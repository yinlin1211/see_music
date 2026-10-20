#!/usr/bin/env python3
"""
MIDI 声乐可视化浏览器 + 播放器
主视图复刻 PNG 图样式：
  [3] RMVPE f0 橙色散点
  [4] Output MIDI 紫色矩形
两轨叠加展示，红线同步播放
"""
import http.server
import socketserver
import os
import json
import urllib.parse
import numpy as np
import datetime
import threading
import argparse

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Browse and compare vocal audio, MIDI, and RMVPE pitch tracks."
    )
    parser.add_argument(
        "--base-dir",
        default=os.environ.get("SEE_MUSIC_BASE_DIR", "."),
        help="Root directory containing classified_vis_* and vocal_midi_* batches.",
    )
    parser.add_argument(
        "--vocal-dir",
        default=os.environ.get("SEE_MUSIC_VOCAL_DIR"),
        help="Directory containing *_vocals.wav or *_vocals.mp3 files.",
    )
    parser.add_argument(
        "--rmvpe-dir",
        default=os.environ.get("SEE_MUSIC_RMVPE_DIR"),
        help="Directory containing *_vocals_f0.npy files.",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SEE_MUSIC_HOST", "127.0.0.1"),
        help="Bind address. Use 0.0.0.0 only when remote access is intended.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SEE_MUSIC_PORT", "8766")),
        help="HTTP port.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
PORT = _ARGS.port
HOST = _ARGS.host
BASE_DIR = os.path.abspath(_ARGS.base_dir)
if not os.path.isdir(BASE_DIR):
    raise SystemExit(f"Base directory does not exist: {BASE_DIR}")

SEPARATED_DIR = os.path.abspath(
    _ARGS.vocal_dir or os.path.join(BASE_DIR, "vocal")
)
RMVPE_DIR = os.path.abspath(
    _ARGS.rmvpe_dir or os.path.join(BASE_DIR, "rmvpe")
)
LOG_FILE      = os.path.join(BASE_DIR, '播放日志.jsonl')
NOTES_FILE    = os.path.join(BASE_DIR, '样本备注.json')   # {vid: note_text}
OPEN_COUNT_FILE = os.path.join(BASE_DIR, '打开次数.json') # {vid: count}
_log_lock = threading.Lock()
_notes_lock = threading.Lock()
_count_lock = threading.Lock()

def _load_json_file(path):
    if os.path.isfile(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def _save_json_file(path, data, lock):
    with lock:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

# 启动时加载到内存
_NOTES = _load_json_file(NOTES_FILE)
_OPEN_COUNTS = _load_json_file(OPEN_COUNT_FILE)

def write_log(entry):
    entry['ts'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    with _log_lock:
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')


def _is_json_line(line):
    """判断一行是否是合法 JSON 对象"""
    s = line.strip()
    if not s:
        return True  # 空行不处理
    try:
        obj = json.loads(s)
        return isinstance(obj, dict)
    except Exception:
        return False


def _log_watcher():
    """
    后台线程：全文扫描日志文件。
    找出所有非 JSON 行（用户手写注释），
    如果该行的下一行不是对应的 user_note，就自动在其后插入。
    """
    import time as _time

    def read_lines():
        if not os.path.isfile(LOG_FILE):
            return []
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return f.readlines()

    def is_user_note_for(text, line):
        """line 是否是 text 对应的 user_note JSON"""
        s = line.strip()
        try:
            obj = json.loads(s)
            return obj.get('action') == 'user_note' and obj.get('text') == text
        except Exception:
            return False

    last_mtime = 0

    while True:
        _time.sleep(1.0)
        try:
            if not os.path.isfile(LOG_FILE):
                continue
            mtime = os.path.getmtime(LOG_FILE)
            if mtime == last_mtime:
                continue  # 文件未变化，跳过

            lines = read_lines()
            result = []
            changed = False
            i = 0
            while i < len(lines):
                raw = lines[i]
                result.append(raw)
                text = raw.strip()
                if text and not _is_json_line(raw):
                    # 检查下一行是否已经是对应的 user_note
                    next_line = lines[i + 1] if i + 1 < len(lines) else ''
                    if not is_user_note_for(text, next_line):
                        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                        note_line = json.dumps(
                            {'action': 'user_note', 'text': text, 'ts': ts},
                            ensure_ascii=False
                        ) + '\n'
                        result.append(note_line)
                        changed = True
                i += 1

            if changed:
                with _log_lock:
                    with open(LOG_FILE, 'w', encoding='utf-8') as f:
                        f.writelines(result)
                last_mtime = os.path.getmtime(LOG_FILE)
            else:
                last_mtime = mtime
        except Exception:
            pass


# 启动日志监听线程
_watcher_thread = threading.Thread(target=_log_watcher, daemon=True)
_watcher_thread.start()

CATS_ORDER = ["A_vocal_clean","B_octave_corrected","C_partial_filtered",
              "D_all_filtered","D_gap_discard","D_pitch_shift"]
SUBS_ORDER = ["90-100","80-90","70-80","60-70","50-60","50以下"]

# ---- 建立索引（YouTube ID 固定11位）----
def build_index(directory, suffix):
    idx = {}
    if not os.path.isdir(directory):
        return idx
    for fname in os.listdir(directory):
        if fname.endswith(suffix):
            stem = fname[:-len(suffix)]
            vid = stem[-11:]
            idx[vid] = os.path.join(directory, fname)
    return idx

print("建立索引...", flush=True)
VOCAL_WAV_INDEX  = build_index(SEPARATED_DIR, "_vocals.wav")
VOCAL_MP3_INDEX  = build_index(SEPARATED_DIR, "_vocals.mp3")
RMVPE_INDEX      = build_index(RMVPE_DIR,     "_vocals_f0.npy")
print(f"WAV:{len(VOCAL_WAV_INDEX)}  RMVPE:{len(RMVPE_INDEX)}", flush=True)


def find_batches():
    batches = []
    for name in sorted(os.listdir(BASE_DIR), reverse=True):
        if name.startswith("classified_vis_") and name.endswith("_按准确率分类"):
            ts = name[len("classified_vis_"):-len("_按准确率分类")]
            midi_dir = os.path.join(BASE_DIR, f"vocal_midi_{ts}")
            vis_acc_dir = os.path.join(BASE_DIR, name)
            # 尝试加载 JSON 索引
            json_index = {}
            json_path = os.path.join(vis_acc_dir, '_match_index.json')
            if not os.path.isfile(json_path):
                json_path = os.path.join(BASE_DIR, f'match_index_{ts}.json')
            if os.path.isfile(json_path):
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    json_index = data.get('samples', {})
                    print(f'  加载 JSON 索引: {ts} ({len(json_index)} 条)', flush=True)
                except Exception as e:
                    print(f'  JSON 加载失败: {e}', flush=True)
            batches.append({
                "ts": ts, "label": ts,
                "vis_acc_dir": vis_acc_dir,
                "midi_dir": midi_dir if os.path.isdir(midi_dir) else None,
                "json_index": json_index,
            })
    return batches

BATCHES   = find_batches()
BATCH_MAP = {b["ts"]: b for b in BATCHES}
print(f"批次: {[b['ts'] for b in BATCHES]}", flush=True)


def get_tree(ts):
    batch = BATCH_MAP.get(ts)
    if not batch: return []
    result = []
    for cat in CATS_ORDER:
        cat_path = os.path.join(batch["vis_acc_dir"], cat)
        if not os.path.isdir(cat_path): continue
        subs = []
        for sub in SUBS_ORDER:
            sub_path = os.path.join(cat_path, sub)
            if os.path.isdir(sub_path):
                count = len([f for f in os.listdir(sub_path) if f.endswith(".png")])
                subs.append({"name": sub, "count": count})
        result.append({"name": cat, "subs": subs})
    return result


def get_images(ts, cat, sub):
    batch = BATCH_MAP.get(ts)
    if not batch: return []
    sub_path = os.path.join(batch["vis_acc_dir"], cat, sub)
    if not os.path.isdir(sub_path): return []
    midi_dir = batch["midi_dir"] or ""
    json_index = batch.get("json_index", {})
    items = []
    for fname in sorted(os.listdir(sub_path)):
        if not fname.endswith(".png"): continue
        vid = fname[:-4]
        img_path = os.path.join(sub_path, fname)
        midi_path = os.path.join(midi_dir, f"{vid}_vocal.mid") if midi_dir else ""
        has_midi = bool(midi_dir) and os.path.isfile(midi_path)
        wav_path   = VOCAL_WAV_INDEX.get(vid, "")
        rmvpe_path = RMVPE_INDEX.get(vid, "")

        # 优先从 JSON 索引读取匹配信息
        rec = json_index.get(vid)
        if rec:
            mismatch    = rec.get('note_mismatch', -1)
            total_notes = rec.get('total_notes', 0)
            keep_ratio  = rec.get('keep_ratio', -1)
            note_match  = rec.get('note_match', 0)
            note_uncertain = rec.get('note_uncertain', 0)
            time_offset_ms = rec.get('time_offset_ms', 0)
            octave_offset  = rec.get('octave_offset', 0)
            has_gap        = rec.get('has_gap', False)
            has_pitch_shift= rec.get('has_pitch_shift', False)
            uncovered_ratio   = rec.get('uncovered_ratio', 0.0)
            uncovered_max_sec = rec.get('uncovered_max_sec', 0.0)
        else:
            # 旧批次没有 JSON，实时计算
            mismatch, total_notes = get_match_score(vid, midi_path if has_midi else "", rmvpe_path)
            keep_ratio = -1
            note_match = note_uncertain = 0
            time_offset_ms = octave_offset = 0
            has_gap = has_pitch_shift = False
            uncovered_ratio = uncovered_max_sec = 0.0

        items.append({
            "id":             vid,
            "img_path":       img_path,
            "midi_path":      midi_path if has_midi else "",
            "has_midi":       has_midi,
            "wav_path":       wav_path,
            "has_wav":        bool(wav_path),
            "mp3_path":       VOCAL_MP3_INDEX.get(vid, ""),
            "has_mp3":        bool(VOCAL_MP3_INDEX.get(vid, "")),
            "rmvpe_path":     rmvpe_path,
            "has_rmvpe":      bool(rmvpe_path),
            "mismatch":       mismatch,
            "total_notes":    total_notes,
            "keep_ratio":     keep_ratio,
            "note_match":     note_match,
            "note_uncertain": note_uncertain,
            "time_offset_ms": time_offset_ms,
            "octave_offset":  octave_offset,
            "has_gap":        has_gap,
            "has_pitch_shift":has_pitch_shift,
            "uncovered_ratio":   uncovered_ratio,
            "uncovered_max_sec": uncovered_max_sec,
        })
    return items


# ---- 匹配分数缓存（vid -> mismatch_count）----
_MATCH_CACHE = {}   # key: vid
_MATCH_LOCK  = threading.Lock()

PITCH_TOL        = 2    # 匹配容差：±2半音
COVER_THRESH     = 0.3  # voiced帧覆盖率阈値
MATCH_THRESH     = 0.3  # 音高匹配率阈値

def compute_match_score(midi_path, rmvpe_path):
    """计算一个样本的不匹配音符数和匹配音符数，返回 (mismatch_count, total_notes)"""
    try:
        import pretty_midi
        pm = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for inst in pm.instruments:
            notes.extend(inst.notes)
        notes.sort(key=lambda n: n.start)
        if not notes:
            return 0, 0

        f0 = np.load(rmvpe_path)
        HOP = 0.01

        mismatch = 0
        for note in notes:
            i0 = int(note.start / HOP)
            i1 = min(int(note.end / HOP) + 1, len(f0))
            if i1 <= i0:
                continue
            seg = f0[i0:i1]
            nz = seg[seg > 0]
            total_frames = len(seg)
            coverage = len(nz) / max(total_frames, 1)
            if coverage < COVER_THRESH:
                # voiced覆盖不足，标记为不确定（不计入不匹配）
                continue
            midi_pitch = note.pitch
            pitch_match = np.mean(np.abs(12 * np.log2(np.maximum(nz, 1e-9) / 440.0) + 69 - midi_pitch) <= PITCH_TOL)
            if pitch_match < MATCH_THRESH:
                mismatch += 1
        return mismatch, len(notes)
    except Exception:
        return -1, 0


def get_match_score(vid, midi_path, rmvpe_path):
    with _MATCH_LOCK:
        if vid in _MATCH_CACHE:
            return _MATCH_CACHE[vid]
    if not midi_path or not rmvpe_path:
        return -1, 0
    mismatch, total = compute_match_score(midi_path, rmvpe_path)
    with _MATCH_LOCK:
        _MATCH_CACHE[vid] = (mismatch, total)
    return mismatch, total


def get_rmvpe_json(rmvpe_path):
    """把 RMVPE npy 转成 JSON：{times:[...], midis:[...], total_sec:...}"""
    try:
        f0 = np.load(rmvpe_path)
        HOP = 0.01  # 10ms
        times, midis = [], []
        for i, hz in enumerate(f0):
            if hz > 0:
                midi = 12 * np.log2(hz / 440.0) + 69
                times.append(round(i * HOP, 3))
                midis.append(round(float(midi), 3))
        total_sec = round(len(f0) * HOP, 3)
        return {"times": times, "midis": midis, "total_sec": total_sec}
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# HTML
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>MIDI 声乐浏览器</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI','PingFang SC',sans-serif;background:#f3f3f3;color:#1e1e1e;display:flex;height:100vh;overflow:hidden}

/* 侧边栏 */
#sidebar{width:230px;min-width:180px;background:#f0f0f0;border-right:1px solid #d0d0d0;display:flex;flex-direction:column;overflow:hidden;flex-shrink:0}
#sidebar-header{padding:12px 14px;border-bottom:1px solid #d0d0d0;flex-shrink:0}
#sidebar-header h2{font-size:13px;color:#0066b8;margin-bottom:7px}
#batch-select{width:100%;padding:5px 7px;background:#ffffff;border:1px solid #c0c0c0;border-radius:4px;color:#1e1e1e;font-size:12px}
#tree{flex:1;overflow-y:auto;padding:4px 0;background:#f0f0f0}
.cat-label{padding:9px 12px 3px;font-size:10px;font-weight:700;color:#888888;text-transform:uppercase;letter-spacing:.8px}
.sub-item{padding:6px 12px 6px 22px;font-size:12px;cursor:pointer;border-left:3px solid transparent;display:flex;justify-content:space-between;align-items:center;transition:all .15s}
.sub-item:hover{background:#e0e0e0}
.sub-item.active{border-left-color:#0066b8;background:#dce9f5;color:#0066b8}
.badge{background:#e0e0e0;padding:1px 6px;border-radius:10px;font-size:11px;color:#555555}
.sub-item.active .badge{background:#b8d4ee;color:#0066b8}
.tag{display:inline-block;padding:1px 5px;border-radius:3px;font-size:10px;font-weight:700;margin-right:3px}
.tag-A{background:#d4f4dd;color:#1a7a35}.tag-B{background:#d4e8f8;color:#0055a0}
.tag-C{background:#fdf4d0;color:#8a6a00}.tag-D{background:#fde8e8;color:#c0392b}

/* 主区域（网格列表） */
#main{flex:1;display:flex;flex-direction:column;overflow:hidden}
#toolbar{padding:9px 14px;background:#f0f0f0;border-bottom:1px solid #d0d0d0;display:flex;align-items:center;gap:10px;flex-shrink:0}
#path-label{font-size:12px;color:#555555;flex-shrink:0}
#search{flex:1;max-width:240px;padding:4px 9px;background:#ffffff;border:1px solid #c0c0c0;border-radius:4px;color:#1e1e1e;font-size:12px}
#search::placeholder{color:#aaaaaa}
#sort-select{padding:4px 7px;background:#ffffff;border:1px solid #c0c0c0;border-radius:4px;color:#1e1e1e;font-size:12px}
#count{font-size:12px;color:#555555;margin-left:auto}
#grid-wrap{flex:1;overflow-y:auto;padding:12px;background:#f3f3f3}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px}
.card{background:#ffffff;border:1px solid #d0d0d0;border-radius:7px;overflow:hidden;cursor:pointer;transition:all .2s}
.card:hover{border-color:#0066b8;transform:translateY(-2px);box-shadow:0 4px 16px rgba(0,102,184,.15)}
.card img{width:100%;height:100px;object-fit:cover;display:block;background:#e8e8e8}
.card-info{padding:6px 9px}
.card-id{font-size:11px;color:#0066b8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card-tags{margin-top:2px;display:flex;gap:3px}
.mini-tag{font-size:10px;padding:1px 4px;border-radius:3px}
.mt-midi{background:#d4e8f8;color:#0055a0}.mt-wav{background:#d4f4dd;color:#1a7a35}
.mt-rmvpe{background:#fde8d0;color:#b85a00}.mt-none{background:#e8e8e8;color:#777777}
.mt-mismatch{background:#fde8e8;color:#c0392b;font-weight:700}
.mt-ok{background:#d4f4dd;color:#1a7a35;font-weight:700}
#empty,#loading{text-align:center;padding:60px;color:#888888;font-size:14px}

/* ========== 模态框 ========== */
#modal{display:none;position:fixed;inset:0;background:rgba(200,200,200,.92);z-index:1000;flex-direction:column}
#modal.open{display:flex}
#modal-hdr{padding:10px 16px;background:#f0f0f0;border-bottom:1px solid #d0d0d0;display:flex;align-items:center;gap:10px;flex-shrink:0}
#modal-title{font-size:14px;font-weight:600;color:#0066b8;white-space:nowrap}
#note-input{flex:1;background:#ffffff;border:1px solid #c0c0c0;border-radius:4px;color:#1e1e1e;font-size:12px;padding:5px 10px;outline:none}
#note-input:focus{border-color:#0066b8}
#note-input::placeholder{color:#aaaaaa}
#open-count{font-size:11px;color:#555555;white-space:nowrap}
#modal-meta{font-size:11px;color:#555555}
#modal-close{background:none;border:none;color:#888888;font-size:22px;cursor:pointer;line-height:1;padding:0 4px}
#modal-close:hover{color:#1e1e1e}

/* 播放器主体 */
#player-body{flex:1;display:flex;flex-direction:column;overflow:hidden;padding:10px 14px;gap:8px;background:#f8f8f8}

/* 两个主轨道 */
.track-block{display:flex;flex-direction:column;gap:3px;flex:1;min-height:0}
.track-label{font-size:11px;color:#555555;display:flex;justify-content:space-between}
.track-label span{color:#0066b8}
.canvas-wrap{position:relative;background:#ffffff;border:1px solid #cccccc;border-radius:5px;overflow:hidden;cursor:crosshair;flex:1;min-height:0}
.canvas-wrap canvas{display:block;width:100%;height:100%}
.playhead-line{position:absolute;top:0;bottom:0;width:2px;background:#ff3333;pointer-events:none;z-index:5;box-shadow:0 0 6px rgba(255,50,50,.7)}

/* 波形轨道（固定高度） */
#wav-block{flex-shrink:0;height:60px;display:flex;flex-direction:column;gap:3px}
#wav-block .canvas-wrap{flex:1}

/* 时间轴 + 控制 */
#ctrl-row{display:flex;align-items:center;gap:10px;flex-shrink:0;padding:4px 0}
#btn-play{padding:8px 28px;border-radius:6px;border:none;background:#4cc9f0;color:#0a0a1a;font-size:14px;font-weight:700;cursor:pointer;transition:all .2s}
#btn-play:hover{background:#74d7f5}
#btn-play:disabled{background:#cccccc;color:#888888;cursor:not-allowed}
#time-display{font-size:13px;color:#0066b8;min-width:130px}
#status-text{font-size:11px;color:#555555;flex:1}

/* 图片折叠区 */
#img-toggle{padding:4px 10px;background:#f0f0f0;border:1px solid #d0d0d0;border-radius:4px;font-size:11px;color:#555555;cursor:pointer;flex-shrink:0;align-self:flex-start}
#img-toggle:hover{color:#1e1e1e}
#img-area{flex-shrink:0;overflow-y:auto;max-height:220px;display:none}
#img-area img{width:100%;border-radius:5px;border:1px solid #d0d0d0}

/* Y轴标签 */
.y-labels{position:absolute;left:0;top:0;bottom:0;width:28px;display:flex;flex-direction:column;justify-content:space-between;pointer-events:none;padding:2px 0}
.y-label{font-size:9px;color:#888888;text-align:right;padding-right:3px;line-height:1}

::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:#f0f0f0}
::-webkit-scrollbar-thumb{background:#c0c0c0;border-radius:3px}
</style>
</head>
<body>

<div id="sidebar">
  <div id="sidebar-header">
    <h2>🎵 MIDI 声乐浏览器</h2>
    <select id="batch-select"></select>
  </div>
  <div id="tree"></div>
</div>

<div id="main">
  <div id="toolbar">
    <span id="path-label">请选择分类</span>
    <input id="search" type="text" placeholder="搜索 ID..." oninput="filterGrid()">
    <select id="sort-select" onchange="sortGrid()">
      <option value="default">原始顺序</option>
      <option value="mismatch_asc">不匹配 ↑ 升序</option>
      <option value="mismatch_desc">不匹配 ↓ 降序</option>
    </select>
    <span id="count"></span>
  </div>
  <div id="grid-wrap">
    <div id="grid"></div>
    <div id="loading" style="display:none">加载中...</div>
    <div id="empty" style="display:none">该目录为空</div>
  </div>
</div>

<!-- 模态框 -->
<div id="modal">
  <div id="modal-hdr">
    <button onclick="navModal(-1)" title="上一个 (←)" style="background:#e8f0f8;border:1px solid #b0c8e0;color:#0066b8;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:14px">◀ 上一个</button>
    <button onclick="navModal(1)" title="下一个 (→)" style="background:#e8f0f8;border:1px solid #b0c8e0;color:#0066b8;padding:5px 12px;border-radius:4px;cursor:pointer;font-size:14px">下一个 ▶</button>
    <span id="modal-title">-</span>
    <input id="note-input" type="text" placeholder="在此输入备注，按 Enter 保存..." onkeydown="if(event.key==='Enter')saveNote()" onblur="saveNote()">
    <span id="open-count"></span>
    <span id="modal-meta"></span>
    <button id="img-toggle" onclick="toggleImg()">📷 对比图</button>
    <a href="/api/log/view" target="_blank" style="padding:6px 10px;border-radius:4px;border:1px solid #d0d0d0;background:#f0f0f0;color:#555555;font-size:11px;text-decoration:none" title="查看播放日志">📋 日志</a>
    <button id="modal-close" onclick="closeModal()">✕</button>
  </div>

  <div id="player-body">
    <!-- 对比图（折叠） -->
    <div id="img-area"><img id="main-img" src="" alt="对比图"></div>

    <!-- [3] RMVPE 轨道（隐藏，仅保留canvas供绘图使用） -->
    <div id="rmvpe-block" style="display:none">
      <canvas id="rmvpe-canvas" style="display:none"></canvas>
      <div class="playhead-line" id="rmvpe-head" style="display:none"></div>
      <span id="rmvpe-voiced"></span>
    </div>

    <!-- [5] Output vs RMVPE overlap 轨道 -->
    <div class="track-block" id="midi-block">
      <div class="track-label">
        <span style="color:#43a047">▪ [5] Output vs RMVPE overlap</span>
        <span style="font-size:10px;margin-left:6px">
          <span style="display:inline-block;width:10px;height:10px;background:#43a047;border-radius:2px;vertical-align:middle"></span> 匹配 
          <span style="display:inline-block;width:10px;height:10px;background:#e53935;border-radius:2px;vertical-align:middle"></span> 不匹配 
          <span style="display:inline-block;width:10px;height:10px;background:rgba(180,120,0,0.9);border-radius:2px;vertical-align:middle"></span> 不确定 
          <span style="display:inline-block;width:8px;height:8px;background:rgba(255,165,0,0.6);border-radius:50%;vertical-align:middle"></span> RMVPE
        </span>
        <span id="midi-info"></span>
      </div>
      <div class="canvas-wrap" id="midi-wrap" onclick="seekByClick(event)">
        <canvas id="midi-canvas"></canvas>
        <div class="playhead-line" id="midi-head"></div>
      </div>
    </div>

    <!-- 波形 -->
    <div id="wav-block">
      <div class="track-label">
        <span style="color:#4cc9f0">▪ 人声波形</span>
        <span id="wav-info"></span>
      </div>
      <div class="canvas-wrap" id="wav-wrap" onclick="seekByClick(event)">
        <canvas id="wav-canvas"></canvas>
        <div class="playhead-line" id="wav-head"></div>
      </div>
    </div>

    <!-- 控制栏 -->
    <div id="ctrl-row">
      <button id="btn-play" disabled onclick="togglePlay()">▶ 播放</button>
      <span id="time-display">0.00 / 0.00 s</span>
      <span id="status-text"></span>
    </div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/@tonejs/midi@2.0.28/build/Midi.js"></script>
<script>
// ============================================================
// 导航
// ============================================================
let allBatches=[], curBatch=null, curCat=null, curSub=null, allItems=[];
const CAT_CLS={A:'tag-A',B:'tag-B',C:'tag-C',D:'tag-D'};
const PITCH_MIN=40, PITCH_MAX=90;

async function init(){
  const r=await fetch('/api/batches');
  allBatches=await r.json();
  const sel=document.getElementById('batch-select');
  allBatches.forEach(b=>{const o=document.createElement('option');o.value=b.ts;o.textContent=b.label;sel.appendChild(o);});
  if(allBatches.length){sel.value=allBatches[0].ts;await selectBatch(allBatches[0].ts);}
  sel.addEventListener('change',()=>selectBatch(sel.value));
}

async function selectBatch(ts){
  curBatch=ts;
  const r=await fetch('/api/tree?ts='+ts);
  const tree=await r.json();
  renderTree(tree);
  for(const cat of tree) for(const sub of cat.subs) if(sub.count>0){await loadGrid(cat.name,sub.name);return;}
}

function renderTree(tree){
  const c=document.getElementById('tree');c.innerHTML='';
  tree.forEach(cat=>{
    const letter=cat.name[0];
    const lbl=document.createElement('div');lbl.className='cat-label';
    lbl.innerHTML=`<span class="tag ${CAT_CLS[letter]||''}">${letter}</span>${cat.name}`;
    c.appendChild(lbl);
    cat.subs.forEach(sub=>{
      const el=document.createElement('div');el.className='sub-item';
      el.dataset.cat=cat.name;el.dataset.sub=sub.name;
      el.innerHTML=`<span>${sub.name}</span><span class="badge">${sub.count}</span>`;
      el.addEventListener('click',()=>loadGrid(cat.name,sub.name));
      c.appendChild(el);
    });
  });
}

async function loadGrid(cat,sub){
  curCat=cat;curSub=sub;
  document.querySelectorAll('.sub-item').forEach(el=>el.classList.toggle('active',el.dataset.cat===cat&&el.dataset.sub===sub));
  document.getElementById('path-label').textContent=`${cat} / ${sub}`;
  document.getElementById('grid').innerHTML='';
  document.getElementById('loading').style.display='block';
  document.getElementById('empty').style.display='none';
  const r=await fetch(`/api/images?ts=${curBatch}&cat=${encodeURIComponent(cat)}&sub=${encodeURIComponent(sub)}`);
  allItems=await r.json();
  document.getElementById('loading').style.display='none';
  renderGrid(allItems);
}

let _displayedItems=[];  // 当前网格中显示的列表
let _curModalIdx=-1;      // 当前打开的样本在列表中的索引

function renderGrid(items){
  _displayedItems=items;  // 记录当前显示列表
  const grid=document.getElementById('grid');grid.innerHTML='';
  document.getElementById('empty').style.display=items.length?'none':'block';
  document.getElementById('count').textContent=`${items.length} 个样本`;
  if(items.length>0){console.log("[DEBUG] first item uncovered_ratio:",items[0].uncovered_ratio,"uncovered_max_sec:",items[0].uncovered_max_sec);}
  items.forEach((item,idx)=>{
    const card=document.createElement('div');card.className='card';
    const mTag=item.has_midi?'<span class="mini-tag mt-midi">MIDI</span>':'';
    const wTag=item.has_wav?'<span class="mini-tag mt-wav">WAV</span>':'';
    const rTag=item.has_rmvpe?'<span class="mini-tag mt-rmvpe">RMVPE</span>':'';
    let mmTag='';
    if(item.mismatch>0){
      mmTag=`<span class="mini-tag mt-mismatch">❌${item.mismatch}</span>`;
    } else if(item.mismatch===0 && item.total_notes>0){
      mmTag=`<span class="mini-tag mt-ok">✔全匹配</span>`;
    }
    const ucRatio = `<span class="mini-tag" style="background:#ede9fe;color:#7c3aed;font-size:9px" title="未覆盖比例">未覆${(item.uncovered_ratio||0).toFixed(1)}%</span>`;
    const ucMax = `<span class="mini-tag" style="background:#e0f2fe;color:#0369a1;font-size:9px" title="最大未覆盖时长">空${item.uncovered_max_sec!=null?item.uncovered_max_sec.toFixed(2):'0.00'}s</span>`;
    card.innerHTML=`<img src="/api/img?path=${encodeURIComponent(item.img_path)}" loading="lazy" alt="${item.id}">
      <div class="card-info"><div class="card-id">${item.id}</div><div class="card-tags">${mTag}${wTag}${rTag}${mmTag}${ucRatio}${ucMax}</div></div>`;
    card.addEventListener('click',()=>{_curModalIdx=idx;openModal(item);});
    grid.appendChild(card);
  });
}

function navModal(delta){
  if(!_displayedItems.length) return;
  _curModalIdx=(_curModalIdx+delta+_displayedItems.length)%_displayedItems.length;
  openModal(_displayedItems[_curModalIdx]);
}

let sortMode='default';
function sortGrid(){
  sortMode=document.getElementById('sort-select').value;
  applyFilterSort();
}

function filterGrid(){
  applyFilterSort();
}

function applyFilterSort(){
  const q=document.getElementById('search').value.toLowerCase();
  let items=q?allItems.filter(t=>t.id.toLowerCase().includes(q)):[...allItems];
  if(sortMode==='mismatch_asc'){
    items.sort((a,b)=>{
      const ma=a.mismatch<0?9999:a.mismatch;
      const mb=b.mismatch<0?9999:b.mismatch;
      return ma-mb;
    });
  } else if(sortMode==='mismatch_desc'){
    items.sort((a,b)=>{
      const ma=a.mismatch<0?-1:a.mismatch;
      const mb=b.mismatch<0?-1:b.mismatch;
      return mb-ma;
    });
  }
  renderGrid(items);
}



// ============================================================
// 播放器状态
// ============================================================
const P = {
  rmvpeTimes:[], rmvpeMidis:[],
  midiNotes:[], minMidi:48, maxMidi:84,
  waveform:[], audioBuffer:null, audioContext:null, audioSource:null,
  totalDuration:0, currentTime:0, startTime:0,
  playing:false, animId:null,
};

function resetPlayer(){
  stopPlayback();
  Object.assign(P,{rmvpeTimes:[],rmvpeMidis:[],midiNotes:[],waveform:[],
    audioBuffer:null,totalDuration:0,currentTime:0});
  clearCanvas('midi-canvas');clearCanvas('wav-canvas');
  document.getElementById('btn-play').disabled=true;
  document.getElementById('time-display').textContent='0.00 / 0.00 s';
  document.getElementById('rmvpe-voiced').textContent='';
  document.getElementById('midi-info').textContent='';
  document.getElementById('wav-info').textContent='';
  updatePlayheads(0);
}

function clearCanvas(id){
  const c=document.getElementById(id);
  c.getContext('2d').clearRect(0,0,c.width,c.height);
}

// ---- 加载 RMVPE ----
async function loadRmvpe(url){
  setStatus('加载 RMVPE...');
  const r=await fetch(url);
  const d=await r.json();
  if(d.error){setStatus('RMVPE 加载失败');return;}
  P.rmvpeTimes=d.times; P.rmvpeMidis=d.midis;
  P.totalDuration=Math.max(P.totalDuration, d.total_sec);
  const pct=d.times.length>0?(d.times.length/(d.total_sec/0.01)*100).toFixed(0):'0';
  document.getElementById('rmvpe-voiced').textContent=`voiced ${pct}%`;
  setStatus('RMVPE 已加载');
}

// ---- 加载 MIDI ----
async function loadMidi(url){
  setStatus('加载 MIDI...');
  const r=await fetch(url);
  const buf=await r.arrayBuffer();
  const midi=new Midi(buf);
  P.midiNotes=[];
  midi.tracks.forEach(t=>t.notes.forEach(n=>P.midiNotes.push({time:n.time,dur:n.duration,midi:n.midi,vel:n.velocity})));
  P.midiNotes.sort((a,b)=>a.time-b.time);
  if(P.midiNotes.length){
    P.minMidi=Math.min(...P.midiNotes.map(n=>n.midi));
    P.maxMidi=Math.max(...P.midiNotes.map(n=>n.midi));
    const last=P.midiNotes[P.midiNotes.length-1];
    P.totalDuration=Math.max(P.totalDuration,last.time+last.dur+0.5);
  }
  document.getElementById('midi-info').textContent=`${P.midiNotes.length} notes`;
  setStatus('MIDI 已加载');
}

// ---- 加载 WAV ----
async function loadWav(url){
  setStatus('加载音频...');
  if(!P.audioContext) P.audioContext=new (window.AudioContext||window.webkitAudioContext)();
  const r=await fetch(url);
  const buf=await r.arrayBuffer();
  P.audioBuffer=await P.audioContext.decodeAudioData(buf);
  P.totalDuration=Math.max(P.totalDuration,P.audioBuffer.duration);
  // 降采样波形
  const raw=P.audioBuffer.getChannelData(0);
  const pts=1500, step=Math.floor(raw.length/pts);
  P.waveform=[];
  for(let i=0;i<pts;i++){let m=0;for(let j=0;j<step;j++)m=Math.max(m,Math.abs(raw[i*step+j]||0));P.waveform.push(m);}
  document.getElementById('wav-info').textContent=`${P.audioBuffer.duration.toFixed(1)}s`;
  setStatus('音频已加载');
}

async function loadAll(item){
  resetPlayer();
  // 先加载 RMVPE + MIDI，完成后立即绘图
  const fastTasks=[];
  if(item.has_rmvpe) fastTasks.push(loadRmvpe(`/api/rmvpe?path=${encodeURIComponent(item.rmvpe_path)}`));
  if(item.has_midi)  fastTasks.push(loadMidi(`/api/midi?path=${encodeURIComponent(item.midi_path)}`));
  await Promise.all(fastTasks);
  // 等浏览器完成布局（canvas 尺寸确定后）再绘图
  requestAnimationFrame(()=>{
    drawMidi();
    document.getElementById('btn-play').disabled=!(P.audioBuffer||P.midiNotes.length);
    updateTimeDisplay();
  });
  // 异步加载音频（MP3 优先，没有则 WAV）
  if(item.has_mp3){
    loadWav(`/api/mp3?path=${encodeURIComponent(item.mp3_path)}`).then(()=>{
      drawWaveform();
      document.getElementById('btn-play').disabled=false;
    }).catch(()=>{});
  } else if(item.has_wav){
    loadWav(`/api/wav?path=${encodeURIComponent(item.wav_path)}`).then(()=>{
      drawWaveform();
      document.getElementById('btn-play').disabled=false;
    }).catch(()=>{});
  }
}

// ============================================================
// 绘图
// ============================================================
function resizeCanvas(id, wrapId){
  const c=document.getElementById(id);
  const w=document.getElementById(wrapId);
  const rect=w.getBoundingClientRect();
  c.width=rect.width||600; c.height=rect.height||150;
  return c;
}

// Y轴：MIDI音高 → 像素（PITCH_MIN~PITCH_MAX，上高下低）
function pitchToY(midi, H){
  return H - ((midi - PITCH_MIN) / (PITCH_MAX - PITCH_MIN)) * H;
}

// 绘制网格（Y轴每10个半音一条线）
function drawGrid(ctx, W, H){
  // 白底背景
  ctx.fillStyle='#ffffff';
  ctx.fillRect(0,0,W,H);
  // 每10半音实线
  ctx.strokeStyle='rgba(0,0,0,0.12)';
  ctx.lineWidth=1;
  for(let p=PITCH_MIN;p<=PITCH_MAX;p+=10){
    const y=pitchToY(p,H);
    ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();
    ctx.fillStyle='rgba(0,0,0,0.35)';
    ctx.font='9px monospace';
    ctx.fillText(p, 2, y-2);
  }
  // 每5个半音虚线
  ctx.strokeStyle='rgba(0,0,0,0.05)';
  for(let p=PITCH_MIN+5;p<PITCH_MAX;p+=10){
    const y=pitchToY(p,H);
    ctx.setLineDash([3,4]);
    ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(W,y);ctx.stroke();
    ctx.setLineDash([]);
  }
}

function drawRmvpe(){
  const c=resizeCanvas('rmvpe-canvas','rmvpe-wrap');
  const ctx=c.getContext('2d'); const W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  drawGrid(ctx,W,H);
  if(!P.rmvpeTimes.length){
    ctx.fillStyle='#999';ctx.font='13px sans-serif';ctx.fillText('无 RMVPE 数据',W/2-50,H/2);return;
  }
  const dur=P.totalDuration||1;
  // 橙色散点，s=1.5 → 半径约1.5px
  ctx.fillStyle='rgba(230,120,0,0.9)';
  for(let i=0;i<P.rmvpeTimes.length;i++){
    const x=(P.rmvpeTimes[i]/dur)*W;
    const y=pitchToY(P.rmvpeMidis[i],H);
    ctx.beginPath();ctx.arc(x,y,1.5,0,Math.PI*2);ctx.fill();
  }
}

function drawMidi(){
  const c=resizeCanvas('midi-canvas','midi-wrap');
  const ctx=c.getContext('2d'); const W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  drawGrid(ctx,W,H);
  const dur=P.totalDuration||1;

  if(!P.midiNotes.length && !P.rmvpeTimes.length){
    ctx.fillStyle='#999';ctx.font='13px sans-serif';ctx.fillText('无数据',W/2-30,H/2);return;
  }

  const HOP=0.01;
  const PITCH_TOL_JS=2;
  const COVER_THRESH_JS=0.3;
  const MATCH_THRESH_JS=0.3;

  // 层一：先画 RMVPE 淡橙色散点（在最下面）
  if(P.rmvpeTimes.length){
    ctx.fillStyle='rgba(255,165,0,0.35)';
    for(let i=0;i<P.rmvpeTimes.length;i++){
      const x=(P.rmvpeTimes[i]/dur)*W;
      const y=pitchToY(P.rmvpeMidis[i],H);
      ctx.beginPath();ctx.arc(x,y,1.5,0,Math.PI*2);ctx.fill();
    }
  }

  // 层二：画 MIDI 音符，根据与 RMVPE 匹配情况着色（绿=匹配，红=不匹配）
  if(P.midiNotes.length){
    P.midiNotes.forEach(note=>{
      const x=(note.time/dur)*W;
      const w=Math.max((note.dur/dur)*W,2);
      const noteH=(H/(PITCH_MAX-PITCH_MIN));
      const GAP=1;
      const y=pitchToY(note.midi,H)-noteH+1+GAP;
      const h=Math.max(noteH-1-GAP*2,2);
      const nx=x+GAP;
      const nw=Math.max(w-GAP*2,1);

      // 计算匹配情况
      let fillColor='#e53935'; // 默认红色
      let strokeColor='rgba(120,0,0,0.6)';
      if(P.rmvpeTimes.length>0){
        const t0=note.time, t1=note.time+note.dur;
        const voicedMidis=[];
        for(let i=0;i<P.rmvpeTimes.length;i++){
          const t=P.rmvpeTimes[i];
          if(t>=t0 && t<t1) voicedMidis.push(P.rmvpeMidis[i]);
          else if(t>=t1) break;
        }
        const totalFrames=Math.max(1,Math.round((t1-t0)/HOP));
        const coverage=voicedMidis.length/totalFrames;
        if(coverage>=COVER_THRESH_JS && voicedMidis.length>0){
          const matched=voicedMidis.filter(m=>Math.abs(m-note.midi)<=PITCH_TOL_JS).length;
          const pitchMatch=matched/voicedMidis.length;
          if(pitchMatch>=MATCH_THRESH_JS){
            fillColor='#43a047'; strokeColor='rgba(0,80,0,0.6)'; // 绿色
          }
        } else {
          fillColor='rgba(180,120,0,0.75)'; strokeColor='rgba(100,60,0,0.5)'; // 橙色=不确定
        }
      }
      ctx.fillStyle=fillColor;
      ctx.fillRect(nx,y,nw,h);
      ctx.strokeStyle=strokeColor;
      ctx.lineWidth=0.8;
      ctx.strokeRect(nx+0.5,y+0.5,nw-1,h-1);
    });
  }
}

function drawWaveform(){
  const c=resizeCanvas('wav-canvas','wav-wrap');
  const ctx=c.getContext('2d'); const W=c.width,H=c.height;
  ctx.clearRect(0,0,W,H);
  if(!P.waveform.length){
    ctx.fillStyle='#999';ctx.font='12px sans-serif';ctx.fillText('无音频数据',W/2-40,H/2);return;
  }
  const cy=H/2;
  const grad=ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'rgba(76,201,240,0.7)');
  grad.addColorStop(0.5,'rgba(76,201,240,0.3)');
  grad.addColorStop(1,'rgba(76,201,240,0.7)');
  ctx.beginPath();ctx.moveTo(0,cy);
  for(let i=0;i<P.waveform.length;i++){const x=(i/P.waveform.length)*W;ctx.lineTo(x,cy-P.waveform[i]*cy*0.9);}
  for(let i=P.waveform.length-1;i>=0;i--){const x=(i/P.waveform.length)*W;ctx.lineTo(x,cy+P.waveform[i]*cy*0.9);}
  ctx.closePath();ctx.fillStyle=grad;ctx.fill();
  ctx.beginPath();ctx.strokeStyle='rgba(76,201,240,0.9)';ctx.lineWidth=1.2;
  for(let i=0;i<P.waveform.length;i++){const x=(i/P.waveform.length)*W;const y=cy-P.waveform[i]*cy*0.9;i===0?ctx.moveTo(x,y):ctx.lineTo(x,y);}
  ctx.stroke();
}

// ============================================================
// 播放控制
// ============================================================
function togglePlay(){ P.playing?pausePlayback():startPlayback(); }

function startPlayback(){
  if(!P.audioBuffer&&!P.midiNotes.length) return;
  P.playing=true;
  document.getElementById('btn-play').textContent='⏸ 暂停';
  logPlay(P.currentTime);
  if(P.audioBuffer){
    if(P.audioContext.state==='suspended') P.audioContext.resume();
    P.audioSource=P.audioContext.createBufferSource();
    P.audioSource.buffer=P.audioBuffer;
    P.audioSource.connect(P.audioContext.destination);
    const offset=Math.min(P.currentTime, P.audioBuffer.duration-0.01);
    P.audioSource.start(0, offset);
    // 用局部变量捕获本次source，避免stop()触发onended时误重置
    const thisSource=P.audioSource;
    P.audioSource.onended=()=>{
      if(P.audioSource!==thisSource) return; // 已被新source替换，忽略
      if(P.playing){P.playing=false;P.currentTime=0;document.getElementById('btn-play').textContent='▶ 播放';updatePlayheads(0);updateTimeDisplay();}
    };
  }
  P.startTime=performance.now()-P.currentTime*1000;
  animate();
}

function pausePlayback(){
  if(P.playing) logPause(P.currentTime);
  P.playing=false;
  document.getElementById('btn-play').textContent='▶ 播放';
  if(P.animId){cancelAnimationFrame(P.animId);P.animId=null;}
  // 先保存当前时间，再stop（stop会触发onended，但playing已为false所以不会重置）
  const savedTime=P.currentTime;
  if(P.audioSource){
    const src=P.audioSource;
    P.audioSource=null; // 先置null，让onended检查失效
    try{src.stop();}catch(e){}
  }
  P.currentTime=savedTime;
}

function stopPlayback(){
  pausePlayback();P.currentTime=0;updatePlayheads(0);updateTimeDisplay();
}

function animate(){
  if(!P.playing) return;
  P.currentTime=(performance.now()-P.startTime)/1000;
  if(P.currentTime>=P.totalDuration){
    P.playing=false;P.currentTime=0;
    document.getElementById('btn-play').textContent='▶ 播放';
    if(P.audioSource){try{P.audioSource.stop();}catch(e){}P.audioSource=null;}
    updatePlayheads(0);updateTimeDisplay();return;
  }
  updatePlayheads(P.currentTime/P.totalDuration);
  updateTimeDisplay();
  P.animId=requestAnimationFrame(animate);
}

function updatePlayheads(frac){
  ['rmvpe-wrap','midi-wrap','wav-wrap'].forEach(wid=>{
    const wrap=document.getElementById(wid);
    if(!wrap) return;
    const head=wrap.querySelector('.playhead-line');
    if(head) head.style.left=(frac*100)+'%';
  });
}

function updateTimeDisplay(){
  document.getElementById('time-display').textContent=
    `${(P.currentTime||0).toFixed(2)} / ${(P.totalDuration||0).toFixed(2)} s`;
}

function seekByClick(e){
  const wrap=e.currentTarget;
  const rect=wrap.getBoundingClientRect();
  const frac=Math.max(0,Math.min(1,(e.clientX-rect.left)/rect.width));
  const was=P.playing;
  if(was) pausePlayback();
  P.currentTime=frac*P.totalDuration;
  logSeek(P.currentTime);
  updatePlayheads(frac);updateTimeDisplay();
  if(was) startPlayback();
}

function setStatus(msg){document.getElementById('status-text').textContent=msg;}

// ============================================================
// 模态框
// ============================================================
let imgVisible=false;
function toggleImg(){
  imgVisible=!imgVisible;
  document.getElementById('img-area').style.display=imgVisible?'block':'none';
  document.getElementById('img-toggle').textContent=imgVisible?'📷 隐藏图':'📷 对比图';
}

let _curNoteVid='';  // 当前备注对应的 vid
let _lastSavedNote=''; // 上次保存的内容，防止重复写入

function saveNote(){
  const inp=document.getElementById('note-input');
  const text=inp.value.trim();
  if(!_curNoteVid || text===_lastSavedNote) return;
  _lastSavedNote=text;
  fetch('/api/note/set',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vid:_curNoteVid,note:text})}).catch(()=>{});
}

function openModal(item){
  document.getElementById('modal-title').textContent=item.id;
  // 先保存上一个的备注
  saveNote();
  _curNoteVid=item.id;
  _lastSavedNote='';
  // 加载备注和打开次数
  fetch('/api/open-count',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({vid:item.id})})
    .then(r=>r.json()).then(d=>{
      document.getElementById('open-count').textContent=`已打开 ${d.count} 次`;
    }).catch(()=>{});
  fetch(`/api/note/get?vid=${encodeURIComponent(item.id)}`)
    .then(r=>r.json()).then(d=>{
      const inp=document.getElementById('note-input');
      inp.value=d.note||'';
      _lastSavedNote=d.note||'';
    }).catch(()=>{});
  const letter=curCat[0];
  // 构建详细信息标签
  let metaHtml=`<span class="tag ${CAT_CLS[letter]||''}">${curCat}</span> ${curSub}`;
  if(item.keep_ratio>=0){
    metaHtml+=` <span style="color:#fb923c;font-size:11px">准确率 ${item.keep_ratio.toFixed(1)}%</span>`;
  }
  if(item.total_notes>0){
    const matchPct=item.note_match>0?(item.note_match/item.total_notes*100).toFixed(0):'0';
    metaHtml+=` <span style="color:#4ade80;font-size:11px">✔${item.note_match}</span>`;
    if(item.mismatch>0) metaHtml+=` <span style="color:#f87171;font-size:11px">❌${item.mismatch}</span>`;
    if(item.note_uncertain>0) metaHtml+=` <span style="color:#fb923c;font-size:11px">?${item.note_uncertain}</span>`;
    metaHtml+=` <span style="color:#6a7aaa;font-size:10px">/ ${item.total_notes}音符</span>`;
  }
  if(item.time_offset_ms && Math.abs(item.time_offset_ms)>5){
    metaHtml+=` <span style="color:#60a5fa;font-size:10px">对齐${item.time_offset_ms>0?'+':''}${item.time_offset_ms}ms</span>`;
  }
  if(item.octave_offset && item.octave_offset!==0){
    metaHtml+=` <span style="color:#a78bfa;font-size:10px">八度${item.octave_offset>0?'+':''}${item.octave_offset}st</span>`;
  }
  if(item.has_gap)   metaHtml+=` <span style="color:#f87171;font-size:10px">[GAP]</span>`;
  if(item.has_pitch_shift) metaHtml+=` <span style="color:#f87171;font-size:10px">[SHIFT]</span>`;
  if(item.uncovered_ratio!=null) metaHtml+=` <span style="background:#ede9fe;color:#7c3aed;font-size:10px;padding:1px 5px;border-radius:3px" title="未覆盖比例">未覆盖 ${(item.uncovered_ratio||0).toFixed(1)}%</span>`;
  if(item.uncovered_max_sec!=null) metaHtml+=` <span style="background:#e0f2fe;color:#0369a1;font-size:10px;padding:1px 5px;border-radius:3px" title="最大未覆盖时长">最大空白 ${item.uncovered_max_sec.toFixed(2)}s</span>`;
  document.getElementById('modal-meta').innerHTML=metaHtml;
  document.getElementById('main-img').src=`/api/img?path=${encodeURIComponent(item.img_path)}`;
  document.getElementById('modal').classList.add('open');
  logOpen(item.id, curCat, curSub);
  loadAll(item);
}

function closeModal(){
  saveNote();  // 关闭时保存备注
  document.getElementById('modal').classList.remove('open');
  logClose();
  stopPlayback();resetPlayer();
  if(imgVisible){imgVisible=false;document.getElementById('img-area').style.display='none';document.getElementById('img-toggle').textContent='📷 对比图';}
}

document.addEventListener('keydown',e=>{
  if(!document.getElementById('modal').classList.contains('open')) return;
  if(e.key==='Escape') closeModal();
  else if(e.key===' '){e.preventDefault();togglePlay();}
  else if(e.key==='ArrowLeft' && !e.shiftKey){e.preventDefault();navModal(-1);}
  else if(e.key==='ArrowRight' && !e.shiftKey){e.preventDefault();navModal(1);}
});

// 窗口resize时重绘
window.addEventListener('resize',()=>{
  if(document.getElementById('modal').classList.contains('open')){
    drawRmvpe();drawMidi();drawWaveform();
  }
});

// ============================================================
// 日志功能
// ============================================================
let _logVid='', _logCat='', _logSub='';
let _playStartSec=0, _playStartWall=0;

function logSend(entry){
  // 异步发送，不阻塞主流
  fetch('/api/log', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify(entry)
  }).catch(()=>{});
}

function logOpen(vid, cat, sub){
  _logVid=vid; _logCat=cat; _logSub=sub;
  logSend({action:'open', vid, cat, sub, text:''});
}

function logPlay(fromSec){
  _playStartSec=fromSec;
  _playStartWall=performance.now();
  logSend({action:'play', vid:_logVid, from_sec:fromSec, text:''});
}

function logPause(atSec){
  const duration=Math.max(0, atSec-_playStartSec);
  logSend({action:'pause', vid:_logVid, at_sec:atSec, from_sec:_playStartSec, duration, text:''});
}

function logSeek(toSec){
  logSend({action:'seek', vid:_logVid, to_sec:toSec, text:''});
}

function logClose(){
  if(_logVid) logSend({action:'close', vid:_logVid, text:''});
  _logVid='';
}

init();
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin","*")
        self.send_header("Cache-Control","no-store")
        super().end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        p = parsed.path
        qs = urllib.parse.parse_qs(parsed.query)

        if p in ("/","/index.html"):
            self._html(HTML)
        elif p=="/api/batches":
            self._json([{"ts":b["ts"],"label":b["label"]} for b in BATCHES])
        elif p=="/api/tree":
            self._json(get_tree(qs.get("ts",[""])[0]))
        elif p=="/api/images":
            self._json(get_images(qs.get("ts",[""])[0],qs.get("cat",[""])[0],qs.get("sub",[""])[0]))
        elif p=="/api/img":
            self._file(qs.get("path",[""])[0],"image/png",BASE_DIR)
        elif p=="/api/midi":
            self._file(qs.get("path",[""])[0],"audio/midi",BASE_DIR)
        elif p=="/api/wav":
            self._file(qs.get("path",[""])[0],"audio/wav",SEPARATED_DIR)
        elif p=="/api/mp3":
            self._file(qs.get("path",[""])[0],"audio/mpeg",SEPARATED_DIR)
        elif p=="/api/rmvpe":
            fpath = qs.get("path",[""])[0]
            abs_path = os.path.realpath(fpath)
            allowed = os.path.realpath(RMVPE_DIR)
            if not abs_path.startswith(allowed) or not os.path.isfile(abs_path):
                self.send_error(403); return
            self._json(get_rmvpe_json(abs_path))
        elif p=="/api/log":
            self._handle_log()
        elif p=="/api/log/view":
            self._view_log(qs.get("n",["200"])[0])
        elif p=="/api/note/get":
            vid=qs.get("vid",[""])[0]
            self._json({"note":_NOTES.get(vid,""),"open_count":_OPEN_COUNTS.get(vid,0)})
        else:
            self.send_error(404)

    def _html(self,html):
        d=html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type","text/html; charset=utf-8")
        self.send_header("Content-Length",len(d))
        self.end_headers();self.wfile.write(d)

    def _json(self,obj):
        d=json.dumps(obj,ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",len(d))
        self.end_headers();self.wfile.write(d)

    def _file(self,file_path,content_type,allowed_root):
        abs_path=os.path.realpath(file_path)
        allowed_root=os.path.realpath(allowed_root)
        if not abs_path.startswith(allowed_root):self.send_error(403);return
        if not os.path.isfile(abs_path):self.send_error(404);return
        size=os.path.getsize(abs_path)
        self.send_response(200)
        self.send_header("Content-Type",content_type)
        self.send_header("Content-Length",size)
        self.send_header("Accept-Ranges","bytes")
        self.end_headers()
        with open(abs_path,"rb") as f:
            while True:
                chunk=f.read(65536)
                if not chunk:break
                self.wfile.write(chunk)

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/log":
            self._handle_log_post()
        elif parsed.path == "/api/note/set":
            self._handle_note_set()
        elif parsed.path == "/api/open-count":
            self._handle_open_count()
        else:
            self.send_error(404)

    def _handle_note_set(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode('utf-8'))
            vid = data.get('vid','')
            note = data.get('note','')
            if vid:
                _NOTES[vid] = note
                _save_json_file(NOTES_FILE, _NOTES, _notes_lock)
                # 同时写入日志
                if note.strip():
                    write_log({'action':'note','vid':vid,'text':note})
            self._json({'ok':True})
        except Exception as e:
            self._json({'ok':False,'error':str(e)})

    def _handle_open_count(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            data = json.loads(body.decode('utf-8'))
            vid = data.get('vid','')
            if vid:
                _OPEN_COUNTS[vid] = _OPEN_COUNTS.get(vid, 0) + 1
                _save_json_file(OPEN_COUNT_FILE, _OPEN_COUNTS, _count_lock)
            self._json({'ok':True,'count':_OPEN_COUNTS.get(vid,0)})
        except Exception as e:
            self._json({'ok':False,'error':str(e)})

    def _handle_log(self):
        # GET 方式，通过查询参数传递日志条目
        parsed = urllib.parse.urlparse(self.path)
        qs = urllib.parse.parse_qs(parsed.query)
        try:
            entry = {k: v[0] for k, v in qs.items()}
            # 将数字字段转换
            for field in ('from_sec','to_sec','duration'):
                if field in entry:
                    entry[field] = round(float(entry[field]), 3)
            write_log(entry)
            self._json({"ok": True})
        except Exception as e:
            self._json({"ok": False, "error": str(e)})

    def _handle_log_post(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(length)
            entry = json.loads(body.decode('utf-8'))
            write_log(entry)
            self._json({"ok": True})
        except Exception as e:
            self._json({"ok": False, "error": str(e)})

    def _view_log(self, n_str):
        """返回最近N条日志的HTML页面"""
        try:
            n = int(n_str)
        except:
            n = 200
        entries = []
        if os.path.isfile(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try: entries.append(json.loads(line))
                        except: pass
        entries = entries[-n:]
        # 统计反复播放片段
        segment_count = {}
        for e in entries:
            if e.get('action') == 'play' and 'from_sec' in e:
                key = f"{e.get('vid','')}|{e.get('from_sec',0):.1f}-{e.get('from_sec',0)+e.get('duration',0):.1f}"
                segment_count[key] = segment_count.get(key, 0) + 1
        hot_segments = sorted(segment_count.items(), key=lambda x: -x[1])
        rows = []
        for e in reversed(entries):
            action = e.get('action','')
            vid = e.get('vid','')
            ts = e.get('ts','')
            if action == 'open':
                color = '#1e3a5f'
                desc = f"▶ 打开 <b>{vid}</b> [{e.get('cat','')} / {e.get('sub','')}]"
            elif action == 'play':
                color = '#1a3a1a'
                desc = f"▶ 播放 {vid} 从 <b>{e.get('from_sec',0):.2f}s</b>"
            elif action == 'pause':
                color = '#2a2a1a'
                desc = f"⏸ 暂停 {vid} 在 <b>{e.get('at_sec',0):.2f}s</b>（播放了 {e.get('duration',0):.2f}s）"
            elif action == 'seek':
                color = '#2a1a1a'
                desc = f"➡ Seek {vid} 到 <b>{e.get('to_sec',0):.2f}s</b>"
            elif action == 'close':
                color = '#2a1a2a'
                desc = f"✕ 关闭 {vid}"
            else:
                color = '#1a1a1a'
                desc = str(e)
            rows.append(f'<tr style="background:{color}"><td style="color:#888;white-space:nowrap;padding:3px 8px;font-size:11px">{ts}</td><td style="padding:3px 8px;font-size:12px">{desc}</td></tr>')
        hot_html = ''.join(f'<tr><td style="padding:2px 8px;font-size:11px;color:#fb923c">{s.split("|")[0]}</td><td style="padding:2px 8px;font-size:11px;color:#fbbf24">{s.split("|")[1]}</td><td style="padding:2px 8px;font-size:11px;color:#f87171">{c}次</td></tr>' for s,c in hot_segments[:20] if c>1)
        html = f"""<!DOCTYPE html><html><head><meta charset=UTF-8><title>播放日志</title>
<style>body{{background:#f8f8f8;color:#1e1e1e;font-family:monospace;padding:16px}}table{{border-collapse:collapse;width:100%}}h2{{color:#0066b8;margin-bottom:12px}}h3{{color:#b85a00;margin:16px 0 8px}}</style></head><body>
<h2>播放日志（最近{n}条） &nbsp;<a href='/api/log/view?n=500' style='font-size:12px;color:#4cc9f0'>500条</a> &nbsp;<a href='/api/log/view?n=9999' style='font-size:12px;color:#4cc9f0'>全部</a></h2>
<h3>热点片段（反复播放超过1次）</h3>
<table><tr style='color:#6a7aaa'><th align=left>样本 ID</th><th align=left>时间段</th><th align=left>次数</th></tr>{hot_html or '<tr><td colspan=3 style="color:#555">(暂无)</td></tr>'}</table>
<h3>详细日志</h3>
<table>{''.join(rows)}</table>
</body></html>"""
        d = html.encode('utf-8')
        self.send_response(200)
        self.send_header('Content-Type','text/html; charset=utf-8')
        self.send_header('Content-Length', len(d))
        self.end_headers(); self.wfile.write(d)

    def log_message(self,fmt,*args):
        if "/api/img" not in str(args[0]) and "/api/rmvpe" not in str(args[0]) and "/api/log" not in str(args[0]):
            print(f"[{self.address_string()}] {args[0]}")


if __name__=="__main__":
    print("="*60)
    print(f"  MIDI 声乐浏览器  http://{HOST}:{PORT}")
    print(f"  数据目录: {BASE_DIR}")
    print(f"  vocal wav: {SEPARATED_DIR}")
    print(f"  RMVPE: {RMVPE_DIR}")
    print("="*60)
    socketserver.ThreadingTCPServer.allow_reuse_address=True
    with socketserver.ThreadingTCPServer((HOST,PORT),Handler) as httpd:
        httpd.serve_forever()
