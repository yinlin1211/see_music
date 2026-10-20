# see_music

`see_music` is a small local browser for checking vocal audio against MIDI
notes and RMVPE pitch tracks. It is designed for reviewing batches of
transcription or MIDI-cleaning results.

The interface provides:

- batch and quality-category navigation;
- PNG comparison previews;
- MIDI note rectangles overlaid with RMVPE pitch points;
- vocal waveform playback;
- synchronized playhead and click-to-seek;
- per-sample notes, open counts, and playback logs.

The repository contains code only. Audio, MIDI, pitch arrays, and generated
comparison images are intentionally excluded.

## Requirements

- Python 3.9 or newer
- NumPy
- A browser with Web Audio support
- Internet access for the Tone.js MIDI parser loaded from jsDelivr

Install the Python dependency:

```bash
python3 -m pip install -r requirements.txt
```

## Start

Run from the repository directory:

```bash
python3 see_music.py --base-dir /path/to/your/dataset
```

Then open <http://127.0.0.1:8766/>.

The default audio and RMVPE directories are:

```text
/path/to/your/dataset/vocal
/path/to/your/dataset/rmvpe
```

Override them when they are stored elsewhere:

```bash
python3 see_music.py \
  --base-dir /path/to/your/dataset \
  --vocal-dir /path/to/vocal \
  --rmvpe-dir /path/to/rmvpe \
  --port 8766
```

For access from another machine, bind explicitly and expose the port through
your own firewall or tunnel:

```bash
python3 see_music.py --base-dir /path/to/your/dataset --host 0.0.0.0
```

## Data Layout

The base directory must contain at least one classified image batch and its
matching MIDI batch:

```text
dataset/
├── classified_vis_<batch>_按准确率分类/
│   └── A_vocal_clean/
│       └── 90-100/
│           └── <video_id>.png
├── vocal_midi_<batch>/
│   └── <video_id>_vocal.mid
├── vocal/
│   └── any_prefix_<video_id>_vocals.mp3
└── rmvpe/
    └── any_prefix_<video_id>_vocals_f0.npy
```

The `<batch>` value must be identical in the image and MIDI directory names.
The viewer uses the final 11 characters before `_vocals` or `_vocals_f0.npy`
as the sample ID for audio and RMVPE files.

WAV files are also supported:

```text
vocal/any_prefix_<video_id>_vocals.wav
```

If both WAV and MP3 exist, the MP3 is preferred by the browser.

## Runtime Files

When you add notes, open samples, or play segments, the server writes these
files into the base directory:

```text
播放日志.jsonl
样本备注.json
打开次数.json
```

They are local review data and are ignored by Git.

## Options

```text
--base-dir PATH       Batch root directory
--vocal-dir PATH      Audio directory
--rmvpe-dir PATH      RMVPE array directory
--host HOST           Bind address, default 127.0.0.1
--port PORT           HTTP port, default 8766
```

The same settings can be provided through `SEE_MUSIC_BASE_DIR`,
`SEE_MUSIC_VOCAL_DIR`, `SEE_MUSIC_RMVPE_DIR`, `SEE_MUSIC_HOST`, and
`SEE_MUSIC_PORT`.
