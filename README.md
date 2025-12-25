# BeatSmith v4 (“SampleSmith”) — MPC One+ Sample Harvesting Engine
BeatSmith began as a deterministic, license-aware beat generator. **v4 pivots the project into a sample harvesting and preparation engine** that outputs **large, curated swaths of pad-ready WAV samples** for the AKAI MPC One+. Instead of rendering full beats, the new deliverable is a **high-volume, high-quality sample library** you import into hardware to build your own programs, sequences, and arrangements.

BeatSmith v4 retains the strongest primitives from v3—deterministic acquisition, reproducible randomness, audio decoding, onset-based slicing, normalization utilities, FX chains, and provenance logging—while replacing beat assembly with a clean, MPC-friendly export pipeline.

---

## Executive Summary (v4)
- **Purpose shift:** from “automatic beat composer” → **“sample pack generator for musicians.”**
- **Core output:** pad-ready WAV files, **44.1 kHz / 16-bit PCM**, **stereo + mono** exports for every sample unit.
- **No time-stretching or duration forcing.** Only trim silence, add tiny fades, and normalize.
- **Optional FX variants** are exported alongside clean samples, in a separate folder with explicit naming.
- **Deterministic + license-aware** Internet Archive harvesting, with **full provenance** (manifest + optional SQLite).
- **Quantization is labeling only:** durations are bucketed (q/e/six/t2, etc.) without resizing.

---

## Quick Start (v4 harvest)
After installing dependencies, generate a sample pack:

    # Minimal harvest (deterministic if you capture seed/salt)
    beatsmith harvest ./out_pack --seed cafe89 --c 10

    # Dry run: print plan only
    beatsmith harvest ./out_pack --dry-run

---

## Installation
Python 3.10+ and `ffmpeg` are required. Use a virtual environment to keep dependencies isolated:

    # System requirements
    - Python 3.10+
    - ffmpeg (for decoding MP3/OGG/etc)
        - macOS: `brew install ffmpeg`
        - Debian/Ubuntu: `sudo apt install ffmpeg`
        - Windows: https://ffmpeg.org/download.html or `choco install ffmpeg`
    - macOS/Linux/WSL tested; Windows should work with ffmpeg on PATH

    # Verify ffmpeg is available
    ffmpeg -version

    # Create env and install the project
    python3 -m venv .venv
    source .venv/bin/activate  # macOS/Linux
    .\.venv\Scripts\activate  # Windows PowerShell
    pip install -e .  # or `pip install .`

---

## What v4 Does (Pipeline)
1) **Acquire** audio from Internet Archive with license filtering + deterministic sampling.
2) **Decode** to 44.1 kHz internal SR, preserving stereo when available.
3) **Extract** one-shot / loop / long-form segments (no time-stretching).
4) **Trim** silence naturally with pad ms; apply tiny fades.
5) **Normalize** (peak or RMS) to consistent loudness with safety ceiling.
6) **Export** clean stereo + mono WAVs (44.1 kHz / 16-bit PCM).
7) **Optionally apply FX** and export variants to a separate folder.
8) **Log provenance** in `pack.json` and optional SQLite DB.

---

## Output Layout (MPC-friendly)
```
OUT_DIR/
  pack.json                 # full manifest (provenance + processing)
  credits.csv               # IA attribution + license URL + retrieval date
  README_PACK.txt           # quick usage + attribution guidance
  samples/
    clean/
      oneshot/
        stereo/*.wav
        mono/*.wav
      loop/
        stereo/*.wav
        mono/*.wav
      longform/
        stereo/*.wav
        mono/*.wav
    fx/
      <fxchain_tag>/         # e.g., “rvb_eq”, “comp_sat”, “echo_mod”
        oneshot/stereo/*.wav
        oneshot/mono/*.wav
        loop/stereo/*.wav
        loop/mono/*.wav
        longform/stereo/*.wav
        longform/mono/*.wav
  metadata/
    waveforms/              # optional PNG waveform renders
    previews/               # optional short MP3 previews
  db/
    samplesmith_v4.db       # sqlite (optional)
```

---

## Filename Conventions
Clean sample naming:

```
<formPrefix>_<srcTag>_<durMs>ms_<grp>_<bpmTag>_<seedTag>_<id>.wav
```

- `formPrefix`: `os` (one-shot), `lp` (loop), `lf` (long-form)
- `srcTag`: short hash of IA identifier + file name
- `durMs`: post-trim duration (ms)
- `grp`: nearest bucket (t2/six/e/q/h/w/b2/b4/...) or `free`
- `bpmTag`: `bpm120a` (assumed) or `bpm128i` (inferred if enabled)
- `seedTag`: `s<seedShort>` (with optional salt)
- `id`: unique short uuid

FX variants add a suffix:

```
...__fx-rvb_eq.wav
```

Stereo/mono are indicated by folder path to keep filenames short.

---

## CLI Reference (v4 Harvest)
Primary command:

    beatsmith harvest [OUT_DIR]

### Core
- `--seed STR` (deterministic)
- `--salt STR` (deterministic alternate take)
- `--cache-dir PATH`
- `--license-allow CSV`
- `--strict-license`
- `--query-bias STR`
- `--reuse-sources`
- `--c INT` (default 100) — total IA source files to download/process
- `--max-samples INT` — hard cap on total exported sample units
- `--min-rms FLOAT` — audible gating threshold
- `--dry-run`
- `--verbose`

### Form selection (randomized per source)
- `--form-modes CSV` (oneshot, loop, longform; default all)
- `--form-variation-range A-B` (default 1-3)
- `--oneshot-seconds A-B` (default 0.08-1.20)
- `--loop-seconds A-B` (default 1.0-16.0)
- `--longform-seconds A-B` (default 30-90)

### Trim + normalize
- `--trim-silence-db FLOAT` (default -45)
- `--trim-pad-ms INT` (default 12)
- `--fade-in-ms INT` (default 2)
- `--fade-out-ms INT` (default 8)
- `--normalize-mode {peak,rms}` (default peak)
- `--normalize-peak-db FLOAT` (default -1.0)
- `--normalize-rms FLOAT` (if rms mode)

### Length labeling (no stretching)
- `--label-bpm-assumption FLOAT` (default 120.0)
- `--label-tolerance FLOAT` (default 0.12)
- `--emit-bpm-infer` (include inferred bpm if confident)
- `--bpm-confidence-min FLOAT` (default 0.55)

### FX variations
- `--c-effects INT` (default 0 => all)
- `--c-effect-randomize-count A-B` (default "0-5"; max 50)
- `--c-effect-variations` (0 or CSV list: reverb,compression,eq,modulation,echo,phaser,tremolo...)
- `--fx-prob FLOAT` (default 1.0)
- `--fx-chain-style {tasteful,aggressive,random}` (default tasteful)
- `--fx-room {small,mid,large}` (optional)
- `--fx-tag-filenames` (default true)

### Output
- `--export-mono` (default true)
- `--export-stereo` (default true)
- `--stereo-prefer` (default true; if mono source, stereo export is dual-mono)
- `--pack-name STR` (default auto `SamplePack_YYYYMMDD_HHMMSS_seed`)
- `--write-manifest` (default true)
- `--write-credits` (default true)
- `--zip-pack` (optional)

---

## Length Grouping (Label Only)
Quantization is **labeling only**—no time-stretching or duration forcing.

- `sec_per_beat = 60 / bpm_assumption`
- `beats = duration_seconds / sec_per_beat`
- Nearest buckets (t2, six, e, q, h, w, b2, b4, …)
- If outside tolerance, label `free`
- Long-form always labeled `lf`

---

## Provenance + Compliance
BeatSmith v4 writes:
- `pack.json` (manifest): run info, source IDs/URLs/licenses, offsets, processing, exports
- `credits.csv`: attribution with license URLs + retrieval timestamps
- Optional SQLite DB with runs/sources/samples/variants/exports

---

## Acceptance Criteria (v4)
Running:

    beatsmith harvest outdir --seed X --c 10

Must yield:
- ≤ `max-samples` exported sample units
- Each unit has **1 stereo WAV + 1 mono WAV**
- Optional FX variants also in stereo+mono
- **No time-stretching** used
- Natural silence trim with pad ms
- Subtle, consistent fades
- Filenames include duration ms, grp label, bpm tag, and FX tag if applicable
- `pack.json` and `credits.csv` present with IA attribution

---

## Testing
    # Install test and lint dependencies
    pip install -e .[test]

    # Run style checks
    ruff check tests

    # Execute unit tests
    pytest

---

## License
MIT (engine). You must still respect third-party audio licenses you fetch and use.
