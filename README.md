# BeatSmith v4 (“SampleSmith”) — Sample Pack Harvesting Engine

BeatSmith is a deterministic, license-aware **sample pack generator**. It harvests longer audio sources (Internet Archive or local files), slices them into **MPC / sampler-ready WAVs**, optionally generates tasteful **FX variants**, and writes full provenance metadata so you can audit where everything came from.

BeatSmith started life as a beat generator (v3). v4 keeps the best primitives (determinism, licensing, audio safety) but pivots to a clean export pipeline: **harvest → slice → trim/fade/normalize → export + manifest**.

---

## Executive Summary (v4)

- **Purpose:** sample pack generation for musicians / producers (not automatic beat composition).
- **Core output:** pad-ready WAV files, **44.1 kHz / 16-bit PCM**, with **stereo + mono** exports per sample (configurable).
- **No time-stretching or duration forcing.** BeatSmith does *not* quantize audio; it only trims/fades/normalizes.
- **Quantization is labeling only:** samples are bucketed (e.g., `q`, `e`, `six`, etc.) based on their natural duration under a BPM assumption (default **120 BPM**)—without resizing.
- **Optional FX variants:** exported to a separate folder with explicit tags in filenames.
- **Deterministic:** `--seed` (+ optional `--salt`) yields reproducible packs.
- **License-aware harvesting:** Internet Archive support with allowlist/strict mode + full provenance (manifest + credits).

---

## Output Layout

A pack directory looks like:

- `samples/clean/oneshot/(stereo|mono)/*.wav`
- `samples/clean/longform/(stereo|mono)/*.wav`
- `samples/fx/oneshot_fx/(stereo|mono)/*.wav` (FX variants derived from one-shots)
- `metadata/manifest.json` (full manifest + options)
- `metadata/credits.csv` (source attribution)
- `README_PACK.txt` (short pack note)

**File permissions:** BeatSmith attempts to set exported files to **0644** (best-effort; non-fatal on restricted environments / Windows).

---

## Modes and Randomization

BeatSmith generates clean samples using **form modes**:

- `oneshot` (default: 0.08–1.20s, onset-aligned windowing)
- `longform` (default: 30–90s)

FX variants are exported as **one-shot FX** (`oneshot_fx`) and always reference a clean one-shot in the manifest.

For each source, BeatSmith randomizes (within the per-form budgets):
- which mode(s) are used (`--form-modes`)
- how many “variations” are cut per source (`--form-variation-range`, default `1-3`)
- the exact duration within each mode’s range

This yields a pack that is **cohesive** (same source family) but **varied** (different windows and durations).

---

## Naming and Duration Buckets

Filenames include metadata such as:
- mode prefix (`oneshot/longform`)
- a short source hash
- duration in milliseconds
- duration bucket label (e.g., `q`, `e`, `six`)
- seed and a short unique suffix

Buckets are derived from the sample’s natural length relative to an assumed BPM (default 120). This is intended as a **creative hint** (e.g., “this one feels like an eighth-note-ish hit/loop”), not a promise of perfect musical grid alignment.

One-shot labels are guaranteed to cover the full range: `w`, `h`, `q`, `e`, `six`, `t32` (whole through thirty-second), at least once per bucket by default.

---

## Providers

### Internet Archive (default)
- Searches Internet Archive and downloads candidate audio.
- Uses a local cache: `~/.beatsmith/cache`
- Tracks used sources: `~/.beatsmith/used_sources.json` (to avoid repeats unless overridden)

### Local Files
- Recursively scans a directory for audio files.
- Supports license sidecar files (e.g., `myfile.wav.license`) for strict filtering.

Use:
- `--provider local --local-root /path/to/audio`

---

## Usage

BeatSmith supports explicit subcommands, but **defaults to harvest mode** if you omit one.

### Harvest (Internet Archive)

    beatsmith harvest ./out_pack --seed 1337 --c 80

Equivalent (defaults to harvest mode):

    beatsmith ./out_pack --seed 1337 --c 80

Dry-run (prints planned sources without downloading/processing):

    beatsmith harvest ./out_pack --dry-run

Strict license filtering (only allow tokens in `--license-allow`):

    beatsmith harvest ./out_pack --strict-license --license-allow "cc0,public domain,cc-by"

Avoid reusing previously harvested sources:

    beatsmith harvest ./out_pack --reuse-sources

### Harvest (Local Provider)

    beatsmith harvest ./out_pack --provider local --local-root /Volumes/Samples/RawAudio

### FX Variants

Enable FX processing for a subset of clean samples:

    beatsmith harvest ./out_pack --c-effects 40 --c-effect-randomize-count 1-3 --oneshot-fx-per-sample 1

### Output Modes

Stereo-first exports (default):

    beatsmith harvest ./out_pack --output-mode stereo

Mono-only exports:

    beatsmith harvest ./out_pack --mono

Stereo-only exports:

    beatsmith harvest ./out_pack --stereo

Stereo + mono exports:

    beatsmith harvest ./out_pack --stereo_mono

### Inspect a Pack

    beatsmith inspect ./out_pack/pack.json

---

## Common Options

- `--seed <str>`: deterministic seed
- `--salt <str>`: alternate “take” on the same seed
- `--c <int>`: how many sources to process (IA/local selection target)
- `--max-samples <int>`: cap total exported clean samples
- `--form-modes oneshot,oneshot_fx,longform` (`loop` is a deprecated alias of `oneshot_fx`)
- `--form-variation-range 1-3`
- `--oneshot-seconds 0.08-1.20`
- `--loop-seconds 1-16` (legacy; loop form now maps to oneshot_fx)
- `--longform-seconds 30-90`
- `--longform-max-ratio 0.25` (caps longform count relative to one-shots)
- `--oneshot-weight 1.0 --loop-weight 0.35 --longform-weight 0.15`
- `--oneshot-full-range / --no-oneshot-full-range`
- `--oneshot-range-min-per-bucket 1`
- `--trim-silence-db -45`
- `--fade-in-ms 2 --fade-out-ms 8`
- `--normalize-mode peak|rms`
- `--output-mode stereo|mono|stereo_mono`
- `--mono / --stereo / --stereo_mono` (convenience flags)
- `--export-stereo / --no-export-stereo` (legacy; still supported)
- `--export-mono / --no-export-mono` (legacy; still supported)
- `--provider ia|local`
- `--local-root <dir>` (only relevant for `--provider local`)
- `--zip-pack` (creates a zip of the output directory)

---

## Generation Behavior

- **Output mix:** one-shots are prioritized and always the largest category. Longform defaults to a 25% cap of one-shots (`--longform-max-ratio`).
- **Full one-shot range:** BeatSmith guarantees at least one one-shot per duration bucket (`w`, `h`, `q`, `e`, `six`, `t32`) unless disabled.
- **One-shot duration selection:** when full-range coverage is enabled, bucket durations are derived from the BPM assumption (and can exceed `--oneshot-seconds`).
- **Stereo-first:** default mode exports stereo when possible, falling back to mono only when stereo is unavailable.
- **Determinism:** `--seed` (+ `--salt`) reproduces both selection and FX decisions.
- **Reliability:** timeouts and emergency synthesis remain active to prevent zero-output runs.

---

## Changelog / Recent Changes

- One-shot dominant distribution with longform caps and a new oneshot_fx output category.
- Full one-shot duration coverage (whole through thirty-second).
- Stereo-first export policy with explicit `--mono`, `--stereo`, and `--stereo_mono` flags.
- "Loop" output bucket repurposed into one-shot FX variants; legacy flags remain supported with warnings.

---

## Installation (Recommended)

Run the installer (macOS/Linux):

    ./scripts/install.sh

The script will:
- ensure Python 3.10+ tooling is available
- install `ffmpeg` (required for reliable audio decoding in many environments)
- create a local `.venv`
- install BeatSmith in editable mode

Once complete, you should be able to run:

    beatsmith --help
    beatsmith harvest --help

---

## Testing

    pip install -e .[test]
    ruff check tests
    pytest

---

## License

MIT (engine). You must still respect third-party audio licenses you fetch and use. BeatSmith records provenance (`pack.json`, `credits.csv`), but compliance is your responsibility.
