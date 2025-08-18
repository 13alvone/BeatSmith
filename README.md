# BeatSmith v3 — Open-Web Autopilot Beat Machine
Deterministic, Internet Archive–powered beat builder that slices audible, onset-aligned chunks from public/CC audio, fits them to your time-signature map, and mixes them through percussive + texture buses with a tasteful FX chain. It supports fully-specified CLI control **and** a zero-args **Autopilot** mode (random banks) so it can surprise you.

## Highlights
- Onset-aligned measure slicing (stronger groove feel than naïve chopping)
- Multi-bus architecture: **percussive** & **texture** lanes with separate selection and mix
- Deterministic seeding (`--seed`, `--salt`) for reproducible chaos
- **Autopilot**: run with no arguments → random signature map, preset, BPM, sources, and FX
- License hygiene: prefer/allowlist CC/PD sources from Internet Archive
- Time-stretch to fit measure length: `--tempo-fit off|loose|strict`
- Built-on layering: mix on top of your own track, optional lookahead sidechain
- FX chain: compressor, 3-band EQ, reverb, tremolo, phaser, echo
- SQLite logging of runs, sources, and per-measure segments for full provenance
- Caching of downloads (`~/.beatsmith/cache` by default)
- Stems per bus/measure (optional)
- **Designed for extensibility**: future FX, source providers, and selection strategies

---

## Quick Start (Autopilot: no flags)
	# Python 3.10+ recommended. ffmpeg required by librosa/audioread.
	python3 -m venv .venv && source .venv/bin/activate
	pip install numpy scipy librosa soundfile requests mido

	# Run with zero args — BeatSmith v3 will pick everything (sig map, BPM, preset, sources, FX).
	python -m beatsmith

	# Output directory is created automatically (e.g., ./beatsmith_auto/bs_YYYYMMDD_HHMMSS_XXXX)
	# The final WAV and beatsmith_v3.db (SQLite) live there; stems if Autopilot enabled them.

---

## Installation
	# System requirements
	- Python 3.10+
	- ffmpeg (for decoding MP3/OGG/etc through librosa/audioread)
	- macOS/Linux/WSL tested; Windows should work with ffmpeg on PATH

	# Create env and install deps
	python3 -m venv .venv && source .venv/bin/activate
	pip install numpy scipy librosa soundfile requests mido

---

## Usage

### 1) Autopilot (random banks, reproducible if you capture the seed)
	# Full Autopilot
	python -m beatsmith

	# Semi-auto: you supply out_dir; everything else randomized
	python -m beatsmith ./out_autogen

	# Force Autopilot even when passing some options
	python -m beatsmith --auto --stems

BeatSmith prints the chosen `seed`/`salt`, `sig_map`, preset, BPM, query bias, FX, etc. Reuse those to reproduce the exact render.

        # Example reproducible run
        $ python -m beatsmith
        [i] Run id=42 preset=lofi BPM=78.3 sig_map=3/4(8) seed='cafe89'
        [i] FX: compressor, reverb 0.12@0.40

        # Recreate that exact beat later
        $ python -m beatsmith out "3/4(8)" --bpm 78.3 --preset lofi \
                --seed cafe89 --compress --reverb-mix 0.12 --reverb-room 0.40

### 2) Fully specified (classic control)
	# 8 measures of 4/4 at 124 BPM, boom-bap preset, stems on
	python -m beatsmith out "4/4(8)" --bpm 124 --preset boom-bap --stems

	# Mixed signatures, license allow-list strict, more sources, tasteful FX
	python -m beatsmith out "4/4(4),5/4(3),6/8(5)" --bpm 132 \
		--license-allow "cc0,cc-by,public domain" --strict-license \
		--num-sources 6 --tempo-fit strict --compress \
		--eq-low +2 --eq-mid -1 --eq-high +3 \
		--reverb-mix 0.22 --reverb-room 0.35 \
		--tremolo-rate 5 --tremolo-depth 0.35 \
		--echo-ms 320 --echo-fb 0.28 --echo-mix 0.22 \
		--stems

	# Build on your existing loop with lookahead sidechain pumping
        python -m beatsmith out "4/4(16)" --bpm 124 --build-on ./my_loop.wav \
                --sidechain 0.6 --sidechain-lookahead-ms 10 --preset edm

### 3) Dry run (plan without downloads)
        # See planned measures and candidate sources, but skip audio fetch
        python -m beatsmith out "4/4(8)" --dry-run

### 4) Inspect previous run
        # Show summary of latest beatsmith_v3.db under current directory
        beatsmith inspect

### 5) Presets
	--preset boom-bap | edm | lofi

Presets bias EQ, FX, and query terms sensibly. You can still override any knob.

---

## CLI Reference (most relevant options)
	Positional (optional in v3):
	  out_dir                     Output directory (created if missing). Autopilot sets this if omitted.
	  sig_map                     Signature program like "4/4(4),5/4(3),6/8(5)". Autopilot picks one if omitted.

	Core:
          --auto                      Force Autopilot (random banks).
          --dry-run                   Print planned sources/measures and exit.
	  --bpm FLOAT                 Global BPM (Autopilot picks a sane range per preset).
	  --seed STR                  Deterministic seed.
	  --salt STR                  Additional salt to create alternate takes deterministically.
	  --num-sources INT           Number of IA sources to pull (default 6; Autopilot randomizes).
	  --query-bias STR            IA search bias (Autopilot selects one).
	  --license-allow STR         Comma list of allowed licenses (tokens matched in license URL).
	  --strict-license            Enforce allow-list strictly (no fallback).
	  --cache-dir PATH            Download cache (~/.beatsmith/cache default).
	  --min-rms FLOAT             Audible threshold for slices (default 0.02).
	  --crossfade FLOAT           Crossfade seconds between measures (default ~0.02; Autopilot varies).
	  --tempo-fit {off,loose,strict}  Fit each slice to the exact measure duration.
	  --stems                     Write stems per bus/measure.
	  --microfill                 Add subtle end-of-measure sparkle on texture bus.
	  --preset {boom-bap,edm,lofi}
	  --verbose                   Debug logs.

	Build-on:
	  --build-on PATH             Mix underneath an existing base track.
	  --sidechain FLOAT           Duck new beat against base (0..1).
	  --sidechain-lookahead-ms FLOAT  Lookahead for sidechain envelope.

	FX (master bus):
	  --compress                  Enable compressor
	  --comp-thresh FLOAT         Threshold dB (default -18)
	  --comp-ratio  FLOAT         Ratio (default 4)
	  --comp-makeup FLOAT         Makeup gain dB (default +2)
	  --eq-low/--eq-mid/--eq-high FLOAT  3-band EQ gains (dB)
	  --reverb-mix FLOAT          0..1 (0 disables)
	  --reverb-room FLOAT         0..1
	  --tremolo-rate FLOAT        Hz (0 disables)
	  --tremolo-depth FLOAT       0..1
	  --phaser-rate FLOAT         Hz (0 disables)
	  --phaser-depth FLOAT        0..1
	  --echo-ms FLOAT             ms (0 disables)
	  --echo-fb FLOAT             0..1 feedback
	  --echo-mix FLOAT            0..1 wet mix

---

## How it Works (Pipeline)
	1) Query Internet Archive Advanced Search (audio mediatype), bias by preset/query term, filter licenses.
	2) For each picked file:
	   - Decode to mono @ 44.1kHz (librosa/audioread).
	   - Compute metrics: onset density, ZCR, spectral flatness → classify as percussive vs texture.
	3) Build the measure timeline from your signature map (e.g., 4/4(4),5/4(3),6/8(5)).
	4) For each measure & bus:
	   - Choose a source of that bus (weighted randomly).
	   - Pick an onset-aligned, sufficiently audible window.
	   - Time-stretch to measure duration (off/loose/strict).
	   - (Optional) micro-fill at tail for texture.
	5) Crossfade-concat measures per bus, balance buses, optional base mix & sidechain.
	6) Master FX (compressor → EQ → reverb → optional tremolo/phaser/echo), normalize, write WAV.
	7) Log everything to SQLite: runs, sources, per-measure segments & parameters.

---

## Data & Reproducibility
	- Deterministic seeding: same (seed, salt, CLI) → same result.
	- Provenance: beatsmith_v3.db records sources (URLs, license URLs), selected segments, and parameters per run.
	- Caching: downloads are content-addressed by URL hash under ~/.beatsmith/cache by default.

---

## Extensibility Notes (for contributors)
The codebase is laid out to make it easy to add sources, selectors, FX, and buses:

	- Source providers: isolate IA calls in a provider layer so we can add others (Freesound*, Wikimedia*, Librivox*, local dirs).
	- Selection strategies: encapsulate “choose slice start” (onset-aligned, RMS-max, beat-tracked).
	- Time fitting: current off/loose/strict adapter can grow into a strategy registry (phase-vocoder, WSOLA, rubberband).
	- FX chain: each FX is a pure function (np.ndarray -> np.ndarray). It’s straightforward to add new modules.
	- Buses: keep “perc” and “tex” as first-class; support more buses with independent FX and mixing.
	- SQLite: expand schema with migration guards (PRAGMA user_version).

\* Respect service ToS, API keys, and licensing.

See `ROADMAP.md` for concrete plugin/registry plans and task breakdowns.

---

## Testing
        # Install test and lint dependencies
        pip install -e .[test]

        # Run style checks
        ruff check tests

        # Execute unit tests
        pytest

---

## Licensing & Ethics
	- BeatSmith is an art/sonification engine. You are responsible for how you use the outputs.
	- We *prefer* Creative Commons / Public Domain assets and expose a license allow-list.
	- Always verify source licenses; IA metadata can be incomplete. Use `--strict-license` to fail fast.

---

## Troubleshooting
	- “No backend to decode MP3/OGG”: ensure `ffmpeg` is installed and on PATH.
	- “No sources available”: loosen `--license-allow` or remove `--strict-license`; try a different `--query-bias`.
	- “Artifacts with --tempo-fit strict”: try `--tempo-fit loose` (snaps to simple musical ratios).
	- “Clipping”: the exporter peak-normalizes to ~-0.8dBFS; if you overdrive echo/comp, artifacts can stack. Back off gains or reverbs.

---

## Examples
	# Groovey EDM Autopilot with stems
	python -m beatsmith --auto --preset edm --stems

	# Tight 7/8 texture piece from ambient sources
	python -m beatsmith out "7/8(12)" --bpm 126 --preset lofi \
		--query-bias "ambient texture OR pad" --tempo-fit loose --stems

---

## Maintainer Directions & Preferences (baked into the project)
	- CLI: argparse only; positional requireds kept minimal; sensible defaults.
	- Logging: use logging.* only; prefixes "[i] ", "[!]", "[DEBUG]", "[x]" throughout code.
	- Robustness: defensive error handling around network/decoding; hard caps on download size; cache with fallbacks.
	- SQLite: always persist run metadata; migrations via PRAGMA user_version in future splits.
	- No placeholders: any new module should ship **complete, working** code; tests preferred.
	- Performance: keep algorithms vectorized; only loop when necessary; consider numba/torch optional accels later.
	- Determinism: all randomness should route through a seedable RNG; log chosen parameters.
	- Tests: add property tests for envelope, onset picking, crossfade boundaries, and time-stretch invariants.

---

## Future: Random/Patterned Stretch-Squish (planned)
We will add options to *randomly* or via a *pattern* expand/compress slices **during the final mix**, independent of measure-fit:

	--stretch-mix random[:min-max][:prob]        e.g., --stretch-mix random:0.85-1.25:0.3
	--stretch-mix pattern "<seq>"                e.g., --stretch-mix pattern "1.0,0.92,1.08,1.0"
	--stretch-mix-scope {bus,global}             Apply per bus or globally
	--stretch-mix-mode {loose,strict,off}        Which stretcher to use for these mutations

See ROADMAP.md for details.

---

## License
MIT (engine). You must still respect third-party audio licenses you fetch and use.

---

