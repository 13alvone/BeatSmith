# ROADMAP.md

# BeatSmith Roadmap — toward the most open, random, and extensible beat machine

This roadmap breaks down near-term hardening, medium-term features (including **random/pattern stretch-squish** and **new FX**), and long-term platform goals (plugins, GUIs, new providers). It’s designed so the project “grows like crazy” without turning into spaghetti.

## 0. Current (v3) — Baseline we ship today
- Autopilot zero-args flow (random signature/preset/BPM/query/FX/stems)
- Onset-aligned slicing; percussive vs texture buses
- Tempo fit: off | loose (musical ratios) | strict (exact phase-vocoder)
- Master FX: compressor, 3-band EQ, reverb, tremolo, phaser, echo
- Base layering + lookahead sidechain
- SQLite run/source/segment logs; download cache; license allow-list
- Deterministic RNG via (seed, salt)

---

## 1. Near-term hardening (v3.x)
### 1.1 Stability & UX
- Add `beatsmith inspect` subcommand:
	- Print last run’s config from SQLite (replay helper).
	- Summarize source licenses, durations, and per-measure selections.
- Add `--dry-run` to resolve sources and print a plan without downloading audio.
- Add retry/backoff on IA requests; user-agent + courteous pacing.

### 1.2 Slicing/Timing
- Beat-track alignment option (librosa beat tracker) for sources that are steady-tempo loops.
- Optional quantize boundary nudges so crossfades land on energy valleys.

### 1.3 Testing
- Audio snapshot tests (hash of rendered 10-sec fixtures).
- Property tests:
	- Crossfade continuity (no clicks: last N samples + first N samples overlap constraints).
	- Time-stretch invariants (len, RMS bounds).
- Fuzz tests for sig_map parser and IA result handling.

---

## 2. Stretch-/Squish-During-Mix feature (requested) (v4)
Introduce secondary, **creative time warping** on top of measure-fit to add motion and swagger.

### 2.1 Spec (CLI)
- Random:
	--stretch-mix random[:min-max][:prob]
	Examples:
		--stretch-mix random            (defaults: 0.9–1.1, prob 0.25)
		--stretch-mix random:0.85-1.25:0.35

- Patterned:
	--stretch-mix pattern "<seq>"
	Examples:
		--stretch-mix pattern "1.00,0.92,1.08,1.00"     # bar loop
		--stretch-mix pattern "rand(0.9,1.1),1.0,1.0"   # allow random tokens

- Scope & mode:
	--stretch-mix-scope {bus,global}
	--stretch-mix-mode {loose,strict,off}  # reuse existing fit engines

### 2.2 Engine
- Implement a **warp lane** applied after per-measure assembly but before master FX.
- Core algorithm: split into N sub-windows/bar (e.g., 2–8), apply per-window stretch factors, stitch with micro-Xfades.
- Musical guards: avoid extreme factors back-to-back; maintain overall length within tolerance (±10ms).

### 2.3 Tests & Metrics
- Ensure final render length tolerance; no DC offset growth; click-free joins.
- Expose warp decisions in SQLite (new table `warps`).

---

## 3. FX Expansion Pack (v4.x)
New FX modules (pure functions) with consistent parameter naming and sensible defaults:

- Limiter (lookahead brickwall)
- Saturation/soft-clip (tanh waveshaper)
- Bitcrusher/Downsampler
- Chorus/Flanger (LFO modulated delay)
- Auto-pan & Stereo widener (MS processing)
- Convolution reverb (IR loader; license-clean IRs)
- Transient shaper (attack/sustain envelopes)
- Multi-band compressor (3 bands, optional upward comp)

CLI pattern (examples):
	--sat-drive 4.0
	--limiter  --limit-thresh -0.1 --limit-lookahead 2.0
	--chorus-rate 0.8 --chorus-depth 0.5 --chorus-mix 0.35
All FX are optional; Autopilot may randomly pick tasteful settings.

---

## 4. Source Providers & Data (v5)
Abstract the “provider” concept (keep IA as default):

- Providers:
	- Internet Archive (current)
	- Local folder (user’s sample pack)
	- Librivox (spoken word textures)
	- Wikimedia Commons (CC audio)
	- Freesound (requires key; obey ToS)
- Provider API:
	- search(query) → list[Asset]
	- fetch(asset) → audio bytes
	- license(asset) → normalized enum
- Provider selection and fallback strategy; per-provider rate caps & polite delays.

---

## 5. Plugin/Registry Architecture (v5.x)
Split into modules and registries so contributors can drop-in new algorithms without editing core:

- Registries:
	- `SliceStrategy`: onset, beat-track, RMS-max, spectral-novelty
	- `TimeFit`: off, loose, strict, rubberband (optional)
	- `FX`: map of name → callable with metadata (params schema, default ranges)
	- `Bus`: definitions with per-bus FX chains and balance defaults
	- `Provider`: IA/local/WMC/etc.

- Configuration:
	- Keep argparse for CLI.
	- Allow optional YAML scene file (`--scene path.yml`) to define complex chains, signatures, and automation.
	- SQLite migrations via `user_version`; schema defs live in `db/schema.sql`.

- Packaging:
	- Publish as `beatsmith` Python package with console scripts: `beatsmith`, `beatsmith-inspect`.

---

## 6. Performance & Quality (v6)
- Parallel fetching/decoding with bounded executors.
- Optional numba acceleration for envelopes and crossfades.
- Streaming rendering to avoid double buffers on long pieces.
- Psychoacoustic loudness normalization (ITU-R BS.1770) before limiter.
- Noise-shaped dither on 16-bit export (when limiter present).

---

## 7. Creative Intelligence (v7)
- Groove inference: detect implied kick/snare from percussive bus → derive optional MIDI lane export.
- Auto-arrangement: high-level A/B/A’ forms from entropy/novelty curves.
- Style guides: preset packs for drill, amapiano, house, garage, halftime, jungle (without genre lock-in).
- (Optional) ML seasoning:
	- Classify percussive vs texture with learned features when available.
	- CLIP-like embeddings to bias source selection to user prompt.
	- **Note**: must remain deterministic when seeded; provide offline mode.

---

## 8. Tooling & Collaboration
- CI: lint (ruff), type check (mypy), unit tests (pytest), minimal audio fixtures.
- Pre-commit hooks: whitespace, large file guard.
- Issue templates: bug / feature / provider request.
- Contribution guide: style (snake_case, logging prefix rules), PR review checklist.

---

## 9. Docs & UX
- `beatsmith inspect` and `beatsmith plan` (dry-run) docs.
- Examples gallery with reproducible seeds.
- Tutorial: how to add a new FX or provider in <100 lines.
- FAQ: licensing, reproducibility, artifacts.

---

## 10. Directions & Preferences (project ethos)
- **Argparse** interface, transparent defaults; positional requireds kept minimal.
- **No placeholders** — PRs must include working code and basic tests.
- **Logging only** (no print); prefix messages: “[i] ” info, “[!] ” warning, “[DEBUG] ” debug, “[x] ” error.
- **SQLite first** for any run/session data; avoid ad-hoc text/CSV.
- **Determinism** — every random decision must be seed-driven and recorded.
- **Extensibility** — isolate algorithms behind registries; keep pure functions for DSP modules.
- **License hygiene** — default to CC/PD; allow strict enforcement; always store provenance in DB.

---

## Appendix: Proposed DB additions
- `warps` table (for stretch-mix):
	- run_id, measure_index, bus, window_index, factor, mode
- `fx_params` table:
	- run_id, fx_name, param_json (for exact reconstruction of master chain)
- `providers` table:
	- run_id, provider_name, config_json

---

## Appendix: Security & Safety
- Enforce max download size (already in v3).
- Sanitize file names on disk; never execute remote content.
- Backoff on HTTP 429; respect robots and service-specific ToS.
- Provide a “verified license only” switch for production pipelines.

---

