# ROADMAP.md

# BeatSmith Roadmap — deterministic sample pack harvesting (v4+)

BeatSmith’s center of gravity is **v4 (“SampleSmith”)**: harvest audio sources, slice into usable samples, apply trim/fade/normalization, optionally generate FX variants, and export with provenance.

This roadmap is written to keep the project:
- deterministic and inspectable,
- license-aware,
- practical for real sampling workflows (MPC/DAW),
- extensible without becoming fragile.

---

## 0. Current (v4) — What we ship today

### 0.1 Harvest pipeline (primary workflow)
- Provider: **Internet Archive** (default)
- Provider: **Local filesystem** (recursive scan)
- Deterministic builds: `--seed` (+ optional `--salt`)
- Slice forms:
  - `oneshot` (onset-aligned windowing)
  - `loop` (fixed-length windows)
  - `longform` (fixed-length windows)
- Post-processing:
  - silence trim + pad
  - fades
  - normalization (`peak` or `rms`)
  - safety checks (finite samples; no NaN/Inf)
- Export:
  - WAV **44.1kHz / 16-bit PCM**
  - stereo + mono variants per sample (configurable)
  - best-effort **0644** permissions on outputs
- Provenance:
  - `pack.json` manifest (run options + sample metadata + sources)
  - `credits.csv` attribution export
- Optional:
  - FX variants (tasteful/aggressive/random chain styles)
  - `--zip-pack`
- Inspection:
  - `beatsmith inspect pack.json`

### 0.2 Legacy / secondary tooling (v3 remnants)
- Earlier beat-generation and pattern tooling exists in the repo (e.g., DB + track assembly logic).
- `tools/drum_patterns.py` exists as a separate MIDI pattern utility.
- v3 code is not the primary “happy path” and is treated as legacy unless explicitly revived.

---

## 1. Near-term hardening (v4.1)

### 1.1 CLI UX and predictability
- Ensure CLI help is explicit about “default-to-harvest” behavior.
- Add clearer examples for:
  - strict licensing
  - local provider usage (`--provider local --local-root ...`)
  - deterministic rebuilds (`--seed`/`--salt`)
- Add a `--print-effective-config` option to dump resolved defaults and derived ranges.

### 1.2 Provider robustness
- Internet Archive:
  - tighten retry/backoff semantics and telemetry for 429/5xx
  - improve search query composition controls (add include/exclude keyword flags)
  - optional per-run “max bytes downloaded” safety cap
- Local:
  - add include/exclude glob patterns (e.g., `--include "*.wav" --exclude "*_preview*"`)
  - optional “follow symlinks” flag (default: off)
  - optional “max files scanned” guardrail for very large trees

### 1.3 Output correctness guarantees
- Enforce safe write semantics:
  - write to temp + atomic rename for `pack.json`/`credits.csv`
  - avoid partial/corrupt artifacts on interruption
- Add validation pass after harvest:
  - confirm all manifest paths exist
  - confirm all WAVs are readable and finite
  - confirm export sample rate and subtype match expectations

### 1.4 Filename and labeling polish
- Optional filename token standardization:
  - keep current short bucket codes (`q`, `e`, `six`, etc.)
  - optionally support explicit tokens (e.g., `q=quarter`) behind a flag
- Add a `--pack-slug` option (normalized folder/zip naming) separate from `--pack-name`.

### 1.5 Logging and debuggability
- Add structured run summary at end:
  - sources processed
  - candidates skipped (low RMS, decode failures, etc.)
  - exports count by mode (oneshot/loop/longform)
  - FX variants generated
- Add `--log-json` option for machine-readable logs (CI / pipelines).

---

## 2. Quality and curation upgrades (v4.2)

### 2.1 Better slicing decisions (without time-stretching)
BeatSmith intentionally does **not** stretch audio to “fit a grid.” Improvements here are about selecting *better windows*, not warping time.

- Smarter onset window selection:
  - multi-onset fallback windows
  - transient “density” heuristic to avoid dead air or overly busy segments
- Loop boundary smoothing:
  - optional zero-crossing alignment
  - optional short crossfade at boundaries for loops
- Add “minimum transient energy” threshold for oneshots (reduce dull/empty hits).

### 2.2 Normalization and loudness options
- Add optional LUFS-style loudness normalization (off by default).
- Add DC offset removal (cheap and beneficial).
- Add clip detection + gentle limiter option (explicitly opt-in).

### 2.3 Metadata improvements
- Record more sample-level metadata into `pack.json`:
  - channel count, peak/RMS per channel
  - trim amount and post-trim duration
  - onset timestamp (for oneshots) when available
- Add a stable sample ID (hash of source + window + options) to improve dedupe and reproducibility.

### 2.4 Artifact indexing
- Optional “index file” for pack browsing:
  - a lightweight JSON/CSV listing for DAW import assistants
  - (future) UI-friendly thumbnails or wave previews (generated offline)

---

## 3. FX expansion and control (v4.3)

### 3.1 Safer, more musical defaults
- Add “guardrails” to FX params so outputs remain usable:
  - reverb wet bounds
  - echo mix bounds
  - prevent runaway feedback/decay
- Add an explicit “FX preset” concept:
  - `--fx-preset airy` / `--fx-preset grime` / `--fx-preset tape` (examples)

### 3.2 FX pipeline extensibility
- Standardize FX modules as pure functions with explicit parameter schemas.
- Add a “dry/wet mix per effect” policy that is consistent across FX types.
- Add a per-run FX report (counts by FX type and average params).

---

## 4. Provider expansion (v5)

### 4.1 Additional sources (optional)
- Add new providers only if they can be:
  - license-auditable,
  - stable to operate,
  - deterministic enough for reproducible builds.

Examples (subject to feasibility):
- public-domain libraries with structured metadata
- user-provided URL lists (with explicit allowlist and caching)

### 4.2 Unified provenance model
- Standardize fields across providers:
  - stable source identifiers
  - license metadata and confidence level
  - retrieval timestamp + retrieval method

---

## 5. Plugin/registry architecture (v5.x)

Goal: allow new providers / slicers / post-processors / FX packs without modifying core logic.

- Provider plugin interface (search, fetch, metadata)
- Slicer strategy interface (oneshot/loop/longform could become strategies)
- Post-processing chain registry (trim/fade/normalize/validate)
- FX pack registry (preset schemas + parameter ranges)
- “Pack export targets” registry (future: Ableton/NI/MPC helpers)

---

## 6. Performance and reliability (v6)

- Parallelism:
  - concurrent downloads (IA)
  - parallel slicing/export (bounded worker pool)
- Memory efficiency:
  - stream decode where possible
  - avoid duplicating large arrays when exporting stereo/mono
- Caching improvements:
  - content-addressed cache keys
  - cache eviction policy (`--cache-max-gb`)
- CI hardening:
  - deterministic golden tests with fixed seeds
  - decode tests that do not depend on network

---

## 7. Docs and UX (ongoing)

- “Getting started” walkthrough with:
  - seed/salt determinism examples
  - strict licensing examples
  - local provider examples
- Add a “Pack Quality Checklist” section to README:
  - what knobs to turn for more transient hits vs more ambience
- CONTRIBUTING guidelines:
  - how to add a provider
  - how to add an FX module
  - how to add tests that are stable in CI

---

## 8. Security and safety (non-negotiable)

- Treat all remote content as untrusted:
  - bounded downloads
  - bounded decode time
  - robust exception handling and skip policies
- File safety:
  - no directory traversal on outputs
  - atomic writes for manifests and credits
- Licensing:
  - strict mode should remain conservative
  - provenance should be first-class and difficult to bypass accidentally

---

## Notes on scope (important)

- BeatSmith does **not** aim to be a DAW, sequencer, or automatic composer in v4+.
- BeatSmith does **not** time-stretch samples to “fit” musical values. Duration bucketing is a **label**, not a warp.
- If beat-generation is revived, it should re-enter as a separate, clearly-scoped module/tool so the harvester remains stable.

