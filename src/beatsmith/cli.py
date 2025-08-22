import argparse
import json
import logging
import os
import sys
import time
import math
import uuid
import zipfile
import shutil
import subprocess
from typing import Any, Dict, Optional, Tuple, Set

import numpy as np
import soundfile as sf

from . import li, lw, le, log
from .audio import (
    parse_sig_map, seconds_per_measure, seeded_rng, normalize_peak,
    load_audio_file, pick_sources, build_measures, assemble_track, TARGET_SR,
    preview_sources, safe_audio, pick_onset_aligned_window, stack_slices,
)
from .fx import (
    compressor, eq_three_band, reverb_schroeder, tremolo, phaser, echo, lookahead_sidechain
)
from .providers.internet_archive import InternetArchiveProvider
from .providers.local import LocalProvider
from .pattern_adapter import apply_pattern_to_quantized_slices

USED_SOURCES_REGISTRY = os.path.expanduser("~/.beatsmith/used_sources.json")
FX_CHANCE = 0.2  # probability gate for applying any given effect

def load_used_sources() -> Set[str]:
    try:
        with open(USED_SOURCES_REGISTRY, "r") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return set(str(x) for x in data)
    except Exception:
        pass
    return set()

def save_used_sources(entries: Set[str]) -> None:
    os.makedirs(os.path.dirname(USED_SOURCES_REGISTRY), exist_ok=True)
    with open(USED_SOURCES_REGISTRY, "w") as fh:
        json.dump(sorted(entries), fh)

# ---------------------------- Presets ----------------------------
def apply_preset(args: argparse.Namespace):
    pres = (args.preset or "").lower().strip()
    if not pres:
        return
    li(f"Applying preset: {pres}")
    if pres == "boom-bap":
        if args.eq_low == 0.0: args.eq_low = +3.0
        if args.eq_mid == 0.0: args.eq_mid = -1.0
        if args.eq_high == 0.0: args.eq_high = +1.0
        if not args.compress: args.compress = True
        if args.query_bias is None: args.query_bias = "drums OR percussion OR vinyl OR breakbeat"
    elif pres == "edm":
        if args.eq_low == 0.0: args.eq_low = +4.0
        if args.eq_mid == 0.0: args.eq_mid = -2.0
        if args.eq_high == 0.0: args.eq_high = +2.0
        if not args.compress: args.compress = True
        if args.echo_ms == 0.0: args.echo_ms = 300.0
        if args.echo_mix == 0.0: args.echo_mix = 0.2
        if args.query_bias is None: args.query_bias = "electronic OR drum machine OR loop"
    elif pres == "lofi":
        if args.eq_low == 0.0: args.eq_low = +2.0
        if args.eq_mid == 0.0: args.eq_mid = +1.5
        if args.eq_high == 0.0: args.eq_high = -2.0
        if args.query_bias is None: args.query_bias = "jazz OR vinyl OR mellow OR ambient"
    else:
        lw(f"Unknown preset '{args.preset}', ignoring.")

# ---------------------------- Autopilot ----------------------------
def autopilot_config(
    rng,
    num_sounds_range: Tuple[int, int] = (15, 30),
    force_reverb: bool = False,
    force_tremolo: bool = False,
) -> Dict[str, Any]:
    long_sig_opts = ["4/4(16)", "3/4(16)", "7/8(16)", "5/4(16)"]
    short_sig_opts = [
        "4/4(8)",
        "4/4(4),5/4(3),6/8(5)",
        "3/4(8)",
        "7/8(8)",
        "5/4(8)",
    ]
    sig_opts = long_sig_opts * 2 + short_sig_opts
    sig_str = rng.choice(sig_opts)
    preset = rng.choice(["boom-bap", "edm", "lofi"])
    if preset == "boom-bap":
        bpm = rng.uniform(80, 96)
    elif preset == "edm":
        bpm = rng.uniform(120, 136)
    else:
        bpm = rng.uniform(60, 84)
    sig_map = parse_sig_map(sig_str)
    est_len = sum(seconds_per_measure(bpm, ms.numer, ms.denom) * ms.count for ms in sig_map)
    if est_len < 60.0:
        reps = int(math.ceil(60.0 / est_len))
        for ms in sig_map:
            ms.count *= reps
    out_dir = os.path.join(
        "beatsmith_auto",
        f"bs_{time.strftime('%Y%m%d_%H%M%S')}_{rng.randrange(10000):04d}"
    )
    reverb_on = force_reverb or rng.random() < 0.1
    tremolo_on = force_tremolo or rng.random() < 0.1
    cfg: Dict[str, Any] = {
        "out_dir": out_dir,
        "sig_map": sig_map,
        "preset": preset,
        "bpm": float(bpm),
        "num_sources": rng.randint(4, 8),
        "num_sounds": rng.randint(num_sounds_range[0], num_sounds_range[1]),
        "crossfade": rng.uniform(0.01, 0.04),
        "stems": rng.random() < 0.3,
        "microfill": rng.random() < 0.5,
        "beat_align": rng.random() < 0.5,
        "mix_layers": 6,
        "tempo_fit": rng.choice(["off", "loose", "strict"]),
        "compress": rng.random() < 0.5,
        "eq_low": rng.uniform(-3, 3),
        "eq_mid": rng.uniform(-3, 3),
        "eq_high": rng.uniform(-3, 3),
        "reverb_mix": rng.uniform(0.05, 0.25) if reverb_on else 0.0,
        "reverb_room": rng.uniform(0.2, 0.6) if reverb_on else 0.0,
        "tremolo_rate": rng.uniform(3.0, 7.0) if tremolo_on else 0.0,
        "tremolo_depth": rng.uniform(0.2, 0.6) if tremolo_on else 0.0,
        "phaser_rate": rng.uniform(0.1, 1.0) if rng.random() < 0.2 else 0.0,
        "phaser_depth": rng.uniform(0.3, 0.7),
        "echo_ms": rng.uniform(200, 500) if rng.random() < 0.4 else 0.0,
        "echo_fb": rng.uniform(0.2, 0.5),
        "echo_mix": rng.uniform(0.1, 0.4),
    }
    return cfg


def stack_fx(y: np.ndarray, args: argparse.Namespace, rng) -> np.ndarray:
    """Apply per-groove compression/EQ with probability gating."""
    if args.compress and rng.random() < FX_CHANCE:
        li("Applying per-stack compressor...")
        y = compressor(
            y,
            TARGET_SR,
            thresh_db=args.comp_thresh,
            ratio=args.comp_ratio,
            makeup_db=args.comp_makeup,
        )
    if any(abs(g) > 1e-6 for g in [args.eq_low, args.eq_mid, args.eq_high]) and rng.random() < FX_CHANCE:
        li(f"Applying per-stack EQ: low={args.eq_low} mid={args.eq_mid} high={args.eq_high}")
        y = eq_three_band(
            y,
            TARGET_SR,
            low_db=args.eq_low,
            mid_db=args.eq_mid,
            high_db=args.eq_high,
        )
    return y

# ---------------------------- CLI ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BeatSmith v3: pull CC/PD audio from selected providers, onset-align slices per signature map, run percussive+texture buses, FX, and render a beat.",
        epilog="FX order: per-groove compressor/EQ then post-loop reverb/tremolo/phaser/echo (each ~20%% chance).",
    )
    p.add_argument("out_dir", nargs="?", default=None, help="Output directory (created if missing).")
    p.add_argument("sig_map", nargs="?", default=None, type=parse_sig_map, help="Signature map like '4/4(4),5/4(3),6/8(5)'.")
    p.add_argument("--bpm", type=float, default=120.0, help="Beats per minute (default: 120).")
    p.add_argument("--seed", type=str, default=None, help="Deterministic seed.")
    p.add_argument("--salt", type=str, default=None, help="Additional salt for alternate takes.")
    # Deprecated, but keep for backward compatibility
    p.add_argument("--num-sources", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--query-bias", type=str, default=None, help="Optional search bias string.")
    p.add_argument("--provider", choices=["ia", "local"], default="ia", help="Audio provider (default: ia).")
    p.add_argument("--license-allow", type=str, default="cc0,creativecommons,public domain,publicdomain,cc-by,cc-by-sa", help="Comma list of license tokens allowed.")
    p.add_argument("--strict-license", action="store_true", help="Enforce license allow-list strictly (no fallback).")
    p.add_argument("--reuse-sources", action="store_true", help="Allow reusing sources from previous runs.")
    p.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.beatsmith/cache"), help="Download cache directory.")
    p.add_argument("--min-rms", type=float, default=0.02, help="Minimum RMS for audible slice.")
    p.add_argument(
        "--num-sounds",
        type=str,
        default=None,
        help="Total slices or range like '20-40' (default: random 15-30).",
    )
    p.add_argument(
        "--mix-layers",
        type=int,
        default=6,
        help="Number of independent mixes to stack for denser output (default: 6).",
    )
    p.add_argument("--crossfade", type=float, default=0.02, help="Seconds of crossfade between measures.")
    p.add_argument("--tempo-fit", choices=["off","loose","strict"], default="strict", help="Time-stretch mode to fit global measure length.")
    p.add_argument("--stems", action="store_true", help="Write stems per bus/measure.")
    p.add_argument("--microfill", action="store_true", help="Enable tiny end-of-measure fills on texture bus.")
    p.add_argument("--beat-align", action="store_true", help="Align slices to detected beats instead of onsets.")
    p.add_argument(
        "--no-boundary-refine",
        dest="boundary_refine",
        action="store_false",
        help="Disable RMS-based refinement of slice boundaries.",
    )
    p.set_defaults(boundary_refine=True)
    p.add_argument("--preset", type=str, default=None, help="Preset: boom-bap | edm | lofi")
    p.add_argument("--auto", action="store_true", help="Autopilot mode: randomize signature map, BPM, preset, sources, FX.")
    p.add_argument("--force-reverb", action="store_true", help="Autopilot: always include reverb")
    p.add_argument("--force-tremolo", action="store_true", help="Autopilot: always include tremolo")
    p.add_argument("--dry-run", action="store_true", help="Print planned sources and measures without downloading audio.")
    p.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    p.add_argument("--build-on", type=str, default=None, help="Path to an existing base track to mix under.")
    p.add_argument("--sidechain", type=float, default=0.0, help="Sidechain duck amount 0..1 against base (default 0).")
    p.add_argument("--sidechain-lookahead-ms", type=float, default=0.0, help="Lookahead (ms) for sidechain.")
    p.add_argument("--compress", action="store_true", help="Per-groove compressor (~20%% chance).")
    p.add_argument("--comp-thresh", type=float, default=-18.0)
    p.add_argument("--comp-ratio", type=float, default=4.0)
    p.add_argument("--comp-makeup", type=float, default=2.0)
    p.add_argument("--eq-low", type=float, default=0.0, help="Low shelf gain dB (per-groove, ~20%% chance)")
    p.add_argument("--eq-mid", type=float, default=0.0, help="Mid band gain dB (per-groove, ~20%% chance)")
    p.add_argument("--eq-high", type=float, default=0.0, help="High shelf gain dB (per-groove, ~20%% chance)")
    p.add_argument("--reverb-mix", type=float, default=0.0, help="0..1 wet mix post-loop (~20%% chance)")
    p.add_argument("--reverb-room", type=float, default=0.3, help="0..1 room size-ish")
    p.add_argument("--tremolo-rate", type=float, default=0.0, help=">0 to enable post-loop (Hz, ~20%% chance)")
    p.add_argument("--tremolo-depth", type=float, default=0.5, help="0..1")
    p.add_argument("--phaser-rate", type=float, default=0.0, help=">0 to enable post-loop (Hz, ~20%% chance)")
    p.add_argument("--phaser-depth", type=float, default=0.6, help="0..1")
    p.add_argument("--echo-ms", type=float, default=0.0, help=">0 to enable post-loop (ms delay, ~20%% chance)")
    p.add_argument("--echo-fb", type=float, default=0.3, help="Feedback 0..1")
    p.add_argument("--echo-mix", type=float, default=0.25, help="Wet mix 0..1")

    # Pattern options
    p.add_argument("--pattern-enable", action="store_true", help="Enable drum pattern overlay")
    p.add_argument("--pattern-db", type=str, default=None, help="Path to drum pattern SQLite DB")
    p.add_argument("--pattern-sig", type=str, default=None, help="Pattern signature filter")
    p.add_argument("--pattern-bars", type=int, default=None, help="Pattern bars filter")
    p.add_argument("--pattern-subdivs", type=int, default=None, help="Pattern subdivision filter")
    p.add_argument("--pattern-swing", type=float, default=0.0, help="Swing amount 0..1 for pattern")
    p.add_argument("--pattern-humanize-ms", type=float, default=0.0, help="Timing humanization in ms")
    p.add_argument("--pattern-vel-scale", type=float, default=1.0, help="Velocity scale for pattern hits")
    p.add_argument(
        "--pattern-lane-map",
        type=str,
        default=None,
        help="Comma list mapping pattern lanes to sample keys, e.g. 'kick=perc,snare=perc'",
    )
    p.add_argument(
        "--pattern-silence-missing",
        action="store_true",
        help="Silence missing lanes instead of warning",
    )
    return p

def main():
    args = build_parser().parse_args()
    if "--num-sources" in sys.argv:
        lw("'--num-sources' is deprecated; it will be removed in a future release.")
    # Parse num_sounds argument which may be a single integer or range "a-b"
    ns_val: Optional[int] = None
    ns_range = (15, 30)
    if args.num_sounds:
        if "-" in args.num_sounds:
            a, b = args.num_sounds.split("-", 1)
            ns_range = (int(a), int(b))
        else:
            ns_val = int(args.num_sounds)
            ns_range = (ns_val, ns_val)
    args.num_sounds = ns_val
    args.num_sounds_range = ns_range
    if args.verbose:
        log.setLevel(logging.DEBUG)
        li("Verbose logging enabled.")
    seed = args.seed or f"auto-{time.time_ns()}"
    rng = seeded_rng(seed, args.salt)
    if args.auto or args.out_dir is None or args.sig_map is None:
        auto = autopilot_config(
            rng,
            num_sounds_range=args.num_sounds_range,
            force_reverb=args.force_reverb,
            force_tremolo=args.force_tremolo,
        )
        if args.auto or args.out_dir is None:
            args.out_dir = auto["out_dir"]
        if args.auto or args.sig_map is None:
            args.sig_map = auto["sig_map"]
        for k, v in auto.items():
            if k in ("out_dir", "sig_map"):
                continue
            cur = getattr(args, k, None)
            if args.auto or cur in (None, 0, 0.0, False, "off"):
                setattr(args, k, v)
    if args.num_sources is None:
        args.num_sources = 6
    if args.num_sounds is None:
        args.num_sounds = rng.randint(*args.num_sounds_range)
    args.seed = seed
    apply_preset(args)
    if args.provider == "local":
        provider = LocalProvider()
    else:
        provider = InternetArchiveProvider()
    if args.dry_run:
        measures = build_measures(args.sig_map)
        total_sec = sum(seconds_per_measure(args.bpm, n, d) for n, d in measures)
        li(f"Total measures: {len(measures)}  est length ≈ {total_sec:.1f}s")
        allow_tokens = [t.strip() for t in (args.license_allow or "").split(",") if t.strip()]
        strict = bool(args.strict_license)
        li("Planning sources...")
        plans = preview_sources(
            provider,
            rng,
            wanted=max(2, args.num_sources),
            query_bias=args.query_bias,
            allow_tokens=allow_tokens,
            strict=strict,
        )
        for idx, pinfo in enumerate(plans, 1):
            title = pinfo.get("title") or pinfo.get("file") or "unknown"
            li(f"Source {idx}: {title} ({pinfo.get('url')})")
        li("Dry run complete. No audio downloaded or files written.")
        return
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    # Unique base name for all outputs
    ts = time.strftime("%Y%m%d_%H%M%S")
    base = f"bs_{ts}_{uuid.uuid4().hex[:8]}"
    stems_root = os.path.join(out_dir, base + "_stems")
    perc_dir = os.path.join(stems_root, "perc")
    tex_dir = os.path.join(stems_root, "tex")
    os.makedirs(perc_dir, exist_ok=True)
    os.makedirs(tex_dir, exist_ok=True)
    stems_dirs = {"perc": perc_dir, "tex": tex_dir}
    sig_str = ",".join(f"{ms.numer}/{ms.denom}({ms.count})" for ms in args.sig_map)
    li(
        f"Run preset={args.preset or 'none'} BPM={args.bpm} sig_map={sig_str} seed='{seed}' salt='{args.salt or ''}'"
    )
    fx_bits = []
    if args.compress: fx_bits.append("compressor")
    if any(abs(x) > 1e-6 for x in [args.eq_low, args.eq_mid, args.eq_high]):
        fx_bits.append(f"eq[{args.eq_low:.1f},{args.eq_mid:.1f},{args.eq_high:.1f}]")
    if args.reverb_mix > 0.0:
        fx_bits.append(f"reverb {args.reverb_mix:.2f}@{args.reverb_room:.2f}")
    if args.tremolo_rate > 0.0:
        fx_bits.append(f"trem {args.tremolo_rate:.1f}Hz@{args.tremolo_depth:.2f}")
    if args.phaser_rate > 0.0:
        fx_bits.append(f"phaser {args.phaser_rate:.1f}Hz@{args.phaser_depth:.2f}")
    if args.echo_ms > 0.0:
        fx_bits.append(f"echo {args.echo_ms:.0f}ms fb={args.echo_fb:.2f} mix={args.echo_mix:.2f}")
    if fx_bits:
        li("FX: " + ", ".join(fx_bits))
    allow_tokens = [t.strip() for t in (args.license_allow or "").split(",") if t.strip()]
    strict = bool(args.strict_license)
    li("Selecting sources...")
    existing_used = load_used_sources()
    used_for_pick = set() if args.reuse_sources else existing_used.copy()
    sources = pick_sources(
        None,
        None,
        rng,
        provider,
        wanted=max(2, args.num_sources),
        query_bias=args.query_bias,
        allow_tokens=allow_tokens,
        strict=strict,
        cache_dir=args.cache_dir,
        used_registry=used_for_pick,
    )
    if not sources:
        le("No sources available, aborting.")
        sys.exit(2)
    tokens = {s.url or s.ia_identifier for s in sources if s.url or s.ia_identifier}
    existing_used.update(tokens)
    save_used_sources(existing_used)
    measures = build_measures(args.sig_map)
    total_sec = sum(seconds_per_measure(args.bpm, n, d) for n, d in measures)
    li(f"Total measures: {len(measures)}  est length ≈ {total_sec:.1f}s")
    pattern_mix = None
    if args.pattern_enable:
        if not args.pattern_db or not os.path.isfile(args.pattern_db):
            le(f"Pattern DB not found: {args.pattern_db}")
        else:
            script = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools", "drum_patterns.py"))
            cmd = [sys.executable, script, "sample", args.pattern_db]
            if args.pattern_sig:
                cmd += ["--signature", args.pattern_sig]
            if args.pattern_bars is not None:
                cmd += ["--bars", str(args.pattern_bars)]
            if args.pattern_subdivs is not None:
                cmd += ["--subdivision", str(args.pattern_subdivs)]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, check=False)
                if res.returncode != 0:
                    lw("Pattern sample failed; continuing without pattern")
                else:
                    pat = json.loads(res.stdout.strip() or "{}")
                    if pat:
                        lane_map = {}
                        if args.pattern_lane_map:
                            for part in args.pattern_lane_map.split(","):
                                if not part:
                                    continue
                                if "=" in part:
                                    k, v = part.split("=", 1)
                                elif ":" in part:
                                    k, v = part.split(":", 1)
                                else:
                                    continue
                                lane_map[k.strip()] = v.strip()
                        step_dur_s = (60.0 / max(args.bpm, 1e-6)) * (
                            (int(pat.get("signature", "4/4").split("/")[0]) * 4.0 / int(pat.get("signature", "4/4").split("/")[1]))
                            / float(pat.get("subdivision", 16))
                        )
                        sample_keys = set(lane_map.values()) if lane_map else {"perc"}
                        sample_registry: Dict[str, list[np.ndarray]] = {}
                        for key in sample_keys:
                            pool = [s for s in sources if s.bus == key]
                            if not pool:
                                if args.pattern_silence_missing:
                                    sample_registry[key] = [np.zeros(int(step_dur_s * TARGET_SR), dtype=np.float32)]
                                else:
                                    lw(f"No sources for pattern lane '{key}'")
                                continue
                            sample_registry[key] = []
                            for src in pool:
                                try:
                                    s0, s1, _ = pick_onset_aligned_window(
                                        src.y,
                                        src.sr,
                                        step_dur_s,
                                        rng=rng,
                                        min_rms=args.min_rms,
                                        refine_edges=args.boundary_refine,
                                    )
                                    seg = src.y[s0:s1].astype(np.float32)
                                    sample_registry[key].append(seg)
                                except Exception:
                                    continue
                        if sample_registry:
                            placements = apply_pattern_to_quantized_slices(
                                pat,
                                sample_registry,
                                args.bpm,
                                swing=args.pattern_swing,
                                humanize_s=args.pattern_humanize_ms / 1000.0,
                                velocity_scale=args.pattern_vel_scale,
                                lane_map=lane_map if lane_map else None,
                                rng=rng,
                            )
                            track_len = int(total_sec * TARGET_SR)
                            pattern_slices = []
                            for pl in placements:
                                idx = int(pl.start_s * TARGET_SR)
                                if idx >= track_len:
                                    continue
                                pattern_slices.append((pl.data, idx))
                            pattern_mix = stack_slices(track_len, pattern_slices)
                    else:
                        lw("No pattern returned; continuing without pattern")
            except Exception as e:
                lw(f"Pattern sampling failed: {e}")
    align_name = "beat" if args.beat_align else "onset"
    li(f"Assembling {align_name}-aligned measures (perc + tex buses) across {args.mix_layers} layers...")
    tex_gain = 10 ** (-3.0 / 20.0)
    layers = []
    for layer in range(args.mix_layers):
        mix_perc, mix_tex = assemble_track(
            None,
            None,
            sources,
            measures,
            bpm=args.bpm,
            rng=rng,
            min_rms=args.min_rms,
            crossfade_s=args.crossfade,
            tempo_mode=args.tempo_fit,
            stems_dirs=stems_dirs if layer == 0 else {},
            microfill=args.microfill,
            beat_align=bool(args.beat_align),
            refine_boundaries=args.boundary_refine,
            num_sounds=args.num_sounds,
        )
        L = max(len(mix_perc), len(mix_tex))
        if len(mix_perc) < L: mix_perc = np.pad(mix_perc, (0, L-len(mix_perc)))
        if len(mix_tex)  < L: mix_tex  = np.pad(mix_tex,  (0, L-len(mix_tex)))
        layers.append(mix_perc + tex_gain * mix_tex)
    if pattern_mix is not None:
        layers.append(pattern_mix)
    max_len = max(len(l) for l in layers)
    mix = np.zeros(max_len, dtype=np.float32)
    for l in layers:
        if len(l) < max_len:
            l = np.pad(l, (0, max_len - len(l)))
        mix += l
    if args.build_on:
        try:
            li(f"Loading base track: {args.build_on}")
            base, _ = load_audio_file(args.build_on, sr=TARGET_SR)
            LL = max(len(mix), len(base))
            if len(mix)  < LL: mix  = np.pad(mix,  (0, LL-len(mix)))
            if len(base) < LL: base = np.pad(base, (0, LL-len(base)))
            if args.sidechain > 0.0:
                li(f"Applying lookahead sidechain amount={args.sidechain:.2f} lookahead={args.sidechain_lookahead_ms}ms")
                mix = lookahead_sidechain(mix, base, TARGET_SR, amount=float(max(0.0, min(args.sidechain, 1.0))), look_ms=float(max(0.0, args.sidechain_lookahead_ms)))
            mix = (0.5*base + 0.5*mix).astype(np.float32)
        except Exception as e:
            lw(f"Base layering failed: {e}")
    loop_src = mix.copy()
    mix = stack_fx(loop_src.copy(), args, rng)
    min_len = int(60.0 * TARGET_SR)
    if len(mix) < min_len:
        li(f"Mix length {len(mix)/TARGET_SR:.1f}s < 60s; looping with per-stack FX")
        loops = [mix]
        total = len(mix)
        while total < min_len:
            loop = stack_fx(loop_src.copy(), args, rng)
            loops.append(loop)
            total += len(loop)
        mix = np.concatenate(loops)
    if args.reverb_mix > 0.0 and rng.random() < FX_CHANCE:
        li(f"Applying reverb (mix={args.reverb_mix:.2f}, room={args.reverb_room:.2f})")
        mix = reverb_schroeder(mix, TARGET_SR, room_size=args.reverb_room, mix=args.reverb_mix)
    if args.tremolo_rate > 0.0 and rng.random() < FX_CHANCE:
        li(f"Applying tremolo (rate={args.tremolo_rate}Hz, depth={args.tremolo_depth})")
        mix = tremolo(mix, TARGET_SR, rate_hz=args.tremolo_rate, depth=args.tremolo_depth)
    if args.phaser_rate > 0.0 and rng.random() < FX_CHANCE:
        li(f"Applying phaser (rate={args.phaser_rate}Hz, depth={args.phaser_depth})")
        mix = phaser(mix, TARGET_SR, rate_hz=args.phaser_rate, depth=args.phaser_depth)
    if args.echo_ms > 0.0 and rng.random() < FX_CHANCE:
        li(f"Applying echo (delay={args.echo_ms}ms, fb={args.echo_fb}, mix={args.echo_mix})")
        wet = echo(mix.copy(), TARGET_SR, delay_ms=args.echo_ms, feedback=args.echo_fb, mix=args.echo_mix)
        mix = (0.75*mix + 0.25*wet).astype(np.float32)
    mix = normalize_peak(mix, peak_db=-0.8)
    mix = safe_audio(mix)
    out_wav = os.path.join(out_dir, base + ".wav")
    sf.write(out_wav, mix, TARGET_SR, subtype="PCM_16")
    # Archive stems
    zip_path = os.path.join(out_dir, base + "_stems.zip")
    with zipfile.ZipFile(zip_path, "w") as zf:
        for root, _, files in os.walk(stems_root):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, stems_root)
                zf.write(full, arc)
    shutil.rmtree(stems_root, ignore_errors=True)
    metadata = {
        "seed": seed,
        "bpm": float(args.bpm),
        "sig_map": sig_str,
        "preset": args.preset,
        "sources": [
            {
                "url": s.url,
                "title": s.title,
                "license": s.licenseurl,
                "bus": s.bus,
            }
            for s in sources
        ],
        "stems_zip": os.path.basename(zip_path),
        "mix_wav": os.path.basename(out_wav),
    }
    out_json = os.path.join(out_dir, base + ".json")
    with open(out_json, "w", encoding="utf-8") as jf:
        json.dump(metadata, jf, ensure_ascii=False, indent=2)
    li(f"Wrote: {out_wav}")
    li(f"Metadata: {out_json}")
    li("Done.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        lw("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        le(f"Unhandled error: {e}")
        sys.exit(1)
