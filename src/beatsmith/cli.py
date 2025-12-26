import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
import uuid
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf

from . import ld, le, li, lw, log
from .audio import (
    TARGET_SR,
    seeded_rng,
    load_audio_from_bytes,
    downmix_to_mono,
    trim_silence,
    apply_fades,
    normalize_audio,
    duration_bucket,
    infer_bpm,
    safe_audio,
    pick_onset_aligned_window,
)
from .fx import (
    compressor,
    eq_three_band,
    reverb_schroeder,
    tremolo,
    phaser,
    echo,
)
from .providers.internet_archive import InternetArchiveProvider
from .providers.local import LocalProvider

USED_SOURCES_REGISTRY = os.path.expanduser("~/.beatsmith/used_sources.json")

FX_TYPES = {
    "reverb": "rvb",
    "compression": "comp",
    "eq": "eq",
    "modulation": "mod",
    "echo": "echo",
    "phaser": "phaser",
    "tremolo": "trem",
}


@dataclass
class SourceAudio:
    identifier: Optional[str]
    filename: Optional[str]
    title: Optional[str]
    url: Optional[str]
    licenseurl: Optional[str]
    audio: np.ndarray  # shape: (channels, samples)
    sr: int


def load_used_sources() -> set[str]:
    try:
        with open(USED_SOURCES_REGISTRY, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return set(str(x) for x in data)
    except Exception:
        pass
    return set()


def save_used_sources(entries: set[str]) -> None:
    os.makedirs(os.path.dirname(USED_SOURCES_REGISTRY), exist_ok=True)
    with open(USED_SOURCES_REGISTRY, "w", encoding="utf-8") as fh:
        json.dump(sorted(entries), fh)


def parse_range(value: str, kind: str = "float") -> Tuple[float, float]:
    if "-" not in value:
        raise argparse.ArgumentTypeError(f"Invalid range '{value}'. Expected 'a-b'.")
    a, b = value.split("-", 1)
    if kind == "int":
        return float(int(a)), float(int(b))
    return float(a), float(b)


def parse_csv(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def short_hash(value: str, length: int = 8) -> str:
    h = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return h[:length]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SampleSmith v4: harvest IA audio into MPC-ready sample packs."
    )
    p.add_argument("out_dir", nargs="?", default=".", help="Output directory for the sample pack.")
    p.add_argument("--seed", type=str, default=None, help="Deterministic seed.")
    p.add_argument("--salt", type=str, default=None, help="Additional salt for alternate takes.")
    p.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.beatsmith/cache"))
    p.add_argument("--license-allow", type=str, default="cc0,creativecommons,public domain,publicdomain,cc-by,cc-by-sa")
    p.add_argument("--strict-license", action="store_true")
    p.add_argument("--query-bias", type=str, default=None)
    p.add_argument("--reuse-sources", action="store_true")
    p.add_argument("--c", type=int, default=100, help="Total IA source files to download/process.")
    p.add_argument("--max-samples", type=int, default=None, help="Cap total exported clean samples.")
    p.add_argument("--min-rms", type=float, default=0.02)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--verbose", action="store_true")

    # Robustness controls
    p.add_argument(
        "--source-timeout-s",
        type=float,
        default=60.0,
        help="Max seconds spent trying to acquire sources before fallback (or abort).",
    )
    p.add_argument(
        "--no-progress-timeout-s",
        type=float,
        default=60.0,
        help="If no samples are exported for this many seconds, trigger fallback mode.",
    )
    p.add_argument(
        "--emergency-synth",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true, synthesize fallback sources so a run always produces output.",
    )

    p.add_argument("--form-modes", type=str, default="oneshot,loop,longform")
    p.add_argument("--form-variation-range", type=str, default="1-3")
    p.add_argument("--oneshot-seconds", type=str, default="0.08-1.20")
    p.add_argument("--loop-seconds", type=str, default="1.0-16.0")
    p.add_argument("--longform-seconds", type=str, default="30-90")

    p.add_argument("--trim-silence-db", type=float, default=-45.0)
    p.add_argument("--trim-pad-ms", type=int, default=12)
    p.add_argument("--fade-in-ms", type=int, default=2)
    p.add_argument("--fade-out-ms", type=int, default=8)
    p.add_argument("--normalize-mode", choices=["peak", "rms"], default="peak")
    p.add_argument("--normalize-peak-db", type=float, default=-1.0)
    p.add_argument("--normalize-rms", type=float, default=None)

    p.add_argument("--label-bpm-assumption", type=float, default=120.0)
    p.add_argument("--label-tolerance", type=float, default=0.12)
    p.add_argument("--emit-bpm-infer", action="store_true")
    p.add_argument("--bpm-confidence-min", type=float, default=0.55)

    p.add_argument("--c-effects", type=int, default=0)
    p.add_argument("--c-effect-randomize-count", type=str, default="0-5")
    p.add_argument("--c-effect-variations", type=str, default="0")
    p.add_argument("--fx-prob", type=float, default=1.0)
    p.add_argument("--fx-chain-style", choices=["tasteful", "aggressive", "random"], default="tasteful")
    p.add_argument("--fx-room", choices=["small", "mid", "large"], default=None)
    p.add_argument("--fx-tag-filenames", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--export-mono", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--export-stereo", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--stereo-prefer", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--pack-name", type=str, default=None)
    p.add_argument("--write-manifest", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--write-credits", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--zip-pack", action="store_true")

    p.add_argument("--provider", choices=["ia", "local"], default="ia")
    return p


def _fx_params(style: str, rng, room: Optional[str]) -> Dict[str, Dict[str, float]]:
    if style == "aggressive":
        comp = {"thresh_db": -24.0, "ratio": 6.0, "makeup_db": 4.0}
        eqp = {"low_db": 4.0, "mid_db": -2.0, "high_db": 3.0}
        trem = {"rate_hz": rng.uniform(4.0, 8.0), "depth": 0.6}
    elif style == "random":
        comp = {"thresh_db": rng.uniform(-28, -12), "ratio": rng.uniform(2.0, 6.0), "makeup_db": rng.uniform(1.0, 6.0)}
        eqp = {"low_db": rng.uniform(-4, 4), "mid_db": rng.uniform(-3, 3), "high_db": rng.uniform(-4, 4)}
        trem = {"rate_hz": rng.uniform(2.0, 9.0), "depth": rng.uniform(0.2, 0.7)}
    else:
        comp = {"thresh_db": -18.0, "ratio": 3.5, "makeup_db": 2.0}
        eqp = {"low_db": 2.0, "mid_db": -1.0, "high_db": 1.5}
        trem = {"rate_hz": rng.uniform(2.0, 5.0), "depth": 0.4}

    room_size = {"small": 0.2, "mid": 0.4, "large": 0.6}.get(room, 0.35)
    rvb = {"room_size": room_size, "mix": 0.18 if style != "aggressive" else 0.3}
    echo_params = {"delay_ms": rng.uniform(180, 420), "feedback": 0.3, "mix": 0.2}
    ph = {"rate_hz": rng.uniform(0.2, 1.0), "depth": 0.5}
    return {
        "compression": comp,
        "eq": eqp,
        "tremolo": trem,
        "reverb": rvb,
        "echo": echo_params,
        "phaser": ph,
    }


def apply_fx_chain(y: np.ndarray, sr: int, chain: Sequence[str], params: Dict[str, Dict[str, float]]) -> np.ndarray:
    def apply_chain(mono: np.ndarray) -> np.ndarray:
        out = mono.copy()
        for fx in chain:
            if fx == "compression":
                p = params["compression"]
                out = compressor(out, sr, thresh_db=p["thresh_db"], ratio=p["ratio"], makeup_db=p["makeup_db"])
            elif fx == "eq":
                p = params["eq"]
                out = eq_three_band(out, sr, low_db=p["low_db"], mid_db=p["mid_db"], high_db=p["high_db"])
            elif fx == "reverb":
                p = params["reverb"]
                out = reverb_schroeder(out, sr, room_size=p["room_size"], mix=p["mix"])
            elif fx == "tremolo":
                p = params["tremolo"]
                out = tremolo(out, sr, rate_hz=p["rate_hz"], depth=p["depth"])
            elif fx == "phaser":
                p = params["phaser"]
                out = phaser(out, sr, rate_hz=p["rate_hz"], depth=p["depth"])
            elif fx == "echo":
                p = params["echo"]
                wet = echo(out, sr, delay_ms=p["delay_ms"], feedback=p["feedback"], mix=p["mix"])
                out = (0.75 * out + 0.25 * wet).astype(np.float32)
        return out

    if y.ndim == 1:
        return apply_chain(y)
    return np.vstack([apply_chain(ch) for ch in y])


def write_audio(path: str, y: np.ndarray, sr: int) -> None:
    if y.ndim == 1:
        data = y
    else:
        data = y.T
    sf.write(path, data, sr, subtype="PCM_16")


def compute_metrics(y: np.ndarray) -> Tuple[float, float]:
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    peak = float(np.max(np.abs(y)) + 1e-12)
    return rms, peak


def pick_window(y: np.ndarray, sr: int, dur_s: float, rng) -> Tuple[int, int]:
    n = y.shape[-1]
    win = max(int(dur_s * sr), 1)
    if n <= win:
        return 0, n
    s0 = rng.randrange(0, n - win + 1)
    return s0, s0 + win


def _synthesize_emergency_sources(rng) -> List[SourceAudio]:
    li("Entering emergency mode: synthesizing fallback sources so output is guaranteed.")

    dur_s = 10.0
    n = int(TARGET_SR * dur_s)
    t = np.arange(n, dtype=np.float32) / float(TARGET_SR)

    # Click track bursts
    click = np.zeros(n, dtype=np.float32)
    period = max(int(TARGET_SR * 0.25), 1)
    for i in range(0, n, period):
        w = min(int(TARGET_SR * 0.01), n - i)
        if w > 1:
            burst = (np.sin(2.0 * np.pi * 1200.0 * t[:w]) * np.hanning(w)).astype(np.float32)
            click[i:i + w] += 0.6 * burst

    # Gentle noise bed
    noise = (np.random.default_rng(rng.randint(0, 2**31 - 1)).standard_normal(n).astype(np.float32) * 0.05)

    # Simple tonal drone (texture)
    drone = (0.12 * np.sin(2.0 * np.pi * 110.0 * t) + 0.08 * np.sin(2.0 * np.pi * 220.0 * t)).astype(np.float32)

    def stereoize(x: np.ndarray) -> np.ndarray:
        # Slight channel variance so stereo isn't identical.
        shift = rng.randint(0, 128)
        left = x
        right = np.roll(x, shift).astype(np.float32)
        return np.vstack([left, right]).astype(np.float32)

    return [
        SourceAudio(
            identifier="synth://clicks",
            filename=None,
            title="synthetic clicks",
            url="synth://clicks",
            licenseurl="publicdomain",
            audio=stereoize(safe_audio(click)),
            sr=TARGET_SR,
        ),
        SourceAudio(
            identifier="synth://noise",
            filename=None,
            title="synthetic noise",
            url="synth://noise",
            licenseurl="publicdomain",
            audio=stereoize(safe_audio(noise)),
            sr=TARGET_SR,
        ),
        SourceAudio(
            identifier="synth://drone",
            filename=None,
            title="synthetic drone",
            url="synth://drone",
            licenseurl="publicdomain",
            audio=stereoize(safe_audio(drone)),
            sr=TARGET_SR,
        ),
    ]


def select_sources(
    provider,
    rng,
    count: int,
    query_bias: Optional[str],
    allow_tokens: List[str],
    strict: bool,
    cache_dir: str,
    reuse_sources: bool,
    source_timeout_s: float,
    emergency_synth: bool,
) -> List[SourceAudio]:
    start = time.monotonic()
    used = load_used_sources()
    selected: List[SourceAudio] = []

    strict_current = bool(strict)
    reuse_current = bool(reuse_sources)

    stats = {
        "search_calls": 0,
        "candidates_seen": 0,
        "fetch_empty": 0,
        "fetch_small": 0,
        "decode_fail": 0,
        "used_skip": 0,
        "too_short": 0,
        "selected": 0,
    }

    li(f"Selecting up to {count} sources (timeout={source_timeout_s:.0f}s, strict={strict_current}, reuse={reuse_current})")

    while len(selected) < count and (time.monotonic() - start) < max(source_timeout_s, 1.0):
        elapsed = time.monotonic() - start
        remaining = source_timeout_s - elapsed
        if remaining <= 0:
            break

        stats["search_calls"] += 1
        wanted_now = max(10, (count - len(selected)) * 3)

        files = provider.search(rng, wanted_now, query_bias, allow_tokens, strict_current)
        if not files and strict_current:
            lw("No search results under strict license. Relaxing strict filter and retrying.")
            strict_current = False
            continue
        if not files:
            lw("Provider returned zero candidates. Retrying shortly.")
            time.sleep(min(1.0, max(0.1, remaining)))
            continue

        for f in files:
            if len(selected) >= count:
                break
            if (time.monotonic() - start) >= source_timeout_s:
                break

            stats["candidates_seen"] += 1
            token = f.get("url") or f.get("identifier")
            if token and (not reuse_current) and token in used:
                stats["used_skip"] += 1
                if stats["used_skip"] % 25 == 0:
                    ld(f"Skipping previously used sources: {stats['used_skip']} so far")
                continue

            url = f.get("url")
            b = provider.fetch(f, cache_dir)

            if not url or not b:
                stats["fetch_empty"] += 1
                continue
            if len(b) < 2048:
                stats["fetch_small"] += 1
                continue

            try:
                fname = f.get("name")
                if not fname and url:
                    fname = os.path.basename(url)
                y, sr = load_audio_from_bytes(b, sr=TARGET_SR, filename=fname, mono=False)
                if y.ndim == 1:
                    y = y[np.newaxis, :]

                if y.shape[-1] < int(0.25 * sr):
                    stats["too_short"] += 1
                    continue

                src = SourceAudio(
                    identifier=f.get("identifier"),
                    filename=f.get("name"),
                    title=f.get("title"),
                    url=url,
                    licenseurl=provider.license(f),
                    audio=y.astype(np.float32),
                    sr=sr,
                )
                selected.append(src)
                stats["selected"] = len(selected)
                if token:
                    used.add(token)

                li(f"Loaded source {len(selected)}/{count}: {src.title or src.filename or 'unknown'}")
                if len(selected) % 5 == 0:
                    li(f"Source progress: {len(selected)}/{count} selected (elapsed {time.monotonic() - start:.1f}s)")

            except Exception as e:
                stats["decode_fail"] += 1
                lw(f"Decode failed: {e}")

            # polite pacing (also helps avoid IA throttling cascades)
            time.sleep(rng.uniform(0.05, 0.25))

        # If we're getting nowhere, relax reuse after half the timeout
        elapsed = time.monotonic() - start
        if not selected and (not reuse_current) and elapsed >= max(5.0, source_timeout_s * 0.5):
            lw("No sources selected yet; temporarily enabling source reuse to improve odds.")
            reuse_current = True

    save_used_sources(used)

    li(
        "Source selection summary: "
        f"selected={stats['selected']} "
        f"search_calls={stats['search_calls']} "
        f"candidates={stats['candidates_seen']} "
        f"used_skip={stats['used_skip']} "
        f"fetch_empty={stats['fetch_empty']} "
        f"fetch_small={stats['fetch_small']} "
        f"too_short={stats['too_short']} "
        f"decode_fail={stats['decode_fail']} "
        f"elapsed={time.monotonic() - start:.1f}s"
    )

    if selected:
        return selected

    if emergency_synth:
        return _synthesize_emergency_sources(rng)

    return []


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        for h in logging.getLogger().handlers:
            try:
                h.setLevel(logging.DEBUG)
            except Exception:
                pass
        log.setLevel(logging.DEBUG)
        li("Verbose logging enabled.")

    seed = args.seed or f"auto-{time.time_ns()}"
    rng = seeded_rng(seed, args.salt)

    form_modes = parse_csv(args.form_modes)
    if not form_modes:
        form_modes = ["oneshot", "loop", "longform"]

    form_var_min, form_var_max = parse_range(args.form_variation_range, kind="int")
    oneshot_min, oneshot_max = parse_range(args.oneshot_seconds)
    loop_min, loop_max = parse_range(args.loop_seconds)
    long_min, long_max = parse_range(args.longform_seconds)

    if args.max_samples is None:
        args.max_samples = int(args.c * max(1.0, form_var_max))

    allow_tokens = [t.strip() for t in (args.license_allow or "").split(",") if t.strip()]
    strict = bool(args.strict_license)

    provider = LocalProvider() if args.provider == "local" else InternetArchiveProvider()

    if args.dry_run:
        li("Planning sources (dry-run)...")
        sources = provider.search(rng, args.c, args.query_bias, allow_tokens, strict)
        if not sources:
            lw("Dry-run: provider returned zero candidates. Consider removing --strict-license or adjusting --query-bias.")
        for idx, f in enumerate(sources[: args.c], 1):
            title = f.get("title") or f.get("name") or "unknown"
            li(f"Source {idx}: {title} ({f.get('url')})")
        li("Dry run complete. No audio downloaded or files written.")
        return

    out_dir = os.path.abspath(args.out_dir)
    ensure_dir(out_dir)
    pack_name = args.pack_name or f"SamplePack_{time.strftime('%Y%m%d_%H%M%S')}_{short_hash(seed)}"

    clean_root = os.path.join(out_dir, "samples", "clean")
    fx_root = os.path.join(out_dir, "samples", "fx")
    meta_root = os.path.join(out_dir, "metadata")
    ensure_dir(clean_root)
    ensure_dir(fx_root)
    ensure_dir(meta_root)

    for form in ("oneshot", "loop", "longform"):
        ensure_dir(os.path.join(clean_root, form, "stereo"))
        ensure_dir(os.path.join(clean_root, form, "mono"))

    li(f"Harvesting {args.c} sources into {pack_name}")

    sources = select_sources(
        provider=provider,
        rng=rng,
        count=args.c,
        query_bias=args.query_bias,
        allow_tokens=allow_tokens,
        strict=strict,
        cache_dir=args.cache_dir,
        reuse_sources=args.reuse_sources,
        source_timeout_s=float(args.source_timeout_s),
        emergency_synth=bool(args.emergency_synth),
    )
    if not sources:
        le("No sources available and emergency synthesis disabled; aborting.")
        sys.exit(2)

    manifest: Dict[str, Any] = {
        "run_id": uuid.uuid4().hex,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "salt": args.salt,
        "pack_name": pack_name,
        "options": vars(args),
        "sources": [],
        "samples": [],
        "variants": [],
    }

    credits: List[Dict[str, str]] = []

    for idx, src in enumerate(sources, 1):
        manifest["sources"].append(
            {
                "id": idx,
                "identifier": src.identifier,
                "file": src.filename,
                "title": src.title,
                "url": src.url,
                "license": src.licenseurl,
                "retrieved_at": time.strftime("%Y-%m-%d"),
            }
        )
        credits.append(
            {
                "identifier": src.identifier or "",
                "file": src.filename or "",
                "title": src.title or "",
                "url": src.url or "",
                "license": src.licenseurl or "",
                "retrieved_at": time.strftime("%Y-%m-%d"),
            }
        )

    seed_tag = f"s{short_hash(seed + (args.salt or ''))}"
    clean_samples: List[Dict[str, Any]] = []

    skip_stats = {
        "trim_empty": 0,
        "segment_empty": 0,
        "low_rms": 0,
        "exported": 0,
        "attempts": 0,
        "trim_recovered": 0,
    }

    last_export_t = time.monotonic()
    gen_start_t = time.monotonic()

    def _should_trigger_no_progress() -> bool:
        if args.no_progress_timeout_s is None:
            return False
        return (time.monotonic() - last_export_t) >= max(float(args.no_progress_timeout_s), 1.0)

    def _log_gen_progress(force: bool = False) -> None:
        if not force and (skip_stats["attempts"] % 50 != 0):
            return
        li(
            "Generation progress: "
            f"exported={skip_stats['exported']} "
            f"attempts={skip_stats['attempts']} "
            f"trim_empty={skip_stats['trim_empty']} "
            f"trim_recovered={skip_stats['trim_recovered']} "
            f"low_rms={skip_stats['low_rms']} "
            f"elapsed={time.monotonic() - gen_start_t:.1f}s"
        )

    def _attempt_generate(relaxed: bool = False) -> None:
        nonlocal last_export_t

        # Relaxed mode is intended to break “zero output” scenarios.
        min_rms = float(args.min_rms)
        trim_db = abs(float(args.trim_silence_db))
        normalize_peak_db = float(args.normalize_peak_db)

        if relaxed:
            min_rms = min(min_rms, 0.005)
            trim_db = min(trim_db, 25.0)
            normalize_peak_db = min(normalize_peak_db, -0.8)

            li(f"Fallback mode enabled: min_rms={min_rms}, trim_db={trim_db}")

        for src_idx, src in enumerate(sources, 1):
            if len(clean_samples) >= int(args.max_samples):
                break

            variations = rng.randint(int(form_var_min), int(form_var_max))
            for _ in range(variations):
                if len(clean_samples) >= int(args.max_samples):
                    break

                if _should_trigger_no_progress():
                    lw(
                        f"No samples exported for {float(args.no_progress_timeout_s):.0f}s; triggering fallback behavior."
                    )
                    return

                skip_stats["attempts"] += 1
                form = rng.choice(form_modes)

                if form == "oneshot":
                    dur_s = rng.uniform(oneshot_min, oneshot_max)
                    s0, s1, _energy = pick_onset_aligned_window(
                        downmix_to_mono(src.audio),
                        src.sr,
                        dur_s,
                        rng=rng,
                        min_rms=min_rms,
                        refine_edges=True,
                    )
                elif form == "loop":
                    dur_s = rng.uniform(loop_min, loop_max)
                    s0, s1 = pick_window(src.audio, src.sr, dur_s, rng)
                else:
                    dur_s = rng.uniform(long_min, long_max)
                    s0, s1 = pick_window(src.audio, src.sr, dur_s, rng)

                seg = src.audio[:, s0:s1]
                if seg.size == 0 or seg.shape[-1] == 0:
                    skip_stats["segment_empty"] += 1
                    ld("Skipping empty segment window.")
                    _log_gen_progress()
                    continue

                # Trim; if trim nukes it, retry without trim.
                seg_t, trim_start, trim_end = trim_silence(seg, src.sr, trim_db, int(args.trim_pad_ms))
                if seg_t.size == 0 or seg_t.shape[-1] == 0:
                    skip_stats["trim_empty"] += 1
                    ld("trim_silence produced empty output; retrying without trim.")
                    seg_t = seg
                    trim_start, trim_end = 0, seg.shape[-1]
                    if seg_t.size == 0 or seg_t.shape[-1] == 0:
                        skip_stats["segment_empty"] += 1
                        _log_gen_progress()
                        continue
                    skip_stats["trim_recovered"] += 1
                seg = seg_t

                seg = apply_fades(seg, src.sr, int(args.fade_in_ms), int(args.fade_out_ms))
                seg = normalize_audio(seg, args.normalize_mode, normalize_peak_db, args.normalize_rms)
                seg = safe_audio(seg)

                mono = downmix_to_mono(seg)
                rms, peak = compute_metrics(mono)

                # In relaxed mode, we still keep a floor, but lower it.
                rms_floor = min_rms if not relaxed else min(min_rms, 0.002)
                if rms < rms_floor:
                    skip_stats["low_rms"] += 1
                    _log_gen_progress()
                    continue

                duration_s = float(seg.shape[-1] / src.sr)
                bucket = duration_bucket(
                    duration_s,
                    float(args.label_bpm_assumption),
                    float(args.label_tolerance),
                    longform=(form == "longform"),
                )

                bpm_tag = f"bpm{int(args.label_bpm_assumption)}a"
                bpm_val = None
                bpm_conf = 0.0
                if args.emit_bpm_infer and form in ("loop", "longform"):
                    bpm_val, bpm_conf = infer_bpm(mono, src.sr)
                    if bpm_val and bpm_conf >= float(args.bpm_confidence_min):
                        bpm_tag = f"bpm{int(round(bpm_val))}i"
                    else:
                        bpm_val = None

                src_tag = short_hash(f"{src.identifier}:{src.filename}")
                dur_ms = int(round(duration_s * 1000))
                form_prefix = {"oneshot": "os", "loop": "lp", "longform": "lf"}[form]
                uid = uuid.uuid4().hex[:8]
                filename = f"{form_prefix}_{src_tag}_{dur_ms}ms_{bucket}_{bpm_tag}_{seed_tag}_{uid}.wav"

                exports: Dict[str, str] = {}
                if args.export_stereo:
                    stereo = seg
                    if stereo.shape[0] == 1 and args.stereo_prefer:
                        stereo = np.repeat(stereo, 2, axis=0)
                    stereo_path = os.path.join(clean_root, form, "stereo", filename)
                    write_audio(stereo_path, stereo, src.sr)
                    exports["stereo"] = os.path.relpath(stereo_path, out_dir)
                if args.export_mono:
                    mono_path = os.path.join(clean_root, form, "mono", filename)
                    write_audio(mono_path, mono, src.sr)
                    exports["mono"] = os.path.relpath(mono_path, out_dir)

                sample_id = uuid.uuid4().hex
                sample_entry = {
                    "sample_id": sample_id,
                    "source_id": src_idx,
                    "form": form,
                    "start_s": float(s0 / src.sr),
                    "end_s": float(s1 / src.sr),
                    "trim_start_s": float(trim_start / src.sr),
                    "trim_end_s": float(trim_end / src.sr),
                    "duration_s": duration_s,
                    "rms": rms,
                    "peak": peak,
                    "label_group": bucket,
                    "bpm_tag": bpm_tag,
                    "bpm_inferred": bpm_val,
                    "bpm_confidence": bpm_conf,
                    "exports": exports,
                }

                manifest["samples"].append(sample_entry)
                clean_samples.append({"entry": sample_entry, "audio": seg})

                skip_stats["exported"] += 1
                last_export_t = time.monotonic()
                if skip_stats["exported"] % 10 == 0:
                    li(f"Exported {skip_stats['exported']} samples so far (elapsed {time.monotonic() - gen_start_t:.1f}s)")
                _log_gen_progress()

                if len(clean_samples) >= int(args.max_samples):
                    break

    _attempt_generate(relaxed=False)

    # If we didn't generate anything (or got stuck), attempt fallback.
    if not clean_samples or _should_trigger_no_progress():
        lw(
            "No (or insufficient) samples generated under current thresholds. "
            "Attempting fallback mode to guarantee output."
        )
        # If we still have zero samples and emergency_synth is enabled, ensure synth sources exist.
        if not clean_samples and args.emergency_synth:
            # If sources already include synth, fine; otherwise extend.
            if not any((s.url or "").startswith("synth://") for s in sources):
                sources.extend(_synthesize_emergency_sources(rng))

        _attempt_generate(relaxed=True)

    if not clean_samples:
        le(
            "No samples generated even after fallback. "
            "This typically indicates an environment/decoder issue or output permissions problem."
        )
        le(
            f"Final stats: attempts={skip_stats['attempts']} trim_empty={skip_stats['trim_empty']} "
            f"trim_recovered={skip_stats['trim_recovered']} low_rms={skip_stats['low_rms']} "
            f"segment_empty={skip_stats['segment_empty']}"
        )
        sys.exit(2)

    # FX generation section (unchanged from your baseline, but keeps safe_audio)
    fx_pool = clean_samples
    if args.c_effects > 0:
        fx_pool = rng.sample(clean_samples, min(args.c_effects, len(clean_samples)))

    fx_variations = parse_csv(args.c_effect_variations)
    fx_types = list(FX_TYPES.keys()) if args.c_effect_variations == "0" else fx_variations
    fx_range_min, fx_range_max = parse_range(args.c_effect_randomize_count, kind="int")

    for sample in fx_pool:
        if rng.random() > float(args.fx_prob):
            continue
        count = rng.randint(int(fx_range_min), int(fx_range_max))
        if count <= 0:
            continue
        for _ in range(count):
            chain_len = rng.randint(1, min(3, len(fx_types)))
            chain = rng.sample(fx_types, chain_len)
            params = _fx_params(args.fx_chain_style, rng, args.fx_room)
            seg = sample["audio"]
            fx_out = apply_fx_chain(seg, TARGET_SR, chain, params)
            fx_out = safe_audio(fx_out)

            # Optional FX filename tagging
            base_entry = sample["entry"]
            form = base_entry["form"]
            base_exports = base_entry["exports"]
            base_any = base_exports.get("stereo") or base_exports.get("mono") or "unknown.wav"
            base_name = os.path.splitext(os.path.basename(base_any))[0]
            fx_tag = "_".join(FX_TYPES.get(x, x) for x in chain) if args.fx_tag_filenames else "fx"
            fx_name = f"{base_name}_{fx_tag}_{uuid.uuid4().hex[:6]}.wav"

            exports: Dict[str, str] = {}
            if args.export_stereo:
                stereo = fx_out
                if stereo.ndim == 1:
                    stereo = np.vstack([stereo, stereo])
                fx_path = os.path.join(fx_root, form, "stereo")
                ensure_dir(fx_path)
                out_path = os.path.join(fx_path, fx_name)
                write_audio(out_path, stereo, TARGET_SR)
                exports["stereo"] = os.path.relpath(out_path, out_dir)
            if args.export_mono:
                mono = downmix_to_mono(fx_out)
                fx_path = os.path.join(fx_root, form, "mono")
                ensure_dir(fx_path)
                out_path = os.path.join(fx_path, fx_name)
                write_audio(out_path, mono, TARGET_SR)
                exports["mono"] = os.path.relpath(out_path, out_dir)

            manifest["variants"].append(
                {
                    "variant_id": uuid.uuid4().hex,
                    "source_sample_id": base_entry["sample_id"],
                    "chain": chain,
                    "params": params,
                    "exports": exports,
                }
            )

    # Write manifest + credits
    if args.write_manifest:
        manifest_path = os.path.join(meta_root, "manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        li(f"Wrote manifest: {manifest_path}")

    if args.write_credits:
        credits_path = os.path.join(meta_root, "credits.csv")
        with open(credits_path, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["identifier", "file", "title", "url", "license", "retrieved_at"])
            w.writeheader()
            for row in credits:
                w.writerow(row)
        li(f"Wrote credits: {credits_path}")

    # Optional zip
    if args.zip_pack:
        zip_name = f"{pack_name}.zip"
        zip_path = os.path.join(out_dir, zip_name)
        li(f"Zipping pack to: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for root, _dirs, files in os.walk(out_dir):
                for fn in files:
                    full = os.path.join(root, fn)
                    if full == zip_path:
                        continue
                    rel = os.path.relpath(full, out_dir)
                    z.write(full, rel)

    li(
        "Run complete. "
        f"exported_clean={len(clean_samples)} "
        f"variants={len(manifest.get('variants', []))} "
        f"elapsed={time.monotonic() - gen_start_t:.1f}s"
    )

