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

from . import li, lw, le, log
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
    audio: np.ndarray
    sr: int


def load_used_sources() -> set[str]:
    try:
        with open(USED_SOURCES_REGISTRY, "r") as fh:
            data = json.load(fh)
            if isinstance(data, list):
                return set(str(x) for x in data)
    except Exception:
        pass
    return set()


def save_used_sources(entries: set[str]) -> None:
    os.makedirs(os.path.dirname(USED_SOURCES_REGISTRY), exist_ok=True)
    with open(USED_SOURCES_REGISTRY, "w") as fh:
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
    p.add_argument("--fx-tag-filenames", action="store_true", default=True)

    p.add_argument("--export-mono", action="store_true", default=True)
    p.add_argument("--export-stereo", action="store_true", default=True)
    p.add_argument("--stereo-prefer", action="store_true", default=True)
    p.add_argument("--pack-name", type=str, default=None)
    p.add_argument("--write-manifest", action="store_true", default=True)
    p.add_argument("--write-credits", action="store_true", default=True)
    p.add_argument("--zip-pack", action="store_true")

    p.add_argument("--provider", choices=["ia", "local"], default="ia")
    return p


def _fx_params(style: str, rng, room: Optional[str]) -> Dict[str, Dict[str, float]]:
    if style == "aggressive":
        comp = {"thresh_db": -24.0, "ratio": 6.0, "makeup_db": 4.0}
        eq = {"low_db": 4.0, "mid_db": -2.0, "high_db": 3.0}
        trem = {"rate_hz": rng.uniform(4.0, 8.0), "depth": 0.6}
    elif style == "random":
        comp = {"thresh_db": rng.uniform(-28, -12), "ratio": rng.uniform(2.0, 6.0), "makeup_db": rng.uniform(1.0, 6.0)}
        eq = {"low_db": rng.uniform(-4, 4), "mid_db": rng.uniform(-3, 3), "high_db": rng.uniform(-4, 4)}
        trem = {"rate_hz": rng.uniform(2.0, 9.0), "depth": rng.uniform(0.2, 0.7)}
    else:
        comp = {"thresh_db": -18.0, "ratio": 3.5, "makeup_db": 2.0}
        eq = {"low_db": 2.0, "mid_db": -1.0, "high_db": 1.5}
        trem = {"rate_hz": rng.uniform(2.0, 5.0), "depth": 0.4}

    room_size = {"small": 0.2, "mid": 0.4, "large": 0.6}.get(room, 0.35)
    rvb = {"room_size": room_size, "mix": 0.18 if style != "aggressive" else 0.3}
    echo_params = {"delay_ms": rng.uniform(180, 420), "feedback": 0.3, "mix": 0.2}
    ph = {"rate_hz": rng.uniform(0.2, 1.0), "depth": 0.5}
    return {
        "compression": comp,
        "eq": eq,
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
    s0 = rng.randrange(0, n - win)
    return s0, s0 + win


def select_sources(
    provider,
    rng,
    count: int,
    query_bias: Optional[str],
    allow_tokens: List[str],
    strict: bool,
    cache_dir: str,
    reuse_sources: bool,
) -> List[SourceAudio]:
    files = provider.search(rng, count, query_bias, allow_tokens, strict)
    if not files:
        lw("No search results (strict filter?). Will try non-strict as fallback.")
        files = provider.search(rng, count, query_bias, allow_tokens, False)
    used = load_used_sources()
    selected: List[SourceAudio] = []
    tried = 0
    for f in files:
        if len(selected) >= count:
            break
        token = f.get("url") or f.get("identifier")
        if token and not reuse_sources and token in used:
            continue
        url = f.get("url")
        b = provider.fetch(f, cache_dir)
        if not url or not b or len(b) < 2048:
            continue
        try:
            fname = f.get("name")
            if not fname and url:
                fname = os.path.basename(url)
            y, sr = load_audio_from_bytes(b, sr=TARGET_SR, filename=fname, mono=False)
            if y.ndim == 1:
                y = y[np.newaxis, :]
            selected.append(
                SourceAudio(
                    identifier=f.get("identifier"),
                    filename=f.get("name"),
                    title=f.get("title"),
                    url=url,
                    licenseurl=provider.license(f),
                    audio=y,
                    sr=sr,
                )
            )
            if token:
                used.add(token)
        except Exception as e:
            lw(f"Decode failed: {e}")
        tried += 1
        if tried > count * 18 and selected:
            break
    save_used_sources(used)
    return selected


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    if args.verbose:
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
        li("Planning sources...")
        sources = provider.search(rng, args.c, args.query_bias, allow_tokens, strict)
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
        provider,
        rng,
        args.c,
        args.query_bias,
        allow_tokens,
        strict,
        args.cache_dir,
        args.reuse_sources,
    )
    if not sources:
        le("No sources available, aborting.")
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

    for src_idx, src in enumerate(sources, 1):
        variations = rng.randint(int(form_var_min), int(form_var_max))
        for _ in range(variations):
            if len(clean_samples) >= args.max_samples:
                break
            form = rng.choice(form_modes)
            if form == "oneshot":
                dur_s = rng.uniform(oneshot_min, oneshot_max)
                s0, s1, _ = pick_onset_aligned_window(
                    downmix_to_mono(src.audio),
                    src.sr,
                    dur_s,
                    rng=rng,
                    min_rms=args.min_rms,
                    refine_edges=True,
                )
            elif form == "loop":
                dur_s = rng.uniform(loop_min, loop_max)
                s0, s1 = pick_window(src.audio, src.sr, dur_s, rng)
            else:
                dur_s = rng.uniform(long_min, long_max)
                s0, s1 = pick_window(src.audio, src.sr, dur_s, rng)

            seg = src.audio[:, s0:s1]
            seg, trim_start, trim_end = trim_silence(seg, src.sr, abs(args.trim_silence_db), args.trim_pad_ms)
            if seg.size == 0:
                continue
            seg = apply_fades(seg, src.sr, args.fade_in_ms, args.fade_out_ms)
            seg = normalize_audio(seg, args.normalize_mode, args.normalize_peak_db, args.normalize_rms)
            seg = safe_audio(seg)

            mono = downmix_to_mono(seg)
            rms, peak = compute_metrics(mono)
            if rms < args.min_rms:
                continue

            duration_s = float(seg.shape[-1] / src.sr)
            bucket = duration_bucket(duration_s, args.label_bpm_assumption, args.label_tolerance, longform=form == "longform")

            bpm_tag = f"bpm{int(args.label_bpm_assumption)}a"
            bpm_val = None
            bpm_conf = 0.0
            if args.emit_bpm_infer and form in ("loop", "longform"):
                bpm_val, bpm_conf = infer_bpm(mono, src.sr)
                if bpm_val and bpm_conf >= args.bpm_confidence_min:
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
        if len(clean_samples) >= args.max_samples:
            break

    if not clean_samples:
        le("No samples generated. Check thresholds or source availability.")
        sys.exit(2)

    fx_pool = clean_samples
    if args.c_effects > 0:
        fx_pool = rng.sample(clean_samples, min(args.c_effects, len(clean_samples)))

    fx_variations = parse_csv(args.c_effect_variations)
    fx_types = list(FX_TYPES.keys()) if args.c_effect_variations == "0" else fx_variations
    fx_range_min, fx_range_max = parse_range(args.c_effect_randomize_count, kind="int")

    for sample in fx_pool:
        if rng.random() > args.fx_prob:
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
            fx_tag = "_".join(FX_TYPES.get(c, c) for c in chain)

            form = sample["entry"]["form"]
            base_filename = os.path.basename(next(iter(sample["entry"]["exports"].values())))
            fx_filename = base_filename.replace(".wav", f"__fx-{fx_tag}.wav")
            fx_dir = os.path.join(fx_root, fx_tag, form)
            ensure_dir(os.path.join(fx_dir, "stereo"))
            ensure_dir(os.path.join(fx_dir, "mono"))

            exports: Dict[str, str] = {}
            if args.export_stereo:
                stereo = fx_out
                if stereo.ndim == 1:
                    stereo = stereo[np.newaxis, :]
                if stereo.shape[0] == 1 and args.stereo_prefer:
                    stereo = np.repeat(stereo, 2, axis=0)
                stereo_path = os.path.join(fx_dir, "stereo", fx_filename)
                write_audio(stereo_path, stereo, TARGET_SR)
                exports["stereo"] = os.path.relpath(stereo_path, out_dir)
            if args.export_mono:
                mono_path = os.path.join(fx_dir, "mono", fx_filename)
                write_audio(mono_path, downmix_to_mono(fx_out), TARGET_SR)
                exports["mono"] = os.path.relpath(mono_path, out_dir)

            manifest["variants"].append(
                {
                    "sample_id": sample["entry"]["sample_id"],
                    "fx_tag": fx_tag,
                    "chain": chain,
                    "params": params,
                    "exports": exports,
                }
            )

    if args.write_manifest:
        manifest_path = os.path.join(out_dir, "pack.json")
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)

    if args.write_credits:
        credits_path = os.path.join(out_dir, "credits.csv")
        with open(credits_path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=["identifier", "file", "title", "url", "license", "retrieved_at"])
            writer.writeheader()
            for row in credits:
                writer.writerow(row)

    readme_path = os.path.join(out_dir, "README_PACK.txt")
    with open(readme_path, "w", encoding="utf-8") as fh:
        fh.write(
            "SampleSmith pack generated by BeatSmith v4.\n"
            "Import samples/clean and samples/fx into your sampler or MPC.\n"
            "See pack.json for full provenance.\n"
        )

    if args.zip_pack:
        zip_path = os.path.join(out_dir, f"{pack_name}.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for root, _, files in os.walk(out_dir):
                for fn in files:
                    if fn == os.path.basename(zip_path):
                        continue
                    full = os.path.join(root, fn)
                    arc = os.path.relpath(full, out_dir)
                    zf.write(full, arc)

    li(f"Pack written to: {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        lw("Interrupted by user.")
        sys.exit(130)
    except Exception as e:
        le(f"Unhandled error: {e}")
        sys.exit(1)
