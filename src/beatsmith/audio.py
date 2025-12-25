import argparse
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Set
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import librosa

from . import li, lw
from .providers import Provider

# Global target sample rate used throughout the module
TARGET_SR = 44100

# ---------------------------- Sig map ----------------------------
@dataclass
class MeasureSpec:
    numer: int
    denom: int
    count: int

def parse_sig_map(sig_map: str) -> List[MeasureSpec]:
    parts = [p.strip() for p in sig_map.split(",") if p.strip()]
    result: List[MeasureSpec] = []
    for p in parts:
        try:
            sig, cnt = p.split("(")
            cnt = cnt.strip().rstrip(")")
            num, den = sig.split("/")
            ms = MeasureSpec(int(num), int(den), int(cnt))
            if ms.numer <= 0 or ms.denom not in (1,2,4,8,16,32) or ms.count <= 0:
                raise ValueError("invalid signature block")
            result.append(ms)
        except Exception:
            raise argparse.ArgumentTypeError(f"Invalid --sig_map segment: '{p}'")
    return result

def seconds_per_measure(bpm: float, numer: int, denom: int) -> float:
    spb = (60.0 / max(bpm, 1e-6)) * (4.0 / float(denom))
    return spb * float(numer)


def duration_samples_map(bpm: float, sr: int = TARGET_SR) -> Dict[str, int]:
    """Map common musical durations to sample lengths.

    The mapping spans standard note lengths from ``1/16`` up to a whole note
    and their dotted equivalents.  Durations are quantized using the provided
    BPM and sample rate.
    """

    beat_sec = 60.0 / max(bpm, 1e-6)  # duration of a quarter note in seconds
    unit = int(round((beat_sec / 4.0) * sr))  # sixteenth note in samples
    multiples = {
        "1/16": 1,
        "1/8": 2,
        "1/4": 4,
        "1/2": 8,
        "1": 16,
    }
    out: Dict[str, int] = {}
    for name, mult in multiples.items():
        base = unit * mult
        out[name] = base
        out[f"dotted {name}"] = base * 3 // 2
    return out

# ---------------------------- RNG ----------------------------
def seeded_rng(seed: Optional[str], salt: Optional[str]) -> random.Random:
    import hashlib
    seed_str = (seed or f"default-seed-{time.time_ns()}") + "::" + (salt or "")
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return random.Random(int.from_bytes(h, "big"))

# ---------------------------- Audio utils ----------------------------
def load_audio_from_bytes(
    b: bytes,
    sr: int = TARGET_SR,
    filename: Optional[str] = None,
    mono: bool = True,
) -> Tuple[np.ndarray, int]:
    """Decode audio from a byte string.

    Parameters
    ----------
    b: bytes
        Raw audio data.
    sr: int
        Target sample rate for decoding/resampling.
    filename: Optional[str]
        If provided, the file's name or URL. The suffix (e.g., ``.mp3``) will
        be used for the temporary file to help backends choose the correct
        decoder.
    """
    import tempfile

    suffix = ".bin"
    if filename:
        path = urlparse(filename).path
        ext = os.path.splitext(os.path.basename(path))[1]
        if ext:
            suffix = ext

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(b)
        tmp.flush()

    try:
        y, srr = librosa.load(tmp_path, sr=sr, mono=mono)
    finally:
        os.remove(tmp_path)

    if not np.isfinite(y).any() or y.size == 0:
        raise ValueError("Decoded audio is empty/invalid")
    return y.astype(np.float32), srr

def load_audio_file(path: str, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    y, srr = librosa.load(path, sr=sr, mono=True)
    if not np.isfinite(y).any() or y.size == 0:
        raise ValueError("Decoded audio is empty/invalid")
    return y.astype(np.float32), srr

def downmix_to_mono(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return y.astype(np.float32)
    return np.mean(y, axis=0).astype(np.float32)

def trim_silence(
    y: np.ndarray,
    sr: int,
    top_db: float,
    pad_ms: int,
) -> Tuple[np.ndarray, int, int]:
    mono = downmix_to_mono(y)
    if mono.size == 0:
        return y, 0, 0
    yt, idx = librosa.effects.trim(mono, top_db=top_db)
    start, end = int(idx[0]), int(idx[1])
    pad = int(sr * pad_ms / 1000.0)
    start = max(0, start - pad)
    end = min(mono.size, end + pad)
    if start >= end:
        return y, 0, mono.size
    if y.ndim == 1:
        return y[start:end].astype(np.float32), start, end
    return y[:, start:end].astype(np.float32), start, end

def apply_fades(y: np.ndarray, sr: int, fade_in_ms: int, fade_out_ms: int) -> np.ndarray:
    if y.size == 0:
        return y
    fade_in = max(int(sr * fade_in_ms / 1000.0), 0)
    fade_out = max(int(sr * fade_out_ms / 1000.0), 0)
    n = y.shape[-1] if y.ndim > 1 else len(y)
    if fade_in > 0:
        ramp = np.linspace(0.0, 1.0, min(fade_in, n), dtype=np.float32)
        if y.ndim == 1:
            y[: len(ramp)] *= ramp
        else:
            y[:, : len(ramp)] *= ramp
    if fade_out > 0:
        ramp = np.linspace(1.0, 0.0, min(fade_out, n), dtype=np.float32)
        if y.ndim == 1:
            y[-len(ramp):] *= ramp
        else:
            y[:, -len(ramp):] *= ramp
    return y

def normalize_rms(y: np.ndarray, target_rms: float) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y**2) + 1e-12))
    if rms <= 0.0:
        return y.astype(np.float32)
    g = target_rms / rms
    return (y * g).astype(np.float32)

def normalize_audio(
    y: np.ndarray,
    mode: str,
    peak_db: float,
    target_rms: Optional[float] = None,
) -> np.ndarray:
    if mode == "rms" and target_rms is not None:
        y = normalize_rms(y, target_rms)
    y = normalize_peak(y, peak_db=peak_db)
    return y

def duration_bucket(
    duration_s: float,
    bpm_assumption: float,
    tolerance: float,
    longform: bool = False,
) -> str:
    if longform:
        return "lf"
    sec_per_beat = 60.0 / max(bpm_assumption, 1e-6)
    beats = duration_s / sec_per_beat
    buckets = [
        (0.125, "t2"),
        (0.25, "six"),
        (0.5, "e"),
        (1.0, "q"),
        (2.0, "h"),
        (4.0, "w"),
        (8.0, "b2"),
        (16.0, "b4"),
    ]
    best = min(buckets, key=lambda b: abs(beats - b[0]))
    if abs(beats - best[0]) / max(best[0], 1e-6) <= tolerance:
        return best[1]
    return "free"

def infer_bpm(y: np.ndarray, sr: int) -> Tuple[Optional[float], float]:
    mono = downmix_to_mono(y)
    if mono.size == 0:
        return None, 0.0
    tempo, beats = librosa.beat.beat_track(y=mono, sr=sr, units="time")
    if tempo <= 0 or not np.isfinite(tempo):
        return None, 0.0
    if len(beats) < 2:
        return float(tempo), 0.2
    duration = float(len(mono) / sr)
    expected_beats = max(duration / (60.0 / tempo), 1.0)
    confidence = min(1.0, len(beats) / expected_beats)
    return float(tempo), float(confidence)

def normalize_peak(y: np.ndarray, peak_db: float = -0.8) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-12
    target = 10 ** (peak_db / 20.0)
    g = target / peak
    return (y * g).astype(np.float32)


def safe_audio(y: np.ndarray) -> np.ndarray:
    """Clamp audio to [-1, 1] and remove NaN/Inf values.

    Any non-finite values are replaced with zeros before clamping to the
    valid floating-point audio range.  The result is always ``float32``.
    """

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(y, -1.0, 1.0).astype(np.float32)

def onset_envelope(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    oenv = librosa.onset.onset_strength(y=y, sr=sr)
    times = librosa.times_like(oenv, sr=sr)
    return oenv.astype(np.float32), times.astype(np.float32)

def classify_source(y: np.ndarray, sr: int) -> Dict[str, float]:
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)))
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-12
    flat = float(np.median(librosa.feature.spectral_flatness(S=S)))
    oenv, _ = onset_envelope(y, sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=False, units="frames")
    dens = float(len(onset_frames) / (len(y)/sr + 1e-9))
    return {"zcr": zcr, "flatness": flat, "onset_density": dens}

def bus_of_source(metrics: Dict[str, float]) -> str:
    if metrics["onset_density"] >= 2.0 or (metrics["zcr"] >= 0.08 and metrics["onset_density"] >= 1.0):
        return "perc"
    return "tex"


def _rms_min_index(
    y: np.ndarray, idx: int, sr: int, search_ms: float = 10.0, win_ms: float = 5.0
) -> int:
    """Return the index near ``idx`` with minimum RMS energy.

    A small neighbourhood around ``idx`` is scanned using a sliding window of
    ``win_ms`` milliseconds to locate the sample with minimal RMS.  This helps
    find low-energy points for seamless crossfades.
    """

    if y.size == 0:
        return idx
    search = max(int(sr * search_ms / 1000.0), 1)
    win = max(int(sr * win_ms / 1000.0), 1)
    lo = max(0, idx - search)
    hi = min(len(y) - win, idx + search)
    if hi <= lo:
        return max(0, min(idx, len(y) - 1))
    frames = librosa.util.frame(y[lo : hi + win], frame_length=win, hop_length=1)
    rms = np.sqrt(np.mean(frames**2, axis=0) + 1e-12)
    return lo + int(np.argmin(rms))


def _refine_edges(y: np.ndarray, sr: int, start: int, end: int) -> Tuple[int, int]:
    s = _rms_min_index(y, start, sr)
    e = _rms_min_index(y, end, sr)
    if e <= s:
        e = s + 1
    return s, e

def pick_onset_aligned_window(
    y: np.ndarray,
    sr: int,
    dur_s: float,
    rng: random.Random,
    min_rms: float = 0.02,
    attempts: int = 50,
    refine_edges: bool = True,
) -> Tuple[int, int, float]:
    n = len(y)
    win = max(int(dur_s * sr), 1)
    if n <= win:
        reps = int(math.ceil(win / max(n,1))) + 1
        y = np.tile(y, reps)
        n = len(y)
    oenv, _ = onset_envelope(y, sr)
    onset_idx = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, units="samples", backtrack=True)
    if onset_idx.size == 0:
        onset_idx = np.arange(0, n - win, max(int(0.1*sr), 512))
    best = (0, win, 0.0)
    for _ in range(attempts):
        s0 = int(rng.choice(onset_idx))
        s0 = max(0, min(s0, n - win - 1))
        s1 = s0 + win
        if refine_edges:
            s0, s1 = _refine_edges(y, sr, s0, s1)
        seg = y[s0:s1]
        energy = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if energy > best[2]:
            best = (s0, s1, energy)
        if energy >= min_rms:
            return s0, s1, energy
    step = max(int(sr * 0.05), 256)
    for s0 in range(0, n - win, step):
        s1 = s0 + win
        if refine_edges:
            s0, s1 = _refine_edges(y, sr, s0, s1)
        seg = y[s0:s1]
        energy = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if energy > best[2]:
            best = (s0, s1, energy)
    return best


def pick_beat_aligned_window(
    y: np.ndarray,
    sr: int,
    dur_s: float,
    rng: random.Random,
    min_rms: float = 0.02,
    attempts: int = 50,
    refine_edges: bool = True,
) -> Tuple[int, int, float]:
    """Pick an audio window starting on a detected beat.

    This mirrors :func:`pick_onset_aligned_window` but uses
    :func:`librosa.beat.beat_track` to find beat locations and ensures that the
    returned segment begins at one of those beats.  If no beats are detected,
    the function falls back to a uniform scan similar to onset alignment.
    """

    n = len(y)
    win = max(int(dur_s * sr), 1)
    if n <= win:
        reps = int(math.ceil(win / max(n, 1))) + 1
        y = np.tile(y, reps)
        n = len(y)

    # Detect beat locations in samples
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    beats = beats.astype(int)
    beats = beats[beats < n - win]
    if beats.size == 0:
        # Fall back to fixed steps if no beats are detected
        beats = np.arange(0, n - win, max(int(0.1 * sr), 512))

    best = (0, win, 0.0)
    for _ in range(attempts):
        s0 = int(rng.choice(beats))
        s1 = s0 + win
        if refine_edges:
            s0, s1 = _refine_edges(y, sr, s0, s1)
        seg = y[s0:s1]
        energy = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if energy > best[2]:
            best = (s0, s1, energy)
        if energy >= min_rms:
            return s0, s1, energy

    # Exhaustive search fallback
    step = max(int(sr * 0.05), 256)
    for s0 in range(0, n - win, step):
        s1 = s0 + win
        if refine_edges:
            s0, s1 = _refine_edges(y, sr, s0, s1)
        seg = y[s0:s1]
        energy = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if energy > best[2]:
            best = (s0, s1, energy)
    return best

def time_stretch_to_length(seg: np.ndarray, sr: int, target_len: int, mode: str) -> Tuple[np.ndarray, float]:
    cur = len(seg)
    if target_len <= 0 or cur <= 0:
        return seg, 1.0
    factor = cur / float(target_len)
    if abs(1.0 - factor) < 1e-3 or mode == "off":
        if cur > target_len:
            return seg[:target_len], 1.0
        elif cur < target_len:
            reps = int(math.ceil(target_len / cur))
            return np.tile(seg, reps)[:target_len], 1.0
        else:
            return seg, 1.0
    if mode == "loose":
        candidates = [0.5, 2/3, 3/4, 4/5, 5/6, 1.0, 6/5, 5/4, 4/3, 3/2, 2.0]
        best = min(candidates, key=lambda r: abs(r - 1.0/factor))
        stretch = best
    else:
        stretch = 1.0 / factor
    y = librosa.effects.time_stretch(seg.astype(np.float32), rate=stretch)
    if len(y) > target_len:
        y = y[:target_len]
    elif len(y) < target_len:
        reps = int(math.ceil(target_len / len(y)))
        y = np.tile(y, reps)[:target_len]
    return y.astype(np.float32), stretch


def stack_slices(length: int, placements: List[Tuple[np.ndarray, int]]) -> np.ndarray:
    """Place ``segments`` at the specified ``start`` positions and sum them.

    Parameters
    ----------
    length : int
        Length of the output timeline in samples.
    placements : List[Tuple[np.ndarray, int]]
        Each tuple contains the audio slice and the start index where it should
        be added.

    Returns
    -------
    np.ndarray
        Timeline with all slices summed; slices that would exceed ``length`` are
        truncated.
    """

    out = np.zeros(length, dtype=np.float32)
    for seg, start in placements:
        if start >= length or len(seg) == 0:
            continue
        end = min(length, start + len(seg))
        out[start:end] += seg[: end - start]
    return out


def crossfade_concat(chunks: List[np.ndarray], sr: int, fade_s: float = 0.02) -> np.ndarray:
    if not chunks:
        return np.zeros(1, dtype=np.float32)
    if len(chunks) == 1:
        return chunks[0]
    out = chunks[0].copy()
    fade = max(int(fade_s * sr), 1)
    for nxt in chunks[1:]:
        a = out
        b = nxt
        m = min(fade, len(a), len(b))
        if m > 1:
            t = np.linspace(0, 1, m, dtype=np.float32)
            a_tail = a[-m:] * (1.0 - t)
            b_head = b[:m] * t
            out = np.concatenate([a[:-m], a_tail + b_head, b[m:]])
        else:
            out = np.concatenate([a, b])
    return out

# ---------------------------- Core Build ----------------------------
@dataclass
class SourceRef:
    id: int
    url: str
    ia_identifier: Optional[str]
    ia_file: Optional[str]
    title: Optional[str]
    licenseurl: Optional[str]
    y: np.ndarray
    sr: int
    bus: str
    metrics: Dict[str, float]


def pick_sources(
    conn: Optional[sqlite3.Connection],
    run_id: Optional[int],
    rng: random.Random,
    provider: Provider,
    wanted: int,
    query_bias: Optional[str],
    allow_tokens: List[str],
    strict: bool,
    cache_dir: str,
    pause_s: float = 0.6,
    used_registry: Optional[Set[str]] = None,
) -> List[SourceRef]:
    files = provider.search(rng, wanted, query_bias, allow_tokens, strict)
    if not files:
        lw("No search results (strict filter?). Will try non-strict as fallback.")
        files = provider.search(rng, wanted, query_bias, allow_tokens, False)
    picked: List[SourceRef] = []
    tried = 0
    for f in files:
        if len(picked) >= wanted:
            break
        token = f.get("url") or f.get("identifier")
        if used_registry and token in used_registry:
            continue
        url = f.get("url")
        b = provider.fetch(f, cache_dir)
        if not url or not b or len(b) < 2048:
            continue
        try:
            fname = f.get("name")
            if not fname and url:
                fname = os.path.basename(urlparse(url).path)
            y, sr = load_audio_from_bytes(b, sr=TARGET_SR, filename=fname)
            metrics = classify_source(y, sr)
            bus = bus_of_source(metrics)
            sid: int
            if conn is not None:
                cur = conn.execute(
                    "INSERT INTO sources(run_id,ia_identifier,ia_file,url,title,licenseurl,picked,bus,duration_s,zcr,flatness,onset_density) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        f.get("identifier"),
                        f.get("name"),
                        url,
                        f.get("title"),
                        provider.license(f),
                        1,
                        bus,
                        float(len(y) / sr),
                        float(metrics["zcr"]),
                        float(metrics["flatness"]),
                        float(metrics["onset_density"]),
                    ),
                )
                sid = cur.lastrowid
            else:
                sid = len(picked) + 1
            picked.append(
                SourceRef(
                    sid,
                    url,
                    f.get("identifier"),
                    f.get("name"),
                    f.get("title"),
                    provider.license(f),
                    y,
                    sr,
                    bus,
                    metrics,
                )
            )
            li(
                f"Loaded source #{len(picked)} [{bus}] len={len(y)/sr:.1f}s  zcr={metrics['zcr']:.3f} flat={metrics['flatness']:.3f} dens={metrics['onset_density']:.2f}/s"
            )
        except Exception as e:
            lw(f"Decode/classify failed: {e}")
        tried += 1
        time.sleep(pause_s * (0.75 + 0.5 * rng.random()))
        if tried > wanted * 18 and picked:
            break
    if not picked:
        li("Synthesizing emergency noise and click sources.")
        t = np.arange(TARGET_SR * 8) / TARGET_SR
        clicks = (np.sin(2 * np.pi * 1000 * t) * (t % 0.5 < 0.01)).astype(np.float32) * 0.2
        if conn is not None:
            cur = conn.execute(
                "INSERT INTO sources(run_id,ia_identifier,ia_file,url,title,licenseurl,picked,bus,duration_s,zcr,flatness,onset_density) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    None,
                    None,
                    "synth://clicks",
                    "synthetic clicks",
                    "publicdomain",
                    1,
                    "perc",
                    float(len(clicks) / TARGET_SR),
                    0.1,
                    0.4,
                    3.0,
                ),
            )
            sid1 = cur.lastrowid
        else:
            sid1 = len(picked) + 1
        picked.append(
            SourceRef(
                sid1,
                "synth://clicks",
                None,
                None,
                "synthetic clicks",
                "publicdomain",
                clicks,
                TARGET_SR,
                "perc",
                {"zcr": 0.1, "flatness": 0.4, "onset_density": 3.0},
            )
        )
        noise = (
            np.random.default_rng(0)
            .standard_normal(int(TARGET_SR * 8))
            .astype(np.float32)
            * 0.03
        )
        if conn is not None:
            cur = conn.execute(
                "INSERT INTO sources(run_id,ia_identifier,ia_file,url,title,licenseurl,picked,bus,duration_s,zcr,flatness,onset_density) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run_id,
                    None,
                    None,
                    "synth://noise",
                    "synthetic noise",
                    "publicdomain",
                    1,
                    "tex",
                    float(len(noise) / TARGET_SR),
                    0.05,
                    0.9,
                    0.2,
                ),
            )
            sid2 = cur.lastrowid
        else:
            sid2 = len(picked) + 1
        picked.append(
            SourceRef(
                sid2,
                "synth://noise",
                None,
                None,
                "synthetic noise",
                "publicdomain",
                noise,
                TARGET_SR,
                "tex",
                {"zcr": 0.05, "flatness": 0.9, "onset_density": 0.2},
            )
        )
    return picked


def preview_sources(
    provider: Provider,
    rng: random.Random,
    wanted: int,
    query_bias: Optional[str],
    allow_tokens: List[str],
    strict: bool,
) -> List[Dict[str, Optional[str]]]:
    """Plan candidate sources without downloading audio."""
    files = provider.search(rng, wanted, query_bias, allow_tokens, strict)
    if not files:
        files = provider.search(rng, wanted, query_bias, allow_tokens, False)
    planned: List[Dict[str, Optional[str]]] = []
    for f in files:
        planned.append(
            {
                "identifier": f.get("identifier"),
                "file": f.get("name"),
                "title": f.get("title"),
                "licenseurl": provider.license(f),
                "url": f.get("url"),
            }
        )
        if len(planned) >= wanted:
            break
    return planned

def choose_by_bus(sources: List[SourceRef], bus: str, rng: random.Random) -> SourceRef:
    cands = [s for s in sources if s.bus == bus]
    if not cands:
        cands = sources
    return rng.choice(cands)

def build_measures(sig_specs: List[MeasureSpec]) -> List[Tuple[int,int]]:
    out = []
    for ms in sig_specs:
        for _ in range(ms.count):
            out.append((ms.numer, ms.denom))
    return out

def assemble_track(
    conn: Optional[sqlite3.Connection],
    run_id: Optional[int],
    sources: List[SourceRef],
    measures: List[Tuple[int, int]],
    bpm: float,
    rng: random.Random,
    min_rms: float,
    crossfade_s: float,
    tempo_mode: str,
    stems_dirs: Dict[str, str],
    microfill: bool,
    beat_align: bool,
    refine_boundaries: bool = True,
    num_sounds: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate percussion and texture mixes from source material.

    Parameters
    ----------
    conn : sqlite3.Connection, optional
        Database handle used to log each generated segment.  When ``None``,
        no database logging is performed.
    run_id : int, optional
        Identifier for the current generation run.
    sources : List[SourceRef]
        Pool of candidate sources with pre-loaded audio data.
    measures : List[Tuple[int, int]]
        Measure specifications given as ``(numerator, denominator)`` pairs.
    bpm : float
        Tempo of the output in beats per minute.
    rng : random.Random
        Random number generator for choosing sources and offsets.
    min_rms : float
        Minimum RMS energy allowed when selecting audio windows.
    crossfade_s : float
        Length of crossfades between consecutive segments, in seconds.
    tempo_mode : str
        Strategy for time-stretching segments to the target duration.
    stems_dirs : Dict[str, str]
        Optional mapping of stem types (``"perc"`` or ``"tex"``) to directory
        paths where individual measure stems are written.  If empty, no stems
        are saved.
    microfill : bool
        When ``True``, occasionally blend a short trailing snippet from the
        source into each texture segment to add variation.
    beat_align : bool
        If ``True``, windows are aligned to detected beats instead of onsets.
    refine_boundaries : bool
        When ``True``, search for low-energy points near segment boundaries
        before extracting audio.  This helps avoid clicks when crossfading
        segments together.

    Returns
    -------
    mix_perc : np.ndarray
        Final percussion mix produced by concatenating all percussion segments
        with crossfades.
    mix_tex : np.ndarray
        Final texture mix produced by concatenating all texture segments with
        crossfades.

    Algorithm
    ---------
    1. Distribute ``num_sounds`` quantized slices across all measures.
    2. Place each slice on a quantized timeline for its measure and stack all
       slices per bus.
    3. Record metadata in ``conn`` and optionally write individual stems.
    4. Crossfade-concatenate the per-measure grooves to obtain the two returned
       mixes.
    """
    pick_fn = pick_beat_aligned_window if beat_align else pick_onset_aligned_window
    num_sounds = num_sounds or rng.randint(15, 30)
    m = max(len(measures), 1)
    base, rem = divmod(num_sounds, m)
    per_measure = [base] * m
    if rem:
        for i in rng.sample(range(m), rem):
            per_measure[i] += 1
    dur_map = duration_samples_map(bpm)
    quant = dur_map["1/16"]
    perc_chunks, tex_chunks = [], []
    for idx, (numer, denom) in enumerate(measures):
        dur = seconds_per_measure(bpm, numer, denom)
        target_len = int(dur * TARGET_SR)
        perc_slices: List[Tuple[np.ndarray, int]] = []
        tex_slices: List[Tuple[np.ndarray, int]] = []
        for _ in range(per_measure[idx]):
            bus = rng.choice(["perc", "tex"])
            src = choose_by_bus(sources, bus, rng)
            dur_name, dur_samples = rng.choice(list(dur_map.items()))
            dur_samples = min(dur_samples, target_len)
            start = rng.randrange(0, max(1, target_len - dur_samples + quant), quant)
            dur_s = dur_samples / TARGET_SR
            s0, s1, energy = pick_fn(
                src.y,
                src.sr,
                dur_s,
                rng=rng,
                min_rms=min_rms if bus == "perc" else min_rms * 0.5,
                refine_edges=refine_boundaries,
            )
            seg = src.y[s0:s1]
            seg, factor = time_stretch_to_length(seg, src.sr, dur_samples, tempo_mode)
            if bus == "tex" and microfill and rng.random() < 0.25:
                fill_len = max(int(0.2 * TARGET_SR), 1)
                fill_len = min(fill_len, len(seg))
                s0f = max(0, s1 - fill_len - 1)
                fill = src.y[s0f : s0f + fill_len]
                if len(fill) < fill_len:
                    fill = np.pad(fill, (0, fill_len - len(fill)))
                seg[-fill_len:] = 0.6 * seg[-fill_len:] + 0.4 * fill
            placement = (seg.astype(np.float32), start)
            if bus == "perc":
                perc_slices.append(placement)
            else:
                tex_slices.append(placement)
            if conn is not None:
                conn.execute(
                    "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id,energy,tempo_factor) VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        idx,
                        numer,
                        denom,
                        bus,
                        float(s0 / src.sr),
                        float(dur_s),
                        src.id,
                        float(energy),
                        float(factor),
                    ),
                )
        groove_p = stack_slices(target_len, perc_slices)
        groove_t = stack_slices(target_len, tex_slices)
        perc_chunks.append(groove_p)
        tex_chunks.append(groove_t)
        if stems_dirs:
            if stems_dirs.get("perc"):
                sf.write(
                    os.path.join(stems_dirs["perc"], f"perc_{idx:03d}_{numer}-{denom}.wav"),
                    safe_audio(groove_p),
                    TARGET_SR,
                    subtype="PCM_16",
                )
            if stems_dirs.get("tex"):
                sf.write(
                    os.path.join(stems_dirs["tex"], f"tex_{idx:03d}_{numer}-{denom}.wav"),
                    safe_audio(groove_t),
                    TARGET_SR,
                    subtype="PCM_16",
                )
    if conn is not None:
        conn.commit()
    mix_perc = crossfade_concat(perc_chunks, TARGET_SR, fade_s=crossfade_s)
    mix_tex = crossfade_concat(tex_chunks, TARGET_SR, fade_s=crossfade_s * 0.8)
    return mix_perc, mix_tex
