import argparse
import math
import os
import random
import sqlite3
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import librosa

from . import li, lw
from .providers import Provider

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

# ---------------------------- RNG ----------------------------
def seeded_rng(seed: Optional[str], salt: Optional[str]) -> random.Random:
    import hashlib
    seed_str = (seed or f"default-seed-{time.time_ns()}") + "::" + (salt or "")
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return random.Random(int.from_bytes(h, "big"))

# ---------------------------- Audio utils ----------------------------
TARGET_SR = 44100


def load_audio_from_bytes(
    b: bytes, sr: int = TARGET_SR, filename: Optional[str] = None
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
        y, srr = librosa.load(tmp_path, sr=sr, mono=True)
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

def normalize_peak(y: np.ndarray, peak_db: float = -0.8) -> np.ndarray:
    peak = np.max(np.abs(y)) + 1e-12
    target = 10 ** (peak_db / 20.0)
    g = target / peak
    return (y * g).astype(np.float32)

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

def pick_onset_aligned_window(y: np.ndarray, sr: int, dur_s: float, rng: random.Random,
                              min_rms: float = 0.02, attempts: int = 50) -> Tuple[int, int, float]:
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
        seg = y[s0:s0+win]
        energy = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if energy > best[2]:
            best = (s0, s0+win, energy)
        if energy >= min_rms:
            return s0, s0+win, energy
    step = max(int(sr * 0.05), 256)
    for s0 in range(0, n - win, step):
        seg = y[s0:s0+win]
        energy = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if energy > best[2]:
            best = (s0, s0+win, energy)
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
    conn: sqlite3.Connection,
    run_id: int,
    rng: random.Random,
    provider: Provider,
    wanted: int,
    query_bias: Optional[str],
    allow_tokens: List[str],
    strict: bool,
    cache_dir: str,
    pause_s: float = 0.6,
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

def assemble_track(conn: sqlite3.Connection, run_id: int, sources: List[SourceRef],
                     measures: List[Tuple[int,int]], bpm: float, rng: random.Random,
                     min_rms: float, crossfade_s: float, tempo_mode: str,
                     stems_dirs: Dict[str, str], microfill: bool) -> Tuple[np.ndarray, np.ndarray]:
    perc_chunks, tex_chunks = [], []
    for idx, (numer, denom) in enumerate(measures):
        dur = seconds_per_measure(bpm, numer, denom)
        target_len = int(dur * TARGET_SR)
        src_p = choose_by_bus(sources, "perc", rng)
        s0, s1, e_p = pick_onset_aligned_window(src_p.y, src_p.sr, dur, rng=rng, min_rms=min_rms)
        seg_p = src_p.y[s0:s1]
        seg_p, factor_p = time_stretch_to_length(seg_p, src_p.sr, target_len, tempo_mode)
        perc_chunks.append(seg_p.astype(np.float32))
        conn.execute(
            "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id,energy,tempo_factor) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, idx, numer, denom, "perc", float(s0/src_p.sr), float(dur), src_p.id, float(e_p), float(factor_p))
        )
        src_t = choose_by_bus(sources, "tex", rng)
        s0t, s1t, e_t = pick_onset_aligned_window(src_t.y, src_t.sr, dur, rng=rng, min_rms=min_rms*0.5)
        seg_t = src_t.y[s0t:s1t]
        seg_t, factor_t = time_stretch_to_length(seg_t, src_t.sr, target_len, tempo_mode)
        if microfill and rng.random() < 0.25:
            fill_len = max(int(0.2 * TARGET_SR), 1)
            s0f = max(0, s1t - fill_len - 1)
            fill = src_t.y[s0f:s0f+fill_len]
            if len(fill) < fill_len:
                fill = np.pad(fill, (0, fill_len - len(fill)))
            seg_t[-fill_len:] = 0.6*seg_t[-fill_len:] + 0.4*fill
        tex_chunks.append(seg_t.astype(np.float32))
        conn.execute(
            "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id,energy,tempo_factor) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, idx, numer, denom, "tex", float(s0t/src_t.sr), float(dur), src_t.id, float(e_t), float(factor_t))
        )
        if stems_dirs:
            if stems_dirs.get("perc"):
                sf.write(os.path.join(stems_dirs["perc"], f"perc_{idx:03d}_{numer}-{denom}.wav"), seg_p, TARGET_SR, subtype="PCM_16")
            if stems_dirs.get("tex"):
                sf.write(os.path.join(stems_dirs["tex"], f"tex_{idx:03d}_{numer}-{denom}.wav"), seg_t, TARGET_SR, subtype="PCM_16")
    conn.commit()
    mix_perc = crossfade_concat(perc_chunks, TARGET_SR, fade_s=crossfade_s)
    mix_tex  = crossfade_concat(tex_chunks, TARGET_SR, fade_s=crossfade_s*0.8)
    return mix_perc, mix_tex
