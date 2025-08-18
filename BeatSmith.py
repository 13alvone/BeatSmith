#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BeatSmith v2: Internet-Archive powered stochastic beat builder with onset alignment,
license allow-list, percussive/texture buses, tempo-fit, caching, presets, stems,
and deterministic seeding.

Dependencies:
  pip install numpy scipy librosa soundfile requests mido

System deps:
  ffmpeg (for decoding via librosa/audioread)

Examples:
  ./beatsmith_v2.py out "4/4(8)" --bpm 124 --preset boom-bap --seed funk --salt v2 --stems

  ./beatsmith_v2.py out "4/4(4),5/4(3),6/8(5)" --bpm 132 \
    --license-allow "cc0,cc-by,publicdomain" \
    --num-sources 6 --tempo-fit strict --compress \
    --eq-low +2 --eq-mid -1 --eq-high +3 \
    --reverb-mix 0.22 --reverb-room 0.35 \
    --tremolo-rate 5 --tremolo-depth 0.35 \
    --echo-ms 320 --echo-fb 0.28 --echo-mix 0.22 \
    --stems --verbose

  # Build atop an existing track with lookahead sidechain
  ./beatsmith_v2.py out "4/4(16)" --bpm 124 --build-on ./my_loop.wav \
    --sidechain 0.6 --sidechain-lookahead-ms 10
"""
import argparse
import hashlib
import json
import logging
import math
import os
import random
import sqlite3
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import requests
import soundfile as sf
from scipy.signal import butter, lfilter
import librosa

try:
    import mido  # optional for MIDI export (future use)
    HAVE_MIDO = True
except Exception:
    HAVE_MIDO = False

# ---------------------------- Logging ----------------------------
LOG_FORMAT = "%(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
log = logging.getLogger("BeatSmithV2")
def li(msg: str): log.info("[i] " + msg)
def lw(msg: str): log.warning("[!] " + msg)
def ld(msg: str): log.debug("[DEBUG] " + msg)
def le(msg: str): log.error("[x] " + msg)

# ---------------------------- SQLite ----------------------------
SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  out_dir TEXT NOT NULL,
  bpm REAL NOT NULL,
  sig_map TEXT NOT NULL,
  seed TEXT,
  salt TEXT,
  params_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources(
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  ia_identifier TEXT,
  ia_file TEXT,
  url TEXT NOT NULL,
  title TEXT,
  licenseurl TEXT,
  picked INTEGER NOT NULL DEFAULT 0,
  bus TEXT NOT NULL,            -- 'perc' or 'tex'
  duration_s REAL NOT NULL,
  zcr REAL,
  flatness REAL,
  onset_density REAL,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS segments(
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  measure_index INTEGER NOT NULL,
  numer INTEGER NOT NULL,
  denom INTEGER NOT NULL,
  bus TEXT NOT NULL,            -- 'perc' or 'tex'
  start_s REAL NOT NULL,
  dur_s REAL NOT NULL,
  source_id INTEGER NOT NULL,
  energy REAL,
  tempo_factor REAL,
  FOREIGN KEY(run_id) REFERENCES runs(id),
  FOREIGN KEY(source_id) REFERENCES sources(id)
);
"""

def db_open(path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(SCHEMA)
        return conn
    except Exception as e:
        le(f"DB open/init failed: {e}")
        sys.exit(1)

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
    seed_str = (seed or f"default-seed-{time.time_ns()}") + "::" + (salt or "")
    h = hashlib.sha256(seed_str.encode("utf-8")).digest()
    return random.Random(int.from_bytes(h, "big"))

# ---------------------------- IA ----------------------------
IA_ADV_URL = "https://archive.org/advancedsearch.php"
IA_META_URL = "https://archive.org/metadata/"
AUDIO_EXTS = {".wav", ".wave", ".aif", ".aiff", ".flac", ".mp3", ".ogg", ".oga", ".m4a"}

def ia_license_ok(licenseurl: Optional[str], allow_tokens: List[str]) -> bool:
    if not licenseurl:
        return False
    low = licenseurl.lower()
    for tok in allow_tokens:
        if tok.strip() and tok.strip().lower() in low:
            return True
    return False

def ia_search_random(rng: random.Random, rows: int, query_bias: Optional[str],
                     allow_tokens: List[str], strict: bool) -> List[Dict]:
    q_parts = ['mediatype:(audio)']
    if query_bias:
        q_parts.append(f'({query_bias})')
    q = " AND ".join(q_parts)
    params = {
        "q": q,
        "fl[]": ["identifier", "title", "licenseurl", "downloads"],
        "sort[]": "downloads desc",
        "rows": max(10, rows),
        "page": 1,
        "output": "json",
    }
    headers = {"User-Agent": "BeatSmith/2.0 (+open source art engine)"}
    try:
        r = requests.get(IA_ADV_URL, params=params, timeout=20, headers=headers)
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        lw(f"IA search failed: {e}")
        return []
    docs = data.get("response", {}).get("docs", [])
    if strict:
        docs = [d for d in docs if ia_license_ok(d.get("licenseurl"), allow_tokens)]
    rng.shuffle(docs)
    return docs

def ia_pick_files_for_item(identifier: str) -> List[Dict]:
    url = IA_META_URL + identifier
    headers = {"User-Agent": "BeatSmith/2.0 (+open source art engine)"}
    try:
        r = requests.get(url, timeout=20, headers=headers)
        r.raise_for_status()
        meta = r.json()
    except Exception as e:
        lw(f"IA metadata failed for {identifier}: {e}")
        return []
    files = meta.get("files", []) or []
    picked = []
    for f in files:
        name = f.get("name") or ""
        lower = name.lower()
        if any(lower.endswith(ext) for ext in AUDIO_EXTS):
            picked.append({
                "identifier": identifier,
                "name": name,
                "title": meta.get("metadata", {}).get("title"),
                "licenseurl": meta.get("metadata", {}).get("licenseurl"),
                "url": f"https://archive.org/download/{identifier}/{name}"
            })
    return picked

# ---------------------------- Cache + HTTP ----------------------------
def cache_path_for(url: str, cache_dir: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    h = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return os.path.join(cache_dir, h + ".bin")

def http_get_cached(url: str, cache_dir: str, timeout: int = 30, max_bytes: int = 50_000_000) -> Optional[bytes]:
    path = cache_path_for(url, cache_dir)
    if os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                b = f.read()
                if b:
                    ld(f"Cache hit: {url}")
                    return b
        except Exception as e:
            lw(f"Cache read failed (will re-download): {e}")
    headers = {"User-Agent": "BeatSmith/2.0 (+open source art engine)"}
    try:
        with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
            r.raise_for_status()
            chunks = []
            size = 0
            for chunk in r.iter_content(chunk_size=65536):
                if chunk:
                    chunks.append(chunk)
                    size += len(chunk)
                    if size > max_bytes:
                        lw(f"Truncating download > {max_bytes} bytes: {url}")
                        break
            b = b"".join(chunks)
            if b:
                try:
                    with open(path, "wb") as f:
                        f.write(b)
                except Exception as e:
                    lw(f"Cache write failed: {e}")
            return b
    except Exception as e:
        lw(f"Download failed: {e} :: {url}")
        return None

# ---------------------------- Audio utils ----------------------------
TARGET_SR = 44100

def load_audio_from_bytes(b: bytes, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=True) as tmp:
        tmp.write(b)
        tmp.flush()
        y, srr = librosa.load(tmp.name, sr=sr, mono=True)
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
    # Zero-crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=512)))
    # Spectral flatness
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256)) + 1e-12
    flat = float(np.median(librosa.feature.spectral_flatness(S=S)))
    # Onset density (per second)
    oenv, _ = onset_envelope(y, sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, backtrack=False, units="frames")
    dens = float(len(onset_frames) / (len(y)/sr + 1e-9))
    return {"zcr": zcr, "flatness": flat, "onset_density": dens}

def bus_of_source(metrics: Dict[str, float]) -> str:
    # Heuristic: percussive has higher onset density and zcr; texture has lower density & flatter spectra
    if metrics["onset_density"] >= 2.0 or (metrics["zcr"] >= 0.08 and metrics["onset_density"] >= 1.0):
        return "perc"
    # very noisy/flat also tends to be texture pad/noise bed
    return "tex"

def pick_onset_aligned_window(y: np.ndarray, sr: int, dur_s: float, rng: random.Random,
                              min_rms: float = 0.02, attempts: int = 50) -> Tuple[int, int, float]:
    """
    Choose a window start near a strong onset; ensure average RMS above threshold.
    Fallback to loudest window.
    """
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
    # fallback: loudest coarse scan
    step = max(int(sr * 0.05), 256)
    for s0 in range(0, n - win, step):
        seg = y[s0:s0+win]
        energy = float(np.sqrt(np.mean(seg**2) + 1e-12))
        if energy > best[2]:
            best = (s0, s0+win, energy)
    return best

def time_stretch_to_length(seg: np.ndarray, sr: int, target_len: int, mode: str) -> Tuple[np.ndarray, float]:
    """
    Stretch/compress seg to exactly target_len samples. Returns (stretched, factor).
    mode: 'off'|'loose'|'strict' (strict uses precise factor; loose rounds to ~nearest musical ratio)
    """
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
        # snap to simple musical ratios to reduce artifacts
        candidates = [0.5, 2/3, 3/4, 4/5, 5/6, 1.0, 6/5, 5/4, 4/3, 3/2, 2.0]
        best = min(candidates, key=lambda r: abs(r - 1.0/factor))
        stretch = best
    else:
        stretch = 1.0 / factor
    # librosa time_stretch expects rate>0; it returns len≈len(seg)/rate
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

# ---------------------------- FX ----------------------------
def rms_envelope(y: np.ndarray, frame: int = 1024, hop: int = 256) -> np.ndarray:
    if len(y) < frame:
        return np.sqrt(np.mean(y**2) + 1e-12) * np.ones(1, dtype=np.float32)
    y2 = librosa.util.frame(y, frame_length=frame, hop_length=hop, axis=0)
    rms = np.sqrt((y2**2).mean(axis=0) + 1e-12)
    return rms.astype(np.float32)

def compressor(y: np.ndarray, sr: int, thresh_db=-18.0, ratio=4.0, attack=0.01, release=0.1, makeup_db=2.0) -> np.ndarray:
    frame = max(int(sr * 0.01), 256)
    hop = frame // 2
    env = rms_envelope(y, frame, hop)
    env_sr = np.interp(np.arange(len(y)), np.arange(len(env)) * hop + frame/2.0, env, left=env[0], right=env[-1]) + 1e-12
    thresh = 10 ** (thresh_db / 20.0)
    over = env_sr / thresh
    comp = np.where(over > 1.0, (over ** (1.0 - 1.0/ratio)), 1.0)
    out = np.empty_like(y)
    g = 1.0
    atk = math.exp(-1.0 / (attack * sr))
    rel = math.exp(-1.0 / (release * sr))
    for i in range(len(y)):
        target = 1.0 / comp[i]
        if target < g:
            g = atk * (g - target) + target
        else:
            g = rel * (g - target) + target
        out[i] = y[i] * g
    out *= 10 ** (makeup_db / 20.0)
    return out.astype(np.float32)

def _butter_band(sr, low=None, high=None, order=4):
    ny = sr * 0.5
    if low is None and high is None:
        raise ValueError("low or high must be set")
    if low is None:
        Wn = high / ny
        b, a = butter(order, Wn, btype='low')
    elif high is None:
        Wn = low / ny
        b, a = butter(order, Wn, btype='high')
    else:
        Wn = [low/ny, high/ny]
        b, a = butter(order, Wn, btype='band')
    return b, a

def eq_three_band(y: np.ndarray, sr: int, low_db=0.0, mid_db=0.0, high_db=0.0,
                  lo_cut=200.0, hi_cut=2000.0) -> np.ndarray:
    bL, aL = _butter_band(sr, high=lo_cut)
    bM, aM = _butter_band(sr, low=lo_cut, high=hi_cut)
    bH, aH = _butter_band(sr, low=hi_cut)
    low = lfilter(bL, aL, y)
    mid = lfilter(bM, aM, y)
    high = lfilter(bH, aH, y)
    gl = 10 ** (low_db / 20.0)
    gm = 10 ** (mid_db / 20.0)
    gh = 10 ** (high_db / 20.0)
    out = low*gl + mid*gm + high*gh
    return out.astype(np.float32)

def reverb_schroeder(y: np.ndarray, sr: int, room_size=0.3, mix=0.2) -> np.ndarray:
    base = np.array([29.7, 37.1, 41.1, 43.7]) * (0.5 + room_size)
    gains = np.array([0.805, 0.827, 0.783, 0.764])
    wet = np.zeros_like(y)
    for d_ms, g in zip(base, gains):
        d = int(sr * (d_ms / 1000.0))
        if d <= 1: 
            continue
        fb = np.zeros_like(y)
        for i in range(len(y)):
            prev = fb[i-d] if i-d >= 0 else 0.0
            fb[i] = y[i] + g * prev
        wet += fb
    # two allpasses (simple)
    def allpass(x, delay_ms=5.0, g=0.7):
        d = int(sr * (delay_ms / 1000.0))
        if d <= 1: return x
        out = np.zeros_like(x)
        z = 0.0
        for i in range(len(x)):
            xin = x[i]
            xdel = out[i-d] if i-d>=0 else 0.0
            out[i] = -g * xin + xdel + g * (z if i>0 else 0.0)
            z = xin
        return out
    wet = allpass(wet, 5.0, 0.7)
    wet = allpass(wet, 1.7, 0.7)
    return ((1.0 - mix) * y + mix * wet).astype(np.float32)

def tremolo(y: np.ndarray, sr: int, rate_hz=5.0, depth=0.5) -> np.ndarray:
    t = np.arange(len(y)) / float(sr)
    mod = (1.0 - depth) + depth * 0.5 * (1.0 + np.sin(2*np.pi*rate_hz*t))
    return (y * mod).astype(np.float32)

def phaser(y: np.ndarray, sr: int, rate_hz=0.2, depth=0.6, stages=4) -> np.ndarray:
    t = np.arange(len(y)) / float(sr)
    mod = 0.5 * (1.0 + np.sin(2*np.pi*rate_hz*t))
    x = y.copy()
    for _ in range(stages):
        out = np.zeros_like(x)
        a = 0.2 + depth*0.78*mod
        z = 0.0
        for i in range(len(x)):
            out[i] = -a[i]*x[i] + z + a[i]*(out[i-1] if i>0 else 0.0)
            z = x[i]
        x = out
    return x.astype(np.float32)

def echo(y: np.ndarray, sr: int, delay_ms=350.0, feedback=0.3, mix=0.25) -> np.ndarray:
    d = int(sr * (delay_ms / 1000.0))
    if d <= 1:
        return y
    out = y.copy()
    fb_buf = y.copy()
    for i in range(d, len(y)):
        out[i] += mix * fb_buf[i-d]
        fb_buf[i] += feedback * fb_buf[i-d]
    return out.astype(np.float32)

def lookahead_sidechain(beat: np.ndarray, key: np.ndarray, sr: int, amount: float, look_ms: float) -> np.ndarray:
    # amount 0..1, lookahead in ms
    env = rms_envelope(key, frame=max(int(sr*0.02),512), hop=max(int(sr*0.01),256))
    env_sr = np.interp(np.arange(len(beat)), np.arange(len(env))*max(int(sr*0.01),256)+max(int(sr*0.02),512)/2.0, env, left=env[0], right=env[-1])
    env_sr = env_sr / (env_sr.max() + 1e-12)
    # shift env earlier for lookahead
    shift = int(sr * (max(0.0, look_ms)/1000.0))
    env_sr = np.concatenate([env_sr[shift:], np.repeat(env_sr[-1], shift)])
    gain = 1.0 - max(0.0, min(amount,1.0)) * env_sr
    return (beat * gain).astype(np.float32)

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
        if args.reverb_mix == 0.0: args.reverb_mix = 0.08
        if args.query_bias is None: args.query_bias = "drums OR percussion OR vinyl OR breakbeat"
    elif pres == "edm":
        if args.eq_low == 0.0: args.eq_low = +4.0
        if args.eq_mid == 0.0: args.eq_mid = -2.0
        if args.eq_high == 0.0: args.eq_high = +2.0
        if not args.compress: args.compress = True
        if args.reverb_mix == 0.0: args.reverb_mix = 0.15
        if args.echo_ms == 0.0: args.echo_ms = 300.0
        if args.echo_mix == 0.0: args.echo_mix = 0.2
        if args.query_bias is None: args.query_bias = "electronic OR drum machine OR loop"
    elif pres == "lofi":
        if args.eq_low == 0.0: args.eq_low = +2.0
        if args.eq_mid == 0.0: args.eq_mid = +1.5
        if args.eq_high == 0.0: args.eq_high = -2.0
        if args.tremolo_rate == 0.0: args.tremolo_rate = 4.0
        if args.tremolo_depth == 0.0: args.tremolo_depth = 0.25
        if args.reverb_mix == 0.0: args.reverb_mix = 0.12
        if args.query_bias is None: args.query_bias = "jazz OR vinyl OR mellow OR ambient"
    else:
        lw(f"Unknown preset '{args.preset}', ignoring.")

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

def pick_sources(conn: sqlite3.Connection, run_id: int, rng: random.Random,
                 wanted: int, query_bias: Optional[str],
                 allow_tokens: List[str], strict: bool,
                 cache_dir: str, pause_s: float = 0.6) -> List[SourceRef]:
    docs = ia_search_random(rng, rows=max(50, wanted*15), query_bias=query_bias,
                            allow_tokens=allow_tokens, strict=strict)
    if not docs:
        lw("No IA search results (strict filter?). Will try non-strict as fallback.")
        docs = ia_search_random(rng, rows=max(50, wanted*15), query_bias=query_bias,
                                allow_tokens=allow_tokens, strict=False)
    picked: List[SourceRef] = []
    tried = 0
    for doc in docs:
        if len(picked) >= wanted: break
        ident = doc.get("identifier")
        files = ia_pick_files_for_item(ident) if ident else []
        rng.shuffle(files)
        for f in files:
            if len(picked) >= wanted: break
            url = f["url"]
            b = http_get_cached(url, cache_dir)
            if not b or len(b) < 2048:
                continue
            try:
                y, sr = load_audio_from_bytes(b, sr=TARGET_SR)
                metrics = classify_source(y, sr)
                bus = bus_of_source(metrics)
                cur = conn.execute(
                    "INSERT INTO sources(run_id,ia_identifier,ia_file,url,title,licenseurl,picked,bus,duration_s,zcr,flatness,onset_density) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, f.get("identifier"), f.get("name"), url, f.get("title"), f.get("licenseurl"),
                     1, bus, float(len(y)/sr), float(metrics["zcr"]), float(metrics["flatness"]), float(metrics["onset_density"]))
                )
                sid = cur.lastrowid
                picked.append(SourceRef(sid, url, f.get("identifier"), f.get("name"), f.get("title"),
                                        f.get("licenseurl"), y, sr, bus, metrics))
                li(f"Loaded source #{len(picked)} [{bus}] len={len(y)/sr:.1f}s  zcr={metrics['zcr']:.3f} flat={metrics['flatness']:.3f} dens={metrics['onset_density']:.2f}/s")
            except Exception as e:
                lw(f"Decode/classify failed: {e}")
            tried += 1
            time.sleep(pause_s * (0.75 + 0.5*rng.random()))
        if tried > wanted*18 and picked:
            break
    if not picked:
        # Emergency synthetic source
        li("Synthesizing emergency noise and click sources.")
        # Perc: clicks
        t = np.arange(TARGET_SR*8)/TARGET_SR
        clicks = (np.sin(2*np.pi*1000*t)*(t%0.5<0.01)).astype(np.float32)*0.2
        cur = conn.execute(
            "INSERT INTO sources(run_id,ia_identifier,ia_file,url,title,licenseurl,picked,bus,duration_s,zcr,flatness,onset_density) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, None, None, "synth://clicks", "synthetic clicks", "publicdomain",
             1, "perc", float(len(clicks)/TARGET_SR), 0.1, 0.4, 3.0)
        )
        sid1 = cur.lastrowid
        picked.append(SourceRef(sid1, "synth://clicks", None, None, "synthetic clicks",
                                "publicdomain", clicks, TARGET_SR, "perc",
                                {"zcr":0.1,"flatness":0.4,"onset_density":3.0}))
        # Tex: noise pad
        noise = np.random.default_rng(0).standard_normal(int(TARGET_SR*8)).astype(np.float32)*0.03
        cur = conn.execute(
            "INSERT INTO sources(run_id,ia_identifier,ia_file,url,title,licenseurl,picked,bus,duration_s,zcr,flatness,onset_density) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, None, None, "synth://noise", "synthetic noise", "publicdomain",
             1, "tex", float(len(noise)/TARGET_SR), 0.05, 0.9, 0.2)
        )
        sid2 = cur.lastrowid
        picked.append(SourceRef(sid2, "synth://noise", None, None, "synthetic noise",
                                "publicdomain", noise, TARGET_SR, "tex",
                                {"zcr":0.05,"flatness":0.9,"onset_density":0.2}))
    return picked

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
    """
    Returns (mix_perc, mix_tex) buses separately (caller mixes/processes them).
    """
    perc_chunks, tex_chunks = [], []
    for idx, (numer, denom) in enumerate(measures):
        dur = seconds_per_measure(bpm, numer, denom)
        target_len = int(dur * TARGET_SR)

        # Percussive layer
        src_p = choose_by_bus(sources, "perc", rng)
        s0, s1, e_p = pick_onset_aligned_window(src_p.y, src_p.sr, dur, rng=rng, min_rms=min_rms)
        seg_p = src_p.y[s0:s1]
        seg_p, factor_p = time_stretch_to_length(seg_p, src_p.sr, target_len, tempo_mode)
        perc_chunks.append(seg_p.astype(np.float32))
        conn.execute(
            "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id,energy,tempo_factor) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, idx, numer, denom, "perc", float(s0/src_p.sr), float(dur), src_p.id, float(e_p), float(factor_p))
        )

        # Texture layer
        src_t = choose_by_bus(sources, "tex", rng)
        s0t, s1t, e_t = pick_onset_aligned_window(src_t.y, src_t.sr, dur, rng=rng, min_rms=min_rms*0.5)
        seg_t = src_t.y[s0t:s1t]
        seg_t, factor_t = time_stretch_to_length(seg_t, src_t.sr, target_len, tempo_mode)
        # optional micro-fill at end of measure for texture sparkle
        if microfill and rng.random() < 0.25:
            fill_len = max(int(0.2 * TARGET_SR), 1)
            s0f = max(0, s1t - fill_len - 1)
            fill = src_t.y[s0f:s0f+fill_len]
            if len(fill) < fill_len:
                fill = np.pad(fill, (0, fill_len - len(fill)))
            # place in last 200ms with small fade-in
            seg_t[-fill_len:] = 0.6*seg_t[-fill_len:] + 0.4*fill
        tex_chunks.append(seg_t.astype(np.float32))
        conn.execute(
            "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id,energy,tempo_factor) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run_id, idx, numer, denom, "tex", float(s0t/src_t.sr), float(dur), src_t.id, float(e_t), float(factor_t))
        )

        # Stems
        if stems_dirs:
            if stems_dirs.get("perc"):
                sf.write(os.path.join(stems_dirs["perc"], f"perc_{idx:03d}_{numer}-{denom}.wav"), seg_p, TARGET_SR, subtype="PCM_16")
            if stems_dirs.get("tex"):
                sf.write(os.path.join(stems_dirs["tex"], f"tex_{idx:03d}_{numer}-{denom}.wav"), seg_t, TARGET_SR, subtype="PCM_16")

    conn.commit()
    mix_perc = crossfade_concat(perc_chunks, TARGET_SR, fade_s=crossfade_s)
    mix_tex  = crossfade_concat(tex_chunks, TARGET_SR, fade_s=crossfade_s*0.8)
    return mix_perc, mix_tex

# ---------------------------- CLI ----------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="BeatSmith v2: pull CC/PD audio from Internet Archive, onset-align slices per signature map, run percussive+texture buses, FX, and render a beat."
    )
    # Required positional
    p.add_argument("out_dir", help="Output directory (created if missing).")
    p.add_argument("sig_map", type=parse_sig_map, help="Signature map like '4/4(4),5/4(3),6/8(5)'.")

    # Core
    p.add_argument("--bpm", type=float, default=120.0, help="Beats per minute (default: 120).")
    p.add_argument("--seed", type=str, default=None, help="Deterministic seed.")
    p.add_argument("--salt", type=str, default=None, help="Additional salt for alternate takes.")
    p.add_argument("--num-sources", type=int, default=6, help="How many IA sources to fetch (default: 6).")
    p.add_argument("--query-bias", type=str, default=None, help="Optional IA search bias string.")
    p.add_argument("--license-allow", type=str, default="cc0,creativecommons,public domain,publicdomain,cc-by,cc-by-sa", help="Comma list of license tokens allowed.")
    p.add_argument("--strict-license", action="store_true", help="Enforce license allow-list strictly (no fallback).")
    p.add_argument("--cache-dir", type=str, default=os.path.expanduser("~/.beatsmith/cache"), help="Download cache directory.")
    p.add_argument("--min-rms", type=float, default=0.02, help="Minimum RMS for audible slice.")
    p.add_argument("--crossfade", type=float, default=0.02, help="Seconds of crossfade between measures.")
    p.add_argument("--tempo-fit", choices=["off","loose","strict"], default="strict", help="Time-stretch mode to fit global measure length.")
    p.add_argument("--stems", action="store_true", help="Write stems per bus/measure.")
    p.add_argument("--microfill", action="store_true", help="Enable tiny end-of-measure fills on texture bus.")
    p.add_argument("--preset", type=str, default=None, help="Preset: boom-bap | edm | lofi")
    p.add_argument("--verbose", action="store_true", help="Enable debug logs.")

    # Build-on options
    p.add_argument("--build-on", type=str, default=None, help="Path to an existing base track to mix under.")
    p.add_argument("--sidechain", type=float, default=0.0, help="Sidechain duck amount 0..1 against base (default 0).")
    p.add_argument("--sidechain-lookahead-ms", type=float, default=0.0, help="Lookahead (ms) for sidechain.")

    # FX (master)
    p.add_argument("--compress", action="store_true", help="Enable master bus compressor.")
    p.add_argument("--comp-thresh", type=float, default=-18.0)
    p.add_argument("--comp-ratio", type=float, default=4.0)
    p.add_argument("--comp-makeup", type=float, default=2.0)

    p.add_argument("--eq-low", type=float, default=0.0, help="Low shelf gain dB")
    p.add_argument("--eq-mid", type=float, default=0.0, help="Mid band gain dB")
    p.add_argument("--eq-high", type=float, default=0.0, help="High shelf gain dB")

    p.add_argument("--reverb-mix", type=float, default=0.0, help="0..1 wet mix; 0 disables")
    p.add_argument("--reverb-room", type=float, default=0.3, help="0..1 room size-ish")

    p.add_argument("--tremolo-rate", type=float, default=0.0, help=">0 to enable (Hz)")
    p.add_argument("--tremolo-depth", type=float, default=0.5, help="0..1")

    p.add_argument("--phaser-rate", type=float, default=0.0, help=">0 to enable (Hz)")
    p.add_argument("--phaser-depth", type=float, default=0.6, help="0..1")

    p.add_argument("--echo-ms", type=float, default=0.0, help=">0 to enable (ms delay)")
    p.add_argument("--echo-fb", type=float, default=0.3, help="Feedback 0..1")
    p.add_argument("--echo-mix", type=float, default=0.25, help="Wet mix 0..1")

    return p

def main():
    args = build_parser().parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)
        li("Verbose logging enabled.")

    apply_preset(args)

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    stems_dirs = {}
    if args.stems:
        stems_root = os.path.join(out_dir, "stems")
        os.makedirs(stems_root, exist_ok=True)
        perc_dir = os.path.join(stems_root, "perc")
        tex_dir  = os.path.join(stems_root, "tex")
        os.makedirs(perc_dir, exist_ok=True)
        os.makedirs(tex_dir, exist_ok=True)
        stems_dirs = {"perc": perc_dir, "tex": tex_dir}

    db_path = os.path.join(out_dir, "beatsmith_v2.db")
    conn = db_open(db_path)

    seed = args.seed or f"default-seed-{time.time_ns()}"
    rng = seeded_rng(seed, args.salt)

    # Record run
    params = vars(args).copy()
    params["sig_map"] = None
    sig_str = ",".join(f"{ms.numer}/{ms.denom}({ms.count})" for ms in args.sig_map)
    row = conn.execute(
        "INSERT INTO runs(created_at,out_dir,bpm,sig_map,seed,salt,params_json) VALUES (datetime('now'),?,?,?,?,?,?)",
        (out_dir, float(args.bpm), sig_str, seed, args.salt or "", json.dumps(params, ensure_ascii=False))
    )
    run_id = row.lastrowid
    li(f"Run id={run_id} BPM={args.bpm} sig_map={sig_str} seed='{seed}' salt='{args.salt or ''}'")

    # Fetch sources
    allow_tokens = [t.strip() for t in (args.license_allow or "").split(",") if t.strip()]
    strict = bool(args.strict_license)
    li("Selecting Internet Archive sources...")
    sources = pick_sources(
        conn, run_id, rng, wanted=max(2, args.num_sources),
        query_bias=args.query_bias, allow_tokens=allow_tokens, strict=strict,
        cache_dir=args.cache_dir
    )
    if not sources:
        le("No sources available, aborting.")
        sys.exit(2)

    # Build measure plan
    measures = build_measures(args.sig_map)
    total_sec = sum(seconds_per_measure(args.bpm, n, d) for n, d in measures)
    li(f"Total measures: {len(measures)}  est length ≈ {total_sec:.1f}s")

    # Assemble buses
    li("Assembling onset-aligned measures (perc + tex buses)...")
    mix_perc, mix_tex = assemble_track(
        conn, run_id, sources, measures, bpm=args.bpm, rng=rng,
        min_rms=args.min_rms, crossfade_s=args.crossfade,
        tempo_mode=args.tempo_fit, stems_dirs=stems_dirs, microfill=args.microfill
    )

    # Balance buses (simple)
    # Perc bus slightly louder; texture -3 dB baseline
    tex_gain = 10 ** (-3.0 / 20.0)
    L = max(len(mix_perc), len(mix_tex))
    if len(mix_perc) < L: mix_perc = np.pad(mix_perc, (0, L-len(mix_perc)))
    if len(mix_tex)  < L: mix_tex  = np.pad(mix_tex,  (0, L-len(mix_tex)))
    mix = mix_perc + tex_gain * mix_tex

    # Build-on layering
    if args.build_on:
        try:
            li(f"Loading base track: {args.build_on}")
            base, _ = load_audio_file(args.build_on, sr=TARGET_SR)
            LL = max(len(mix), len(base))
            if len(mix)  < LL: mix  = np.pad(mix,  (0, LL-len(mix)))
            if len(base) < LL: base = np.pad(base, (0, LL-len(base)))
            if args.sidechain > 0.0:
                li(f"Applying lookahead sidechain amount={args.sidechain:.2f} lookahead={args.sidechain_lookahead_ms if hasattr(args,'sidechain_lookahead_ms') else 0}ms")
                mix = lookahead_sidechain(mix, base, TARGET_SR, amount=float(max(0.0, min(args.sidechain, 1.0))), look_ms=float(max(0.0, args.sidechain_lookahead_ms)))
            # equal-power-ish mix
            mix = (0.5*base + 0.5*mix).astype(np.float32)
        except Exception as e:
            lw(f"Base layering failed: {e}")

    # Master FX
    if args.compress:
        li("Applying master compressor...")
        mix = compressor(mix, TARGET_SR, thresh_db=args.comp_thresh, ratio=args.comp_ratio, makeup_db=args.comp_makeup)
    if any(abs(g)>1e-6 for g in [args.eq_low, args.eq_mid, args.eq_high]):
        li(f"Applying master EQ: low={args.eq_low} mid={args.eq_mid} high={args.eq_high}")
        mix = eq_three_band(mix, TARGET_SR, low_db=args.eq_low, mid_db=args.eq_mid, high_db=args.eq_high)
    if args.reverb_mix > 0.0:
        li(f"Applying reverb (mix={args.reverb_mix:.2f}, room={args.reverb_room:.2f})")
        mix = reverb_schroeder(mix, TARGET_SR, room_size=args.reverb_room, mix=args.reverb_mix)
    if args.tremolo_rate > 0.0:
        li(f"Applying tremolo (rate={args.tremolo_rate}Hz, depth={args.tremolo_depth})")
        mix = tremolo(mix, TARGET_SR, rate_hz=args.tremolo_rate, depth=args.tremolo_depth)
    if args.phaser_rate > 0.0:
        li(f"Applying phaser (rate={args.phaser_rate}Hz, depth={args.phaser_depth})")
        mix = phaser(mix, TARGET_SR, rate_hz=args.phaser_rate, depth=args.phaser_depth)
    if args.echo_ms > 0.0:
        li(f"Applying echo (delay={args.echo_ms}ms, fb={args.echo_fb}, mix={args.echo_mix})")
        # Use wet-only plus dry to avoid runaway when combined with compressor
        wet = echo(mix.copy(), TARGET_SR, delay_ms=args.echo_ms, feedback=args.echo_fb, mix=args.echo_mix)
        mix = (0.75*mix + 0.25*wet).astype(np.float32)

    mix = normalize_peak(mix, peak_db=-0.8)
    out_wav = os.path.join(out_dir, f"beatsmith_v2_{run_id}.wav")
    sf.write(out_wav, mix, TARGET_SR, subtype="PCM_16")
    li(f"Wrote: {out_wav}")

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

