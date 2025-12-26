import math
from typing import Optional

import numpy as np
from scipy.signal import butter, lfilter
import librosa


__all__ = [
    # Canonical primitives (original API)
    "rms_envelope",
    "compressor",
    "eq_three_band",
    "reverb_schroeder",
    "tremolo",
    "phaser",
    "echo",
    "lookahead_sidechain",
    # Compatibility surface (apply_* API)
    "apply_compression",
    "apply_eq_three_band",
    "apply_eq",
    "apply_reverb_schroeder",
    "apply_reverb",
    "apply_tremolo",
    "apply_phaser",
    "apply_modulation",
    "apply_echo",
    "apply_delay",
]


# ---------------------------- Helpers ----------------------------
def _as_mono(y: np.ndarray) -> np.ndarray:
    """Ensure 1D float32 mono."""
    if y is None:
        return np.zeros(1, dtype=np.float32)
    y = np.asarray(y)
    if y.size == 0:
        return np.zeros(1, dtype=np.float32)
    if y.ndim == 1:
        return y.astype(np.float32, copy=False)
    # If caller passes (channels, n) or (n, channels), downmix safely.
    if y.shape[0] <= 8 and y.shape[0] < y.shape[-1]:
        mono = np.mean(y, axis=0)
    else:
        mono = np.mean(y, axis=-1)
    mono = np.nan_to_num(mono, nan=0.0, posinf=0.0, neginf=0.0)
    return mono.astype(np.float32, copy=False)


def _clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


# ---------------------------- Core DSP ----------------------------
def rms_envelope(y: np.ndarray, frame: int = 1024, hop: int = 256) -> np.ndarray:
    y = _as_mono(y)
    frame = max(int(frame), 1)
    hop = max(int(hop), 1)

    if len(y) < frame:
        return (np.sqrt(np.mean(y**2) + 1e-12) * np.ones(1, dtype=np.float32)).astype(np.float32)

    y2 = librosa.util.frame(y, frame_length=frame, hop_length=hop, axis=0)
    rms = np.sqrt((y2**2).mean(axis=0) + 1e-12)
    return rms.astype(np.float32)


def compressor(
    y: np.ndarray,
    sr: int,
    thresh_db: float = -18.0,
    ratio: float = 4.0,
    attack: float = 0.01,
    release: float = 0.1,
    makeup_db: float = 2.0,
) -> np.ndarray:
    y = _as_mono(y)
    sr = max(int(sr), 1)

    frame = max(int(sr * 0.01), 256)
    hop = max(frame // 2, 1)

    env = rms_envelope(y, frame=frame, hop=hop)

    # Up-sample envelope to sample-rate with midpoint alignment.
    env_sr = np.interp(
        np.arange(len(y), dtype=np.float64),
        (np.arange(len(env), dtype=np.float64) * hop) + (frame / 2.0),
        env.astype(np.float64, copy=False),
        left=float(env[0]) if len(env) else 0.0,
        right=float(env[-1]) if len(env) else 0.0,
    ).astype(np.float64, copy=False) + 1e-12

    thresh = 10 ** (float(thresh_db) / 20.0)
    ratio = max(float(ratio), 1e-6)
    over = env_sr / max(thresh, 1e-12)

    comp = np.where(over > 1.0, (over ** (1.0 - 1.0 / ratio)), 1.0)

    out = np.empty_like(y, dtype=np.float32)
    g = 1.0

    attack = max(float(attack), 1e-6)
    release = max(float(release), 1e-6)
    atk = math.exp(-1.0 / (attack * sr))
    rel = math.exp(-1.0 / (release * sr))

    for i in range(len(y)):
        target = 1.0 / float(comp[i])
        if target < g:
            g = atk * (g - target) + target
        else:
            g = rel * (g - target) + target
        out[i] = float(y[i]) * float(g)

    out *= 10 ** (float(makeup_db) / 20.0)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def _butter_band(sr: int, low: Optional[float] = None, high: Optional[float] = None, order: int = 4):
    sr = max(int(sr), 1)
    ny = sr * 0.5
    order = max(int(order), 1)

    if low is None and high is None:
        raise ValueError("low or high must be set")

    if low is None:
        Wn = float(high) / ny
        b, a = butter(order, Wn, btype="low")
    elif high is None:
        Wn = float(low) / ny
        b, a = butter(order, Wn, btype="high")
    else:
        Wn = [float(low) / ny, float(high) / ny]
        b, a = butter(order, Wn, btype="band")
    return b, a


def eq_three_band(
    y: np.ndarray,
    sr: int,
    low_db: float = 0.0,
    mid_db: float = 0.0,
    high_db: float = 0.0,
    low_cut: float = 200.0,
    high_cut: float = 4000.0,
) -> np.ndarray:
    y = _as_mono(y)
    sr = max(int(sr), 1)

    low_cut = float(max(10.0, low_cut))
    high_cut = float(max(low_cut + 10.0, high_cut))

    b_l, a_l = _butter_band(sr, high=low_cut)
    b_h, a_h = _butter_band(sr, low=high_cut)
    b_m, a_m = _butter_band(sr, low=low_cut, high=high_cut)

    low = lfilter(b_l, a_l, y)
    high = lfilter(b_h, a_h, y)
    mid = lfilter(b_m, a_m, y)

    gl = 10 ** (float(low_db) / 20.0)
    gm = 10 ** (float(mid_db) / 20.0)
    gh = 10 ** (float(high_db) / 20.0)

    out = (gl * low) + (gm * mid) + (gh * high)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def reverb_schroeder(y: np.ndarray, sr: int, room_size: float = 0.3, mix: float = 0.2) -> np.ndarray:
    y = _as_mono(y)
    sr = max(int(sr), 1)

    room_size = _clamp01(float(room_size))
    mix = _clamp01(float(mix))

    def comb(x: np.ndarray, delay_ms: float = 50.0, g: float = 0.7) -> np.ndarray:
        delay = max(int(sr * delay_ms / 1000.0), 1)
        buf = np.zeros(delay, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float64)
        g = float(np.clip(g, -0.9999, 0.9999))
        for i in range(len(x)):
            idx = i % delay
            yi = float(x[i]) + g * buf[idx]
            out[i] = yi
            buf[idx] = yi
        out = np.nan_to_num(out)
        return np.clip(out, -1.0, 1.0)

    def allpass(x: np.ndarray, delay_ms: float = 5.0, g: float = 0.7) -> np.ndarray:
        delay = max(int(sr * delay_ms / 1000.0), 1)
        buf = np.zeros(delay, dtype=np.float64)
        out = np.zeros_like(x, dtype=np.float64)
        g = float(np.clip(g, -0.9999, 0.9999))
        for i in range(len(x)):
            idx = i % delay
            yi = -g * float(x[i]) + buf[idx]
            out[i] = yi
            buf[idx] = float(x[i]) + g * yi
        out = np.nan_to_num(out)
        return np.clip(out, -1.0, 1.0)

    dry = y.astype(np.float64, copy=False)
    x = dry.copy()

    # Simple Schroeder topology: comb bank + allpass diffusion.
    for dm in [50, 56, 61, 68]:
        x = comb(x, delay_ms=float(dm), g=0.805 + room_size * 0.1)
    for dm in [5, 1.7, 6.3]:
        x = allpass(x, delay_ms=float(dm), g=0.7)

    wet = x
    out = (1.0 - mix) * dry + mix * wet
    out = np.nan_to_num(out)
    out = np.clip(out, -1.0, 1.0)
    return out.astype(np.float32)


def tremolo(y: np.ndarray, sr: int, rate_hz: float = 5.0, depth: float = 0.5) -> np.ndarray:
    y = _as_mono(y)
    sr = max(int(sr), 1)

    rate_hz = float(max(0.0, rate_hz))
    depth = _clamp01(float(depth))

    t = np.arange(len(y), dtype=np.float64) / float(sr)
    mod = (1.0 - depth) + depth * np.sin(2.0 * np.pi * rate_hz * t)
    out = y.astype(np.float64, copy=False) * mod
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def phaser(y: np.ndarray, sr: int, rate_hz: float = 0.2, depth: float = 0.6, stages: int = 4) -> np.ndarray:
    y = _as_mono(y)
    sr = max(int(sr), 1)

    rate_hz = float(max(0.0, rate_hz))
    depth = _clamp01(float(depth))
    stages = max(int(stages), 1)

    t = np.arange(len(y), dtype=np.float64) / float(sr)
    mod = (np.sin(2.0 * np.pi * rate_hz * t) + 1.0) * 0.5

    out = y.copy().astype(np.float32)
    for _ in range(stages):
        delay = (mod * depth * 1024.0).astype(int) + 1
        buf = np.zeros(2048, dtype=np.float32)
        for i in range(len(out)):
            out[i] = out[i] + buf[i % 2048]
            buf[i % 2048] = out[i] - buf[i % 2048]
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def echo(y: np.ndarray, sr: int, delay_ms: float = 350.0, feedback: float = 0.3, mix: float = 0.25) -> np.ndarray:
    y = _as_mono(y)
    sr = max(int(sr), 1)

    delay_ms = float(max(1.0, delay_ms))
    feedback = float(np.clip(feedback, -0.999, 0.999))
    mix = _clamp01(float(mix))

    delay = max(int(sr * delay_ms / 1000.0), 1)
    buf = np.zeros(delay, dtype=np.float32)
    out = np.zeros_like(y, dtype=np.float32)

    for i in range(len(y)):
        out[i] = y[i] + buf[i % delay]
        buf[i % delay] = y[i] + buf[i % delay] * feedback

    out = ((1.0 - mix) * y + mix * out).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


def lookahead_sidechain(beat: np.ndarray, key: np.ndarray, sr: int, amount: float, look_ms: float) -> np.ndarray:
    beat = _as_mono(beat)
    key = _as_mono(key)
    sr = max(int(sr), 1)

    amount = _clamp01(float(amount))
    look_ms = float(max(0.0, look_ms))

    frame = max(int(sr * 0.02), 512)
    hop = max(int(sr * 0.01), 256)

    env = rms_envelope(key, frame=frame, hop=hop)
    if len(env) == 0:
        return beat.astype(np.float32)

    env_sr = np.interp(
        np.arange(len(beat), dtype=np.float64),
        (np.arange(len(env), dtype=np.float64) * hop) + (frame / 2.0),
        env.astype(np.float64, copy=False),
        left=float(env[0]),
        right=float(env[-1]),
    ).astype(np.float64, copy=False)

    env_sr = env_sr / (float(np.max(env_sr)) + 1e-12)

    shift = int(sr * (look_ms / 1000.0))
    if shift > 0 and shift < len(env_sr):
        env_sr = np.concatenate([env_sr[shift:], np.repeat(env_sr[-1], shift)])
    elif shift >= len(env_sr):
        env_sr = np.repeat(env_sr[-1], len(env_sr))

    gain = 1.0 - amount * env_sr
    out = beat.astype(np.float64, copy=False) * gain
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out.astype(np.float32)


# ---------------------------- Compatibility wrappers ----------------------------
def apply_compression(
    y: np.ndarray,
    sr: int,
    thresh_db: float = -18.0,
    ratio: float = 4.0,
    attack: float = 0.01,
    release: float = 0.1,
    makeup_db: float = 2.0,
    **kwargs,
) -> np.ndarray:
    # Common alias used by some CLIs/configs.
    if "threshold_db" in kwargs and kwargs["threshold_db"] is not None:
        thresh_db = float(kwargs["threshold_db"])
    return compressor(
        y=y,
        sr=sr,
        thresh_db=thresh_db,
        ratio=ratio,
        attack=attack,
        release=release,
        makeup_db=makeup_db,
    )


def apply_eq_three_band(
    y: np.ndarray,
    sr: int,
    low_db: float = 0.0,
    mid_db: float = 0.0,
    high_db: float = 0.0,
    low_cut: float = 200.0,
    high_cut: float = 4000.0,
    **kwargs,
) -> np.ndarray:
    return eq_three_band(
        y=y,
        sr=sr,
        low_db=low_db,
        mid_db=mid_db,
        high_db=high_db,
        low_cut=low_cut,
        high_cut=high_cut,
    )


def apply_eq(
    y: np.ndarray,
    sr: int,
    low_db: float = 0.0,
    mid_db: float = 0.0,
    high_db: float = 0.0,
    low_cut: float = 200.0,
    high_cut: float = 4000.0,
    **kwargs,
) -> np.ndarray:
    # Generic “apply_eq” expected by some CLI surfaces.
    return apply_eq_three_band(
        y=y,
        sr=sr,
        low_db=low_db,
        mid_db=mid_db,
        high_db=high_db,
        low_cut=low_cut,
        high_cut=high_cut,
    )


def apply_reverb_schroeder(
    y: np.ndarray,
    sr: int,
    room_size: float = 0.3,
    mix: float = 0.2,
    **kwargs,
) -> np.ndarray:
    return reverb_schroeder(y=y, sr=sr, room_size=room_size, mix=mix)


def apply_reverb(
    y: np.ndarray,
    sr: int,
    room_size: float = 0.3,
    mix: float = 0.2,
    **kwargs,
) -> np.ndarray:
    # Many pipelines just call it “apply_reverb”.
    return apply_reverb_schroeder(y=y, sr=sr, room_size=room_size, mix=mix)


def apply_tremolo(
    y: np.ndarray,
    sr: int,
    rate_hz: float = 5.0,
    depth: float = 0.5,
    **kwargs,
) -> np.ndarray:
    return tremolo(y=y, sr=sr, rate_hz=rate_hz, depth=depth)


def apply_phaser(
    y: np.ndarray,
    sr: int,
    rate_hz: float = 0.2,
    depth: float = 0.6,
    stages: int = 4,
    **kwargs,
) -> np.ndarray:
    return phaser(y=y, sr=sr, rate_hz=rate_hz, depth=depth, stages=stages)


def apply_modulation(
    y: np.ndarray,
    sr: int,
    kind: str = "tremolo",
    rate_hz: float = 5.0,
    depth: float = 0.5,
    stages: int = 4,
    **kwargs,
) -> np.ndarray:
    """
    Generic modulation entrypoint (compat).

    Routes to:
      - phaser if kind/effect/mode indicates "phaser" OR stages is provided
      - otherwise tremolo
    """
    k = kind
    for alias in ("effect", "mode"):
        if alias in kwargs and kwargs[alias] is not None:
            k = str(kwargs[alias])
            break
    k = (k or "tremolo").strip().lower()

    # If caller explicitly provides stages, treat as phaser intent.
    if "stages" in kwargs and kwargs["stages"] is not None:
        try:
            stages = int(kwargs["stages"])
            k = "phaser"
        except Exception:
            pass

    if k == "phaser":
        if "rate" in kwargs and kwargs["rate"] is not None:
            rate_hz = float(kwargs["rate"])
        if "depth" in kwargs and kwargs["depth"] is not None:
            depth = float(kwargs["depth"])
        return phaser(y=y, sr=sr, rate_hz=rate_hz, depth=depth, stages=stages)

    # Default: tremolo
    if "rate" in kwargs and kwargs["rate"] is not None:
        rate_hz = float(kwargs["rate"])
    if "depth" in kwargs and kwargs["depth"] is not None:
        depth = float(kwargs["depth"])
    return tremolo(y=y, sr=sr, rate_hz=rate_hz, depth=depth)


def apply_echo(
    y: np.ndarray,
    sr: int,
    delay_ms: float = 350.0,
    feedback: float = 0.3,
    mix: float = 0.25,
    **kwargs,
) -> np.ndarray:
    return echo(y=y, sr=sr, delay_ms=delay_ms, feedback=feedback, mix=mix)


def apply_delay(
    y: np.ndarray,
    sr: int,
    delay_ms: float = 350.0,
    feedback: float = 0.3,
    mix: float = 0.25,
    **kwargs,
) -> np.ndarray:
    # Common synonym for echo in some pipelines.
    return apply_echo(y=y, sr=sr, delay_ms=delay_ms, feedback=feedback, mix=mix)

