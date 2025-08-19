import math
import numpy as np
from scipy.signal import butter, lfilter
import librosa

__all__ = [
    "rms_envelope", "compressor", "eq_three_band", "reverb_schroeder",
    "tremolo", "phaser", "echo", "lookahead_sidechain"
]

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
                  low_cut=200.0, high_cut=4000.0) -> np.ndarray:
    b_l, a_l = _butter_band(sr, high=low_cut)
    b_h, a_h = _butter_band(sr, low=high_cut)
    b_m, a_m = _butter_band(sr, low=low_cut, high=high_cut)
    low = lfilter(b_l, a_l, y)
    high = lfilter(b_h, a_h, y)
    mid = lfilter(b_m, a_m, y)
    gl = 10 ** (low_db / 20.0)
    gm = 10 ** (mid_db / 20.0)
    gh = 10 ** (high_db / 20.0)
    out = gl*low + gm*mid + gh*high
    return out.astype(np.float32)

def reverb_schroeder(y: np.ndarray, sr: int, room_size=0.3, mix=0.2) -> np.ndarray:
    def comb(x, delay_ms=50.0, g=0.7):
        delay = int(sr * delay_ms / 1000.0)
        buf = np.zeros(delay, dtype=np.float32)
        out = np.zeros_like(x)
        for i in range(len(x)):
            yi = x[i] + g * buf[i % delay]
            out[i] = yi
            buf[i % delay] = yi
        return out

    def allpass(x, delay_ms=5.0, g=0.7):
        delay = int(sr * delay_ms / 1000.0)
        buf = np.zeros(delay, dtype=np.float32)
        out = np.zeros_like(x)
        for i in range(len(x)):
            yi = buf[i % delay] + x[i] - g * out[i-1 if i>0 else 0]
            out[i] = yi
            buf[i % delay] = x[i] + g * yi
        return out

    x = y.copy()
    for dm in [50, 56, 61, 68]:
        x = comb(x, delay_ms=dm, g=0.805 + room_size*0.1)
    for dm in [5, 1.7, 6.3]:
        x = allpass(x, delay_ms=dm, g=0.7)
    wet = x.astype(np.float32)
    return (1.0 - mix) * y + mix * wet

def tremolo(y: np.ndarray, sr: int, rate_hz=5.0, depth=0.5) -> np.ndarray:
    t = np.arange(len(y)) / sr
    mod = (1.0 - depth) + depth * np.sin(2*np.pi*rate_hz*t)
    return (y * mod).astype(np.float32)

def phaser(y: np.ndarray, sr: int, rate_hz=0.2, depth=0.6, stages=4) -> np.ndarray:
    t = np.arange(len(y)) / sr
    mod = (np.sin(2*np.pi*rate_hz*t) + 1) * 0.5
    out = y.copy().astype(np.float32)
    for _ in range(stages):
        delay = (mod * depth * 1024).astype(int) + 1
        buf = np.zeros(2048, dtype=np.float32)
        for i in range(len(out)):
            d = delay[i]  # noqa: F841
            out[i] = out[i] + buf[i % 2048]
            buf[i % 2048] = out[i] - buf[i % 2048]
    return out

def echo(y: np.ndarray, sr: int, delay_ms=350.0, feedback=0.3, mix=0.25) -> np.ndarray:
    delay = int(sr * delay_ms / 1000.0)
    buf = np.zeros(delay, dtype=np.float32)
    out = np.zeros_like(y)
    for i in range(len(y)):
        out[i] = y[i] + buf[i % delay]
        buf[i % delay] = y[i] + buf[i % delay] * feedback
    return ((1.0 - mix) * y + mix * out).astype(np.float32)

def lookahead_sidechain(beat: np.ndarray, key: np.ndarray, sr: int, amount: float, look_ms: float) -> np.ndarray:
    env = rms_envelope(key, frame=max(int(sr*0.02),512), hop=max(int(sr*0.01),256))
    env_sr = np.interp(np.arange(len(beat)), np.arange(len(env))*max(int(sr*0.01),256)+max(int(sr*0.02),512)/2.0, env, left=env[0], right=env[-1])
    env_sr = env_sr / (env_sr.max() + 1e-12)
    shift = int(sr * (max(0.0, look_ms)/1000.0))
    env_sr = np.concatenate([env_sr[shift:], np.repeat(env_sr[-1], shift)])
    gain = 1.0 - max(0.0, min(amount,1.0)) * env_sr
    return (beat * gain).astype(np.float32)
