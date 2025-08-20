import argparse
import base64
import io

import numpy as np
import pytest
import soundfile as sf
import librosa
import random
import audioread
from hypothesis import given, strategies as st, settings

from beatsmith.audio import (
    MeasureSpec,
    parse_sig_map,
    crossfade_concat,
    time_stretch_to_length,
    load_audio_from_bytes,
    pick_onset_aligned_window,
    pick_beat_aligned_window,
)


# ---------------------- parse_sig_map ----------------------

def test_parse_sig_map_valid():
    sig = "4/4(2), 3/8(1)"
    specs = parse_sig_map(sig)
    assert specs == [MeasureSpec(4, 4, 2), MeasureSpec(3, 8, 1)]


def test_parse_sig_map_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_sig_map("4/5(2)")


@given(st.text())
@settings(max_examples=50, deadline=None)
def test_parse_sig_map_fuzz(s):
    try:
        parse_sig_map(s)
    except argparse.ArgumentTypeError:
        pass


# ---------------------- crossfade_concat ----------------------

def test_crossfade_concat_continuity():
    sr = 1000
    a = np.ones(sr, dtype=np.float32)
    b = np.ones(sr, dtype=np.float32)
    out = crossfade_concat([a, b], sr, fade_s=0.1)
    fade = int(0.1 * sr)
    assert len(out) == sr * 2 - fade
    assert np.allclose(out, 1.0, atol=1e-6)


@given(
    a_val=st.floats(-1.0, 1.0),
    b_val=st.floats(-1.0, 1.0),
    length=st.integers(min_value=2, max_value=1000),
    fade=st.integers(min_value=1, max_value=500),
)
@settings(max_examples=50, deadline=None)
def test_crossfade_concat_property(a_val, b_val, length, fade):
    sr = 1000
    a = np.full(length, a_val, dtype=np.float32)
    b = np.full(length, b_val, dtype=np.float32)
    fade = min(fade, length)
    out = crossfade_concat([a, b], sr, fade_s=fade / sr)
    m = min(fade, len(a), len(b))
    if m > 1:
        t = np.linspace(0, 1, m, dtype=np.float32)
        expected = np.concatenate([
            np.full(len(a) - m, a_val, dtype=np.float32),
            a_val * (1.0 - t) + b_val * t,
            np.full(len(b) - m, b_val, dtype=np.float32),
        ])
    else:
        expected = np.concatenate([a, b])
    assert np.allclose(out, expected, atol=1e-5)


# ---------------------- time_stretch_to_length ----------------------

@pytest.mark.parametrize("mode", ["off", "loose", "strict"])
@pytest.mark.parametrize("target", [50, 150])
def test_time_stretch_to_length_length(mode, target):
    seg = np.linspace(0, 1, 100).astype(np.float32)
    y, _ = time_stretch_to_length(seg, sr=100, target_len=target, mode=mode)
    assert len(y) == target


@given(
    seg_len=st.integers(min_value=2048, max_value=4096),
    target=st.integers(min_value=1024, max_value=4096),
    mode=st.sampled_from(["off", "loose", "strict"]),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
@settings(max_examples=10, deadline=None)
def test_time_stretch_invariants(seg_len, target, mode, seed):
    rng = np.random.default_rng(seed)
    seg = rng.standard_normal(seg_len).astype(np.float32)
    y, factor = time_stretch_to_length(seg, sr=22050, target_len=target, mode=mode)
    assert len(y) == target
    in_rms = float(np.sqrt(np.mean(seg**2)))
    out_rms = float(np.sqrt(np.mean(y**2)))
    assert not np.isnan(out_rms)
    if in_rms > 0:
        assert out_rms <= in_rms * 10.0


# ---------------------- alignment window helpers ----------------------

def _steady_click_track(sr=22050, bpm=120, beats=16):
    beat_dur = 60.0 / bpm
    times = np.arange(beats) * beat_dur
    length = int((beats + 1) * beat_dur * sr)
    y = librosa.clicks(times=times, sr=sr, click_duration=0.03, length=length)
    return y.astype(np.float32), sr, int(round(beat_dur * sr))


def test_pick_onset_aligned_window_click_track():
    y, sr, beat_samples = _steady_click_track()
    dur = 60.0 / 120.0
    s0, s1, _ = pick_onset_aligned_window(y, sr, dur, rng=random.Random(0))
    assert min(abs(s0 - i * beat_samples) for i in range(16)) < sr * 0.03


def test_pick_beat_aligned_window_click_track():
    y, sr, _ = _steady_click_track()
    dur = 60.0 / 120.0
    s0, s1, _ = pick_beat_aligned_window(y, sr, dur, rng=random.Random(0))
    _, beats = librosa.beat.beat_track(y=y, sr=sr, units="samples")
    assert min(abs(int(s0) - b) for b in beats) < sr * 0.03


@given(st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=10, deadline=None)
def test_crossfade_click_free_boundaries(seed):
    rng_np = np.random.default_rng(seed)
    sr = 8000
    length = sr * 2
    y = (rng_np.standard_normal(length) * 0.1).astype(np.float32)
    # Inject sharp impulses to create strong onsets
    pos1 = rng_np.integers(100, sr - 100)
    pos2 = rng_np.integers(sr + 100, length - 100)
    y[pos1] += rng_np.uniform(1.0, 3.0)
    y[pos2] -= rng_np.uniform(1.0, 3.0)
    s0, s1, _ = pick_onset_aligned_window(
        y, sr, 0.5, rng=random.Random(int(seed)), refine_edges=True, min_rms=0.0
    )
    seg = y[s0:s1]
    out = crossfade_concat([seg, seg], sr, fade_s=0.01)
    fade = max(int(0.01 * sr), 1)
    join = len(seg) - fade
    rms = float(np.sqrt(np.mean(seg**2) + 1e-12))
    diff = float(abs(out[join] - out[join - 1]))
    assert diff <= rms * 4.0


# ---------------------- load_audio_from_bytes ----------------------

MP3_BASE64 = (
    "SUQzBAAAAAAAI1RTU0UAAAAPAAADTGF2ZjYwLjE2LjEwMAAAAAAAAAAAAAAA/+NIwAAAAAAAAAAA"
    "AEluZm8AAAAPAAAABAAABaAAZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmZmmZmZmZmZmZmZmZmZmZmZ"
    "mZmZmZmZmZmZmczMzMzMzMzMzMzMzMzMzMzMzMzMzMzMzMz/////////////////////////////////"
    "////AAAAAExhdmM2MC4zMQAAAAAAAAAAAAAAACQDoAAAAAAAAAWgpumPwwAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
    "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA/+NIxAA5Yv5w"
    "AVoYAUA6Acu+XLLloB0V0x1jqCJCIBC5BcgvAg4oIsRU6EstmWTLJlx23vJzgIIkaZVecWyenqfv"
    "2fXWdmGakWpqydeBdwvAg4po1yWZyt/3Lct/43b05hAAAAEocAEbuegABgb/XDgYAAIiIgAAAYGL"
    "d3cDAwAAEIiBAAAAwMDd3DgYGAAAAiIEAAADAwMW7hwMDAAAAEIgQAAAMDAxbu4GLAAAAREQIIAw"
    "MW7u7iwAAREREQW7u7u7iEAAAAAw8PDw8AAAAAAw8PDw8AAAAAAw8PDw8AAAAAAw8PDw8AAAAAAw"
    "8PDw8AAFAIDAgCMEkJAwmwoDDDD+MFQK2GYq+0S5vPzC8DNMPwU8wfQ+DElFefp0Y6xH/+NIxC1D"
    "uzZMOZ6oAHbypTHrKeMOwaE+DGIjUHO8UOMIYFA+HVaDIKCEh8wXw3zxysrMdULEDTkzA5pEQN/R"
    "EDI6SA0ujANGpwgRbIqTJ4vFstNUwGJCuBl4tgZaM4GExwBj8fAY/JAGCBOBiwVmTmKLJKfbbAxY"
    "MwMAhoDD4eAw+JACAiBhQKgYUDICQQBg8HAYNCCK/9tvwHAMDAwHAwMCQbpAYDAQGAwEGBQbJBsG"
    "h0QNxBZF/+1tBn8TaFwoYpFeDFQfMMaIKiySHCghxFEXL//tQ+zK5VOEWL6ZdLCKRDS8kOUTJdFm"
    "kOIt/9C0dAlyuUSi0XBqBe9JN1TCQvMSAsGjAwsNdfrJw4HmVM0cRUDAgEhgJ3Ho1MpJmBwG/+NI"
    "xDFJw3qluZzQAm46oYmBwhBhgoZmQiCYqDOsu4usiYvSIM4gQAjEsgoDTme9SmI0lDVwugwAYgUJ"
    "BC4BgACXCEguQaeICm5rqpk0wGFmgdmHJFu8eb3+9+vRFRBxtExFBHkUEWJFDLKQKGQbMgdLrIvm"
    "KKpfJKmGG//////+qRiEgWI1ycYg7l5iD8cYcjKYIO/yRpgAMuRtLM5oYl4f///////0UGWYpEOJ"
    "tMB1M1NIEzU0kFCjyijHUeUwYBS9Vy6KXq1XRS9/////////7a6JNTronZeyyll7LKWXuJVl7iTK"
    "IqaIAApmpGgACrSjaWRX4jaWZZIhNLMsQQEltUxBTUUzLjEwMKqqqqqqqqqqqqqqqqqqqqqqqqqq"
    "/+NIxAAAAANIAcAAAExBTUUzLjEwMKqqqqqqqqqqqqqqTEFNRTMuMTAwqqqqqqqqqqqqqqqqqqqq"
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
    "qqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqqq"
    "qqqq"
)


def _sine_bytes(fmt: str) -> bytes:
    sr = 8000
    t = np.linspace(0, 0.1, int(0.1 * sr), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    buf = io.BytesIO()
    if fmt == "wav":
        sf.write(buf, y, sr, format="WAV")
    elif fmt == "ogg":
        sf.write(buf, y, sr, format="OGG", subtype="VORBIS")
    else:
        raise ValueError(fmt)
    return buf.getvalue()


@pytest.mark.parametrize(
    "data_fn,name",
    [
        (lambda: base64.b64decode(MP3_BASE64), "t.mp3"),
        (lambda: _sine_bytes("ogg"), "t.ogg"),
        (lambda: _sine_bytes("wav"), "t.wav"),
    ],
)
def test_load_audio_from_bytes_formats(data_fn, name):
    data = data_fn()
    try:
        y, sr = load_audio_from_bytes(data, sr=8000, filename=name)
    except audioread.exceptions.NoBackendError:
        pytest.skip("No backend for decoding")
    assert sr == 8000
    assert y.ndim == 1 and y.size > 0
    assert np.isfinite(y).all()
