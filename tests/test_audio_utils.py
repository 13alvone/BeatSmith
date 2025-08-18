import argparse
import numpy as np
import pytest

from beatsmith.audio import (
    MeasureSpec,
    parse_sig_map,
    crossfade_concat,
    time_stretch_to_length,
)


# ---------------------- parse_sig_map ----------------------

def test_parse_sig_map_valid():
    sig = "4/4(2), 3/8(1)"
    specs = parse_sig_map(sig)
    assert specs == [MeasureSpec(4, 4, 2), MeasureSpec(3, 8, 1)]


def test_parse_sig_map_invalid():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_sig_map("4/5(2)")


# ---------------------- crossfade_concat ----------------------

def test_crossfade_concat_continuity():
    sr = 1000
    a = np.ones(sr, dtype=np.float32)
    b = np.ones(sr, dtype=np.float32)
    out = crossfade_concat([a, b], sr, fade_s=0.1)
    fade = int(0.1 * sr)
    assert len(out) == sr * 2 - fade
    assert np.allclose(out, 1.0, atol=1e-6)


# ---------------------- time_stretch_to_length ----------------------

@pytest.mark.parametrize("mode", ["off", "loose", "strict"])
@pytest.mark.parametrize("target", [50, 150])
def test_time_stretch_to_length_length(mode, target):
    seg = np.linspace(0, 1, 100).astype(np.float32)
    y, _ = time_stretch_to_length(seg, sr=100, target_len=target, mode=mode)
    assert len(y) == target
