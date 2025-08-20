import numpy as np
from beatsmith.fx import reverb_schroeder

def test_reverb_schroeder_finite_and_non_silent():
    sr = 44100
    rng = np.random.default_rng(0)
    y = rng.standard_normal(sr).astype(np.float32)
    out = reverb_schroeder(y, sr)
    assert np.all(np.isfinite(out))
    assert np.any(np.abs(out) > 1e-7)
