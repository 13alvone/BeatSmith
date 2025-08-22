import random

import numpy as np

from beatsmith.pattern_adapter import apply_pattern_to_quantized_slices


def test_step_to_time_conversion():
    pattern = {
        "signature": "4/4",
        "bars": 1,
        "subdivision": 16,
        "lanes": {"kick": [0, 1, 2]},
    }
    sample = np.ones(10, dtype=np.float32)
    out = apply_pattern_to_quantized_slices(pattern, {"kick": [sample]}, 120.0)
    times = [p.start_s for p in out]
    assert np.allclose(times, [0.0, 0.125, 0.25])


def test_swing_offsets():
    pattern = {"signature": "4/4", "bars": 1, "subdivision": 16, "lanes": {"kick": [0, 1]}}
    sample = np.ones(10, dtype=np.float32)
    out = apply_pattern_to_quantized_slices(pattern, {"kick": [sample]}, 120.0, swing=0.5)
    times = [p.start_s for p in out]
    assert np.allclose(times, [0.0, 0.1875])


def test_humanize_jitter():
    pattern = {"signature": "4/4", "bars": 1, "subdivision": 16, "lanes": {"kick": [0]}}
    sample = np.ones(10, dtype=np.float32)
    rng = random.Random(0)
    out = apply_pattern_to_quantized_slices(
        pattern, {"kick": [sample]}, 120.0, humanize_s=0.01, rng=rng
    )
    assert len(out) == 1
    assert abs(out[0].start_s - 0.006888437) < 1e-6


def test_velocity_scaling():
    pattern = {"signature": "4/4", "bars": 1, "subdivision": 16, "lanes": {"snare": [(0, 0.5)]}}
    sample = np.ones(4, dtype=np.float32)
    out = apply_pattern_to_quantized_slices(
        pattern, {"snare": [sample]}, 120.0, velocity_scale=0.2
    )
    assert np.allclose(out[0].data, 0.1)
