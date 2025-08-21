import numpy as np

from beatsmith.audio import duration_samples_map, stack_slices, TARGET_SR

def test_duration_samples_map_quantization():
    bpm = 120.0
    mapping = duration_samples_map(bpm)
    unit = mapping["1/16"]
    # quarter note at 120 BPM is 0.5s -> ≈22050 samples (allow 2-sample rounding)
    assert abs(mapping["1/4"] - int(0.5 * TARGET_SR)) <= 2
    # dotted eighth is 1.5 times an eighth
    assert mapping["dotted 1/8"] == int(round(1.5 * mapping["1/8"]))
    for name, samples in mapping.items():
        if "dotted" not in name:
            assert samples % unit == 0

def test_stack_slices_overlap():
    length = 16
    a = np.ones(4, dtype=np.float32)
    b = np.full(4, 0.5, dtype=np.float32)
    out = stack_slices(length, [(a, 0), (b, 2)])
    expected = np.zeros(length, dtype=np.float32)
    expected[0:4] += a
    expected[2:6] += b
    assert np.allclose(out, expected)
