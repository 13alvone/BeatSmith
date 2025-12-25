from beatsmith.audio import duration_bucket


def test_duration_bucket_quarter_note():
    assert duration_bucket(0.5, bpm_assumption=120.0, tolerance=0.12) == "q"


def test_duration_bucket_free():
    assert duration_bucket(0.7, bpm_assumption=120.0, tolerance=0.05) == "free"


def test_duration_bucket_longform():
    assert duration_bucket(10.0, bpm_assumption=120.0, tolerance=0.12, longform=True) == "lf"
