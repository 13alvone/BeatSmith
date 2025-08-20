import io

import numpy as np
import soundfile as sf

from beatsmith.audio import normalize_peak, safe_audio, TARGET_SR


def test_safe_audio_removes_nans_before_write():
    y = np.array([0.25, -0.25, 0.5, -0.5], dtype=np.float32)
    y = normalize_peak(y)
    y[2] = np.nan  # effect introduces NaN
    buf = io.BytesIO()
    sf.write(buf, safe_audio(y), TARGET_SR, format="WAV")
    buf.seek(0)
    out, sr = sf.read(buf, dtype="float32")
    assert np.isfinite(out).all()
    assert np.max(np.abs(out)) > 0.0
