import io
import hashlib

import numpy as np
import soundfile as sf

from beatsmith.audio import TARGET_SR, seeded_rng


def render_fixture(seed: str, salt: str) -> bytes:
    rng_py = seeded_rng(seed, salt)
    rng_np = np.random.default_rng(rng_py.randrange(2**32))
    y = rng_np.standard_normal(TARGET_SR * 10).astype("float32")
    buf = io.BytesIO()
    sf.write(buf, y, TARGET_SR, format="wav")
    return buf.getvalue()


def test_audio_snapshot():
    data = render_fixture("snapshot", "fixed")
    h = hashlib.sha256(data).hexdigest()
    assert h == "bfc6745a66f379dc51bacfcc0998225047bda3dc43e1aee75bb152fa99c872f2"
