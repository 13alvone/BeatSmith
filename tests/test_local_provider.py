import random
from pathlib import Path

from beatsmith.providers.local import LocalProvider


def _touch(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"data")


def test_query_filtering(tmp_path):
    _touch(tmp_path / "cat.wav")
    _touch(tmp_path / "dog.mp3")
    _touch(tmp_path / "bird.wav")

    provider = LocalProvider(str(tmp_path))
    rng = random.Random(0)
    results = provider.search(rng, wanted=10, query_bias="cat dog", allow_tokens=[], strict=False)
    names = sorted(f["name"] for f in results)
    assert names == ["cat.wav", "dog.mp3"]


def test_strict_license_enforcement(tmp_path):
    allow = tmp_path / "allow.wav"
    _touch(allow)
    (allow.with_suffix(".wav.license")).write_text("CC0 1.0")

    deny = tmp_path / "deny.wav"
    _touch(deny)
    (deny.with_suffix(".wav.license")).write_text("GPL")

    no_license = tmp_path / "nolicense.wav"
    _touch(no_license)

    provider = LocalProvider(str(tmp_path))
    rng = random.Random(0)
    results = provider.search(rng, wanted=10, query_bias=None, allow_tokens=["cc0"], strict=True)
    names = sorted(f["name"] for f in results)
    assert names == ["allow.wav"]
    assert results[0]["licenseurl"] == "CC0 1.0"
