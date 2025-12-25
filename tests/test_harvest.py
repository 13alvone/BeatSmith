import json
from pathlib import Path

import numpy as np
import soundfile as sf

from beatsmith import cli


def test_harvest_writes_pack(tmp_path, monkeypatch):
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    tone = (0.4 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    stereo = np.stack([tone, tone], axis=1)
    src = tmp_path / "src.wav"
    sf.write(src, stereo, sr)

    out_dir = tmp_path / "out"
    monkeypatch.chdir(tmp_path)
    cli.main([
        str(out_dir),
        "--provider",
        "local",
        "--c",
        "1",
        "--max-samples",
        "1",
        "--form-modes",
        "oneshot",
        "--oneshot-seconds",
        "0.1-0.2",
        "--min-rms",
        "0.0",
    ])

    pack = out_dir / "pack.json"
    credits = out_dir / "credits.csv"
    assert pack.exists()
    assert credits.exists()

    data = json.loads(pack.read_text())
    assert len(data["samples"]) == 1
    exports = data["samples"][0]["exports"]
    assert "stereo" in exports and "mono" in exports

    stereo_path = out_dir / exports["stereo"]
    mono_path = out_dir / exports["mono"]
    assert stereo_path.exists()
    assert mono_path.exists()

    for expected in ("samples/clean/oneshot/stereo", "samples/clean/oneshot/mono"):
        assert (out_dir / expected).exists()
