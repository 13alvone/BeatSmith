import json
import sys

import numpy as np
import soundfile as sf

from beatsmith import cli


def test_cli_writes_json_and_stems(tmp_path, monkeypatch):
    # create simple audio sources for the local provider
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    for i in range(2):
        sf.write(tmp_path / f"src{i}.wav", y, sr)

    out_dir = tmp_path / "out"
    argv = [
        "bs",
        str(out_dir),
        "4/4(1)",
        "--provider",
        "local",
        "--num-sources",
        "2",
        "--num-sounds",
        "1",
        "--tempo-fit",
        "off",
        "--no-boundary-refine",
    ]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", argv)
    cli.main()

    wavs = list(out_dir.glob("*.wav"))
    jsons = list(out_dir.glob("*.json"))
    zips = list(out_dir.glob("*.zip"))
    assert len(wavs) == 1
    assert len(jsons) == 1
    assert len(zips) == 1

    with open(jsons[0], "r", encoding="utf-8") as jf:
        data = json.load(jf)
    assert data["stems_zip"] == zips[0].name
    assert isinstance(data.get("sources"), list) and data["sources"]
