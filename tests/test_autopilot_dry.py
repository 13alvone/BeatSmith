import random

from beatsmith.cli import autopilot_config


def test_autopilot_usually_dry():
    rng = random.Random(0)
    wet_reverb = 0
    wet_trem = 0
    N = 100
    for _ in range(N):
        cfg = autopilot_config(rng)
        if cfg["reverb_mix"] > 0:
            wet_reverb += 1
        else:
            assert cfg["reverb_room"] == 0
        if cfg["tremolo_rate"] > 0:
            wet_trem += 1
        else:
            assert cfg["tremolo_depth"] == 0
    assert wet_reverb <= N * 0.2
    assert wet_trem <= N * 0.2


def test_autopilot_force_effects():
    rng = random.Random(123)
    cfg = autopilot_config(rng, force_reverb=True, force_tremolo=True)
    assert cfg["reverb_mix"] > 0 and cfg["reverb_room"] > 0
    assert cfg["tremolo_rate"] > 0 and cfg["tremolo_depth"] > 0
