from pathlib import Path
import importlib.util
import random

import numpy as np
from mido import MidiFile, MidiTrack, Message

from beatsmith.pattern_adapter import apply_pattern_to_quantized_slices

spec = importlib.util.spec_from_file_location(
    "drum_patterns", Path(__file__).resolve().parents[1] / "tools" / "drum_patterns.py"
)
drum_patterns = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drum_patterns)


def _write_test_midi(path: Path) -> None:
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    tpb = mid.ticks_per_beat
    track.append(Message("note_on", note=36, velocity=100, time=0))
    track.append(Message("note_on", note=38, velocity=100, time=2 * tpb))
    mid.save(path)


def test_ingest_and_apply(tmp_path):
    midi_path = tmp_path / "simple.mid"
    _write_test_midi(midi_path)
    db = tmp_path / "patterns.db"
    drum_patterns.ingest(db, [midi_path], subdivision=16)
    pattern = drum_patterns.sample(db)
    assert pattern and pattern["lanes"] == {"kick": [0], "snare": [8]}
    registry = {
        "kick": [np.ones(1, dtype=np.float32)],
        "snare": [np.ones(1, dtype=np.float32)],
    }
    placements = apply_pattern_to_quantized_slices(
        pattern, registry, 120.0, rng=random.Random(0)
    )
    times = [p.start_s for p in placements]
    assert np.allclose(times, [0.0, 1.0])

