"""Utilities for adapting drum patterns to quantized sample slices.

This module provides :func:`apply_pattern_to_quantized_slices` which maps a
quantized drum pattern onto available audio slices and returns placements that
can be consumed by existing mixing utilities (e.g. :func:`stack_slices`).

The implementation purposely keeps the data model simple – a placed slice is
represented by the :class:`PlacedSlice` dataclass containing the audio buffer and
its start time (in seconds).  Mixing code can convert ``start_s`` to a sample
index using the desired sample rate.

The function handles several musical niceties:

* General MIDI note numbers are mapped to normalized lane names via
  :data:`GM_NOTE_TO_LANE`.
* A default lane map (:data:`DEFAULT_LANE_MAP`) is provided so callers can
  easily remap pattern lane names to their own sample registry keys.
* Step indices are converted to seconds using the project's BPM and the
  pattern's subdivision and time signature.
* Optional swing and humanization offsets can be applied to each step to add
  groove and timing variance.
* Velocities can be scaled and applied directly to the returned audio
  buffers.

The goal is to stay lightweight while being flexible enough for tests and
future features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence
import random

import numpy as np

from .pattern_constants import GM_NOTE_TO_LANE, LANES, DEFAULT_LANE_MAP  # noqa: F401


# ---------------------------------------------------------------------------
# Dataclass used by ``apply_pattern_to_quantized_slices``
# ---------------------------------------------------------------------------

@dataclass
class PlacedSlice:
    """Represents an audio slice placed on a timeline.

    Parameters
    ----------
    data
        The audio buffer for this slice.  The caller is expected to provide
        mono ``float32`` arrays but the type is not strictly enforced.
    start_s
        Start position in **seconds**.
    """

    data: np.ndarray
    start_s: float


# ---------------------------------------------------------------------------
# Core functionality
# ---------------------------------------------------------------------------


def _step_to_seconds(step: int, step_dur_s: float, swing: float) -> float:
    """Convert a step index to seconds applying swing if requested."""

    offset = 0.0
    if swing and step % 2 == 1:
        offset = swing * step_dur_s
    return step * step_dur_s + offset


def apply_pattern_to_quantized_slices(
    pattern: Mapping[str, object],
    sample_registry: Mapping[str, Sequence[np.ndarray]],
    bpm: float,
    *,
    swing: float = 0.0,
    humanize_s: float = 0.0,
    velocity_scale: float = 1.0,
    lane_map: Optional[Mapping[str, str]] = None,
    rng: Optional[random.Random] = None,
) -> List[PlacedSlice]:
    """Map a quantized drum ``pattern`` onto available audio ``sample_registry``.

    Parameters
    ----------
    pattern
        Mapping describing the drum pattern.  The expected structure matches
        the output of :mod:`tools.drum_patterns`, namely::

            {
                "signature": "4/4",
                "bars": 2,
                "subdivision": 16,
                "lanes": {"kick": [0, 8, ...], "snare": [4, ...]}
            }

        Step lists may optionally contain ``(step, velocity)`` tuples where
        ``velocity`` is normalized ``0.0``–``1.0``.
    sample_registry
        Mapping from lane name to a sequence of audio slices.  A slice is a
        NumPy array containing the audio data.
    bpm
        The target BPM for the project; this controls the conversion from step
        indices to seconds.
    swing
        Amount of swing to apply to odd-numbered steps, expressed as a fraction
        of the step duration.  For example, ``swing=0.5`` delays every second
        step by 50% of ``step_dur_s``.
    humanize_s
        Maximum absolute random offset (in seconds) added to each slice's start
        time.
    velocity_scale
        Global scaling applied to per-hit velocities.
    lane_map
        Optional mapping from pattern lane names to sample registry keys.
    rng
        Optional random number generator used for selecting samples and
        applying humanization.  If ``None`` a new :class:`random.Random`
        instance is created.

    Returns
    -------
    list[PlacedSlice]
        Ordered list of slice placements ready for mixing.
    """

    if rng is None:
        rng = random.Random()

    lane_map = dict(DEFAULT_LANE_MAP if lane_map is None else lane_map)

    signature = pattern.get("signature", "4/4")
    try:
        numer, denom = [int(x) for x in str(signature).split("/")]
    except Exception:  # pragma: no cover - defensive, but unlikely in tests
        numer, denom = 4, 4
    subdivision = int(pattern.get("subdivision", 16))
    beats_per_bar = numer * 4.0 / float(denom)
    step_dur_s = (60.0 / max(bpm, 1e-6)) * (beats_per_bar / subdivision)

    lanes: Mapping[str, Sequence] = pattern.get("lanes", {})  # type: ignore

    out: List[PlacedSlice] = []
    for lane, hits in lanes.items():
        mapped_lane = lane_map.get(lane, lane)
        samples = sample_registry.get(mapped_lane)
        if not samples:
            continue  # nothing to place for this lane
        for hit in hits:
            if isinstance(hit, (list, tuple)) and len(hit) >= 2:
                step = int(hit[0])
                vel = float(hit[1])
            else:
                step = int(hit)
                vel = 1.0

            start_s = _step_to_seconds(step, step_dur_s, swing)
            if humanize_s:
                start_s += rng.uniform(-humanize_s, humanize_s)

            sample = rng.choice(samples)
            # Apply velocity scaling directly to the audio buffer
            amp = vel * velocity_scale
            data = sample.astype(np.float32) * float(amp)
            out.append(PlacedSlice(data=data, start_s=start_s))

    out.sort(key=lambda p: p.start_s)
    return out
