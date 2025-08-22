"""Shared constants for normalizing drum patterns.

The mapping groups related General MIDI drum notes under a single
canonical *lane* name (e.g. all snare variants map to ``"snare"``).
Grouping in this way keeps downstream code simple and consistent.
"""

from __future__ import annotations

from typing import Dict, List

# Map each normalized lane to the General MIDI note numbers that trigger it.
# Notes are grouped by their musical role rather than strict GM categories.
LANE_TO_GM_NOTES: Dict[str, List[int]] = {
    "kick": [35, 36],
    "snare": [38, 40, 37, 39],
    "hh_closed": [42, 44],
    "hh_open": [46],
    "tom_low": [41, 43],
    "tom_mid": [45, 47],
    "tom_high": [48, 50],
    "crash": [49, 57, 55],
    "ride": [51, 53, 59, 52],
}

# Reverse lookup mapping of GM note numbers to lane names.
GM_NOTE_TO_LANE: Dict[int, str] = {
    note: lane for lane, notes in LANE_TO_GM_NOTES.items() for note in notes
}

# Normalized list of lane names encountered above.
LANES: List[str] = sorted(LANE_TO_GM_NOTES)

# Default lane remapping used if a caller does not supply their own mapping.
# It simply maps each lane to itself but allows a single point of customization.
DEFAULT_LANE_MAP: Dict[str, str] = {lane: lane for lane in LANES}
