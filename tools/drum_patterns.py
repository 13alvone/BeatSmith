import argparse
import json
import logging
import math
import sqlite3
from pathlib import Path

from mido import MidiFile, merge_tracks

from beatsmith.pattern_constants import GM_NOTE_TO_LANE, LANES


class PrefixFormatter(logging.Formatter):
    PREFIX = {
        logging.INFO: "[i]",
        logging.WARNING: "[!]",
        logging.DEBUG: "[DEBUG]",
        logging.ERROR: "[x]",
        logging.CRITICAL: "[x]",
    }

    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        prefix = self.PREFIX.get(record.levelno, "[i]")
        record.msg = f"{prefix} {record.getMessage()}"
        return super().format(record)


def get_logger(level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger("drum_patterns")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(PrefixFormatter("%(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = get_logger()


def parse_midi(path: Path, subdivision: int) -> dict:
    """Parse a MIDI file into a normalized drum pattern."""
    mid = MidiFile(path)
    ticks_per_beat = mid.ticks_per_beat
    track = merge_tracks(mid.tracks)
    tempo = 500000  # default 120 BPM
    time_sig = (4, 4)
    current_ticks = 0
    lanes: dict[str, list[int]] = {lane: [] for lane in LANES}

    for msg in track:
        current_ticks += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
        elif msg.type == "time_signature":
            time_sig = (msg.numerator, msg.denominator)
        elif msg.type == "note_on" and msg.velocity > 0:
            lane = GM_NOTE_TO_LANE.get(msg.note)
            if lane is None:
                continue
            beats = current_ticks / ticks_per_beat
            beats_per_bar = time_sig[0] * 4 / time_sig[1]
            step = int(round(beats / beats_per_bar * subdivision))
            lanes[lane].append(step)

    total_beats = current_ticks / ticks_per_beat
    beats_per_bar = time_sig[0] * 4 / time_sig[1]
    bars = int(math.ceil(total_beats / beats_per_bar))
    bpm = 60_000_000 / tempo
    for hits in lanes.values():
        hits.sort()
    return {
        "signature": f"{time_sig[0]}/{time_sig[1]}",
        "bars": bars,
        "subdivision": subdivision,
        "bpm": bpm,
        "lanes": {k: v for k, v in lanes.items() if v},
    }


def ingest(db_path: Path, midi_files: list[Path], subdivision: int) -> None:
    logger.info(f"Ingesting {len(midi_files)} MIDI file(s) into {db_path}")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                signature TEXT NOT NULL,
                bars INTEGER NOT NULL,
                subdivision INTEGER NOT NULL,
                bpm REAL NOT NULL,
                data TEXT NOT NULL
            )
            """
        )
        for midi_path in midi_files:
            try:
                pattern = parse_midi(midi_path, subdivision)
            except Exception as exc:  # pragma: no cover
                logger.error(f"Failed to parse {midi_path}: {exc}")
                continue
            conn.execute(
                "INSERT INTO patterns(signature,bars,subdivision,bpm,data) VALUES(?,?,?,?,?)",
                (
                    pattern["signature"],
                    pattern["bars"],
                    pattern["subdivision"],
                    pattern["bpm"],
                    json.dumps(pattern["lanes"]),
                ),
            )
    conn.close()


def sample(
    db_path: Path,
    signature: str | None = None,
    bars: int | None = None,
    subdivision: int | None = None,
    min_bpm: float | None = None,
    max_bpm: float | None = None,
) -> dict | None:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    query = "SELECT signature,bars,subdivision,bpm,data FROM patterns WHERE 1=1"
    params: list = []
    if signature:
        query += " AND signature = ?"
        params.append(signature)
    if bars is not None:
        query += " AND bars = ?"
        params.append(bars)
    if subdivision is not None:
        query += " AND subdivision = ?"
        params.append(subdivision)
    if min_bpm is not None:
        query += " AND bpm >= ?"
        params.append(min_bpm)
    if max_bpm is not None:
        query += " AND bpm <= ?"
        params.append(max_bpm)
    query += " ORDER BY RANDOM() LIMIT 1"
    cur = conn.execute(query, params)
    row = cur.fetchone()
    conn.close()
    if row is None:
        return None
    lanes = json.loads(row["data"])
    return {
        "signature": row["signature"],
        "bars": row["bars"],
        "subdivision": row["subdivision"],
        "bpm": row["bpm"],
        "lanes": lanes,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Drum pattern tools")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="Ingest MIDI files into the pattern database")
    p_ingest.add_argument("db", type=Path, help="SQLite database path")
    p_ingest.add_argument("midis", nargs="+", type=Path, help="MIDI files to ingest")
    p_ingest.add_argument(
        "--subdivision",
        type=int,
        default=16,
        help="Steps per bar to quantize hits",
    )

    p_sample = sub.add_parser("sample", help="Sample a random pattern from the database")
    p_sample.add_argument("db", type=Path, help="SQLite database path")
    p_sample.add_argument("--signature", help="Time signature filter, e.g. 4/4")
    p_sample.add_argument("--bars", type=int, help="Number of bars")
    p_sample.add_argument("--subdivision", type=int, help="Subdivision per bar")
    p_sample.add_argument("--min-bpm", type=float, help="Minimum BPM")
    p_sample.add_argument("--max-bpm", type=float, help="Maximum BPM")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ingest":
        ingest(args.db, args.midis, args.subdivision)
    elif args.command == "sample":
        result = sample(
            args.db,
            signature=args.signature,
            bars=args.bars,
            subdivision=args.subdivision,
            min_bpm=args.min_bpm,
            max_bpm=args.max_bpm,
        )
        if result is None:
            logger.warning("No pattern matched the query")
        else:
            print(json.dumps(result))


if __name__ == "__main__":
    main()
