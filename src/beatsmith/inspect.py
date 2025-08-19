from typing import Optional
import argparse
from collections import defaultdict

from . import li, le
from .db import db_open, find_latest_db, read_last_run


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize latest BeatSmith run.")
    p.add_argument("--root", type=str, default=".", help="Search root for beatsmith_v3.db")
    p.add_argument("--db", type=str, default=None, help="Explicit path to beatsmith_v3.db")
    return p


def main(argv: Optional[list[str]] = None):
    args = build_parser().parse_args(argv)
    db_path = args.db or find_latest_db(args.root)
    if not db_path:
        le("No beatsmith_v3.db found")
        return
    conn = db_open(db_path)
    run = read_last_run(conn)
    if not run:
        le("No runs recorded in DB")
        return
    li(f"DB: {db_path}")
    li(
        f"Run id={run['id']} created={run['created_at']} BPM={run['bpm']} "
        f"sig_map={run['sig_map']} seed={run['seed']} salt={run['salt']}"
    )
    if run["params"]:
        li("Params:")
        for k in sorted(run["params"].keys()):
            li(f"  {k}: {run['params'][k]}")
    li(f"Sources: {run['num_sources']}  segments: {run['num_segments']}")

    cur = conn.execute(
        """
        SELECT s.title, s.bus, s.licenseurl,
               COUNT(seg.id) AS ct,
               COALESCE(SUM(seg.dur_s), 0.0) AS dur
        FROM sources s
        LEFT JOIN segments seg ON seg.source_id = s.id
        WHERE s.run_id = ?
        GROUP BY s.id
        ORDER BY s.bus, s.title
        """,
        (run["id"],),
    )
    rows = cur.fetchall()
    if rows:
        li("Source usage:")
        for title, bus, lic, ct, dur in rows:
            lic = lic or "-"
            li(f"  {bus:<4} {ct:>3}x {dur:>6.2f}s {title} {lic}")

    cur = conn.execute(
        """
        SELECT seg.measure_index, seg.bus, s.title
        FROM segments seg
        JOIN sources s ON seg.source_id = s.id
        WHERE seg.run_id = ?
        ORDER BY seg.measure_index, seg.bus
        """,
        (run["id"],),
    )
    rows = cur.fetchall()
    if rows:
        li("Segments:")
        by_measure: dict[int, list[str]] = defaultdict(list)
        for m, bus, title in rows:
            by_measure[m].append(f"{bus}:{title}")
        for m in sorted(by_measure):
            li(f"  {m:03d}: " + ", ".join(by_measure[m]))


__all__ = ["main"]
