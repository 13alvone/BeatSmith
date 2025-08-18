from typing import Optional
import argparse

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


__all__ = ["main"]
