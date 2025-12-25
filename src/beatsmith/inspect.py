from typing import Optional
import argparse
import json
from pathlib import Path

from . import li, le


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Summarize latest SampleSmith pack.")
    p.add_argument("--root", type=str, default=".", help="Search root for pack.json")
    p.add_argument("--pack", type=str, default=None, help="Explicit path to pack.json")
    return p


def find_latest_pack(root: str) -> Optional[str]:
    paths = list(Path(root).rglob("pack.json"))
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return str(latest)


def main(argv: Optional[list[str]] = None):
    args = build_parser().parse_args(argv)
    pack_path = args.pack or find_latest_pack(args.root)
    if not pack_path:
        le("No pack.json found")
        return
    with open(pack_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    li(f"Pack: {pack_path}")
    li(f"Run id={data.get('run_id')} seed={data.get('seed')} salt={data.get('salt')}")
    li(f"Samples: {len(data.get('samples', []))} variants: {len(data.get('variants', []))}")

    sources = data.get("sources", [])
    if sources:
        li("Sources:")
        for src in sources:
            title = src.get("title") or src.get("file") or src.get("identifier") or "unknown"
            li(f"  {title} {src.get('license') or '-'}")


__all__ = ["main"]
