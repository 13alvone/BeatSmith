import json
import sqlite3
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from . import le

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY,
  created_at TEXT NOT NULL,
  out_dir TEXT NOT NULL,
  bpm REAL NOT NULL,
  sig_map TEXT NOT NULL,
  seed TEXT,
  salt TEXT,
  params_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sources(
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  ia_identifier TEXT,
  ia_file TEXT,
  url TEXT NOT NULL,
  title TEXT,
  licenseurl TEXT,
  picked INTEGER NOT NULL DEFAULT 0,
  bus TEXT NOT NULL,            -- 'perc' or 'tex'
  duration_s REAL NOT NULL,
  zcr REAL,
  flatness REAL,
  onset_density REAL,
  FOREIGN KEY(run_id) REFERENCES runs(id)
);
CREATE TABLE IF NOT EXISTS segments(
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL,
  measure_index INTEGER NOT NULL,
  numer INTEGER NOT NULL,
  denom INTEGER NOT NULL,
  bus TEXT NOT NULL,            -- 'perc' or 'tex'
  start_s REAL NOT NULL,
  dur_s REAL NOT NULL,
  source_id INTEGER NOT NULL,
  energy REAL,
  tempo_factor REAL,
  FOREIGN KEY(run_id) REFERENCES runs(id),
  FOREIGN KEY(source_id) REFERENCES sources(id)
);
"""

def db_open(path: str) -> sqlite3.Connection:
    try:
        conn = sqlite3.connect(path)
        conn.execute("PRAGMA foreign_keys=ON;")
        conn.executescript(SCHEMA)
        return conn
    except Exception as e:
        le(f"DB open/init failed: {e}")
        sys.exit(1)


def find_latest_db(root: str = ".") -> Optional[str]:
    """Locate the most recently modified beatsmith_v3.db under root."""
    paths = list(Path(root).rglob("beatsmith_v3.db"))
    if not paths:
        return None
    latest = max(paths, key=lambda p: p.stat().st_mtime)
    return str(latest)


def read_last_run(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    """Return info for the most recent run or None."""
    cur = conn.execute(
        "SELECT id,created_at,out_dir,bpm,sig_map,seed,salt,params_json FROM runs ORDER BY id DESC LIMIT 1"
    )
    row = cur.fetchone()
    if not row:
        return None
    run: Dict[str, Any] = {
        "id": row[0],
        "created_at": row[1],
        "out_dir": row[2],
        "bpm": row[3],
        "sig_map": row[4],
        "seed": row[5],
        "salt": row[6],
        "params": json.loads(row[7]) if row[7] else {},
    }
    run_id = run["id"]
    run["num_sources"] = conn.execute(
        "SELECT COUNT(*) FROM sources WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    run["num_segments"] = conn.execute(
        "SELECT COUNT(*) FROM segments WHERE run_id=?", (run_id,)
    ).fetchone()[0]
    return run
