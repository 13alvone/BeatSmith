import sqlite3
import sys

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
