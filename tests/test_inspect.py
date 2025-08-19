import logging

from beatsmith.db import db_open
from beatsmith import inspect


def _seed_db(conn):
    conn.execute(
        "INSERT INTO runs(created_at,out_dir,bpm,sig_map,seed,salt,params_json) VALUES (datetime('now'),?,?,?,?,?,?)",
        ("out", 120.0, "4/4(1)", "seed", "salt", "{}"),
    )
    run_id = conn.execute("SELECT id FROM runs").fetchone()[0]
    conn.execute(
        "INSERT INTO sources(run_id,url,title,licenseurl,bus,duration_s,picked) VALUES (?,?,?,?,?,?,1)",
        (run_id, "u1", "src1", "lic1", "perc", 1.0),
    )
    conn.execute(
        "INSERT INTO sources(run_id,url,title,licenseurl,bus,duration_s,picked) VALUES (?,?,?,?,?,?,1)",
        (run_id, "u2", "src2", "lic2", "tex", 2.0),
    )
    conn.execute(
        "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id) VALUES (?,?,?,?,?,?,?,1)",
        (run_id, 0, 4, 4, "perc", 0.0, 1.0),
    )
    conn.execute(
        "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id) VALUES (?,?,?,?,?,?,?,2)",
        (run_id, 1, 4, 4, "tex", 1.0, 2.0),
    )
    conn.commit()
    return run_id


def test_inspect_reports_sources_and_segments(tmp_path, caplog):
    db_path = tmp_path / "beatsmith_v3.db"
    conn = db_open(str(db_path))
    _seed_db(conn)

    with caplog.at_level(logging.INFO):
        inspect.main(["--db", str(db_path)])
    out = caplog.text

    assert "lic1" in out and "lic2" in out
    assert "000: perc:src1" in out
    assert "001: tex:src2" in out

