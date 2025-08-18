import time

from beatsmith.db import db_open, find_latest_db, read_last_run


def test_read_last_run(tmp_path):
    db_path = tmp_path / "beatsmith_v3.db"
    conn = db_open(str(db_path))
    conn.execute(
        "INSERT INTO runs(created_at,out_dir,bpm,sig_map,seed,salt,params_json) VALUES (datetime('now'),?,?,?,?,?,?)",
        ("out", 120.0, "4/4(1)", "seed", "salt", '{"preset":"boom"}')
    )
    run_id = conn.execute("SELECT id FROM runs").fetchone()[0]
    conn.execute(
        "INSERT INTO sources(run_id,url,bus,duration_s,picked) VALUES (?,?,?,?,1)",
        (run_id, "u", "perc", 1.0)
    )
    conn.execute(
        "INSERT INTO segments(run_id,measure_index,numer,denom,bus,start_s,dur_s,source_id) VALUES (?,?,?,?,?,?,?,1)",
        (run_id, 0, 4, 4, "perc", 0.0, 1.0)
    )
    conn.commit()
    info = read_last_run(conn)
    assert info and info["id"] == run_id
    assert info["num_sources"] == 1
    assert info["num_segments"] == 1
    assert info["params"]["preset"] == "boom"


def test_find_latest_db(tmp_path):
    older = tmp_path / "a" / "beatsmith_v3.db"
    newer = tmp_path / "b" / "beatsmith_v3.db"
    older.parent.mkdir()
    newer.parent.mkdir()
    older.touch()
    time.sleep(0.1)
    newer.touch()
    found = find_latest_db(str(tmp_path))
    assert found and found.endswith(str(newer))
