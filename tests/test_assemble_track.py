import random
import numpy as np

from beatsmith.audio import assemble_track, TARGET_SR, SourceRef
from beatsmith.db import db_open


def test_assemble_track_respects_num_sounds():
    rng = random.Random(0)
    conn = db_open(":memory:")
    conn.execute(
        "INSERT INTO runs(id,created_at,out_dir,bpm,sig_map,seed,salt,params_json) VALUES (1,'','',120.0,'','','','{}')"
    )
    conn.execute(
        "INSERT INTO sources(id,run_id,ia_identifier,ia_file,url,title,licenseurl,picked,bus,duration_s,zcr,flatness,onset_density) VALUES (1,1,NULL,NULL,'','','',1,'perc',1.0,0.0,0.0,0.0)"
    )
    y = np.ones(int(TARGET_SR * 2), dtype=np.float32)
    src = SourceRef(1, '', None, None, None, None, y, TARGET_SR, 'perc', {})
    measures = [(4, 4)] * 4
    assemble_track(
        conn,
        1,
        [src],
        measures,
        bpm=120.0,
        rng=rng,
        min_rms=0.0,
        crossfade_s=0.0,
        tempo_mode='off',
        stems_dirs={},
        microfill=False,
        beat_align=False,
        refine_boundaries=False,
        num_sounds=10,
    )
    count = conn.execute("SELECT COUNT(*) FROM segments WHERE run_id=1").fetchone()[0]
    assert count == 10
