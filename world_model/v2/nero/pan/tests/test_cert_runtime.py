"""cert/backups.py + cert/runtime.py: stored backups have no online choices and never drag the pan; the gate refuses."""
from world_model.v2.nero.pan import segments as S
from world_model.v2.nero.pan.cert import runtime as RT

NOMINAL = dict(placement="none", hidden_mass=None, perturb=dict(dx=0.0, dy=0.0, yaw_deg=0.0, mu_scale=1.0))


def _cert(ok):
    cell = dict(est_lo=(-1, -1, -90), est_hi=(1, 1, 90))
    return dict(cell=cell, segments={s: (dict(ok=True, backup_retreat=0.03) if ok.get(s, True) else dict(ok=False, best_retreat=0.03))
                                     for s in S.SEGMENTS})


def test_gate_refuses_at_B0_and_leaves_pan_untouched():
    r = RT.run_task(NOMINAL, "T0", _cert(dict(pinch=False)))
    assert r["outcome"].startswith("refused before pinch") and r["end_inR"] and r["U_entered"] is None


def test_gate_refusal_mid_task_runs_previous_backup():
    r = RT.run_task(NOMINAL, "T1", _cert(dict(push=False)))
    assert r["outcome"] == "refused before push (not certified); backup of pinch run"
    assert r["steps"][0]["seg"] == "pinch" and r["steps"][-1]["gate"] == "refused"
    assert r["U_entered"] is None and r["end_inR"], r["end_R_fail"]
