"""cert/backups.py + cert/runtime.py: stored backups have no online choices and never drag the pan; the gate refuses."""
import pytest

from world_model.v2.nero.pan import segments as S
from world_model.v2.nero.pan.cert import backups as BK
from world_model.v2.nero.pan.cert import runtime as RT

NOMINAL = dict(placement="none", hidden_mass=None, perturb=dict(dx=0.0, dy=0.0, yaw_deg=0.0, mu_scale=1.0))


@pytest.fixture(scope="module")
def ep():
    e = S.Episode(NOMINAL, observe=None, **RT.PLAN)
    assert not e.refused
    return e


def test_stored_backups_are_fixed(ep):
    for seg in S.SEGMENTS:
        bk = BK.make(ep, seg, 0.03)
        assert not BK.has_online_choice(bk), seg
        kinds = [p[0] for p in bk.phases]
        assert kinds[0] == "stop" and ("retreat_fixed", "left", 0.03) in bk.phases
        # no backup drags the pan back to the push start (the v1 undo that popped the lid): the only left-claw move_to is the
        # lift backup's vertical lowering to lift_start (its first motion after the stop)
        left_moves = [i for i, p in enumerate(bk.phases) if p[0] == "move_to" and "left" in p[1]]
        assert left_moves == ([1] if seg == "lift" else []), seg


def test_v1_backup_flagged(ep):
    v1 = BK.Backup(seg="pinch", retreat=0.0, phases=tuple(ep.backup_phases("pinch")))
    assert BK.has_online_choice(v1)


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
