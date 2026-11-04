"""ARCHIVED 11-03 (AgileX removal): cert/backups push/clamp-flow tests; they were skipped under the NYU gripper (pan push/clamp flow dropped 11-01) and ran only under the removed AgileX build.
Moved unchanged from pan/tests/test_cert_runtime.py except the skipif markers (is_nyu() no longer exists); not collected (archive/conftest.py)."""
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
