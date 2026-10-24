"""the default scene / task are unchanged by the opt-in pick option (builds the h5 rig on CPU; ~20 s)."""
import inspect

import numpy as np

from world_model.v2.house.teleop import scene as SC, sim as S


def test_signatures_default_off():
    sig = inspect.signature(S.TeleopSim.__init__).parameters
    assert sig["task"].default is None and sig["pick_props"].default is None
    assert inspect.signature(SC.build_rig).parameters["pick_props"].default is None
    assert inspect.signature(SC.new_rig).parameters["pick_props"].default is None


def test_default_rig_has_no_pick_objects_and_same_counts():
    sim = S.TeleopSim("h5", grasped=True, with_people=False)
    m = sim.m
    names = [m.body(i).name or "" for i in range(m.nbody)]
    assert not any(n.startswith(("pp_", "pickt_")) for n in names)
    assert all(m.camera(i).name != "pick_head" for i in range(m.ncam))
    assert (m.nbody, m.ngeom, m.nmocap) == (239, 1445, 1)          # measured before the pick option existed (same as the origin default scene)
    assert sim.task is None and sim.grasped and "pick_props" not in sim.rig.info
    # the default carry task still holds the tray: weld active, arms at the original hold pose
    assert sim.d.eq_active[sim.eq["left"]] == 1 and sim.d.eq_active[sim.eq["right"]] == 1
    assert np.allclose(sim.ctl.q_hold[0], [-0.522, 2.159, -0.947, -0.567, -1.188, -0.551], atol=1e-3)


def test_pick_task_is_additive():
    sim = S.TeleopSim("h5", with_people=False, task="pick", pick_props="single")
    m = sim.m
    assert sim.task == "pick" and not sim.grasped
    assert m.body("pp_bottle").id > 0 and m.body("pickt_table").id > 0
    assert sim.d.eq_active[sim.eq["left"]] == 0                       # nothing welded: free arms
    from world_model.v2.house.pick import props as PP
    assert np.allclose(sim.ctl.q_hold[0], PP.READY_Q["left"])
