"""gpusense/fixture.py -- build the equivalence fixture: the h5 teleop model (.mjb) + 8 sim states (6 s .. 24 s of a carry episode) with the
CPU ray-cast reference frames (head + both wrist cameras) on exactly those states.  SIMULATION ONLY.  ~1 min (scene compile)."""
import pickle
import sys
from pathlib import Path

import mujoco

OUT = Path("/data/local/ss5772/molmospaces/percept/gpu")


def build(out=OUT):
    from world_model.v2.house.teleop.sim import TeleopSim
    from world_model.v2.house.teleop import operator as OP
    from world_model.v2.house.people_track import camera as CAM
    from world_model.v2.house.grasp.wrist import WristCam
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sim = TeleopSim("h5", seed=0)
    sc = sim.scene
    wp = sc.waypoints
    op = OP.SloppyOperator(0, "carry", sc.out_route, pick_pose=wp["cafe_counter_pick"], place_pose=wp["lab_desk_place"], params=None, back_route=sc.back_route,
                           back_mode=sc.back_mode)
    cam = CAM.HeadCamera(sim)
    wl, wr = WristCam(sim.m, "left"), WristCam(sim.m, "right")
    mujoco.mj_saveModel(sim.m, str(out / "h5.mjb"), None)
    fix = []
    for k in range(1200):
        sim.tick(op.step(sim.observe()))
        if k % 150 == 149:
            d = sim.d
            f = cam.render(sim.k, 0)
            fr = dict(k=sim.k, qpos=d.qpos.copy(), mpos=d.mocap_pos.copy(), mquat=d.mocap_quat.copy(), eq=d.eq_active.copy(), lift=sim.lift_actual(), head=f,
                      pose=cam.pose(), pose_base=sim.pose())
            for s, w in (("left", wl), ("right", wr)):
                fr["wrist_" + s] = w.capture(d, sim.k, seed=0)
            fix.append(fr)
    pickle.dump(dict(fix=fix, person_geoms=cam.person_geoms, self_geom=cam.self_geom, root=cam.root), open(out / "fix.pkl", "wb"))
    return out


if __name__ == "__main__":
    print(build(*sys.argv[1:]))
