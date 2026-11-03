"""Task 3/5/6: passive pan+lid+puck scene (robot absent, or present and locked), drift, speed, exact restore."""
import sys, json, time
import numpy as np
from pydrake.all import Simulator, RigidTransform, RotationMatrix, Quaternion
from world_model.v2.nero.pan.drake_spike.scene import build, mujoco


def pose_init(plant, ctx, bodies, mm):
    for nm, body in bodies.items():
        bid = mm.body(nm).id
        q = mm.body_quat[bid]
        plant.SetFreeBodyPose(ctx, body, RigidTransform(RotationMatrix(Quaternion(*q)), mm.body_pos[bid]))


def run(dt=0.005, T=5.0, with_robot=False, walls_hydro=True, restore=True, approx="sap"):
    b, plant, sg, bodies, mm, info = build(dt, approx, with_robot=with_robot, walls_hydro=walls_hydro)
    plant.Finalize()
    diagram = b.Build(); sim = Simulator(diagram)
    ctx = sim.get_mutable_context(); pctx = plant.GetMyMutableContextFromRoot(ctx)
    pose_init(plant, pctx, bodies, mm)
    if with_robot:   # hold robot at q=0 (all arm/gripper joints locked)
        for i in range(plant.num_joints()):
            from pydrake.all import JointIndex
            j = plant.get_joint(JointIndex(i))
            if j.num_positions() == 1: j.Lock(pctx)
    res = dict(dt=dt, approx=approx, with_robot=with_robot, walls_hydro=walls_hydro, **info)
    def P(nm): return plant.EvalBodyPoseInWorld(pctx, bodies[nm])
    p0 = {k: P(k) for k in bodies}
    t0 = time.time(); sim.AdvanceTo(T); wall = time.time() - t0
    p1 = {k: P(k) for k in bodies}
    lid_rel0 = p0["pan"].inverse() @ p0["lid"]; lid_rel1 = p1["pan"].inverse() @ p1["lid"]
    hid_rel0 = p0["pan"].inverse() @ p0["hidden"]; hid_rel1 = p1["pan"].inverse() @ p1["hidden"]
    d = lambda a, c: 1e3 * (c.translation() - a.translation())
    def ang(a, c): return float(np.degrees(np.arccos(np.clip((np.trace(a.rotation().matrix().T @ c.rotation().matrix()) - 1) / 2, -1, 1))))
    res.update(sim_T=T, wall_s=wall, wall_per_sim_s=wall / T,
               pan_drift_world_mm=d(p0["pan"], p1["pan"]).tolist(), pan_drift_norm_xy_mm=float(np.linalg.norm(d(p0["pan"], p1["pan"])[:2])),
               pan_rot_deg=ang(p0["pan"], p1["pan"]),
               lid_vs_pan_drift_mm=d(lid_rel0, lid_rel1).tolist(), lid_vs_pan_rot_deg=ang(lid_rel0, lid_rel1),
               puck_vs_pan_drift_mm=d(hid_rel0, hid_rel1).tolist(),
               pan_z_settle_mm=float(1e3 * (p1["pan"].translation()[2] - p0["pan"].translation()[2])))
    cr = plant.get_contact_results_output_port().Eval(pctx)
    res.update(n_hydro_contacts_end=cr.num_hydroelastic_contacts(), n_point_contacts_end=cr.num_point_pair_contacts())
    return res


def restore_test(dt=0.005, with_robot=False, kick=False):
    b, plant, sg, bodies, mm, info = build(dt, with_robot=with_robot)
    plant.Finalize(); diagram = b.Build(); sim = Simulator(diagram)
    ctx = sim.get_mutable_context(); pctx = plant.GetMyMutableContextFromRoot(ctx)
    pose_init(plant, pctx, bodies, mm)
    sim.Initialize(); sim.AdvanceTo(1.0)
    if kick:   # pan slides/spins on the stove through the restore interval (harder than a body at rest)
        from pydrake.all import SpatialVelocity
        plant.SetFreeBodySpatialVelocity(pctx, bodies['pan'], SpatialVelocity([0, 0, 1.0], [0.25, 0.1, 0]))
    saved = ctx.Clone()                       # full Context copy at t=1 s
    x_saved = ctx.get_discrete_state_vector().CopyToVector().copy() if ctx.num_discrete_state_groups() else None
    sim.AdvanceTo(2.0)
    qa = plant.GetPositions(pctx).copy(); va = plant.GetVelocities(pctx).copy()
    out = {}
    # (A) restore by Context.SetTimeAndContinuousState-equivalent: SetTimeStateAndParametersFrom(saved)
    ctx.SetTimeStateAndParametersFrom(saved)
    sim.Initialize(); sim.AdvanceTo(2.0)
    qb = plant.GetPositions(pctx).copy(); vb = plant.GetVelocities(pctx).copy()
    out["A_SetTimeStateAndParametersFrom"] = dict(max_abs_dq=float(np.abs(qa - qb).max()), max_abs_dv=float(np.abs(va - vb).max()),
                                                    bitwise_equal=bool(np.array_equal(qa, qb) and np.array_equal(va, vb)))
    # (B) fresh simulator on a clone of the saved context
    ctx2 = saved.Clone()
    sim2 = Simulator(diagram, ctx2); sim2.Initialize(); sim2.AdvanceTo(2.0)
    p2 = plant.GetMyContextFromRoot(ctx2)
    qc = plant.GetPositions(p2).copy(); vc = plant.GetVelocities(p2).copy()
    out["B_clone_new_simulator"] = dict(max_abs_dq=float(np.abs(qa - qc).max()), max_abs_dv=float(np.abs(va - vc).max()),
                                         bitwise_equal=bool(np.array_equal(qa, qc) and np.array_equal(va, vc)))
    # (C) uses state only: copy positions+velocities into a brand-new context (no clone) and rerun
    b2, plant2, sg2, bodies2, mm2, _ = build(dt, with_robot=with_robot); plant2.Finalize(); diagram2 = b2.Build()
    sim3 = Simulator(diagram2); c3 = sim3.get_mutable_context(); pc3 = plant2.GetMyMutableContextFromRoot(c3)
    ps = plant.GetMyContextFromRoot(saved)
    plant2.SetPositions(pc3, plant.GetPositions(ps)); plant2.SetVelocities(pc3, plant.GetVelocities(ps)); c3.SetTime(saved.get_time())
    sim3.Initialize(); sim3.AdvanceTo(2.0)
    qd = plant2.GetPositions(pc3).copy(); vd = plant2.GetVelocities(pc3).copy()
    out["C_new_context_q_v_only"] = dict(max_abs_dq=float(np.abs(qa - qd).max()), max_abs_dv=float(np.abs(va - vd).max()),
                                          bitwise_equal=bool(np.array_equal(qa, qd) and np.array_equal(va, vd)))
    return out


if __name__ == "__main__":
    which = sys.argv[1]
    if which == "passive":
        print(json.dumps(run(dt=float(sys.argv[2]), with_robot=(len(sys.argv) > 3 and sys.argv[3] == "robot"), walls_hydro=not (len(sys.argv) > 4 and sys.argv[4] == "nowalls")), indent=1))
    elif which == "restore":
        print(json.dumps(restore_test(float(sys.argv[2]), with_robot=(len(sys.argv) > 3 and sys.argv[3] == "robot"), kick=(len(sys.argv) > 4 and sys.argv[4] == "kick")), indent=1))
