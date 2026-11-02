"""Unit tests (a) TWIST and (b) CREEP in Drake (hydroelastic + discrete SAP). Spec: see SUMMARY.md.
usage: python -m world_model.v2.nero.pan.drake_spike.unit_tests [dt ...]"""
import sys, json, time
import numpy as np
from pydrake.all import (RigidTransform, ExternallyAppliedSpatialForce, SpatialForce, RotationMatrix, PrismaticJoint,
                         JointActuator, Simulator, SpatialInertia, UnitInertia, Box, CoulombFriction)
from pydrake.common.value import Value
from world_model.v2.nero.pan.drake_spike.common import make_plant, add_box_body, hydro_props, HYDRO_MODULUS, DISSIPATION, RES_HINT

MU = 1.0
S_PAD = 0.02


def yaw_of(R):
    return float(np.arctan2(R[1, 0], R[0, 0]))


def twist(dt, approx="sap", model="hydroelastic_with_fallback", F=10.0, rate=0.05, tmax=4.0, res=RES_HINT, modulus=HYDRO_MODULUS):
    b, plant, sg = make_plant(dt, approx, gravity=False, model=model)
    blk = add_box_body(plant, "block", (0.10, 0.10, 0.045), 0.5, MU, res=res, modulus=modulus, hydro=(model != "point"))
    pads, acts = [], []
    for nm, z in (("pad_top", 0.0225 + 0.005), ("pad_bot", -0.0225 - 0.005)):
        pad = add_box_body(plant, nm, (S_PAD, S_PAD, 0.010), 0.05, MU, res=res, modulus=modulus, hydro=(model != "point"))
        # prismatic z joint from world, pad frame origin starts at z (touching the block)
        fr = plant.AddFrame(__import__("pydrake.all", fromlist=["FixedOffsetFrame"]).FixedOffsetFrame(
            nm + "_anchor", plant.world_frame(), RigidTransform([0, 0, z])))
        j = plant.AddJoint(PrismaticJoint(nm + "_z", fr, pad.body_frame(), [0, 0, 1], -0.05, 0.05))
        acts.append(plant.AddJointActuator(nm + "_act", j))
        pads.append(pad)
    plant.Finalize()
    diagram = b.Build()
    sim = Simulator(diagram)
    ctx = sim.get_mutable_context()
    pctx = plant.GetMyMutableContextFromRoot(ctx)
    plant.SetFreeBodyPose(pctx, blk, RigidTransform())
    # actuation: top pad pushes down (-F), bottom pad pushes up (+F). Actuator order = order added.
    plant.get_actuation_input_port().FixValue(pctx, [-F, F])
    port = plant.get_applied_spatial_force_input_port()
    n = int(round(tmax / dt))
    rows = []
    slip_T = None
    yaw_1s = None
    unwrapped = 0.0; prev = 0.0
    t0 = time.time()
    for k in range(n):
        t = k * dt
        Tz = rate * t
        f = ExternallyAppliedSpatialForce()
        f.body_index = blk.index(); f.p_BoBq_B = np.zeros(3)
        f.F_Bq_W = SpatialForce(tau=[0, 0, Tz], f=[0, 0, 0])
        port.FixValue(pctx, Value[list[ExternallyAppliedSpatialForce]]([f]))
        sim.AdvanceTo((k + 1) * dt)
        V = plant.EvalBodySpatialVelocityInWorld(pctx, blk)
        wz = float(V.rotational()[2])
        yaw = yaw_of(plant.EvalBodyPoseInWorld(pctx, blk).rotation().matrix())
        d = (yaw - prev + np.pi) % (2 * np.pi) - np.pi; unwrapped += d; prev = yaw
        if slip_T is None and abs(wz) > 0.05:
            slip_T = rate * (k + 1) * dt
        if yaw_1s is None and (k + 1) * dt >= 1.0 - 1e-9:
            yaw_1s = unwrapped
    cr = plant.get_contact_results_output_port().Eval(pctx)
    nh = cr.num_hydroelastic_contacts(); npt = cr.num_point_pair_contacts()
    print("   twist end: pad z", [float(plant.EvalBodyPoseInWorld(pctx, p).translation()[2]) for p in pads], "blk z", float(plant.EvalBodyPoseInWorld(pctx, blk).translation()[2]), flush=True)
    return dict(test="twist", n_hydro_contacts_final=nh, n_point_contacts_final=npt, dt=dt, approx=approx, contact_model=model, slip_torque_Nm=slip_T, yaw_at_1s_rad=yaw_1s,
                final_yaw_rad=unwrapped, wall_s=time.time() - t0, analytic_slip_Nm=2 * MU * F * 0.3826 * S_PAD)


def creep(dt, approx="sap", model="hydroelastic_with_fallback", res=RES_HINT, modulus=HYDRO_MODULUS):
    g = 9.81; m = 1.0
    b, plant, sg = make_plant(dt, approx, gravity=True, model=model)
    plate = add_box_body(plant, "plate", (0.4, 0.4, 0.02), 10.0, MU, pose=RigidTransform([0, 0, -0.01]), welded=True, res=res, modulus=modulus, hydro=(model != "point"))
    blk = add_box_body(plant, "block", (0.10, 0.10, 0.02), m, MU, res=res, modulus=modulus, hydro=(model != "point"))
    plant.Finalize()
    diagram = b.Build(); sim = Simulator(diagram)
    ctx = sim.get_mutable_context(); pctx = plant.GetMyMutableContextFromRoot(ctx)
    plant.SetFreeBodyPose(pctx, blk, RigidTransform([0, 0, 0.01]))
    port = plant.get_applied_spatial_force_input_port()
    Fx = 0.5 * MU * m * g
    xs, ts = [], []
    t0 = time.time()
    n = int(round(3.5 / dt))
    for k in range(n):
        t = k * dt
        f = ExternallyAppliedSpatialForce(); f.body_index = blk.index(); f.p_BoBq_B = np.zeros(3)
        f.F_Bq_W = SpatialForce(tau=[0, 0, 0], f=[Fx if t >= 0.5 - 1e-12 else 0.0, 0, 0])
        port.FixValue(pctx, Value[list[ExternallyAppliedSpatialForce]]([f]))
        sim.AdvanceTo((k + 1) * dt)
        xs.append(float(plant.EvalBodyPoseInWorld(pctx, blk).translation()[0])); ts.append((k + 1) * dt)
    xs = np.array(xs); ts = np.array(ts)
    i05 = int(round(0.5 / dt)) - 1; i15 = int(round(1.5 / dt)) - 1; iend = -1
    cr = plant.get_contact_results_output_port().Eval(pctx)
    nh = cr.num_hydroelastic_contacts(); npt = cr.num_point_pair_contacts()
    return dict(test="creep", n_hydro_contacts_final=nh, n_point_contacts_final=npt, dt=dt, approx=approx, contact_model=model,
                mean_speed_last2s_mm_s=1e3 * (xs[iend] - xs[i15]) / (ts[iend] - ts[i15]),
                total_slide_mm=1e3 * (xs[iend] - xs[i05]), settle_z_mm=1e3 * (float(plant.EvalBodyPoseInWorld(pctx, blk).translation()[2]) - 0.01),
                wall_s=time.time() - t0)


if __name__ == "__main__":
    dts = [0.001, 0.005]
    out = []
    for model in ("hydroelastic_with_fallback", "point"):   # 'point' = Drake point contact (no hydroelastic), for contrast
        for dt in dts:
            for fn in (twist, creep):
                try:
                    r = fn(dt, model=model)
                except Exception as e:
                    r = dict(test=fn.__name__, dt=dt, contact_model=model, error=str(e)[:200])
                print(json.dumps(r), flush=True); out.append(r)
    json.dump(dict(assumed=dict(hydro_modulus_Pa=HYDRO_MODULUS, dissipation_s_per_m=DISSIPATION, res_hint_m=RES_HINT,
                                note="all ASSUMED; point-contact rows use Drake default point stiffness/dissipation, i.e. not tuned"), results=out),
              open("/common/users/ss5772/bimanual/world_model/v2/nero/results/drake_spike_2026-10-30/unit_tests.json", "w"), indent=1)
