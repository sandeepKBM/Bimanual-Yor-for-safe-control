"""experiments/baselines/mjpc/make_planning_model.py

Builds the MJPC PLANNING model (an MJCF file, NEW, generated -- the scene XMLs
on disk are never edited) for the bimanual cup-collect crossing:

  * same YOR robot + table + arm-arm/arm-table collision patching as the demo
    (cct.load_yor_cup_collect_model's _patch_real_collision_geoms);
  * cup/balls/cubes REMOVED (the demo's cup is kinematically re-pinned to the
    left EE and never loads the arm; its geometry enters the cost exactly the
    way it enters FL-MPC's barrier -- as cup keypoint spheres on the left EE,
    passed per episode via userdata);
  * the 12 arm position servos (disabled at runtime in the demo; torques go in
    via qfrc_applied) replaced by 12 torque MOTORS, ctrlrange = +-TAU_MAX=100 Nm
    (sim_common.TAU_MAX, from the unmodified YOR model's effort=100), with
    dyntype="user"+actearly: the server's mjcb_act_dyn realises the plant's
    TorqueCommandFilter (low-pass + 400 Nm/s slew) as actuator state;
  * finger servos removed (fingers are passive in the plan; their state is
    copied from the plant each step);
  * MJPC residual user-sensors + <custom> agent/planner numerics + static
    keypoint tables (sim_common.barrier_keypoints, identical to FL-MPC's).

Visual-only geoms are dropped (explicit inertials are frozen first, so the
multibody dynamics are unchanged) so the file loads in MJPC's pinned MuJoCo
(3.2.3) with no mesh assets.

Usage: python make_planning_model.py --out <file.xml> [--planner 0|2|...] ...
(mjpc_planner.py calls build_planning_xml() directly.)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[3]
for _p in (str(REPO_ROOT), str(REPO_ROOT / "scripts"),
           str(REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"), str(REPO_ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only)
from scripts import cup_collect_task_utils as cct  # noqa: E402  (read-only)

# ---- userdata layout: MUST match cpp/bimanual_cup_task.h ----
UD_T0, UD_DTREF, UD_NK, UD_CUPAXIS, UD_HOLDPOS, UD_QHOLD, UD_NCUP, UD_CUPKP = 0, 1, 2, 3, 6, 9, 15, 16
MAX_CUP = 24
UD_REF = UD_CUPKP + 4 * MAX_CUP
MAX_KNOTS = 160
N_USERDATA = UD_REF + 12 * MAX_KNOTS

# ---- default residual weights (the TUNED set; see README.md for the tuning log) ----
DEFAULT_WEIGHTS = dict(
    RTrackQ=600.0,     # == FL-MPC W_POS_INT
    RTrackQd=8.0,      # == FL-MPC W_VEL_INT
    LVel=15.0,         # == FL-MPC W_VEL_EGO (ego_hold=False: damping only)
    LHold=0.0,         # == FL-MPC ego_hold=False (W_POS_EGO -> 0)
    Leash=2.0e5,       # FL-MPC: hard leash 0.08 m
    Tilt=50.0,         # mild upright preference
    TiltViol=2.0e5,    # FL-MPC: hard 0.10 rad tilt bound -> steep rectified cost
    Clear=2.0e5,       # FL-MPC: hard arm-arm HOCBF -> steep rectified cost on h < H_SAFE
    Table=2.0e5,       # FL-MPC: hard table-floor rows
    Effort=1.0e-4,     # torque effort (Nm^2)
    TauRate=1.0e-6,    # applied-torque rate (Nm/s)^2 -- the control-rate / smoothness term
    Accel=1.0e-3,      # joint accel (FL-MPC R_ACC=1e-4 on accel + W_JERK on its change)
)
RESIDUAL_DIMS = [("RTrackQ", 6), ("RTrackQd", 6), ("LVel", 6), ("LHold", 6), ("Leash", 3),
                 ("Tilt", 2), ("TiltViol", 1), ("Clear", 1), ("Table", 1),
                 ("Effort", 12), ("TauRate", 12), ("Accel", 12)]
DEFAULT_PARAMS = dict(HSafe=0.02, DMin=None, TiltLim=0.08, Leash=0.08, ZFloor=None)
DEFAULT_AGENT = dict(agent_planner=0, agent_horizon=0.35, agent_timestep=0.01,
                     sampling_trajectories=64, sampling_spline_points=6, sampling_exploration=0.08,
                     sampling_representation=2, gradient_spline_points=6, ilqg_num_rollouts=10)

# (element, attribute) pairs that MuJoCo 3.9's to_xml emits but MJPC's pinned
# MuJoCo 3.2.3 parser rejects (found by loading the file in the MJPC server).
_DROP_ATTRS = [("texture", "colorspace"), ("light", "type")]


def _flat_kps(kps):
    return [float(v) for bid, off, r in kps for v in (bid, off[0], off[1], off[2], r)]


# PD-setpoint parameterisation (act_mode="pd"): the demo's own computed-torque
# gains (sim_common.KP_TASK=600, KD_TASK=2*sqrt(600)) applied through the arm's
# mass-matrix DIAGONAL at the home pose (diag(M) = [0.157 0.177 0.209 0.0082 0.0278
# 0.0079] kg m^2, both arms) -> kp = 600*M_ii, kd = 49*M_ii (critically damped,
# ~3.9 Hz). The plant evaluates the identical clip(kp(u-q)-kd*qd, +-100 Nm).
_M_DIAG = np.array([0.1572, 0.1773, 0.2094, 0.0082, 0.0278, 0.0079])
PD_KP = np.tile(600.0 * _M_DIAG, 2)
PD_KD = np.tile(2.0 * np.sqrt(600.0) * _M_DIAG, 2)


def build_planning_xml(out_path: str, table_top_z: float, weights: dict | None = None,
                       params: dict | None = None, agent: dict | None = None,
                       act_mode: str = "torque") -> str:
    assert act_mode in ("torque", "pd"), act_mode
    w = dict(DEFAULT_WEIGHTS)
    w.update(weights or {})
    p = dict(DEFAULT_PARAMS)
    p.update(params or {})
    if p["DMin"] is None:
        p["DMin"] = float(sc.D_MIN)
    if p["ZFloor"] is None:
        p["ZFloor"] = float(table_top_z) - 0.006     # demo TABLE_FLOOR_MARGIN_M
    ag = dict(DEFAULT_AGENT)
    ag.update(agent or {})

    spec = mujoco.MjSpec.from_file(cct.YOR_CUP_COLLECT_XML)
    sc._patch_real_collision_geoms(spec)

    # 1) freeze inertials from the full compiled model (so dropping visual geoms keeps dynamics)
    full = spec.compile()
    for b in spec.bodies:
        if b.name == "world" or not b.name:
            continue
        bid = full.body(b.name).id
        b.explicitinertial = True
        b.mass = float(full.body_mass[bid])
        b.ipos = full.body_ipos[bid].copy()
        b.iquat = full.body_iquat[bid].copy()
        b.inertia = full.body_inertia[bid].copy()

    # 1b) mesh-FITTED primitive geoms (e.g. the arms' collision capsules are
    # type=capsule + mesh=... -> MuJoCo fits size/pos/quat to the mesh at compile)
    # -> freeze the compiled primitive so no mesh asset is needed.
    for g in spec.geoms:
        if g.meshname and g.type != mujoco.mjtGeom.mjGEOM_MESH:
            gid = g.id
            assert full.geom_type[gid] == g.type, (g.name, full.geom_type[gid], g.type)
            g.size = full.geom_size[gid].copy()
            g.pos = full.geom_pos[gid].copy()
            g.quat = full.geom_quat[gid].copy()
            g.meshname = ""
            try:
                g.fitscale = 1.0
            except Exception:
                pass

    # 2) remove cup / balls; cubes become MOCAP (kinematic) obstacles whose pose
    #    is copied from the plant every step (MJPC's State carries mocap), so the
    #    planner sees the un-held cubes it could bump into (the held cube is
    #    parked far away -- its arm collisions are disabled in the plant too).
    drop = [b for b in spec.bodies if b.name == cct.CUP_BODY_NAME or b.name.startswith("ball_")]
    for b in drop:
        spec.delete(b)
    for b in spec.bodies:
        if b.name.startswith("cube_"):
            for j in list(b.joints):
                spec.delete(j)
            for fj in list(getattr(b, "freejoints", []) or []):
                spec.delete(fj)
            b.mocap = True

    # 3) drop visual-only geoms + all meshes/textures/materials
    for b in list(spec.bodies) + [spec.worldbody]:
        for g in list(b.geoms):
            if g.contype == 0 and g.conaffinity == 0:
                spec.delete(g)
    # the base/lift collision meshes (contype=0, conaffinity=1) can only touch
    # contype-bit-1 geoms (cup/cubes/balls -- removed); arms use bits 4/8/16/32
    # and the lift is welded to the world with the table (static-static pairs
    # never collide) -> they are inert in the planning model; drop them.
    for g in list(spec.geoms):
        is_mesh = g.type == mujoco.mjtGeom.mjGEOM_MESH or bool(g.meshname)
        if not is_mesh or g.contype != 0:
            continue
        if g.parent.name.startswith(("left_arm_", "right_arm_")):
            continue
        spec.delete(g)
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_MESH or g.meshname:
            raise RuntimeError(f"collision mesh geom {g.name!r} on {g.parent.name} ct={g.contype} ca={g.conaffinity} mesh={g.meshname} survives")
    for x in list(spec.meshes):         # textures/materials are all builtin/rgba -> kept
        spec.delete(x)
    for x in list(spec.keys):
        spec.delete(x)
    for x in list(spec.equalities):
        spec.delete(x)
    for x in list(spec.cameras):
        spec.delete(x)
    for x in list(spec.sensors):
        spec.delete(x)

    # 4) actuators: 12 arm torque motors with the plant-filter actuator dynamics
    for a in list(spec.actuators):
        spec.delete(a)
    for side in ("left", "right"):
        for i in range(1, 7):
            a = spec.add_actuator()
            a.name = f"tau_{side}_arm_joint{i}"
            a.target = f"{side}_arm_joint{i}"
            a.trntype = mujoco.mjtTrn.mjTRN_JOINT
            a.dyntype = mujoco.mjtDyn.mjDYN_USER
            a.actdim = 1
            a.actearly = True
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.gainprm[0] = 1.0
            a.biastype = mujoco.mjtBias.mjBIAS_NONE
            a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
            if act_mode == "torque":
                a.ctrlrange = [-sc.TAU_MAX, sc.TAU_MAX]
            else:   # joint setpoint, within the joint range
                a.ctrlrange = full.jnt_range[full.joint(f"{side}_arm_joint{i}").id].copy()
            a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
            a.forcerange = [-sc.TAU_MAX, sc.TAU_MAX]

    # 5) custom numerics: agent + residual params (ORDER matters for residual_*)
    for k, v in ag.items():
        n = spec.add_numeric()
        n.name, n.data = k, [float(v)]
    for k, v in (("bimanual_act_mode", [0.0 if act_mode == "torque" else 1.0]),
                 ("bimanual_kp", list(PD_KP)), ("bimanual_kd", list(PD_KD))):
        n = spec.add_numeric()
        n.name, n.data = k, [float(x) for x in v]
    for k in ("HSafe", "DMin", "TiltLim", "Leash", "ZFloor"):
        n = spec.add_numeric()
        n.name, n.data = f"residual_{k}", [float(p[k])]

    # 6) residual user sensors (must be FIRST sensors)
    for name, dim in RESIDUAL_DIMS:
        s = spec.add_sensor()
        s.name, s.type, s.dim = name, mujoco.mjtSensor.mjSENS_USER, dim
        s.needstage = mujoco.mjtStage.mjSTAGE_ACC
        s.datatype = mujoco.mjtDataType.mjDATATYPE_REAL
        s.userdata = [0.0, float(w[name]), 0.0, float(max(10.0 * w[name], 1.0))]
    spec.nuserdata = N_USERDATA
    spec.nuser_sensor = 4

    # 7) static keypoint tables -- need the compiled planning model's body ids
    m_plan = spec.compile()
    sc._KEYPOINT_CACHE.clear()
    left = sc.barrier_keypoints(m_plan, sc.EGO_PREFIX)
    right = sc.barrier_keypoints(m_plan, sc.INT_PREFIX)
    tips = []
    for gname, r in [("right_arm_link7_pad", 0.006), ("right_arm_link8_pad", 0.006)]:
        gid = m_plan.geom(gname).id
        tips.append((int(m_plan.geom_bodyid[gid]), m_plan.geom_pos[gid].copy(), r))
    tips.append((m_plan.body("right_arm_gripper_base").id, np.zeros(3), 0.01))
    sc._KEYPOINT_CACHE.clear()
    for name, kps in [("kp_left", left), ("kp_right", right), ("kp_left_table", left),
                      ("kp_right_table", list(right) + tips)]:
        n = spec.add_numeric()
        n.name, n.data = name, _flat_kps(kps)

    m_plan = spec.compile()
    xml = spec.to_xml()
    for el, attr in _DROP_ATTRS:
        xml = re.sub(r'(<%s\b[^>]*?)\s%s="[^"]*"' % (el, attr), r"\1", xml)
    Path(out_path).write_text(xml)
    return out_path


def planning_mocap_cubes(m_plan):
    """[(mocap_id, cube_body_name)] for the cube mocap obstacles of the planning model."""
    out = []
    for b in range(m_plan.nbody):
        name = m_plan.body(b).name
        if name.startswith("cube_") and m_plan.body_mocapid[b] >= 0:
            out.append((int(m_plan.body_mocapid[b]), name))
    return out


def planning_joint_map(m_plan, m_sim):
    """qpos/qvel index maps sim -> plan for every (hinge/slide) joint the planning model has."""
    qmap, vmap = [], []
    for j in range(m_plan.njnt):
        name = m_plan.joint(j).name
        js = m_sim.joint(name).id
        qmap.append((int(m_plan.jnt_qposadr[j]), int(m_sim.jnt_qposadr[js])))
        vmap.append((int(m_plan.jnt_dofadr[j]), int(m_sim.jnt_dofadr[js])))
    return qmap, vmap


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--table-top-z", type=float, default=0.55)
    a = ap.parse_args()
    print(build_planning_xml(a.out, a.table_top_z))
