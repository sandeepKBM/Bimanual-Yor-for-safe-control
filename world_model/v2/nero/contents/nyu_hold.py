"""nyu_hold.py -- hold poses of the contents scenes (tray / box with side tabs) for the NYU gripper (11-03 grasp-geometry fix).

The AgileX capture (results/.../capture.json) came from the pan Episode's pre-grasp IK with the AgileX TCP (ee_grasp_z 0.120): the NYU jaws overlapped
the object / world at those poses.  Here the hold pose is re-solved for the NYU TCP (nyu_gripper.EE_GRASP_Z, 40 % of pivot-to-tip):
  * ee_grasp site on obj_grasp_left / obj_grasp_right, approach from the SIDE (left -y, right +y, as before), jaw closing axis = world z;
  * jaws set so the inner gap at the TCP = tab thickness + 2 x GRASP_CLEARANCE_M (gripper.opening_for_width);
  * among converged IK solutions: clearance >= CLEARANCE_MIN (5 mm, ASSUMED) of every robot geom against the world, against the object (arm links and the
    OTHER gripper; each welded gripper vs the object is exempt, see weldcontact) and left vs right arm, at the grasp pose AND the 5 cm lift pose; then the
    largest joint-limit margin.
Result file: contents/capture_nyu.json (same layout as capture.json["pot"], cached separately from the earlier capture.json, which nothing loads any more).
    python -m world_model.v2.nero.contents.nyu_hold --write
"""
from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np

from .. import config as C
from .. import gripper as G
from .. import clearance as CL
from ..ik import ArmIK
from ..democ import kin as K
from ..pan import scene as PS
from . import scenes as SC
from . import envelope as E

CAP_NYU = pathlib.Path(__file__).resolve().parent / "capture_nyu.json"
APPROACH = {"left": np.array([0.0, -1.0, 0.0]), "right": np.array([0.0, 1.0, 0.0])}
JAW = np.array([0.0, 0.0, 1.0])
LIFT = 0.05
# 11-03 jaw/body fix (contents tabs only; the pot keeps gripper.TCP_FRAC and its opening):
SHIFT_M = 0.055         # ASSUMED: ee_grasp site sits this far OUTSIDE the tab centre, along -approach, so the tab lies in the distal part of the pads and the closed jaw
                        # tips stop short of the object body (grasp depth = EE_GRASP_Z + SHIFT_M; tray needs >= 0.053: 0.045 left the tips 2.8 mm inside the rim).  Old value 0.0 (site on the tab centre).
BODY_CLEAR_MIN = 0.005  # ASSUMED (= CL.CLEARANCE_MIN): every jaw / palm geom of a welded gripper vs the object BODY geoms (tab_* excluded)
JAW_Z_FAR = 0.2110      # MEASURED nyu_gripper.JAW_TIP_Z_OPEN_M: deepest point of the pad region the tab can reach (gap is checked up to here)


def _model(world):
    m = PS.apply_sim_defaults(SC.build_spec(world).compile())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    return m, d


def hold_opening(w=SC.TAB_Z, shift=None, z_far=JAW_Z_FAR, clearance=None, tab_outer=0.21, grasp_y=None):
    """jaw opening for the tab grasp with the grasp point `shift` outside the tab: the most closed jaw angle whose pad-to-pad gap is >= w + 2 clearance at
    EVERY depth from the tab's outer end to z_far (the jaws are a V: closing on the tab at the TCP only (gripper.opening_for_width) makes the gap shrink
    below the tab thickness deeper along the pad, 14.5 mm at 0.184 m for a 20 mm tab).  -> dict(frac, nut, theta, gap_min_over_tab, z_lo, z_hi, ...).
    The pot path does not call this."""
    ng = G._nyu()
    cl = ng.GRASP_CLEARANCE_M if clearance is None else float(clearance)   # ASSUMED 2 mm per side, as gripper.opening_for_width
    shift = SHIFT_M if shift is None else float(shift)
    target = float(w) + 2.0 * cl
    z_lo = G.tcp_offset() + ((SC.GRASP_Y if grasp_y is None else grasp_y) + shift - tab_outer)     # MODEL: depth along the pads of the tab's outer end
    zs = np.linspace(z_lo, z_far, 12)
    best = None
    for th in np.arange(0.0, ng.THETA_MAX_RAD, 0.002):
        if min(G.jaw_gap_m(th, z) for z in zs) >= target:
            best = th
    if best is None:
        raise RuntimeError("no jaw angle keeps the gap >= tab thickness over the pad")
    u = G.theta_to_nut(best)
    return dict(frac=1.0 - u / ng.NUT_TRAVEL_M, nut=u, theta=float(best), target=target, z_lo=float(z_lo), z_hi=float(z_far),
                gap_min_over_tab=float(min(G.jaw_gap_m(best, z) for z in zs)))


def hold_finger_frac(w=SC.TAB_Z):
    """hold_opening() as the opening fraction that carry.HOLD_FINGER / gripper.set_opening take (gripper.opening_for_width(TAB_Z)["frac"] would put
    the jaws 9.4 mm inside the tab, this one leaves them 1.6 mm clear)."""
    return hold_opening(w)["frac"]


def set_grip(m, d, w=SC.TAB_Z, shift=None, world=None):
    """world: the depth of the tab along the pads follows the world's tabs (BoxWorld: crate end wall); None = the nominal tab."""
    o = hold_opening(w, shift) if world is None else hold_opening(w, shift, tab_outer=getattr(world, "tab_outer_y", 0.21), grasp_y=world.grasp_y)
    for s in C.SIDES:
        G.set_opening(m, d, s, o["frac"])
    return o


def jaw_body_min(m, d, obj_body="obj"):
    """-> (min distance [m], (gripper geom, body geom)) of every jaw / palm geom of both grippers vs the object BODY geoms (tab_* geoms excluded: the tab is
    between the pads).  Negative = penetration.  Own GJK (clearance._dist)."""
    robot, _, obj = CL.classify(m, obj_body)
    nm = lambda g: m.geom(g).name or f"g{g}"
    body = [g for g in obj if not nm(g).startswith("tab")]
    best = (CL._DMAX, None)
    for s in C.SIDES:
        for g in robot[s]:
            if CL._is_grip(m, g):
                for o in body:
                    v = CL._dist(m, d, g, o)
                    if v < best[0]:
                        best = (v, (nm(g), nm(o)))
    return best


def _pose_clear(m, d, ik, q, d_obj=0.0):
    for s in C.SIDES:
        d.qpos[ik[s].qa] = q[s]
    mujoco.mj_forward(m, d)
    rep = CL.report(m, d, "obj", welded=C.SIDES, detail=True)
    jb, pair = jaw_body_min(m, d)
    rep["jaw_body"] = jb
    rep["pairs"]["jaw_body"] = pair
    rep["min"] = min(rep["min"], jb)          # BODY_CLEAR_MIN == CL.CLEARANCE_MIN (5 mm, ASSUMED), so one threshold covers all
    return rep


def _tips_toward_object(ik, sols, approach):
    """ArmIK's approach error is a cross product, so it also accepts the anti-parallel gripper (jaw tips pointing AWAY from the object, palm over the body).
    Keep only the solutions whose gripper z axis points along `approach` (11-03 fix: at an outward-shifted grasp the right arm only had anti-parallel ones)."""
    keep = []
    for q, e in sols:
        ik.set_q(q)
        if float(ik.d.xmat[ik.gb].reshape(3, 3)[:, 2] @ approach) > 0.9:
            keep.append((q, e))
    return keep


def _srs_candidates(m, d, side, tgt, app, ik, n=12, n_psi=360):
    """11-19: exact closed-form (SRS) IK candidates at the same target as ArmIK.collect: every arm angle psi on a grid x both jaw signs x all branches inside the MJCF
    joint limits with residual <= 1e-5 m / 1e-4 rad (so the 1 mm / 1 deg grasp tolerance is not loosened); greedy picks by joint margin, >= 0.08 rad apart.  -> [(q, err)]"""
    from ...stack.kin.srs_ik import SRSArm
    arm = SRSArm(m, side)
    arm.refresh_base(d)
    found = []
    for sg in (1.0, -1.0):
        y = sg * JAW
        T = np.eye(4)
        T[:3, :3] = np.column_stack([np.cross(y, app), y, app])
        T[:3, 3] = tgt
        for psi in np.linspace(-np.pi, np.pi, n_psi, endpoint=False):
            for r in arm.ik_full(T, psi):
                if r["pos_err"] <= 1e-5 and r["rot_err"] <= 1e-4 and r["in_limits"]:
                    found.append((ik.margin(r["q"]), r["q"], r["pos_err"], r["rot_err"]))
    found.sort(key=lambda t: -t[0])
    out = []
    for mg, q, pe, re in found:
        if all(np.abs(q - o[0]).max() > 0.08 for o in out):
            out.append((q, (pe, re, 0.0)))
        if len(out) >= n:
            break
    return out


def solve(world=None, n_sol=12, seeds=80, verbose=False, shift=None):
    """-> dict(q={side: q7}, clear=report, opening=..., cands=(nL, nR))"""
    world = world or SC.TrayWorld()
    m, d = _model(world)
    shift = SHIFT_M if shift is None else float(shift)
    op = set_grip(m, d, shift=shift, world=world)
    ik = {s: ArmIK(m, s) for s in C.SIDES}
    # ee_grasp site goes `shift` outside the tab centre (along -approach): the tab then lies `shift` deeper along the pads
    Ro = d.xmat[m.body("obj").id].reshape(3, 3)       # 11-19: the approach follows the object's start yaw (tab frame); Ro = I at yaw 0 (the captured pose)
    app = {s: Ro @ APPROACH[s] for s in C.SIDES}
    tgt = {s: d.site_xpos[m.site(f"obj_grasp_{s}").id].copy() - shift * app[s] for s in C.SIDES}
    sols = {s: _tips_toward_object(ik[s], ik[s].collect(tgt[s], app[s], JAW, n_sol=n_sol, seeds=seeds, seed=i), app[s]) for i, s in enumerate(C.SIDES)}
    best = None
    for stage in (0, 1):     # stage 0: the DLS candidates only (reproduces capture_nyu.json at the nominal world); stage 1 (only if no pair passed): + exact SRS candidates (one rule for every world)
        if stage == 1:
            for s in C.SIDES:
                sols[s] = sols[s] + [c for c in _srs_candidates(m, d, s, tgt[s], app[s], ik[s]) if all(np.abs(c[0] - o[0]).max() > 0.08 for o in sols[s])]
        for qa, _ in sols["left"]:
            for qb, _ in sols["right"]:
                q = {"left": qa, "right": qb}
                rep = _pose_clear(m, d, ik, q)
                if rep["min"] < CL.CLEARANCE_MIN:
                    continue
                ql = {}
                for s in C.SIDES:
                    pik = K.PoseIK(m, s)
                    p0, R0 = pik.ik.pose(q[s])
                    qq, err = pik.solve(p0 + [0, 0, LIFT], R0, q[s], q_post=q[s], k_post=0.1, iters=200, tol=1e-9)
                    ql[s] = qq
                rl = _pose_clear(m, d, ik, ql)
                if rl["min"] < CL.CLEARANCE_MIN:
                    continue
                score = min(ik[s].margin(q[s]) for s in C.SIDES)
                if best is None or score > best[0]:
                    best = (score, q, rep, rl)
        if best is not None:
            break
    if best is None:
        raise RuntimeError(f"no hold pose with >= {CL.CLEARANCE_MIN*1e3:.0f} mm clearance ({len(sols['left'])}/{len(sols['right'])} IK solutions incl. exact SRS)")
    score, q, rep, rl = best
    return dict(q=q, clear=rep, clear_lift=rl, margin=float(score), opening=op, shift=shift, cands=(len(sols["left"]), len(sols["right"])), world=world,
                tcp=tgt)


from ...stack.planners.closed_chain import PlanRefused      # noqa: E402  (a refusal for behind_job / the planners: outcome refused_planner)


class GraspUnreachable(PlanRefused):
    """no hold pose with both claws on the world's tabs and >= CLEARANCE_MIN clearance: a PlanRefused with reason "grasp_unreachable"."""

    def __init__(self, msg):
        super().__init__(msg, reason="grasp_unreachable")


# world fields that do not move the arms' hold pose (contents mass / position / size, fills): the pose is a kinematic IK to the tabs + clearance of the geometry
_HOLD_IRRELEVANT = ("contents_kg", "contents_xy", "contents_half", "contents_mu", "fills_kg", "box_mu", "box_mass", "tray_mu", "tray_mass", "cup_mu", "cup_mass", "counter_mu")
_HOLD_MEMO = {}


_FP_SOURCES = ("contents/nyu_hold.py", "contents/scenes.py", "ik.py", "clearance.py", "gripper.py", "nyu_gripper.py", "config.py", "democ/kin.py", "pan/scene.py",
               "../stack/kin/srs_ik.py")      # relative to nero/: every file whose code the solve runs (a source edit changes the key)
_FP_MODEL_ARRAYS = ("jnt_range", "body_pos", "body_quat", "body_mass", "body_inertia", "body_ipos", "geom_size", "geom_pos", "geom_quat", "geom_type", "geom_contype",
                    "geom_conaffinity", "geom_margin", "geom_gap", "geom_rbound", "dof_armature", "dof_damping", "actuator_forcerange", "actuator_gear", "site_pos", "site_quat")
_FP_MODEL_MEMO = {}


def _simple_attrs(mod):
    """the module's public constants (numbers, strings, tuples / lists / dicts of them, arrays): what a test may monkeypatch."""
    out = {}
    for k in sorted(vars(mod)):
        v = vars(mod)[k]
        if k.startswith("_") or isinstance(v, (type, type(mod))) or callable(v):
            continue
        out[k] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def _cpu_tag():
    """CPU model + machine: hold poses from different CPUs differ at the IK tolerance level (0.0024 rad measured 11-19), so a cache entry is per CPU."""
    import platform
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip() + "|" + platform.machine()
    except OSError:
        pass
    return platform.processor() + "|" + platform.machine()


def solver_fingerprint():
    """sha1 hex of every input of solve() other than the world: source of the files the solve runs, the module constants of nyu_hold / scenes / config / gripper /
    clearance / the NYU gripper module, every BIMANUAL_* env var, the identity of the model-building callables, and the arrays of the compiled robot model the IK /
    clearance read.  The cheap part is hashed per call; the compiled-model part is computed once per process for each distinct cheap part."""
    import hashlib
    import os
    nero = pathlib.Path(__file__).resolve().parents[1]
    h = hashlib.sha1()
    for rel in _FP_SOURCES:
        try:
            h.update((nero / rel).read_bytes())
        except OSError:
            h.update(rel.encode())
    import sys
    for mod in (sys.modules[__name__], SC, C, G, CL, G._nyu()):
        h.update(repr(sorted(_simple_attrs(mod).items(), key=lambda t: t[0])).encode())
    h.update(repr(sorted((k, v) for k, v in os.environ.items() if k.startswith("BIMANUAL"))).encode())
    h.update(_cpu_tag().encode())          # USER 11-19 ("Tag cache by CPU"): the iterative IK stops at its tolerance at a CPU-dependent point (AMD EPYC 7352 vs Intel Xeon Gold 5220: 0.0024 rad)
    try:
        h.update(repr((float(C.claw_default()), int(C.substeps_default()), float(G.tcp_offset()))).encode())
    except Exception as e:      # a patched callable that fails must still change the key
        h.update(repr(e).encode())
    cheap = h.hexdigest()
    mk = (cheap, id(PS.apply_sim_defaults), id(SC.build_spec), id(ArmIK.solve), id(CL.report))
    if mk not in _FP_MODEL_MEMO:
        m = PS.apply_sim_defaults(SC.build_spec(SC.BoxWorld()).compile())
        hm = hashlib.sha1()
        for a in _FP_MODEL_ARRAYS:
            hm.update(np.ascontiguousarray(getattr(m, a)).tobytes())
        hm.update(repr(float(m.opt.timestep)).encode())
        _FP_MODEL_MEMO[mk] = hm.hexdigest()
    return hashlib.sha1((cheap + _FP_MODEL_MEMO[mk]).encode()).hexdigest()


def _hold_key(world):
    """cache key: the world's geometry fields + the solver fingerprint (a patched / edited solver input gives another key)."""
    import dataclasses
    import hashlib
    d = {k: v for k, v in dataclasses.asdict(world).items() if k not in _HOLD_IRRELEVANT}
    return type(world).__name__ + "_" + hashlib.sha1(repr(sorted(d.items())).encode()).hexdigest()[:16] + "_" + solver_fingerprint()[:16]


def _write_atomic(f, text):
    """tmp file in the same directory + os.replace: a reader never sees a partial entry, concurrent writers of one key leave one complete file."""
    import os
    tmp = f.with_name(f".{f.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, f)


def hold_pose(world):
    """MODEL: {side: q (7,)} arm configuration with each claw's ee_grasp site on its grasp tab of `world`'s object (tab positions from the world's obj_xy, obj_yaw
    and counter height) at the start, before the lift.  One rule for every world: solve(world) (the procedure that wrote capture_nyu.json; at the nominal tray /
    box world it reproduces the captured W0_liftoff exactly, error 0.0).  Cached per world geometry (memory + $BIMANUAL_PLAN_CACHE/hold_pose, default
    stack/planners/plan_cache/hold_pose).  Raises GraspUnreachable when no pose with >= CLEARANCE_MIN clearance exists."""
    import os
    if isinstance(world, SC.PotWorld):     # EXCEPTION (reported 11-19): the pot has handles, not tabs; solve() finds no >= 5 mm pose for it, and no geometry prior varies it -> its captured pose
        cap = json.loads(CAP_NYU.read_text())["pot"]["q"]["W0_liftoff"]
        return {s: np.array(cap[s], float) for s in C.SIDES}
    key = _hold_key(world)
    if key in _HOLD_MEMO:
        r = _HOLD_MEMO[key]
        if isinstance(r, str):
            raise GraspUnreachable(r)
        return {s: np.array(v) for s, v in r.items()}
    root = pathlib.Path(os.environ.get("BIMANUAL_PLAN_CACHE", pathlib.Path(__file__).resolve().parents[2] / "stack" / "planners" / "plan_cache")) / "hold_pose"
    f = root / f"{key}.json"
    try:
        r = json.loads(f.read_text())
    except (OSError, ValueError):
        r = None
    if r is None:
        try:
            q = solve(world)["q"]
            r = {s: [float(x) for x in q[s]] for s in C.SIDES}
        except RuntimeError as e:
            r = f"grasp_unreachable: {e}"
        try:
            root.mkdir(parents=True, exist_ok=True)
            _write_atomic(f, json.dumps(r))
        except OSError:
            pass
    _HOLD_MEMO[key] = r
    if isinstance(r, str):
        raise GraspUnreachable(r)
    return {s: np.array(v) for s, v in r.items()}


def capture(world=None, **kw):
    """capture_nyu.json content: capture.json['pot'] layout, q / tcp per waypoint (continuation IK, orientation held, as envelope.capture)."""
    r = solve(world, **kw)
    world = r["world"]
    m, d = _model(world)
    set_grip(m, d, shift=r['shift'], world=world)
    q, tcp, clear = {}, {}, {}
    ik = {s: ArmIK(m, s) for s in C.SIDES}
    for s in C.SIDES:
        pik = K.PoseIK(m, s)
        q0 = np.asarray(r["q"][s])
        p0, R0 = pik.ik.pose(q0)
        for wp, dl in E.WAYPOINTS.items():
            qq = q0.copy()
            for i in range(1, 61):
                qq, err = pik.solve(p0 + np.asarray(dl, float) * i / 60, R0, qq, q_post=q0, k_post=0.1, iters=60, tol=1e-9)
            q.setdefault(wp, {})[s] = qq.tolist()
            tcp.setdefault(wp, {})[s] = pik.ik.pose(qq)[0].tolist()
            if err > 1e-4:
                q[wp][s + "_ik_err"] = float(err)
    for wp in q:
        for s in C.SIDES:
            d.qpos[ik[s].qa] = q[wp][s]
        clear[wp] = {k: v for k, v in CL.report(m, d, "obj", welded=C.SIDES).items()}
        clear[wp]["jaw_body"] = jaw_body_min(m, d)[0]
    old = json.loads((E.RESULTS / "capture.json").read_text())["pot"]      # object-side numbers do not depend on the gripper (pot: o0, R0, empty mass / COM)
    grip = float(m.body_subtreemass[m.body("left_arm_gripper_base").id])
    cap = dict(obj="pot", grasp_frac=old["grasp_frac"], sides=["left", "right"], q=q, tcp=tcp, qpos_rest=[], o0=old["o0"], R0=old["R0"],
               empty_mass=old["empty_mass"], empty_com_world=old["empty_com_world"], gripper_mass=grip,
               provenance=dict(gripper="nyu", tcp_z=G.tcp_offset(), grasp_shift_m=r['shift'], grasp_shift_label='ASSUMED (11-03; old 0.0)', body_clear_min_m=BODY_CLEAR_MIN, tab_thickness=SC.TAB_Z, opening=r["opening"], clearance_min_required_m=CL.CLEARANCE_MIN,
                               clearance_at_hold=r["clear"], clearance_at_lift=r["clear_lift"], clearance_by_waypoint=clear, n_ik_solutions=list(r["cands"]),
                               joint_margin=r["margin"],
                               note="solved by contents/nyu_hold.py; o0/R0/empty_* copied from results/contents_envelope_2026-11-01/capture.json (object side, gripper independent)"))
    return cap


def write(path=CAP_NYU, **kw):
    cap = capture(**kw)
    pathlib.Path(path).write_text(json.dumps({"pot": cap}, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    return cap


if __name__ == "__main__":
    import sys
    c = write()
    print(json.dumps(c["provenance"], indent=1, default=str))
