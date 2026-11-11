"""stack/skills/pregrasp.py -- PreGrasp: SEARCH for a two-hand pre-grasp pose (NYU hinged-jaw gripper).  SIMULATION ONLY, CPU.

Generalises democ/pregrasp_probe.py.  Inputs: WorldState (object pose + BoxShape from the ObjectState -- never the sim's truth), a SkillCall
(args: obj, faces, constraints), and a Sim used ONLY for IK / collision queries (sim.m, sim.d, sim.handles).
sim.handles keys read here:  "obj_geoms" {oid: geom name}, "obj_joint" {oid: free-joint name} (optional; puts the object at the ESTIMATED pose
before the collision queries), "benign_geoms" (geom names a contact with which is not a failure: object, support, floor).

Grasp feature (the thing a hand grips) = GraspFeature(name, side, pos, approach, jaw, thickness), all in the OBJECT frame.  A box gives two
(its +-y end regions, thickness = 2 hz); a pot handle is one more feature (frame + thickness).  Search parameters per candidate (all hands share
them, as in the probe): inset [m] (site depth inside the face along the approach), tilt_deg (approach pitched toward -z), roll_deg (jaw axis
about the approach), dz [m] (site height offset), open (jaw opening fraction).
Per-candidate checks (all MODEL, from the compiled MJCF):  ik (residual <= IK_TOL), free (open jaws: no robot contact + arm-arm h >= H_MIN),
grips (jaws fully closed: BOTH jaws of every active hand touch the object).  Score = min joint-limit margin [rad] over the best passing pair.
Search = coarse grid over (inset, tilt, roll, dz) -> pattern-search refinement around the best cells.  Why not CEM: a cell costs ~5 s (IK
dominates), the space is 4-D with a few discrete branches (jaw sign, IK solution) and the objective is non-smooth in the contact checks, so a
deterministic grid + coordinate pattern search is cheaper, reproducible and every evaluated cell is reported; `open` is not searched, it is
swept inside a cell (IK does not depend on it).
"""
from __future__ import annotations

import itertools
import multiprocessing as mp
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from world_model.v2.nero import adapters as A, config as C, gripper as G
from world_model.v2.nero.democ import kin as K
from world_model.v2.tray_rep.sets.ir import BoxShape
from ..types import BimanualPlan, Bound, Segment, SkillCall, Verdict, WorldState, SIDES

# ------------------------------------------------------------------------------------------------------------------ bounds (labelled)
IK_TOL = 1e-6        # MODEL: kin.collect_pose "ok" threshold on [pos m; rotvec rad] residual
H_MIN = 0.03         # ASSUMED (inherited from kin.find_pregrasp min_h, arm-arm proxy barrier h)
N_SOL, SEEDS = 6, 25 # ASSUMED search effort (as pregrasp_probe; fewer IK branches can only MISS solutions, never admit bad ones)
GRID = dict(inset=(0.0, 0.01, 0.02), tilt_deg=(0.0, 10.0, 20.0), roll_deg=(0.0,), dz=(0.0,))   # ASSUMED coarse grid (search range, not a world bound)
STEP = dict(inset=0.005, tilt_deg=5.0, roll_deg=5.0, dz=0.005)                                  # ASSUMED refinement steps
LIM = dict(inset=(-0.02, 0.05), tilt_deg=(0.0, 35.0), roll_deg=(-20.0, 20.0), dz=(-0.02, 0.02)) # ASSUMED hard limits of the search space
N_OPEN = 5           # ASSUMED number of opening fractions swept in [min needed, cap]
SEARCH_ASSUMPTIONS = {
    "ik_tol": Bound(IK_TOL, IK_TOL, "MODEL", "kin.collect_pose ok threshold"),
    "h_min": Bound(H_MIN, H_MIN, "ASSUMED", "kin.find_pregrasp min_h, arm-arm proxy barrier", "m"),
    "n_sol": Bound(N_SOL, N_SOL, "ASSUMED", "IK branches kept per side/jaw sign"),
    "ik_seeds": Bound(SEEDS, SEEDS, "ASSUMED", "IK restarts per side/jaw sign"),
    "grid_inset": Bound(min(GRID["inset"]), max(GRID["inset"]), "ASSUMED", "coarse grid, search range", "m"),
    "grid_tilt": Bound(min(GRID["tilt_deg"]), max(GRID["tilt_deg"]), "ASSUMED", "coarse grid, search range", "deg"),
    "clearance": Bound(0.002, 0.002, "ASSUMED", "nyu_gripper.GRASP_CLEARANCE_M per side (USER example 11-03)", "m"),
}
STAGES = ("ik", "contact", "arm_arm", "grip")      # order in which a candidate can fail; the furthest stage reached names the reason


@dataclass(frozen=True)
class GraspFeature:
    """one thing a hand can grip, in the OBJECT frame.  approach: unit vector the hand moves along (into the object); jaw: unit finger-closing axis;
    thickness: object extent along the jaw axis at the feature [m] (a pot handle: its diameter)."""
    name: str
    side: str
    pos: tuple
    approach: tuple
    jaw: tuple
    thickness: float
    geom: str | tuple | None = None   # MuJoCo geom(s) the jaws must touch when closed (None = handles["obj_geoms"][oid]); "body:<name>" = every geom of that body
    advance: float = 0.0              # [m] the candidate q is the STAND-OFF pose `advance` back along the approach; the closed-jaw test runs at the grasp pose
                                      # (PoseIK from q, orientation held).  0 = the candidate q is the grasp pose (handles, box ends)


def box_features(half) -> dict:
    """Demo C convention (kin.box_grasp_targets): left hand grips the +y end (approach -y), right the -y end; jaw axis = local z."""
    hx, hy, hz = (float(x) for x in half)
    return {"+y": GraspFeature("+y", "left", (0, hy, 0), (0, -1, 0), (0, 0, 1), 2 * hz),
            "-y": GraspFeature("-y", "right", (0, -hy, 0), (0, 1, 0), (0, 0, 1), 2 * hz)}


def handle_features(p, frac_left: float = 1.0, frac_right: float = 1.0) -> dict:
    """target adapter for HANDLES (pan: one, pot: two), in the pan BODY frame (origin = centre of the underside, scene._pan_local).
    Same convention as the box ends: left hand on the +axis handle (approach -axis, from the free end towards the pot), right hand on the -axis
    handle (approach +axis); jaw axis = z, so the gripped thickness = handle_t.  pos = the point `frac` of the handle length from the wall (default 1.0 = the free
    END face, like the box ends: with the NYU gripper the TCP must sit within ~5 cm of the tip, the palm hits a longer overlap); the search `inset`
    moves the TCP along the approach (towards the wall).  `p` = scene.PanParams (prior geometry, not sim truth).
    Returns {"handle": left, "handle2": right} (pan: only "handle").  # MODEL pan/scene._pan_local, handle2_local"""
    ax = np.asarray(p.handle_axis(), float)
    R0, t0 = p.pan_radius + p.pan_wall_t, p.pan_bottom_t + p.pan_wall_h - p.handle_t / 2
    fl = frac_left
    out = {"handle": GraspFeature("handle", "left", tuple(ax * (R0 + fl * p.handle_len) + [0, 0, t0]), tuple(-ax), (0.0, 0.0, 1.0), float(p.handle_t),
                                  geom="pan_handle")}
    if p.handle2_len > 0:
        fr = frac_right
        out["handle2"] = GraspFeature("handle2", "right", tuple(-ax * (R0 + fr * p.handle2_len) + [0, 0, t0]), tuple(ax), (0.0, 0.0, 1.0),
                                      float(p.handle_t), geom="pan_handle2")
    return out


def rim_features(p, stack_h: float, advance: float, lid_t: float) -> dict:
    """target adapter for the pan RIM clamp (right hand, from the far side): jaws close on the pan bottom + wall + lid stack of height `stack_h`
    at the -axis rim, approach = +axis, jaw z, mid-height of the stack.  pos in the pan body frame, rim outer surface.  The caller evaluates it
    with the pan at its planned POST-PUSH pose (the claw only reaches the rim after the left claw pushed).  # MODEL pan/scene._pan_local"""
    ax = np.asarray(p.handle_axis(), float)
    z_mid = 0.5 * (p.pan_bottom_t + p.pan_wall_h + lid_t)
    return {"rim": GraspFeature("rim", "right", tuple(-ax * (p.pan_radius + p.pan_wall_t) + [0, 0, z_mid]), tuple(ax), (0.0, 0.0, 1.0), float(stack_h),
                                geom=("body:pan", "body:lid"), advance=float(advance))}


def _quat_to_R(q):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(q, float))
    return R.reshape(3, 3)


def _rot(axis, ang):
    k = np.asarray(axis, float)
    k = k / np.linalg.norm(k)
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * Kx + (1 - np.cos(ang)) * Kx @ Kx


def gap_at_frac(frac: float) -> float:
    """inner jaw gap at the TCP [m] for an opening fraction (MODEL: gripper.jaw_gap_m via nyu THETA_POLY)."""
    ng = G._nyu()
    u = (1.0 - float(frac)) * ng.NUT_TRAVEL_M
    return float(G.jaw_gap_m(float(np.polyval(np.asarray(ng.THETA_POLY)[::-1], u))))


def grasp_target(obj_R, obj_p, feat: GraspFeature, cell: dict, jaw_sign: float):
    """(site position, gripper_base rotation) in the world for a feature + a candidate cell.  tilt rotates approach+jaw about
    approach x (-z) so the approach points down by tilt_deg (identical to pregrasp_probe.targets for the box)."""
    ap0 = obj_R @ np.asarray(feat.approach, float)
    jaw = obj_R @ np.asarray(feat.jaw, float) * jaw_sign
    ap = ap0
    tilt = np.radians(cell.get("tilt_deg", 0.0))
    ax = np.cross(ap0, [0.0, 0.0, -1.0])
    if tilt != 0.0 and np.linalg.norm(ax) > 1e-6:
        Rt = _rot(ax, tilt)
        ap, jaw = Rt @ ap0, Rt @ jaw
    roll = np.radians(cell.get("roll_deg", 0.0))
    jaw = np.cos(roll) * jaw + np.sin(roll) * np.cross(ap, jaw)
    pos = obj_p + obj_R @ np.asarray(feat.pos, float) + ap0 * cell.get("inset", 0.0) + np.array([0.0, 0.0, cell.get("dz", 0.0)])
    return pos, K.grasp_rot(ap, jaw)


# ------------------------------------------------------------------------------------------------------------------ worker context
_CTX: dict = {}


def _set_state(sides, q: dict, opens: dict):
    m, d = _CTX["m"], _CTX["d"]
    d.qpos[:] = _CTX["qpos0"]
    for s in SIDES:
        if s in q:
            qa, _ = A.arm_addrs(m, s)
            d.qpos[qa] = q[s]
            for i in range(7):
                d.ctrl[m.actuator(f"{s}_arm_joint{i+1}_pos").id] = q[s][i]
        if s in opens:
            G.set_opening(m, d, s, opens[s])
    mujoco.mj_forward(m, d)


def _bad_contacts():
    m, d = _CTX["m"], _CTX["d"]
    ok = _CTX["benign"]
    return [(m.geom(c.geom1).name, m.geom(c.geom2).name) for c in d.contact[:d.ncon]
            if not (m.geom(c.geom1).name in ok and m.geom(c.geom2).name in ok)]


def jaws_straddle(m, d, geom_id: int, sides) -> tuple[bool, dict]:
    """with the current d (jaws as set): do BOTH jaws of every side touch `geom_id` (an int, or {side: int} when each side grips its own geom)?"""
    gid = geom_id if isinstance(geom_id, dict) else {s: geom_id for s in sides}
    gid = {s: (set(v) if isinstance(v, (set, frozenset, list, tuple)) else {int(v)}) for s, v in gid.items()}
    jaws = {s: (set(G.col_geoms(m, s, "finger1")), set(G.col_geoms(m, s, "finger2"))) for s in sides}
    touch = {s: [False, False] for s in sides}
    for c in d.contact[:d.ncon]:
        g1, g2 = int(c.geom1), int(c.geom2)
        for s in sides:
            if g1 in gid[s]:
                g = g2
            elif g2 in gid[s]:
                g = g1
            else:
                continue
            for k in (0, 1):
                if g in jaws[s][k]:
                    touch[s][k] = True
    return all(all(v) for v in touch.values()), touch


def _eval_cell(cell: dict) -> dict:
    """IK once per (side, jaw sign), then every (jaw-sign, solution) pair x opening.  -> record with the best passing item (or the failing stage)."""
    m, d = _CTX["m"], _CTX["d"]
    sides, feats, R, p = _CTX["sides"], _CTX["feats"], _CTX["R"], _CTX["p"]
    rec = dict(cell=dict(cell), stage="ik", n_pairs=0, n_free=0, n_pass=0, best=None, fail_contacts={})
    sols = {}
    for s in sides:
        for js in (1.0, -1.0):
            pos, Rg = grasp_target(R, p, feats[s], cell, js)
            pos = pos - Rg[:, 2] * feats[s].advance          # stand-off pose (advance = 0: the grasp pose itself)
            sols[(s, js)] = K.collect_pose(_CTX["pik"][s], pos, Rg, N_SOL, SEEDS, seed=0 if s == "left" else 1, ok=IK_TOL)
    if not all(any(sols[(s, js)] for js in (1.0, -1.0)) for s in sides):
        return rec
    rec["stage"] = "contact"
    stage_i = 1
    opens = _CTX["opens"]
    options = [[(js, q, e) for js in (1.0, -1.0) for q, e in sols[(s, js)]] for s in sides]
    for combo in itertools.product(*options):
        q = {s: c[1] for s, c in zip(sides, combo)}
        rec["n_pairs"] += 1
        ok_opens = []
        hmin = np.inf
        for o in opens:
            _set_state(sides, q, {s: o for s in sides})
            bad = _bad_contacts()
            if bad:
                for pr in set(bad):
                    k = "|".join(pr)
                    rec["fail_contacts"][k] = rec["fail_contacts"].get(k, 0) + 1
                continue
            stage_i = max(stage_i, 2)
            h = _CTX["pb"].min_h(_CTX["d"]) if len(sides) == 2 else np.inf
            if h < H_MIN:
                continue
            ok_opens.append(o)
            hmin = min(hmin, h)
        if not ok_opens:
            continue
        rec["n_free"] += 1
        stage_i = max(stage_i, 3)
        qg = dict(q)
        for s, c in zip(sides, combo):                       # stand-off -> grasp pose along the approach (kinematic, orientation held)
            if feats[s].advance > 0:
                pos, Rg = grasp_target(R, p, feats[s], cell, c[0])
                qg[s], eg = _CTX["pik"][s].solve(pos, Rg, q[s], q_post=q[s], k_post=0.1, iters=120, tol=1e-9)
                if eg > 1e-4:                                # MODEL: grasp pose not reachable at the reference orientation (0.1 mm / 0.1 mrad)
                    qg = None
                    break
        if qg is None:
            continue
        _set_state(sides, qg, {s: 0.0 for s in sides})
        ok, touch = jaws_straddle(m, d, _CTX["geom"], sides)
        if not ok:
            continue
        rec["n_pass"] += 1
        jm = min(min(_CTX["pik"][s].margin(q[s])[0], _CTX["pik"][s].margin(qg[s])[0]) for s in sides)
        if rec["best"] is None or jm > rec["best"]["margin"]:
            o_mid = float(ok_opens[len(ok_opens) // 2])           # middle of the passing window = robust to opening error
            rec["best"] = dict(margin=float(jm), q={s: q[s].tolist() for s in sides}, jaw_sign={s: c[0] for s, c in zip(sides, combo)},
                               ik_res={s: float(c[2]) for s, c in zip(sides, combo)}, open=o_mid, ok_opens=[float(x) for x in ok_opens],
                               h=None if not np.isfinite(hmin) else float(hmin))
    rec["stage"] = STAGES[stage_i] if rec["best"] is None else "pass"
    return rec


def _key(cell):
    return tuple(round(cell[k], 6) for k in ("inset", "tilt_deg", "roll_deg", "dz"))


# ------------------------------------------------------------------------------------------------------------------ result
@dataclass
class PreGraspResult:
    ok: bool
    reason: str
    best: dict | None
    ranked: list = field(default_factory=list)          # passing cells, best first
    cells: list = field(default_factory=list)           # every evaluated cell (record without the q arrays for failures)
    wall_s: float = 0.0
    n_cells: int = 0
    sides: tuple = ()
    feature: dict = field(default_factory=dict)
    assumptions: dict = field(default_factory=dict)


def _margins(best, feat_thick, sides):
    gap = gap_at_frac(best["open"])
    return dict(joint_limit_rad=best["margin"], ik_residual_slack=IK_TOL - max(best["ik_res"].values()),
                arm_arm_h_slack_m=None if best["h"] is None else best["h"] - H_MIN, open_window_frac=float(max(best["ok_opens"]) - min(best["ok_opens"])),
                jaw_gap_over_thickness_m=gap - max(feat_thick[s] for s in sides))


class PreGrasp:
    name = "pregrasp"

    def __init__(self):
        self._sim = None
        self._last: PreGraspResult | None = None

    # ------------------------------------------------------------- protocol
    def pre(self, ws: WorldState, call: SkillCall):
        oid = call.args.get("obj")
        o = ws.objects.get(oid)
        if o is None:
            return False, f"object {oid!r} not in the WorldState"
        if not o.visible:
            return False, f"object {oid!r} is not currently observed (visible=False)"
        pose = ws.world_pose(oid)
        if not (np.all(np.isfinite(pose.p)) and np.all(np.isfinite(pose.R))):
            return False, f"object {oid!r} pose unknown (non-finite)"
        if call.args.get("features") is None and not isinstance(o.shape, BoxShape):
            return False, f"shape {type(o.shape).__name__} has no default grasp features (pass args['features'])"
        return True, ""

    def post(self, ws: WorldState, call: SkillCall):
        """jaws straddle the object: re-run the closed-jaw test at the executed (ws.robot.q, if present) else the planned q."""
        if self._last is None or self._sim is None or not self._last.ok:
            return False, "no successful PreGrasp plan to check"
        oid = call.args["obj"]
        sim, sides = self._sim, self._last.sides
        q = {s: np.asarray(ws.robot.q[s]) if s in ws.robot.q and len(ws.robot.q[s]) == 7 else np.asarray(self._last.best["q"][s]) for s in sides}
        self._setup(sim, ws, oid, sides, self._last.feature)
        _set_state(sides, q, {s: 0.0 for s in sides})
        ok, touch = jaws_straddle(_CTX["m"], _CTX["d"], _CTX["geom"], sides)
        return ok, "" if ok else f"jaws do not both touch the object when closed: {touch}"

    def plan(self, ws: WorldState, call: SkillCall, planner, sim) -> BimanualPlan:
        res = self.search(ws, call, sim)
        prov = dict(skill=self.name, planner=getattr(planner, "name", None), ok=res.ok, reason=res.reason, wall_s=res.wall_s, n_cells=res.n_cells,
                    ranked=[dict(cell=r["cell"], margin=r["best"]["margin"], open=r["best"]["open"], n_pass=r["n_pass"]) for r in res.ranked[:10]],
                    stages={st: sum(1 for c in res.cells if c["stage"] == st) for st in STAGES + ("pass",)},
                    object_unc_used=False)        # the object-pose uncertainty (ObjectState.unc) is NOT inflated into the checks yet
        seg = []
        if res.ok:
            b = res.best
            sides = res.sides
            thick = {s: res.feature[s]["thickness"] for s in sides}
            marg = _margins(b, thick, sides)
            verdict = Verdict("pregrasp", True, "sampled", marg, {k: v.__dict__ for k, v in res.assumptions.items()},
                              "single-configuration MODEL checks; IK branches/seeds sampled")
            qt = {s: np.asarray(b["q"][s]) for s in sides}
            seg = [Segment("pregrasp", 0.0, 0.0, q_ref=lambda t, qt=qt: {s: (qt[s], np.zeros(7), np.zeros(7)) for s in qt},
                           gripper={s: b["open"] for s in sides},
                           meta=dict(q_target={s: b["q"][s] for s in sides}, cell=res.ranked[0]["cell"], jaw_sign=b["jaw_sign"],
                                     margins=marg, verdict=verdict, ik_res=b["ik_res"]))]
        return BimanualPlan(seg, "pregrasp_search", prov, {k: v.__dict__ for k, v in res.assumptions.items()})

    # ------------------------------------------------------------- search
    def _setup(self, sim, ws, oid, sides, feats_dict, opens=None, benign=None):
        m = sim.m
        d = mujoco.MjData(m)
        d.qpos[:] = sim.d.qpos
        pose = ws.world_pose(oid)
        R, p = np.asarray(pose.R, float), np.asarray(pose.p, float)
        h = sim.handles
        gname = h["obj_geoms"][oid]
        jn = h.get("obj_joint", {}).get(oid)
        if jn is not None:                                   # put the object where the ESTIMATE says it is
            ba = m.jnt_qposadr[m.joint(jn).id]
            att = [m.jnt_qposadr[m.joint(a).id] for a in h.get("obj_attached_joints", {}).get(oid, ())]   # free bodies riding on it (the pan's lid)
            Ro_, po_ = np.zeros(9), d.qpos[ba:ba + 3].copy()
            mujoco.mju_quat2Mat(Ro_, d.qpos[ba + 3:ba + 7].copy())
            Ro_ = Ro_.reshape(3, 3)
            Dm = R @ Ro_.T                                   # rigid move old pose -> estimate, applied to the attached bodies too
            for aq in att:
                pa, qa = d.qpos[aq:aq + 3].copy(), d.qpos[aq + 3:aq + 7].copy()
                Ra = np.zeros(9)
                mujoco.mju_quat2Mat(Ra, qa)
                Rn = Dm @ Ra.reshape(3, 3)
                d.qpos[aq:aq + 3] = p + Dm @ (pa - po_)
                mujoco.mju_mat2Quat(qa, Rn.reshape(-1))
                d.qpos[aq + 3:aq + 7] = qa
            q4 = np.zeros(4)
            mujoco.mju_mat2Quat(q4, R.reshape(-1))
            d.qpos[ba:ba + 3], d.qpos[ba + 3:ba + 7] = p, q4
        mujoco.mj_forward(m, d)
        feats = {s: GraspFeature(**{**f, "pos": tuple(f["pos"]), "approach": tuple(f["approach"]), "jaw": tuple(f["jaw"])}) if isinstance(f, dict) else f
                 for s, f in feats_dict.items()}
        benign = set(h.get("benign_geoms", K.NAMES_OK)) | {gname}
        if h.get("benign_non_robot"):      # every geom outside the robot subtree (pan scene: world, pan, lid, hidden contents)
            robot_root = m.body(h["benign_non_robot"]).id
            benign |= {m.geom(g).name for g in range(m.ngeom) if m.body_rootid[m.geom_bodyid[g]] != robot_root and m.geom(g).name}
        def _ids(spec):
            out = set()
            for g in ((spec,) if isinstance(spec, str) else tuple(spec)):
                if g.startswith("body:"):
                    b = m.body(g[5:]).id
                    out |= {i for i in range(m.ngeom) if m.geom_bodyid[i] == b}
                else:
                    out.add(m.geom(g).id)
            return out
        fgeo = {s: _ids(f.geom or gname) for s, f in feats.items()}
        for f in feats.values():                             # the gripped geoms count as object geoms (no "bad contact" for them vs the support)
            benign |= {m.geom(i).name for i in _ids(f.geom or gname)}
        _CTX.update(m=m, d=d, qpos0=d.qpos.copy(), sides=tuple(sides), feats=feats, R=R, p=p, geom=fgeo,
                    benign=benign, opens=opens or [1.0],
                    pb=A.PairBarriers(m), pik={s: K.PoseIK(m, s) for s in sides})

    def search(self, ws: WorldState, call: SkillCall, sim, procs: int | None = None) -> PreGraspResult:
        t0 = time.time()
        a, cons = call.args, dict(call.args.get("constraints", {}))
        oid = a["obj"]
        ok, why = self.pre(ws, call)
        if not ok:
            return PreGraspResult(False, "pre: " + why, None, wall_s=time.time() - t0)
        o = ws.objects[oid]
        sides = tuple(s for s in call.sides if s in SIDES)
        feats_all = a.get("features") or box_features(o.shape.half)
        faces = a.get("faces") or list(feats_all)
        feats = {}
        for f in faces:
            if f in feats_all and feats_all[f].side in sides:
                feats[feats_all[f].side] = feats_all[f]
        if set(feats) != set(sides):
            miss = sorted(set(sides) - set(feats))
            return PreGraspResult(False, f"no allowed grasp face for side(s) {miss} (faces={faces})", None, wall_s=time.time() - t0)
        # gripper opening range: need gap >= thickness + 2 clearance, capped by the request
        cap = float(cons.get("open_cap", 1.0))
        clr = float(cons.get("clearance", SEARCH_ASSUMPTIONS["clearance"].lo))
        thick = max(f.thickness for f in feats.values())
        if gap_at_frac(cap) < thick + 2 * clr:
            return PreGraspResult(False, f"jaw opening cap {cap:.2f} gives gap {gap_at_frac(cap)*1e3:.1f} mm < object thickness "
                                  f"{thick*1e3:.1f} mm + 2x{clr*1e3:.0f} mm clearance: the jaws cannot straddle it", None, wall_s=time.time() - t0, sides=sides)
        wf = G.opening_for_width(thick, clearance=clr)
        lo_open = min(cap, float(wf["frac"]))
        opens = [float(x) for x in np.linspace(lo_open, cap, N_OPEN)] if cap > lo_open + 1e-9 else [cap]
        if cons.get("opens"):                                  # caller pins the opening(s) (e.g. the executor's approach opening)
            opens = [float(x) for x in cons["opens"]]
        grid = {**GRID, **cons.get("grid", {})}
        procs = int(cons.get("procs", 1)) if procs is None else procs
        rounds = int(cons.get("refine_rounds", 3))
        lim = {**LIM, **cons.get("lim", {})}
        self._setup(sim, ws, oid, sides, feats, opens)
        self._sim = sim

        def run(cells):
            if procs > 1 and len(cells) > 1:
                with mp.get_context("fork").Pool(min(procs, len(cells)), maxtasksperchild=3) as pool:
                    return pool.map(_eval_cell, cells, chunksize=1)
            return [_eval_cell(c) for c in cells]

        seen: dict = {}

        def add(cells):
            new = [c for c in cells if _key(c) not in seen]
            for c, r in zip(new, run(new)):
                seen[_key(c)] = r

        add([dict(inset=i, tilt_deg=t, roll_deg=r, dz=z) for i, t, r, z in itertools.product(grid["inset"], grid["tilt_deg"], grid["roll_deg"], grid["dz"])])
        score = lambda r: -1e9 if r["best"] is None else r["best"]["margin"]
        mult = 1.0
        for _ in range(rounds):
            top = max(seen.values(), key=score)
            if top["best"] is None:
                break                                          # nothing passes yet -> refinement has no gradient; the grid is the answer
            nb = []
            for k in STEP:
                for sgn in (-1, 1):
                    c = dict(top["cell"])
                    c[k] = float(np.clip(c[k] + sgn * mult * STEP[k], *lim[k]))
                    nb.append(c)
            before = score(top)
            add(nb)
            if score(max(seen.values(), key=score)) <= before + 1e-9:
                mult *= 0.5                                    # no improvement: halve the step
        cells = list(seen.values())
        ranked = sorted([r for r in cells if r["best"] is not None], key=lambda r: -r["best"]["margin"])
        fe = {s: dict(name=f.name, side=f.side, pos=f.pos, approach=f.approach, jaw=f.jaw, thickness=f.thickness, geom=f.geom) for s, f in feats.items()}
        asm = dict(SEARCH_ASSUMPTIONS)
        asm["open_cap"] = Bound(cap, cap, "ASSUMED", "request constraint", "frac")
        wall = time.time() - t0
        if ranked:
            res = PreGraspResult(True, "", ranked[0]["best"], ranked, cells, wall, len(cells), sides, fe, asm)
        else:
            far = max(STAGES.index(r["stage"]) for r in cells)
            ncls = {st: sum(1 for r in cells if r["stage"] == st) for st in STAGES}
            top_c = {}
            for r in cells:
                for k, v in r["fail_contacts"].items():
                    top_c[k] = top_c.get(k, 0) + v
            top_c = dict(sorted(top_c.items(), key=lambda kv: -kv[1])[:3])
            msg = {0: "no IK solution for the grasp targets (residual > %.0e) in any cell" % IK_TOL,
                   1: "every IK pair has a robot contact with open jaws (top: %s)" % top_c,
                   2: "every contact-free pair violates arm-arm h >= %.2f m" % H_MIN,
                   3: "no pair has both jaws touching the object when closed"}[far]
            res = PreGraspResult(False, f"{msg}; furthest stage '{STAGES[far]}', cells per stage {ncls}", None, [], cells, wall, len(cells), sides, fe, asm)
        self._last = res
        return res
