"""adapters/rule1_contents.py -- the physics callables of Rule 1 (stack/checks/rule1.admit_step) for the contents carry tasks.  SIMULATION ONLY.

Tasks: box, tray2/4/6, box_behind, tray6_behind (not pot).  Planner: carry.resolve_planner() (env BIMANUAL_PLANNER, default "scripted"); the table-behind tasks
always use the closed_chain default WaypointPlan (they have no scripted plan).

What is checked (all STATIC rows are proof under MODEL = the rows of stack/checks/admit: torque / payload u <= U_ADMIT of the lever split on the arm
configurations of the PLANNED path, open-content limit, support at the landing; guarantee text in checks/payload.GUARANTEE):
  stop      the stored backup's ("stop", T) phase: deceleration |v_branch| / T (v from the planned object path at the branch time) at the branch pose and the
            samples just beyond it (the stop overshoots by 0.5 v T), then held still
  retrace   the stored backup's retrace / advance: object acceleration A_BACKUP (rest to rest per segment) over the path samples between the branch and the landing
  landing   S = support row at the landing (set down) or torque at the hold poses (held); R_K = a CONSTRUCTIVE WITNESS: the planner's remaining plan admitted by
            checks/admit.admit on the landing's theta set (depth 1 <= K, 1 node <= NODE_CAP); a landing with no witness is a MISSING row, never a pass
Programs per step (D4): `stored_backup` (the fixed one-bin program = the planner's stored backup) and, when the backup moves the object,
`read_then_retrace` (prefix = stop + hold, reading = claw load at hold, bins = equal-width total-mass bins, plan = the stored retrace at the highest acceleration
admitted on that bin, capped at A_BACKUP).  The theta set is the TOTAL carried mass (field `total_kg`, kg), the quantity the claw-load reading measures.

    python -m world_model.v2.adapters.rule1_contents --task box --cells 1.0-1.5,4.5-5.0 [--n-sampled 0] --out DIR
(cells in the prior's unit of carry_cert.cells(task): box = contents kg, tray = fill per mug kg).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import pathlib
import time

import numpy as np

from world_model.v2.adapters import collision_rows as RCOL
from world_model.v2.adapters import contents_specs as CS
from world_model.v2.adapters import payload_envelope as PE
from world_model.v2.nero.contents import carry as CR
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.nero.contents import envelope as E
from world_model.v2.nero.contents import objects as OB
from world_model.v2.nero.contents import scenes as SC
from world_model.v2.nero.contents import thresholds as THR
from world_model.v2.stack import executor as EX
from world_model.v2.stack import tasks as TK
from world_model.v2.stack import types as TY
from world_model.v2.stack import theta as TH
from world_model.v2.stack.checks import admit as AD
from world_model.v2.stack.checks import collision as KC
from world_model.v2.stack.checks import dynamic as DY
from world_model.v2.stack.checks import open_content as OCN
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks import base_stop as BS
from world_model.v2.stack.checks import fixability as FX
from world_model.v2.stack.checks import rule1 as R1
from world_model.v2.stack.checks import support as SUP
from world_model.v2.stack.checks.row import ORDER, Row, missing

TOTAL_FIELD = "total_kg"          # the theta field: TOTAL carried mass (empty object + contents) [kg] = what the claw-load reading measures (probe.py)
BIN_WIDTH_KG = 0.5                # ASSUMED (to be raised to the user): widest reading bin; the bins are EQUAL width, n = ceil(range / BIN_WIDTH_KG), so each is <= this
N_STATIC = 6                      # path samples per move (= closed_chain.N_STATIC = carry_cert.N_PER_MOVE; the static admission's samples)
A_BACKUP = CR.A_BACKUP            # ASSUMED 0.25 m/s^2 (carry.A_BACKUP): the stored backup's retrace acceleration
A_MIN = AD.A_MIN                  # ASSUMED 0.05 m/s^2 (checks/admit): a retrace slower than this = the bin is refused (the recovery stalls)
# stop time when a stored backup does not start with ("stop", T) (none does today): TY.STOP_T_S (ASSUMED, one source in stack/types.py, read at call time)
READ_PRODUCER = "claw_load"
READ_HOLD_S = 0.4                 # ASSUMED (11-17): the hold before the in-carry claw-load read [s] = probe.T_HOLD (reused); the executor holds still and reads over the last N_READ_RT steps
N_READ_RT = 40                    # ASSUMED (11-17): control steps averaged by the read (0.2 s) = probe.N_READ (reused)
HONEST = dict(between_samples="not bounded (samples only)", branch_grid="0.25 s grid; triggers between grid points not checked")      # what a "proof" label here does NOT cover


# ----------------------------------------------------------------------------------------------------------------------- small helpers
def _merge(name: str, rows: list) -> Row:
    """several rows of one motion / landing -> one Row: ok = all ok; guarantee = weakest (a failed non-missing row = refuted); margin = the smallest margin."""
    bad = [r for r in rows if not r.ok]
    if bad:
        g = "refuted" if any(r.guarantee != "missing" for r in bad) else "missing"
    else:
        g = min((r.guarantee for r in rows), key=ORDER.index) if rows else "proof"
    ms = [r.margin for r in rows if r.margin is not None]
    ns = [r.n for r in rows if r.n]
    worst = {}
    for r in rows:
        d = dict(ok=r.ok, margin=r.margin, unit=r.unit, guarantee=r.guarantee)
        if r.why:
            d["why"] = r.why
        if isinstance(r.worst, dict):
            d.update({k: r.worst[k] for k in ("u_max", "a", "n_samples", "m", "c", "wp", "a_open", "a_plan", "action", "a_admit", "depth", "nodes", "backup_retreat") if k in r.worst})
        if r.n:
            d.update(n=r.n, eps_upper=r.eps_upper)
        worst[r.name] = d
    return Row(name, not bad, margin=(min(ms) if ms else None), unit="mixed", guarantee=g, worst=worst, n=(min(ns) if ns else None),
               assumptions={r.name: r.assumptions for r in rows}, why="; ".join(f"{r.name}: {r.why}" for r in bad if r.why))


def _hull(theta: TH.ThetaSet) -> tuple:
    """(lo, hi) of the total mass over the cells (a union of cells is checked on its hull: a sound superset)."""
    lo = min(c.bounds[TOTAL_FIELD].lo for c in theta.cells)
    hi = max(c.bounds[TOTAL_FIELD].hi for c in theta.cells)
    return float(lo), float(hi)


class NoCalibration(Exception):
    """the claw_guard calibration has no entry for this task: the reading program is not offered (never another task's calibration)."""


def claw_read_spec(task: str, nb: float) -> TH.ReadingSpec:
    """the claw_guard read of the TOTAL carried mass: y = M + n, |n| <= EPS_FREE + nb.
      EPS_FREE  MEASURED: probe_calib_claw_guard_2026-11-12/CALIB.json eps.EPS_FREE (max over the calibration worlds of every calibrated task of the noise-free read residual,
                free-hanging at the probe pose).  Its transfer to a mid-carry read at another pose is ASSUMED.
      nb        ASSUMED: the probe's propagated joint-torque noise, TAU_NOISE (0.05 Nm per joint, ASSUMED, probe.py) through |pinv(J^T)_z| of both arms, summed over joints and arms,
                = probe.read()'s nb formula, here the MAX over this task's path samples (Ctx.read_nb).
    The calibration must contain the task (per_task): box, tray6, pot today; tray2 / tray4 / *_behind have none -> NoCalibration."""
    from world_model.v2.nero.contents import probe as PR
    path = pathlib.Path(PR.NEW_CALIB["claw_guard"])
    cal = json.loads(path.read_text())
    if task not in cal.get("per_task", {}):
        raise NoCalibration(f"{path}: no claw_guard calibration for task {task!r} (has {sorted(cal.get('per_task', {}))}); the reading program is not offered")
    eps = float(cal["eps"]["EPS_FREE"])
    return TH.ReadingSpec(producer=READ_PRODUCER, statistic="vertical load at hold from joint torques [kg]", coef={TOTAL_FIELD: 1.0}, offset=0.0, noise=eps + float(nb),
                          noise_tag="ASSUMED", noise_src=f"EPS_FREE {eps} kg MEASURED ({path} eps.EPS_FREE, free-hanging at the probe pose; transfer to a mid-carry read ASSUMED) "
                                                         f"+ nb {float(nb):.4f} kg ASSUMED (probe.TAU_NOISE {PR.TAU_NOISE} Nm ASSUMED, propagated through |pinv(J^T)_z|, max over this task's path samples)")


def cell_total(task: str, cell) -> tuple:
    """a carry_cert cell (prior unit: box = contents kg, tray = fill per mug kg) -> the total carried mass interval [kg]."""
    td = TK.task(task)
    em = CS.empty_props(task)[0]
    lo, hi = cell
    if td.n_cups:
        return float(em + td.n_cups * (OB.CUP["empty"] + lo)), float(em + td.n_cups * (OB.CUP["empty"] + hi))
    return float(em + lo), float(em + hi)


def total_to_cell(task: str, lo: float, hi: float) -> tuple:
    """inverse of cell_total over an interval (tray: per-mug fill hull, clipped to the fill range)."""
    td = TK.task(task)
    em = CS.empty_props(task)[0]
    if td.n_cups:
        f = lambda M: float(np.clip((M - em) / td.n_cups - OB.CUP["empty"], 0.0, OB.CUP["fill_max_kg"]))
        return f(lo), f(hi)
    return float(lo - em), float(hi - em)


def theta_box(lo: float, hi: float, source="prior") -> TH.ThetaSet:
    return TH.ThetaSet.box(source=source, **{TOTAL_FIELD: (lo, hi)})


# ----------------------------------------------------------------------------------------------------------------------- recovery motion reference
@dataclasses.dataclass(frozen=True)
class MotionRef:
    """the opaque `plan` of a RecoveryProgram bin / prefix.
    kind: none | stop_hold (stop, hold in place) | stop_back (stop, retrace back to the start of `target`) | stop_adv (stop, advance to the end of `target`)
          (the three = the planner's STORED backups) | read_hold (stop + hold to read: the prefix of a reading program)
          | back_at / adv_at (retrace / advance only, at object acceleration `a`: a bin plan) | refuse (no admissible acceleration on this bin)"""
    kind: str
    seg: str
    target: str | None = None
    a: float | None = None
    stop_T: float = dataclasses.field(default_factory=lambda: TY.STOP_T_S)      # call-time read of the single ASSUMED stop time
    note: str = ""
    segs: "tuple | None" = None   # 11-17: the stored backup's FULL retrace / advance segment tuple (the executor runs it); None = not recorded (target is its last element)


@dataclasses.dataclass
class Br:
    """one branch point of a segment of the planned path."""
    t: float
    v: float                     # object speed [m/s] at the branch (planned path)
    key: str                     # path sample at the branch pose
    ahead: str | None            # path sample at the END of the stop: the first sample at or beyond 0.5 v T along the path (continuing into the next segments' samples);
                                 # None = the overshoot runs past the last planned sample (no checked pose exists: the motion rows are MISSING)
    idx_b: int | None = None     # closed_chain: geometry sample indices of key / ahead (the collision rows walk every index in between)
    idx_e: int | None = None
    finish: bool = False         # 11-16: the branch is in the last stop_T seconds of its segment -> the stop FINISHES the planned segment (executor: the reference finishes at rest);
                                 # `ahead` = the segment's last path sample
    a_stop: float | None = None  # finish only: the stop's deceleration = the plan's own acceleration `a` (the planned remainder's acceleration cap)
    ext: dict | None = None      # 11-17 (USER "check the overshoot pose"): the stop's overshoot runs past the last planned sample -> the path is EXTENDED by a straight-line continuation of the
                                 # last segment (MODEL) with both arms' SRS IK at the end: dict(ok, excess, idx [virtual geometry indices], keys, i_last, key_last, why).  ok False = no in-limit IK
                                 # solution (a refuted row).  None: no extension (planned samples cover the stop, or the extension is switched off / not available).


@dataclasses.dataclass
class PathModel:
    a: float
    names: list
    offs: dict                   # key -> object offset from S [m] (every key of `cap`)
    cap: dict                    # arm configurations per key (envelope.Arms format)
    static: dict                 # seg -> keys of the static admission's samples (N_STATIC + 1; the shared boundary key repeats)
    order: list                  # every key along the whole path (static + branch + stop-end samples)
    branches: dict               # seg -> [Br]
    T: dict                      # seg -> duration of the move [s]
    disp: np.ndarray             # object displacement S -> goal [m]
    desc: str = ""
    cc: object = None            # closed_chain: the ClosedChainPlan (collision rows); geo: its Geometry
    geo: object = None
    ext_store: dict = dataclasses.field(default_factory=dict)      # virtual geometry index (>= len(geo.offs)) -> dict(q={left,right}, off, i_last): the extended samples of Br.ext

    def __post_init__(self):
        self._pos = {k: i for i, k in enumerate(self.order)}

    def path_samples(self, first=0) -> AD.PathSamples:
        """the static admission's samples of segments names[first:] (checks/admit.PathSamples)."""
        per, order = {}, []
        for s in self.names[first:]:
            per[s] = list(self.static[s])
            order += [k for k in self.static[s] if k not in order]
        return AD.PathSamples({k: self.offs[k] for k in order}, per, self.cap, np.asarray(self.disp, float))

    def br(self, seg, t) -> Br:
        return min(self.branches[seg], key=lambda b: abs(b.t - t))

    def _span(self, k0, k1) -> list:
        return self.order[self._pos[k0]:self._pos[k1] + 1]

    def _ext_ok(self, br: Br) -> bool:
        return br.ext is not None and bool(br.ext.get("ok"))

    def keys_back(self, seg, br: Br, target) -> list:
        """every sample from the END of the stop back to the start of `target` (retrace): the static samples plus every branch / stop-end sample in between (+ the extended samples)."""
        if self._ext_ok(br):
            return self._span(self.static[target][0], br.ext["key_last"]) + list(br.ext["keys"])
        return self._span(self.static[target][0], br.ahead)

    def keys_adv(self, seg, br: Br, target) -> list:
        """every sample from the branch pose forward to the end of `target` (advance); the extended samples of an overshoot past the last sample are added."""
        return self._span(br.key, self.static[target][-1]) + (list(br.ext["keys"]) if self._ext_ok(br) else [])

    def keys_stop(self, seg, br: Br) -> list:
        """every sample from the branch pose to the END of the stop (through the extended samples when the overshoot runs past the last planned one)."""
        if self._ext_ok(br):
            return self._span(br.key, br.ext["key_last"]) + list(br.ext["keys"])
        return self._span(br.key, br.ahead)


class _SharedBackend:
    """payload_envelope.EnvelopeBackend with ONE Arms per capture (Arms builds the MJCF terms of every key; the rows below reuse it)."""

    def __init__(self):
        self._arms = {}

    def o_ref(self):
        return PE.o_ref()

    def make_envelope(self, spec, cap, offs):
        hit = self._arms.get(id(cap))
        if hit is None or hit[0] is not cap:
            hit = (cap, E.Arms(cap))
            self._arms[id(cap)] = hit
        return E.Envelope(spec, hit[1], PE.o_ref(), np.eye(3), ["left", "right"], grip_mass=PE.caps()["pot"]["gripper_mass"], offsets=offs)


# ----------------------------------------------------------------------------------------------------------------------- the task context
class Ctx:
    """everything physics for one task: the world, the planned path samples with branch points, the envelope, the support geometry and the caches."""

    def __init__(self, task: str, planner: str | None = None, wp=None):
        self.task = task
        self.td = TK.task(task)
        if self.td.kind == "pot":
            raise ValueError("rule1_contents: the pot is not built here (box, tray, *_behind)")
        self.planner = "closed_chain" if (self.td.base or wp is not None) else CR.resolve_planner(planner)
        self.em, self.ec = CS.empty_props(task)
        self.backend = _SharedBackend()
        self.inp = None
        if self.planner == "closed_chain":
            from world_model.v2.stack.planners import closed_chain as CCP
            self.CCP = CCP
            self.inp = CCP._nominal_inputs(task, wp)          # wp: a WaypointPlan in place of the task default (z3 oracle); None = the default plan
            self.world = self.inp.world
            self.wp = self.inp.wp
        else:
            self.CCP = None
            self.world = CC.world_default(task)
            self.wp = None
        self.models, self._envs, self._wit, self._specs, self._memo, self._recs = {}, {}, {}, {}, {}, {}
        self.cell_spec = {}              # (lo, hi) total interval -> the carry_cert cell ObjectSpec (the K2-consistent spec when theta is exactly that cell)
        self.n_sampled, self.procs = 0, 1

    # ------------------------------------------------------------------ specs
    def spec_for(self, lo, hi):
        k = (round(lo, 6), round(hi, 6))
        if k in self.cell_spec:
            return self.cell_spec[k]
        if k not in self._specs:
            self._specs[k] = CS.spec_from_total(self.task, max(lo - self.em, 0.0), hi - self.em)
        return self._specs[k]

    def register_cell(self, cell, lo, hi):
        self.cell_spec[(round(lo, 6), round(hi, 6))] = CS.spec_from_cell(self.task, *cell)

    # ------------------------------------------------------------------ plan model
    def backup(self, seg):
        if self.planner == "closed_chain":
            return self.CCP.backup_phases_for(seg, self.wp)
        from world_model.v2.stack.planners import scripted as SP
        return SP.backup_phases(seg)

    def _stop_T(self, seg):
        ph = self.backup(seg)
        return float(ph[0][1]) if ph and ph[0][0] == "stop" else TY.STOP_T_S

    def model(self, a: float) -> PathModel:
        k = round(float(a), 6)
        if k not in self.models:
            self.models[k] = self._model_cc(k) if self.planner == "closed_chain" else self._model_scripted(k)
        return self.models[k]

    def _post_of(self, g, seg, i) -> bool:
        """is geometry sample i after the base move of a table-behind plan (the moving segment wins at a shared boundary index)?"""
        if self.wp is None or not self.wp.base:
            return False
        k = self.wp.base[0][0]
        a, b = g.bounds[seg]
        sg = seg if a <= i <= b else next(s_ for s_, (x, y) in g.bounds.items() if x <= i <= y)
        return list(g.bounds).index(sg) >= k

    def _model_cc(self, a) -> PathModel:
        CCP = self.CCP
        cc = CCP.plan_for(self.inp, a)
        g = cc.geo
        names = list(g.bounds)
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(g.offs, axis=0), axis=1))]        # arc length along the whole planned path, per geometry sample
        n_all = len(g.offs)
        static, need = {}, set()
        for sg in names:
            i0, i1 = g.bounds[sg]
            static[sg] = [int(i) for i in np.unique(np.round(np.linspace(i0, i1, CCP.N_STATIC + 1)).astype(int))]
            need |= set(static[sg])
        idx_br, T, ext_store = {}, {}, {}
        for sg in names:
            i0, i1 = g.bounds[sg]
            pc = cc.pieces[sg]
            n, sT = pc.n, self._stop_T(sg)
            vel = np.linalg.norm(np.gradient(pc.p, CCP.DT, axis=0), axis=1)
            T[sg] = n * CCP.DT
            lst = []
            for t in R1.branch_times(T[sg]):
                kk = min(n, int(round(t / CCP.DT)))
                ib = i0 + int(np.argmin(np.linalg.norm(g.offs[i0:i1 + 1] - pc.p[kk], axis=1)))
                v = float(vel[kk]) if kk < n else 0.0
                fin, ext = False, None
                if v <= 1e-9:
                    ie = ib
                elif (not EX.legacy_stop()) and 1e-9 < T[sg] - t <= sT + 1e-9:
                    ie, fin = i1, True       # 11-16: the stop finishes the segment (executor stop branch); end of stop = the segment's last geometry sample
                else:
                    # END of the stop: first sample at or beyond 0.5 v T (+ DS/2: the nearest sample may sit up to half a spacing behind the branch pose), continuing into the next
                    # segments' samples; None = beyond the last planned sample
                    want = arc[ib] + 0.5 * v * sT + 0.5 * CCP.DS
                    ie = next((i for i in range(ib, n_all) if arc[i] >= want), None)
                    if ie is not None and self._post_of(g, sg, ie) != self._post_of(g, sg, ib):
                        ie = None            # the overshoot would run through the base move: no planned sample at the end of the stop
                    if ie is None and os.environ.get("BIMANUAL_LEGACY_NO_ARM_EXT") != "1":
                        ext = self._extend_stop(g, sg, ib, want, arc, ext_store, n_all)       # 11-17: check the overshoot pose (straight-line continuation + both arms' IK)
                if ext is not None and ext["ok"]:
                    need |= set(range(ib, ext["i_last"] + 1)) | set(ext["idx"])
                else:
                    need |= {ib} | (set(range(ib, ie + 1)) if ie is not None else set())
                lst.append((float(t), v, ib, ie, fin, ext))
            idx_br[sg] = lst
        key = lambda i: f"g{i}"
        gq = lambda i, s_: np.asarray(ext_store[i]["q"][s_] if i >= n_all else g.q[s_][i], float).tolist()
        q = {key(i): {s: gq(i, s) for s in ("left", "right")} for i in sorted(need)}
        cap = dict(PE.caps()["pot"], q=q)
        offs = {key(i): np.asarray(ext_store[i]["off"] if i >= n_all else g.offs[i], float) for i in sorted(need)}
        branches = {}
        for sg in names:
            bl = []
            for t, v, ib, ie, fin, ext in idx_br[sg]:
                if ext is not None and ext["ok"]:
                    ext = dict(ext, keys=[key(i) for i in ext["idx"]], key_last=key(ext["i_last"]))
                    bl.append(Br(t, v, key(ib), ext["keys"][-1], ib, ext["idx"][-1], fin, None, ext))
                else:
                    bl.append(Br(t, v, key(ib), None if ie is None else key(ie), ib, ie, fin, (float(a) if fin else None), ext))
            branches[sg] = bl
        return PathModel(a, names, offs, cap, {s: [key(i) for i in static[s]] for s in names}, [key(i) for i in sorted(need)], branches, T, np.asarray(g.offs[-1], float),
                         desc="closed_chain: planned joint path (SRS IK + TOPP-RA), samples at the planned configurations", cc=cc, geo=g, ext_store=ext_store)

    def _extend_stop(self, g, seg, ib, want, arc, ext_store, n_all) -> dict:
        """USER 11-17 "check the overshoot pose".  The stop's end (arc length `want` along the planned path) is past the last planned sample on this side of the base move: continue the
        LAST segment's direction in a straight line by the excess (MODEL), subdivided at DS, solve both arms' SRS IK (closed_chain._nodes: every in-limit solution over the arm-angle grid and
        both elbow roots) at the end pose and take the one nearest the last planned configuration; the samples in between are linear in q (MODEL: stays inside the joint box).
        -> dict(ok, excess, i_last, idx [virtual indices], why).  ok False: no in-limit IK solution at the end pose (the row is refuted with the reason)."""
        CCP, inp = self.CCP, self.inp
        i_last = ib
        while i_last + 1 < n_all and self._post_of(g, seg, i_last + 1) == self._post_of(g, seg, ib):
            i_last += 1
        excess = float(max(want - arc[i_last], 0.0))
        d = None
        for j in range(i_last - 1, -1, -1):
            dd = np.asarray(g.offs[i_last], float) - np.asarray(g.offs[j], float)
            if float(np.linalg.norm(dd)) > 1e-9:
                d = dd / np.linalg.norm(dd)
                break
        if d is None:
            return dict(ok=False, excess=excess, i_last=i_last, idx=[], why="the last segment has no direction (zero-length path): the overshoot pose cannot be built")
        n = max(1, int(math.ceil(excess / CCP.DS - 1e-9)))
        end = np.asarray(g.offs[i_last], float) + d * excess
        qe = {}
        for s_ in ("left", "right"):
            arm = inp.arms[s_]
            Q, _, _ = CCP._nodes(arm, CCP._tcp_pose(inp.T0[s_], end), CCP._psi_grid(arm, inp.q0[s_])[0])
            if len(Q) == 0:
                miss = CCP._limit_violation(arm, CCP._tcp_pose(inp.T0[s_], end), CCP._psi_grid(arm, inp.q0[s_])[0][::2])
                return dict(ok=False, excess=excess, i_last=i_last, idx=[], arm=s_,
                            why=f"{s_} arm: no in-limit SRS solution at the overshoot pose (the last planned sample continued {excess * 1e3:.1f} mm along the last segment; the closest "
                                f"configuration is {miss:.3f} rad outside a joint limit, inf = out of reach)")
            ql = np.asarray(g.q[s_][i_last], float)
            qe[s_] = Q[int(np.argmin(np.max(np.abs(Q - ql), axis=1)))]
        dqm = float(max(np.max(np.abs(qe[s_] - np.asarray(g.q[s_][i_last], float))) for s_ in ("left", "right")))
        if dqm > CCP.DQ_MAX * n + 1e-12:
            return dict(ok=False, excess=excess, i_last=i_last, idx=[], dq_max=dqm,
                        why=f"no in-limit IK solution continuous with the planned path at the overshoot pose ({excess * 1e3:.1f} mm past the last planned sample): the nearest solution is "
                            f"{dqm:.3f} rad away in joint space (> DQ_MAX {CCP.DQ_MAX} x {n} samples, closed_chain.DQ_MAX ASSUMED)")
        idx = []
        for k in range(1, n + 1):
            f = min(1.0, k * CCP.DS / excess) if excess > 0 else 1.0
            i = n_all + len(ext_store)
            ext_store[i] = dict(q={s_: np.asarray(g.q[s_][i_last], float) + f * (qe[s_] - np.asarray(g.q[s_][i_last], float)) for s_ in ("left", "right")},
                                off=np.asarray(g.offs[i_last], float) + d * excess * f, i_last=i_last)
            idx.append(i)
        return dict(ok=True, excess=excess, i_last=i_last, idx=idx, n=n, dq_max=dqm, why="")

    def _model_scripted(self, a) -> PathModel:
        moves = CR.plan_moves(CC.plan_key(self.task))
        names = [s for s, _ in moves]
        glob = [(0.0, "p0", np.zeros(3))]            # (order, key, offset): global path order for the continuation IK
        static, branches, T, raws = {}, {}, {}, {}
        cum, prev_key, order0 = np.zeros(3), "p0", 0.0
        for seg, dl in moves:
            dl = np.asarray(dl, float)
            d = float(np.linalg.norm(dl))
            T[seg] = CR.move_time(d, a)
            ent = [(i / N_STATIC, f"{seg}:{i}") for i in range(1, N_STATIC + 1)]
            raw = []
            for t in R1.branch_times(T[seg]):
                s = min(t / T[seg], 1.0)
                frac = s * s * (3 - 2 * s)
                v = d * 6.0 * s * (1 - s) / T[seg]
                kb = prev_key if frac <= 1e-12 else (f"{seg}:{N_STATIC}" if frac >= 1 - 1e-12 else f"{seg}@{t:.2f}")
                if kb.startswith(f"{seg}@"):
                    ent.append((frac, kb))
                raw.append((float(t), v, kb))
            ent.sort()
            for f, k in ent:
                glob.append((order0 + f, k, cum + dl * f))
            static[seg] = [prev_key] + [f"{seg}:{i}" for i in range(1, N_STATIC + 1)]
            raws[seg] = (raw, d)
            prev_key, cum, order0 = f"{seg}:{N_STATIC}", cum + dl, order0 + 1.0
        glob.sort(key=lambda r: r[0])
        order = [k for _, k, _ in glob]
        offs = {k: o for _, k, o in glob}
        arc = {k: c for k, c in zip(order, np.r_[0.0, np.cumsum([np.linalg.norm(offs[order[i + 1]] - offs[order[i]]) for i in range(len(order) - 1)])])}
        for seg in names:
            sT, lst = self._stop_T(seg), []
            for t, v, kb in raws[seg][0]:
                fin = False
                if v <= 1e-9:
                    ke = kb
                elif (not EX.legacy_stop()) and 1e-9 < T[seg] - t <= sT + 1e-9:
                    ke, fin = f"{seg}:{N_STATIC}", True      # 11-16: the stop finishes the segment: end of stop = the segment's end sample
                else:       # the END of the stop: first sample at or beyond 0.5 v T along the path (continuing into the next segments); None = past the last sample
                    want = arc[kb] + 0.5 * v * sT
                    ke = next((k for k in order[order.index(kb):] if arc[k] >= want - 1e-12), None)
                lst.append(Br(t, v, kb, ke, finish=fin, a_stop=(float(a) if fin else None)))
            branches[seg] = lst
        cap = THR.capture_path(PE.caps()["pot"], offs)
        disp = sum(np.asarray(dl, float) for _, dl in moves)
        return PathModel(a, names, offs, cap, static, order, branches, T, np.asarray(disp, float),
                         desc="scripted: carry.PLANS straight lines, smoothstep timing, continuation IK (thresholds.capture_path)")

    def env(self, a):
        k = round(float(a), 6)
        if k not in self._envs:
            m = self.model(a)
            self._envs[k] = self.backend.make_envelope(self._prior_spec(), m.cap, m.offs)
        return self._envs[k]

    def _prior_spec(self):
        lo, hi = CS.prior_total(self.task)
        return CS.spec_from_total(self.task, lo, hi)

    # ------------------------------------------------------------------ geometry of the landings
    def start_support(self):
        return SC.support_tops(self.world)["counter"]

    def goal(self, model: PathModel):
        """(support top dict, dest xy) of the plan's goal, as stack/adapters/static_admission does."""
        o = np.asarray(PE.o_ref(), float)
        if self.td.base:
            from world_model.v2.nero.base import scene as BS
            return BS.support_tops_behind(self.world)[self.td.goal_support], np.asarray(BS.place_pose_xy(self.world), float)
        return SC.support_tops(self.world)[self.td.goal_support], o[:2] + np.asarray(model.disp, float)[:2]

    def admit_disp(self, model: PathModel) -> np.ndarray:
        d = np.asarray(model.disp, float).copy()
        if self.td.base:
            _, dest = self.goal(model)
            d[:2] = dest - np.asarray(PE.o_ref(), float)[:2]
        return d

    # ------------------------------------------------------------------ rows
    def _worst(self, model, lo, hi, keys, a):
        k = (model.a, round(lo, 6), round(hi, 6), tuple(keys), round(float(a), 6))
        if k not in self._memo:
            self._memo[k] = self.env(model.a).worst(self.spec_for(lo, hi).contents, list(keys), float(a), PAY.MODE)
        return self._memo[k]

    def torque_row(self, model, theta, keys, a, name) -> Row:
        """u <= U_ADMIT over `keys` at peak object acceleration `a` for every contents vertex of the theta hull (proof under MODEL; the LP best-split u is not computed)."""
        lo, hi = _hull(theta)
        u, det = self._worst(model, lo, hi, keys, a)
        return Row(name, bool(u <= PAY.U_ADMIT), margin=float(PAY.U_ADMIT - u), unit="utilisation", guarantee="proof",
                   worst={**{k: v for k, v in det.items() if k in ("wp", "m", "c", "Fz", "twist")}, "u_max": float(u), "a": float(a), "n_samples": len(keys)},
                   assumptions=dict(model=PAY.GUARANTEE + f" over the theta hull [{lo:.3f}, {hi:.3f}] kg x acceleration box at {len(keys)} path samples", **HONEST),
                   why="" if u <= PAY.U_ADMIT else f"utilisation {u:.3f} > {PAY.U_ADMIT} at a = {a:.3f} m/s^2")

    def open_row(self, theta, a, name) -> Row | None:
        if "open_content" not in self.td.rows:
            return None
        lo, hi = _hull(theta)
        a_open = OCN.a_open_of(self.spec_for(lo, hi))
        return OCN.open_content_row(a, a_open, name=name) if a_open < np.inf else None

    def motion_row(self, model: PathModel, seg: str, plan: MotionRef, theta: TH.ThetaSet, t: float) -> Row:
        """check_motion of Rule 1: a static (proof under MODEL) row of the recovery motion `plan` on theta from the branch point t of segment `seg`."""
        nm = f"{plan.kind}[{seg}]"
        if plan.kind == "none" or (seg == model.names[0] and t <= 1e-9):
            return Row(nm, True, margin=None, unit="", guarantee="proof", worst=dict(why="no motion: the object rests on its start support at the start of the first step"),
                       assumptions=dict(model="start of step 0: nothing to recover from (the claws have not lifted yet)"))
        if plan.kind == "refuse":
            return Row(nm, False, margin=None, unit="", guarantee="refuted", why=plan.note or "no acceleration level is admitted on this bin")
        br = model.br(seg, t)
        if br.ahead is None and br.ext is not None and not br.ext.get("ok"):
            return Row(nm, False, margin=None, unit="", guarantee="refuted", worst=dict(t=float(t), excess_m=float(br.ext["excess"]), arm=br.ext.get("arm")),
                       assumptions=dict(model="the stop overshoot runs past the last planned sample; the path is extended by a straight-line continuation of the last segment (MODEL) and the arms' SRS IK is solved there"),
                       why=f"stop at t = {t:.2f} s: the overshoot pose (0.5 v T past the last planned sample) cannot be reached: " + str(br.ext.get("why")))
        if br.ahead is None:
            return missing(nm, f"the stop overshoot (0.5 v T = {0.5 * br.v * plan.stop_T * 1e3:.1f} mm at t = {t:.2f} s) runs past the last planned path sample: the pose at the end of the stop is "
                               "not checked, so nothing is admitted")
        rows = []
        if plan.kind.startswith("stop") or plan.kind == "read_hold":
            a_stop = br.a_stop if br.finish else br.v / max(plan.stop_T, 1e-9)
            ks = model.keys_stop(seg, br)
            rows.append(self.torque_row(model, theta, ks, a_stop, "stop"))
            rows.append(self.torque_row(model, theta, [br.ahead], 0.0, "hold_after_stop"))
            o = self.open_row(theta, a_stop, "open_content[stop]")
            if o is not None:
                rows.append(o)
        a_mv = A_BACKUP if plan.kind in ("stop_back", "stop_adv") else plan.a
        if plan.kind in ("stop_back", "back_at"):
            rows.append(self.torque_row(model, theta, model.keys_back(seg, br, plan.target), a_mv, "retrace"))
        elif plan.kind in ("stop_adv", "adv_at"):
            rows.append(self.torque_row(model, theta, model.keys_adv(seg, br, plan.target), a_mv, "advance"))
        if plan.kind in ("stop_back", "stop_adv", "back_at", "adv_at"):
            o = self.open_row(theta, a_mv, "open_content[move]")
            if o is not None:
                rows.append(o)
        rows.append(self.collision_row(model, seg, br, plan))
        if self.n_sampled > 0 and plan.kind in ("stop_back", "stop_adv", "stop_hold", "back_at", "adv_at"):
            rows.append(self.sampled_row(model, seg, theta, plan))
        r = _merge(nm, rows)
        if model._ext_ok(br):
            r.assumptions["stop_overshoot_extension"] = dict(tag="MODEL", excess_m=float(br.ext["excess"]), n_samples=int(br.ext["n"]), dq_max_rad=float(br.ext["dq_max"]),
                                                              note="the overshoot (0.5 v T) runs past the last planned sample: the path is continued in a straight line along the last segment (MODEL), both arms' "
                                                                   "SRS IK is solved at its end (nearest in-limit solution, linear in q between), and the same torque / collision rows are evaluated there")
        return r

    # ------------------------------------------------------------------ collision rows of the recovery motion
    def _scene(self, model):
        if not hasattr(self, "_scenes"):
            self._scenes, self._cd, self._crow = {}, {}, {}
        if model.a not in self._scenes:
            self._scenes[model.a] = RCOL._Scene(model.cc)
        return self._scenes[model.a]

    def _coll_state(self, model, idx, post):
        k = (model.a, idx, post)
        if k not in self._cd:
            sc = self._scene(model)
            g = model.geo
            if idx >= len(g.offs):                                   # an extended sample of an overshoot past the last planned one (Br.ext)
                e = model.ext_store[idx]
                sc.set_state(e["q"]["left"], e["q"]["right"], e["off"], post)
            else:
                sc.set_state(g.q["left"][idx], g.q["right"][idx], g.offs[idx], post)
            dd = sc.distances()
            X, R = sc.geom_state()
            self._cd[k] = ({c: np.where(dd[c][1], sc.cap[c], dd[c][0]) for c in RCOL.CLASSES}, X, R)
        return self._cd[k]

    def _coll_args(self, model, idx, post):
        """(ql, qr, p, post) of geometry sample idx (the arguments _coll_state passes to the scene)."""
        g = model.geo
        if idx >= len(g.offs):
            e = model.ext_store[idx]
            return (e["q"]["left"], e["q"]["right"], e["off"], post)
        return (g.q["left"][idx], g.q["right"][idx], g.offs[idx], post)

    def recovery_indices(self, model, seg, br, plan) -> list:
        """closed_chain geometry sample indices walked by the recovery from branch br: the stop (branch pose -> end of the stop), then the retrace back to the start of
        plan.target (descending) or the advance to the end of plan.target (ascending)."""
        g = model.geo
        if model._ext_ok(br):          # the overshoot runs past the last planned sample: planned samples, then the extended ones (virtual indices)
            walk = list(range(br.idx_b, br.ext["i_last"] + 1)) + list(br.ext["idx"])
            i_end = br.ext["i_last"]
        else:
            walk, i_end = list(range(br.idx_b, br.idx_e + 1)), br.idx_e
        idx = list(walk)
        if plan.kind in ("stop_back", "back_at"):
            idx += list(reversed(walk[:-1])) + list(range(br.idx_b - 1, g.bounds[plan.target][0] - 1, -1))          # back to the start of the target (equal to range(idx_e - 1, start - 1, -1) when start <= idx_b)
        elif plan.kind in ("stop_adv", "adv_at"):
            idx += list(range(i_end + 1, g.bounds[plan.target][1] + 1))
        return idx

    def collision_row(self, model, seg, br, plan) -> Row:
        """USER 11-16 D9 + arm-arm / arm-world on the recovery motion (checks/collision swept row on adapters/collision_rows' per-sample distances) over every 1 cm geometry sample
        from the branch pose through the stop overshoot and the retrace / advance.  Geometric: independent of theta and of the retrace acceleration, so cached.  The swept term uses
        unit time per sample (dt = 1: speed bound = geom displacement + rotation * radius per sample), the module's MODEL (linear in q between samples)."""
        nm = "collision"
        if self.planner != "closed_chain" or model.geo is None:
            return missing(nm, "scripted plan: the recovery is executed by fresh continuation IK, its arm configurations are not known in advance, so no collision model exists; "
                               "scripted programs are not admitted until one is built")
        idxs = self.recovery_indices(model, seg, br, plan)
        sc = self._scene(model)
        ck = (model.a, seg, tuple(idxs))
        if ck in self._crow:
            return self._crow[ck]

        def post(i):          # is sample i after the base move?  the moving segment wins at a shared boundary index
            if i >= len(model.geo.offs):
                i = model.ext_store[i]["i_last"]          # an extended sample is on the side of the base move of the last planned sample it continues
            b = model.geo.bounds
            return sc.post_base(seg if b[seg][0] <= i <= b[seg][1] else next(s_ for s_, (a_, b_) in b.items() if a_ <= i <= b_))
        st = [self._coll_state(model, i, post(i)) for i in idxs]
        n = len(idxs)
        tt, n_ref = np.arange(n, dtype=float), 0
        Dst = {c: np.stack([x[0][c] for x in st]) for c in RCOL.CLASSES}
        Xst, Rst = np.stack([x[1] for x in st]), np.stack([x[2] for x in st])
        if n > 1 and not RCOL.swept_coarse():          # USER 11-17: adaptive refinement of the tight intervals (collision_rows.refine_samples); legacy BIMANUAL_LEGACY_SWEPT_COARSE=1
            tt, Dst, Xst, Rst, n_ref = RCOL.refine_samples(sc, [self._coll_args(model, i, post(i)) for i in idxs], Dst, Xst, Rst, RCOL.d_min_table())
        m_ = len(tt)
        h = np.diff(tt)
        sp = RCOL._geom_speeds(sc, Xst, Rst, 1.0) if m_ > 1 else {}
        d, v, names = {}, {}, {}
        for c in RCOL.CLASSES:
            D = Dst[c]
            if D.shape[1] == 0:
                d[c], v[c], names[c] = np.full(m_, sc.cap[c]), np.zeros(m_), [""] * m_
                continue
            if m_ > 1:
                S = RCOL._pair_speed(sc, c, sp, m_ - 1)          # displacement + rotation * radius per interval (S is per unit sample index)
                dm_p = np.minimum(D[:-1], D[1:])
                mask = dm_p <= (dm_p.min(1) + S.max(1))[:, None]
                vv = np.where(mask, S, 0.0).max(1) / h           # speed over the (sub-)interval, time = its length in sample indices
                v[c] = np.r_[vv, vv[-1]]
            else:
                v[c] = np.zeros(1)
            d[c] = D.min(1)
            names[c] = [f"{sc.nm(sc.pairs[c][i][0])}|{sc.nm(sc.pairs[c][i][1]) if sc.pairs[c][i][1] >= 0 else 'floor'}" for i in D.argmin(1)]
        row = KC.swept_clearance_row(d, v, tt, RCOL.d_min_table(), name=nm, pair_names=names)
        row.assumptions = dict(row.assumptions, samples=f"{n} geometry samples (indices {idxs[0]}..{idxs[-1]}), swept bound per 1 cm interval (dt = 1 per sample)",
                               swept_refinement=("none (BIMANUAL_LEGACY_SWEPT_COARSE=1)" if RCOL.swept_coarse() else
                                                 f"adaptive: an interval with bound < d_min (endpoints >= d_min) is halved up to {RCOL.SWEPT_MAX_DEPTH} times (min step = 1/{2 ** RCOL.SWEPT_MAX_DEPTH} of the interval, "
                                                 f"ASSUMED, USER to confirm), state linear in q and p (the row's MODEL), own speed bound per sub-interval; {n_ref} midpoints added, {m_} points tested"),
                               not_covered="arm links vs the carried object, self-collision, base / lift column, anything the speed model misses; same d_min as collision_rows", **HONEST)
        self._crow[ck] = row
        return row

    def read_nb(self, model) -> float:
        """probe.read()'s noise bound nb [kg] = TAU_NOISE * sum over both arms of sum_j |pinv(J^T)_z,j| / g, MAX over this task's path samples (J = the TCP-site Jacobian of envelope.Arms)."""
        from world_model.v2.nero.contents import probe as PR
        terms = self.env(model.a).arms.terms
        best = 0.0
        for k in model.order:
            best = max(best, sum(float(np.abs(np.linalg.pinv(terms[(k, s)]["J"].T)[2]).sum()) for s in ("left", "right")))
        return best * PR.TAU_NOISE / OB.G

    def read_spec(self, model):
        """(ReadingSpec | None, note): None when the task has no claw_guard calibration."""
        if not hasattr(self, "_rspec"):
            self._rspec = {}
        if model.a not in self._rspec:
            nb = self.read_nb(model)
            try:
                self._rspec[model.a] = (claw_read_spec(self.task, nb), f"nb = {nb:.4f} kg")
            except NoCalibration as e:
                self._rspec[model.a] = (None, str(e))
        return self._rspec[model.a]

    # ------------------------------------------------------------------ sampled rows (optional)
    def sampled_row(self, model, seg, theta, plan) -> Row:
        """worlds drawn from the theta hull through carry_cert.sample_world (rejection on the total mass), run by carry.world_trial at the plan's acceleration with the
        STORED backups branched every 0.25 s; checks/dynamic.family_row counts.  Only the stored backup is run by the executor: a reading-program bin plan is not."""
        nm = f"dynamic[{seg}]"
        if plan.kind in ("back_at", "adv_at"):
            return missing(nm, "the executor runs the stored backups only; a reading program's bin plan (slower retrace) is not simulated")
        if self.td.base:
            return missing(nm, "carry_cert.sample_world / _trial are not wired for the table-behind tasks")
        lo, hi = _hull(theta)
        key = (round(lo, 6), round(hi, 6), model.a)
        if key not in self._recs:
            cell = total_to_cell(self.task, lo, hi)
            cp = CC.C.carry_physics()
            fc = bool(cp["fric_comp"])
            ws = CC.WELD_STIFF if cp["weld_solref"] is not None else None
            m_hat = 0.5 * ((lo - self.em) + (hi - self.em))
            jobs = []
            for i in range(self.n_sampled):
                for attempt in range(60):
                    seed = int(np.random.SeedSequence([CC.TASK_ID.get(self.task, 0), int(round(1000 * lo)), int(round(1000 * hi)), i, attempt]).generate_state(1)[0])
                    w = CC.sample_world(self.task, cell, np.random.default_rng(seed))
                    if lo - 1e-9 <= self._world_total(w) <= hi + 1e-9:
                        jobs.append((self.task, cell, i, seed, model.a, m_hat, fc, ws))
                        break
            if self.procs > 1:
                import multiprocessing as mp
                with mp.get_context("spawn").Pool(self.procs, maxtasksperchild=3) as pool:
                    res = list(pool.imap_unordered(CC._trial, jobs))
            else:
                res = [CC._trial(j) for j in jobs]
            self._recs[key] = [r for _, _, r in res if "error" not in r]
        r = DY.family_row(self._recs[key], seg, (0.0,))
        r.assumptions = dict(r.assumptions, worlds="uniform on the theta hull (rejection on the total mass), carry_cert.sample_world; stored backup branched every 0.25 s")
        return r

    def _world_total(self, w) -> float:
        if self.td.n_cups:
            return float(self.em + sum(w.fills_kg) + self.td.n_cups * OB.CUP["empty"])
        return float(self.em + w.contents_kg)

    # ------------------------------------------------------------------ landings
    def landing_row(self, model: PathModel, landing: str, theta: TH.ThetaSet) -> Row:
        """landing_ok of Rule 1: S (support at the landing / torque at the hold poses) and R_K (a constructive witness within depth K and NODE_CAP)."""
        lo, hi = _hull(theta)
        key = (self.task, model.a, landing, round(lo, 6), round(hi, 6), os.environ.get("BIMANUAL_LEGACY_HOLD_IS_SAFE"))       # the flag is read at call time: part of the cache key
        if key in self._wit:
            return self._wit[key]
        kind, _, arg = landing.partition(":")
        nm = f"landing[{landing}]"
        em, ec, cont = self.em, self.ec, self.spec_for(lo, hi).contents
        if kind == "set_down" and arg == "start":
            sup = self.start_support()
            s = SUP.support_row(self.td, np.asarray(PE.o_ref(), float)[:2], sup["x"], sup["y"], em, ec, cont)
            rows = [s, self.witness(model, 0, theta, "plan from the start: the whole plan, depth 1")]
        elif kind == "set_down" and arg == "goal":
            sup, dest = self.goal(model)
            s = SUP.support_row(self.td, dest, sup["x"], sup["y"], em, ec, cont)
            w = Row("R_K[goal]", True, margin=None, unit="", guarantee="proof", worst=dict(depth=0, nodes=0, K=R1.K, node_cap=R1.NODE_CAP, action="the landing is the plan's goal"),
                    assumptions=dict(model="zero remaining moves: the object is set down at the goal of the plan (placement error NOT covered)"))
            rows = [s, w]
        elif kind == "hold" and arg in model.names and os.environ.get("BIMANUAL_LEGACY_HOLD_IS_SAFE") != "1":
            ns = Row("landing.not_supported", False, margin=None, unit="", guarantee="refuted", worst=dict(landing=landing),
                     assumptions=dict(model="USER 11-17: safe = resting on a support; an object held in the arms is not"), why="held in the arms: not on a support (USER 11-17)")
            rows = [ns, self.torque_row(model, theta, model.static[arg], 0.0, "hold_poses"),
                    missing("R_K", "not computed: hold is not a safe landing (USER 11-17)")]
        elif kind == "hold" and arg in model.names:
            rows = [self.torque_row(model, theta, model.static[arg], 0.0, "hold_poses"),
                    self.witness(model, model.names.index(arg), theta, f"remaining plan from {arg}: depth 1 (covers every hold pose of the segment)")]
        else:
            rows = [missing("S", f"landing {landing!r}: no S check built (let go in the air / set down elsewhere than the start or the goal)"),
                    missing("R_K", f"R_K witness not built for landing {landing}")]
        self._wit[key] = _merge(nm, rows)
        return self._wit[key]

    def witness(self, model: PathModel, first: int, theta: TH.ThetaSet, note: str) -> Row:
        """R_K witness: the planner's remaining plan (segments names[first:]) admitted by checks/admit.admit on theta (static rows, proof under MODEL)."""
        lo, hi = _hull(theta)
        sup, _ = self.goal(model)
        ps = model.path_samples(first)
        ps = AD.PathSamples(ps.offs, ps.per_seg, ps.cap, self.admit_disp(model))
        adm = AD.admit(self.td, self.spec_for(lo, hi), ps, self.backend, (sup["x"], sup["y"]))
        ok = bool(adm.verdict.admit) and not adm.action.startswith("refuse")
        ms = [m for m in adm.verdict.margins.values() if m is not None and math.isfinite(m)]
        return Row(f"R_K[{self.task}:{model.names[first]}..]", ok, margin=(min(ms) if ms else None), unit="mixed", guarantee="proof" if ok else "refuted",
                   worst=dict(depth=1, nodes=1, K=R1.K, node_cap=R1.NODE_CAP, action=adm.action, a_admit=adm.a_admit, remaining=model.names[first:], note=note),
                   assumptions=dict(model="constructive witness = the planned remaining path admitted by the static rows on this theta set (checks/admit.admit); "
                                          "'not found' would be not shown, not refuted"),
                   why="" if ok else f"the remaining plan {model.names[first:]} is refused on [{lo:.3f}, {hi:.3f}] kg: {adm.action}; {adm.verdict.why}")

    # ------------------------------------------------------------------ programs
    def parse_backup(self, seg, model: PathModel):
        """the stored backup's phase list -> (MotionRef, landing name).  See the module doc.  11-17: the MotionRef keeps the FULL retrace / advance tuple in `segs`."""
        ph = self.backup(seg)
        if ph is None:
            return None, None
        sT = float(ph[0][1]) if ph[0][0] == "stop" else TY.STOP_T_S
        back = adv = None
        segs = None
        weld_off = False
        for p in ph:
            if p[0] == "retrace":
                back = p[1][-1]
                segs = tuple(p[1])
            elif p[0] == "to":
                back = str(p[1]).replace("start_", "", 1)
            elif p[0] == "advance":
                adv = p[1][-1]
                segs = tuple(p[1])
            elif p[0] == "weld" and not p[1]:
                weld_off = True
        if adv is not None:
            return MotionRef("stop_adv", seg, adv, None, sT, segs=segs), ("set_down:goal" if weld_off and adv == model.names[-1] else f"unsupported:advance:{adv}")
        if back is not None:
            if weld_off:
                return MotionRef("stop_back", seg, back, None, sT, segs=segs), ("set_down:start" if back == model.names[0] else f"unsupported:set_down_at:{back}")
            return MotionRef("stop_back", seg, back, None, sT, segs=segs), f"hold:{back}"
        return MotionRef("stop_hold", seg, None, None, sT), (f"let_go:{seg}" if weld_off else f"hold:{seg}")

    def programs(self, model: PathModel, seg: str, theta_t: TH.ThetaSet):
        """-> (programs in order of preference, fallback program) for the step; ([], None) when the step has no stored recovery."""
        ref, landing = self.parse_backup(seg, model)
        if ref is None:
            return [], None
        fixed = TH.RecoveryProgram.fixed("stored_backup", ref, landing)
        progs = [fixed]
        if ref.kind in ("stop_back", "stop_adv") and self.read_spec(model)[0] is not None:
            progs.append(self.read_program(model, seg, ref, landing, theta_t))
        return progs, fixed

    def read_program(self, model, seg, ref: MotionRef, landing, theta_t) -> TH.RecoveryProgram:
        spec = self.read_spec(model)[0]
        R = TH.reading_range(spec, theta_t)
        nb = max(1, int(math.ceil(R.width / BIN_WIDTH_KG - 1e-12)))
        edges = [R.lo + R.width * i / nb for i in range(nb)] + [R.hi]
        ivs = [TH.Interval(edges[i], edges[i + 1]) for i in range(nb)]
        kind = "back_at" if ref.kind == "stop_back" else "adv_at"
        prefix = MotionRef("read_hold", seg, None, None, ref.stop_T)
        tmp = TH.RecoveryProgram("read_then_retrace", prefix, spec, tuple((iv, MotionRef(kind, seg, ref.target, segs=ref.segs)) for iv in ivs), landing)
        reach = [b for b in model.branches[seg] if b.ahead is not None]       # the branch whose stop ends farthest along the path: its retrace covers the samples of every other branch
        last = max(reach, key=lambda b: model._pos[b.ahead])
        keys = model.keys_back(seg, last, ref.target) if ref.kind == "stop_back" else model.keys_adv(seg, model.branches[seg][0], ref.target)
        bins = []
        for (j, th_j, _ex), iv in zip(TH.bin_sets(tmp, theta_t), ivs):
            if th_j.is_empty():
                bins.append((iv, MotionRef(kind, seg, ref.target, A_BACKUP, segs=ref.segs)))
                continue
            lo, hi = _hull(th_j)
            cont = self.spec_for(lo, hi).contents
            a_t = self.env(model.a).a_max(cont, keys, PAY.MODE, limit=PAY.U_ADMIT)
            a_open = OCN.a_open_of(self.spec_for(lo, hi)) if "open_content" in self.td.rows else np.inf
            a_bin = None if a_t is None else min(A_BACKUP, a_t, a_open)
            if a_bin is None or a_bin < A_MIN:
                bins.append((iv, MotionRef("refuse", seg, ref.target, None, ref.stop_T, note=f"no acceleration >= A_MIN = {A_MIN} admitted on [{lo:.3f}, {hi:.3f}] kg "
                                                                                           f"(a_max {a_t}, a_open {a_open:.3f})")))
            else:
                bins.append((iv, MotionRef(kind, seg, ref.target, float(a_bin), ref.stop_T, segs=ref.segs)))
        return TH.RecoveryProgram("read_then_retrace", prefix, spec, tuple(bins), landing)

    # ------------------------------------------------------------------ the executable certificate (11-17)
    @staticmethod
    def _json_phases(ph):
        """backup phase list -> JSON lists (tuples -> lists); the executor turns them back."""
        return [[(list(x) if isinstance(x, (tuple, list)) else x) for x in p] for p in ph]

    def program_spec(self, model: PathModel, seg: str, prog: TH.RecoveryProgram, rows=(), theta_t=None) -> dict:
        """the program as plain JSON that stack/executor.py runs (`("program", spec)`), see the module doc of the executor.
        fixed program (the stored backup): dict(name, kind='fixed', stop_T, phases, landing).
        reading program: dict(name, kind='read', stop_T, prefix_phases [stop, read_hold], read {producer, hold_s, n_read, spec}, bins [{j, lo, hi, motion, segs, a, verified, theta_j}],
        tail_phases (the stored backup's phases after its retrace / advance), landing, executable).  `verified` = Theta_j is non-empty AND every row of bin j (motion rows at every
        branch time + the landing row) passed in `rows` (a bin with an empty Theta_j got A_BACKUP unchecked: verified False)."""
        ph = self.backup(seg)
        phs = [] if ph is None else self._json_phases(ph)
        if prog.reading is None:
            ref = prog.bins[0][1]
            return dict(name=prog.name, kind="fixed", stop_T=float(ref.stop_T), phases=phs, landing=prog.landing, executable=ph is not None,
                        a_min=float(A_MIN), a_backup=float(A_BACKUP))
        sets = TH.bin_sets(prog, theta_t) if theta_t is not None else [None] * len(prog.bins)
        i_m = next((i for i, p in enumerate(ph or []) if p[0] in ("retrace", "advance")), None)
        tail = [] if i_m is None else phs[i_m + 1:]
        bins = []
        for j, (iv, ref) in enumerate(prog.bins):
            th_j = None if sets[j] is None else sets[j][1]
            mine = [r for r in rows if f":bin{j}[" in r.name or f":landing[bin{j}]" in r.name]
            ok_rows = bool(mine) and all(r.ok for r in mine)
            nonempty = th_j is not None and not th_j.is_empty()
            motion = "refuse" if ref.kind == "refuse" else ("retrace" if ref.kind == "back_at" else "advance")
            bins.append(dict(j=j, lo=float(iv.lo), hi=float(iv.hi), motion=motion, segs=(None if ref.segs is None else list(ref.segs)),
                             a=(None if ref.a is None else float(ref.a)), verified=bool(nonempty and ok_rows and ref.kind != "refuse" and ref.segs is not None),
                             theta_j=(None if th_j is None else th_j.to_json())))
        stop_T = float(prog.prefix.stop_T)
        exe = (self.planner == "closed_chain" and i_m is not None and os.environ.get("BIMANUAL_LEGACY_STORED_BACKUP_ONLY") != "1")
        return dict(name=prog.name, kind="read", stop_T=stop_T, prefix_phases=[["stop", stop_T], ["read_hold", READ_HOLD_S]],
                    read=dict(producer=READ_PRODUCER, hold_s=READ_HOLD_S, n_read=N_READ_RT, spec=prog.reading.to_json()), bins=bins, tail_phases=tail,
                    landing=prog.landing, executable=bool(exe), a_min=float(A_MIN), a_backup=float(A_BACKUP))

    def plan_key(self, model: PathModel):
        """the binding of a certificate to a plan: dict(geo_key = the closed_chain PlanInputs key, a_plan); None for a scripted plan.  The executor compares it with ep.cc."""
        if self.planner != "closed_chain" or self.inp is None:
            return None
        return dict(geo_key=self.inp.key(), a_plan=round(float(model.a), 6))

    def _read_assumptions(self, model) -> list:
        spec, note = self.read_spec(model)
        A = lambda n, v, tag, src: dict(name=n, value=v, tag=tag, src=src)
        if spec is None:
            return [A("reading_program", "not offered", "MODEL", note)]
        return [A("READ_HOLD_S", READ_HOLD_S, "ASSUMED", "hold before the in-carry claw-load read [s] (= probe.T_HOLD reused; 11-17 executor phase)"),
                A("N_READ_RT", N_READ_RT, "ASSUMED", "control steps averaged by the in-carry read (= probe.N_READ reused)"),
                A("EPS_FREE_transfer", "EPS_FREE measured free-hanging at the probe pose is applied to a mid-carry read at the branch pose", "ASSUMED", "claw_guard CALIB"),
                A("read_nb", self.read_nb(model), "ASSUMED", "probe.read() noise bound formula at this task's path samples (max); TAU_NOISE 0.05 Nm ASSUMED")]

    def callables(self, model: PathModel, seg: str):
        return (lambda plan, th, t: self.motion_row(model, seg, plan, th, t)), (lambda landing, th: self.landing_row(model, landing, th))

    def assumptions(self, model: PathModel, seg: str) -> list:
        A = lambda n, v, tag, src: dict(name=n, value=v, tag=tag, src=src)
        return [A("BIN_WIDTH_KG", BIN_WIDTH_KG, "ASSUMED", "reading bin width (to be raised to the user); equal-width bins, n = ceil(range / BIN_WIDTH_KG)"),
                A("A_BACKUP", A_BACKUP, "ASSUMED", "carry.A_BACKUP: stored backup retrace acceleration"),
                A("A_MIN", A_MIN, "ASSUMED", "checks/admit.A_MIN: a retrace slower than this refuses the bin"),
                A("stop_T", self._stop_T(seg), "MODEL", "stored backup phase (stop, T): deceleration = v_branch / T (Hermite stop: constant deceleration v / T); finish branches use a_stop = a (see stop_finish)"),
                A("path", model.desc, "MODEL", f"a_plan = {model.a} m/s^2; {N_STATIC} static samples per move + the branch-pose samples"),
                A("theta_hull", "a union of cells is checked on its total-mass hull", "MODEL", "sound superset"),
                A("between_samples", "not bounded (samples only)", "MODEL", "torque / support rows are evaluated at the path samples only (a 'proof' label does not cover between them)"),
                A("branch_grid", "0.25 s grid; triggers between grid points not checked", "MODEL", "branch points every 0.25 s; a trigger can come at any control step"),
                A("stop_end", "first path sample at or beyond 0.5 v T (+ DS/2 nearest-sample margin on closed_chain), continuing into the next segments' samples", "MODEL",
                  "Hermite stop q1 = q + 0.5 v T; a stop that ends past the last sample gives a MISSING row"),
                A("stop_finish", "a branch within the last stop_T s of its segment: the stop FINISHES the segment (end of stop = the segment's last path sample), deceleration a_stop = the plan's acceleration a "
                  "(the planned remainder is the plan's own motion, whose acceleration cap is a); the collision swept term is geometric per sample (dt = 1), so it needs no speed for the remainder", "MODEL",
                  "executor stop branch 11-16 (every reference entry ends at rest: the move finishes); BIMANUAL_LEGACY_STOP=1 restores the Hermite 0.5 v T overshoot model for every branch"),
                A("collision", "closed_chain: swept row (D9 obj_world + arm_arm + arm_world) on the stop + retrace / advance samples; scripted: MISSING (no model)", "MODEL", "adapters/collision_rows"),
                *self._read_assumptions(model),
                A("tray_bins", "bins and the start-landing witness use spec_from_total (one mug may hold all the fill): conservative for the spill limit", "MODEL", "adapters/contents_specs")]


_CTX = {}


def get_ctx(task, planner=None, wp=None) -> Ctx:
    k = (task, planner) if wp is None else (task, planner, wp)
    if k not in _CTX:
        _CTX[k] = Ctx(task, planner, wp)
    return _CTX[k]


# ----------------------------------------------------------------------------------------------------------------------- plan admission + steps
def plan_admission(ctx: Ctx, cell, lo, hi):
    """the static admission of the whole plan on the cell (the K2-consistent spec) -> (AD.Admission, PathModel at the admitted acceleration | None)."""
    ctx.register_cell(cell, lo, hi)
    m = ctx.model(AD.A_NOM)
    sup, _ = ctx.goal(m)
    for _ in range(2):
        ps = m.path_samples(0)
        ps = AD.PathSamples(ps.offs, ps.per_seg, ps.cap, ctx.admit_disp(m))
        adm = AD.admit(ctx.td, ctx.spec_for(lo, hi), ps, ctx.backend, (sup["x"], sup["y"]))
        if adm.a_admit is None or abs(adm.a_admit - m.a) < 1e-6:
            break
        m = ctx.model(adm.a_admit)
    return adm, (m if adm.a_admit is not None else None)


def steps_of(ctx: Ctx, model: PathModel) -> list:
    """[(index, name)] in plan order: the arm moves, with the base move ("base_move") where a table-behind plan has it.  The release is not a step (no stored backup)."""
    out = list(model.names)
    if ctx.wp is not None and ctx.wp.base:
        out.insert(ctx.wp.base[0][0], "base_move")
    return list(enumerate(out))


def prior_cells(ctx: Ctx) -> TH.ThetaSet:
    cs = []
    for (c,) in CC.cells(ctx.task):
        lo, hi = cell_total(ctx.task, c)
        ctx.register_cell(c, lo, hi)                    # a prior cell is checked with carry_cert's own cell spec (tray: per-mug spill limit), as K2 does
        cs.append(TH.Cell({TOTAL_FIELD: TH.Interval(lo, hi)}, "prior"))
    return TH.ThetaSet(tuple(cs))


# ----------------------------------------------------------------------------------------------------------------------- the base-stop recovery of a base_move step (11-16)
BASE_STOP_CARD = pathlib.Path(__file__).resolve().parents[1] / "nero" / "results" / "base_stop_card_2026-11-16" / "CARD.json"      # MEASURED stop card (may not exist yet)


def _finite_nonneg(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(float(x)) and float(x) >= 0.0


def base_stop_card(path=None, level=None):
    """the MEASURED base-stop card -> dict(extra_overshoot=dict(line=m, turn=rad), settle_s, extra_lateral_m | None, extra_yaw_rad | None, n, src, levels, plan, card_level) | None.
    The card is a property of the base at a speed level, measured fresh (USER 11-17): scoped by LEVEL only, no task / plan match.
    None (the card-dependent rows are MISSING, never a default) when: the file is missing / unreadable; `level` is not given or is not in card["levels"]; n_not_at_rest != 0 (a stop that did not come to rest); U_kinds is not empty (a U flag during a stop); or any value used is not finite and >= 0.
    extra_lateral_m / extra_yaw_rad (sideways and yaw drift during the stops, max over n) may be absent from an older card: they come back None and the rows that need them are MISSING."""
    try:
        c = json.loads(pathlib.Path(path or BASE_STOP_CARD).read_text())
        levels = [str(x) for x in (c.get("levels") or [])]
        if level is None or str(level) not in levels:
            return None
        plan = c.get("plan")
        if isinstance(c.get("per_level"), dict) and os.environ.get("BIMANUAL_LEGACY_CARD_MAX_LEVELS") != "1":
            if str(level) not in c["per_level"]:                  # 11-17 (USER "measure per level"): the level's own measured values; legacy flag = max over levels
                return None
            c = dict(c, **c["per_level"][str(level)])
        if c.get("n_not_at_rest") != 0 or c.get("U_kinds") != []:
            return None
        vals = [c["extra_overshoot_m"], c["extra_overshoot_rad"], c["settle_s"]]
        drift = [c.get("extra_lateral_m"), c.get("extra_yaw_rad")]
        if not all(_finite_nonneg(v) for v in vals) or not all(v is None or _finite_nonneg(v) for v in drift):
            return None
        return dict(extra_overshoot=dict(line=float(vals[0]), turn=float(vals[1])), settle_s=float(vals[2]),
                    extra_lateral_m=None if drift[0] is None else float(drift[0]), extra_yaw_rad=None if drift[1] is None else float(drift[1]),
                    n=c.get("n"), src=c.get("src"), levels=levels, plan=plan, card_level=str(level))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _worst_sweep(results: list) -> dict:
    """the worst (smallest clearance margin) of several check_continuation results (the drift corners of one pose); ok = all ok."""
    w = min(results, key=lambda r: r["min_dist_m"] - r.get("clearance_min_m", 0.0))
    return dict(w, ok=all(bool(r["ok"]) for r in results), n_variants=len(results))


def _sweep_job(a):
    """pool worker: one BR.check_continuation (world, geo, prev, pose, rem); the backend is the default one of the worker."""
    from world_model.v2.stack.planners import base_route as BR
    world, geo, prev, pose, rem = a
    return BR.check_continuation(world, geo, prev, pose, rem, backend=BR.default_backend())


def base_stop_record(ctx: Ctx, model: PathModel, theta_t: TH.ThetaSet, ix: int, skills=None, card=None, level=None) -> dict:
    """the Rule 1 record fields of the base_move step: program "base_stop_finish" (USER 11-17) = base_stop -> hold (pause; hold-torque row as a PREFIX row) -> FINISH THE PLANNED ROUTE
    (base_route.remaining_route from the LANDED pose = stop distance + MEASURED extra + MEASURED lateral / yaw drift, four corners) -> the plan's remaining arm steps -> landing on the goal
    support (S = the support row at the goal on theta_t).  Checked at every branch time of the route (R1.branch_times over route_duration) plus each moving segment's decel start (analytic
    worst case) on theta_t.  skills = the step's recovery list (None: the stored default, hold).  level = the base level actually run (None: the plan's level, else the default); the card must
    cover it.  card = base_stop_card(level=, task=) (None: read the file; no usable card -> the rows that need it are MISSING).  BIMANUAL_LEGACY_STOP_DRIFT=1: no drift (the 11-17 first version).
    base_stop.next_step is MISSING here; run_cell fills it from the later steps' records (link_base_next).  -> dict to merge into the step record."""
    from world_model.v2.adapters import base_envelope as BE
    from world_model.v2.stack.planners import base_route as BR
    after, route, lv0 = ctx.wp.base[0]
    route = tuple(route)
    lv = level or lv0 or ctx.CCP.BASE_LEVEL_DEFAULT
    be = BR.default_backend()
    caps = be.levels()[lv]
    segs = BS.segment_table(route, caps, BR.GAP_S)
    dur = BR.route_duration(route, lv, be.levels())
    times = R1.branch_times(dur)
    card = base_stop_card(level=lv) if card is None else card
    extra = None if card is None else card["extra_overshoot"]
    legacy_drift = os.environ.get("BIMANUAL_LEGACY_STOP_DRIFT") == "1"
    drift = None                                       # (lateral m, yaw rad) MEASURED max over the card's n; None = not used (legacy) or not on the card
    drift_missing = False
    if card is not None and not legacy_drift:
        lat, yw = card.get("extra_lateral_m"), card.get("extra_yaw_rad")
        if lat is None or yw is None:
            drift_missing = True
        else:
            drift = (max(0.0, float(lat)), max(0.0, float(yw)))
    prev = model.names[after - 1]
    lo, hi = _hull(theta_t)
    spec = ctx.spec_for(lo, hi)
    _, lvl_rows = BE.pick_level(ctx.task, spec, (lv,), "measured")
    p0 = be.start_pose(ctx.world)
    pert = (lambda pose: BS.drift_poses(pose, drift[0], drift[1])) if drift is not None else (lambda pose: [tuple(float(x) for x in pose)])
    can_sweep = extra is not None and not drift_missing
    sweeps, by_pose = {}, {}
    pending = None                                     # procs > 1: first pass records the distinct (pose, route) sweeps, they run in a pool, second pass reads the cache

    def sweep_at(pose, rem):                           # check_continuation, cached by (pose, route)
        k = (tuple(round(float(x), 4) for x in pose), tuple((a_, round(float(b_), 4)) for a_, b_ in rem))
        if k not in by_pose:
            if pending is not None:
                pending[k] = (pose, rem)               # first occurrence, as in the serial path
                return dict(ok=True, min_dist_m=1.0, clearance_min_m=0.0, worst={})
            by_pose[k] = BR.check_continuation(ctx.world, model.geo, prev, pose, rem, backend=be)
        return by_pose[k]

    def run_sweeps():
        sw = {}
        if can_sweep:
            for c in BS.overshoot_cases(segs, times, extra):                           # R3: the stop runs past the segment end -> sweep the robot at the overshoot pose (all drift corners)
                sw[BS.case_key(c)] = _worst_sweep([sweep_at(pv, ()) for pv in pert(BS.overshoot_pose(p0, route, c))])
        return sw
    # finish the planned route from the landed pose (stop distance + MEASURED extra + MEASURED drift), arms at the hold pose of the move before the base move

    def run_results():
        out = []
        if can_sweep and segs:
            for t in BS.eval_times(segs, times):
                p_land = BS.pose_at(p0, route, segs, t, stop=True, extra=extra)
                i_seg = BS.interrupted_index(segs, t)
                per = []
                for pv in pert(p_land):
                    rem = BR.remaining_route(p0, route, i_seg, pv)
                    per.append(dict(sweep_at(pv, rem), remaining=[list(x) for x in rem]))
                r = _worst_sweep(per)
                out.append(dict(t=float(t), ok=bool(r["ok"]), min_dist_m=float(r["min_dist_m"]), clearance_min_m=float(r["clearance_min_m"]), worst=r["worst"],
                                pose=[float(x) for x in p_land], remaining=r["remaining"], n_variants=r["n_variants"]))
        return out
    if can_sweep and getattr(ctx, "procs", 1) > 1:
        pending = {}
        run_sweeps()
        run_results()
        jobs = list(pending.items())
        pending = None
        order = sorted(range(len(jobs)), key=lambda i: -len(BR.route_poses(jobs[i][1][0], jobs[i][1][1])))        # biggest first (load balance); results are stored by key
        args = [(ctx.world, model.geo, prev, jobs[i][1][0], jobs[i][1][1]) for i in order]
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(min(ctx.procs, max(1, len(args))), maxtasksperchild=3) as pool:
            for i, r in zip(order, pool.imap(_sweep_job, args, chunksize=1)):
                by_pose[jobs[i][0]] = r
    sweeps = run_sweeps()
    miss_drift = lambda nm: missing(nm, "the MEASURED stop card lacks the lateral / yaw drift of the stop (extra_lateral_m, extra_yaw_rad): the overshoot / landed pose is not known")
    rows = [BS.recovery_row(skills),
            miss_drift("base_stop.in_route") if drift_missing and extra is not None else BS.in_route_row(segs, times, extra, sweeps),
            BS.at_rest_row(segs, times, None if card is None else card["settle_s"])]
    rows += BS.base_rows_on_theta(lvl_rows.get(lv))
    rows.append(BS.landing_hold_row(ctx.torque_row(model, theta_t, model.static[prev][-1:], 0.0, "rest_payload")))          # PREFIX row: the hold at the pause (not S)
    results = run_results()
    if card is None:
        rows.append(missing("base_stop.continuation", "no usable MEASURED stop card for this level / task (CARD.json missing, other level, not all stops at rest, or a U flag): the landed pose is not known"))
    elif drift_missing:
        rows.append(miss_drift("base_stop.continuation"))
    else:
        rows.append(BS.continuation_row(results))
    rows.append(BS.goal_support_row(ctx.landing_row(model, "set_down:goal", theta_t)))
    rows.append(BS.next_step_row(None))
    for r in rows:
        if card is not None and r.name in ("base_stop.in_route", "base_stop.at_rest", "base_stop.continuation"):
            r.assumptions = dict(r.assumptions, card=dict(n=card["n"], src=card["src"], tag="MEASURED", levels=card["levels"], plan=card.get("plan")),
                                 drift=("none (BIMANUAL_LEGACY_STOP_DRIFT=1)" if legacy_drift else (None if drift is None else dict(lateral_m=drift[0], yaw_rad=drift[1], tag="MEASURED max over n; applied as a symmetric box, four corners swept (MODEL)"))))
    rec = _base_rec(model, lv, times, rows, card, prev)
    rec["card_level"] = None if card is None else card.get("card_level")
    return rec


def _base_rec(model, lv, times, rows, card, prev) -> dict:
    d = [R1.row_to_dict(r) for r in rows]
    bad = [x for x in d if not x["ok"]]
    ok = not bad
    g = "refuted" if any((not r.ok) and r.guarantee != "missing" for r in rows) else min((r.guarantee for r in rows), key=ORDER.index)
    if g == "proof":
        g = "sampled"                       # a 0.25 s branch grid (ASSUMED): never a proof (stack/checks/rule1.BRANCH_DT)
    rec = dict(a_plan=model.a, base_level=lv, branch_times=times, programs=["base_stop_finish"], rows=(d if ok else bad), all_rows=d, guarantee=g,
               landings={"base_stop_finish": "set_down:goal"}, card=card, hold_pose_after=prev)
    if ok:
        asm = [dict(name="base_stop_card", value=card, tag="MEASURED", src=str(BASE_STOP_CARD)),
               dict(name="executable", value=True, tag="MODEL", src="stack/executor.py phase base_stop, stored backup [base_stop, hold] (closed_chain._base_segment); resume = remaining_route (USER 11-17)"),
               dict(name="guarantee", value=g, tag="MODEL", src="weakest row, capped at sampled"),
               dict(name="branch_dt", value=R1.BRANCH_DT, tag="ASSUMED", src="R1.branch_times: 0.25 s grid; triggers between grid points not checked (each moving segment's decel start is added analytically)")]
        rec.update(admitted=True, executable=True, program="base_stop_finish", admission=dict(recovery="base_stop_finish", rows=d, assumptions=asm, fallback_ok_on=None))
    else:
        rec.update(admitted=False, refusal="base_stop recovery: " + "; ".join(f"{x['name']}: {x['why'] or x['guarantee']}" for x in bad)[:900])
    return rec


def link_base_next(res: list) -> None:
    """fill base_stop.next_step of the base_move record from the records of the arm steps after it (each its own Rule 1 step) and rebuild the record's verdict (in place)."""
    ib = next((i for i, r in enumerate(res) if r["seg"] == "base_move" and "all_rows" in r), None)
    if ib is None:
        return
    r = res[ib]
    def _tg(x):                                        # the fixability tag of a refused later step (step_tag: FIXABLE only if one program's failing leaves are all fixable)
        if x.get("admitted"):
            return None
        t = FX.step_tag(x)
        return dict(kind=t.kind, knobs=list(t.knobs), key=t.key)
    later = [(x["seg"], bool(x.get("admitted")), x.get("refusal") or "; ".join(f"{f['name']}" for pp in (x.get("per_program") or {}).values() for f in pp.get("failing", [])[:2]) or "refused", _tg(x))
             for x in res[ib + 1:]]
    rows = [Row(**{k: v for k, v in d.items() if k in Row.__dataclass_fields__}) for d in r["all_rows"]]
    rows = [BS.next_step_row(later) if x.name == "base_stop.next_step" else x for x in rows]
    for k in ("admitted", "executable", "program", "admission", "refusal"):
        r.pop(k, None)
    r.update(_base_rec(type("M", (), {"a": r["a_plan"]})(), r["base_level"], r["branch_times"], rows, r["card"], r["hold_pose_after"]))
    r["card_level"] = (r["card"] or {}).get("card_level")
    if r.get("out_file"):
        pathlib.Path(r["out_file"]).write_text(json.dumps(r, indent=1, default=float))


def run_cell(task, cell, out: pathlib.Path, n_sampled=0, procs=1, planner=None, fallback=True, wp=None, a=None, level=None) -> list:
    """a=None: the steps are checked at the static admission's a_admit (as before).  a given (Part B, 11-16, the Z3 oracle passes the plan's a): the steps use ctx.model(a), provided
    the static admission admits at a (a <= a_admit + 1e-9); else every step is a refusal record (the oracle refuses such plans at stage b; run_cell stays self-consistent).
    level (11-17): the base level actually run (probe_select passes the admitted level); None = the plan's own base level (ctx.wp.base[0]).  The base-stop rows use it (record base_level, card_level)."""
    ctx = get_ctx(task, planner, wp)
    ctx.n_sampled, ctx.procs = n_sampled, procs
    lo, hi = cell_total(task, cell)
    theta_t = theta_box(lo, hi)
    adm0, model = plan_admission(ctx, cell, lo, hi)
    base = dict(task=task, cell=list(cell), planner=ctx.planner, theta_t=theta_t.to_json(), plan_action=adm0.action, plan_a_admit=adm0.a_admit, n_sampled=n_sampled, fallback_evaluated=bool(fallback))
    theta_0 = prior_cells(ctx) if fallback else None
    a_refused = None
    if a is not None and model is not None:
        if float(a) > float(adm0.a_admit) + 1e-9:
            a_refused = f"the plan's acceleration a = {float(a):.4f} m/s^2 exceeds the static admission's a_admit = {float(adm0.a_admit):.4f} m/s^2 on this cell: the plan is not admitted at a"
        else:
            model = ctx.model(float(a))
    base["plan_a"] = None if a is None else float(a)
    res = []
    m_ref = model if model is not None else ctx.model(AD.A_NOM)
    for ix, seg in steps_of(ctx, m_ref):
        rec = dict(base, step=ix, seg=seg)
        if a_refused is not None:
            rec.update(admitted=False, refusal=a_refused, rows=[], guarantee="refuted")
        elif model is None:
            rows = [R1.row_to_dict(r) for r in adm0.rows if r.name.startswith(("torque_payload", "support_dest", "open_content")) and not r.ok] or [R1.row_to_dict(r) for r in adm0.rows]
            rec.update(admitted=False, refusal=f"the plan itself is refused by the static admission ({adm0.action}); no step is started", rows=rows, guarantee=adm0.verdict.guarantee)
        elif seg == "base_move" and os.environ.get("BIMANUAL_LEGACY_NO_BASE_BACKUP") == "1":
            r = missing("recovery[base_move]", "no stored recovery for the base move (backup_phases is None unless BIMANUAL_BASE_BACKUP=1); the base-stop landing check is not built here (base_tip)")
            rec.update(admitted=False, refusal="no recovery program", rows=[R1.row_to_dict(r)], guarantee="missing")
        elif seg == "base_move":
            rec.update(base_stop_record(ctx, model, theta_t, ix, level=level))
        else:
            progs, fb = ctx.programs(model, seg, theta_t)
            times = R1.branch_times(model.T[seg])
            cm, lo_ = ctx.callables(model, seg)
            adm, per = R1.admit_step(ix, theta_t, progs, times, cm, lo_, fb, theta_0, assumptions_extra=ctx.assumptions(model, seg))
            rec.update(a_plan=model.a, branch_times=times, programs=[p.name for p in progs],
                       per_program={n: dict(ok=all(r.ok for r in rs), n_rows=len(rs), failing=[R1.row_to_dict(r) for r in rs if not r.ok][:12]) for n, rs in per.items()},
                       landings={p.name: p.landing for p in progs})
            if adm is not None:
                specs = {p.name: ctx.program_spec(model, seg, p, per.get(p.name, []), theta_t) for p in progs}
                # 11-17: the executor runs the stored backup AND the reading program (stack/executor.py ("program", spec): stop, read, bin retrace); BIMANUAL_LEGACY_STORED_BACKUP_ONLY=1 = stored backups only
                ex = adm.recovery == "stored_backup" or bool(specs[adm.recovery].get("executable"))
                adm.assumptions.append(dict(name="executable", value=ex, tag="MODEL", src="stack/executor.py runs the stored backup and the reading program (BIMANUAL_LEGACY_STORED_BACKUP_ONLY=1: stored backups only)"))
                rspec = ctx.read_spec(model)[0]
                rec.update(executable=ex, admitted=True, program=adm.recovery, admission=adm.to_json(), guarantee=next(a["value"] for a in adm.assumptions if a["name"] == "guarantee"),
                           program_spec=specs[adm.recovery], programs_spec=specs, fallback_spec=(None if fb is None else specs.get(fb.name)), plan_key=ctx.plan_key(model),
                           theta_0=(None if theta_0 is None else theta_0.to_json()), read_spec=(None if rspec is None else rspec.to_json()))
                if not ex and os.environ.get("BIMANUAL_LEGACY_NONEXEC_ADMIT") != "1":
                    # 11-17: a step admitted ONLY by a program the executor cannot run is not admitted
                    nx = Row("not_executable", False, margin=None, unit="", guarantee="refuted", worst=dict(program=adm.recovery),
                             assumptions=dict(model="stack/executor.py cannot run this program (stored backups only under BIMANUAL_LEGACY_STORED_BACKUP_ONLY=1, or no closed_chain retrace)"),
                             why="admitted only by a program the executor cannot run")
                    rec.update(admitted=False, refusal="admitted only by a program the executor cannot run yet", rows=[R1.row_to_dict(nx)], guarantee="refuted", refusal_tag="NOT_FIXABLE:not_executable")
            else:
                rec.update(admitted=False, refusal="no recovery program is viable on theta_t", guarantee="refuted", rows=[])
        fn = out / f"{task}_{cell[0]}-{cell[1]}_step{ix}_{seg}.json"
        if seg == "base_move":
            rec["out_file"] = str(fn)
        fn.write_text(json.dumps(rec, indent=1, default=float))
        res.append(rec)
    link_base_next(res)                  # USER 11-17: the base program finishes the route and then runs the later steps: they must each be admitted
    return res


def _summ(rec) -> str:
    head = f"{rec['task']} cell {rec['cell']} step {rec['step']} {rec['seg']}"
    if rec.get("admitted"):
        fb = rec["admission"]["fallback_ok_on"]
        nfb = "not evaluated" if not rec.get("fallback_evaluated") else (f"{len(fb['cells'])} prior cell(s)" if fb else "none")
        return f"{head}: ADMITTED program={rec['program']} guarantee={rec['guarantee']} (fallback passes on Theta_0: {nfb})"
    why = rec.get("refusal", "")
    fails = []
    for n, pp in rec.get("per_program", {}).items():
        for r in pp["failing"][:3]:
            fails.append(f"[{n}] {r['name']}: {r['why'] or r['guarantee']}"[:230])
    if not fails:
        fails = [f"{r['name']}: {r.get('why') or r['guarantee']}"[:230] for r in rec.get("rows", [])[:4]]
    return f"{head}: REFUSED ({why})" + "".join("\n      " + f for f in fails)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True, help="comma list of: box, tray2, tray4, tray6, box_behind, tray6_behind")
    ap.add_argument("--cells", default=None, help="comma list lo-hi in carry_cert's prior unit (box: contents kg; tray: fill per mug kg); default = every carry_cert cell")
    ap.add_argument("--n-sampled", type=int, default=0, help="worlds per theta set for the optional sampled row (default 0 = static only)")
    ap.add_argument("--procs", type=int, default=1)
    ap.add_argument("--planner", choices=CR.PLANNERS, default=None)
    ap.add_argument("--no-fallback", action="store_true", help="skip evaluating the fallback on the prior cells (fallback_ok_on stays None)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.planner is not None:
        os.environ["BIMANUAL_PLANNER"] = a.planner
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for task in a.task.split(","):
        cells = [tuple(float(x) for x in c.split("-")) for c in a.cells.split(",")] if a.cells else [c for (c,) in CC.cells(task)]
        for cell in cells:
            t0 = time.time()
            for rec in run_cell(task, cell, out, a.n_sampled, a.procs, a.planner, fallback=not a.no_fallback):
                print(_summ(rec), flush=True)
            print(f"  ({task} {cell}: {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
