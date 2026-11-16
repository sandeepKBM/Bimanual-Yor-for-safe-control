"""adapters/rule4_contents.py -- the contents-carry binding of Rule 4 (stack/checks/rule4.py: check_target + the propose-check loop).  SIMULATION ONLY.
Not registered anywhere.  The proposer here is the MOCK only (no VLM / LLM is ever called).

State = (task, planned path model, segment, branch time t): the arms are mid-step, the object is carried (a point on the closed-chain planned path).
Targets (what a proposer can ask for; the planner turns each into a transfer on the SAME planned path, reversed or held):
  set_down_start      retrace the planned path back to the start of the plan and set the object down on the start counter     landing set_down:start
  hold_here           stop here and keep the object in the claws                                                              landing hold:<seg>
  hold_lower          retrace to the start of the previous segment and hold (lower pose; the first segment: its own start)    landing hold:<prev seg>
  let_go              stop and let go in the air                                                                              landing let_go:<seg>  (S not built -> MISSING)
  set_down_support:s  set down on an intermediate support s                                                                   landing set_down:s    (not built -> MISSING)
The transfer = a Rule 1 program: prefix = a STOP (read_hold rows: stop deceleration + hold at the pose just beyond it), plan = retrace at acceleration a
(back_at), landing = the target's.  Both are DERIVED on the theta_t set (not the stored backup's fixed 0.3 s / A_BACKUP):
  stop time T = the first of STOP_T0 * 2^k, k = 0..STOP_K_MAX (0.3 .. 4.8 s) for which the torque row (stop + hold at the end-of-stop pose) AND the open-content row
                pass on theta_t (initial stop and every in-flight stop alike); none passes -> `refuse`; the overshoot (0.5 v T + DS/2) running past the plan end
                -> MISSING (a shorter stop fails the rows, a longer one has no checked pose);
  retrace a = min(A_BACKUP, a_max over the retrace samples, open-content limit), halved up to RETRACE_HALVINGS times (never below A_MIN) until every branch point passes.
  11-16: a stop whose move (the segment, or the retrace) has remaining time <= T FINISHES the move (executor stop branch): end sample = the move's end, deceleration = a (not v / T,
  not 0.5 v T); BIMANUAL_LEGACY_STOP=1 restores the old model.
  Branch points of the transfer = every 0.25 s of its duration plus the end.  The in-flight stops ARE enumerated: at each branch point the recovery (stop, then continue
  to the target) is checked with its own stop time from the ladder (T = 0.3 s included, whose deceleration v / T can exceed a) and, in the accelerating half of the
  retrace, also T = v / a; the chosen T and end sample per branch point are recorded in admission['transfer']['per_branch'].
What "the task can continue" means in this code (R_K): the landing's witness (rule1_contents.landing_row): the planner's REMAINING plan from the target
is admitted by the static rows on theta_t (depth 1 <= K, 1 node <= NODE_CAP; proof under MODEL).  In addition an in-air target (hold_*, let_go) must
have Rule 1 viable on the NEXT step from it (the step is still refused if the arms still carry the object); a set-down on a support needs no such row
(it relieves the arms; the retry of the step is the orchestrator's next Rule 1 call).

    python -m world_model.v2.adapters.rule4_contents --trap box --cell 2.5-3.0 --seg carry --t 0.75 --proposer set_down|non_set_down
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math

import numpy as np

from world_model.v2.adapters import rule1_contents as RC
from world_model.v2.nero.contents import carry_cert as CC
from world_model.v2.stack import executor as EX
from world_model.v2.stack import theta as TH
from world_model.v2.stack import types as TY
from world_model.v2.stack.checks import admit as AD
from world_model.v2.stack.checks import open_content as OCN
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks import rule1 as R1
from world_model.v2.stack.checks import rule4 as R4
from world_model.v2.stack.checks.row import Row, missing
from world_model.v2.stack.proposer import mock as MK

# STOP_T0 (shortest stop time of the ladder) = TY.STOP_T_S: the stored backups' stop time, ONE ASSUMED source (stack/types.py), read at call time by _stop_t0()
RETRACE_HALVINGS = 3   # ASSUMED (to be raised to the user): the transfer's retrace acceleration may be halved up to 3 times (never below A_MIN) so that a stop at every branch point fits
def _stop_t0() -> float:
    return TY.STOP_T_S


STOP_K_MAX = 4         # ASSUMED (to be raised to the user): the stop time may be doubled up to 4 times (0.3 -> 4.8 s)

# the proposer's recovery skill (schema.RECOVERY_SKILLS) -> the target kind it is read as
SKILL_TARGET = {"set_down": "set_down_start", "retrace": "hold_lower", "hold": "hold_here", "let_go": "let_go"}


# ----------------------------------------------------------------------------------------------------------------------- state, targets, transfers
@dataclasses.dataclass(frozen=True)
class CarryState:
    ctx: object
    model: object
    seg: str
    t: float


@dataclasses.dataclass(frozen=True)
class Target:
    kind: str
    landing: str
    requires_next: bool
    support: str | None = None

    @property
    def name(self) -> str:
        return self.kind + (f":{self.support}" if self.support else "")


@dataclasses.dataclass(frozen=True)
class Transfer:
    """the opaque transfer motion (JSON-safe via asdict)."""
    target: str
    seg: str
    t0: float
    retrace_to: str | None
    stop_T: float | None
    a: float | None
    note: str = ""


def make_target(st: CarryState, kind: str, support: str | None = None) -> Target:
    names = st.model.names
    i = names.index(st.seg)
    if kind == "set_down_start":
        return Target(kind, "set_down:start", False)
    if kind == "set_down_support":
        return Target(kind, f"set_down:{support}", False, support)
    if kind == "hold_here":
        return Target(kind, f"hold:{st.seg}", True)
    if kind == "hold_lower":
        return Target(kind, f"hold:{names[max(i - 1, 0)]}", True)
    if kind == "let_go":
        return Target(kind, f"let_go:{st.seg}", True)
    raise ValueError(f"unknown target kind {kind!r}")


def _retrace_to(st: CarryState, tg: Target):
    names = st.model.names
    if tg.kind in ("set_down_start", "set_down_support"):
        return names[0]
    if tg.kind == "hold_lower":
        return names[max(names.index(st.seg) - 1, 0)]
    return None


# --- the transfer trajectory on the planned geometry samples (closed_chain only; fixed-base tasks)
@dataclasses.dataclass
class TP:
    """the opaque plan of the transfer program.  Trajectory on the arc length s of the planned path (g.offs, 1 cm samples):
    phase 1  the stop from the branch state (ib, v0): constant deceleration v0 / stop_T, ascending, ends at sample ie (0.5 v0 T + DS/2 beyond, rule1_contents' end-of-stop rule);
    phase 2  rest-to-rest retrace at object acceleration a from ie down to the start sample it of the target segment (absent for a hold target)."""
    seg: str
    t0: float
    ib: int
    v0: float
    stop_T: float | None
    ie: int | None
    a: float | None
    it: int | None
    problem: str = ""            # "" | "missing:<why>" | "refuse:<why>"
    L: float = 0.0
    a_stop: float | None = None  # 11-16: set when the initial stop FINISHES the segment (remaining time <= stop_T): the stop's deceleration = the plan's a (None = Hermite v0 / stop_T)
    per_branch: dict = dataclasses.field(default_factory=dict, compare=False, repr=False)    # tau -> the in-flight recovery admitted at that branch point (filled by transfer_row)

    @property
    def T_r(self) -> float:
        return 0.0 if (self.it is None or self.a is None or self.L <= 1e-12) else 2.0 * math.sqrt(self.L / self.a)

    @property
    def duration(self) -> float:
        return float((self.stop_T or 0.0) + self.T_r)


def _geo(model):
    g = model.geo
    return g, np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(g.offs, axis=0), axis=1))]


def ext_model(ctx, model):
    """a copy of the PathModel whose key set is EVERY geometry sample (the transfer's stops end at poses the stored model has no key for).  Registered in ctx.models under
    a + 1e-6 (a label only: that is the key ctx.env / the torque memo use; the collision rows and the landing witnesses use the original model)."""
    cache = ctx.__dict__.setdefault("_r4_ext", {})
    if model.a not in cache:
        g = model.geo
        n = len(g.offs)
        q = {f"g{i}": {s: g.q[s][i].tolist() for s in ("left", "right")} for i in range(n)}
        ext = dataclasses.replace(model, a=round(model.a + 1e-6, 6), cap=dict(model.cap, q=q), offs={f"g{i}": np.asarray(g.offs[i], float) for i in range(n)},
                                  order=[f"g{i}" for i in range(n)])
        ctx.models[round(ext.a, 6)] = ext
        cache[model.a] = ext
    return cache[model.a]


def _end_idx(ctx, model, seg, arc, ib, dist, direction):
    """sample at the END of a stop of travelled distance `dist` from ib (direction +1 ascending / -1 descending): the first sample at or beyond dist + DS/2; None = past the plan end
    (or, ascending, through the base move) -> no checked pose.  dist = 0 -> ib."""
    if dist <= 1e-12:
        return ib
    ds = ctx.CCP.DS
    g = model.geo
    if direction > 0:
        want = arc[ib] + dist + 0.5 * ds
        ie = next((i for i in range(ib, len(arc)) if arc[i] >= want), None)
        if ie is not None and ctx._post_of(g, seg, ie) != ctx._post_of(g, seg, ib):
            return None
        return ie
    want = arc[ib] - dist - 0.5 * ds
    ie = next((i for i in range(ib, -1, -1) if arc[i] <= want), None)
    if ie is not None and ctx._post_of(g, seg, ie) != ctx._post_of(g, seg, ib):
        return None
    return ie


def _choose_stop(ctx, model, ext, theta, seg, arc, ib, v, direction, it=None, a_brake=None, t_rem=None, end_idx=None, a_fin=None):
    """the stop time of a stop from speed v at sample ib: the first candidate T (ladder STOP_T0 * 2^k, plus v / a_brake when given), ascending (a longer stop overshoots farther),
    for which the torque row (stop + hold at the end-of-stop pose) AND the open-content row pass.  -> dict(status 'ok' | 'refuse' | 'missing', T, e, why).
    missing = the end-of-stop pose of a candidate is past the plan end / target start / through the base move (earlier candidates failed their rows)."""
    K = lambda idxs: [f"g{i}" for i in idxs]
    cands = sorted([_stop_t0() * 2 ** k for k in range(STOP_K_MAX + 1)] + ([v / a_brake] if a_brake and v > 1e-9 and v / a_brake >= _stop_t0() else []))
    last = None
    for T in cands:
        fin = (not EX.legacy_stop()) and t_rem is not None and end_idx is not None and a_fin is not None and v > 1e-9 and 1e-9 < t_rem <= T + 1e-9
        if fin:      # 11-16 (executor stop branch): a stop in the last T s of a rest-to-rest move lets the move finish: end = the move's end sample, deceleration = the move's own a
            e, acc = int(end_idx), float(a_fin)
        else:
            e, acc = _end_idx(ctx, model, seg, arc, ib, 0.5 * v * T, direction), v / T
        if e is None or (direction < 0 and it is not None and e < it):
            return dict(status="missing", T=T, e=e, why=f"the stop overshoot (0.5 v T = {0.5 * v * T * 1e3:.1f} mm, v = {v:.3f} m/s, T = {T:.2f} s) runs past the plan end / target start / base move: "
                                                         "the end-of-stop pose is not checked" + ("" if last is None else f" (shorter stops up to T = {last['T']:.2f} s fail the torque or open-content row)"))
        span = range(ib, e + 1) if direction > 0 else range(e, ib + 1)
        o = ctx.open_row(theta, acc, "open_content[stop]")
        ok = ctx.torque_row(ext, theta, K(span), acc, "stop").ok and ctx.torque_row(ext, theta, [f"g{e}"], 0.0, "hold").ok and (o is None or o.ok)
        last = dict(status="refuse", T=T, e=e, why="", finish=bool(fin), a_stop=acc)
        if ok:
            return dict(status="ok", T=T, e=e, why="", finish=bool(fin), a_stop=acc)
    last["why"] = f"no stop time <= {_stop_t0() * 2 ** STOP_K_MAX:.1f} s passes the torque and open-content rows"
    return last


def _state_at(ctx, model, seg, t):
    """EXACT branch state at time t of segment seg (not snapped to the 0.25 s grid): (ib, v) from the planned piece at the control period, as closed_chain._model_cc does at its grid times."""
    CCP, g = ctx.CCP, model.geo
    pc = model.cc.pieces[seg]
    kk = min(pc.n, max(0, int(round(t / CCP.DT))))
    i0, i1 = g.bounds[seg]
    ib = i0 + int(np.argmin(np.linalg.norm(g.offs[i0:i1 + 1] - pc.p[kk], axis=1)))
    vel = np.linalg.norm(np.gradient(pc.p, CCP.DT, axis=0), axis=1)
    return ib, (float(vel[kk]) if kk < pc.n else 0.0)


def build_transfer(st: CarryState, tg: Target, theta_t: TH.ThetaSet):
    """-> (Transfer record, RecoveryProgram of the transfer).  Stop time and retrace acceleration derived on theta_t (module doc); the end-of-stop pose is that of the stop time USED."""
    ctx, m, seg = st.ctx, st.model, st.seg
    to = _retrace_to(st, tg)
    prob = ""
    if ctx.planner != "closed_chain" or m.geo is None or ctx.td.base or (ctx.wp is not None and ctx.wp.base):
        prob = "missing:transfer rows are built for closed_chain plans without a base move only (scripted: no arm configurations along the recovery; base tasks / a plan with a base move: not built)"
        tp = TP(seg, float(st.t), 0, 0.0, None, None, None, None, prob)
        return Transfer(tg.name, seg, float(st.t), to, None, None, prob), TH.RecoveryProgram.fixed(f"transfer:{tg.name}", tp, tg.landing)
    g, arc = _geo(m)
    ext = ext_model(ctx, m)
    ib, v0 = _state_at(ctx, m, seg, st.t)
    lo, hi = RC._hull(theta_t)
    cont = ctx.spec_for(lo, hi).contents
    env = ctx.env(ext.a)
    ch = _choose_stop(ctx, m, ext, theta_t, seg, arc, ib, v0, +1, t_rem=float(m.T[seg]) - float(st.t), end_idx=int(g.bounds[seg][1]), a_fin=float(m.a))
    if ch["status"] == "missing":
        tp = TP(seg, float(st.t), ib, v0, None, None, None, None, "missing:" + ch["why"])
        return Transfer(tg.name, seg, float(st.t), to, None, None, tp.problem), TH.RecoveryProgram.fixed(f"transfer:{tg.name}", tp, tg.landing)
    if ch["status"] == "refuse":
        tp = TP(seg, float(st.t), ib, v0, ch["T"], ch["e"], None, None, f"refuse:{ch['why']} on [{lo:.3f}, {hi:.3f}] kg")
        return Transfer(tg.name, seg, float(st.t), to, ch["T"], None, tp.problem), TH.RecoveryProgram.fixed(f"transfer:{tg.name}", tp, tg.landing)
    stop_T, ie = ch["T"], ch["e"]
    it = None if to is None else int(g.bounds[to][0])
    a_tr = None
    if it is not None:
        keys = [f"g{i}" for i in range(it, ie + 1)]
        a_t = env.a_max(cont, keys, PAY.MODE, limit=PAY.U_ADMIT)
        a_open = OCN.a_open_of(ctx.spec_for(lo, hi)) if "open_content" in ctx.td.rows else math.inf
        a_tr = None if a_t is None else min(R1_A_BACKUP, a_t, a_open)
        if a_tr is None or a_tr < RC.A_MIN:
            prob = f"refuse:no retrace acceleration >= A_MIN = {RC.A_MIN} on [{lo:.3f}, {hi:.3f}] kg (a_max {a_t}, a_open {a_open})"
            a_tr = None
    L = 0.0 if it is None else float(arc[ie] - arc[it])
    tp = TP(seg, float(st.t), ib, v0, stop_T, ie, a_tr, it, prob, L=L, a_stop=(ch["a_stop"] if ch.get("finish") else None))
    if it is not None and a_tr is not None:
        # the retrace acceleration is the largest of a_tr * 2^-j (j = 0..RETRACE_HALVINGS, >= A_MIN) for which the transfer passes at EVERY branch point of the transfer (a stop in the
        # accelerating half must fit the torque limit and the remaining path); none passes -> the last one, whose failing rows are reported
        for j in range(RETRACE_HALVINGS + 1):
            aj = a_tr * 0.5 ** j
            if aj < RC.A_MIN and j > 0:
                break
            tp = TP(seg, float(st.t), ib, v0, stop_T, ie, aj, it, prob, L=L, a_stop=(ch["a_stop"] if ch.get("finish") else None))
            if all(transfer_row(ctx, m, tp, theta_t, tau).ok for tau in R1.branch_times(tp.duration)):
                break
        a_tr = tp.a
    return Transfer(tg.name, seg, float(st.t), to, stop_T, a_tr, prob), TH.RecoveryProgram.fixed(f"transfer:{tg.name}", tp, tg.landing)


R1_A_BACKUP = RC.A_BACKUP


class _IdxCtx:
    """stand-in `self` for rule1_contents.Ctx.collision_row with the geometry sample indices of the transfer's recovery (descending samples included)."""

    def __init__(self, ctx, idxs):
        self._c, self._i = ctx, idxs

    def recovery_indices(self, *a, **k):
        return self._i

    def __getattr__(self, n):
        return getattr(self._c, n)


def _coll(ctx, model, seg, idxs):
    return RC.Ctx.collision_row(_IdxCtx(ctx, idxs), model, seg, None, None)


def transfer_row(ctx, model, tp: TP, theta, tau):
    """check_motion for the transfer program: the RECOVERY FROM THE TRANSFER'S STATE at time tau (Rule 1 applied to the transfer, one row per branch time of the transfer):
      phase 1 (stopping)        finish the stop (end pose fixed), then retrace to the target
      phase 2 accelerating half stop (stop time = first of the ladder whose torque row passes; its OWN end-of-stop pose, past the target start / plan end -> MISSING), then continue the retrace
      phase 2 decelerating half continue the planned deceleration to rest at the target
    rows: stop torque + hold at the end-of-stop pose + open-content + retrace torque + open-content + collision (stop + retrace samples, rule1_contents' swept row)."""
    nm = f"transfer@tau={tau:.2f}"
    if tp.problem.startswith("missing:"):
        return missing(nm, tp.problem[8:])
    if tp.problem.startswith("refuse:"):
        return Row(nm, False, margin=None, unit="", guarantee="refuted", why=tp.problem[7:])
    seg = tp.seg
    g, arc = _geo(model)
    ext = ext_model(ctx, model)
    K = lambda idxs: [f"g{i}" for i in idxs]
    rows = []
    a = tp.a
    Ts = tp.stop_T
    if tau <= Ts + 1e-9:                                                  # phase 1
        v = tp.v0 * max(0.0, 1.0 - tau / Ts)
        p = arc[tp.ib] + tp.v0 * tau - tp.v0 * tau * tau / (2 * Ts)
        ibk = min(int(np.argmin(np.abs(arc - p))), tp.ie)
        stops = [(+1, ibk, tp.ie, tp.a_stop if tp.a_stop is not None else tp.v0 / Ts)]
        tp.per_branch[round(float(tau), 6)] = dict(t=float(tau), mode="finish_stop_then_retrace" if tp.it is not None else "finish_stop", T=float(Ts - tau), end_idx=int(tp.ie), v=float(v),
                                                   begin_idx=int(ibk))
        ie = tp.ie
        down = False
    else:
        tau2 = tau - Ts
        Tr, L = tp.T_r, tp.L
        accel = tau2 <= Tr / 2 + 1e-9
        s = 0.5 * a * tau2 * tau2 if accel else L - 0.5 * a * (Tr - tau2) ** 2
        v = a * (tau2 if accel else max(Tr - tau2, 0.0))
        p = arc[tp.ie] - s
        ibk = int(np.argmin(np.abs(arc - p)))
        ibk = max(min(ibk, tp.ie), tp.it)
        down = True
        if not accel:                                                     # continue the planned deceleration to rest at the target start
            tp.per_branch[round(float(tau), 6)] = dict(t=float(tau), mode="continue_decel_to_target", T=float(max(Tr - tau2, 0.0)), end_idx=int(tp.it), v=float(v), begin_idx=int(ibk))
            rows.append(ctx.torque_row(ext, theta, K(range(tp.it, ibk + 1)), a, "decel_to_target"))
            o = ctx.open_row(theta, a, "open_content[move]")
            if o is not None:
                rows.append(o)
            rows.append(_coll(ctx, model, seg, list(range(ibk, tp.it - 1, -1))))
            return RC._merge(nm, rows)
        ch = _choose_stop(ctx, model, ext, theta, seg, arc, ibk, v, -1, it=tp.it, a_brake=a, t_rem=float(Tr - tau2), end_idx=int(tp.it), a_fin=float(a))
        if ch["status"] == "missing":
            return missing(nm, f"in-flight stop at tau = {tau:.2f} s: " + ch["why"])
        ie, Tk = ch["e"], ch["T"]
        tp.per_branch[round(float(tau), 6)] = dict(t=float(tau), mode="stop_then_retrace", T=float(Tk), end_idx=int(ie), v=float(v), begin_idx=int(ibk), finish=bool(ch.get("finish")))
        stops = [(-1, ibk, ie, ch["a_stop"])]
    d, i0, i1, acc = stops[0]
    span = list(range(i0, i1 + 1)) if d > 0 else list(range(i1, i0 + 1))
    rows.append(ctx.torque_row(ext, theta, K(span), acc, "stop"))
    rows.append(ctx.torque_row(ext, theta, [f"g{i1}"], 0.0, "hold_after_stop"))
    o = ctx.open_row(theta, acc, "open_content[stop]")
    if o is not None:
        rows.append(o)
    idxs = (list(range(i0, i1 + 1)) if d > 0 else list(range(i0, i1 - 1, -1)))
    if tp.it is not None:
        end = i1
        rows.append(ctx.torque_row(ext, theta, K(range(tp.it, end + 1)), a, "retrace"))
        o = ctx.open_row(theta, a, "open_content[move]")
        if o is not None:
            rows.append(o)
        idxs += list(range(end - 1, tp.it - 1, -1))
    rows.append(_coll(ctx, model, seg, idxs))
    return RC._merge(nm, rows)


def programs_at_for(st: CarryState, theta_t: TH.ThetaSet):
    """-> programs_at(target, phase) for rule4.check_target (and the transfers it built, by target name)."""
    transfers = {}
    base_cm, base_lo = st.ctx.callables(st.model, st.seg)

    def programs_at(tg: Target, phase: str):
        ctx, m = st.ctx, st.model
        if phase == "transfer":
            tr, prog = build_transfer(st, tg, theta_t)
            transfers[tg.name] = (tr, prog.bins[0][1])
            return [prog]
        if phase == "here":
            seg = _retrace_to(st, tg) or st.seg
            return [TH.RecoveryProgram.fixed(f"rest:{tg.name}", RC.MotionRef("none", seg), tg.landing)]
        if phase == "next":
            if tg.kind == "let_go":
                return None                                              # nothing is left to continue after letting go in the air
            seg = _retrace_to(st, tg) or st.seg
            progs, _ = ctx.programs(m, seg, theta_t)
            cm, lo = ctx.callables(m, seg)
            return progs, R1.branch_times(m.T[seg]), cm, lo
        raise ValueError(phase)
    return programs_at, transfers


def continuation(st: CarryState, tg: Target, theta_t: TH.ThetaSet, adm=None) -> str:
    """what the record says about continuing the task from the landing, from a Rule 1 result that was actually run (never assumed)."""
    m, ctx = st.model, st.ctx
    if tg.requires_next:                      # the next-step admission of check_target (part `next_step`): the stored step re-run from ITS start
        ok = adm is not None and adm["parts"].get("next_step") is not None
        return ("rule1_admitted_whole_step_from_t0 (proxy: the stored step re-run from its start, not a resume from the hold pose)" if ok else
                "static_only (Rule 1 refuses the next step from the in-air target)")
    if tg.kind != "set_down_start":
        return "unknown (no next-step model for this landing)"
    seg = m.names[0]
    progs, fb = ctx.programs(m, seg, theta_t)
    cm, lo = ctx.callables(m, seg)
    a1, per = R1.admit_step(0, theta_t, progs, R1.branch_times(m.T[seg]), cm, lo)
    if a1 is not None:
        return "rule1_admitted_step0 (later steps not run)"
    return "static_only (Rule 1 refuses the next step: safe park, the task may not continue)"


def make_check(st: CarryState, theta_t: TH.ThetaSet):
    programs_at, transfers = programs_at_for(st, theta_t)
    base_cm, lo = st.ctx.callables(st.model, st.seg)

    def cm(plan, th, t):
        return transfer_row(st.ctx, st.model, plan, th, t) if isinstance(plan, TP) else base_cm(plan, th, t)

    def check(tg: Target):
        transfers.pop(tg.name, None)
        _, prog = build_transfer(st, tg, theta_t)                         # to know the duration: the transfer's own branch times (every 0.25 s + the end)
        tp = prog.bins[0][1]
        times = R1.branch_times(tp.duration) if not tp.problem else [0.0]
        ok, rows, adm = R4.check_target(tg, None, theta_t, programs_at, times, cm, lo)
        if adm is not None:
            tr, tp_ = transfers[tg.name]
            adm["transfer"] = dict(dataclasses.asdict(tr), branch_times=times, duration=tp.duration, t_state=float(st.t),
                                   per_branch=[tp_.per_branch[k] for k in sorted(tp_.per_branch)])
            adm["continuation"] = continuation(st, tg, theta_t, adm)
            adm["assumptions"] += [dict(name="STOP_K_MAX", value=STOP_K_MAX, tag="ASSUMED", src="rule4_contents: stop time ladder STOP_T0 * 2^k, k <= STOP_K_MAX (to be raised to the user)"),
                                   dict(name="RETRACE_HALVINGS", value=RETRACE_HALVINGS, tag="ASSUMED", src="rule4_contents: retrace acceleration halved at most this often, never below A_MIN"),
                                   dict(name="stop_finish", value="a stop (initial: remaining time of the segment; in-flight in the accelerating half of the retrace: remaining time of the retrace) with remaining time <= T finishes the move: end = the move's end sample, deceleration = the plan's a (initial) / the retrace a (in-flight)", tag="MODEL",
                                        src="executor stop branch 11-16 (rest-to-rest references); BIMANUAL_LEGACY_STOP=1 restores the 0.5 v T Hermite overshoot"),
                                   dict(name="STOP_T0", value=_stop_t0(), tag="ASSUMED", src="rule4_contents: shortest stop time = the stored backups' stop time (carry backup_phases)")]
        return ok, rows, adm
    return check, transfers


# ----------------------------------------------------------------------------------------------------------------------- the mock proposer
class MockTargetProposer:
    """scripted target proposals (NOT a model).  The recovery skills come from the mock proposer's fixture reply (stack/proposer/mock.fixture_plan: its
    `recoveries` = set_down on every motion step); mock.py has no target vocabulary of its own, so skill -> target is SKILL_TARGET.
    allow_set_down=True : round 0 = [hold, retrace] (in-air), round 1 = the fixture's skill (set_down), later rounds nothing new
    allow_set_down=False: round 0 = [hold, retrace], round 1 = [let_go], then only repeats -> the loop must fail  (every proposed target is non-set-down)"""

    def __init__(self, task, allow_set_down=True):
        self.task, self.allow = task, bool(allow_set_down)
        fx = MK.fixture_plan(task)
        self.fixture_skills = sorted({r["skill"] for v in fx["recoveries"].values() for r in v})
        self.calls = []

    def __call__(self, state: CarryState, theta_t, failing_rows):
        rnd = len(self.calls)
        self.calls.append([r.name if hasattr(r, "name") else str(r) for r in failing_rows][:6])
        if rnd == 0:
            skills = ["hold", "retrace"]
        elif rnd == 1:
            skills = [s for s in self.fixture_skills if s == "set_down"] if self.allow else ["let_go"]
        else:
            skills = ["hold"] if not self.allow else []
        return [make_target(state, SKILL_TARGET[s]) for s in skills]


# ----------------------------------------------------------------------------------------------------------------------- the trap
def build_trap(task="box", cell=(2.5, 3.0), seg="carry", t=0.75, planner=None):
    """a state mid-carry on a heavy cell where Rule 1 refuses (box contents 2.5-3.0 kg: every step is refused, rule1_contents).
    -> dict(state, theta_t, rule1_refused: bool, failing_rows: [Row], plan_action, a_plan)."""
    ctx = RC.get_ctx(task, planner)
    lo, hi = RC.cell_total(task, cell)
    theta_t = RC.theta_box(lo, hi)
    adm0, model = RC.plan_admission(ctx, cell, lo, hi)
    if model is None:
        raise RuntimeError(f"the plan itself is refused on this cell ({adm0.action}): the trap needs a plan admitted at a reduced acceleration")
    if not 0.0 <= t <= model.T[seg]:
        raise ValueError(f"t = {t} is outside segment {seg!r} (duration {model.T[seg]:.2f} s)")
    st = CarryState(ctx, model, seg, float(t))           # the EXACT time: the transfer starts from the planned state at t (not snapped to the 0.25 s grid)
    progs, fb = ctx.programs(model, seg, theta_t)
    cm, lo_ = ctx.callables(model, seg)
    adm, per = R1.admit_step(model.names.index(seg), theta_t, progs, R1.branch_times(model.T[seg]), cm, lo_, fb, None)
    fails = [r for rs in per.values() for r in R1.failing(rs)]
    return dict(state=st, theta_t=theta_t, rule1_refused=adm is None, failing_rows=fails, plan_action=adm0.action, a_plan=model.a)


def run_trap(allow_set_down: bool, **kw):
    """the Rule 4 loop on the trap with the mock proposer -> (accepted Target | None, log, trap dict)."""
    tr = build_trap(**kw)
    st, th = tr["state"], tr["theta_t"]
    check0, _ = make_check(st, th)

    def check(tg):
        ok, rows, adm = check0(tg)
        if adm is not None:
            tr["admission"] = adm
        return ok, rows, adm
    prop = MockTargetProposer(st.ctx.task, allow_set_down)
    acc, log = R4.loop(st, th, prop, check, failing_rows=tr["failing_rows"][:12])
    return acc, log, tr


def print_log(acc, log, tr):
    print(f"trap: task={tr['state'].ctx.task} theta_t={[(c.bounds[RC.TOTAL_FIELD].lo, c.bounds[RC.TOTAL_FIELD].hi) for c in tr['theta_t'].cells]} kg seg={tr['state'].seg} "
          f"t={tr['state'].t} a_plan={tr['a_plan']} plan_action={tr['plan_action']!r} rule1_refused={tr['rule1_refused']} (failing rows: {len(tr['failing_rows'])})")
    for e in log:
        print(f"round {e['round']}: candidates={e['candidates']} accepted={e['accepted']}" + (f" note={e['note']}" if e.get("note") else ""))
        for v in e["verdicts"]:
            print(f"   {v['target']}: ok={v['ok']} guarantee={v['guarantee']}")
            for f in v["failing"][:3]:
                print(f"      FAIL {f}")
    print("ACCEPTED:", None if acc is None else acc.name)
    if acc is not None and "admission" in tr:
        ad = tr["admission"]
        t_ = ad["transfer"]
        print(f"  guarantee={ad['guarantee']} continuation={ad['continuation']!r}")
        print(f"  transfer: t_state={t_['t_state']} stop_T={t_['stop_T']} a={t_['a']} duration={t_['duration']:.2f} s, {len(t_['branch_times'])} branch points {t_['branch_times'][:4]}..{t_['branch_times'][-1]}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trap", default="box")
    ap.add_argument("--cell", default="2.5-3.0")
    ap.add_argument("--seg", default="carry")
    ap.add_argument("--t", type=float, default=0.75)
    ap.add_argument("--proposer", choices=("set_down", "non_set_down"), default="set_down")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    lo, hi = (float(x) for x in a.cell.split("-"))
    acc, log, tr = run_trap(a.proposer == "set_down", task=a.trap, cell=(lo, hi), seg=a.seg, t=a.t)
    print_log(acc, log, tr)
    if a.json:
        json.dump(dict(accepted=None if acc is None else acc.name, log=log), open(a.json, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
