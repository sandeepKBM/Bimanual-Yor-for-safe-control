"""checks/admit.py -- static admission of a plan on a contents set: action = two arms / two arms, slow / refuse.  SIMULATION ONLY.

Reproduces nero/contents/carry_cert.static_cell's decision logic (rows are proof under MODEL, exact vertex checks):
  torque_payload   u <= U_ADMIT at the path samples of every move (checks/payload)
  open_content     planned peak acceleration <= the static open-content limit (spill / lid slide)  (checks/open_content)
  support_dest     total COM over the destination support shape (checks/support)
Action (ASSUMED rule, user-confirmed): two arms at a_nom if u(a_nom) <= U_ADMIT and a_nom <= a_open; else two arms slowed to a* = min(a_max(u <= U_ADMIT),
a_open) if a* >= A_MIN; else refuse.  A destination that fails the support row refuses ("refuse (placement)").  The one-arm utilisation is reported only.
Inputs are passed in: `spec` (ObjectSpec-like: contents, open_limits, empty_mass, empty_com), `path` (PathSamples: samples, per-segment keys, captured arm
configurations, object displacement), `backend` (make_envelope(spec, cap, offs), o_ref()), `support` = ((x0, x1), (y0, y1)) of the destination support top.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from world_model.v2.stack.checks import coverage as COV
from world_model.v2.stack.checks import open_content as OCN
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks import support as SUP
from world_model.v2.stack.checks.row import Row, combine, missing
from world_model.v2.stack.tasks import TaskDef
from world_model.v2.stack.types import Verdict

A_NOM = 0.5        # m/s^2, ASSUMED nominal peak acceleration (nero/contents/envelope.py A_NOM; adapter test asserts equality)
A_MIN = 0.05       # m/s^2, ASSUMED: slower than this = refuse (the task stalls)  (carry_cert.A_MIN)
N_PER_MOVE = 6     # path samples per move


@dataclass
class PathSamples:
    offs: dict          # key -> object offset [m] (3,)
    per_seg: dict       # segment -> [keys] (start key first)
    cap: dict           # captured arm configurations per key
    disp: np.ndarray    # object displacement start -> goal [m]


@dataclass
class Admission:
    verdict: Verdict
    action: str
    a_admit: float | None
    a_open: float
    u_nom: float
    u_one_nom: float
    rows: list = field(default_factory=list)
    # fields carried over from carry_cert.static_cell's dict, so step 2 can switch the callers without reshaping
    worst_nom: dict = field(default_factory=dict)       # binding vertex of u_nom (wp, m, c, Fz, twist)
    worst_one: dict = field(default_factory=dict)       # binding detail of the one-arm utilisation (wp, m, c, worst_joint, binding)
    contents_m: tuple = ()                              # (m_lo, m_hi) of the contents set [kg]
    seg_rows: dict = field(default_factory=dict)        # {segment: {row name: dict}} in static_cell's "rows" shape (plus "release": {"support_dest": ...})
    planner: str = "scripted"
    why: str = ""                                       # set when the planner refused (no path)


def path_offsets(t: TaskDef, moves=None):
    """object offsets along the planned moves: N_PER_MOVE samples per move (straight line, smoothstep timing).  -> ({key: offset}, {seg: [keys]})"""
    offs, per_seg, cur = {"p0": np.zeros(3)}, {}, np.zeros(3)
    k = 0
    for seg, dl in (t.moves if moves is None else moves):
        keys = []
        for i in range(1, N_PER_MOVE + 1):
            k += 1
            offs[f"p{k}"] = cur + np.asarray(dl, float) * i / N_PER_MOVE
            keys.append(f"p{k}")
        per_seg[seg] = ["p0" if not per_seg else list(per_seg.values())[-1][-1]] + keys
        cur = cur + np.asarray(dl, float)
    return offs, per_seg


def admit(t: TaskDef, spec, path: PathSamples, backend, support, a_nom: float = A_NOM, coverage=None) -> Admission:
    """coverage (11-09, probe-selection step 5): a checks.coverage.Coverage.  Given, an UNCOVERED phenomenon (enforce mode) refuses before any row ("refuse (uncovered)",
    a missing `coverage` row); covered, the coverage row is listed with the others.  The caller builds `spec` on the coverage's Theta_t (coverage.contents_mass).
    None (default) = the pre-11-09 behaviour, rows unchanged (golden)."""
    if coverage is not None and coverage.refuses:
        return refused_uncovered(t, spec, coverage)
    env = backend.make_envelope(spec, path.cap, path.offs)
    cont = spec.contents
    a_open = OCN.a_open_of(spec)
    keys = list(path.offs)
    u_nom, det_nom = env.worst(cont, keys, a_nom, PAY.MODE)
    u_one, det_one = env.worst(cont, keys, a_nom, "one")                 # reported only (one-arm carry not built)
    if u_nom <= PAY.U_ADMIT and a_nom <= a_open:
        a_adm, action = a_nom, "two arms"
    else:
        a_t = PAY.a_max(env, cont, keys)
        a_adm = None if a_t is None else min(a_t, a_open)
        action = "two arms, slow" if (a_adm is not None and a_adm >= A_MIN) else "refuse"
        if action == "refuse":
            a_adm = None
    aa = a_adm if a_adm is not None else a_nom
    rows: list[Row] = []
    for seg, ks in path.per_seg.items():
        rows.append(PAY.torque_row(env, cont, ks, aa, name=f"torque_payload[{seg}]"))
        if a_open < np.inf and "open_content" in t.rows:
            rows.append(OCN.open_content_row(aa, a_open, name=f"open_content[{seg}]"))
    dest = np.asarray(backend.o_ref(), float)[:2] + np.asarray(path.disp, float)[:2]
    sup_row = SUP.support_row(t, dest, support[0], support[1], spec.empty_mass, spec.empty_com, cont)
    rows.append(sup_row)
    if coverage is not None:
        rows.append(COV.row(coverage))
    if action != "refuse" and not sup_row.ok:
        action, a_adm = "refuse (placement)", None
    seg_rows = _seg_rows(rows, list(path.per_seg), sup_row)
    return Admission(combine(rows, "admit"), action, a_adm, float(a_open), float(u_nom), float(u_one), rows,
                     worst_nom={k: v for k, v in det_nom.items() if k in _KEYS_NOM}, worst_one={k: v for k, v in det_one.items() if k in _KEYS_ONE},
                     contents_m=(float(cont.m_lo), float(cont.m_hi)), seg_rows=seg_rows)


_KEYS_NOM = ("wp", "m", "c", "Fz", "twist")
_KEYS_ONE = ("wp", "m", "c", "worst_joint", "binding")


def _seg_rows(rows, segs, sup_row):
    """the rows in static_cell's dict shape: {seg: {"torque_payload": {...}, "open_content": {...}}, "release": {"support_dest": {...}}}; the support row is also
    filed under the last segment (the place move), as static_cell does."""
    out: dict = {}
    for r in rows:
        if r.name == "support_dest":
            continue
        base, _, seg = r.name.partition("[")
        seg = seg.rstrip("]")
        if base == "torque_payload":
            out.setdefault(seg, {})[base] = dict(ok=r.ok, u_max=round(r.worst["u_max"], 4), u_best_split=round(r.worst["u_best_split"], 4), a=r.worst["a"],
                                                 n_samples=r.worst["n_samples"], worst={k: v for k, v in r.worst.items() if k in _KEYS_NOM},
                                                 guarantee=r.assumptions.get("model", ""))
        else:
            out.setdefault(seg, {})[base] = dict(ok=r.ok, a_plan=r.worst["a_plan"], a_open=r.worst["a_open"], guarantee=r.assumptions.get("model", ""))
    sd = dict(ok=sup_row.ok, margin_m=round(sup_row.margin, 4), worst={k: sup_row.worst[k] for k in ("m", "c") if k in sup_row.worst},
              region=sup_row.worst.get("region"), shape=sup_row.worst.get("shape"), guarantee=sup_row.assumptions.get("model", ""))
    out["release"] = dict(support_dest=sd)
    if segs:
        out.setdefault(segs[-1], {})["support_dest"] = sd
    return out


def refused(t: TaskDef, spec, why: str, planner: str = "closed_chain") -> Admission:
    """the planner refused the goal (no path, e.g. closed_chain.PlanRefused): no admission rows, action "refuse (unreachable)" -- the counterpart of
    carry_cert.refused_static.  The verdict is not admitted and its guarantee is "missing" (a planner failure certifies nothing)."""
    cont = spec.contents
    r = missing("plan", why)
    return Admission(combine([r], "admit"), "refuse (unreachable)", None, float(OCN.a_open_of(spec)), float("inf"), float("inf"), [r],
                     contents_m=(float(cont.m_lo), float(cont.m_hi)), planner=planner, why=why)


def refused_uncovered(t: TaskDef, spec, coverage) -> Admission:
    """a phenomenon of PHI(task) is neither measured nor on an approved fallback (enforce mode): refuse; the coverage row is MISSING (never admits)."""
    cont = spec.contents
    r = COV.row(coverage)
    return Admission(combine([r], "admit"), "refuse (uncovered)", None, float(OCN.a_open_of(spec)), float("inf"), float("inf"), [r],
                     contents_m=(float(cont.m_lo), float(cont.m_hi)), planner="coverage", why=r.why)
