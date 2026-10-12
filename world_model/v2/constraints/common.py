"""constraints/common.py -- state container for the expressiveness study, the linear toy decision model, and the three checks.
SIMULATION / SAVED DATA ONLY.

PoseState         poses {name: (R, p)} (+ optional uncertainty {name: Unc}); entities read their pose from it.
state.moved(jac, du)   the state the LINEAR toy model predicts at u = u0 + du: p += Jp du; the axis (x, y components) += Jc du, R rotated minimally.
Task              protocol every task module implements (see tasks/*.py).
check_margin      margin equivalence: IR margin vs a hand-coded geometric reference on N states (max |err|, sign agreement).
check_rows        rows vs the margin: (a) rows-implied slack vs margin along random du, (b) finite-difference gradient of the margin vs the
                  active-row gradient, (c) soundness (rows feasible => margin >= 0), plus timing of margin / rows (ms).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from world_model.v2.tray_rep.sets import ir
from world_model.v2.tray_rep.sets.ir import Entity, Pose, Unc
from world_model.v2.constraints import ir_ext


def min_rot(c0, c1):
    """rotation matrices (..., 3, 3) of the minimal rotation taking unit c0 to unit c1."""
    c0, c1 = np.asarray(c0, float), np.asarray(c1, float)
    v = np.cross(c0, c1)
    d = (c0 * c1).sum(-1)
    Kx = np.zeros(v.shape[:-1] + (3, 3))
    Kx[..., 0, 1], Kx[..., 0, 2], Kx[..., 1, 0] = -v[..., 2], v[..., 1], v[..., 2]
    Kx[..., 1, 2], Kx[..., 2, 0], Kx[..., 2, 1] = -v[..., 0], -v[..., 1], v[..., 0]
    return np.eye(3) + Kx + (Kx @ Kx) / (1.0 + d)[..., None, None]


class PoseState:
    def __init__(self, poses: dict, unc: dict | None = None, extra: dict | None = None):
        self.poses = {k: (np.asarray(R, float), np.asarray(p, float)) for k, (R, p) in poses.items()}
        self.unc = unc or {}
        self.extra = extra or {}

    def moved(self, jac, du):
        du = np.asarray(du, float)
        poses = dict(self.poses)
        for name, J in jac.items():
            if name == "u0" or name not in poses:
                continue
            R, p = poses[name]
            if "p" in J:
                Jp = np.asarray(J["p"], float)
                p = p + np.einsum("...iu,u->...i", Jp, du)
            if "axis" in J:
                Jc = np.asarray(J["axis"], float)
                c0 = R[..., :, 2]
                cxy = c0[..., :2] + np.einsum("...iu,u->...i", Jc, du)
                cz = np.sqrt(np.maximum(1.0 - (cxy ** 2).sum(-1), 1e-12))
                c1 = np.concatenate([cxy, cz[..., None]], -1)
                R = min_rot(c0, c1) @ R
            poses[name] = (R, p)
        s = PoseState.__new__(PoseState)
        s.poses, s.unc, s.extra = poses, self.unc, self.extra
        return s


def ent(name, shape, parent=None, with_unc=True):
    """Entity whose pose / uncertainty come from a PoseState."""
    return Entity(name, shape, (lambda s, n=name: Pose(*s.poses[n])),
                  (lambda s, n=name: s.unc.get(n, Unc())) if with_unc else None, parent)


@dataclass
class Task:
    """protocol. `states` are PoseStates; `ref(state)` is the hand-coded margin (float) in the same units as the IR margin."""
    name: str
    spec_dict: dict
    build: Callable            # () -> (spec Node, ents dict)
    sample: Callable           # (n, rng) -> list[PoseState]
    ref: Callable              # state -> float   (hand-coded geometric margin)
    toy: Callable              # (state, rng) -> jac dict (with "u0")
    toy_norot: Callable | None = None   # same without the rotation (axis) Jacobians, to separate translation-only row errors
    notes: dict = field(default_factory=dict)


# ====================================================================================== checks
def check_margin(task: Task, states, robust=False, tol_sign_band=1e-9):
    spec, _ = task.build()
    m_ir = np.array([ir.margin(spec, s, robust) for s in states])
    m_ref = np.array([task.ref(s) for s in states])
    err = np.abs(m_ir - m_ref)
    decided = np.abs(m_ref) > tol_sign_band
    sign_agree = float(np.mean(np.sign(m_ir[decided]) == np.sign(m_ref[decided]))) if decided.any() else float("nan")
    pos = m_ref > 0
    near = np.abs(m_ref) < 0.05
    return dict(n=len(states), max_abs_err=float(err.max()), mean_abs_err=float(err.mean()), sign_agree=sign_agree,
                max_abs_err_pos=float(err[pos].max()) if pos.any() else float("nan"),
                max_abs_err_near=float(err[near].max()) if near.any() else float("nan"), n_near=int(near.sum()),
                frac_pos=float(np.mean(m_ref > 0)), frac_neg=float(np.mean(m_ref < 0)), m_ref_range=[float(m_ref.min()), float(m_ref.max())],
                err=err, m_ir=m_ir, m_ref=m_ref)


def _row_slack(A, b, u):
    return b - A @ u


def check_rows(task: Task, states, rng, n_du=12, du_scale=0.3, fd_h=1e-6, robust=False, inside_only=True, spec=None, toy_fn=None):
    """per state: u0 random in [-1, 1]^nu, jac from the task, rows at u0; compare with the margin of the state at u = u0 + du."""
    if spec is None:
        spec, _ = task.build()
    slack_err, sound_viol, n_sound = [], 0, 0
    n_pos, n_pos_feas = 0, 0
    tfn = toy_fn or task.toy
    grad_err, grad_rel, grad_n = [], [], 0
    by_tag = {}
    row_margin0 = []
    for s in states:
        jac = tfn(s, rng)
        u0 = jac["u0"]
        try:
            A, b, tags = ir.rows(spec, s, jac, robust)
        except ir.NotConvex:
            return dict(rows="NotConvex")
        if len(b) == 0:
            continue
        m0 = ir.margin(spec, s, robust)
        sl0 = _row_slack(A, b, u0)
        if m0 > 0:
            row_margin0.append(sl0.min() - m0)
        # (a) along random du
        for _ in range(n_du):
            du = rng.normal(size=len(u0)) * du_scale
            su = s.moved(jac, du)
            mu = ir.margin(spec, su, robust)
            rs = _row_slack(A, b, u0 + du).min()
            if (not inside_only) or (m0 > 0 and mu > 0 and rs > 0):
                slack_err.append(abs(rs - mu))
            n_sound += 1
            if rs >= 0 and mu < -1e-9:
                sound_viol += 1
            if mu > 0:
                n_pos += 1
                n_pos_feas += rs >= 0
        # (b) FD gradient of the margin at u0 vs the active-row gradient
        if m0 > 0:
            order = np.argsort(sl0)
            if len(sl0) < 2 or sl0[order[1]] - sl0[order[0]] > 1e-4:           # not at a kink
                g_fd = np.array([(ir.margin(spec, s.moved(jac, fd_h * e), robust) - ir.margin(spec, s.moved(jac, -fd_h * e), robust))
                                 / (2 * fd_h) for e in np.eye(len(u0))])
                g_row = -A[order[0]]
                grad_err.append(np.abs(g_fd - g_row).max())
                grad_rel.append(np.linalg.norm(g_fd - g_row) / max(np.linalg.norm(g_fd), 1e-9))
                grad_n += 1
                by_tag.setdefault(str(tags[order[0]][0]), []).append(grad_rel[-1])
    f = lambda a, fn: float(fn(a)) if len(a) else float("nan")
    return dict(rows="ok", n_states=len(states), n_rows=int(len(b)), slack_max_err=f(slack_err, np.max), slack_med_err=f(slack_err, np.median),
                grad_max_abs_err=f(grad_err, np.max), grad_med_rel_err=f(grad_rel, np.median), grad_n=grad_n,
                sound_violations=int(sound_viol), sound_n=int(n_sound), completeness=(n_pos_feas / n_pos if n_pos else float('nan')), n_margin_pos=int(n_pos),
                grad_rel_by_active_tag={k: [float(np.median(v)), len(v)] for k, v in by_tag.items()}, row0_minus_margin0_max=f(np.abs(row_margin0), np.max))


def time_ms(fn, reps=30, warm=3):
    for _ in range(warm):
        fn()
    t = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t) / reps * 1e3


def check_time(task: Task, states, rng, reps=20):
    spec, _ = task.build()
    s = states[len(states) // 2]
    jac = task.toy(s, rng)
    ms_margin = time_ms(lambda: ir.margin(spec, s, False), reps)
    ms_margin_rob = time_ms(lambda: ir.margin(spec, s, True), reps)
    ms_status = time_ms(lambda: ir.status(spec, s), reps)
    try:
        ms_rows = time_ms(lambda: ir.rows(spec, s, jac, False), reps)
    except ir.NotConvex:
        ms_rows = float("nan")
    return dict(ms_margin=ms_margin, ms_margin_robust=ms_margin_rob, ms_status=ms_status, ms_rows=ms_rows)


def check_rows_per_atom(task: Task, states, rng, toy_fn=None, **kw):
    """rows-vs-margin for every top-level conjunct of the spec on its own (And of And flattened one level)."""
    spec, _ = task.build()
    kids = []
    def walk(n):
        if isinstance(n, ir.And):
            for c in n.c:
                walk(c)
        else:
            kids.append(n)
    walk(spec)
    out = {}
    for i, k in enumerate(kids):
        tag = getattr(k, "tag", "") or type(k).__name__
        key = f"{i}:{type(k).__name__}:{tag}"
        r = check_rows(task, states, np.random.default_rng(i + 1), spec=k, toy_fn=toy_fn, **kw)
        out[key] = r
    return out
