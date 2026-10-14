"""ctl.py -- ZonoCtl: the unmodified FL-MPC (phase2.Ctl2: tray tilt cone, phase1.Ctl plumbing) + perceived-zonotope HOCBF rows as hard
constraints, injected through FLMPC.step(table_rows=...) every control step.  SIMULATION ONLY.

Nothing in mpc_fl.py / fast_barriers.py / sim_common.py is edited: the rows use the table_rows slot FLMPC already has (per-arm
`a @ tau <= b`, applied to the FIRST-BLOCK torque columns, substituted tau = M a0 + c by FL-MPC).  Same control rate as the caller (one
tau() per mj_step, FLMPC DT = physics dt for the house demos), no extra MuJoCo rollouts, torque/acceleration scaling untouched.

Optional, all default OFF (= the original behaviour bit-for-bit):
  relax="uniform"   graceful relaxation: if the final solve of a step falls back (fl.last_fallback != 0) re-solve with b_zono + s on the ZONOTOPE
                    rows only (both arms), the smallest certified s found by relax.relax_search.  Table rows passed by the caller, the tray_level
                    rows and everything inside FL-MPC stay HARD.  Needs the 'rt' FL-MPC (BIMANUAL_FLMPC_IMPL=rt: exact QP, trustworthy
                    fallback flag); with impl 'py' it warns and is disabled.  NOTE qp_stats() (the StepMonitor wraps fl.step) also counts the extra
                    solves, like the refine solves.
  relax="selective" PER-ROW relaxation instead of one uniform scalar (a single hopeless row, e.g. a tracked ball that already touches, no longer
                    loosens every other zonotope row).  (1) screen: a zonotope row that demands > relax_frac of the best effort the torque box allows
                    (b < -frac * TAU_MAX * sum|a|, relax.screen_hopeless) is lifted to that level, all others untouched; if that certifies, done.
                    (2) otherwise the uniform search is run on top, then the certified torque is turned into per-row slack s_i = max(0, a_i tau - b_i)
                    (relax.residual_slack: zero on every row that tau already satisfies, <= the uniform s) and the QP is re-solved with that vector (still
                    certifying; if it does not, the uniform result is kept).  Same exact-QP requirement as "uniform"; torque box and all other rows HARD.
  level_relax=True  LAST-RESORT tier (needs relax + tray_level): if the step is STILL on FL-MPC's a=0 fallback after the zonotope relaxation (typically the
                    hard tilt / grip-band rows themselves are infeasible: tray already outside the tilt cone while the arm is fast), search ONE uniform
                    slack s applied to the zonotope rows AND the tray_level rows together (relax_search again); torque box and FL-MPC's own rows stay hard.
  brake=True        a=0 fallback replaced by a velocity brake: a = -qd/brake_tau (FL form, tau = M a + c) scaled into the torque box by FLMPC's closed-form
                    backstop.  a=0 only holds the CURRENT joint velocity (tau = bias): an arm moving at 8 rad/s keeps going, which is how the tray flipped.
  ref_clip=c [rad]  ANTI-WINDUP of the joint reference: q_ref_eff = q + clip(q_ref - q, +-c) per joint, restored after the step.  While a zonotope row holds an arm
                    away from its reference (e.g. pressed against an obstacle) the tracking error grows without bound; when the rows release, FL-MPC
                    spends the whole acceleration / torque box on it, the joints reach ~8 rad/s and the hard tilt row can no longer brake in time
                    (infeasible -> a=0 keeps that velocity -> the tray flips).  Bounding the error bounds the release transient.  None = off.
  tray_level=True   keep the tray level: HARD right-arm tilt row (the built-in cone only covers the LEFT arm), HARD grip-height band rows for both
                    arms, and the tray rim / centre keypoints become TRACKED keypoints (measured tray pose + velocity).  The band is measured relative
                    to the body both arms are mounted on (rows.common_mount_body -> 'Lift_Top'); its acceleration is NOT modelled.
  track_bodies=[..] TRACKED object keypoints (rows.tracked_keypoints): world point from the body's pose in MjData (the SIMULATION stand-in for a
                    perception tracker), measured world velocity, J/Jd as if rigidly attached to each arm's gripper_base; the object's own rolling
                    acceleration is an unmodelled disturbance.  'ball' tracked => the fixed ball spheres are dropped; 'tray' tracked => the fixed
                    tray rim points are dropped.
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import mujoco

from .. import phase2 as P2
from . import rows as R
from .relax import relax_search, screen_hopeless, residual_slack

_EMPTY = (np.zeros((0, 6)), np.zeros(0))


class ZonoCtl(P2.Ctl2):
    def __init__(self, C, zonos=None, margin=0.02, alpha1=R.ALPHA1, alpha2=R.ALPHA2, d_act=0.25, max_rows=8, use_tray=None,
                 tilt=True, tau_cap=None, names=None, kp_radius=True, tray_spacing=0.05, refine=2, refine_add=4, refine_tol=0.02,
                 relax=None, relax_s0=0.05, relax_smax=50.0, relax_bisect=3, relax_max_solves=12, relax_frac=0.9,
                 tray_level=False, level_delta=0.015, track_bodies=None, level_relax=False, brake=False, brake_tau=0.04, ref_clip=None,
                 proprio=None, proprio_ball="prior"):
        super().__init__(C, tilt=tilt, tau_cap=tau_cap)
        m = C.m
        self.kps = {s: R.arm_keypoints(m, s) for s in ("left", "right")}
        if use_tray is None:
            use_tray = bool(C.info.get("has_tray"))
        self.use_tray = use_tray
        track_bodies = list(track_bodies) if track_bodies else []
        # PROPRIO-ONLY (default = env BIMANUAL_PROPRIO_ONLY, off): no `d.xpos` of the tray / ball as 'tracked' points.  The tray rim is the hand-FK path
        # (fixed offsets in each gripper frame from the weld relpose: gripper_base pose = encoders + known geometry, base twist via v_extra), exactly as
        # arm_rows.ArmZonoRows does.  The ball has NO perception in this pipeline -> proprio_ball='prior' keeps the 3 fixed ball spheres on the tray
        # centre line (a conservative prior, not a measurement), 'drop' removes ball points altogether.
        self.proprio = (os.environ.get("BIMANUAL_PROPRIO_ONLY", "0") == "1") if proprio is None else bool(proprio)
        self.proprio_ball = proprio_ball
        if proprio_ball not in ("prior", "drop"):
            raise ValueError("proprio_ball must be 'prior' or 'drop'")
        if self.proprio:
            track_bodies = [b for b in track_bodies if b not in ("tray", "ball")]
        self.tray_level = bool(tray_level)
        if self.tray_level and not use_tray:
            raise ValueError("tray_level needs the tray welded to both grippers (use_tray)")
        tray_tracked = (self.tray_level or "tray" in track_bodies) and not self.proprio
        ball_tracked = "ball" in track_bodies
        ball_fixed = not ball_tracked and not (self.proprio and self.proprio_ball == "drop")
        if use_tray:       # the tray is welded to BOTH grippers: express its points in each gripper frame so each arm's row sees them
            self.kps = {s: self.kps[s] + R.tray_keypoints(m, s, spacing=tray_spacing, include_rim=not tray_tracked, include_ball=ball_fixed)
                        for s in ("left", "right")}
        specs = (["tray"] if tray_tracked else []) + [b for b in track_bodies if b != "tray"]
        self.tracked = R.tracked_keypoints(m, specs, spacing=tray_spacing) if specs else None
        self.gb = {"left": m.body("left_arm_gripper_base").id, "right": m.body("right_arm_gripper_base").id}
        self.level_delta = float(level_delta)
        self.mount = R.common_mount_body(m) if self.tray_level else None
        self.z_ref = {}
        if relax not in (None, "uniform", "selective"):
            raise ValueError(f"relax must be None, 'uniform' or 'selective', got {relax!r}")
        self.relax_frac = float(relax_frac)
        if relax and getattr(self.fl, "impl", None) != "rt":
            warnings.warn("ZonoCtl relax needs the 'rt' FL-MPC (BIMANUAL_FLMPC_IMPL=rt): the default 'py' QP feasible flag is unreliable -> relax DISABLED")
            relax = None
        self.relax = relax
        self.relax_kw = dict(s0=relax_s0, s_max=relax_smax, n_bisect=relax_bisect, max_solves=relax_max_solves)
        self._s_prev = 0.0
        # last-resort tiers (default OFF = the previous behaviour bit-for-bit), see the module docstring
        self.level_relax = bool(level_relax)
        self.brake, self.brake_tau = bool(brake), float(brake_tau)
        self.ref_clip = None if ref_clip is None else float(ref_clip)
        if self.level_relax and not (self.relax and self.tray_level):
            warnings.warn("ZonoCtl level_relax needs relax and tray_level: DISABLED")
            self.level_relax = False
        self.zonos = R.as_zonoset(zonos) if zonos is not None else None
        if self.zonos is not None and names is not None:
            self.zonos.names = list(names)
        self.margin, self.alpha1, self.alpha2, self.d_act = margin, alpha1, alpha2, d_act
        self.max_rows, self.kp_radius = max_rows, kp_radius
        self.refine, self.refine_add, self.refine_tol = refine, refine_add, refine_tol
        self.n_refined = 0
        self.base_twist = None                       # (vx, vy, wz, ox, oy)
        self.stats = []                              # (rows_left, rows_right, ms_rows, ms_total, h_min, feasible, fallback)

    # ---- inputs
    def set_zonos(self, zonos, names=None):
        self.zonos = R.as_zonoset(zonos)
        if names is not None:
            self.zonos.names = list(names)

    def set_base_motion(self, vxy, wz=0.0, origin_xy=(0.0, 0.0)):
        """planar velocity (m/s) and yaw rate (rad/s) of the kinematic base, and its origin: the keypoints' world velocity then includes
        v + wz e_z x (p - origin) (J qd only sees the arm joints).  Base acceleration is NOT modelled.  None disables."""
        self.base_twist = None if vxy is None else (float(vxy[0]), float(vxy[1]), float(wz), float(origin_xy[0]), float(origin_xy[1]))

    def _vextra(self):
        if self.base_twist is None:
            return None
        vx, vy, wz, ox, oy = self.base_twist
        return lambda p: np.stack([vx - wz * (p[:, 1] - oy), vy + wz * (p[:, 0] - ox), np.zeros(len(p))], 1)

    # ---- one control step
    def zono_rows(self, d, max_rows="default"):
        """(rows_left, rows_right, info) for the current state; mj_forward must have been called."""
        mr = self.max_rows if max_rows == "default" else max_rows
        m = self.m
        M_full = self.sc.full_mass_matrix(m, d)
        out, infos = [], []
        meas = R.tracked_measured(m, d, self.tracked) if (self.tracked is not None and len(self.tracked) and R.FAST) else None     # arm-independent: once per step
        for side, da in (("left", self.ld), ("right", self.rd)):
            Ma = M_full[np.ix_(da, da)]
            c = d.qfrc_bias[da] - d.qfrc_passive[da]              # == FLMPC's c_e / c_i  -> the row is exact in acceleration space
            inf = {}
            out.append(R.zono_barrier_rows(m, d, self.kps[side], da, np.linalg.inv(Ma), c, self.zonos, margin=self.margin,
                                           alpha1=self.alpha1, alpha2=self.alpha2, d_act=self.d_act, kp_radius=self.kp_radius,
                                           v_extra=self._vextra(), max_rows=mr, info=inf, tracked=self.tracked, tracked_gb=self.gb[side], tracked_meas=meas))
            infos.append(inf)
        return out[0], out[1], infos

    def level_rows(self, d):
        """HARD keep-the-tray-level rows: ((a_l, b_l), (a_r, b_r)).  Left: grip-height band only (FL-MPC's cone already covers the left EE);
        right: tilt row + grip-height band.  z_ref per arm is captured on the first call (mj_forward must have been called)."""
        m = self.m
        M_full = self.sc.full_mass_matrix(m, d)
        out = []
        for side, da in (("left", self.ld), ("right", self.rd)):
            Ma = M_full[np.ix_(da, da)]
            Mi = np.linalg.inv(Ma)
            c = d.qfrc_bias[da] - d.qfrc_passive[da]
            if side not in self.z_ref:
                self.z_ref[side] = R.tray_origin_height(m, d, side, self.mount)[1]
            parts = []
            if side == "right":
                parts.append(R.tray_tilt_row(m, d, side, da, Mi, c, P2.TILT_MAX, self.alpha1, self.alpha2))
            parts.append(R.grip_height_rows(m, d, side, da, Mi, c, self.mount, self.z_ref[side], self.level_delta, self.alpha1, self.alpha2))
            out.append((np.vstack([a for a, _ in parts]), np.concatenate([b for _, b in parts])))
        return out[0], out[1]

    def _relax_step(self, solve_at, orig, a_pre, a_post):
        """graceful relaxation of one step.  solve_at(s) -> (te, ti, ok, fallback) re-solves with b_zono + s.  orig = (te, ti, ok, fb) of the
        failed s=0 solve; a_pre / a_post = FL-MPC's (a_ego_prev, a_int_prev) before / after that solve.  The pre-state is restored before every
        extra solve; if nothing certifies the original output AND post-state are put back (FL-MPC's own fallback is kept)."""
        fl = self.fl
        keep = {}
        t0 = time.perf_counter()
        attrs = ("last_fallback", "last_feasible", "last_oi")           # per-solve status fields the FL-MPC leaves behind
        snap = lambda: {k: (v.copy() if hasattr(v, "copy") else v) for k in attrs if hasattr(fl, k) for v in [getattr(fl, k)]}
        st_orig = snap()

        def solve(s):
            fl.a_ego_prev, fl.a_int_prev = a_pre[0].copy(), a_pre[1].copy()
            te_, ti_, ok_, fb_ = solve_at(s)
            keep[s] = (te_, ti_, ok_, fb_, fl.a_ego_prev.copy(), fl.a_int_prev.copy(), snap())
            return fb_ == 0

        res = relax_search(solve, s_prev=self._s_prev, **self.relax_kw)
        out = dict(n_solves=res["n_solves"], ms=(time.perf_counter() - t0) * 1e3)
        if res["certified"]:
            s = res["s"]
            te, ti, ok, fb, ae, ai, st = keep[s]
            fl.a_ego_prev, fl.a_int_prev = ae, ai
            for k, v in st.items():
                setattr(fl, k, v)                                       # the ACCEPTED solve's status, not that of the last (failed) trial
            out.update(s=s, te=te, ti=ti, ok=ok, fb=fb, unrecovered=False)
        else:
            fl.a_ego_prev, fl.a_int_prev = a_post[0].copy(), a_post[1].copy()
            for k, v in st_orig.items():
                setattr(fl, k, v)
            out.update(s=0.0, te=orig[0], ti=orig[1], ok=orig[2], fb=orig[3], unrecovered=True)
        return out

    def _brake(self, d):
        """(tau_left, tau_right) of the velocity brake  a = -qd / brake_tau  (tau = M a + c), scaled into the torque box (closed form, only shrinks a)."""
        M_full = self.sc.full_mass_matrix(self.m, d)
        out, a_new = [], []
        for da in (self.ld, self.rd):
            Ma = M_full[np.ix_(da, da)]
            c = d.qfrc_bias[da] - d.qfrc_passive[da]
            a = self.fl._torque_safe_scale(Ma, c, -np.asarray(d.qvel[da], float) / self.brake_tau, self.sc.TAU_MAX)
            out.append(Ma @ a + c); a_new.append(a)
        self.fl.a_ego_prev, self.fl.a_int_prev = a_new[0].copy(), a_new[1].copy()
        return out[0], out[1]

    _STATUS = ("last_fallback", "last_feasible", "last_oi")

    def _status_snap(self):
        fl = self.fl
        return {k: (v.copy() if hasattr(v, "copy") else v) for k in self._STATUS if hasattr(fl, k) for v in [getattr(fl, k)]}

    def _selective_step(self, solve_at, rl, rr, nh_l, nh_r, orig, a_pre, a_post):
        """PER-ROW relaxation of one step (see the module docstring, relax="selective").  solve_at((s_l, s_r)) -> (te, ti, ok, fallback) re-solves with
        b_zono_left += s_l (array, one entry per zonotope row), b_zono_right += s_r.  rl / rr = the unslackened row sets (hard rows first, nh_* of them).
        Returns the same dict as _relax_step plus n_rows_slack, n_screened."""
        fl = self.fl
        t0 = time.perf_counter()
        lim = float(self.sc.TAU_MAX)
        Al, bl, Ar, br = rl[0][nh_l:], rl[1][nh_l:], rr[0][nh_r:], rr[1][nh_r:]
        sc_l = screen_hopeless(Al, bl, lim, self.relax_frac)
        sc_r = screen_hopeless(Ar, br, lim, self.relax_frac)
        n_scr = int((sc_l > 0).sum() + (sc_r > 0).sum())
        st0 = self._status_snap()
        n_solves = 0

        def done(res, sl, sr):
            res.update(n_solves=res.get("n_solves", 0) + n_solves, ms=(time.perf_counter() - t0) * 1e3,
                       n_rows_slack=int((sl > 0).sum() + (sr > 0).sum()), n_screened=n_scr)
            return res

        # (1) screen only
        if n_scr:
            fl.a_ego_prev, fl.a_int_prev = a_pre[0].copy(), a_pre[1].copy()
            te, ti, ok, fb = solve_at((sc_l, sc_r))
            n_solves += 1
            if fb == 0:
                return done(dict(s=float(max(sc_l.max(initial=0.0), sc_r.max(initial=0.0))), te=te, ti=ti, ok=ok, fb=fb, unrecovered=False, n_solves=0), sc_l, sc_r)
            fl.a_ego_prev, fl.a_int_prev = a_post[0].copy(), a_post[1].copy()
            for k, v in st0.items():
                setattr(fl, k, v)
        # (2) uniform on top of the screen, then shrink to the residual slack
        res = self._relax_step(lambda sv: solve_at((sc_l + sv, sc_r + sv)), orig, a_pre, a_post)
        if res["unrecovered"]:
            return done(res, np.zeros_like(sc_l), np.zeros_like(sc_r))
        te_u, ti_u = res["te"], res["ti"]
        keep = (fl.a_ego_prev.copy(), fl.a_int_prev.copy(), self._status_snap())
        v_l, v_r = residual_slack(Al, bl, te_u), residual_slack(Ar, br, ti_u)
        fl.a_ego_prev, fl.a_int_prev = a_pre[0].copy(), a_pre[1].copy()
        te, ti, ok, fb = solve_at((v_l, v_r))
        n_solves += 1
        if fb == 0:
            res.update(s=float(max(v_l.max(initial=0.0), v_r.max(initial=0.0))), te=te, ti=ti, ok=ok, fb=fb)
            return done(res, v_l, v_r)
        fl.a_ego_prev, fl.a_int_prev = keep[0], keep[1]                       # shrink did not certify: keep the uniform (still certified) result
        for k, v in keep[2].items():
            setattr(fl, k, v)
        return done(res, np.full_like(sc_l, res["s"]), np.full_like(sc_r, res["s"]))

    def tau(self, d, table_rows=None):
        if self.ref_clip is None:
            return self._tau(d, table_rows)
        ref = (self.qe_ref.copy(), self.qi_ref.copy())
        try:
            for r_, qadr in ((self.qe_ref, self.lq), (self.qi_ref, self.rq)):
                q = np.asarray(d.qpos[qadr], float)
                r_[:] = q + np.clip(r_ - q, -self.ref_clip, self.ref_clip)
            return self._tau(d, table_rows)
        finally:
            self.qe_ref[:], self.qi_ref[:] = ref

    def _tau(self, d, table_rows=None):
        """as Ctl2.tau, plus zonotope rows; `table_rows` = optional existing (a_e, b_e, a_i, b_i) (e.g. table-plane rows), stacked in."""
        t0 = time.perf_counter()
        mujoco.mj_forward(self.m, d)
        pairs = self.sc.all_pair_barriers(self.m, d, self.ld, self.rd)
        rl = rr = (np.zeros((0, 6)), np.zeros(0))
        infos = [{}, {}]
        tz0 = time.perf_counter()
        if self.zonos is not None and len(self.zonos):
            rl, rr, infos = self.zono_rows(d)
            self._ids_l, self._ids_r = list(infos[0].get("ids", [])), list(infos[1].get("ids", []))
        t_rows = (time.perf_counter() - tz0) * 1e3
        hard_l = hard_r = None                                  # rows that are never relaxed: caller table rows, then the tray_level rows
        nt_l = nt_r = 0                                         # how many of those are the caller's table rows (level_relax slackens only what follows)
        if table_rows is not None:
            ae, be, ai, bi = table_rows
            hard_l, hard_r = (ae, be), (ai, bi)
            nt_l, nt_r = len(be), len(bi)
        if self.tray_level:
            lv_l, lv_r = self.level_rows(d)
            hard_l = R.merge_rows(*[x for x in (hard_l, lv_l) if x is not None])
            hard_r = R.merge_rows(*[x for x in (hard_r, lv_r) if x is not None])
        nh_l = len(hard_l[1]) if hard_l is not None else 0
        nh_r = len(hard_r[1]) if hard_r is not None else 0
        if hard_l is not None or hard_r is not None:
            rl = R.merge_rows(*[x for x in (hard_l, rl) if x is not None]) or _EMPTY
            rr = R.merge_rows(*[x for x in (hard_r, rr) if x is not None]) or _EMPTY
        rows = (rl[0], rl[1], rr[0], rr[1])
        t1 = time.perf_counter()
        kw = dict(left_ee_body=self.lee, tilt_max_rad=P2.TILT_MAX, cup_cone_facets=P2.CONE_FACETS) if self.tilt else {}
        a_prev = (self.fl.a_ego_prev.copy(), self.fl.a_int_prev.copy())
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z, pairs,
                                  table_rows=rows, **kw)
        # ---- active-set refinement: FLMPC's QP is a dual-coordinate-ascent solver run on the capped row set; rows that were NOT passed can be
        # violated by the solution.  Check ALL candidate rows against the returned torque, add the worst violators, re-solve (<= `refine` times).
        if self.refine and self.zonos is not None and len(self.zonos) and self.max_rows is not None:
            full = self.zono_rows(d, max_rows=None)
            for _ in range(self.refine):
                added = False
                new = []
                for side, (ra, rb), tau, inf in (("l", rl, te, full[2][0]), ("r", rr, ti, full[2][1])):
                    fa, fb = (full[0] if side == "l" else full[1])
                    if not len(fb):
                        new.append((ra, rb)); continue
                    viol = fa @ tau - fb
                    have = set(map(tuple, (infl := (self._ids_l if side == "l" else self._ids_r))))
                    cand = [i for i in np.argsort(-viol) if viol[i] > self.refine_tol and tuple(inf["ids"][i]) not in have][: self.refine_add]
                    if cand:
                        added = True
                        ra = np.vstack([ra, fa[cand]]); rb = np.concatenate([rb, fb[cand]])
                        infl.extend(inf["ids"][i] for i in cand)
                    new.append((ra, rb))
                if not added:
                    break
                rl, rr = new
                rows = (rl[0], rl[1], rr[0], rr[1])
                self.fl.a_ego_prev, self.fl.a_int_prev = a_prev[0].copy(), a_prev[1].copy()
                te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z, pairs,
                                          table_rows=rows, **kw)
                self.n_refined += 1
        # ---- graceful relaxation (optional): minimal uniform slack on the ZONOTOPE rows only
        fb0 = int(getattr(self.fl, "last_fallback", 0))
        s_used, n_extra, ms_extra, unrec = 0.0, 0, 0.0, 0
        n_slack = n_scr = 0
        if self.relax and fb0 != 0 and (len(rl[1]) - nh_l) + (len(rr[1]) - nh_r) > 0:
            a_post = (self.fl.a_ego_prev.copy(), self.fl.a_int_prev.copy())
            rl_, rr_ = rl, rr

            def solve_at(sv):
                sl, sr = sv if isinstance(sv, tuple) else (sv, sv)
                bl, br = rl_[1].copy(), rr_[1].copy()
                bl[nh_l:] += sl; br[nh_r:] += sr
                te_, ti_, ok_ = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z, pairs,
                                             table_rows=(rl_[0], bl, rr_[0], br), **kw)
                return te_, ti_, bool(ok_), int(self.fl.last_fallback)

            if self.relax == "selective":
                res = self._selective_step(solve_at, rl_, rr_, nh_l, nh_r, (te, ti, ok, fb0), a_prev, a_post)
                n_slack, n_scr = res["n_rows_slack"], res["n_screened"]
            else:
                res = self._relax_step(solve_at, (te, ti, ok, fb0), a_prev, a_post)
            te, ti, ok = res["te"], res["ti"], res["ok"]
            s_used, n_extra, ms_extra, unrec = res["s"], res["n_solves"], res["ms"], int(res["unrecovered"])
            self._s_prev = s_used
        elif self.relax:
            self._s_prev = 0.0
        s_lv, braked = 0.0, 0
        if self.level_relax and int(getattr(self.fl, "last_fallback", 0)) == 2 and nh_l + nh_r > nt_l + nt_r:
            a_post3 = (self.fl.a_ego_prev.copy(), self.fl.a_int_prev.copy())
            rl3, rr3 = rl, rr

            def solve_lv(sv):
                bl, br = rl3[1].copy(), rr3[1].copy()
                bl[nt_l:] += sv; br[nt_r:] += sv                     # level rows AND zonotope rows (the caller's table rows stay hard)
                te_, ti_, ok_ = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z, pairs,
                                             table_rows=(rl3[0], bl, rr3[0], br), **kw)
                return te_, ti_, bool(ok_), int(self.fl.last_fallback)

            res3 = self._relax_step(solve_lv, (te, ti, ok, 2), a_prev, a_post3)
            n_extra += res3["n_solves"]; ms_extra += res3["ms"]
            if not res3["unrecovered"]:
                te, ti, ok = res3["te"], res3["ti"], res3["ok"]
                s_lv = res3["s"]
        if self.brake and int(getattr(self.fl, "last_fallback", 0)) == 2:
            te, ti = self._brake(d)
            braked = 1
        resid = 0.0
        if self.relax == "selective":                           # TRUE row residual of the returned torque against the UNSLACKENED zonotope rows [N m]
            resid = max(0.0, float(np.max(rl[0][nh_l:] @ te - rl[1][nh_l:], initial=-np.inf)), float(np.max(rr[0][nh_r:] @ ti - rr[1][nh_r:], initial=-np.inf)))
        t2 = time.perf_counter()
        if not ok:
            self.n_infeasible += 1
        self.last_ok = ok
        hm = min([i.get("h_min", np.inf) for i in infos] + [np.inf])
        self.stats.append((len(rl[1]), len(rr[1]), t_rows, (t2 - t0) * 1e3, hm, bool(ok), int(getattr(self.fl, "last_fallback", 0)), (t2 - t1) * 1e3,
                           s_used, n_extra, ms_extra, unrec, fb0, n_slack, n_scr, resid, s_lv, braked))
        return te, ti

    def summary(self, skip=1):
        s = np.array(self.stats[skip:], float)
        if not len(s):
            return {}
        return dict(steps=int(len(s)), rows_left_mean=float(s[:, 0].mean()), rows_right_mean=float(s[:, 1].mean()),
                    rows_per_step_mean=float((s[:, 0] + s[:, 1]).mean()), rows_per_step_max=int((s[:, 0] + s[:, 1]).max()),
                    steps_with_rows=int(((s[:, 0] + s[:, 1]) > 0).sum()),
                    ms_rows_build_median=float(np.median(s[:, 2])), ms_rows_build_p95=float(np.percentile(s[:, 2], 95)),
                    ms_flmpc_step_median=float(np.median(s[:, 7])), ms_flmpc_step_p95=float(np.percentile(s[:, 7], 95)),
                    ms_step_total_median=float(np.median(s[:, 3])), ms_step_total_p95=float(np.percentile(s[:, 3], 95)),
                    ms_step_total_max=float(s[:, 3].max()), n_infeasible=int((s[:, 5] == 0).sum()),
                    n_relaxed_fallback=int((s[:, 6] == 1).sum()), n_zero_fallback=int((s[:, 6] == 2).sum()), refine_resolves=int(self.n_refined),
                    h_min_row=float(s[:, 4].min()), qp=self.qp_stats(), **self._relax_summary(s))

    def _relax_summary(self, s):
        """relax / level bookkeeping.  n_zero_fallback (above) is the FINAL count (after relaxation); ms_relax_p95 is over the steps where relaxation
        was attempted; relax_s_* over the steps where it certified."""
        if not self.relax:
            return {}
        sel = {}
        if self.relax == "selective":
            rel = s[:, 8] > 0
            sel = dict(n_rows_slack_mean_on_relaxed=float(s[rel, 13].mean()) if rel.any() else 0.0, n_rows_slack_max=int(s[:, 13].max()),
                       n_steps_screened=int((s[:, 14] > 0).sum()), row_resid_max=float(s[:, 15].max()), row_resid_p99=float(np.percentile(s[:, 15], 99)),
                       n_steps_resid_gt_1=int((s[:, 15] > 1.0).sum()))
        att, used = s[:, 9] > 0, s[:, 8] > 0
        lv = {}
        if self.level_relax or self.brake:
            lv = dict(n_level_relaxed_steps=int((s[:, 16] > 0).sum()), level_relax_s_max=float(s[:, 16].max()), n_braked_steps=int(s[:, 17].sum()))
        return dict(n_relaxed_steps=int(used.sum()), relax_s_max=float(s[used, 8].max()) if used.any() else 0.0,
                    relax_s_p95=float(np.percentile(s[used, 8], 95)) if used.any() else 0.0, n_relax_unrecovered=int(s[:, 11].sum()),
                    relax_extra_solves_max=int(s[:, 9].max()), n_relax_attempted=int(att.sum()),
                    ms_relax_p95=float(np.percentile(s[att, 10], 95)) if att.any() else 0.0,
                    n_zero_fallback_pre_relax=int((s[:, 12] == 2).sum()), n_fallback_pre_relax=int((s[:, 12] != 0).sum()), **sel, **lv)
