"""nero/contents/rules.py -- opt-in SAFETY RULE rows for the FL-MPC carry controllers ("rules='full'"), SIMULATION ONLY (11-03).

What: (a) arm-vs-table-top HOCBF rows (FL-MPC `table_rows`), (b) joint position + speed limits (FL-MPC `q_lim` / `qd_lim`).
The torque box (16 Nm) and the arm-arm clearance rows are FL-MPC's own and are on in every mode.

Bounds (labelled per AGENTS.md):
  support surfaces   MODEL    static box geoms "pan_table_top" (top 0.76), "counter" (top 0.82), "shelf" (box scenes; top 0.97): top / bottom z and the xy
                              footprint come from geom_pos / geom_size (upright, axis-aligned; asserted).  A row for (point, surface) exists only while the
                              point's xy is inside the footprint + the point's radius AND its centre z >= the slab bottom (a point below a slab, e.g. under
                              the shelf, is not constrained by that slab: its underside is not modelled).  The footprint switch is evaluated on the CURRENT
                              xy, not predicted over the horizon (a point entering the footprint mid-horizon gets its row at the next 5 ms step).
  TABLE_MARGIN       ASSUMED  0.0 m (0.01 m would exclude both gripper_base spheres at the hold pose): proxy-sphere SURFACE (centre z - radius) must stay >= table top + margin.
  ALPHA1, ALPHA2     ASSUMED  30, 30 (= experiments/ab_smooth_vs_bangbang/sim_common.ALPHA1/2, the HOCBF class-K gains used for the arm-arm rows).
  pruning (SOUND)    MODEL    see the comment above `Rules.bounds`: a row is left out only if, with |a_j| <= A_j, the constraint cannot be reached inside
                              the horizon T = NP*DT.  A_j = min(fl.a_box, sum_k |Minv_jk| (max(tau_max_k, |c_k|) + |c_k|))  (torque box -> acceleration box).
  q limits           MODEL    MJCF jnt_range of the arm joints (= config.URDF_LIMITS); a joint without a finite range gets no row.  Effective bound per call:
                              lo_eff = min(lo, q_now), hi_eff = max(hi, q_now) (a joint already past its range may not go further; the row stays feasible).
                              Speed: qd_eff = max(qd_lim, |qd_now|).
  qd limits          SPEC     vendor datasheet config.MANUAL['joint_speed_deg_s'] = 180,180,180,225,225,225,225 deg/s (joints 1-7), no safety factor.
Proxy points = world_model/v2/nero/proxy.py sphere set (the same one FL-MPC's arm-arm rows use, per gripper variant); link0 (fixed base) excluded.
The welded object is not an arm point, so it is not in the table rows.  A (point, surface) pair in scope with clearance h < 0 gets no row that step (the HOCBF row would be infeasible): `overhang` if the sphere does not touch the slab
(3-D distance >= r), `pen` if it does (a real penetration, counted, not constrained).  The t = 0 lists are carried in `info()`.
"""
from __future__ import annotations

import mujoco
import numpy as np

from .. import adapters as A
from .. import config as C
from .. import proxy as P
from ...flmpc import _sc

TABLE_MARGIN = 0.0      # ASSUMED m (no extra margin: the proxy radii already bound the links; 0.01 would exclude both gripper_base spheres, which start 6-9 mm above the table)
ALPHA1, ALPHA2 = 30.0, 30.0   # ASSUMED
SURFACES = ("pan_table_top", "counter", "shelf")      # static box geoms (MODEL)
EPS_JD = 1e-4           # s, numerical Jdot qdot step
SIDES = ("left", "right")
_SCR = {}


def _scratch(m):
    if id(m) not in _SCR:
        _SCR[id(m)] = mujoco.MjData(m)
    return _SCR[id(m)]


def joint_limits(m):
    """(lo, hi) each (2, 7): MJCF jnt_range of the arm joints (MODEL); +-inf where the joint is unlimited."""
    lo, hi = np.full((2, 7), -np.inf), np.full((2, 7), np.inf)
    for k, s in enumerate(SIDES):
        _, dofs = A.arm_addrs(m, s)
        for j, dof in enumerate(dofs):
            jid = int(m.dof_jntid[dof])
            if m.jnt_limited[jid]:
                lo[k, j], hi[k, j] = m.jnt_range[jid]
    return lo, hi


def speed_limits():
    """(2, 7) rad/s from the datasheet (SPEC)."""
    v = np.radians(np.asarray(C.MANUAL["joint_speed_deg_s"], float))
    return np.stack([v, v])


class Rules:
    def __init__(self, m, table=True, limits=True, margin=TABLE_MARGIN, prune="sound"):
        if prune not in ("sound", "none"):
            raise ValueError("prune must be 'sound' or 'none'")
        self.m, self.table, self.limits, self.margin, self.prune = m, bool(table), bool(limits), float(margin), prune
        self.surf = []                                   # MODEL: static box geoms; each dict(name, top, bottom, x=(lo,hi), y=(lo,hi))
        for nm in SURFACES:
            gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, nm)
            if gid < 0:
                continue
            assert int(m.geom_type[gid]) == int(mujoco.mjtGeom.mjGEOM_BOX) and int(m.geom_bodyid[gid]) == 0, nm
            assert np.allclose(m.geom_quat[gid], [1, 0, 0, 0]), f"{nm}: rotated box not supported"
            p, z = m.geom_pos[gid], m.geom_size[gid]
            self.surf.append(dict(name=nm, top=float(p[2] + z[2]), bottom=float(p[2] - z[2]),
                                  x=(float(p[0] - z[0]), float(p[0] + z[0])), y=(float(p[1] - z[1]), float(p[1] + z[1]))))
        self.z_table = self.surf[0]["top"] if self.surf else None                # MODEL (kept for callers / info)
        spec = P.load()
        self.kp = {s: [k for k in P.keypoints(m, f"{s}_arm_", spec) if not m.body(k[0]).name.endswith("link0")] for s in SIDES}
        self.dofs = {s: A.arm_addrs(m, s)[1] for s in SIDES}
        self.qadr = {s: A.arm_addrs(m, s)[0] for s in SIDES}
        self.q_lim = joint_limits(m)
        self.qd_lim = speed_limits()
        self.L = {s: self._chain_len(s) for s in SIDES}   # (n_points, 7): |r_ij| upper bound, see bounds()
        self.keep = None                        # set True at the first call (calibrate: report lists only)
        self.excluded = []
        self.stats = dict(calls=0, tab_pen=0, tab_overhang=0, tab_cand=0, tab_pruned=0, tab_out_scope=0, q_cand=0, q_pruned=0, qd_cand=0, qd_pruned=0, n_q_past=0,
                          q_past_joint=np.zeros((2, 7), int), a_bound_max=np.zeros((2, 7)))

    # ---- soundness helpers ----
    def _chain_len(self, side):
        """L[i, j] >= |r_ij| for ALL configurations: r_ij = vector from joint j's axis anchor to proxy point i.  Walk anchor_j -> origin(body_j) -> ... -> origin(body_i)
        -> point; every segment is a constant-length rigid offset (|jnt_pos|, |body_pos| of each child, |point offset|), so the triangle inequality gives a
        configuration-independent bound.  0 if joint j is not an ancestor of the point's body."""
        m = self.m
        jb = [int(m.dof_bodyid[d_]) for d_ in self.dofs[side]]
        jp = [float(np.linalg.norm(m.jnt_pos[int(m.dof_jntid[d_])])) for d_ in self.dofs[side]]
        L = np.zeros((len(self.kp[side]), 7))
        for i, (bid, off, _r) in enumerate(self.kp[side]):
            for j in range(7):
                tot, b, ok = float(np.linalg.norm(off)), int(bid), False
                while b > 0:
                    if b == jb[j]:
                        ok = True
                        break
                    tot += float(np.linalg.norm(m.body_pos[b]))
                    b = int(m.body_parentid[b])
                L[i, j] = tot + jp[j] if ok else 0.0
        return L

    def bounds(self, fl, side, M_a, c_a):
        """A_j: sound bound on |commanded acceleration| of joint j this step (MODEL).
        The QP's torque box is |tau| <= tl_k with tl_k = max(tau_max_k, |c_k|) (FL-MPC F6 guard) and tau = M a + c, so a = Minv (tau - c) and
        |a_j| <= sum_k |Minv_jk| (tl_k + |c_k|); the acceleration box |a| <= fl.a_box is also hard.  The relaxed fallback slackens only collision/limit/table
        rows (torque box hard) and the zero fallback is a = 0, so the bound holds on every path.
        Soundness of the omissions (horizon T = NP*DT, constant-M,c over the horizon as in the QP, a(s) bounded by A_j for all s in [0, T]):
          position: |q_j(t) - q_j(0)| <= |qd_j| t + 0.5 A_j t^2 <= |qd_j| T + 0.5 A_j T^2 =: D_j   -> a limit row is omitted only if its distance > D_j;
          speed:    |qd_j(t)| <= |qd_j| + A_j t                                                   -> a speed row is omitted only if |qd_j| + A_j T < limit;
          table:    z-displacement of point i <= sum_j L_ij (|qd_j| T + 0.5 A_j T^2)  (|Jz_ij| <= L_ij, see _chain_len) -> row omitted only if clearance > that.
        What this does NOT cover: the HOCBF row is stricter than the plain constraint 'clearance >= 0 for t <= T', so an omitted row can differ from the unpruned QP
        solution when that stricter row would have been active; the test compares both."""
        Minv = np.linalg.inv(M_a)
        tm = np.broadcast_to(np.asarray(fl.tau_max_vec if fl.tau_max_vec is not None else _sc.TAU_MAX, float), (7,))
        tl = np.maximum(tm, np.abs(c_a))
        return np.minimum(float(fl.a_box), np.abs(Minv) @ (tl + np.abs(c_a)))

    # ---- geometry ----
    def _pts(self, d, s):
        b = np.array([k[0] for k in self.kp[s]])
        o = np.array([k[1] for k in self.kp[s]])
        R = d.xmat[b].reshape(-1, 3, 3)
        return d.xpos[b] + np.einsum("kij,kj->ki", R, o)

    def _scope(self, P0, s):
        """(n_points, n_surf) bool: point xy inside footprint + radius and centre z >= slab bottom."""
        r = np.array([k[2] for k in self.kp[s]])
        out = np.zeros((len(r), len(self.surf)), bool)
        for q, S in enumerate(self.surf):
            out[:, q] = ((P0[:, 0] >= S["x"][0] - r) & (P0[:, 0] <= S["x"][1] + r) & (P0[:, 1] >= S["y"][0] - r) & (P0[:, 1] <= S["y"][1] + r)
                         & (P0[:, 2] >= S["bottom"]))
        return out

    def clearance(self, d):
        """{side: (n_points, n_surf) h = z_k - r_k - top - margin} for all candidate (point, surface) pairs (no scope, pruning or exclusion)."""
        out = {}
        for s in SIDES:
            P0 = self._pts(d, s)
            r = np.array([k[2] for k in self.kp[s]])
            out[s] = np.stack([P0[:, 2] - r - S["top"] - self.margin for S in self.surf], 1)
        return out

    def _box_dist(self, p, S):
        """exact distance from a point to the (axis-aligned) slab box of surface S (MODEL)."""
        c = np.array([min(max(p[0], S["x"][0]), S["x"][1]), min(max(p[1], S["y"][0]), S["y"][1]), min(max(p[2], S["bottom"]), S["top"])])
        return float(np.linalg.norm(p - c))

    def calibrate(self, d):
        """first call: lists for the report.  excluded = (point, surface) in scope whose sphere really intersects the slab (3-D distance < r) at t = 0;
        overhang = in scope with h < 0 (the sphere's lowest point is below the top plane) but NOT touching the slab (it hangs over the edge): no row,
        no penetration.  Rows are decided every step (h >= 0 needed), nothing is excluded forever."""
        h = self.clearance(d)
        self.excluded, self.overhang_t0, self.out_of_scope_t0 = [], [], {}
        for s in SIDES:
            P0 = self._pts(d, s)
            sc = self._scope(P0, s)
            self.out_of_scope_t0[s] = [self.m.body(self.kp[s][i][0]).name for i in range(len(self.kp[s])) if not sc[i, 0]]
            for i, q in zip(*np.where(sc & (h[s] < 0.0))):
                r = self.kp[s][i][2]
                rec = dict(side=s, body=self.m.body(self.kp[s][i][0]).name, radius=r, surface=self.surf[q]["name"], h0_m=float(h[s][i, q]),
                           dist_to_slab_m=self._box_dist(P0[i], self.surf[q]))
                (self.excluded if rec["dist_to_slab_m"] < r else self.overhang_t0).append(rec)
        self.keep = True

    def info(self):
        return dict(surfaces=self.surf, margin=self.margin, alpha=(ALPHA1, ALPHA2), prune=self.prune, excluded=self.excluded, overhang_t0=getattr(self, "overhang_t0", None),
                    out_of_table_footprint_t0=getattr(self, "out_of_scope_t0", None),
                    n_points={s: len(self.kp[s]) for s in SIDES},
                    q_lim_finite=int(np.isfinite(self.q_lim[0]).sum()), qd_lim_deg_s=C.MANUAL["joint_speed_deg_s"],
                    stats={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in self.stats.items()})

    # ---- rows ----
    def table_rows(self, d, M_e, c_e, M_i, c_i, fl=None, A_b=None):
        """(a_tab_e, b_tab_e, a_tab_i, b_tab_i) in FL-MPC's torque convention (a_tab @ tau <= b_tab) for arm dynamics M_a, c_a (WITH the payload
        extras that FL-MPC adds; the caller passes exactly those).  HOCBF on the z-height of each (proxy sphere, surface) pair in scope (relative degree 2)."""
        m = self.m
        if self.keep is None:
            self.calibrate(d)
        sd = _scratch(m)
        sd.qpos[:] = d.qpos
        mujoco.mj_integratePos(m, sd.qpos, d.qvel, EPS_JD)
        mujoco.mj_kinematics(m, sd)
        mujoco.mj_comPos(m, sd)
        jp = np.zeros((3, m.nv))
        jp2 = np.zeros((3, m.nv))
        T = None if fl is None else fl.NP * fl.DT
        out = []
        for k, (s, M_a, c_a) in enumerate((("left", M_e, c_e), ("right", M_i, c_i))):
            dofs = self.dofs[s]
            Minv = np.linalg.inv(M_a)
            qd = d.qvel[dofs]
            P0 = self._pts(d, s)
            sc = self._scope(P0, s)
            if self.prune == "sound" and A_b is not None:
                dz = self.L[s] @ (np.abs(qd) * T + 0.5 * A_b[k] * T * T)          # (n_points,) upper bound on the z-displacement within the horizon
            rows_a, rows_b = [], []
            for i, (bid, off, r) in enumerate(self.kp[s]):
                jac_done = False
                for q, S in enumerate(self.surf):
                    if not sc[i, q]:
                        self.stats["tab_out_scope"] += 1
                        continue
                    h = float(P0[i, 2] - r - S["top"] - self.margin)
                    if h < 0.0:          # no valid HOCBF row (would demand lifting the sphere out of the plane): overhang over the edge, or a real penetration
                        self.stats["tab_pen" if self._box_dist(P0[i], S) < r else "tab_overhang"] += 1
                        continue
                    self.stats["tab_cand"] += 1
                    if self.prune == "sound" and A_b is not None and h > dz[i]:
                        self.stats["tab_pruned"] += 1
                        continue
                    if not jac_done:
                        mujoco.mj_jac(m, d, jp, None, P0[i], bid)
                        p2 = sd.xpos[bid] + sd.xmat[bid].reshape(3, 3) @ off
                        mujoco.mj_jac(m, sd, jp2, None, p2, bid)
                        Jz = jp[2, dofs].copy()
                        Jdqd = float((jp2[2, dofs] - Jz) @ qd / EPS_JD)
                        hdot = float(Jz @ qd)
                        A_h = Jz @ Minv
                        jac_done = True
                    psi1 = hdot + ALPHA1 * h
                    rows_a.append(-A_h)
                    rows_b.append(-float(A_h @ c_a) + Jdqd + ALPHA1 * hdot + ALPHA2 * psi1)
            out += [np.array(rows_a).reshape(-1, 7), np.array(rows_b)]
        return tuple(out)

    def limit_bounds(self, d, A_b=None, T=None):
        """effective (lo, hi, qd_lim) each (2, 7) with +-inf / inf where a row is unnecessary (no finite limit, or pruned soundly).  Also counts q past range."""
        lo, hi = self.q_lim[0].copy(), self.q_lim[1].copy()
        vm = self.qd_lim.copy()
        past = False
        for k, s in enumerate(SIDES):
            q = d.qpos[self.qadr[s]]
            qd = d.qvel[self.dofs[s]]
            pl, ph = np.isfinite(lo[k]) & (q < lo[k]), np.isfinite(hi[k]) & (q > hi[k])
            past |= bool(np.any(pl | ph))
            self.stats["q_past_joint"][k] += (pl | ph)
            lo[k] = np.where(pl, q, lo[k])
            hi[k] = np.where(ph, q, hi[k])
            vm[k] = np.maximum(vm[k], np.abs(qd))
            fin_lo, fin_hi = np.isfinite(lo[k]), np.isfinite(hi[k])
            self.stats["q_cand"] += int(fin_lo.sum() + fin_hi.sum())
            self.stats["qd_cand"] += 7
            if self.prune == "sound" and A_b is not None:
                D = np.abs(qd) * T + 0.5 * A_b[k] * T * T
                dl = np.where(fin_lo, np.abs(q - lo[k]), 0.0)
                dh = np.where(fin_hi, np.abs(hi[k] - q), 0.0)
                pr_lo, pr_hi = fin_lo & (dl > D), fin_hi & (dh > D)
                pr_v = (np.abs(qd) + A_b[k] * T) < vm[k]
                self.stats["q_pruned"] += int(pr_lo.sum() + pr_hi.sum())
                self.stats["qd_pruned"] += int(pr_v.sum())
                lo[k] = np.where(pr_lo, -np.inf, lo[k])
                hi[k] = np.where(pr_hi, np.inf, hi[k])
                vm[k] = np.where(pr_v, np.inf, vm[k])
        self.stats["n_q_past"] += int(past)
        return lo, hi, vm

    def fl_kwargs(self, d, M_e, c_e, M_i, c_i, fl=None):
        """kwargs for FLMPC.step.  fl (the controller) supplies NP, DT, a_box, tau_max_vec for the sound pruning; fl=None -> no pruning."""
        self.stats["calls"] += 1
        A_b, T = None, None
        if fl is not None and self.prune == "sound":
            A_b = np.stack([self.bounds(fl, "left", M_e, c_e), self.bounds(fl, "right", M_i, c_i)])
            T = fl.NP * fl.DT
            self.stats["a_bound_max"] = np.maximum(self.stats["a_bound_max"], A_b)
        kw = {}
        if self.table:
            kw["table_rows"] = self.table_rows(d, M_e, c_e, M_i, c_i, fl=fl, A_b=A_b)
        if self.limits:
            lo, hi, vm = self.limit_bounds(d, A_b, T)
            kw["q_lim"] = (lo, hi)
            kw["qd_lim"] = vm
        return kw


def arm_dyn(m, d, dofs, M_extra=None, bias_extra=None):
    """(M_a, c_a) exactly as FL-MPC builds them: arm mass block + extra, qfrc_bias - qfrc_passive + extra."""
    M = np.zeros((m.nv, m.nv))
    mujoco.mj_fullM(m, M, d.qM)
    Ma = M[np.ix_(dofs, dofs)].copy()
    ca = (d.qfrc_bias[dofs] - d.qfrc_passive[dofs]).copy()
    if M_extra is not None:
        Ma = Ma + M_extra
    if bias_extra is not None:
        ca = ca + bias_extra
    return Ma, ca
