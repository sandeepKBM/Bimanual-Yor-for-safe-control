"""rt_pipe (phase 1b): pipelined / anytime FL-MPC. SIMULATION ONLY -- nothing here opens
any hardware / network connection.

A solver THREAD computes the plan for step k+1 while step k executes; the control loop
never waits on the solver beyond the step's deadline.

Mechanism (reuses the existing F5 delay compensation, delay_steps = 1):
  * the plan requested at step k targets step k+1; its initial state is x_{k+1} forward-
    predicted from the measured x_k with the already-committed acceleration a_k
    (FLMPC._predict_over_delay, semi-implicit Euler -- exact for the FL double integrator);
    the committed a_k is the acceleration the plant REALIZES (the demo's existing F5 hook
    `replace_last_applied` computes it from the filtered/clipped torque and posts it here);
    references are sampled one step ahead (the demo's cf_delay offset = 1);
  * the solve runs in rt_kernels.rt_full (@njit nogil) inside a Python threading.Thread, so it
    truly overlaps the main loop; plans are published by an atomic reference swap into a
    short newest-first list (double buffer: the loop only ever reads complete, immutable plans);
  * DEADLINE: the plan for step k must be complete within `budget` (budget_ms, default 4 ms of
    the 5 ms period) of its post time. The loop waits for it at most until that deadline. The
    simulator runs slower than real time, so availability is judged on a VIRTUAL real-time clock:
    a plan with post->done latency delta counts as available at step s iff delta <= budget +
    (s - target) * DT; a late plan is a MISS even if it has arrived by the time the simulator gets
    there. On a miss the newest available plan is used, SHIFTED by its age (block age//L of its
    acceleration profile);
  * EXECUTION-TIME CHECKS on whatever is applied (rt_kernels.exec_check, at the measured x_k):
    the FL torque uses the CURRENT M, c; barrier re-check (top-K pairs at x_k, exact FL
    prediction of the shifted plan); torque box -> _torque_safe_scale; a failed re-check or a
    plan older than age_max -> the brake a = -INFEAS_BRAKE_KD * M^-1 qd (the demo's existing
    infeasibility brake law, FL form), then the torque-box scale;
  * everything is COUNTED: misses, plan age histogram, brakes (age cap / re-check), scales,
    wait time, loop latency (sense -> command), solver time, injected delays.

SAFETY SEMANTICS (explicit): a fresh plan (age 0) is exactly as safe as the synchronous rt
solve at the predicted state (the prediction is exact under FL). A STALE shifted plan is only
NOMINALLY safe: it was feasible for a state that the plant has since left, and disturbances /
model error (friction, contacts, torque filtering) accumulate with its age. Hence (i) the
barrier re-check at the measured state on every step, and (ii) the hard plan-age cap
(age_max, default 4 steps = 20 ms) after which the conservative brake is forced.
"""
from __future__ import annotations

import collections
import sys
import threading
import time

import numpy as np

import rt_flmpc
import rt_kernels as K
import rt_qp
import sim_common as sc


class Plan:
    __slots__ = ("target", "x", "status", "fallback", "t_post", "t_done", "hmin", "solve_ms", "solve_cpu_ms",
                 "injected_ms", "seq")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _Req:
    __slots__ = ("target", "qpos", "qvel", "refs", "opts", "a_cmd", "a_real", "t_post", "posted", "lean", "args")


class _LeanSnap:
    """Immutable copy of the robot-only MjData fields the planner reads (taken by the main
    thread right after its own lean forward at x_k -- the F5 delay mechanism evaluates M, c,
    Jacobians and pairs at the MEASURED state and only predicts q, qd). The solver thread then
    needs no MuJoCo call and no lean sync: only a short Python prologue + the nogil kernel."""
    __slots__ = ("qpos", "qvel", "xpos", "xmat", "cdof", "cdof_dot", "cvel", "subtree_com", "qfrc_bias",
                 "qfrc_passive", "Mfull")

    def __init__(self, d, Mfull):
        for k in self.__slots__[:-1]:
            setattr(self, k, np.array(getattr(d, k), copy=True))
        self.Mfull = np.array(Mfull, copy=True)


class PipelinedRTFLMPC(rt_flmpc.RTFLMPC):
    """Execution side (this object: the demo's `fl`) + a planner RTFLMPC in a worker thread."""

    def __init__(self, model, *, budget_ms: float = 4.0, age_max: int = 4, inject_p: float = 0.0,
                 inject_ms=(10.0,), inject_seed: int = 0, recheck_tol: float = 0.002, brake_kd: float = 8.0,
                 switch_interval: float = 1e-4, **kw):
        kw = dict(kw)
        if int(kw.get("delay_steps", 0)) != 1:
            raise ValueError("rt_pipe needs delay_steps=1 (the demo sets cf_delay=1 for --controller-impl rt_pipe)")
        super().__init__(model, **kw)
        self.planner = rt_flmpc.RTFLMPC(model, **kw)
        self.budget = float(budget_ms) * 1e-3
        self.age_max = int(age_max)
        self.inject_p = float(inject_p)
        self.inject_ms = tuple(float(v) for v in (inject_ms if np.ndim(inject_ms) else (inject_ms,)))
        self._rng = np.random.default_rng(int(inject_seed))
        self.recheck_tol = float(recheck_tol)
        self.brake_kd = float(brake_kd)
        self._switch_interval = float(switch_interval)
        self._cond = threading.Condition()
        self._plans = collections.deque(maxlen=16)     # newest first
        self._req = None                                # latest posted request (latest wins)
        self._pending = None                            # request built at step k, posted on replace_last_applied
        self._stop = False
        self._thread = None
        self._worker_exc = None
        self._k = 0
        self._ex = np.zeros(K.N_EX)
        self._seq = 0
        # stats
        self.n_pipe_steps = 0
        self.n_miss = 0
        self.n_bootstrap = 0
        self.n_brake_age = 0
        self.n_brake_recheck = 0
        self.n_exec_scaled = 0
        self.n_injected = 0
        self.n_late_post = 0
        self.age_hist = np.zeros(64, np.int64)
        self.pipe_records = []       # (k, target, age, miss, braked, recheck_fail, scaled, latency, wait, hmin)

    # ------------------------------------------------------------------ wiring
    def configure_lean(self, lean, scene_model, *args, **kw):
        super().configure_lean(lean, scene_model, *args, **kw)
        self.planner.configure_lean(lean.clone(), scene_model, *args, **kw)
        # compile / cache-load exec_check now (dry run, forced brake, nothing recorded)
        t0 = time.perf_counter()
        cfg = self._lean_cfg
        ld = lean.data
        lean.set_state(lean.model.qpos0.copy(), np.zeros(lean.model.nv))
        lean.forward()
        e_body, e_off, e_r, e_mask = self._ego_pack(cfg["ego_extra"])
        i_body, i_off, i_r, i_mask = cfg["int_kps"]
        self._ensure_ws(1, 1, len(e_body), len(i_body))
        ws = self._ws
        self._gcfg[K.GI_E] = len(e_body)
        self._gcfg[K.GI_I] = len(i_body)
        self._gcfg[K.GI_ROOT] = cfg["root"]
        self._gcfg[K.GI_LEFT_EE] = cfg["lee"]
        ws.tau_lim_e[:] = sc.TAU_MAX
        ws.tau_lim_i[:] = sc.TAU_MAX
        K.exec_check(self._ci, self._coll_steps, self.DT, self.H_MARGIN, self._gcfg, self._gpar, self._flg,
                     ld.xpos, ld.xmat, ld.cdof, ld.cdof_dot, ld.cvel, ld.subtree_com, ld.qvel, ld.qfrc_bias,
                     ld.qfrc_passive, lean.full_M(), cfg["ldofs"], cfg["rdofs"], cfg["lmask_ee"],
                     e_body, e_off, e_r, e_mask, i_body, i_off, i_r, i_mask,
                     self._z3, self._eye3, self._cup_mI, cfg["gvec"], ws.xout, 0, 0, 1e9,
                     self.recheck_tol, self.brake_kd, ws.tau_lim_e, ws.tau_lim_i, ws.W,
                     ws.a_e0, ws.a_i0, ws.tau_e, ws.tau_i, self._ex)
        self.jit_warmup_s += self.planner.jit_warmup_s + (time.perf_counter() - t0)
        sys.setswitchinterval(self._switch_interval)   # bound GIL hand-over latency (sim loop is Python-heavy)
        self._thread = threading.Thread(target=self._worker, name="rt_pipe_solver", daemon=True)
        self._thread.start()

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    # ------------------------------------------------------------------ worker
    def _prepare_req(self, req, delay_on=True):
        """MAIN THREAD (holds the GIL anyway): the planner's Python prologue for this request --
        F5 forward prediction with the committed (realized) acceleration, payload / keypoint /
        option packing -- into a self-contained argument tuple. The solver thread then only
        executes the compiled nogil kernel (no MuJoCo call, no Python prologue under GIL
        contention)."""
        P = self.planner
        if not delay_on:
            P.delay_steps = 0
        try:
            req.args = P._prepare(req.lean, req.lean.Mfull, *req.refs, a_prev=req.a_cmd,
                                  a_queue=[req.a_real] if delay_on else [], **req.opts)
        finally:
            P.delay_steps = 1

    def _solve(self, req):
        P = self.planner
        t0 = time.perf_counter()
        c0 = time.thread_time()
        P._run(req.args, t0, t0)
        solve = time.perf_counter() - t0
        solve_cpu = time.thread_time() - c0
        ws = P._ws
        m_coll = P.K_PAIRS * len(P.coll_steps)
        x = ws.xout.copy()
        hmin = P.H_MARGIN + float(np.min(ws.b[:m_coll] - ws.A[:m_coll] @ x)) if m_coll else 1e9
        return Plan(target=req.target, x=x, status=int(P.last_oi[K.OI_HARD_STATUS]), fallback=int(P.last_fallback),
                    t_post=req.t_post, t_done=0.0, hmin=hmin, solve_ms=solve * 1e3, solve_cpu_ms=solve_cpu * 1e3,
                    injected_ms=0.0, seq=0)

    def _worker(self):
        while True:
            with self._cond:
                while self._req is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                req = self._req
                self._req = None
            try:
                plan = self._solve(req)
            except BaseException as e:      # never die silently: surfaced by the next step_lean
                self._worker_exc = e
                return
            if self.inject_p > 0.0 and self._rng.random() < self.inject_p:
                ms = float(self.inject_ms[int(self._rng.integers(len(self.inject_ms)))])
                time.sleep(ms * 1e-3)          # artificial slow solve (releases the GIL)
                plan.injected_ms = ms
            plan.t_done = time.perf_counter()
            with self._cond:
                self._seq += 1
                plan.seq = self._seq
                self._plans.appendleft(plan)
                self._cond.notify_all()

    def _post_req(self, req):
        """Post a request to the worker; the deadline of the plan it produces is t_post + budget."""
        req.t_post = time.perf_counter()
        req.posted = True
        self._t_post_prev = req.t_post
        with self._cond:
            self._req = req
            self._cond.notify_all()

    # ------------------------------------------------------------------ F5 hooks (demo) -> post
    def replace_last_applied(self, a_e, a_i):
        """Demo F5 hook (after its torque filter): the acceleration the plant REALIZES this step
        is the committed a_k for the forward prediction of the NEXT plan -> post the request."""
        req = self._pending
        if req is not None and not req.posted:
            req.a_real = (np.asarray(a_e, float).copy(), np.asarray(a_i, float).copy())
            self._prepare_req(req)
            self._post_req(req)

    def push_applied(self, a_e, a_i):
        """Another controller commanded this step (lift phase): keep the committed accel."""
        self._last_pushed = (np.asarray(a_e, float).copy(), np.asarray(a_i, float).copy())

    # ------------------------------------------------------------------ main-loop step
    def step_lean(self, scene_data, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, **opts):
        t_enter = time.perf_counter()
        if self._worker_exc is not None:
            raise RuntimeError("rt_pipe solver thread died") from self._worker_exc
        k = self._k
        self._k += 1
        cfg = self._lean_cfg
        lean = cfg["lean"]
        if self._pending is not None and not self._pending.posted:
            # caller did not call replace_last_applied: post with the commanded accel (late)
            self._pending.a_real = self._pending.a_cmd
            self._prepare_req(self._pending)
            self._post_req(self._pending)
            self.n_late_post += 1
        lean.sync_from_scene(scene_data)
        lean.forward()
        Mfull = lean.full_M()
        ld = lean.data
        qpos_k = ld.qpos.copy()
        qvel_k = ld.qvel.copy()
        refs = tuple(np.ascontiguousarray(r, np.float64).copy() for r in (q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref))
        # scene-dependent option payloads are copied (the worker runs later)
        opts_c = dict(opts)
        for key in ("cup_pose", "cube"):
            if opts_c.get(key) is not None:
                opts_c[key] = tuple(np.array(v, copy=True) if isinstance(v, np.ndarray) else v for v in opts_c[key])
        wait = 0.0
        plan = None
        miss = 0
        if not self._plans:
            # bootstrap: synchronous solve for THIS step (no prediction)
            req = _Req()
            req.target, req.qpos, req.qvel, req.refs, req.opts = k, qpos_k, qvel_k, refs, opts_c
            req.lean = _LeanSnap(ld, Mfull)
            req.a_cmd = (self.planner.a_ego_prev.copy(), self.planner.a_int_prev.copy())
            req.a_real = req.a_cmd
            req.t_post = time.perf_counter()
            req.posted = True
            self._prepare_req(req, delay_on=False)
            plan = self._solve(req)
            plan.t_done = time.perf_counter()
            with self._cond:
                self._plans.appendleft(plan)
            self.n_bootstrap += 1
            deadline = plan.t_done
        else:
            deadline = self._t_post_prev + self.budget
            t_w0 = time.perf_counter()
            with self._cond:
                while True:
                    newest = self._plans[0]
                    if newest.target >= k:
                        break
                    now = time.perf_counter()
                    if now >= deadline:
                        break
                    self._cond.wait(deadline - now)
                cands = list(self._plans)
            wait = time.perf_counter() - t_w0
            # REAL-TIME (virtual-clock) eligibility, independent of how slowly the simulator itself
            # steps: a plan whose post->done latency is delta was, in a real 200 Hz loop, available
            # from the first step s with delta <= budget + (s - target) * DT (i.e. its own deadline,
            # or that many periods later as a stale plan). Newest eligible plan wins.
            for p in cands:                       # newest first
                if p.target <= k and (p.t_done - p.t_post) <= self.budget + (k - p.target) * self.DT + 1e-12:
                    plan = p
                    break
            if plan is None or plan.target != k:
                miss = 1
        age = (k - plan.target) if plan is not None else 10 ** 6
        force = 1 if (plan is None or age > self.age_max) else 0
        # ---- execution-time checks on the measured state ----
        e_body, e_off, e_r, e_mask = self._ego_pack(cfg["ego_extra"] if isinstance(opts.get("ego_extra_kps", "cfg"), str)
                                                    else opts.get("ego_extra_kps"))
        i_body, i_off, i_r, i_mask = cfg["int_kps"]
        self._ensure_ws(1, 1, len(e_body), len(i_body))
        ws = self._ws
        g = self._gcfg
        g[K.GI_ROOT] = cfg["root"]
        g[K.GI_LEFT_EE] = cfg["lee"]
        g[K.GI_E] = len(e_body)
        g[K.GI_I] = len(i_body)
        self._gpar[K.GF_DMIN] = sc.D_MIN
        flg = self._flg
        cup_p, cup_R = self._z3, self._eye3
        flg[K.FL_CUP_ON] = 0
        cp = opts.get("cup_pose")
        if cp is not None:
            cb = cfg["cup_body"]
            self._cup_mI[0] = cfg["scene_model"].body_mass[cb]
            self._cup_mI[1:4] = cfg["scene_model"].body_inertia[cb]
            cup_p = np.ascontiguousarray(cp[0], np.float64)
            cup_R = np.ascontiguousarray(np.asarray(cp[1], np.float64).reshape(3, 3))
            flg[K.FL_CUP_ON] = 1
        c_all = ld.qfrc_bias - ld.qfrc_passive
        if self.tau_max_vec is None:
            ws.tau_lim_e[:] = sc.TAU_MAX
            ws.tau_lim_i[:] = sc.TAU_MAX
        else:
            ws.tau_lim_e[:] = np.maximum(self.tau_max_vec, np.abs(c_all[cfg["ldofs"]]))
            ws.tau_lim_i[:] = np.maximum(self.tau_max_vec, np.abs(c_all[cfg["rdofs"]]))
        px = plan.x if plan is not None else ws.xout
        hmin_plan = plan.hmin if plan is not None else 1e9
        K.exec_check(self._ci, self._coll_steps, self.DT, self.H_MARGIN, g, self._gpar, flg,
                     ld.xpos, ld.xmat, ld.cdof, ld.cdof_dot, ld.cvel, ld.subtree_com, ld.qvel, ld.qfrc_bias,
                     ld.qfrc_passive, Mfull, cfg["ldofs"], cfg["rdofs"], cfg["lmask_ee"],
                     e_body, e_off, e_r, e_mask, i_body, i_off, i_r, i_mask,
                     cup_p, cup_R, self._cup_mI, cfg["gvec"], px, int(min(age, 10 ** 6)), force, hmin_plan,
                     self.recheck_tol, self.brake_kd, ws.tau_lim_e, ws.tau_lim_i, ws.W,
                     ws.a_e0, ws.a_i0, ws.tau_e, ws.tau_i, self._ex)
        ex = self._ex
        braked = int(ex[K.EX_BRAKED])
        rfail = int(ex[K.EX_RECHECK_FAIL])
        scaled = int(ex[K.EX_SCALED])
        # ---- bookkeeping ----
        self.n_pipe_steps += 1
        self.n_miss += miss
        self.age_hist[min(age, 63)] += 1
        if force:
            self.n_brake_age += 1
        if rfail:
            self.n_brake_recheck += 1
        self.n_exec_scaled += scaled
        a_e = ws.a_e0.copy()
        a_i = ws.a_i0.copy()
        self.a_ego_prev = a_e
        self.a_int_prev = a_i
        feasible = plan is not None and plan.status == rt_qp.ST_OPTIMAL and not braked
        self.last_feasible = feasible
        self.last_fallback = (plan.fallback if plan is not None else 2) if not braked else 3
        self.last_M_e, self.last_c_e, self.last_M_i, self.last_c_i = ws.M_e.copy(), ws.c_e.copy(), ws.M_i.copy(), ws.c_i.copy()
        # planner counters mirrored for the demo log (fl_relaxed_steps etc.)
        P = self.planner
        self.n_relaxed, self.n_zero_fallback, self.n_taumax_relaxed = P.n_relaxed, P.n_zero_fallback, P.n_taumax_relaxed
        # ---- request for step k+1 (posted by replace_last_applied with the realized accel) ----
        req = _Req()
        req.target, req.qpos, req.qvel, req.refs, req.opts = k + 1, qpos_k, qvel_k, refs, opts_c
        req.lean = _LeanSnap(ld, Mfull)
        req.a_cmd = (a_e.copy(), a_i.copy())
        req.a_real = req.a_cmd
        req.posted = False
        req.t_post = 0.0
        self._pending = req
        t_ret = time.perf_counter()
        self.pipe_records.append((k, plan.target if plan is not None else -1, age, miss, braked, rfail, scaled,
                                  (t_ret - t_enter) * 1e3, wait * 1e3, float(ex[K.EX_HMIN]),
                                  plan.solve_ms if plan is not None else np.nan,
                                  plan.injected_ms if plan is not None else 0.0,
                                  plan.solve_cpu_ms if plan is not None else np.nan,
                                  int(plan.t_done - plan.t_post <= self.budget + 1e-12) if plan is not None else 0))
        return ws.tau_e.copy(), ws.tau_i.copy(), feasible

    def rt_log_fields(self):
        """Episode-end counters (also stops the solver thread: the demo calls this once at the end)."""
        self.close()
        P = self.planner
        R = np.array(self.pipe_records, dtype=float) if self.pipe_records else np.zeros((0, 14))
        lat = R[:, 7] if len(R) else np.zeros(1)
        sol = R[:, 10][np.isfinite(R[:, 10])] if len(R) else np.zeros(1)
        cpu = R[:, 12][np.isfinite(R[:, 12])] if len(R) else np.zeros(1)
        # CPU-time view of the deadline: would the plan used this step have made its budget on an
        # idle core? (wall-clock misses in the sim also contain GIL/CPU contention with the
        # Python-heavy simulator loop running in the same process)
        cpu_miss = int(np.sum(R[:, 12] > self.budget * 1e3)) if len(R) else 0
        waits = R[:, 8] if len(R) else np.zeros(1)
        return dict(fl_rt_steps=self.n_pipe_steps, fl_rt_pipe_miss_steps=self.n_miss,
                    fl_rt_pipe_bootstrap=self.n_bootstrap, fl_rt_pipe_brake_age=self.n_brake_age,
                    fl_rt_pipe_brake_recheck=self.n_brake_recheck, fl_rt_pipe_exec_scaled=self.n_exec_scaled,
                    fl_rt_pipe_late_post=self.n_late_post,
                    fl_rt_pipe_age_hist=np.trim_zeros(self.age_hist, "b"),
                    fl_rt_pipe_injected=int(np.sum(R[:, 11] > 0)) if len(R) else 0,
                    fl_rt_pipe_latency_ms=np.array([lat.mean(), np.median(lat), np.percentile(lat, 99), lat.max()]),
                    fl_rt_pipe_solve_ms=np.array([sol.mean(), np.median(sol), np.percentile(sol, 99), sol.max()])
                    if len(sol) else np.zeros(4),
                    fl_rt_pipe_solve_cpu_ms=np.array([cpu.mean(), np.median(cpu), np.percentile(cpu, 99), cpu.max()])
                    if len(cpu) else np.zeros(4),
                    fl_rt_pipe_cpu_over_budget_steps=cpu_miss,
                    fl_rt_pipe_wait_ms=np.array([waits.mean(), np.percentile(waits, 99), waits.max()]),
                    fl_rt_pipe_budget_ms=self.budget * 1e3, fl_rt_pipe_age_max=self.age_max,
                    fl_rt_hard_cap_steps=P.n_hard_cap, fl_rt_relax_cap_steps=P.n_relax_cap,
                    fl_rt_hard_infeasible_steps=P.n_hard_infeasible, fl_rt_relax_infeasible_steps=P.n_relax_infeasible,
                    fl_rt_hard_it_hist=np.trim_zeros(P.hard_it_hist, "b"),
                    fl_rt_relax_it_hist=np.trim_zeros(P.relax_it_hist, "b"),
                    fl_rt_it_caps=np.array([P.it_hard, P.it_relax]))
