"""planner.py -- SamplingPlanner: MPPI / CEM / predictive sampling over a smooth base-twist sequence, GPU, warm-started, CUDA-graph captured.

One plan() =
  1. noise  eps ~ N(0, 1) at Hc coarse knots, linearly interpolated to H steps  (smooth perturbations), U_k = clip(mean + sigma * interp(eps_k));
  2. structured candidates replace the tail of the batch:  [mean, brake (0), intent hold, previous best] + random TWO-PHASE constant-velocity primitives
     (heading th1 for a random time then th2): they let the search jump between topologically different routes (left vs right of an obstacle),
     which a Gaussian around one mean never does;
  3. cost(U) on all K (the cost is NON-CONVEX, see cost.py);
  4. update:  mppi  w ~ softmax(-(c - cmin) / (lam * std c)) over the noise samples, mean += sum w (U - mean)
              cem   mean = mean of the best `elite` fraction (all samples)
              ps    mean = best sample (predictive sampling)
  5. candidate refinement: the updated mean, the best primitive and the best sample are re-evaluated and the cheapest becomes the new mean.
All of 1-5 is ONE torch graph with static buffers (CUDA graph after warm-up): no host sync inside; the host copies the state in, draws eps / primitives
into static buffers (torch.randn(out=)) and reads the (H, nu) result back.

warm-up (startup, user requirement):  CUDA context + cuBLAS/grid_sample kernels + graph capture + a few dummy plans, then `ready = True`.
`stats["warmup_ms"]` and `stats["plan_ms"]` are reported separately.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch


def interp_matrix(H, Hc):
    """(H, Hc) linear interpolation from Hc coarse knots to H fine steps."""
    m = np.zeros((H, Hc), np.float32)
    pos = np.linspace(0, Hc - 1, H)
    for i, p in enumerate(pos):
        j = min(int(math.floor(p)), Hc - 2); f = p - j
        m[i, j] = 1 - f; m[i, j + 1] = f
    return m


def shift_matrix(tau, elapsed):
    """(H, H) warm-start shift in TIME: new[i] = old(tau_i + elapsed) by linear interpolation over the step end-times tau (H,), last value held."""
    H = len(tau)
    m = np.zeros((H, H), np.float32)
    for i in range(H):
        q = tau[i] + elapsed
        if q >= tau[-1]:
            m[i, H - 1] = 1.0; continue
        j = int(np.searchsorted(tau, q, side="right") - 1)
        if j < 0:                       # before the first knot: hold the first value
            m[i, 0] = 1.0; continue
        f = (q - tau[j]) / (tau[j + 1] - tau[j])
        m[i, j] += 1 - f; m[i, j + 1] += f
    return m


class SamplingPlanner:
    def __init__(self, problem, K=1024, Hc=6, method="mppi", sigma=(0.10, 0.10, 0.15, 0.02, 0.02), lam=0.4, n_prim=96, elite=0.05, iters=1,
                 prim_speed=(0.9, 0.9), use_graph=True, seed=0, plan_dt=None, compile=False):
        assert method in ("mppi", "cem", "ps")
        self.pb, self.K, self.H, self.Hc, self.method, self.lam, self.elite = problem, K, problem.H, Hc, method, lam, elite
        self.nu, self.dev = problem.nu, problem.dev
        self.n_prim = min(n_prim, K // 2)
        self.ramp_acc = (0.8, 1.2, 1.6, 2.4, 4.0)           # m/s^2 linear acceleration ramps toward {intent, intent/2, 0} (rad/s^2: x2 on yaw)
        self.n_ramp = 3 * len(self.ramp_acc)
        self.n_fix = 4 + self.n_ramp                       # mean, zero, intent, prev-best, ramps
        self.iters = iters
        f32 = dict(dtype=torch.float32, device=self.dev)
        sg = np.zeros(self.nu, np.float32); sg[:min(self.nu, len(sigma))] = sigma[:self.nu]
        self.sigma = torch.as_tensor(sg, **f32)
        self.tau = np.cumsum(problem.dtv.cpu().numpy())
        self.tau_t = torch.as_tensor(self.tau, **f32)
        self.ramp_w = torch.clamp(self.tau_t / 0.3, max=1.0)[None, :, None]          # noise grows from 0 at the current state: samples start CONTINUOUS with the velocity
        self.acc_t = torch.as_tensor([[a, a, 2 * a] for a in self.ramp_acc], **f32)  # (R,3)
        self.Bint = torch.as_tensor(interp_matrix(self.H, Hc), **f32)
        self.vmax = torch.as_tensor(problem.model.v_max, **f32)
        self.prim_speed = prim_speed
        self.gen = torch.Generator(device=self.dev); self.gen.manual_seed(seed)
        nn_ = K - self.n_prim - self.n_fix
        assert nn_ > 8
        self.nn = nn_
        # static buffers
        self.mean = torch.zeros(self.H, self.nu, **f32)
        self.prev_best = torch.zeros(self.H, self.nu, **f32)
        self.eps = torch.zeros(nn_, Hc, self.nu, **f32)
        self.prim = torch.zeros(self.n_prim, self.H, self.nu, **f32)
        self.shift = torch.eye(self.H, **f32)
        self.out = torch.zeros(self.H, self.nu, **f32)
        self.out_v = torch.zeros(self.H, 3, **f32)
        self.out_a = torch.zeros(self.H, 2, **f32)
        self.out_j = torch.zeros((), dtype=torch.long, device=self.dev)
        self.out_cost = torch.zeros((), **f32)
        self.out_hmin = torch.zeros((), **f32)
        self.use_graph, self.graph, self.ready = use_graph, None, False
        self._cost = problem.cost
        if compile:      # inductor-fused rollout + cost (fewer, larger kernels); compiled during warm-up, then captured in the CUDA graph
            self._cost = torch.compile(problem.cost, dynamic=False)
        self.stats = dict(warmup_ms=None, plan_ms=[], graph=False)
        self.plan_dt = plan_dt
        self.choice = [0, 0, 0]          # how often the stage-2 pick was: [updated mean, best structured candidate, best noise sample]
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------- candidates / update (inside the graph)
    def _build(self):
        pb = self.pb
        mean = self.shift @ self.mean                                     # warm start (shifted)
        Un = mean[None] + self.sigma * torch.einsum("hc,kcu->khu", self.Bint, self.eps) * self.ramp_w
        v0 = pb.S["v0"]
        tg = torch.stack([pb.S["u_int"][:3], 0.5 * pb.S["u_int"][:3], torch.zeros_like(v0)])                        # (3,3)
        dv = (tg - v0)[:, None, None, :]                                                                              # (3,1,1,3)
        lim = self.acc_t[None, :, None, :] * self.tau_t[None, None, :, None]                                          # (1,R,H,3)
        ramps = (v0 + torch.minimum(torch.maximum(dv, -lim), lim)).reshape(self.n_ramp, self.H, 3)
        if self.nu > 3:
            ramps = torch.cat([ramps, torch.zeros(self.n_ramp, self.H, self.nu - 3, device=ramps.device)], -1)
        fixed = torch.cat([torch.stack([mean, torch.zeros_like(mean), pb.S["u_int"].expand(self.H, -1), self.shift @ self.prev_best]), ramps], 0)
        U = torch.cat([Un, fixed, self.prim], 0)
        vm = self.vmax
        U = torch.cat([torch.minimum(torch.maximum(U[..., :3], -vm), vm), U[..., 3:]], -1)
        return U, mean

    def _step(self):
        pb = self.pb
        U, mean = self._build()
        c = self._cost(U)
        nn_ = self.nn
        cn = c[:nn_]
        if self.method == "mppi":
            sd = c.std().clamp(min=1e-3)
            w = torch.softmax(-(cn - cn.min()) / (self.lam * sd), 0)
            new = mean + (w[:, None, None] * (U[:nn_] - mean[None])).sum(0)
        elif self.method == "cem":
            ne = max(2, int(self.elite * self.K))
            idx = torch.topk(c, ne, largest=False).indices
            new = U.index_select(0, idx).mean(0)
        else:
            new = U.index_select(0, torch.argmin(c).reshape(1))[0]
        ib = torch.argmin(c)
        ip = nn_ + torch.argmin(c[nn_:])
        cand = torch.cat([new[None], U.index_select(0, ip.reshape(1)), U.index_select(0, ib.reshape(1))], 0)
        c2, parts = pb.cost(cand, return_parts=True)          # batch of 3: uncompiled (tiny)
        j = torch.argmin(c2)
        best = cand.index_select(0, j.reshape(1))[0]
        self.out.copy_(best); self.out_j.copy_(j)
        self.out_v.copy_(parts["rollout"]["v"].index_select(0, j.reshape(1))[0]); self.out_a.copy_(parts["rollout"]["a"].index_select(0, j.reshape(1))[0])
        self.out_cost.copy_(c2.index_select(0, j.reshape(1))[0]); self.out_hmin.copy_(parts["h_min"].index_select(0, j.reshape(1))[0])
        self.mean.copy_(best); self.prev_best.copy_(best)

    # ---------------------------------------------------------------- host side
    def _fill_random(self):
        torch.randn(self.eps.shape, out=self.eps, generator=self.gen)
        n, H = self.n_prim, self.H
        r = torch.rand(n, 7, device=self.dev, generator=self.gen)
        th1, th2 = r[:, 0] * 2 * math.pi, r[:, 1] * 2 * math.pi
        s1, s2 = r[:, 2] * self.prim_speed[0], r[:, 3] * self.prim_speed[1]
        ts = (r[:, 4] * 0.8 + 0.1) * H
        steps = torch.arange(H, device=self.dev)[None, :]
        first = (steps < ts[:, None]).float()
        vx = first * (s1 * torch.cos(th1))[:, None] + (1 - first) * (s2 * torch.cos(th2))[:, None]
        vy = first * (s1 * torch.sin(th1))[:, None] + (1 - first) * (s2 * torch.sin(th2))[:, None]
        self.prim.zero_()
        self.prim[..., 0] = vx * self.vmax[0]; self.prim[..., 1] = vy * self.vmax[1]
        if self.nu >= 3:
            self.prim[..., 2] = (r[:, 5:6] - 0.5) * 0.2 * self.vmax[2] * first

    def plan(self, elapsed=None):
        """one planning call at the state already set in problem.S.  `elapsed` [s] since the previous plan (warm-start shift).  Returns (U (H,nu) np, info)."""
        t0 = time.perf_counter()
        el = elapsed if elapsed is not None else (self.plan_dt or float(self.pb.dtv[0]))
        self.shift.copy_(torch.as_tensor(shift_matrix(self.tau, el)))
        self.pb.resample_mu()
        self.pb.flush()                                                     # ONE host->device copy for the whole state (+ mu draws)
        for _ in range(self.iters):
            self._fill_random()
            if self.graph is not None:
                self.graph.replay()
            else:
                self._step()
            if self.iters > 1:
                self.shift.copy_(torch.eye(self.H, device=self.dev))
        U = self.out.cpu().numpy()                                          # sync
        self.choice[int(self.out_j)] += 1
        self.v_pred, self.a_pred = self.out_v.cpu().numpy(), self.out_a.cpu().numpy()
        ms = (time.perf_counter() - t0) * 1e3
        if self.ready:
            self.stats["plan_ms"].append(ms)
        return U, dict(cost=float(self.out_cost), h_min=float(self.out_hmin), ms=ms)

    def reset(self):
        self.mean.zero_(); self.prev_best.zero_()

    def warmup(self, n=8):
        """CUDA context, kernels, graph capture, dummy plans.  Leaves the planner state reset.  Sets ready."""
        t0 = time.perf_counter()
        torch.zeros(1, device=self.dev)
        S = self.pb.S
        saved = self.pb.flat.clone(); saved_h = self.pb._host.copy()
        for _ in range(3):                                                # eager passes (also the required side-stream warm-up before capture)
            self._fill_random(); self.pb.resample_mu(); self.pb.flush(); self._step()
        if self.use_graph and torch.cuda.is_available():
            torch.cuda.synchronize()
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(2):
                    self._step()
            torch.cuda.current_stream().wait_stream(s)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._step()
            self.graph = g
            self.stats["graph"] = True
        for _ in range(n):
            self.plan()
        torch.cuda.synchronize()
        self.pb.flat.copy_(saved); self.pb._host[:] = saved_h
        self.reset()
        self.stats["plan_ms"] = []
        self.stats["warmup_ms"] = (time.perf_counter() - t0) * 1e3
        self.ready = True
        return self.stats["warmup_ms"]

    def summary(self):
        a = np.array(self.stats["plan_ms"]) if self.stats["plan_ms"] else np.zeros(1)
        return dict(K=self.K, H=self.H, method=self.method, graph=self.stats["graph"], warmup_ms=self.stats["warmup_ms"], n_plans=len(self.stats["plan_ms"]),
                    plan_ms_median=float(np.median(a)), plan_ms_p95=float(np.percentile(a, 95)), plan_ms_max=float(a.max()), stage2_choice=list(self.choice), gpu_mem_peak_MB=float(torch.cuda.max_memory_allocated() / 2**20))
