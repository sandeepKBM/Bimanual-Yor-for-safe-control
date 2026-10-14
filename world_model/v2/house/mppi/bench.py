"""bench.py -- warm-up vs steady-state planner timing on the real S2 problem (perceived zonotope volume, 227 robot points, H=24).  SIMULATION ONLY.
    CUDA_VISIBLE_DEVICES=5 python -m world_model.v2.house.mppi.bench --K 256 1024 4096 --out $MPPI_OUT/bench.json
Reports: cuda_init_ms (first CUDA op in a fresh process), field build, per-K warm-up (kernels + graph capture + dummy plans), then 300 steady plans:
host wall ms/plan (median/p95/max, includes the state copy + readback sync) and pure GPU ms/plan (cuda events around the graph replay), eager vs graph."""
import argparse, json, os, time
t_start = time.perf_counter()
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, nargs="+", default=[256, 1024, 4096])
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--graph-only", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    t0 = time.perf_counter(); torch.zeros(1, device="cuda"); torch.cuda.synchronize(); cuda_init = (time.perf_counter() - t0) * 1e3
    from . import s2 as S, planner as PL
    from world_model.v2.house.zono_rows.ctl import ZonoCtl
    from world_model.v2.house.zono_rows import base_filter as BF
    sc = S.build_scene(); C = sc["C"]
    ctl = ZonoCtl(C, zonos=sc["zs"], names=sc["names"], margin=0.02, alpha1=10.0, alpha2=10.0, d_act=0.15, max_rows=64, tilt=True, refine=0, relax="uniform", tray_level=True, level_delta=0.015, track_bodies=["tray", "ball"])
    bf = BF.BaseFilter(ctl, BF.BaseCfg(), origin_xy=(sc["traj"].x[0], sc["traj"].y[0]), yaw=float(sc["traj"].yaw[0]), data=C.d)
    t0 = time.perf_counter(); prob, pinfo = S.build_problem(sc, ctl, bf); torch.cuda.synchronize(); t_field = (time.perf_counter() - t0) * 1e3
    res = dict(cuda_init_ms=cuda_init, field_build_ms=t_field, n_pts=pinfo["n_pts"], H=prob.H, runs={})
    rng = np.random.default_rng(0)
    for K in a.K:
        for graph in ((True,) if a.graph_only else (True, False)):
            torch.cuda.reset_peak_memory_stats()
            pl = PL.SamplingPlanner(prob, K=K, sigma=(0.08, 0.08, 0.02), plan_dt=0.05, use_graph=graph, compile=a.compile)
            wu = pl.warmup()
            ts, gs = [], []
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            for i in range(a.n):
                y = 0.0 + 0.4 * (i / a.n)                                   # move through the scene so caches/branches vary
                prob.set_state(pose=[3.99 + 0.1 * math_sin(i), 4.505 + y, 1.5708], v=[0.05, 0.1, 0], p0=pinfo["p0"], t_hat=pinfo["t_hat"], s_end=pinfo["s_end"], yaw_ref=pinfo["yaw_ref"])
                t0 = time.perf_counter()
                ev0.record(); pl.plan(); ev1.record(); torch.cuda.synchronize()
                ts.append((time.perf_counter() - t0) * 1e3); gs.append(ev0.elapsed_time(ev1))
            res["runs"][f"K{K}_{'graph' if graph else 'eager'}{'_compiled' if a.compile else ''}"] = dict(warmup_ms=wu, plan_host_ms_median=float(np.median(ts)), plan_host_ms_p95=float(np.percentile(ts, 95)), plan_host_ms_max=float(np.max(ts)),
                                                                    plan_gpu_ms_median=float(np.median(gs)), gpu_mem_peak_MB=float(torch.cuda.max_memory_allocated() / 2**20))
            print(K, "graph" if graph else "eager", a.compile, list(res["runs"].values())[-1], flush=True)
            del pl
    res["total_startup_to_ready_s"] = time.perf_counter() - t_start
    if a.out:
        open(a.out, "w").write(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def math_sin(i):
    import math
    return math.sin(i / 20.0)


if __name__ == "__main__":
    main()
