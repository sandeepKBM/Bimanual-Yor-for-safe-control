"""wm/run_arm.py -- one cell of the tray "coverage vs caution" experiment. SIMULATION / SAVED DATA ONLY
(MuJoCo CPU + torch; no hardware, no network).

  python -m world_model.v2.tray_rep.wm.run_arm --arm {N0,R,Rc} --budget b --seed s --obs-mode {full,tray_only,pc_est}

  N0  nominal data only (no branches); budget is ignored (use 0)
  R   `b` law-R branches (train_kinwm.make_branches law, MuJoCo truth) mixed into every batch (branch-frac 0.5)
  Rc  `b` law-Rc branches (roots drawn uniformly from the not-yet-failed states)
Pipeline (each stage is skipped if its output exists, so a killed run resumes):
  1. WM        TrayTrainer, --wm-steps steps; wm_ckpt.pt every --ckpt-every steps; wm.pt, wm_curve.json at the end
  2. value     reach_value.train on TrayDyn, --value-iters iterations; value/value.pt
  3. gate      P(V > 0) on the oracle's TEST states by label (savable / hold_ok / uncertified); WM margin error
  4. closed loop  WM-greedy on certified-savable TEST states (oracle label 1), T ticks of MuJoCo; failure rate + Wilson CI
  5. references   hold-still and oracle policy on the same states (cached in <data>/oracle/cl_ref_*.json)
  result.json, then _DONE.
Output: <runs>/<tag>/<arm>_<mode>_b<budget>_s<seed>/   (default runs = /common/users/ss5772/bimanual_data/tray_rep/runs)
Branch files: <data>/branches/<law>_b<budget>_s<seed>.npz (made on demand if absent; pre-make them with wm/launch_local.sh).
Seeds: WM init / batches = seed; branches = seed + 1 (the vdirect convention); reference normalisation seed 12345.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep import cl_eval_mj as CL
from world_model.v2.tray_rep import oracle as OR
from world_model.v2.tray_rep.store import read_meta
from world_model.v2.bead_rep import util as U
from world_model.v2.tray_rep.wm import common as C
from world_model.v2.tray_rep.wm import model as M

RUNS = K.ROOT / "runs"
DEFAULT_SETUP = "piper"
CL_T = 50


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def default_data(setup, mode):
    return K.ROOT / f"data_{setup}_{mode}"


def branch_path(data, law, budget, seed):
    return Path(data) / "branches" / f"{law}_b{budget}_s{seed}.npz"


def get_branches(data, law, budget, seed, workers, dev):
    """torch dict of branches; makes the npz (atomic) if missing. branch seed = seed + 1 (vdirect convention)."""
    from world_model.v2.tray_rep import branches as BR
    p = branch_path(data, law, budget, seed)
    if not p.exists():
        t0 = time.time()
        out, meta = BR.make_branches_mj(data, "train", law, budget, seed=seed + 1, workers=workers)
        C.atomic_json(meta, p.with_suffix(".json"))             # json first: the npz's existence implies both are there
        U.atomic_savez(p, **out)
        log(f"[branches {law}] made {budget} in {time.time()-t0:.0f}s: fail {meta['branch_fail_frac']:.3f} "
            f"tick-fail {meta['tick_fail_frac']:.3f} root-failed {meta['root_failed_frac']:.3f}")
    meta = json.loads(p.with_suffix(".json").read_text())
    return BR.load_branches(p, dev), meta


def train_wm(args, data, out, norm, dev):
    from world_model.v2.tray_rep.wm.trainer import TrayTrainer, dump_json
    import torch
    t0 = time.time()
    T = TrayTrainer(args.seed, norm, data, total_steps=args.wm_steps, dev=dev, batch=args.batch, lr=args.lr,
                    branch_frac=args.branch_frac)
    meta = dict(kind=args.arm, budget=args.budget, window_eps=T.window_eps)
    if args.arm != "N0":
        BR, bm = get_branches(data, args.arm, args.budget, args.seed, args.workers, dev)
        T.add_branches(BR)
        meta.update(bm)
        log(f"[branches {args.arm}] b={args.budget} fail_frac={bm['branch_fail_frac']:.3f} "
            f"tick_fail_frac={bm['tick_fail_frac']:.3f} (check before trusting the recipe: failure-heavy branches)")
    ck = out / "wm_ckpt.pt"
    if ck.exists():
        T.load_ckpt(ck)
        log(f"[wm] resumed from step {T.step}")
    tag = f" {args.arm} b{args.budget} s{args.seed} {args.obs_mode}"
    while T.step < args.wm_steps:
        T.train(min(args.ckpt_every, args.wm_steps - T.step), log_every=args.log_every, tag=tag)
        T.save_ckpt(ck)
    wm = T.frozen_wm()
    T.save_wm(out / "wm.pt", extra=dict(norm=norm))
    dump_json(T.curve, out / "wm_curve.json")
    dump_json(meta, out / "wm_meta.json")
    meta["wm_sec"] = time.time() - t0
    del T
    if os.path.exists(ck):
        os.remove(ck)
    return wm, meta


def oracle_gate(wm, net, dev, data, oracle):
    """P(V_wm > 0) by oracle label on TEST states + WM margin error on those states."""
    import torch
    D = C.load_data(data, "test")
    eps = np.unique(oracle["ep"])
    Z = dict(zip(eps.tolist(), C.filtered_latents(wm, D, dev, eps=eps)))
    z = torch.tensor(np.stack([Z[int(e)][int(t)] for e, t in zip(oracle["ep"], oracle["t"])]), device=dev)
    with torch.no_grad():
        lh = wm.margin(z[:, :wm.d_h], z[:, wm.d_h:])
        V = net(z, lh).cpu().numpy(); lh = lh.cpu().numpy()
    lab, l0 = oracle["label"], oracle["l0"]
    res = {}
    for k, name in OR.LABELS.items():
        m = lab == k
        res[name] = dict(n=int(m.sum()), frac_V_pos=float((V[m] > 0).mean()) if m.any() else None,
                         frac_lhat_pos=float((lh[m] > 0).mean()) if m.any() else None,
                         V_mean=float(V[m].mean()) if m.any() else None)
    res["margin_mae_vs_l0"] = float(np.abs(lh - l0).mean())
    return res


def cl_references(data, oracle_path, T, max_states, a_scale=None):
    """hold-still and oracle-policy closed-loop failure on the certified states; cached beside the oracle file."""
    p = Path(oracle_path).with_name(f"{Path(oracle_path).stem}_clref_T{T}_n{max_states}.json")
    if p.exists():
        return json.loads(p.read_text())
    o = OR.load_oracle(oracle_path)
    ref = {}
    for name, pol in (("hold_still", CL.hold_still), ("oracle", CL.OraclePolicy())):
        ref[name], _ = CL.evaluate(pol, data, o, "test", T, max_states)
    C.atomic_json(ref, p)
    return ref


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", required=True, choices=["N0", "R", "Rc"])
    ap.add_argument("--budget", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obs-mode", required=True, choices=list(E.OBS_DIMS) + ["pc_est"])
    ap.add_argument("--setup", default=DEFAULT_SETUP, help="data_<setup>_<obs_mode> under the tray_rep root; also the runs sub-dir")
    ap.add_argument("--data", default="")
    ap.add_argument("--oracle", default="", help="default <data>/oracle/test_v0.npz")
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--wm-steps", type=int, default=15000)
    ap.add_argument("--value-iters", type=int, default=10000)
    ap.add_argument("--value-batch", type=int, default=2048)
    ap.add_argument("--n-imag", type=int, default=200000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--branch-frac", type=float, default=0.5)
    ap.add_argument("--n-ref", type=int, default=C.N_REF, help="reference branches for the fixed normalisation")
    ap.add_argument("--cl-T", type=int, default=CL_T)
    ap.add_argument("--cl-max", type=int, default=None, help="max certified states in the closed loop (default all)")
    ap.add_argument("--ckpt-every", type=int, default=5000)
    ap.add_argument("--log-every", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=4, help="CPU workers for on-demand branch / norm generation")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.arm == "N0":
        a.budget = 0
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    data = Path(a.data) if a.data else default_data(a.setup, a.obs_mode)
    meta = read_meta(data)
    assert meta["obs_mode"] == a.obs_mode, f"{data} was generated with obs_mode={meta['obs_mode']}"
    oracle_path = Path(a.oracle) if a.oracle else data / "oracle" / "test_v0.npz"
    out = Path(a.out) if a.out else Path(a.runs) / a.setup / f"{a.arm}_{a.obs_mode}_b{a.budget}_s{a.seed}"
    U.safe_mkdir(out)
    if (out / "_DONE").exists():
        log(f"[run_arm] {out} is done"); return
    log(f"[run_arm] arm={a.arm} b={a.budget} s={a.seed} mode={a.obs_mode} dev={dev} "
        f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} data={data} out={out} wm_steps={a.wm_steps} "
        f"value_iters={a.value_iters}")
    a_scale, w_scale, alpha = meta.get("a_scale", 1.0), meta.get("w_scale"), meta.get("alpha")
    renv = E.make_env(meta)                           # only for U / scales (the closed loop makes its own)
    U_set = renv.U
    timing = {}

    # ---- 1. WM
    t0 = time.time()
    if (out / "wm.pt").exists():
        wm = M.load_wm(out / "wm.pt", dev)
        wmeta = json.loads((out / "wm_meta.json").read_text()) if (out / "wm_meta.json").exists() else {}
        log("[wm] found wm.pt")
    else:
        norm = C.get_norm(data, a.n_ref, C.REF_SEED, a.workers, log)
        wm, wmeta = train_wm(a, data, out, norm, dev)
    timing["wm_sec"] = time.time() - t0

    # ---- 2. value
    from world_model.v2.tray_rep.wm import value as V
    t0 = time.time()
    vdir = out / "value"
    if (vdir / "value.pt").exists():
        net, _ = V.load_value(vdir / "value.pt", dev)
        log("[value] found value.pt")
    else:
        net, _ = V.train_value(wm, data, dev, vdir, a.value_iters, a.seed, U_set, alpha, batch=a.value_batch,
                               n_imag=a.n_imag, log=log,
                               extra_meta=dict(a_scale=a_scale, w_scale=w_scale, obs_mode=a.obs_mode))
    timing["value_sec"] = time.time() - t0

    # ---- 3. gate
    t0 = time.time()
    oracle = OR.load_oracle(oracle_path)
    gate = oracle_gate(wm, net, dev, data, oracle)
    log("[gate] " + json.dumps({k: (v["frac_V_pos"] if isinstance(v, dict) else v) for k, v in gate.items()}))
    timing["gate_sec"] = time.time() - t0

    # ---- 4. closed loop
    from world_model.v2.tray_rep.wm.policy import WMGreedy
    t0 = time.time()
    pol = WMGreedy(wm, net, dev, U_set, alpha)
    cl, per = CL.evaluate(pol, data, oracle, "test", a.cl_T, a.cl_max)
    np.savez_compressed(out / "cl_states.npz", ep=oracle["ep"][per["sel"]], t=oracle["t"][per["sel"]],
                        fail=per["fail"], fail_tick=per["fail_tick"], dmin=per["dmin"])
    log(f"[closed-loop wm_greedy] fail {cl['fail_rate']:.3f} ci95 {cl['fail_ci95']} n={cl['n']} ({time.time()-t0:.0f}s)")
    timing["cl_sec"] = time.time() - t0

    # ---- 5. references
    t0 = time.time()
    ref = cl_references(data, oracle_path, a.cl_T, a.cl_max)
    log(f"[refs] hold_still fail {ref['hold_still']['fail_rate']:.3f}  oracle fail {ref['oracle']['fail_rate']:.3f}")
    timing["ref_sec"] = time.time() - t0

    timing["total_sec"] = time.time() - t_start
    cur = json.loads((out / "wm_curve.json").read_text()) if (out / "wm_curve.json").exists() else []
    result = dict(arm=a.arm, budget=a.budget, seed=a.seed, obs_mode=a.obs_mode, setup=a.setup, data=str(data),
                  a_scale=a_scale, w_scale=w_scale, alpha=alpha, wm_steps=a.wm_steps, value_iters=a.value_iters,
                  device=dev, wm_meta=wmeta, wm_last_val=cur[-1]["val"] if cur else None,
                  wm_last_train=cur[-1]["train"] if cur else None, gate=gate, cl=cl, ref=ref,
                  n_certified_in_oracle=int((oracle["label"] == OR.SAVABLE).sum()), timing=timing)
    (out / "result.json").write_text(json.dumps(result, indent=1, default=float))
    (out / "_DONE").write_text(time.strftime("%FT%TZ", time.gmtime()))
    log(f"[result] {a.arm} b{a.budget} s{a.seed} {a.obs_mode}: wm_greedy fail {cl['fail_rate']:.3f} "
        f"hold {ref['hold_still']['fail_rate']:.3f} oracle {ref['oracle']['fail_rate']:.3f} total {timing['total_sec']:.0f}s")


if __name__ == "__main__":
    main()
