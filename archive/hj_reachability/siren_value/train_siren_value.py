#!/usr/bin/env python3
"""Self-supervised training of a SIREN safety value function via the Fisac
discounted-safety Bellman operator, for
``hj_reachability.dynamics.BimanualPairAvoidance`` (bimanual arm-arm
avoidance differential game).

No dataset: each step samples a fresh batch of states via
``BimanualPairAvoidance.sample_states_mixed`` (workspace-oversampled, see
that method's docstring), computes the one-step Bellman target with a
frozen/target copy of the network (``fisac_safety.bellman_target``), and
minimizes MSE against the online network's prediction. The target net is
periodically hard-updated from the online net (standard DQN-style target
network, needed because the backup target itself is a function of the
network being trained -- without freezing it, the regression target moves
every step and the recursion cannot settle).

This is intentionally SCAFFOLD/SMOKE code (see the task's own instruction:
"Do NOT run long training (the owner runs that)") -- defaults below are
sized for a real-scale run; the smoke invocation used far smaller
``--iters``/``--batch`` (see README.md's "Smoke result" section) since
``boundary_fn`` is a per-sample, non-batched MuJoCo FK call (~4-5ms/call on
CPU; see ``fisac_safety.boundary_fn_batch``'s docstring), so wall-clock is
dominated by that loop, not the network.

Usage:
    python train_siren_value.py --iters 300 --batch 512 --gamma 0.999 \\
        --dt 0.05 --width 128 --depth 3 --out outputs/smoke_run
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # hj_reachability/
import dynamics  # noqa: E402  (hj_reachability/dynamics.py -- BimanualPairAvoidance, reused as-is)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # hj_reachability/siren_value/
from siren import SIREN  # noqa: E402
from fisac_safety import FisacConfig, bellman_target, boundary_fn_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--dt", type=float, default=0.05, help="game timestep [s] for the one-step Bellman backup")
    p.add_argument("--width", type=int, default=128)
    p.add_argument("--depth", type=int, default=3, help="number of hidden SineLayers")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--target-update-every", type=int, default=10,
                   help="hard-update the frozen target net from the online net every N iters")
    p.add_argument("--eval-every", type=int, default=25,
                   help="how often to log the sign-agreement(V, l) sanity metric")
    p.add_argument("--eval-batch", type=int, default=256, help="held-out batch size for the sign-agreement metric")
    p.add_argument("--task-region-frac", type=float, default=0.5,
                   help="fraction of sampled states drawn near the shared workspace (see sample_states_mixed)")
    p.add_argument("--u-bar", type=float, default=None,
                   help="OPTIONAL: override BOTH u_bar_ego and u_bar_int (rad/s). "
                        "This is a KINEMATIC game (no link mass/friction in the dynamics), so "
                        "domain-randomization over physical params reduces, for this abstraction, to "
                        "randomizing u_bar (achievable joint speed) and boundary geometry -- see README.md's "
                        "'Physical-param note'. This flag is the hook for that; NOT wired to any randomization "
                        "loop yet (out of scope for this scaffold).")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, required=True, help="output directory for checkpoint.pt + metrics.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    dyn = dynamics.BimanualPairAvoidance()
    if args.u_bar is not None:
        dyn.cfg = dynamics.BimanualPairAvoidanceConfig(
            u_bar_ego=args.u_bar, u_bar_int=args.u_bar,
            r_ego=dyn.cfg.r_ego, r_int=dyn.cfg.r_int, d_min=dyn.cfg.d_min,
        )
    cfg = FisacConfig(gamma=args.gamma, dt=args.dt,
                       u_bar_ego=dyn.cfg.u_bar_ego, u_bar_int=dyn.cfg.u_bar_int)

    device = torch.device(args.device)
    online_net = SIREN(in_features=dyn.state_dim, width=args.width, depth=args.depth).to(device)
    target_net = copy.deepcopy(online_net).to(device)
    for p in target_net.parameters():
        p.requires_grad_(False)

    optimizer = torch.optim.Adam(online_net.parameters(), lr=args.lr)

    # Fixed anchor states for the mandatory deep-safe / deep-unsafe sanity
    # check (task spec): sample a large fixed pool once, up front, and pick
    # the most-safe (max l(x)) and most-unsafe (min l(x)) states in it as
    # anchors so V's value AT THOSE SPECIFIC STATES can be tracked across
    # training (a fresh random batch every iter would make "trend" meaningless).
    anchor_pool_np = dyn.sample_states_mixed(2000, np.random.default_rng(args.seed + 1),
                                              task_region_frac=args.task_region_frac)
    anchor_l = boundary_fn_batch(dyn, anchor_pool_np)
    deep_safe_state = anchor_pool_np[np.argmax(anchor_l)]
    deep_unsafe_state = anchor_pool_np[np.argmin(anchor_l)]
    deep_safe_l, deep_unsafe_l = float(anchor_l.max()), float(anchor_l.min())
    anchors = torch.as_tensor(np.stack([deep_safe_state, deep_unsafe_state]), dtype=torch.float32, device=device)
    print(f"Anchors: deep-safe l(x)={deep_safe_l:.4f}, deep-unsafe l(x)={deep_unsafe_l:.4f}", flush=True)

    metrics = {"iter": [], "bellman_residual": [], "sign_agreement": [],
               "deep_safe_l": deep_safe_l, "deep_unsafe_l": deep_unsafe_l,
               "anchor_V": []}
    t0 = time.time()

    for it in range(args.iters):
        states_np = dyn.sample_states_mixed(args.batch, rng, task_region_frac=args.task_region_frac)
        states = torch.as_tensor(states_np, dtype=torch.float32, device=device)
        l_np = boundary_fn_batch(dyn, states_np)
        l_x = torch.as_tensor(l_np, dtype=torch.float32, device=device)

        target, grad = bellman_target(states, l_x, target_net, cfg)
        assert torch.isfinite(grad).all(), "non-finite gradient from the min-max backup"

        v_pred = online_net(states).squeeze(-1)
        loss = torch.nn.functional.mse_loss(v_pred, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (it + 1) % args.target_update_every == 0:
            target_net.load_state_dict(online_net.state_dict())

        residual = float(loss.item()) ** 0.5  # report RMS residual, sqrt(MSE), in l(x) units
        metrics["iter"].append(it)
        metrics["bellman_residual"].append(residual)

        if it % args.eval_every == 0 or it == args.iters - 1:
            eval_states_np = dyn.sample_states_mixed(args.eval_batch, rng, task_region_frac=args.task_region_frac)
            eval_l = boundary_fn_batch(dyn, eval_states_np)
            with torch.no_grad():
                eval_v = online_net(torch.as_tensor(eval_states_np, dtype=torch.float32, device=device))
                eval_v = eval_v.squeeze(-1).cpu().numpy()
            sign_agreement = float(np.mean(np.sign(eval_v) == np.sign(eval_l)))
            metrics["sign_agreement"].append(sign_agreement)
            with torch.no_grad():
                anchor_v = online_net(anchors).squeeze(-1).cpu().numpy().tolist()
            metrics["anchor_V"].append({"iter": it, "V_deep_safe": anchor_v[0], "V_deep_unsafe": anchor_v[1]})
            elapsed = time.time() - t0
            print(f"[iter {it:5d}/{args.iters}] bellman_residual={residual:.6f} "
                  f"sign_agreement={sign_agreement:.4f} V(deep_safe)={anchor_v[0]:.5f} "
                  f"V(deep_unsafe)={anchor_v[1]:.5f} elapsed={elapsed:.1f}s", flush=True)
        else:
            metrics["sign_agreement"].append(None)

    torch.save({
        "online_state_dict": online_net.state_dict(),
        "target_state_dict": target_net.state_dict(),
        "args": vars(args),
    }, os.path.join(args.out, "checkpoint.pt"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Done. Checkpoint + metrics written to {args.out}")


if __name__ == "__main__":
    main()
