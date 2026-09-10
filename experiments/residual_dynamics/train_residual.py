#!/usr/bin/env python
"""experiments/residual_dynamics/train_residual.py

Trains an MLP or SIREN to predict the residual dynamics
(residual = qacc_realized - a_cmd, see perturbed_plant.py / README.md) from
[q, qd, a] (36-D) -> residual (12-D), on a dataset produced by
collect_residual_data.py.

Decisive metrics (printed + saved to <out>.metrics.json):
  - baseline_rmse: RMSE of the "bare FL" prediction (predicts
    residual=0), i.e. just ||residual|| on the val set -- this is the
    error the FL controller ALREADY has with no learned correction.
  - model_val_rmse: RMSE of (model(x) - residual) on the val set, same
    units (rad/s^2), same elementwise convention as baseline_rmse so the
    two are directly comparable.
  - fraction_removed = 1 - model_val_rmse / baseline_rmse: fraction of the
    FL-vs-real-plant mismatch the learned residual model removes.

Inputs/outputs are standardized (mean/std from the TRAIN split only); the
scaler (mean/std for both x and y) is saved in the checkpoint so a caller
can invert it at inference time.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import build_model  # noqa: E402

EPS = 1e-8


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=str, required=True, help="path to .npz from collect_residual_data.py")
    p.add_argument("--arch", type=str, choices=["mlp", "siren"], required=True)
    p.add_argument("--epochs", type=int, required=True)
    p.add_argument("--out", type=str, required=True, help="checkpoint path (.pt); metrics saved alongside")
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 256])
    p.add_argument("--activation", type=str, default="relu", help="MLP only")
    p.add_argument("--siren-w0-first", type=float, default=30.0,
                    help="SIREN only. Paper-standard default is 30.0, tuned for "
                         "coordinate-like inputs normalized to [-1,1] (images/SDFs). "
                         "On heavy-tailed tabular targets (like this residual-dynamics "
                         "data) with few epochs, w0=30's high-frequency bias can fail "
                         "to converge at all -- see README.md's smoke caveat. Lower "
                         "(e.g. 3-6) if a quick/small run needs to actually fit.")
    p.add_argument("--siren-w0-hidden", type=float, default=1.0, help="SIREN only, hidden-layer w0.")
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    return p


def load_dataset(path: str):
    d = np.load(path)
    state, a_cmd, residual = d["state"], d["a_cmd"], d["residual"]
    x = np.concatenate([state, a_cmd], axis=1).astype(np.float32)   # (N,36) = [q,qd,a]
    y = residual.astype(np.float32)                                  # (N,12)
    return x, y


def standardize_fit(x: np.ndarray):
    mean = x.mean(axis=0)
    std = x.std(axis=0) + EPS
    return mean.astype(np.float32), std.astype(np.float32)


def rmse(err: np.ndarray) -> float:
    """Elementwise RMSE over ALL dims/samples -- sqrt(mean(err**2))."""
    return float(np.sqrt(np.mean(err ** 2)))


def main():
    args = build_arg_parser().parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    x, y = load_dataset(args.data)
    n = x.shape[0]
    perm = rng.permutation(n)
    n_val = max(1, int(round(n * args.val_frac)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train, y_train = x[train_idx], y[train_idx]
    x_val, y_val = x[val_idx], y[val_idx]

    x_mean, x_std = standardize_fit(x_train)
    y_mean, y_std = standardize_fit(y_train)

    def std_x(a):
        return (a - x_mean) / x_std

    def std_y(a):
        return (a - y_mean) / y_std

    def unstd_y(a):
        return a * y_std + y_mean

    x_train_t = torch.from_numpy(std_x(x_train))
    y_train_t = torch.from_numpy(std_y(y_train))
    x_val_t = torch.from_numpy(std_x(x_val))
    y_val_t = torch.from_numpy(std_y(y_val))

    if args.arch == "mlp":
        arch_kwargs = {"activation": args.activation}
    else:
        arch_kwargs = {"w0_first": args.siren_w0_first, "w0_hidden": args.siren_w0_hidden}
    model = build_model(args.arch, hidden_dims=tuple(args.hidden_dims), **arch_kwargs)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    # baseline (predict residual = 0) RMSE on the val set, RAW units.
    baseline_rmse = rmse(y_val)

    n_train = x_train_t.shape[0]
    loss_history = []
    for epoch in range(args.epochs):
        model.train()
        epoch_perm = torch.from_numpy(rng.permutation(n_train))
        epoch_loss = 0.0
        for start in range(0, n_train, args.batch_size):
            idx = epoch_perm[start:start + args.batch_size]
            xb, yb = x_train_t[idx], y_train_t[idx]
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(idx)
        epoch_loss /= n_train
        loss_history.append(epoch_loss)
        print(f"[train:{args.arch}] epoch {epoch + 1}/{args.epochs}  "
              f"train_mse(standardized)={epoch_loss:.5f}")

    model.eval()
    with torch.no_grad():
        pred_val_std = model(x_val_t).numpy()
    pred_val_raw = unstd_y(pred_val_std)
    model_val_rmse = rmse(pred_val_raw - y_val)
    fraction_removed = 1.0 - model_val_rmse / baseline_rmse

    print(f"[train:{args.arch}] baseline_rmse (||residual||, val)   = {baseline_rmse:.4f} rad/s^2")
    print(f"[train:{args.arch}] model_val_rmse                       = {model_val_rmse:.4f} rad/s^2")
    print(f"[train:{args.arch}] fraction_removed = 1 - model/baseline = {fraction_removed:.4f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "arch": args.arch,
        "hidden_dims": args.hidden_dims,
        "activation": args.activation,
        "siren_w0_first": args.siren_w0_first,
        "siren_w0_hidden": args.siren_w0_hidden,
        "state_dict": model.state_dict(),
        "x_mean": x_mean, "x_std": x_std,
        "y_mean": y_mean, "y_std": y_std,
    }, out_path)

    metrics = {
        "arch": args.arch,
        "data": args.data,
        "n_total": int(n), "n_train": int(len(train_idx)), "n_val": int(len(val_idx)),
        "epochs": args.epochs,
        "hidden_dims": args.hidden_dims,
        "siren_w0_first": args.siren_w0_first if args.arch == "siren" else None,
        "siren_w0_hidden": args.siren_w0_hidden if args.arch == "siren" else None,
        "loss_history_standardized_mse": loss_history,
        "baseline_rmse": baseline_rmse,
        "model_val_rmse": model_val_rmse,
        "fraction_removed": fraction_removed,
    }
    metrics_path = out_path.with_suffix(out_path.suffix + ".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train:{args.arch}] wrote checkpoint {out_path} and metrics {metrics_path}")


if __name__ == "__main__":
    main()
