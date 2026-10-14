"""Multi-GPU (DDP) entry point for FOUND FLOQ (ur5e_benchmark.floq, imported
UNMODIFIED) on our bimanual arm-arm avoidance episodes.

Why this file exists (not an edit to train_bimanual_floq.py or to the vendor
ur5e_benchmark.floq.train module): the vendor `train_floq()` loop is a plain
single-process loop -- no DistributedSampler/DDP -- and its evaluate_floq_model
pass over the FULL val+test split every epoch (at eval_flow_samples=8) is the
actual wall-clock bottleneck (see hj_reachability/floq/train_floq.sbatch's
2026-09-06 timing note: job 273001/272949 did not finish even epoch 1 of 15
in 5+ hours; the ~1.4k train batches/epoch are NOT the dominant cost). Plain
DDP on the train loop alone would not fix that. This script instead:

  1. DDP-wraps the model and shards the TRAIN loader with DistributedSampler
     (global batch size held constant -- see `_per_gpu_batch_size` -- so the
     optimization trajectory matches the single-GPU run's hyperparameters).
  2. Also SHARDS the expensive eval forward passes (val + test) across ranks
     via exact (no-padding, no-duplication) index slicing, then
     `dist.all_gather_object`s the raw per-example arrays and feeds them into
     `ur5e_benchmark.utils.safety_metrics.compute_binary_metrics` and the
     same aggregation shape as vendor `evaluate_floq_model` -- i.e. the
     metric MATH is copied verbatim from vendor eval.py, only the expensive
     forward pass is parallelized. Every rank ends up with mathematically
     identical global/per_horizon/per_task metrics (up to float summation
     order) to what a single-process eval would have produced on the same
     full split.
  3. Skips the per-epoch TEST-split eval (not used for best-checkpoint
     selection, which is val_auroc only -- see vendor train.py) and instead
     runs it ONCE at the end on the best checkpoint's weights. This alone
     removes roughly half the per-epoch eval cost.
  4. Reuses the vendor's dataset-cache build function unmodified, but adds
     (a) a "skip rebuild if the cache already looks complete" fast path
     (mission: reuse the existing dataset, never rebuild if intact -- this
     cache was already built once by job 273272) and (b) rank-0-only
     building with a barrier, since `build_floq_dataset_cache` itself is NOT
     idempotency-safe against concurrent writers.

No vendor loss math, model math, or metric math is edited -- `compute_floq_losses`,
`FloQModel`/`HorizonRiskMLP`, `compute_binary_metrics`, and
`build_floq_dataset_cache` are imported and called unmodified. Only
orchestration (sampling, sharding, gathering, rank-gated I/O) is new.

Single-GPU fallback: if launched without torchrun (WORLD_SIZE unset/1), this
script runs identically to a plain single-process job (no dist calls fire).

Usage (single node, N GPUs, via torchrun -- see train_floq_ddp.sbatch):
    torchrun --standalone --nproc_per_node=N train_bimanual_floq_ddp.py \
        --config configs/bimanual_floq_v1.yaml --output-dir OUT
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset

import paths  # noqa: F401  (sys.path setup for ur5e_benchmark.floq)
from ur5e_benchmark.floq.dataset import (
    FloQTensorDataset,
    build_floq_dataset_cache,
    load_floq_dataset_metadata,
)
from ur5e_benchmark.floq.losses import compute_floq_losses
from ur5e_benchmark.floq.train import _init_model, _to_device  # vendor helpers, unmodified
from ur5e_benchmark.floq.utils import save_json, save_yaml, set_seed, utc_timestamp
from ur5e_benchmark.utils.safety_metrics import compute_binary_metrics


# --------------------------------------------------------------------------
# Distributed bootstrap
# --------------------------------------------------------------------------
def _dist_state() -> dict[str, Any]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_distributed = world_size > 1
    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requested (WORLD_SIZE>1) but CUDA is not available.")
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return {
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_distributed": is_distributed,
        "is_main": rank == 0,
        "device": device,
    }


def _log(dstate: dict[str, Any], msg: str) -> None:
    if dstate["is_main"]:
        print(f"[train_floq_ddp] {msg}", flush=True)


# --------------------------------------------------------------------------
# Dataset root resolution: reuse-if-intact + rank-0-only build (vendor's
# build_floq_dataset_cache is not safe to call concurrently from N ranks --
# it unconditionally overwrites train/val/test.npz + metadata.json).
# --------------------------------------------------------------------------
def _cache_looks_complete(root: Path) -> bool:
    required = ["metadata.json", "train.npz", "val.npz", "test.npz"]
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in required)


def _resolve_dataset_root(config: dict[str, Any], run_dir: Path, dstate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    dataset_root_cfg = config.get("dataset_root")
    synthetic_smoke = bool(config.get("synthetic_smoke", False))

    if dataset_root_cfg is not None and not synthetic_smoke and not bool(config.get("rebuild_dataset", False)):
        dataset_root = Path(str(dataset_root_cfg)).expanduser().resolve()
        return dataset_root, load_floq_dataset_metadata(dataset_root)

    dataset_cache_dir = config.get("dataset_cache_dir")
    dataset_root = Path(str(dataset_cache_dir)).expanduser().resolve() if dataset_cache_dir else (run_dir / "dataset_cache")
    dataset_root.mkdir(parents=True, exist_ok=True)

    if _cache_looks_complete(dataset_root):
        _log(dstate, f"reusing existing FLOQ tensor cache -> {dataset_root} (skipping rebuild)")
        return dataset_root, load_floq_dataset_metadata(dataset_root)

    # Cache missing/incomplete: only rank 0 builds it, everyone else waits.
    if dstate["is_main"]:
        _log(dstate, f"building FLOQ tensor cache -> {dataset_root}")
        build_floq_dataset_cache(
            source_roots=config.get("source_roots", config.get("source_root")),
            output_dir=dataset_root,
            horizons=config.get("horizons", [10, 25, 50, 100, 150]),
            feature_key=str(config.get("feature_key", "state_feature_order")),
            action_key=config.get("action_key"),
            task_filter=str(config.get("task_filter", "both")).strip().lower(),
            feature_alignment_mode=str(config.get("feature_alignment_mode", "intersection")),
            use_target_is_safe_fallback=bool(config.get("use_target_is_safe_fallback", True)),
            drop_censored=bool(config.get("drop_censored", True)),
            dry_run=bool(config.get("dataset_dry_run", False)),
            synthetic_smoke=synthetic_smoke,
            seed=int(config.get("seed", 7)),
        )
    if dstate["is_distributed"]:
        dist.barrier()
    return dataset_root, load_floq_dataset_metadata(dataset_root)


# --------------------------------------------------------------------------
# Exact (no padding/duplication) per-rank index sharding for eval loaders.
# --------------------------------------------------------------------------
def _exact_shard_indices(n: int, world_size: int, rank: int) -> list[int]:
    base, rem = divmod(n, world_size)
    start = rank * base + min(rank, rem)
    end = start + base + (1 if rank < rem else 0)
    return list(range(start, end))


# --------------------------------------------------------------------------
# Sharded eval: run the (expensive, eval_flow_samples>train flow_samples)
# forward pass only over this rank's shard, then all_gather the raw arrays
# and hand them to the SAME aggregation math vendor eval.py uses
# (compute_binary_metrics, per-horizon / per-task breakdown) -- copied
# verbatim from ur5e_benchmark/src/ur5e_benchmark/floq/eval.py so metric
# values are identical to a single-process full-split eval.
# --------------------------------------------------------------------------
def _local_eval_pass(*, model, data_loader, device, config: dict[str, Any]) -> dict[str, list]:
    q_mode = str(config.get("q_mode", "direct_q")).strip().lower()
    flow_steps = int(config.get("flow_steps", 16))
    flow_samples = int(config.get("eval_flow_samples", config.get("flow_samples", 1)))

    model.eval()
    out: dict[str, list] = {
        "episode_id": [], "timestep": [], "y": [], "y_mask": [], "prob": [],
        "horizon": [], "task_id": [], "uncertainty": [], "flow_mse": [],
    }
    with torch.no_grad():
        for batch in data_loader:
            out["episode_id"].extend([str(v) for v in batch["episode_id"]])
            out["timestep"].extend([int(v.item()) for v in batch["timestep"]])
            moved = _to_device(batch, device)
            outputs = model(moved, q_mode=q_mode, flow_steps=flow_steps, flow_samples=flow_samples)

            q_prob = torch.sigmoid(outputs["q_logit"])
            uncertainty = torch.sigmoid(outputs["flow_q_std"]) if "flow_q_std" in outputs else torch.zeros_like(q_prob)

            if "flow_pred" in outputs and "flow_target" in outputs:
                raw = torch.mean((outputs["flow_pred"] - outputs["flow_target"]) ** 2, dim=-1)
                has_future = moved.get("has_future")
                if has_future is None:
                    out["flow_mse"].append(float(raw.mean().item()))
                else:
                    weight = has_future.to(dtype=raw.dtype)
                    denom = weight.sum().clamp_min(1.0)
                    out["flow_mse"].append(float(((raw * weight).sum() / denom).item()))

            out["y"].append(moved["y_unsafe"].detach().cpu().numpy())
            out["y_mask"].append(moved["y_mask"].detach().cpu().numpy())
            out["prob"].append(q_prob.detach().cpu().numpy())
            out["horizon"].append(moved["horizon"].detach().cpu().numpy().astype(np.int64))
            out["task_id"].append(moved["task_id"].detach().cpu().numpy().astype(np.int64))
            out["uncertainty"].append(uncertainty.detach().cpu().numpy())
    return out


def _aggregate_global(chunks: list[dict[str, list]], *, q_mode: str, flow_samples: int) -> dict[str, Any]:
    """Concatenate per-rank local eval outputs and compute metrics with the
    exact same math as vendor ur5e_benchmark.floq.eval.evaluate_floq_model's
    post-loop aggregation (compute_binary_metrics is imported unmodified)."""
    episode_id_all: list[str] = []
    timestep_all: list[int] = []
    y_all, y_mask_all, prob_all, horizon_all, task_id_all, uncertainty_all = [], [], [], [], [], []
    flow_mse_all: list[float] = []
    for c in chunks:
        episode_id_all.extend(c["episode_id"])
        timestep_all.extend(c["timestep"])
        y_all.extend(c["y"])
        y_mask_all.extend(c["y_mask"])
        prob_all.extend(c["prob"])
        horizon_all.extend(c["horizon"])
        task_id_all.extend(c["task_id"])
        uncertainty_all.extend(c["uncertainty"])
        flow_mse_all.extend(c["flow_mse"])

    if not y_all:
        empty_metrics = compute_binary_metrics(np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64))
        return {
            "global": empty_metrics, "per_horizon": {}, "per_task": {}, "risk_curve": {},
            "flow": {"val_flow_mse": None, "q_uses_generated_future": bool(q_mode != "direct_q"), "flow_samples": int(flow_samples)},
        }

    y = np.concatenate(y_all, axis=0).astype(np.float64)
    y_mask = np.concatenate(y_mask_all, axis=0).astype(np.float64)
    prob = np.concatenate(prob_all, axis=0).astype(np.float64)
    horizon = np.concatenate(horizon_all, axis=0).astype(np.int64)
    task_id = np.concatenate(task_id_all, axis=0).astype(np.int64)

    valid = y_mask > 0.5
    y_valid = y[valid].astype(np.int64)
    prob_valid = prob[valid]
    horizon_valid = horizon[valid]
    task_valid = task_id[valid]

    global_metrics = compute_binary_metrics(y_valid, prob_valid)

    per_horizon: dict[str, Any] = {}
    risk_curve: dict[str, Any] = {}
    for h in sorted(set(horizon_valid.tolist())):
        mask = horizon_valid == int(h)
        per_horizon[str(int(h))] = compute_binary_metrics(y_valid[mask], prob_valid[mask])
        risk_curve[str(int(h))] = {
            "mean_predicted_risk": float(np.mean(prob_valid[mask])) if bool(mask.any()) else None,
            "empirical_unsafe_rate": float(np.mean(y_valid[mask])) if bool(mask.any()) else None,
            "num_rows": int(mask.sum()),
        }

    per_task: dict[str, Any] = {}
    for task in sorted(set(task_valid.tolist())):
        mask = task_valid == int(task)
        per_task[str(int(task))] = compute_binary_metrics(y_valid[mask], prob_valid[mask])

    return {
        "global": global_metrics,
        "per_horizon": per_horizon,
        "per_task": per_task,
        "risk_curve": risk_curve,
        "flow": {
            "val_flow_mse": float(np.mean(flow_mse_all)) if flow_mse_all else None,
            "q_uses_generated_future": bool(q_mode != "direct_q"),
            "flow_samples": int(flow_samples),
        },
    }


def _sharded_eval(*, model, dataset, batch_size: int, device, config: dict[str, Any], dstate: dict[str, Any]) -> dict[str, Any]:
    n = len(dataset)
    if dstate["is_distributed"]:
        idx = _exact_shard_indices(n, dstate["world_size"], dstate["rank"])
        shard = Subset(dataset, idx)
    else:
        shard = dataset
    loader = DataLoader(shard, batch_size=int(batch_size), shuffle=False, num_workers=0)
    local = _local_eval_pass(model=model, data_loader=loader, device=device, config=config)

    if dstate["is_distributed"]:
        gathered: list[Any] = [None] * dstate["world_size"]
        dist.all_gather_object(gathered, local)
        chunks = gathered
    else:
        chunks = [local]

    q_mode = str(config.get("q_mode", "direct_q")).strip().lower()
    flow_samples = int(config.get("eval_flow_samples", config.get("flow_samples", 1)))
    return _aggregate_global(chunks, q_mode=q_mode, flow_samples=flow_samples)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def train_floq_ddp(*, config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    dstate = _dist_state()
    device = dstate["device"]
    world_size = dstate["world_size"]
    rank = dstate["rank"]

    seed = int(config.get("seed", 7)) + rank  # decorrelate dataloader/dropout RNG across ranks
    set_seed(seed)

    run_dir = Path(output_dir).expanduser().resolve()
    if dstate["is_main"]:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    if dstate["is_distributed"]:
        dist.barrier()

    dataset_root, dataset_metadata = _resolve_dataset_root(config, run_dir, dstate)
    config["dataset_root"] = str(dataset_root)

    global_batch = int(config.get("batch_size", 128))
    per_gpu_batch = max(1, global_batch // world_size) if dstate["is_distributed"] else global_batch
    effective_global_batch = per_gpu_batch * world_size
    _log(
        dstate,
        f"world_size={world_size} global_batch(config)={global_batch} "
        f"per_gpu_batch={per_gpu_batch} effective_global_batch={effective_global_batch} "
        f"(kept ~constant vs single-GPU run; learning_rate left UNCHANGED at "
        f"{config.get('learning_rate')})",
    )

    train_ds = FloQTensorDataset(dataset_root=dataset_root, split="train")
    val_ds = FloQTensorDataset(dataset_root=dataset_root, split="val")
    test_ds = FloQTensorDataset(dataset_root=dataset_root, split="test")

    if dstate["is_distributed"]:
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, seed=int(config.get("seed", 7)), drop_last=True
        )
        train_loader = DataLoader(train_ds, batch_size=per_gpu_batch, shuffle=False, sampler=train_sampler, num_workers=0)
    else:
        train_sampler = None
        train_loader = DataLoader(train_ds, batch_size=per_gpu_batch, shuffle=True, num_workers=0)

    model = _init_model(config, dataset_metadata, device)
    if dstate["is_distributed"]:
        # find_unused_parameters=True: compute_floq_losses's horizon-monotonicity
        # term (use_horizon_monotonicity_loss) calls model.forward_q(...) directly
        # (bypassing DDP.__call__ via attribute passthrough) in ADDITION to the
        # tracked model(...) forward. q_mode="both" means every param is already
        # touched by the tracked forward, so this is defensive insurance, not a
        # known-necessary requirement -- cheap for a model this small.
        model = DDP(model, device_ids=[dstate["local_rank"]], output_device=dstate["local_rank"], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config.get("learning_rate", 1e-3)), weight_decay=float(config.get("weight_decay", 1e-4))
    )

    q_mode = str(config.get("q_mode", "direct_q")).strip().lower()
    flow_steps = int(config.get("flow_steps", 16))
    flow_samples = int(config.get("flow_samples", 1))

    num_epochs = int(config.get("epochs", 20))
    max_batches = int(config.get("max_train_batches", 0))
    train_history: list[dict[str, Any]] = []
    val_history: list[dict[str, Any]] = []

    best_score = -float("inf")
    best_state_dict = None
    best_epoch = None
    best_val_report = None

    t_start = time.time()
    for epoch in range(1, num_epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses: dict[str, list[float]] = {
            "loss_total": [], "loss_q": [], "loss_flow": [], "loss_td": [], "loss_horizon_mono": [],
        }
        for batch_idx, batch in enumerate(train_loader, start=1):
            if max_batches > 0 and batch_idx > max_batches:
                break
            moved = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(moved, q_mode=q_mode, flow_steps=flow_steps, flow_samples=flow_samples)
            # compute_floq_losses calls model.forward_q(...) directly (a custom
            # method, not model.__call__) for the TD target (under no_grad) and
            # the horizon-monotonicity term. A DDP wrapper's nn.Module.__getattr__
            # only resolves its OWN _modules/_parameters/_buffers, not attributes
            # of the wrapped submodule -- so `model.forward_q` raises
            # AttributeError on the DDP object (confirmed by local 2-GPU smoke
            # test, see docs/runbooks/slurm.md's FLOQ DDP notes). Pass the
            # underlying module for this call; the gradients it produces still
            # flow into the SAME parameter tensors DDP is watching, and since
            # `outputs = model(...)` above already armed the reducer for this
            # iteration, DDP's per-parameter hooks fire correctly on the single
            # `losses["loss_total"].backward()` call below regardless of which
            # object's method created a given autograd node.
            loss_model = model.module if isinstance(model, DDP) else model
            losses = compute_floq_losses(model=loss_model, batch=moved, outputs=outputs, config=config)
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("clip_grad", 1.0)))
            optimizer.step()
            for key in epoch_losses:
                epoch_losses[key].append(float(losses[key].item()))

        # Average this rank's epoch losses, then all-reduce across ranks so the
        # SAVED train_history reflects the true global epoch mean, not one
        # shard's -- purely a logging/diagnostics fix, does not change the
        # optimization itself (gradients are already DDP-averaged every step).
        local_means = torch.tensor(
            [float(np.mean(v) if v else 0.0) for v in epoch_losses.values()], dtype=torch.float64, device=device
        )
        if dstate["is_distributed"]:
            dist.all_reduce(local_means, op=dist.ReduceOp.SUM)
            local_means /= world_size
        train_row = {"epoch": epoch, **dict(zip(epoch_losses.keys(), (float(x) for x in local_means.tolist())))}
        train_history.append(train_row)

        # Sharded val eval (every epoch, for best-checkpoint selection).
        val_report = _sharded_eval(
            model=model, dataset=val_ds, batch_size=per_gpu_batch, device=device, config=config, dstate=dstate
        )
        val_row = {"epoch": epoch, "val_global": val_report["global"]}
        val_history.append(val_row)

        val_auroc = val_report["global"].get("auroc")
        val_acc = float(val_report["global"].get("accuracy", 0.0))
        score = float(val_auroc) if val_auroc is not None else val_acc
        elapsed = time.time() - t_start
        _log(dstate, f"epoch {epoch}/{num_epochs} loss_total={train_row['loss_total']:.4f} val_auroc={val_auroc} elapsed={elapsed:.1f}s")

        if score >= best_score:
            best_score = score
            best_epoch = epoch
            best_val_report = val_report
            model_to_save = model.module if isinstance(model, DDP) else model
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model_to_save.state_dict().items()}
            if dstate["is_main"]:
                torch.save(
                    {
                        "epoch": epoch, "model_state_dict": best_state_dict, "optimizer_state_dict": optimizer.state_dict(),
                        "config": config, "dataset_metadata": dataset_metadata, "val_report": val_report,
                    },
                    run_dir / "best_checkpoint.pt",
                )

        if dstate["is_main"]:
            model_to_save = model.module if isinstance(model, DDP) else model
            torch.save(
                {
                    "epoch": epoch, "model_state_dict": model_to_save.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                    "config": config, "dataset_metadata": dataset_metadata, "val_report": val_report,
                },
                run_dir / "checkpoint.pt",
            )
        if dstate["is_distributed"]:
            dist.barrier()

    # Final TEST-split eval, ONCE, on the best checkpoint's weights (not every
    # epoch -- test_report was never used for best-checkpoint selection in the
    # vendor loop either, so this changes nothing about which checkpoint wins,
    # it only removes ~half the per-epoch eval cost).
    model_to_save = model.module if isinstance(model, DDP) else model
    if best_state_dict is not None:
        model_to_save.load_state_dict(best_state_dict)
    final_test_report = _sharded_eval(
        model=model, dataset=test_ds, batch_size=per_gpu_batch, device=device, config=config, dstate=dstate
    )

    if dstate["is_main"]:
        save_json(run_dir / "train_metrics.json", {"history": train_history})
        save_json(run_dir / "val_metrics.json", {"history": val_history})
        save_json(run_dir / "test_metrics.json", {"epoch": best_epoch, "test_global": final_test_report["global"]})
        save_yaml(run_dir / "config_resolved.yaml", config)
        save_json(run_dir / "dataset_metadata.json", dataset_metadata)

        summary_payload = {
            "model_type": str(config.get("model_type", "floq")).strip().lower(),
            "q_mode": q_mode,
            "flow_auxiliary_only": bool(q_mode == "direct_q" and str(config.get("model_type", "floq")).strip().lower() == "floq"),
            "uses_td_loss": bool(config.get("use_td_loss", False)),
            "synthetic_smoke": bool(config.get("synthetic_smoke", False)),
            "dataset_root": str(dataset_root),
            "best_epoch": best_epoch,
            "best_val_auroc": best_val_report["global"].get("auroc") if best_val_report else None,
            "best_val_accuracy": best_val_report["global"].get("accuracy") if best_val_report else None,
            "final_test_auroc": final_test_report["global"].get("auroc"),
            "final_test_accuracy": final_test_report["global"].get("accuracy"),
            "world_size": world_size,
            "effective_global_batch": effective_global_batch,
            "timestamp": utc_timestamp(),
        }
        save_json(run_dir / "run_summary.json", summary_payload)
        print(json.dumps(summary_payload, indent=2, default=str))

    if dstate["is_distributed"]:
        dist.barrier()
        dist.destroy_process_group()

    return {
        "run_dir": str(run_dir),
        "dataset_root": str(dataset_root),
        "best_checkpoint": str(run_dir / "best_checkpoint.pt"),
        "checkpoint": str(run_dir / "checkpoint.pt"),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--epochs", type=int, default=None, help="override config['epochs']")
    p.add_argument("--max-train-batches", type=int, default=None, help="override config['max_train_batches']")
    args = p.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.max_train_batches is not None:
        config["max_train_batches"] = args.max_train_batches

    train_floq_ddp(config=config, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
