"""Thin, read-only-study-derived value-query runtime for a trained
``BimanualArmArmAvoidance`` DeepReach checkpoint.

Adapted from (studied, not imported -- no ``ur5e_benchmark`` dependency added
to this repo):
  - ``ur5e_benchmark.planning.hj_value_runtime.H10ValueRuntime`` (run-config +
    checkpoint loading, batched forward-pass evaluation of a DeepReach BVP
    model at ``(tau, state)``);
  - ``ur5e_benchmark.hj_eval.hj_inference.TaskBHJInference`` (the thin
    "value at a flat state" query wrapper on top of the runtime);
  - ``ur5e_benchmark.safety_models.interface.SafetyValueModel`` (the
    provider-neutral protocol: ``state_dim``, ``safe_when_leq_zero``,
    ``values(flat_states, tau_seconds) -> np.ndarray``, ``describe()``).

Deliberately simplified relative to the ur5e source: no
``variable_horizon``/``horizon_conditioned_scalar``/``survival_risk_monitor``
support (those are ur5e-specific multi-horizon-head features; our
``train_config.yaml`` trains a single scalar ``deepreach_model: vanilla``
output, so ``_batched_forward`` here only implements that one path -- see
``eval_runtime._batched_forward`` in the ur5e source for the full
variable-horizon logic if this is ever extended). Not importing
``ur5e_benchmark`` at all keeps this file usable from either the plain
bimanual env or the ``vlaism-deepreach`` env without adding a cross-repo
package dependency.

SIGN CONVENTION (important, do not mix up with ``dynamics.boundary_fn``):
``dynamics.BimanualPairAvoidance.boundary_fn``/``deepreach_dynamics``'s own
``l(x)`` is positive-safe (l(x) > 0 => safe). But ``export_dataset.py``
writes CSV ``target_value`` with the OPPOSITE (ur5e-style) convention --
``safe_when_leq_zero`` (target_value <= 0 => safe) -- and a network trained
with ``supervised_bootstrap`` against that CSV inherits THAT convention in
its raw output. So ``BimanualSafetyValueModel.safe_when_leq_zero`` below is
``True`` (matching what the checkpoint was actually supervised on), not
``dynamics.py``'s raw l(x) sign.

Not wired into any eval/steering code yet -- this is a read-only interface
study + query-path scaffold. No checkpoint exists yet to load (see
README.md); this module has been import-smoke-tested only (no real
run_dir/checkpoint available at scaffold time).
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_module_from_path(module_name: str, file_path: str):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_run_config(run_dir: Path) -> dict:
    """Verbatim port of ur5e_benchmark.deepreach.eval_runtime._load_run_config."""
    config_path = Path(run_dir) / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found under {run_dir}")
    return dict(yaml.safe_load(config_path.read_text()) or {})


def _resolve_checkpoint(run_dir: Path, checkpoint: str | None) -> Path:
    """Verbatim port of ur5e_benchmark.deepreach.eval_runtime._resolve_checkpoint."""
    run_dir = Path(run_dir)
    if checkpoint:
        path = Path(checkpoint)
        if not path.is_absolute():
            path = run_dir / checkpoint
        return path
    current_path = run_dir / "training" / "checkpoints" / "model_current.pth"
    if current_path.exists():
        return current_path
    candidates = sorted((run_dir / "training" / "checkpoints").glob("model_epoch_*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found under {run_dir / 'training' / 'checkpoints'}")
    return candidates[-1]


def build_model_and_dynamics(
    *,
    run_config: dict,
    data_root: Path,
    checkpoint_path: Path | None,
    device: str,
    deepreach_repo: Path,
):
    """Adapted from ur5e_benchmark.deepreach.model_loading.build_model_and_dynamics,
    simplified to the single-scalar-output (non-variable-horizon) case that
    BimanualArmArmAvoidance / train_config.yaml uses."""
    import torch

    deepreach_repo = str(Path(deepreach_repo).resolve())
    if deepreach_repo not in sys.path:
        sys.path.insert(0, deepreach_repo)
    from dynamics import dynamics as deepreach_dynamics  # type: ignore[import-not-found]
    from utils import modules  # type: ignore[import-not-found]

    dynamics_class = getattr(deepreach_dynamics, str(run_config["dynamics_class"]))
    dyn = dynamics_class(
        data_root=str(Path(data_root).resolve()),
        set_mode=str(run_config.get("set_mode", "avoid")),
        deepreach_model=str(run_config.get("deepreach_model", "vanilla")),
    )
    model = modules.build_bvp_model(run_config, in_features=dyn.input_dim, out_features=1)
    if checkpoint_path is not None:
        checkpoint_obj = torch.load(Path(checkpoint_path).resolve(), map_location=device)
        state_dict = checkpoint_obj["model"] if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj else checkpoint_obj
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, dyn


def _batched_forward(model, dyn, coords, batch_size: int):
    """Simplified port of ur5e_benchmark.deepreach.eval_runtime._batched_forward
    (single scalar output only -- see module docstring)."""
    import torch

    outputs = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], batch_size):
            batch = coords[start:start + batch_size]
            model_input = dyn.coord_to_input(batch)
            result = model({"coords": model_input})
            raw_out = result["model_out"]
            if raw_out.ndim > 1 and raw_out.shape[-1] == 1:
                raw_out = raw_out.squeeze(dim=-1)
            values = dyn.io_to_value(result["model_in"], raw_out)
            outputs.append(values.detach().cpu())
    return torch.cat(outputs, dim=0)


class BimanualHJValueRuntime:
    """Loads a DeepReach BimanualArmArmAvoidance run and evaluates scalar
    value at (tau, state). Port of ur5e_benchmark's H10ValueRuntime.

    State vectors must match dataset_description.json's state_order (i.e.
    dynamics.BimanualPairAvoidance.state_feature_order: q_ego_0..5, q_int_0..5).
    Requires torch + the DeepReach repo importable.
    """

    DEFAULT_DEEPREACH_REPO = "/common/users/ss5772/DeepReach/deepreach"

    def __init__(
        self,
        run_dir: Path | str,
        *,
        checkpoint: str | None = None,
        device: str = "cpu",
        batch_size: int = 256,
        deepreach_repo: Path | str | None = None,
    ) -> None:
        import torch

        self.run_dir = Path(run_dir).resolve()
        self.device = str(device)
        self.batch_size = int(batch_size)
        self._torch = torch
        self._run_config = _load_run_config(self.run_dir)
        data_root = Path(self._run_config["data_root"]).resolve()
        ck = _resolve_checkpoint(self.run_dir, checkpoint)

        repo = Path(deepreach_repo or self.DEFAULT_DEEPREACH_REPO)
        self._model, self._dynamics = build_model_and_dynamics(
            run_config=self._run_config,
            data_root=data_root,
            checkpoint_path=ck,
            device=self.device,
            deepreach_repo=repo,
        )
        self._checkpoint_path = ck
        self._data_root = data_root

    @property
    def state_dim(self) -> int:
        return int(self._dynamics.state_dim)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self._checkpoint_path)

    @property
    def data_root(self) -> Path:
        return Path(self._data_root)

    @property
    def run_config(self) -> dict:
        return dict(self._run_config)

    def values(self, flat_states: np.ndarray, tau_seconds: float | np.ndarray) -> np.ndarray:
        torch = self._torch
        fs = np.asarray(flat_states, dtype=np.float64)
        if fs.ndim == 1:
            fs = fs.reshape(1, -1)
        n = fs.shape[0]
        if fs.shape[1] != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {fs.shape[1]}")

        if np.ndim(tau_seconds) == 0:
            taus = np.full((n,), float(tau_seconds), dtype=np.float64)
        else:
            taus = np.asarray(tau_seconds, dtype=np.float64).reshape(-1)
            if taus.shape[0] != n:
                raise ValueError("tau_seconds must broadcast to batch size")

        states_t = torch.tensor(fs, dtype=torch.float32, device=self.device)
        tau_t = torch.tensor(taus, dtype=torch.float32, device=self.device)
        coords = torch.cat((tau_t.unsqueeze(1), states_t), dim=1)
        out = _batched_forward(self._model, self._dynamics, coords, batch_size=self.batch_size)
        return out.detach().cpu().numpy().reshape(-1)

    def value_one(self, flat_state: np.ndarray, tau_seconds: float) -> float:
        return float(self.values(np.asarray(flat_state).reshape(1, -1), tau_seconds)[0])


@dataclass(frozen=True)
class BimanualHJInferenceConfig:
    tau_seconds: float = 1.0  # tMax placeholder from train_config.yaml (no real horizon semantics yet)
    device: str = "cpu"
    batch_size: int = 512


class BimanualHJInference:
    """Thin "value at a flat 12-d joint state" wrapper -- port of
    ur5e_benchmark.hj_eval.hj_inference.TaskBHJInference, minus the
    TaskWrapper/state_variant projection (this problem's state IS the flat
    12-d joint vector already, no task-state projection needed)."""

    def __init__(self, run_dir: Path | str, *, checkpoint: str | None = None,
                 config: BimanualHJInferenceConfig | None = None) -> None:
        cfg = config or BimanualHJInferenceConfig()
        self.cfg = cfg
        self.runtime = BimanualHJValueRuntime(
            run_dir, checkpoint=checkpoint, device=str(cfg.device), batch_size=int(cfg.batch_size),
        )

    def value_from_flat_state(self, flat_state: np.ndarray, *, tau_seconds: float | None = None) -> float:
        tau = float(self.cfg.tau_seconds if tau_seconds is None else tau_seconds)
        flat = np.asarray(flat_state, dtype=np.float32).reshape(-1)
        return float(self.runtime.value_one(flat, tau))


class BimanualSafetyValueModel:
    """Duck-types ur5e_benchmark.safety_models.interface.SafetyValueModel
    (state_dim, safe_when_leq_zero, values(...), describe()) without
    importing that package, so this can plug into the same
    provider-neutral steering/eval contract described there.

    NOTE (sign convention): True, matching the CSV target_value convention
    the checkpoint was actually trained against (see module docstring) --
    the OPPOSITE of dynamics.boundary_fn's own raw l(x) sign.
    """

    def __init__(self, run_dir: Path | str, *, checkpoint: str | None = None,
                 device: str = "cpu", batch_size: int = 512) -> None:
        self._runtime = BimanualHJValueRuntime(run_dir, checkpoint=checkpoint, device=device, batch_size=batch_size)

    @property
    def state_dim(self) -> int:
        return self._runtime.state_dim

    @property
    def safe_when_leq_zero(self) -> bool:
        return True

    def values(self, flat_states: np.ndarray, tau_seconds: float | np.ndarray) -> np.ndarray:
        return self._runtime.values(flat_states, tau_seconds)

    def describe(self) -> dict[str, Any]:
        return {
            "provider": "bimanual_hj_reachability",
            "dynamics_class": self._runtime.run_config.get("dynamics_class"),
            "run_dir": str(self._runtime.run_dir),
            "checkpoint_path": str(self._runtime.checkpoint_path),
            "data_root": str(self._runtime.data_root),
            "safe_when_leq_zero": True,
            "state_feature_order": "q_ego_0..5, q_int_0..5 (see dynamics.BimanualPairAvoidance.state_feature_order)",
        }
