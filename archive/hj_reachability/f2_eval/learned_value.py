"""F2: load the TRAINED DeepReach bimanual arm-arm value net and expose
value + state-gradient at a (q_left, q_right) joint state.

NEW file (F2 eval only). Reuses (read-only) ``value_runtime.build_model_and_dynamics``
and the DeepReach fork's ``BimanualArmArmAvoidance`` normalisation. Does NOT
edit any existing file.

Honest labelling of what this net is
------------------------------------
Checkpoint ``runs/bimanual_arm_arm_avoidance_bootstrap_v0`` was trained with
``training_objective: supervised_bootstrap`` on ``datasets/bimanual_yor_v2``:
i.i.d. sampled joint states (50% task-region, 50% uniform) labelled
+1 (unsafe, l(x)<=0) / -1 (safe, l(x)>0) from the INSTANTANEOUS faithful
keypoint clearance l(x). The labels do not depend on tau and contain NO
reachability / time-to-collision information -> this is a smooth learned
*classifier of the instantaneous constraint set*, NOT a converged HJ BRT
value. Any "anticipation" can only come from the net's spatial smoothing.

Conventions
-----------
- state order = [q_ego(6), q_int(6)] = [q_LEFT(6), q_RIGHT(6)]
  (mujoco_fk: EGO_PREFIX="left_arm_", INT_PREFIX="right_arm_").
- safe_when_leq_zero: V<=0 safe, V>0 unsafe.
- The training loss supervised tanh(model_out) against +-1
  (utils.modules.extract_signed_model_values, output_value_range=tanh);
  ``value_runtime`` returns the raw pre-tanh ``model_out`` (logit). Both are
  exposed here: ``V = tanh(logit)`` (the trained quantity) and ``logit``.

Name-clash note: ``hj_reachability/dynamics.py`` shadows the DeepReach
``dynamics`` package (and value_runtime's own import fails when
hj_reachability is on sys.path first). We temporarily swap sys.modules /
sys.path while importing the DeepReach modules, then restore them.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

RUN_DIR = Path("/common/users/ss5772/DeepReach/deepreach/runs/bimanual_arm_arm_avoidance_bootstrap_v0")
CKPT = "training/checkpoints/model_final.pth"
DEEPREACH_REPO = "/common/users/ss5772/DeepReach/deepreach"
HJ_DIR = Path(__file__).resolve().parents[1]

_CLASH = ("dynamics", "utils")


def _load_value_runtime_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_f2_value_runtime", str(HJ_DIR / "value_runtime.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_f2_value_runtime"] = mod
    spec.loader.exec_module(mod)
    return mod


class LearnedArmArmValue:
    """Trained value V(q_left, q_right) (tau fixed) with autograd gradient."""

    def __init__(self, run_dir=RUN_DIR, checkpoint=CKPT, tau_seconds: float = 1.0):
        import torch
        torch.set_num_threads(1)
        self.torch = torch
        vr = _load_value_runtime_module()
        run_config = vr._load_run_config(Path(run_dir))
        ck = vr._resolve_checkpoint(Path(run_dir), checkpoint)

        saved_mods = {k: v for k, v in sys.modules.items()
                      if any(k == c or k.startswith(c + ".") for c in _CLASH)}
        for k in saved_mods:
            del sys.modules[k]
        saved_path = list(sys.path)
        # DeepReach's `dynamics`/`utils` are namespace packages (no __init__),
        # which LOSE to any regular dynamics.py/utils.py anywhere on sys.path,
        # so drop clashing entries while importing.
        def _clashes(p):
            pp = Path(p or ".")
            return any((pp / f"{c}.py").exists() or (pp / c / "__init__.py").exists() for c in _CLASH)
        sys.path[:] = [DEEPREACH_REPO] + [p for p in saved_path if not _clashes(p)]
        try:
            model, dyn = vr.build_model_and_dynamics(
                run_config=run_config, data_root=Path(run_config["data_root"]),
                checkpoint_path=ck, device="cpu", deepreach_repo=Path(DEEPREACH_REPO))
        finally:
            for k in [k for k in sys.modules
                      if any(k == c or k.startswith(c + ".") for c in _CLASH)]:
                del sys.modules[k]
            sys.modules.update(saved_mods)
            sys.path[:] = saved_path
        for p in model.parameters():
            p.requires_grad_(False)
        self.model, self.dyn = model, dyn
        self.tau = float(tau_seconds)
        self.checkpoint = str(ck)
        self._mean = dyn.state_mean.double().clone()
        self._var = dyn.state_var.double().clone()

    def _forward_logit(self, states, need_grad: bool):
        torch = self.torch
        x = torch.as_tensor(np.asarray(states, dtype=np.float64).reshape(-1, 12))
        x = x.clone().requires_grad_(need_grad)
        tau = torch.full((x.shape[0], 1), self.tau, dtype=torch.float64)
        inp = torch.cat([tau, (x - self._mean) / self._var], dim=1).float()
        out = self.model({"coords": inp})["model_out"].reshape(-1)
        return x, out

    def values(self, states):
        """(logit, V=tanh(logit)) for states [N,12]=[q_left, q_right]."""
        with self.torch.no_grad():
            _, out = self._forward_logit(states, need_grad=False)
        lg = out.double().numpy()
        return lg, np.tanh(lg)

    def value_and_grad(self, q_left, q_right):
        """Returns (V, logit, dV/dq[12], dlogit/dq[12]) at one state.
        dV/dq = (1 - V^2) dlogit/dq."""
        s = np.concatenate([np.asarray(q_left, float), np.asarray(q_right, float)])
        x, out = self._forward_logit(s, need_grad=True)
        (g,) = self.torch.autograd.grad(out.sum(), x)
        dlg = g.detach().numpy().reshape(-1).astype(float)
        lg = float(out.detach().double().numpy()[0])
        V = float(np.tanh(lg))
        return V, lg, (1.0 - V * V) * dlg, dlg

    def batch_value_and_grad(self, states):
        x, out = self._forward_logit(states, need_grad=True)
        (g,) = self.torch.autograd.grad(out.sum(), x)
        lg = out.detach().double().numpy()
        V = np.tanh(lg)
        dlg = g.detach().numpy()
        return V, lg, (1.0 - V ** 2)[:, None] * dlg, dlg
