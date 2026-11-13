"""world_model/wm_semantic_monitor.py

REAL trained-WM failure monitor -- drop-in replacement for
`eval/safety_module.py`'s `OracleSpillMonitor` placeholder. Implements the
same `SemanticSafety` Protocol (`reset()`/`assess(obs) -> SemanticVerdict`)
structurally (duck-typed; eval/ is only ever imported here, never edited).

Pipeline, matching docs/design/state_spaces.md's S3 restriction (image +
proprio(26D) + action ONLY, never object pose / ground-truth spill flags):

  1. Render the ZED workspace camera (`zed_cam`) from `obs.model`/`obs.data`
     via `mujoco.Renderer` -- this is what a real robot's camera would give,
     NOT a privileged read. Tiled into `image_shape[-1] // 3` copies to match
     the trained WM's channel count (world_model/README.md / collection/
     cameras.py's YorCupCameraRenderer duplicates the same zed_cam frame into
     both `rgb_head`/`rgb_wrist_right` for the cup_yor_full dataset -- i.e.
     the WM's "6-channel image" IS the ZED view twice, not two different
     cameras -- so tiling the live render the same way reproduces exactly
     what the WM saw during training).
  2. Read 26-D proprio (left_q, left_qd, right_q, right_qd,
     left_gripper_state, right_gripper_state) straight off `data.qpos`/
     `data.qvel`, replicating collection/collect_cup.py's own per-step
     recording code byte-for-byte (same joint order, same gripper-state
     formula `qpos(joint7)/GRIPPER_JOINT_RANGE_M` clipped to [0,1]).
  3. Read the 16-D action as the position-actuator ctrl values collect_cup.py
     itself recorded (`data.ctrl[left_pos_act+left_finger_act]` /
     `data.ctrl[right_pos_act+right_finger_act]`) -- KNOWN LIMITATION: in
     this eval harness the LEFT arm's position servos are disabled
     (`sc.disable_left_arm_position_servos`, scene_setup.py) and driven by
     direct joint torque instead, so the left half of this action vector is
     a constant readout (whatever ctrl was left at after grasp-init), not a
     causally-meaningful command, unlike at data-collection time. This is
     the same array slice/convention, just no longer causally load-bearing
     for the left arm -- see this module's README section for the honest
     writeup.
  4. Step the FROZEN WorldModel's RSSM online, one step at a time
     (`dynamics.obs_step`, not the batched `observe` used in training/eval),
     maintaining recurrent state (`self._state`) and the previous action
     across calls -- `reset()` clears both at episode start.
  5. Run the frozen MonitorHead MLP on `dynamics.get_feat(post)` ->
     `P(spill)`, `P(collision)`. `risk = P(spill)`; `veto = risk >=
     risk_threshold`.

Both the WorldModel and MonitorHead are loaded frozen (`eval()`, all calls
under `@torch.no_grad()`) -- this file only ever forwards through them, never
trains.

`LatentValueShield` (below, subclasses `RealWMSpillMonitor`) is the "learned
HJ in the loop" step: it reuses steps 1-4 above UNCHANGED (factored into
`_step_wm`) but replaces step 5's instantaneous `P(spill)` classifier veto
with a proper reachability VALUE `V_theta(z)`
(`world_model/train_latent_safety_value.py`'s `LatentSafetyValue`, trained
via the Fisac et al. discounted-safety-Bellman recursion) -- `veto = V(z) <
v_thresh` instead of `P(spill) >= risk_threshold`. See its own class
docstring for the sign convention and the anticipatory `v_thresh` margin.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import mujoco
import numpy as np
import torch
import yaml

DREAMERV3_ROOT = "../dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)
import models  # noqa: E402  (vendored dreamerv3-torch)

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
REPO_ROOT = WORLD_MODEL_ROOT.parent
_EVAL_DIR = REPO_ROOT / "eval"
_SIM_COMMON_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
for _p in (str(REPO_ROOT), str(_SIM_COMMON_DIR), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from world_model.train_wm import build_spaces  # noqa: E402
from world_model.monitor_head import MonitorHead  # noqa: E402

import sim_common as sc  # noqa: E402  (read-only import; do not edit)
import safety_module as sm  # noqa: E402  (eval/, read-only import; do not edit)

ZED_CAM = "zed_cam"
GRIPPER_JOINT_RANGE_M = 0.035  # matches collect_cup.py's own constant / gripper_convention metadata

# Same actuator-name / joint order collect_cup.py's per-step recording uses.
LEFT_POS_ACT_NAMES = [f"left_arm_joint{i}_pos" for i in range(1, 7)] + [
    "left_arm_joint7_pos", "left_arm_joint8_pos",
]
RIGHT_POS_ACT_NAMES = [f"right_arm_joint{i}_pos" for i in range(1, 7)] + [
    "right_arm_joint7_pos", "right_arm_joint8_pos",
]


def _gripper_state(model, data, prefix: str) -> float:
    jid = model.joint(f"{prefix}_joint7").id
    qadr = model.jnt_qposadr[jid]
    return float(np.clip(data.qpos[qadr] / GRIPPER_JOINT_RANGE_M, 0.0, 1.0))


def _read_proprio(model, data) -> np.ndarray:
    """26-D proprio in dataset_adapter.py's PROPRIO_FIELDS order: left_q(6),
    left_qd(6), right_q(6), right_qd(6), left_gripper_state(1),
    right_gripper_state(1)."""
    _, left_qposadr, left_dofadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, right_qposadr, right_dofadr = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    parts = [
        data.qpos[left_qposadr],
        data.qvel[left_dofadr],
        data.qpos[right_qposadr],
        data.qvel[right_dofadr],
        np.array([_gripper_state(model, data, "left_arm")]),
        np.array([_gripper_state(model, data, "right_arm")]),
    ]
    return np.concatenate(parts).astype(np.float32)


def _read_action(model, data) -> np.ndarray:
    """16-D action = data.ctrl at the same actuator slice collect_cup.py
    itself recorded (left_action=8, right_action=8). See module docstring's
    KNOWN LIMITATION: the left half is not causally meaningful in this
    torque-driven eval harness (position servos disabled), it is still the
    same readout convention."""
    left_ids = [model.actuator(n).id for n in LEFT_POS_ACT_NAMES]
    right_ids = [model.actuator(n).id for n in RIGHT_POS_ACT_NAMES]
    return np.concatenate([data.ctrl[left_ids], data.ctrl[right_ids]]).astype(np.float32)


class RealWMSpillMonitor:
    """REAL system for the `SemanticSafety` role -- structurally satisfies
    eval/safety_module.py's `SemanticSafety` Protocol (`reset()`/`assess()`),
    swapping in a frozen trained WorldModel + MonitorHead for the
    OracleSpillMonitor placeholder. Never reads obs.model/obs.data for
    anything except rendering the ZED camera and proprio/action (all
    physically obtainable on the real robot) -- no ground-truth spill/
    collision/object-pose read, unlike the oracle it replaces."""

    def __init__(
        self,
        wm_checkpoint: str,
        head_checkpoint: str,
        config_path: str | None = None,
        device: str | None = None,
        risk_threshold: float = 0.5,
    ):
        config_path = config_path or str(WORLD_MODEL_ROOT / "train_wm_config.yaml")
        with open(config_path) as f:
            cfg = dict(yaml.safe_load(f))
        if device:
            cfg["device"] = device
        if not torch.cuda.is_available():
            cfg["device"] = "cpu"
        self.device = cfg["device"]
        config = types.SimpleNamespace(**cfg)

        head_ckpt = torch.load(head_checkpoint, map_location=self.device)
        image_shape = tuple(head_ckpt["image_shape"])
        proprio_dim = int(head_ckpt["proprio_dim"])
        num_actions = int(head_ckpt["num_actions"])
        obs_space, act_space = build_spaces(image_shape, proprio_dim, num_actions)

        self.wm = models.WorldModel(obs_space, act_space, step=0, config=config).to(self.device)
        wm_ckpt = torch.load(wm_checkpoint, map_location=self.device)
        self.wm.load_state_dict(wm_ckpt["wm_state_dict"])
        self.wm.eval()

        self.head = MonitorHead(head_ckpt["latent_dim"], hidden=head_ckpt["hidden"]).to(self.device)
        self.head.load_state_dict(head_ckpt["head_state_dict"])
        self.head.eval()

        self.image_shape = image_shape
        self.proprio_dim = proprio_dim
        self.num_actions = num_actions
        self.risk_threshold = risk_threshold
        self.horizon_s = 0.15  # read by run_eval.py's semantic-precision/recall horizon (parity w/ oracle)

        self._renderer = None
        self._renderer_model_id = None
        self._state = None
        self._prev_action = None
        self._t_idx = 0

    def reset(self) -> None:
        self._state = None
        self._prev_action = torch.zeros((1, self.num_actions), dtype=torch.float32, device=self.device)
        self._t_idx = 0
        self._renderer = None  # a fresh model may be loaded per episode; rebuild lazily
        self._renderer_model_id = None

    @torch.no_grad()
    def _step_wm(self, obs):
        """Shared pipeline, factored out of `assess()` below so a subclass
        (e.g. `LatentValueShield`) can reuse the EXACT WM-load + per-vision-
        step ZED render + proprio/action read + `obs_step` machinery
        byte-for-byte, without reimplementing it -- this refactor does not
        change `assess()`'s own behavior at all (it is the same code, just
        split so the latent feature + monitor-head probs are also usable by
        a caller that wants a different final scoring rule).

        Returns (feat, p_spill, p_collision): `feat` is
        `wm.dynamics.get_feat(post)` (the latent `z` a value net like
        `LatentSafetyValue` consumes), `p_spill`/`p_collision` are this
        frozen MonitorHead's classifier outputs on that same feat (kept
        around so a value-based shield can log them side-by-side for an
        anticipation comparison, per this task's smoke ask)."""
        model, data = obs.model, obs.data
        if self._renderer is None or self._renderer_model_id != id(model):
            h, w, c = self.image_shape
            self._renderer = mujoco.Renderer(model, height=h, width=w)
            self._renderer_model_id = id(model)

        self._renderer.update_scene(data, camera=ZED_CAM)
        zed = self._renderer.render().copy()  # (H, W, 3) uint8, real (renderable-on-robot) observation
        n_tiles = max(1, self.image_shape[-1] // 3)
        image = np.concatenate([zed] * n_tiles, axis=-1).astype(np.uint8)[..., : self.image_shape[-1]]

        proprio = _read_proprio(model, data)
        is_first_np = np.array([self._t_idx == 0], dtype=bool)
        obs_dict = {
            "image": image[None],
            "proprio": proprio[None],
            "is_first": is_first_np,
            "is_terminal": np.array([False]),
        }
        data_t = self.wm.preprocess(obs_dict)
        embed = self.wm.encoder(data_t)
        is_first_t = torch.as_tensor(is_first_np, device=self.device)
        post, _ = self.wm.dynamics.obs_step(
            self._state, self._prev_action, embed, is_first_t, sample=False
        )
        self._state = post
        feat = self.wm.dynamics.get_feat(post)
        probs = torch.sigmoid(self.head(feat))[0].detach().cpu().numpy()
        p_spill, p_collision = float(probs[0]), float(probs[1])

        action = _read_action(model, data)
        self._prev_action = torch.as_tensor(action[None], dtype=torch.float32, device=self.device)
        self._t_idx += 1
        return feat, p_spill, p_collision

    @torch.no_grad()
    def assess(self, obs) -> "sm.SemanticVerdict":
        feat, p_spill, p_collision = self._step_wm(obs)
        veto = p_spill >= self.risk_threshold
        return sm.SemanticVerdict(risk=p_spill, veto=veto, spilled_now=False)


class LatentValueShield(RealWMSpillMonitor):
    """REAL "learned HJ in the loop" SHIELD -- the step this task adds.

    Replaces `RealWMSpillMonitor`'s instantaneous `P(spill)` classifier
    veto with a proper reachability VALUE `V_theta(z)`
    (`world_model/train_latent_safety_value.py`'s `LatentSafetyValue`,
    trained via the Fisac et al. discounted-safety-Bellman recursion on the
    same frozen WM's latent `z = wm.dynamics.get_feat(post)`). Sign
    convention (unchanged from training): V(z) < 0 <=> unsafe (inside the
    discounted reachable-failure set), V(z) > 0 <=> safe.

    REUSE, not reinvention: subclasses `RealWMSpillMonitor` so its exact
    WM-load + per-vision-step ZED render + proprio/action read + obs_step
    pipeline (`_step_wm`, factored out above) is shared byte-for-byte --
    this class only adds the value net and swaps the final scoring/veto
    rule. `p_spill` (classifier) is still computed every call (the
    MonitorHead is loaded as part of the parent's `__init__`) and kept on
    `self.last_p_spill` purely so a caller can log V-vs-P(spill) side by
    side for an anticipation comparison (see this task's smoke evidence);
    it plays NO role in this class's own veto decision.

    `v_thresh` (default 0.1, a small POSITIVE number) is deliberately set
    ABOVE the V=0 safe/unsafe boundary so the veto fires a bit BEFORE V
    actually crosses zero -- anticipatory margin, exposed as a parameter so
    the demo/eval caller can tune how early the shield acts.
    """

    def __init__(
        self,
        wm_checkpoint: str,
        head_checkpoint: str,
        value_checkpoint: str,
        config_path: str | None = None,
        device: str | None = None,
        v_thresh: float = 0.1,
        risk_k: float = 4.0,
    ):
        super().__init__(
            wm_checkpoint=wm_checkpoint,
            head_checkpoint=head_checkpoint,
            config_path=config_path,
            device=device,
        )
        from world_model.train_latent_safety_value import LatentSafetyValue  # noqa: E402  (local import; avoids a hard torch-training-file dep at module import time)

        value_ckpt = torch.load(value_checkpoint, map_location=self.device)
        latent_dim = int(value_ckpt["latent_dim"])
        hidden = int(value_ckpt.get("hidden", 64))
        self.value_net = LatentSafetyValue(latent_dim, hidden=hidden).to(self.device)
        self.value_net.load_state_dict(value_ckpt["value_state_dict"])
        self.value_net.eval()

        self.v_thresh = v_thresh
        self.risk_k = risk_k
        # side-by-side logging only (see class docstring) -- NOT used by assess()'s veto.
        self.last_v = None
        self.last_p_spill = None

    @torch.no_grad()
    def assess(self, obs) -> "sm.SemanticVerdict":
        feat, p_spill, p_collision = self._step_wm(obs)
        v = float(self.value_net(feat).reshape(()).item())
        # risk in [0,1]: sigmoid(-k*V) -- ~0.5 at V=0, ->1 as V becomes very
        # negative (unsafe), ->0 as V becomes very positive (safe); k=4.0
        # (default) makes it saturate over roughly V in [-1,1], the value
        # net's own tanh-bounded range.
        risk = float(1.0 / (1.0 + np.exp(self.risk_k * v)))
        veto = v < self.v_thresh
        self.last_v = v
        self.last_p_spill = p_spill
        return sm.SemanticVerdict(risk=risk, veto=veto, spilled_now=False)
