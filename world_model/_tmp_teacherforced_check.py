import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import sys, types, yaml, torch, numpy as np
from pathlib import Path

DREAMERV3_ROOT = "/common/users/ss5772/dreamerv3"
if DREAMERV3_ROOT not in sys.path:
    sys.path.insert(0, DREAMERV3_ROOT)
import tools as dreamer_tools  # noqa: E402

WORLD_MODEL_ROOT = Path(__file__).resolve().parent
if str(WORLD_MODEL_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(WORLD_MODEL_ROOT.parent))
from world_model.eval_wm import build_wm_from_checkpoint, _channels_to_panels  # noqa: E402

with open("world_model/train_wm_config.yaml") as f:
    cfg = dict(yaml.safe_load(f))
config = types.SimpleNamespace(**cfg)
episodes = dreamer_tools.load_episodes(Path("world_model/data/dreamer_fmt/cup_yor_full"))
key = [k for k in episodes if k.startswith("episode_000181")][0]
ep = episodes[key]
wm, image_shape, proprio_dim, num_actions = build_wm_from_checkpoint(
    config, episodes, "world_model/logdir/wm_cup_full/best.pt"
)

T = 160
batch = {k: torch.as_tensor(np.asarray(v[:T]), device=config.device)[None] for k, v in ep.items()}
data = wm.preprocess(batch)
with torch.no_grad():
    embed = wm.encoder(data)
    post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
    feat = wm.dynamics.get_feat(post)
    image_pred = wm.heads["decoder"](feat)["image"].mode()

truth = data["image"][0].cpu().numpy().clip(0, 1)
pred = image_pred[0].cpu().numpy().clip(0, 1)

r0, r1, c0, c1 = 14, 58, 2, 62
idxs = list(range(0, T, 16))
fig, axes = plt.subplots(3, len(idxs), figsize=(len(idxs) * 1.6, 4.8))
for j, i in enumerate(idxs):
    tp = _channels_to_panels(truth[i])[:, :64][r0:r1, c0:c1]
    pp = _channels_to_panels(pred[i])[:, :64][r0:r1, c0:c1]
    ep_ = np.abs(pp - tp)
    axes[0, j].imshow(tp)
    axes[0, j].axis("off")
    axes[0, j].set_title(f"t={i}", fontsize=7)
    axes[1, j].imshow(pp)
    axes[1, j].axis("off")
    axes[2, j].imshow(ep_)
    axes[2, j].axis("off")
axes[0, 0].set_ylabel("true")
axes[1, 0].set_ylabel("recon(teacher-forced)")
axes[2, 0].set_ylabel("|err|")
fig.suptitle("episode_000181 teacher-forced (not imagined) recon -- ball visibility check")
fig.tight_layout()
fig.savefig("artifacts/wm_teacherforced_ballcheck.png", dpi=130)
print("saved")
