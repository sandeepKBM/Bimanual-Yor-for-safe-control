"""Generate episode-shaped train/val/test CSVs (+ feature_schema.json) for
FLOQ's own dataset builder (``ur5e_benchmark.floq.dataset.build_floq_dataset_cache``)
from our bimanual kinematic single-integrator game.

FLOQ (see /common/users/ss5772/VLAism/ur5e_benchmark/src/ur5e_benchmark/floq/)
expects, per source root: a ``feature_schema.json`` with a
``state_feature_order`` list, and ``train.csv``/``val.csv``/``test.csv`` with
an episode-key column, a per-step timestep column, and (for our labeling
path) a ``failure_step`` column -- it derives per-(step, horizon) safe/unsafe
labels itself (``dataset.py::_build_split_arrays``): unsafe iff the episode's
failure step falls within `horizon` steps of `step`, safe iff the episode
survives past `step + horizon` with no failure, else censored (dropped).

Our "failure" = first timestep the FAITHFUL MuJoCo-FK boundary
(``mujoco_fk.MuJoCoFKOracle.boundary_fn``, imported from ``dynamics.py``'s
``BimanualPairAvoidance.boundary_fn``) goes <= 0, i.e. the avoid set is
entered -- exactly the BRT "first hitting time" this problem cares about.

Rollout policy: NOT the adversarial-optimal HJ policy (no trained value
function exists yet to define one -- that's what FLOQ is being trained to
approximate). Both ego (state = q_ego, control u) and intruder (state =
q_int, disturbance d) take a bounded random walk, ||u|| <= u_bar_ego,
||d|| <= u_bar_int, matching BimanualPairAvoidance's ``ẋ = u`` / ``ẋ = d``
kinematic game and control bounds. A ``--task-region-frac`` fraction of
EPISODES start near the shared workspace point via
``BimanualPairAvoidance.sample_states_mixed`` (mirrors
``hj_reachability/export_dataset.py``'s static-dataset oversampling) so a
meaningful fraction of rollouts actually enter the avoid set instead of
wandering the full 12-D joint box forever.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

import paths  # noqa: F401  (sets up sys.path for `dynamics`/`mujoco_fk`)
import dynamics as bimanual_dynamics  # noqa: E402


CSV_FIELDS = (
    ["episode_uid", "timestep_index", "failure_step", "episode_end_step", "task"]
    + [f"q_ego_{i}" for i in range(bimanual_dynamics.N_JOINTS_PER_ARM)]
    + [f"q_int_{i}" for i in range(bimanual_dynamics.N_JOINTS_PER_ARM)]
)


def _random_ball_vector(rng: np.random.Generator, dim: int, radius: float) -> np.ndarray:
    """Roughly-uniform random vector inside the L2 ball of the given radius
    (direction uniform on the sphere via a standard-normal draw, magnitude
    uniform in [0, radius] -- exact uniform-in-ball density is not needed
    here, this is a rollout exploration policy, not a sampling proof)."""
    v = rng.normal(size=dim)
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.zeros(dim)
    v = v / n
    mag = radius * float(rng.uniform(0.0, 1.0))
    return v * mag


def rollout_episode(
    spec: "bimanual_dynamics.BimanualPairAvoidance",
    rng: np.random.Generator,
    *,
    max_steps: int,
    dt: float,
    u_bar_ego: float,
    u_bar_int: float,
    task_region_frac: float,
) -> tuple[list[dict], int | None]:
    lo = bimanual_dynamics.JOINT_RANGES[:, 0]
    hi = bimanual_dynamics.JOINT_RANGES[:, 1]

    state0 = spec.sample_states_mixed(1, rng, task_region_frac=task_region_frac)[0]
    q_ego = state0[: bimanual_dynamics.N_JOINTS_PER_ARM].copy()
    q_int = state0[bimanual_dynamics.N_JOINTS_PER_ARM :].copy()

    rows: list[dict] = []
    failure_step: int | None = None
    for t in range(int(max_steps)):
        state = np.concatenate([q_ego, q_int])
        h = spec.boundary_fn(state)
        rows.append({"t": t, "q_ego": q_ego.copy(), "q_int": q_int.copy()})
        if h <= 0.0:
            failure_step = t
            break
        u = _random_ball_vector(rng, bimanual_dynamics.N_JOINTS_PER_ARM, u_bar_ego)
        d = _random_ball_vector(rng, bimanual_dynamics.N_JOINTS_PER_ARM, u_bar_int)
        q_ego = np.clip(q_ego + dt * u, lo, hi)
        q_int = np.clip(q_int + dt * d, lo, hi)
    return rows, failure_step


def _write_split_csv(path: str, episodes: list[tuple[str, list[dict], int | None]]) -> int:
    n_rows = 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for episode_uid, rows, failure_step in episodes:
            episode_end_step = rows[-1]["t"]
            for row in rows:
                out = {
                    "episode_uid": episode_uid,
                    "timestep_index": row["t"],
                    "failure_step": "" if failure_step is None else int(failure_step),
                    "episode_end_step": episode_end_step,
                    "task": "bimanual",
                }
                for i in range(bimanual_dynamics.N_JOINTS_PER_ARM):
                    out[f"q_ego_{i}"] = float(row["q_ego"][i])
                    out[f"q_int_{i}"] = float(row["q_int"][i])
                writer.writerow(out)
                n_rows += 1
    return n_rows


def build(
    *,
    output_root: str,
    num_episodes: int,
    max_steps: int,
    dt: float,
    u_bar_ego: float,
    u_bar_int: float,
    task_region_frac: float,
    seed: int,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> dict:
    os.makedirs(output_root, exist_ok=True)
    rng = np.random.default_rng(seed)
    spec = bimanual_dynamics.BimanualPairAvoidance(
        bimanual_dynamics.BimanualPairAvoidanceConfig(u_bar_ego=u_bar_ego, u_bar_int=u_bar_int)
    )

    t0 = time.time()
    episodes: list[tuple[str, list[dict], int | None]] = []
    n_failed = 0
    for ep_idx in range(int(num_episodes)):
        rows, failure_step = rollout_episode(
            spec,
            rng,
            max_steps=max_steps,
            dt=dt,
            u_bar_ego=u_bar_ego,
            u_bar_int=u_bar_int,
            task_region_frac=task_region_frac,
        )
        if failure_step is not None:
            n_failed += 1
        episodes.append((f"bimanual_ep_{ep_idx:06d}", rows, failure_step))
    wall = time.time() - t0

    n_train = int(round(num_episodes * train_frac))
    n_val = int(round(num_episodes * val_frac))
    splits = {
        "train": episodes[:n_train],
        "val": episodes[n_train : n_train + n_val],
        "test": episodes[n_train + n_val :],
    }

    split_rows = {}
    for split, split_episodes in splits.items():
        path = os.path.join(output_root, f"{split}.csv")
        split_rows[split] = _write_split_csv(path, split_episodes)

    schema = {
        "state_feature_order": (
            [f"q_ego_{i}" for i in range(bimanual_dynamics.N_JOINTS_PER_ARM)]
            + [f"q_int_{i}" for i in range(bimanual_dynamics.N_JOINTS_PER_ARM)]
        )
    }
    with open(os.path.join(output_root, "feature_schema.json"), "w") as f:
        json.dump(schema, f, indent=2)

    manifest = {
        "output_root": output_root,
        "num_episodes": int(num_episodes),
        "num_episodes_with_failure": int(n_failed),
        "failure_fraction": float(n_failed) / float(num_episodes) if num_episodes else 0.0,
        "max_steps": int(max_steps),
        "dt": float(dt),
        "u_bar_ego": float(u_bar_ego),
        "u_bar_int": float(u_bar_int),
        "task_region_frac": float(task_region_frac),
        "seed": int(seed),
        "split_episode_counts": {k: len(v) for k, v in splits.items()},
        "split_row_counts": split_rows,
        "wall_seconds": wall,
    }
    with open(os.path.join(output_root, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-root", required=True)
    p.add_argument("--num-episodes", type=int, default=30)
    p.add_argument("--max-steps", type=int, default=40)
    p.add_argument("--dt", type=float, default=0.1)
    p.add_argument("--u-bar-ego", type=float, default=bimanual_dynamics.DEFAULT_U_BAR_EGO)
    p.add_argument("--u-bar-int", type=float, default=bimanual_dynamics.DEFAULT_U_BAR_INT)
    p.add_argument("--task-region-frac", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()

    manifest = build(
        output_root=args.output_root,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        dt=args.dt,
        u_bar_ego=args.u_bar_ego,
        u_bar_int=args.u_bar_int,
        task_region_frac=args.task_region_frac,
        seed=args.seed,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
