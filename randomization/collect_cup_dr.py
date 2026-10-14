"""randomization/collect_cup_dr.py -- OOD generalization test-set collector.

Wraps collection/collect_cup.py's cup-task episode loop with
randomization/domain_randomization.py::randomize() so the resulting episodes
are domain-randomized (visual + dynamics + scene's table pose) relative to
the ZERO-randomization data (`collection/data/cup_yor_full`) the WM/monitor
were trained on. Written entirely under `randomization/` per this task's
constraint (does not edit collection/, scripts/, or scenes/ on disk).

## How it hooks in

`collect_cup.collect_episode()` builds its model via
`collection.cameras.build_yor_cup_model()` (module-attribute call, so
monkeypatching `cameras.build_yor_cup_model` at import time redirects that
call site without editing collect_cup.py). This script's
`_dr_build_yor_cup_model` wraps the original builder and calls
`randomize(model, rng, config)` -- the MjModel (numeric) branch -- on the
freshly compiled model before returning it, seeded independently per episode
(`seed + 1_000_000`, decoupled from the behavior/perturbation seed, per
randomization/README.md's two-independent-seeds recommendation).

## Known scope limitation (documented, not fixed here)

`randomize()`'s `scene` family writes cup/ball xy jitter and ball
"parking" into `model.qpos0` (see domain_randomization.py's own "qpos0 only
takes effect on reset" docstring note). But `collect_cup.py`'s
`_init_settled_state()` resets via `mujoco.mj_resetDataKeyframe(model, data,
key)` (the scene's "home" keyframe, i.e. `model.key_qpos`), NOT
`mj_resetData` (which would pick up `qpos0`). So the scene family's
cup/ball-position/count randomization does NOT propagate through this
collection path -- only `table` body_pos jitter (a direct, reset-independent
model array write) from the scene family takes effect. Fixing this cleanly
needs a one-line edit in collect_cup.py's `_init_settled_state` (copy
`model.qpos0` into the keyframe-reset qpos for the cup/ball joints, or reset
via `mj_resetData` instead of the keyframe) -- out of scope here since this
task may only write under world_model/, randomization/, artifacts/.
**Visual and dynamics families are unaffected by this limitation** (both
operate on direct compiled-model arrays -- lights/materials/geom
rgba/cameras, and friction/mass/damping/actuator gain -- with no qpos
dependency), so this test set still exercises real visual-appearance and
physics-parameter OOD shift, just not cup/ball spatial layout shift.

Usage (mirrors collect_cup.py's own invocation convention):
    MUJOCO_GL=egl /common/users/ss5772/conda_envs/safe_env/bin/python \\
        -m randomization.collect_cup_dr \\
        --num-episodes 40 --seed-start 5000 --out-dir collection/data/cup_dr_test
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from collection import cameras  # noqa: E402
from collection import collect_cup  # noqa: E402
from randomization.domain_randomization import RandomizationConfig, randomize  # noqa: E402

# Default scaffold ranges (randomization/README.md) -- ALL nonzero vs the
# zero-DR training data (collection/data/cup_yor_full used no randomization
# at all), so any nonzero draw here is already OOD relative to training.
DR_CONFIG = RandomizationConfig()

_dr_seed_holder = {"seed": 0}
_orig_build_yor_cup_model = cameras.build_yor_cup_model


def _dr_build_yor_cup_model(scene_path: str = cameras.SCENE_YOR_CUP):
    model = _orig_build_yor_cup_model(scene_path)
    rng = random.Random(_dr_seed_holder["seed"])
    randomize(model, rng, DR_CONFIG)  # numeric branch: visual + dynamics + scene(table pos only, see module docstring)
    return model


cameras.build_yor_cup_model = _dr_build_yor_cup_model  # module-attribute patch; collect_cup.py calls cameras.build_yor_cup_model()


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "collection" / "data" / "cup_dr_test")
    p.add_argument("--num-episodes", type=int, default=40)
    p.add_argument("--seed-start", type=int, default=5000)
    p.add_argument("--render-size", type=int, default=64)
    p.add_argument("--render-every", type=int, default=1)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_spilled = n_collision = n_clean = 0
    for i in range(args.num_episodes):
        seed = args.seed_start + i
        _dr_seed_holder["seed"] = seed + 1_000_000  # independent DR-rng stream, decoupled from the behavior seed
        meta = collect_cup.collect_episode(
            episode_index=i,
            seed=seed,
            out_dir=args.out_dir,
            render_size=args.render_size,
            render_every=args.render_every,
        )
        spilled = meta["spilled_count"] > 0
        collision = meta["collision"]
        n_spilled += int(spilled)
        n_collision += int(collision)
        n_clean += int(not spilled and not collision)
        print(
            f"{meta['episode_id']} seed={seed} level={meta['perturbation_level']} "
            f"success={meta['success']} spilled={meta['spilled_count']} "
            f"collision={meta['collision']} steps={meta['num_steps']}"
        )

    dt = time.time() - t0
    print(
        f"wrote {args.num_episodes} DR episodes in {dt:.1f}s -- "
        f"spilled={n_spilled} collision={n_collision} clean={n_clean}"
    )


if __name__ == "__main__":
    main()
