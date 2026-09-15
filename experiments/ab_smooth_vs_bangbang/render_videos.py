"""experiments/ab_smooth_vs_bangbang/render_videos.py

Thin driver that shells out to scripts/make_cup_collect_safety_demo.py to
render the two FLAGSHIP comparison videos on the NOW-FINAL full-fidelity
model (self-collision + arm<->arm + arm<->table + right-arm<->cup +
arm<->cube):
  1. flmpc  (our solution)   --controller flmpc
  2. v4c    (reactive baseline) --controller v4c
both on the same hard geometry (--scenario crossing --seed 0
--cup-jitter-m 0.04 --n-cubes 3), --config geometric_only for BOTH (WM spill
monitor off) so the comparison isolates the controller axis only --
--controller flmpc forces the WM monitor off internally regardless of
--config anyway (see make_cup_collect_safety_demo.py's CHANGE 2 docstring),
so using geometric_only for v4c too keeps it apples-to-apples rather than
letting v4c get WM-monitor help flmpc structurally can't use.

The demo script writes a FIXED filename per (config, controller, scenario,
monitor) combo under artifacts/ and OVERWRITES it on every run (see its
`video_path = ART_DIR / f"cup_collect_safety_demo_{config}{ctrl_suffix}..."`
naming) -- so back-to-back renders of flmpc then v4c would clobber each
other under the default naming when suffixes collide. This driver runs each
render then COPIES the resulting artifact to a distinct stable name so both
coexist:
    artifacts/flagship_flmpc_crossing_seed0.mp4
    artifacts/flagship_v4c_crossing_seed0.mp4
(+ matching _meta.json copies)

Rendering needs an EGL headless GL context and a specific GPU (see repo
convention: GPU 5, others busy) and MUJOCO_GL=egl. NOTE: the task brief for
this harness said "framebuffer max 640x480" but the demo script as it
actually stands renders at 1280x720 (`mujoco.Renderer(model, height=720,
width=1280)` in make_cup_collect_safety_demo.py's run_episode) -- this
driver does not hardcode a resolution override (the demo script owns that),
flagging the discrepancy here rather than silently trusting either number.

THIS DRIVER ITSELF DOES NOT RENDER THE FULL VIDEOS (owner does, it's a
multi-minute-per-video capture+encode job). Run with --smoke to do ONE short
--max-steps-truncated flmpc render to confirm the command/flags/output-copy
plumbing works end-to-end; otherwise it just prints the exact commands.

Usage:
    # print the exact full-render commands (default, no execution):
    python experiments/ab_smooth_vs_bangbang/render_videos.py --dry-run

    # OWNER: actually render both flagship videos (full episode, ~5-10min each):
    python experiments/ab_smooth_vs_bangbang/render_videos.py --run

    # SMOKE (this task): one short truncated flmpc render to confirm plumbing:
    python experiments/ab_smooth_vs_bangbang/render_videos.py --smoke
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEMO_SCRIPT = REPO_ROOT / "scripts" / "make_cup_collect_safety_demo.py"
ART_DIR = REPO_ROOT / "artifacts"
PYTHON = "/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python"
GPU = "5"  # repo convention: GPU 5 only, 1-4 busy (see make_cup_collect_safety_demo.py docstring)

# (label, controller, config) -- config="geometric_only" for BOTH, see module docstring.
FLAGSHIP_RUNS = [
    ("flmpc", "flmpc", "geometric_only"),
    ("v4c", "v4c", "geometric_only"),
]

COMMON_FLAGS = [
    "--scenario", "crossing",
    "--seed", "0",
    "--cup-jitter-m", "0.04",
    "--n-cubes", "3",
]


def video_path_for(config: str, controller: str, scenario: str = "crossing") -> Path:
    ctrl_suffix = "" if controller == "v4c" else f"_{controller}"
    scenario_suffix = "" if scenario == "crossing" else f"_{scenario}"
    return ART_DIR / f"cup_collect_safety_demo_{config}{ctrl_suffix}{scenario_suffix}.mp4"


def meta_path_for(config: str, controller: str, scenario: str = "crossing") -> Path:
    ctrl_suffix = "" if controller == "v4c" else f"_{controller}"
    scenario_suffix = "" if scenario == "crossing" else f"_{scenario}"
    return ART_DIR / f"cup_collect_safety_demo_meta_{config}{ctrl_suffix}{scenario_suffix}.json"


def build_command(controller: str, config: str, max_steps: int | None = None) -> list[str]:
    cmd = [PYTHON, str(DEMO_SCRIPT),
           "--config", config, "--controller", controller] + COMMON_FLAGS
    if max_steps is not None:
        cmd += ["--max-steps", str(max_steps)]
    return cmd


def env_for_render() -> dict:
    env = os.environ.copy()
    env["MUJOCO_GL"] = "egl"
    env["CUDA_VISIBLE_DEVICES"] = GPU
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
        env[v] = "1"
    return env


def print_commands(max_steps: int | None = None) -> None:
    print("=== FLAGSHIP RENDER COMMANDS (owner runs these) ===")
    for label, controller, config in FLAGSHIP_RUNS:
        cmd = build_command(controller, config, max_steps=max_steps)
        print(f"\n# {label}:")
        print(f"MUJOCO_GL=egl CUDA_VISIBLE_DEVICES={GPU} \\")
        print("    " + " \\\n    ".join(cmd))
        print(f"  -> writes {video_path_for(config, controller)}")
        stable = ART_DIR / f"flagship_{label}_crossing_seed0.mp4"
        print(f"  -> this driver then copies to {stable}")


def run_one(label: str, controller: str, config: str, max_steps: int | None, dry_run: bool) -> tuple[bool, str]:
    cmd = build_command(controller, config, max_steps=max_steps)
    print(f"\n[render_videos] {label}: {' '.join(cmd)}", flush=True)
    if dry_run:
        return True, "(dry-run, not executed)"

    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env_for_render(),
                           capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"[render_videos] {label} FAILED (rc={proc.returncode}, wall={wall:.0f}s)")
        print("  stdout tail:", proc.stdout[-1500:])
        print("  stderr tail:", proc.stderr[-1500:])
        return False, f"rc={proc.returncode}"

    src_vid = video_path_for(config, controller)
    src_meta = meta_path_for(config, controller)
    dst_vid = ART_DIR / f"flagship_{label}_crossing_seed0.mp4"
    dst_meta = ART_DIR / f"flagship_{label}_crossing_seed0_meta.json"

    if not src_vid.exists():
        print(f"[render_videos] {label}: expected video not found at {src_vid} "
              f"(demo stdout tail below)")
        print(proc.stdout[-1500:])
        return False, "video not produced"

    shutil.copy2(src_vid, dst_vid)
    if src_meta.exists():
        shutil.copy2(src_meta, dst_meta)
    print(f"[render_videos] {label} OK, wall={wall:.0f}s -> copied {src_vid.name} -> {dst_vid.name}")
    return True, str(dst_vid)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="(default) just print the exact commands, execute nothing")
    g.add_argument("--run", action="store_true", help="OWNER: actually render both FULL flagship videos")
    g.add_argument("--smoke", action="store_true",
                   help="run ONE short --max-steps-truncated flmpc render to confirm the CLI/flags/copy plumbing works")
    ap.add_argument("--smoke-max-steps", type=int, default=300)
    args = ap.parse_args()

    ART_DIR.mkdir(parents=True, exist_ok=True)

    if args.run:
        print("=== FULL FLAGSHIP RENDER (owner) ===")
        results = []
        for label, controller, config in FLAGSHIP_RUNS:
            ok, info = run_one(label, controller, config, max_steps=None, dry_run=False)
            results.append((label, ok, info))
        print("\n=== SUMMARY ===")
        for label, ok, info in results:
            print(f"  {label}: {'OK' if ok else 'FAILED'} -> {info}")
        return

    if args.smoke:
        print(f"=== SMOKE RENDER (flmpc only, --max-steps {args.smoke_max_steps}) ===")
        label, controller, config = FLAGSHIP_RUNS[0]
        ok, info = run_one(label, controller, config, max_steps=args.smoke_max_steps, dry_run=False)
        print(f"\n[render_videos] smoke result: {'PASS' if ok else 'FAIL'} -> {info}")
        if not ok:
            sys.exit(1)
        return

    # default: dry-run, just print commands
    print_commands()
    print("\n(this was --dry-run; pass --run for the full videos [owner] or --smoke for a short confirm render)")


if __name__ == "__main__":
    main()
