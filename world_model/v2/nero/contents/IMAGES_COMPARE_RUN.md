# Proposer with images: Sonnet vs Codex, same worlds (recipe; owner runs this; 11-12)

Opt-in flag, default OFF (`--plan-images none`): a run without it is bit-identical to before (request text and cache keys unchanged, tested).
Images (`nero/contents/plan_images.py`): rendered from the initial state of world `--plan-image-world` (default 0) BEFORE any probe, 640x360 PNG, saved to `<plan-cache>/images/<sha16>.png`.
- `zed`: torso ZED left imager. Mount = the PROPOSED Lift_Top mount of `scripts/zed_camera.py` (not hardware-documented; confirm on hardware), fovy 70 (ASSUMED, ZED nominal), pitch 28 deg (ASSUMED, framed by eye: the 44 deg of the old scene cuts the object off). Left image only; no depth, no noise, no lens model.
- `zed+wrist`: adds `left_arm_wrist_rgb` (RealSense RGB, mount ASSUMED). At the reset pose it sees only the fingers (black frame): not informative, use `zed`.
The plan is proposed once per task, so one world's views stand for the task (the world index is in CONFIG.json and `plans/<task>.json: plan_images`). The prompt gets one line after the task text:
`IMAGES: image 1 is the torso stereo camera (ZED, left imager) at the start of the task, before any probe.` Rule 2 / rejection re-asks attach the same images and the same line.
Cache key = sha256(text, system, model, max tokens, image FILE HASHES): an image request never hits a text-only entry; another render = another key.
Same worlds as run_n20_v2b: tasks box_behind,tray6_behind, n 20, the same Theta list (bounds rule 8). Env for every command (worktree root):

    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=. MUJOCO_GL=osmesa PYTHONUNBUFFERED=1
    PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python
    D=/common/users/ss5772/bimanual_data/contents/table_behind_same_2026-11-09
    AC="--theirs agentchord --agentchord-plan tray6_behind=$D/agentchord_plans/tray6_behind,box_behind=$D/agentchord_plans/box_behind"

## Sonnet arm (Claude subscription; live, cache in `rw`; the re-ask loops it triggers are new requests)
    $PY -m world_model.v2.nero.contents.same_worlds run --tasks box_behind,tray6_behind --n 20 --procs 10 --plan-source codex \
        --plan-spec claude:claude-sonnet-5-5:high --plan-cache $D/sonnet_plans_img --plan-images zed --plan-image-world 0 $AC --out $D/run_n20_sonnet_img
    # then check the replay: same command + --offline --out $D/run_n20_sonnet_img_replay

## Codex arm (needs Codex quota: the owner decides when; 24 worlds ended in quota errors in run_n20_v2b)
    $PY -m world_model.v2.nero.contents.same_worlds run --tasks box_behind,tray6_behind --n 20 --procs 10 --plan-source codex \
        --plan-spec codex:gpt-5.6-sol:high --plan-cache $D/codex_plans_img --plan-images zed --plan-image-world 0 $AC --out $D/run_n20_codex_img

(`--plan-source codex` is the flag name for "the proposer model writes the plan", for either backend.) Both arms get the identical text + the identical image files (the ZED render is
deterministic: the same world gives the same file name; compare `plan_images.paths` in the two CONFIG/plans files). Text-only references for the same worlds: `run_n20_v2b` (Codex) and
`run_n20_sonnet` (Sonnet), caches `ours_plans_v2` / `sonnet_plans_v2` (read only).
Caveats to record (bounds rule 9): model, effort, no temperature/seed; clean sim render (no sensor noise); proposed mount + ASSUMED pitch/fov; one world's view per task; the image shows
what a camera sees (lid closed, fill not visible), no probe data.
