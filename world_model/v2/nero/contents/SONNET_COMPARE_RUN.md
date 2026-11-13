# Codex vs Sonnet proposer comparison: run recipe (owner runs this; 11-12)

Backend: `claude:claude-sonnet-5-5:high` (pan/llm.py ClaudeCodeVLM, local `claude -p`, no tools, empty temp cwd, our system prompt replaces Claude Code's).
Codex side = the cache `ours_plans_v2` (491 requests: 2 first plans, 244 Rule 2 re-asks, 245 rejection re-asks). Codex is never called.
Output: `/common/users/ss5772/bimanual_data/contents/table_behind_same_2026-11-09/sonnet_plans_v2/` (cache JSONs, `cli_meta.jsonl`, `errors.jsonl`, `pairs.jsonl`, `SUMMARY.md`).
Already in that dir from the 11-12 smoke: the box_behind first plan (1 call). Resume skips it.

Env for every command (from the worktree root):
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 PYTHONPATH=.
    PY=/common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python

## Stage 1 (calls: Claude subscription, ~8-30 s and ~0.03 USD-equivalent each at effort high)
1. First plans only (1 call left: tray6_behind):
       $PY -m world_model.v2.nero.contents.proposer_compare send --kinds plan
2. Re-asks, a 40-request sample (first 10 per task x kind by cache key = 40 calls):
       $PY -m world_model.v2.nero.contents.proposer_compare send --kinds rule2,rejected --limit-per-kind 10
3. All re-asks (489 calls, 4 shards in parallel; shard k of N is deterministic, resume-safe):
       for k in 0 1 2 3; do setsid nohup $PY -m world_model.v2.nero.contents.proposer_compare send --kinds plan,rule2,rejected --shard $k/4 > $OUT/send_$k.log 2>&1 & done
   (a re-ask request embeds CODEX's previous plan / checker evidence and is sent verbatim; the Claude answer is to Codex's situation.)
4. Rerun `send` with the same arguments to retry failures listed in errors.jsonl (cut at the cap, CLI error, rate limit): a cut reply is NO answer.
5. Summary (compiles first plans only by default, ~6 s per plan; `--compile-kinds plan,rule2` compiles re-asks too, slow):
       $PY -m world_model.v2.nero.contents.proposer_compare analyze

Dry run (count, no call): add `--dry-run` to `send`.

## Stage 2 (document only, not run): the full v2 table-behind carry with the Sonnet cache, same 40 worlds
Same command as run_n20_v2b (its launch.sh), with `--plan-spec claude:claude-sonnet-5-5:high --plan-cache <sonnet_plans_v2>` and a new output dir; first fill the cache with
the plan + the Rule 2 / rejection loops that the Sonnet plan itself triggers (they differ from Codex's, so they are NEW requests): run once WITHOUT `--offline` (live, `rw` cache),
then rerun with `--offline` to check it replays. Same worlds = same seeds / n=20 per task / same Theta list as run_n20_v2b (bounds rule 8). Number of calls is not known beforehand:
Codex made 19 (box) and 34 (tray) re-asks per world at most, and 24 worlds ended in quota errors.

## Caveats to record with every result (bounds rule 9)
- model claude-sonnet-5-5, effort high (Codex: gpt-5.6-sol, reasoning high); temperature not settable, no seed; max_new_tokens=4096 is in the cache key but the CLI ignores it.
- Claude Code default system prompt replaced by ours; the Codex gateway preface is prepended to the user message for parity. Codex: `codex exec` adds its own agent prompt.
- Settings/hooks/CLAUDE.md/auto-memory off (`--setting-sources ''`, env). The CLI still sends the account e-mail (userEmail context) and an environment block (temp cwd, platform, date).
- No image in these requests (the cache has none). The static verdict on a world's Theta is not in the comparison script (stage 2 gives it).
- Subscription login: rate limits can end a run (errors.jsonl); n of re-ask requests answered may differ between models.
