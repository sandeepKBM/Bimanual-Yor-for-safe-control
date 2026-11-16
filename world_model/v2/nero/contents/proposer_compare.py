"""nero/contents/proposer_compare.py -- Codex vs Claude (Sonnet) as the PROPOSER, on the SAME requests.  SIMULATION / SAVED DATA ONLY.

Every request in the Codex cache (default: table_behind_same_2026-11-09/ours_plans_v2, `codex:gpt-5.6-sol:high`) is sent, byte-identical (text, system, max_new_tokens,
temperature, n; the cache holds no image bytes and the requests carry no image), to `claude:claude-sonnet-5-5:high` through CachedVLM into a NEW cache dir.  Codex is never
called: its answers come from its cache.  Kinds (classified from the request text):
    plan       the FIRST plan request of a task (stack/proposer/prompt: user_prompt).  One per task: the text holds no world seed, so every world of a task shares it.
               Rebuilt here through prompt.user_prompt / prompt.SYSTEM and checked against the cached text byte-for-byte.
    rule2      a Rule 2 re-ask (prompt.reask_prompt: 'YOUR PREVIOUS PLAN' + evidence).  The text embeds CODEX's previous plan and the checker evidence of a Codex run; it is sent
               verbatim (it cannot be rebuilt without that run), so Sonnet answers Codex's situation, not its own.
    rejected   a re-ask after a rejected plan (prompt.rejection_prompt), same remark.
Reported separately per (task, kind).

    send     python -m world_model.v2.nero.contents.proposer_compare send --out DIR [--shard k/N] [--kinds plan,rule2,rejected] [--limit-per-kind N] [--dry-run]
    analyze  python -m world_model.v2.nero.contents.proposer_compare analyze --out DIR      -> DIR/pairs.jsonl, DIR/SUMMARY.md (machine-written)

Resume: a request already in the Sonnet cache is skipped; a failed call (cut at the cap, CLI error, timeout) is appended to DIR/errors.jsonl and retried next time.
Per-call CLI meta (usage, cost, stop reason, wall) is appended to DIR/cli_meta.jsonl (CachedVLM does not store it).
Not computed here: the static verdict on a world's Theta (it needs the per-world carry harness: probe_select.run_world); stage 2 is in SONNET_COMPARE_RUN.md.
Bounds rule 9: results carry model, effort and the caveats of pan/llm.info(spec); the Codex side is whatever its cache record says.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

DATA = Path("/common/users/ss5772/bimanual_data/contents/table_behind_same_2026-11-09")
CODEX_DIR = DATA / "ours_plans_v2"
CLAUDE_DIR = DATA / "sonnet_plans_v2"
CLAUDE_SPEC = "claude:claude-sonnet-5-5:high"
KINDS = ("plan", "rule2", "rejected")


def classify(text):
    """-> (task, kind) from the request text."""
    m = re.match(r"TASK_ID: (\w+)", text)
    task = m.group(1) if m else "?"
    if "REJECTED PLAN 1 (JSON):" in text:
        return task, "rejected"
    if "YOUR PREVIOUS PLAN (JSON):" in text:
        return task, "rule2"
    return task, "plan"


def load_codex(d=CODEX_DIR):
    out = []
    for f in sorted(Path(d).glob("*.json")):
        r = json.loads(f.read_text())
        task, kind = classify(r["request"]["text"])
        out.append(dict(key=r["key"], task=task, kind=kind, rec=r))
    return out


def text_sha(text, system):
    return hashlib.sha256(json.dumps([text, system]).encode()).hexdigest()[:20]


def check_first_plan(item):
    """rebuild the first plan request through the live code path and compare with the cache, byte for byte -> dict(ok, text_equal, system_equal)."""
    from world_model.v2.stack.proposer import prompt as PR
    req = item["rec"]["request"]
    return dict(text_equal=PR.user_prompt(item["task"]) == req["text"], system_equal=PR.SYSTEM == req["system"], images=len(req.get("image_hashes") or []))


def select(items, kinds, shard, limit_per_kind, tasks=None):
    sel = [x for x in items if x["kind"] in kinds and (not tasks or x["task"] in tasks)]
    if limit_per_kind:
        cnt, keep = {}, []
        for x in sel:                                    # sorted by key: a deterministic subset
            k = (x["task"], x["kind"])
            if cnt.get(k, 0) < limit_per_kind:
                cnt[k] = cnt.get(k, 0) + 1
                keep.append(x)
        sel = keep
    k, n = shard
    return [x for j, x in enumerate(sel) if j % n == k]


def send(a):
    from world_model.v2.house.lang.backend import CachedVLM
    from world_model.v2.nero.pan import llm as LLM
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    items = load_codex(a.codex)
    k, n = (int(x) for x in a.shard.split("/"))
    sel = select(items, set(a.kinds.split(",")), (k, n), a.limit_per_kind, set(a.tasks.split(",")) if a.tasks else None)
    inner = LLM.make(a.spec)
    be = CachedVLM(inner, out)
    todo = []
    for x in sel:
        q = x["rec"]["request"]
        if x["kind"] == "plan":
            chk = check_first_plan(x)
            if not (chk["text_equal"] and chk["system_equal"] and chk["images"] == 0):
                print(f"WARNING {x['key']} {x['task']}: rebuilt first-plan request differs from the cache {chk}; the CACHED text is sent", flush=True)
        key = be.key(q["text"], [], q["system"], q["max_new_tokens"], q["temperature"], q["n"])
        if not (out / f"{key}.json").exists():
            todo.append((x, key))
    print(f"shard {k}/{n}: {len(sel)} selected, {len(sel) - len(todo)} already done, {len(todo)} to send ({a.spec})", flush=True)
    if a.dry_run:
        return
    for j, (x, key) in enumerate(todo):
        q = x["rec"]["request"]
        t0 = time.time()
        try:
            o = be.ask(q["text"], images=[], system=q["system"], max_new_tokens=q["max_new_tokens"], temperature=q["temperature"], n=q["n"])
            meta = dict(key=key, codex_key=x["key"], task=x["task"], kind=x["kind"], ok=True, **(o.get("cli") or {}))
            with open(out / "cli_meta.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
            print(f"[{j + 1}/{len(todo)}] {x['task']} {x['kind']} {key} ok {meta.get('wall_s')} s stop={meta.get('stop_reason')} cost={meta.get('cost_usd')}", flush=True)
        except Exception as e:                                   # noqa: BLE001  cut at the cap, CLI error, timeout: no answer, recorded, retried on resume
            with open(out / "errors.jsonl", "a") as f:
                f.write(json.dumps(dict(key=key, codex_key=x["key"], task=x["task"], kind=x["kind"], error=f"{type(e).__name__}: {e}"[:800], wall_s=round(time.time() - t0, 2))) + "\n")
            print(f"[{j + 1}/{len(todo)}] {x['task']} {x['kind']} {key} FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)


# ---- analysis ---------------------------------------------------------------------------------------------------------------------------
_SUP = {}


def _supports(task):
    from world_model.v2.stack.proposer import prompt as PR
    if task not in _SUP:                         # scene_facts takes seconds (it builds the task spec): once per task
        _SUP[task] = tuple(PR.scene_facts(task)["task"]["supports"])
    return _SUP[task]


def describe(reply, task, do_compile=True):
    """parse + validate (schema v2) + compile one reply -> flat dict of what the reply says."""
    from world_model.v2.stack.proposer import compile as PC, prompt as PR, schema as S
    d = dict(parse_ok=False)
    if reply is None:
        return d
    supports = _supports(task)
    try:
        tp = S.validate_reply(reply, task, supports)
    except S.PlanRejected as e:
        return dict(d, stage=e.stage, reasons=list(e.reasons)[:3])
    pl = tp.to_dict()
    pl.update(task=task, plan_id=S.plan_id(pl))
    d.update(parse_ok=True, plan_id=pl["plan_id"], phenomena=sorted(p["id"] if p["id"] != "NEW" else "NEW:" + str(p.get("name")) for p in pl["phenomena"]),
             handles=sorted(f"{p['id']}={'measure:' + p['handle']['measure'] if 'measure' in p['handle'] else 'unmeasurable'}" for p in pl["phenomena"]),
             probes=list(pl["probes"]), steps=[s["skill"] for s in pl["steps"]], n_steps=len(pl["steps"]), n_recoveries=sum(len(v) for v in pl["recoveries"].values()),
             recoveries={k: [r["skill"] for r in v] for k, v in pl["recoveries"].items()}, fallback=[r["skill"] for r in pl["fallback"]])
    if not do_compile:
        return d
    try:
        cp = PC.compile_plan(pl, task, level=("L1" if task in S.BASE_TASKS else None))
        d.update(compiles=True, segments=list(cp.job_plan("refuse")["wp"].names))
    except S.PlanRejected as e:
        d.update(compiles=False, compile_stage=e.stage, compile_reasons=list(e.reasons)[:3])
    except Exception as e:                                       # noqa: BLE001  a builder crash is a finding, not a pass
        d.update(compiles=False, compile_stage="crash", compile_reasons=[f"{type(e).__name__}: {str(e)[:200]}"])
    return d


def jaccard(x, y):
    x, y = set(x or []), set(y or [])
    return 1.0 if not x and not y else len(x & y) / len(x | y)


def analyze(a):
    out = Path(a.out)
    items = load_codex(a.codex)
    meta = {}
    if (out / "cli_meta.jsonl").exists():
        for ln in (out / "cli_meta.jsonl").read_text().splitlines():
            m = json.loads(ln)
            meta[m["key"]] = m
    errs = {}
    if (out / "errors.jsonl").exists():
        for ln in (out / "errors.jsonl").read_text().splitlines():
            e = json.loads(ln)
            errs[e["key"]] = e
    from world_model.v2.house.lang.backend import CachedVLM, Backend

    class _K(Backend):
        model = a.spec.split(":")[1]
    kk = CachedVLM(_K(), out)
    rows = []
    for x in items:
        q = x["rec"]["request"]
        key = kk.key(q["text"], [], q["system"], q["max_new_tokens"], q["temperature"], q["n"])
        f = out / f"{key}.json"
        row = dict(task=x["task"], kind=x["kind"], codex_key=x["key"], claude_key=key, claude_sent=f.exists(), claude_error=(errs.get(key) or {}).get("error"))
        cr = x["rec"]["response"]
        row["codex"] = dict(describe(cr["texts"][0], x["task"], x["kind"] in a.compile_kinds.split(",")), wall_s=cr.get("wall_s"))
        if f.exists():
            r = json.loads(f.read_text())
            m = meta.get(key, {})
            row["claude"] = dict(describe(r["response"]["texts"][0], x["task"], x["kind"] in a.compile_kinds.split(",")), wall_s=r["response"].get("wall_s"), reasoning=r.get("reasoning"), model=r.get("model"),
                                 stop_reason=m.get("stop_reason"), cost_usd=m.get("cost_usd"), out_tokens=(m.get("usage") or {}).get("output_tokens"),
                                 in_tokens=(m.get("usage") or {}).get("input_tokens"))
            c, s = row["codex"], row["claude"]
            if c["parse_ok"] and s["parse_ok"]:
                row["agree"] = dict(phenomena_jaccard=jaccard(c["phenomena"], s["phenomena"]), handles_jaccard=jaccard(c["handles"], s["handles"]), probes_jaccard=jaccard(c["probes"], s["probes"]),
                                    probes_equal=c["probes"] == s["probes"], steps_equal=c["steps"] == s["steps"], same_plan_id=c["plan_id"] == s["plan_id"])
        rows.append(row)
    (out / "pairs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "SUMMARY.md").write_text(summary(rows, a, out))
    print(f"wrote {out/'pairs.jsonl'} and {out/'SUMMARY.md'}")


def _mean(v):
    v = [x for x in v if x is not None]
    return f"{sum(v) / len(v):.2f}" if v else "-"


def summary(rows, a, out):
    from world_model.v2.nero.pan import llm as LLM
    inf = LLM.info(a.spec)
    L = ["# Codex vs Claude proposer, same requests (machine-written by nero/contents/proposer_compare.py)", "",
         f"- Codex: `codex:gpt-5.6-sol:high` from the cache `{a.codex}` (no Codex call made).",
         f"- Claude: model `{inf['model']}`, effort `{inf['reasoning']}`, backend `{inf['backend']}`, cache `{out}`.",
         f"- Caveats (bounds rule 9): {inf['caveats']}.",
         "- Same request = same text, system, max_new_tokens, temperature, n; no image (the cache holds none). Re-ask kinds embed CODEX's previous plan / the checker evidence of a Codex run, sent verbatim.",
         "- Static verdict on a world's Theta: NOT computed here (needs the per-world carry; see SONNET_COMPARE_RUN.md stage 2). Agreement rows are over requests where both replies parse.",
         "- n is small for the plan kind (one request per task). Examples, not rates.", "",
         "| task | kind | n codex | n sent | claude errors/cut | parse codex | parse claude | compile codex | compile claude | phenomena Jaccard | handles Jaccard | probes equal | steps equal | wall s codex | wall s claude | claude out tokens | claude cost USD |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for task in sorted({r["task"] for r in rows}):
        for kind in KINDS:
            g = [r for r in rows if r["task"] == task and r["kind"] == kind]
            if not g:
                continue
            s = [r for r in g if r["claude_sent"]]
            ag = [r["agree"] for r in s if "agree" in r]
            L.append(f"| {task} | {kind} | {len(g)} | {len(s)} | {sum(1 for r in g if r['claude_error'] and not r['claude_sent'])} | "
                     f"{sum(r['codex']['parse_ok'] for r in g)}/{len(g)} | {sum(r['claude']['parse_ok'] for r in s)}/{len(s)} | "
                     + (f"{sum(bool(r['codex'].get('compiles')) for r in g)}/{len(g)} | {sum(bool(r['claude'].get('compiles')) for r in s)}/{len(s)} | " if kind in a.compile_kinds.split(",") else "- | - | ") +
                     f"{_mean([x['phenomena_jaccard'] for x in ag])} | {_mean([x['handles_jaccard'] for x in ag])} | "
                     f"{sum(x['probes_equal'] for x in ag)}/{len(ag)} | {sum(x['steps_equal'] for x in ag)}/{len(ag)} | "
                     f"{_mean([r['codex']['wall_s'] for r in s])} | {_mean([r['claude']['wall_s'] for r in s])} | "
                     f"{_mean([r['claude'].get('out_tokens') for r in s])} | {_mean([r['claude'].get('cost_usd') for r in s])} |")
    L += ["", "Codex wall time is over the same requests that Claude answered (Codex wall_s as stored; Rule 2 loops hit Codex quota limits, those requests have no stored reply and are not in the cache).", ""]
    L += ["## First plan requests, side by side", ""]
    for r in rows:
        if r["kind"] != "plan":
            continue
        L.append(f"### {r['task']}")
        for who in ("codex", "claude"):
            if who not in r:
                L.append(f"- {who}: not sent" + (f" ({r['claude_error']})" if r.get("claude_error") else ""))
                continue
            d = r[who]
            L.append(f"- {who}: parse={d['parse_ok']} compiles={d.get('compiles')} wall={d.get('wall_s')} s" + (f"; phenomena={d['phenomena']}; handles={d['handles']}; probes={d['probes']}; steps={d['steps']}; "
                     f"recoveries={d['recoveries']}; fallback={d['fallback']}" if d["parse_ok"] else f"; rejected stage={d.get('stage')} {d.get('reasons')}"))
        L.append(f"- agreement: {r.get('agree')}")
        L.append("")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("send", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--codex", default=str(CODEX_DIR), help="the Codex cache dir (read only)")
        p.add_argument("--out", default=str(CLAUDE_DIR), help="the Claude cache dir (new)")
        p.add_argument("--spec", default=CLAUDE_SPEC)
    sub.choices["analyze"].add_argument("--compile-kinds", default="plan", help="kinds whose replies are compiled (the builder is slow: ~seconds per plan); others: parse only")
    s = sub.choices["send"]
    s.add_argument("--shard", default="0/1", help="k/N: this process sends requests j with j %% N == k")
    s.add_argument("--kinds", default="plan", help="comma list of plan,rule2,rejected (default plan only: 2 requests)")
    s.add_argument("--tasks", default="", help="comma list of tasks (default: all in the cache)")
    s.add_argument("--limit-per-kind", type=int, default=0, help="per (task, kind): the first N by cache key (0 = all)")
    s.add_argument("--dry-run", action="store_true", help="count and check only; no Claude call")
    a = ap.parse_args(argv)
    if a.cmd == "send" and os.environ.get("BIMANUAL_NO_NETWORK") == "1":
        sys.exit("BIMANUAL_NO_NETWORK=1: refusing to call the Claude CLI")
    (send if a.cmd == "send" else analyze)(a)


if __name__ == "__main__":
    main()
