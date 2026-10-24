"""vlm/codex_eval.py -- evaluate the Codex CLI model (cdx-agent's offload model, gpt-5.6-luna) as a pick-proposer / TEACHER on the SAME
prompts and images our served VLMs get.  SIMULATION DATA ONLY: sends sim renders + prompt text to the Codex CLI (user-approved 2026-10-16).

Each record: Qwen-chat format line from vlm/ftdata (messages = [system, user{image, text}, assistant{gold JSON}], meta).  Codex runs as
`codex exec` in an EMPTY temp dir, read-only sandbox, ephemeral session, with --output-schema for the pick answer (naming records: free text).
    python -m world_model.v2.vlm.codex_eval --data <split.jsonl> --n 60 --workers 3 --effort high --out <dir>
    python -m world_model.v2.vlm.codex_eval --score <dir>/codex_<split>.jsonl            # re-score only
Scores (vs the record's gold answer): parse rate, arms/arm agreement with the assign.py rule label, naming accuracy (category),
mass-range overlap with the hand prior table (log-IoU; the table is an ASSUMPTION, so this is agreement, not accuracy), fragile/handle
agreement, latency.  Small n = examples, not rates.
"""
from __future__ import annotations

import argparse, json, math, os, random, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from world_model.v2.house.pick.planner import parse_vlm_plan

MODEL = "gpt-5.6-luna"
SCHEMA = {"type": "object", "additionalProperties": False,
          "required": ["arms", "arm", "grasp", "mass_kg", "contents", "fragile", "handle", "why"],
          "properties": {"arms": {"type": "integer", "enum": [1, 2]}, "arm": {"type": "string", "enum": ["left", "right", "both"]},
                         "grasp": {"type": "string", "enum": ["pinch", "side", "squeeze"]},
                         "mass_kg": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2},
                         "contents": {"type": "string", "enum": ["liquid", "loose", "solid", "empty", "unknown"]},
                         "fragile": {"type": "boolean"}, "handle": {"type": "boolean"}, "why": {"type": "string"}}}


def _is_naming(rec):
    return "Name the boxed object" in rec["messages"][1]["content"][1]["text"]


def ask_codex(rec, effort="high", timeout_s=900):
    sysm = rec["messages"][0]["content"]
    img = rec["messages"][1]["content"][0]["image"]
    txt = rec["messages"][1]["content"][1]["text"]
    naming = _is_naming(rec)
    prompt = (f"{sysm}\n\n{txt}\n\n" + ("Answer with the name only." if naming else "Answer with ONE JSON object matching the schema."))
    with tempfile.TemporaryDirectory(prefix="cx_", dir=os.environ.get("CODEX_EVAL_TMP")) as d:
        out = Path(d) / "last.txt"
        cmd = ["codex", "exec", "--skip-git-repo-check", "--ephemeral", "-s", "read-only", "-m", MODEL, "-c", f'model_reasoning_effort="{effort}"',
               "-C", d, "-i", img, "-o", str(out)]
        if not naming:
            sp = Path(d) / "schema.json"
            sp.write_text(json.dumps(SCHEMA))
            cmd += ["--output-schema", str(sp)]
        t0 = time.time()
        p = subprocess.run(cmd + [prompt], capture_output=True, text=True, timeout=timeout_s, stdin=subprocess.DEVNULL)
        dt = time.time() - t0
        text = out.read_text() if out.exists() else ""
    return dict(text=text.strip(), seconds=round(dt, 2), rc=p.returncode, err=(p.stderr or "")[-400:] if p.returncode else "")


def ask_svc(rec, timeout_s=900):
    """same record through the running file-queue daemon (Qwen / Cosmos); identical prompt text, the pick answer budget of the planner."""
    from world_model.v2.vlm import service
    from world_model.v2.house.pick.planner import ASK_TOKENS
    naming = _is_naming(rec)
    o = service.ask(rec["messages"][1]["content"][1]["text"], images=[rec["messages"][1]["content"][0]["image"]], system=rec["messages"][0]["content"],
                    max_new_tokens=40 if naming else ASK_TOKENS, temperature=0.0, timeout_s=timeout_s)
    return dict(text=(o.get("texts") or [""])[0].strip(), seconds=o.get("wall_s", o.get("seconds", -1)), rc=0, err="", model=o.get("model"))


def _log_iou(a, b):
    try:
        la, lb = [math.log(max(x, 1e-3)) for x in a], [math.log(max(x, 1e-3)) for x in b]
    except Exception:  # noqa: BLE001
        return None
    lo, hi = max(min(la), min(lb)), min(max(la), max(lb))
    inter = max(0.0, hi - lo)
    union = max(max(la), max(lb)) - min(min(la), min(lb))
    return inter / union if union > 0 else float(la == lb)


def score(rows):
    S = dict(n=len(rows), n_pick=0, parse=0, arms_agree=0, arm_agree=0, mass_iou=[], fragile_agree=0, handle_agree=0, n_name=0, name_ok=0, sec=[])
    for r in rows:
        S["sec"].append(r["seconds"])
        if r["naming"]:
            S["n_name"] += 1
            cat = (r["meta"].get("category") or "").lower()
            disp = (r["gold"] or "").lower()
            ans = r["text"].lower()
            S["name_ok"] += int(bool(ans) and (disp in ans or cat in ans.replace(" ", "")))
            continue
        S["n_pick"] += 1
        p, g = parse_vlm_plan(r["text"]), parse_vlm_plan(r["gold"])
        if p is None or g is None:
            continue
        S["parse"] += 1
        S["arms_agree"] += int(p["arms"] == g["arms"])
        S["arm_agree"] += int(p["arm"] == g["arm"])
        if p.get("mass_range") and g.get("mass_range"):
            S["mass_iou"].append(_log_iou(p["mass_range"], g["mass_range"]))
        S["fragile_agree"] += int(p.get("fragile") == g.get("fragile"))
        S["handle_agree"] += int(p.get("handle") == g.get("handle"))
    n = max(S["n_pick"], 1)
    mi = [x for x in S["mass_iou"] if x is not None]
    sec = sorted(S["sec"]) or [0]
    return dict(n=S["n"], n_pick=S["n_pick"], parse_rate=S["parse"] / n, arms_agree=S["arms_agree"] / n, arm_agree=S["arm_agree"] / n,
                mass_logiou_mean=sum(mi) / len(mi) if mi else None, fragile_agree=S["fragile_agree"] / n, handle_agree=S["handle_agree"] / n,
                n_name=S["n_name"], name_acc=S["name_ok"] / max(S["n_name"], 1), sec_p50=sec[len(sec) // 2], sec_max=sec[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--effort", default="high", help="model_reasoning_effort (config default is xhigh)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="/data/local/ss5772/molmospaces/codex_eval")
    ap.add_argument("--score")
    ap.add_argument("--backend", choices=["codex", "svc"], default="codex", help="svc = the running vlm daemon at $BIMANUAL_VLM_SVC")
    ap.add_argument("--tag", default=None, help="output tag (default: codex effort / svc model)")
    a = ap.parse_args()
    if a.score:
        rows = [json.loads(l) for l in open(a.score) if l.strip()]
        print(json.dumps(score(rows), indent=1))
        return
    recs = [json.loads(l) for l in open(a.data) if l.strip()]
    random.Random(a.seed).shuffle(recs)
    recs = recs[:a.n]
    os.makedirs(a.out, exist_ok=True)
    tag = a.tag or (f"codex_{a.effort}" if a.backend == "codex" else "svc")
    fn = Path(a.out) / f"{tag}_{Path(a.data).stem}.jsonl"
    done = {json.loads(l)["key"] for l in open(fn) if l.strip()} if fn.exists() else set()

    def one(i_rec):
        i, rec = i_rec
        key = rec["messages"][1]["content"][0]["image"] + ("|name" if _is_naming(rec) else "")
        if key in done:
            return None
        try:
            o = ask_codex(rec, a.effort) if a.backend == "codex" else ask_svc(rec)
        except Exception as e:  # noqa: BLE001
            o = dict(text="", seconds=-1, rc=-1, err=f"{type(e).__name__}: {e}"[:300])
        row = dict(key=key, naming=_is_naming(rec), gold=rec["messages"][2]["content"], meta=rec.get("meta", {}), **o)
        with open(fn, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[{i + 1}/{len(recs)}] {row['meta'].get('category')} {o['seconds']}s rc={o['rc']} {o['text'][:90]!r}", flush=True)
        return row

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, enumerate(recs)))
    rows = [json.loads(l) for l in open(fn) if l.strip()]
    print(json.dumps(score(rows), indent=1))


if __name__ == "__main__":
    main()
