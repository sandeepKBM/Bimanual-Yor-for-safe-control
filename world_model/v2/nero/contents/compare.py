"""nero/contents/compare.py -- ours (per-cell certificate) vs AgentChord-style (execute, pose monitors) on the same worlds.

Machine-written table from:
  cert  = carry_cert run dir (STATIC.json + trials.jsonl): our admission per cell + sampled trials of the admitted cells
  base  = baseline_ungated run dir (trials.jsonl): the cells WE refuse, executed anyway
In the cells we admit, the AgentChord-style policy executes the same plan with the same controller on the same worlds, so its nominal
outcome is our nominal outcome (it has no stored backup; its monitors cannot fire under the weld).

    python -m world_model.v2.nero.contents.compare --cert DIR --base DIR [--out FILE]
"""
from __future__ import annotations

import argparse
import collections
import json
import pathlib


def _load(p):
    return [json.loads(l) for l in open(p)] if pathlib.Path(p).exists() else []


def table(cert_dir, base_dir):
    cert_dir, base_dir = pathlib.Path(cert_dir), pathlib.Path(base_dir)
    st = json.loads((cert_dir / "STATIC.json").read_text())["cells"]
    ours = collections.defaultdict(list)
    for r in _load(cert_dir / "trials.jsonl"):
        ours[r["key"]].append(r)
    base = collections.defaultdict(list)
    for r in _load(base_dir / "trials.jsonl"):
        base[f"{r['task']}|{r['cell'][0]}-{r['cell'][1]}"].append(r)
    L = ["cell | contents kg | OURS: decision | nominal unsafe (of n) | worlds with a failed backup || AGENTCHORD-STYLE: executes | "
         "unsafe (of n) | first unsafe kind | ends tipped/not at rest | its monitor fired", ""]
    for key, c in st.items():
        s = c["static"]
        m = f"{s['contents_m'][0]:.2f}-{s['contents_m'][1]:.2f}"
        if s["a_admit"] is not None:
            rs = [r for r in ours.get(key, []) if "error" not in r]
            n = len(rs)
            nom = sum(any(g["nominal"]["viol"] for g in r["segments"]) for r in rs)
            bk = sum(any(g["family"]["0.0"]["n_bad"] for g in r["segments"]) for r in rs)
            kinds = collections.Counter(next((g["nominal"]["viol"]["kind"] for g in r["segments"] if g["nominal"]["viol"]), None)
                                        for r in rs)
            kinds.pop(None, None)
            notR = sum(not r["end_in_R"][0] for r in rs)
            L.append(f"{key} | {m} | {s['action']} | {nom}/{n} | {bk}/{n} || executes (same plan) | {nom}/{n} | "
                     f"{dict(kinds) or '-'} | {notR}/{n} | 0/{n} (weld: no relative motion)")
        else:
            rs = [r for r in base.get(key, []) if "error" not in r]
            n = len(rs)
            u = sum(r["any_U"] for r in rs)
            kinds = collections.Counter((r["first_U"] or {}).get("kind") for r in rs if r["any_U"])
            notR = sum(not r["end_in_R"][0] for r in rs)
            fired = sum(r["monitor"]["fired"] for r in rs)
            L.append(f"{key} | {m} | REFUSE (u_nom {s['u_nom']}) | - | - || executes | {u}/{n} | {dict(kinds) or '-'} | "
                     f"{notR}/{n} | {fired}/{n}")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cert", default="/data/local/ss5772/contents/carry_cert_2026-11-02")
    ap.add_argument("--base", default="/data/local/ss5772/contents/baseline_ungated_2026-11-02")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    t = table(a.cert, a.base)
    print(t)
    if a.out:
        pathlib.Path(a.out).write_text(t + "\n")


if __name__ == "__main__":
    main()
