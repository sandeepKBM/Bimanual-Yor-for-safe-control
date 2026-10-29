"""plan_patch.py -- write a patched COPY of a saved AgentChord plan dir (runs/plans/<src>) as runs/plans/<src>_<tag>.  The source
plan and their code are never edited; every change is counted in PATCH.json.  Replay the copy with PLAN_NAME=<src>_<tag>.

  --rename     names that are not scene objects are mapped to the unique scene object whose name is a token of it
               (tray_left_handle -> tray).  Stands in for the paper's open-vocabulary grounding (SAM3 segments whatever noun
               the planner writes); the released sim instead looks names up exactly and crashes on any other name.
               Caveat: a real 'tray left handle' query would return the handle's pose, not the tray centre's.
  --force-valid
               set every force_valid flag to true: their documented opt-in 'project an invalid target pose to the nearest
               valid pose' (the paper's "target poses are clipped to a predefined workspace").  Tests what the planner
               would have got had it opted in.

    python plan_patch.py SRC --tag rn --rename --objects tray capsule table
    python plan_patch.py SRC --tag fv --force-valid
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

PLANS = Path("/data/local/ss5772/agentchord/runs/plans")
FILES = ("agent_task_graph.json", "agent_recovery_spec.json", "agent_compiled_graph.json")


def obj_names(text):
    return set(re.findall(r'obj_name["\']?\s*[:=]\s*["\']([A-Za-z0-9_]+)["\']', text)) | \
        {o for lst in re.findall(r'"objects"\s*:\s*\[([^\]]*)\]', text) for o in re.findall(r'"([A-Za-z0-9_]+)"', lst)}


def build_map(names, scene):
    out, unresolved = {}, []
    for n in sorted(names - set(scene)):
        hits = [s for s in scene if s in n.split("_")]
        (out.__setitem__(n, hits[0]) if len(hits) == 1 else unresolved.append(n))
    return out, unresolved


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--rename", action="store_true")
    ap.add_argument("--objects", nargs="+", default=["tray", "capsule", "table"])
    ap.add_argument("--force-valid", action="store_true")
    a = ap.parse_args(argv)
    src, dst = PLANS / a.src, PLANS / f"{a.src}_{a.tag}"
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    texts = {f: (src / f).read_text() for f in FILES if (src / f).exists()}
    rec = dict(source=a.src, rename=a.rename, force_valid=a.force_valid, n_changed={})
    if a.rename:
        mp, unres = build_map(set().union(*(obj_names(t) for t in texts.values())), a.objects)
        rec.update(name_map=mp, unresolved=unres)
        for f, t in texts.items():
            n = 0
            for k, v in mp.items():
                t, c = re.subn(rf'(?<![A-Za-z0-9_]){re.escape(k)}(?![A-Za-z0-9_])', v, t)
                n += c
            texts[f] = t
            rec["n_changed"][f"rename:{f}"] = n
    if a.force_valid:
        for f, t in texts.items():
            t, c = re.subn(r'("force_valid"\s*:\s*)false', r'\1true', t)
            texts[f] = t
            rec["n_changed"][f"force_valid:{f}"] = c
    for f, t in texts.items():
        json.loads(t)                                     # still valid JSON
    dst.mkdir()
    for f in src.iterdir():
        if f.name not in texts and f.is_file():
            shutil.copy2(f, dst / f.name)
    for f, t in texts.items():
        (dst / f).write_text(t)
    (dst / "PATCH.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(rec))


if __name__ == "__main__":
    main()
