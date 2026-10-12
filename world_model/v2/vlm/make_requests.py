"""vlm/make_requests.py -- build the shared bake-off request file (same requests for every model).
Order = priority: (b) physical-parameter priors first, then (a) spec authoring.
    python -m world_model.v2.vlm.make_requests [--n_repeats 3] [--scenes <meta.json>] [--out <req.jsonl>]
B1 hand-spec tasks (world_model/v2/constraints/) are NOT wired yet: see spec_tasks() to append them when those files exist."""
import argparse, json
from pathlib import Path

from world_model.v2.vlm import prompts as P
from world_model.v2.vlm import phys2real_prior as PR

DATA = Path("/common/users/ss5772/bimanual_data/vlm")


def spec_tasks(meta):
    """(scene dicts to ask a spec for). Extend here with B1's tasks when world_model/v2/constraints/ exists."""
    return meta


def build(meta, n_repeats=3):
    reqs = []
    for s in meta:                                           # (b) params with range hints, per image, n_repeats samples each
        reqs += PR.build_requests(s["id"], s["kind"], s["images"], n_repeats, hint=True)
    for s in meta:                                           # (b') no range hint, tray_ball only
        if s["kind"] == "tray_ball":
            reqs += PR.build_requests(s["id"], s["kind"], s["images"], n_repeats, hint=False)
    for s in spec_tasks(meta):                               # (a) specs: greedy, both images
        reqs.append(dict(id=f"s0|{s['id']}", images=s["images"], system=P.spec_system(), text=P.spec_user(s), temperature=0.0,
                         max_new_tokens=400))
    for s in spec_tasks(meta):                               # (a') specs sampled
        reqs.append(dict(id=f"s1|{s['id']}", images=s["images"], system=P.spec_system(), text=P.spec_user(s), temperature=0.7, n=3,
                         max_new_tokens=400))
    return reqs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_repeats", type=int, default=3)
    ap.add_argument("--scenes", default=str(DATA / "scenes" / "meta.json"))
    ap.add_argument("--out", default=str(DATA / "req" / "bake.jsonl"))
    ap.add_argument("--specs_v2", action="store_true", help="spec requests only, with the corrected entity docs (ENTITY_DOC_V2)")
    a = ap.parse_args()
    meta = json.load(open(a.scenes))
    if a.specs_v2:
        from world_model.v2.vlm.scenes import ENTITY_DOC_V2
        meta = [dict(s, entities=ENTITY_DOC_V2[s["kind"]]) for s in meta]
        r = [x for x in build(meta, a.n_repeats) if x["id"].split("|")[0] in ("s0", "s1")]
    else:
        r = build(meta, a.n_repeats)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("".join(json.dumps(x) + "\n" for x in r))
    print(len(r), "requests ->", a.out)
