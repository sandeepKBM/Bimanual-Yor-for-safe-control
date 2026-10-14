"""lang/cli.py -- entry points of the language layer.  SIMULATION / SAVED DATA ONLY (the VLM is reached only through the file-queue service).

    make_lang_services(inner, cfg, out, sim, log)       used by flagship.simulate --lang
    python -m world_model.v2.house.lang.cli eval --instruction "..." [--kind desk|counter] [--pkl perception_desk.pkl] [--backend auto|live|mock]
                                          [--out DIR] [--cache DIR] [--mu-queries 4] [--mu-k 1.0] [--mu-image PNG ...]
        grounding + authoring + mu prior on a saved flagship perception result; writes <tag>_lang.json (+ marks png) and prints the score vs the gt keys
        (evaluation only: the gt keys are never shown to the VLM).
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from . import backend as BE, grounding as GR, session as SE, hooks as HK, prior as PR

DEFAULT_PKL = "/data/local/ss5772/molmospaces/flagship/perception/perception_{kind}.pkl"
DEFAULT_OUT = "/data/local/ss5772/molmospaces/lang"


def make_lang_services(inner, cfg, out, sim=None, log=print):
    out = Path(out)
    be = BE.make_backend(cfg.get("backend", "auto"), cache_dir=cfg.get("cache") or out / "lang_cache", log=log)
    sess = SE.LangSession(be, out_dir=out / "lang", log=log)
    (out / "lang").mkdir(parents=True, exist_ok=True)
    return HK.LangPerceptionService(inner, sess, cfg["instruction"], out / "lang", log=log, sim=sim, image=cfg.get("image", "main"))


def run_eval(instruction, kind="desk", pkl=None, backend="auto", out=DEFAULT_OUT, cache=None, tag=None, mu_queries=4, mu_k=1.0, mu_images=None, mode="replace", image=None, log=print):
    out = Path(out)
    with open(pkl or DEFAULT_PKL.format(kind=kind), "rb") as f:
        res = pickle.load(f)
    be = BE.make_backend(backend, cache_dir=cache or out / "cache", log=log)
    sess = SE.LangSession(be, mu_queries=mu_queries, mu_k=mu_k, spec_mode=mode, out_dir=out / "runs", log=log)
    props = GR.proposals_from_result(res)
    tag = tag or f"{kind}_{abs(hash(instruction)) % 10**6}"
    t0 = time.time()
    if image:                                           # e.g. the flat-colour main-sim render through the same camera pose (marks are projected with the result's own K / origin / R)
        import cv2
        rgb = cv2.imread(image)[:, :, ::-1]
    else:
        rgb = np.asarray(res["left"])
    lr = sess.run(instruction, rgb, props, mu_images=mu_images, tag=tag)
    gt = {z["id"]: z["gt_key"] for z in res["zonos"]}
    sc = GR.score_grounding(lr.grounding, gt)
    d = lr.to_json()
    d["grounding_score"] = {k: v for k, v in sc.items() if k != "rows"}
    d["grounding_rows"] = sc["rows"]
    d["mu_vs_truth"] = dict(truth=PR.SIM_TRUTH, hard_coded=list(PR.DEFAULT_MU), vlm=[lr.mu.lo, lr.mu.hi], vlm_covers_truth=lr.mu.covers(PR.SIM_TRUTH))
    d["call_trace"] = getattr(be, "trace", [])
    d["backend_name"] = getattr(be, "name", "")
    d["total_wall_s"] = time.time() - t0
    (out / "runs" / f"{tag}_lang.json").write_text(json.dumps(d, indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
    return lr, d


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--instruction", required=True)
    e.add_argument("--kind", default="desk", choices=["desk", "counter"])
    e.add_argument("--pkl", default=None)
    e.add_argument("--backend", default="auto", choices=["auto", "live", "mock"])
    e.add_argument("--out", default=DEFAULT_OUT)
    e.add_argument("--cache", default=None)
    e.add_argument("--tag", default=None)
    e.add_argument("--mu-queries", type=int, default=4)
    e.add_argument("--mu-k", type=float, default=1.0)
    e.add_argument("--mu-image", action="append", default=None)
    e.add_argument("--mode", default="replace", choices=["replace", "augment"])
    e.add_argument("--image", default=None, help="RGB png to show the VLM instead of the saved twin image (same camera pose / intrinsics as the saved result)")
    a = ap.parse_args(argv)
    lr, d = run_eval(a.instruction, a.kind, a.pkl, a.backend, a.out, a.cache, a.tag, a.mu_queries, a.mu_k, a.mu_image, a.mode, a.image)
    print(lr.spec.pretty())
    print(json.dumps({k: d[k] for k in ("spec_status", "checks", "clearances", "unresolved", "mu", "latency", "grounding_score", "mu_vs_truth")}, indent=1, default=str))


if __name__ == "__main__":
    main()
