"""vlm/phys2real_prior.py -- Phys2Real-style VLM physical-parameter prior (arXiv 2510.11689 protocol, local open VLM instead of GPT-5).

Protocol: prompt for BOTH an estimate and a self-reported sigma; aggregate over several images AND repeated queries (samples at T>0);
    mean        = mean of the estimates
    sigma_vlm   = MEAN of the VLM's reported sigmas        (Phys2Real's choice; they report std-of-estimates gives poor uncertainty)
    std_est     = std of the repeated estimates            (kept so both uncertainty definitions are available)
The prior is then fused with an online estimate by inverse-variance weighting (see `fuse`).

    phys2real_prior(params, images, n_repeats, run_fn, kind=..., scene_id=...) -> {param: dict(mean, sigma_vlm, std_est, samples)}
`run_fn(list_of_requests) -> {id: output}` is e.g. functools.partial(client.run_batch, "name"). For many scenes use
`build_requests` / `aggregate` directly so ALL requests ride in ONE Slurm job.
"""
from __future__ import annotations

import numpy as np

from world_model.v2.vlm import prompts as P
from world_model.v2.vlm.spec_eval import extract_json


def build_requests(scene_id, kind, images, n_repeats=3, hint=True, temperature=0.7, max_new_tokens=300, tag="p"):
    """one request per image (n_repeats samples each); ids '<tag>|<scene>|<hint>|<img index>'."""
    return [dict(id=f"{tag}|{scene_id}|{'h' if hint else 'n'}|{i}", images=[im], system=P.param_system(), text=P.param_user(kind, hint),
                 temperature=temperature, n=n_repeats, max_new_tokens=max_new_tokens) for i, im in enumerate(images)]


def parse_sample(text, kind):
    """text -> {param: (estimate, sigma)} for the params that parsed (positive finite estimate, positive finite sigma)."""
    j = extract_json(text)
    out = {}
    if not isinstance(j, dict):
        return out
    for p in P.KIND_PARAMS[kind]:
        v = j.get(P.PARAM_DEFS[p]["key"])
        try:
            e, s = float(v["estimate"]), float(v["sigma"])
        except Exception:  # noqa: BLE001
            continue
        if np.isfinite(e) and np.isfinite(s) and e > 0 and s > 0:
            out[p] = (e, s)
    return out


def aggregate(outputs_for_scene, kind):
    """outputs_for_scene: list of runner output records (one per image). -> {param: dict} ; n_total counts all samples incl. unparsed."""
    texts = [t for o in outputs_for_scene for t in o["texts"]]
    res = {}
    for p in P.KIND_PARAMS[kind]:
        sm = [parse_sample(t, kind).get(p) for t in texts]
        ok = [s for s in sm if s is not None]
        if not ok:
            res[p] = dict(mean=None, sigma_vlm=None, std_est=None, samples=[], n_total=len(texts))
            continue
        est = np.array([s[0] for s in ok])
        sig = np.array([s[1] for s in ok])
        res[p] = dict(mean=float(est.mean()), sigma_vlm=float(sig.mean()), std_est=float(est.std(ddof=1)) if len(est) > 1 else None,
                      samples=[list(s) for s in ok], n_total=len(texts))
    return res


def phys2real_prior(params, images, n_repeats, run_fn, kind="tray_ball", scene_id="scene", hint=True, **kw):
    reqs = build_requests(scene_id, kind, images, n_repeats, hint, **kw)
    outs = run_fn(reqs)
    agg = aggregate([outs[r["id"]] for r in reqs], kind)
    return {p: agg[p] for p in params}


def fuse(prior_mean, prior_sigma, online_mean, online_sigma):
    """inverse-variance fusion of the VLM prior with an online estimate -> (mean, sigma)."""
    w1, w2 = 1.0 / prior_sigma ** 2, 1.0 / online_sigma ** 2
    return (w1 * prior_mean + w2 * online_mean) / (w1 + w2), (w1 + w2) ** -0.5
