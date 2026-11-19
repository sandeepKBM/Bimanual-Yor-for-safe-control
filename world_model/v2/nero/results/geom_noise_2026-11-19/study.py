"""Noise study of the depth box fit (plan Phase 3).  SIMULATION ONLY, no network.
python -m world_model.v2.nero.results.geom_noise_2026-11-19.study --worlds 0:10 --seeds 0,1,2,3 --noise clean,zed2i_lab --out DIR [--task box_bookcase]
One JSON per (task, world, noise, seed) in DIR/runs/, written atomically, skipped if present.  clean is deterministic: seed 0 only.
Truth = the geom poses read from the model (copied from nero/contents/tests/test_geom_probe_gate.py), NOT geom_probe.true_box.  The GPU comes from the env."""
import argparse
import json
import os
import resource
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
import mujoco

from world_model.v2.nero.base import geom_prior as GPR
from world_model.v2.nero.base import scene as BS
from world_model.v2.nero.base import smoke as SM
from world_model.v2.nero.contents import geom_probe as GP
from world_model.v2.nero.contents import plan_images as PI
from world_model.v2.nero.contents import scenes as SC

BOARDS = ("counter", "bookcase_l1_floor", "bookcase_l1_base", "bookcase_plinth", "bookcase_mid", "bookcase_top", "bookcase_side_left", "bookcase_side_right",
          "bookcase_back", "pan_table")
GEOM_OF = {"pan_table": "pan_table_top"}
FIELDS = ("cx", "cy", "cz", "sx", "sy", "sz", "yaw", "top_z")


def _true_geom_box(m, d, name):
    g = m.geom(name).id
    R = d.geom_xmat[g].reshape(3, 3)
    s = 2 * m.geom_size[g]
    return dict(cx=d.geom_xpos[g][0], cy=d.geom_xpos[g][1], cz=d.geom_xpos[g][2], sx=s[0], sy=s[1], sz=s[2], yaw=float(np.arctan2(R[1, 0], R[0, 0])))


def _true_crate(m, d):
    b = m.body("obj").id
    R = d.xmat[b].reshape(3, 3)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != b or m.geom(g).name.startswith("tab"):
            continue
        Rg = d.geom_xmat[g].reshape(3, 3)
        c = R.T @ (d.geom_xpos[g] - d.xpos[b])
        e = np.abs(R.T @ Rg) @ m.geom_size[g]
        lo, hi = np.minimum(lo, c - e), np.maximum(hi, c + e)
    cw = d.xpos[b] + R @ ((lo + hi) / 2)
    s = hi - lo
    return dict(cx=cw[0], cy=cw[1], cz=cw[2], sx=s[0], sy=s[1], sz=s[2], yaw=float(np.arctan2(R[1, 0], R[0, 0])))


def _fold(t):
    t = dict(t)
    y = (t["yaw"] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    k = int(round((t["yaw"] - y) / (np.pi / 2))) % 2
    if k:
        t["sx"], t["sy"] = t["sy"], t["sx"]
    t["yaw"] = y
    t["top_z"] = t["cz"] + t["sz"] / 2
    return t


def _err(f, v, t):
    """signed error (m; yaw in deg, folded mod 90 deg)."""
    if f == "yaw":
        return float(np.degrees((v - t + np.pi / 4) % (np.pi / 2) - np.pi / 4))
    return float(v - t)


def _scene(task, i):
    _, w = SM.behind_world(task, i)
    m = BS.build_spec(w).compile()
    d = mujoco.MjData(m)
    SC.settle(m, d, PI.SETTLE_S)
    mujoco.mj_forward(m, d)
    return w, m, d


def _entity_rec(fit, truth):
    fields = {}
    for f in FIELDS:
        rec = fit.get("fields", {}).get(f)
        if rec is None:
            continue
        obs = bool(rec.get("observed"))
        v = float(rec["value"]) if obs and rec.get("value") is not None and np.isfinite(rec["value"]) else None
        fields[f] = dict(value=v, observed=obs and v is not None, truth=float(truth[f]), error=_err(f, v, truth[f]) if v is not None else None)
    return dict(fit_ok=bool(fit.get("fit_ok")), why=fit.get("why"), n_points=fit.get("n_points"), n_cams=fit.get("n_cams"), fields=fields)


def _write(path, obj):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=1, default=float)
    os.replace(tmp, path)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", default="0:1")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--noise", default="clean,zed2i_lab")
    ap.add_argument("--out", required=True)
    ap.add_argument("--task", default="box_bookcase")
    a = ap.parse_args(argv)
    lo, hi = (int(x) for x in a.worlds.split(":"))
    seeds = [int(s) for s in a.seeds.split(",")]
    noises = a.noise.split(",")
    runs = os.path.join(a.out, "runs")
    os.makedirs(runs, exist_ok=True)
    for i in range(lo, hi):
        todo = [(nz, s) for nz in noises for s in ([seeds[0]] if nz == "clean" else seeds)
                if not os.path.exists(os.path.join(runs, f"{a.task}_w{i}_{nz}_s{s}.json"))]
        if not todo:
            continue
        w, m, d = _scene(a.task, i)
        sup = float(SC.support_tops(w)["counter"]["top"])
        geom = GPR.draw(a.task, i)
        truths = {"crate": _fold(_true_crate(m, d))}
        for n in BOARDS:
            truths[n] = _fold(_true_geom_box(m, d, GEOM_OF.get(n, n)))
        for nz, s in todo:
            t0 = time.time()
            clouds = GP.lab_clouds(m, d, w, noise=nz, rng=np.random.default_rng(s))
            t1 = time.time()
            ents, tfit = {}, {}
            for n in truths:
                t2 = time.time()
                fit = GP.measure_box(clouds, GP.entity_geoms(m, n), support_top_z=sup if n == "crate" else None)
                tfit[n] = time.time() - t2
                ents[n] = _entity_rec(fit, truths[n])
            rec = dict(task=a.task, world=i, noise=nz, seed=s, geom_draw=geom, support_top_z=sup, entities=ents,
                       note="clean is deterministic: seed 0 only" if nz == "clean" else None,
                       cams=[c["label"] for c in clouds], image_size=list(PI.room_size()),
                       timing=dict(clouds_s=t1 - t0, fit_s=tfit, fit_total_s=sum(tfit.values())),
                       maxrss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
            _write(os.path.join(runs, f"{a.task}_w{i}_{nz}_s{s}.json"), rec)
            print(f"w{i} {nz} s{s}: clouds {t1 - t0:.1f}s fits {sum(tfit.values()):.1f}s rss {rec['maxrss_mb']:.0f}MB", flush=True)


if __name__ == "__main__":
    main()
