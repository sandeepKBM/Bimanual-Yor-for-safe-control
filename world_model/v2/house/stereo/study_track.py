"""stereo/study_track.py -- STAGE 2b: the people tracker (ocs.PeopleProducer -> people_track DepthDetector + MultiTracker) fed by three depth sources on the SAME
recorded runs:  stereo (SGBM, sg_*) | ray-cast + noise model with the SAME geometry (rz_*) | ray-cast, default tracker camera (rd_*, fovy 55).  SIMULATION ONLY.

    python -m world_model.v2.house.stereo.study_track --tags speckle_s1_640x360 speckle_s2_640x360 ... [--noise fitted.json]

The tracker runs at 50 Hz ticks (frames every 5th tick = 10 Hz, latency 3 ticks as in the live pipeline); localisation is the TRUE base pose (isolates the depth source).
Metrics at each snapshot, over person-visible snapshots (ZED FOV + line of sight + range < 4.5 m, truth):
  detection rate   an associated confirmed track within 0.5 m of the person (also within 1.0 m)
  position RMSE    over the associated snapshots (nearest track within 1 m)
  false tracks     track-seconds of confirmed tracks farther than 1.0 m from every person / total seconds, and the count of distinct false track ids
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.sets.ir import Pose
from world_model.v2.house.ocs.state import WorldState, RobotState, Kind
from world_model.v2.house.ocs import util as U
from world_model.v2.house.people_track import camera as CAM
from .cfg import StereoCfg
from .twin import OUT
from . import adapters as AD

TICK = 0.02
RMAX = 4.5


class _View:
    def __init__(self):
        self.k, self.t, self.frame, self.base = 0, 0.0, None, np.zeros(3)

    def odom(self):
        return dict(pose=self.base)

    def head_frame(self):
        return self.frame


def run_source(tags, prefix, spec, out_root=OUT, tracker_cfg=None, r_max=RMAX, gate=(0.5, 1.0), min_pts=None):
    from world_model.v2.house.ocs.producers import PeopleProducer
    per_tag = []
    for tag in tags:
        files = sorted((Path(out_root) / "cache" / tag).glob("fr_*.npz"))
        seed = int(tag.split("_s")[1].split("_")[0])
        snap = np.load(Path(out_root) / "snap" / f"carry_s{seed}.npz", allow_pickle=True)
        pp = AD.make_people_producer(spec, tracker_cfg=tracker_cfg, r_max=r_max, **({} if min_pts is None else dict(min_pts=min_pts)))
        v = _View()
        rows = []
        false_ids, false_sec, total_sec = set(), 0.0, 0.0
        k_prev = None
        for f in files:
            z = np.load(f)
            i, k = int(z["i"]), int(z["k"])
            # ticks between snapshots: predict-only updates
            if k_prev is not None:
                for kk in range(k_prev + 1, k):
                    v.k, v.t, v.frame = kk, kk * TICK, None
                    ws = WorldState(t=v.t, robot=RobotState(base=U.base_pose(*v.base)))
                    pp.update(ws, v, v.t)
            hf = dict(tick=k, origin=z[f"{prefix}_origin"], R=z[f"{prefix}_R"], base_true=snap["base"][i], pts=z[f"{prefix}_pts"], z=z[f"{prefix}_z"], pix=z[f"{prefix}_pix"],
                      zs=z[f"{prefix}_zs"], person=z[f"{prefix}_person"])
            v.k, v.t, v.frame, v.base = k, k * TICK, hf, snap["base"][i]
            ws = WorldState(t=v.t, robot=RobotState(base=U.base_pose(*v.base)))
            pp.update(ws, v, v.t)
            k_prev = k
            tracks = [(o.oid, ws.world_pose(o.oid).p[:2].copy()) for o in ws.objects.values() if o.kind == Kind.PERSON]
            truth = snap["ppl_xy"][i]
            vis = z["vis_zed"] if len(z["vis_zed"]) else np.zeros((0, 3))
            dt = 5 * TICK
            total_sec += dt
            for oid, p in tracks:
                if len(truth) == 0 or np.min(np.linalg.norm(truth - p, axis=1)) > gate[1]:
                    false_ids.add(oid)
                    false_sec += dt
            for pi in range(len(truth)):
                inf, los, rng = vis[pi]
                if not (inf and los and rng < r_max):
                    continue
                if tracks:
                    d = np.array([np.linalg.norm(p - truth[pi]) for _, p in tracks])
                    j = int(np.argmin(d))
                    rows.append(dict(i=i, person=pi, rng=float(rng), err=float(d[j]), det05=bool(d[j] < gate[0]), det10=bool(d[j] < gate[1])))
                else:
                    rows.append(dict(i=i, person=pi, rng=float(rng), err=None, det05=False, det10=False))
        per_tag.append(dict(tag=tag, rows=rows, false_tracks=len(false_ids), false_sec=false_sec, total_sec=total_sec))
    return per_tag


def summarize(per_tag):
    rows = [r for p in per_tag for r in p["rows"]]
    n = len(rows)
    e05 = np.array([r["err"] for r in rows if r["det05"]])
    e10 = np.array([r["err"] for r in rows if r["det10"]])
    rng = np.array([r["rng"] for r in rows])
    out = dict(n_visible=n, det05=float(np.mean([r["det05"] for r in rows])) if n else None, det10=float(np.mean([r["det10"] for r in rows])) if n else None,
               rmse_m=float(np.sqrt((e10 ** 2).mean())) if len(e10) else None, rmse05_m=float(np.sqrt((e05 ** 2).mean())) if len(e05) else None,
               median_err_m=float(np.median(e10)) if len(e10) else None, p95_err_m=float(np.percentile(e10, 95)) if len(e10) else None,
               false_tracks=int(sum(p["false_tracks"] for p in per_tag)), false_track_frac=float(sum(p["false_sec"] for p in per_tag) / max(sum(p["total_sec"] for p in per_tag), 1e-9)),
               total_s=float(sum(p["total_sec"] for p in per_tag)))
    bins = []
    for lo, hi in ((0, 1.5), (1.5, 2.5), (2.5, 3.5), (3.5, 4.5)):
        m = [r for r in rows if lo <= r["rng"] < hi]
        if len(m) >= 5:
            ee = np.array([r["err"] for r in m if r["det10"]])
            bins.append(dict(lo=lo, hi=hi, n=len(m), det05=float(np.mean([r["det05"] for r in m])), rmse=float(np.sqrt((ee ** 2).mean())) if len(ee) else None))
    out["by_range"] = bins
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    ap.add_argument("--noise", default=None, help="json with sigma_disp_px / sigma_floor (fitted) for the stereo spec")
    ap.add_argument("--name", default="track")
    a = ap.parse_args()
    cfg = StereoCfg(W=640, H=360)
    kw = {}
    if a.noise:
        kw = json.loads(Path(a.noise).read_text())
    res = {}
    spec_s = AD.stereo_spec(cfg, 4, **kw)
    from dataclasses import replace
    spec_z = replace(CAM.head_spec(), fovy=cfg.fovy, W=160, H=90, native_H=cfg.H, name="head_zed")
    spec_d = CAM.head_spec(W=160, H=90)
    for name, prefix, spec, mp in (("stereo_sgbm_minpts6", "sg", spec_s, 6), ("stereo_sgbm", "sg", spec_s, AD.MIN_PTS_STEREO), ("raycast_zedgeom", "rz", spec_z, 6),
                                   ("raycast_default", "rd", spec_d, 6)):
        res[name] = summarize(run_source(a.tags, prefix, spec, min_pts=mp))
        print(name, json.dumps({k: v for k, v in res[name].items() if k != "by_range"}), flush=True)
    res["tags"] = a.tags
    (Path(OUT) / f"{a.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
