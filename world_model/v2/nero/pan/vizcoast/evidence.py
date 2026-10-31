"""Recorded sim runs as a counterexample store.  SIMULATION ONLY; read-only on the result folders.

Each entry: sig (same tuple as domain.Plan.signature), theta (true-world parameters of that run), held, cat, src.
Sources (2026-10-28 sweeps, pan_handle_grasp_2026-10-28/): one-claw (v1 force100, v2 one-claw fractions, c4 one-claw),
two claws on the handle (v2/v3/c4 with a second claw), under-rim (ur_*, robust/, robust_L70/, robust_mass_known/), and any
new Stage-B runs (vizcoast/*.jsonl).
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

from ...config import REPO
from . import domain as D

RES = Path(REPO) / "world_model/v2/nero/results/pan_handle_grasp_2026-10-28"
PAN_LID_KG = 0.304845189535163          # MODEL: pan + lid mass in the compiled covered scene (handle_grasp nominal_mass_kg)


def category(r):
    if "error" in r:
        return "no_clean_pregrasp" if "pre-grasp" in r["error"] else "error"
    if r.get("held"):
        return "held"
    sq = r.get("hold_last1s_squeeze_N")
    if isinstance(sq, dict):
        if sq.get("right", 99) < 1.0:
            return "right_claw_lost_or_missed"
        if sq.get("left", 99) < 1.0:
            return "left_claw_lost_or_missed"
    if r.get("hold_last1s_support_N", r.get("hold_last1s_support_Fz_N", 0.0)) > 0.1:
        return "still_on_support"
    if r.get("lift_achieved_m", 0.0) < 0.04:
        return "sag_or_no_lift"
    return "slip_or_tilt"


def _theta(r):
    p = r.get("perturb") or {}
    pl = r.get("placement", "none")
    hm = 0.0 if pl == "none" else float(r.get("hidden_mass") if r.get("hidden_mass") is not None else (0.6 if pl == "overload" else 0.2))
    return dict(dx=float(p.get("dx", 0.0)), dy=float(p.get("dy", 0.0)), yaw_deg=float(p.get("yaw_deg", 0.0)),
                mu_scale=float(p.get("mu_scale", 1.0)), hidden_mass=hm, placement=pl)


def _under_rim_sig(r):
    pm = r.get("payload_model_mass_kg")
    known = r.get("payload_mass") == "true" or (pm is not None and pm > PAN_LID_KG + 1e-6)
    sig = ((("L", "handle"), ("R", "rim")), float(r.get("grasp_frac", 0.6)), float(r.get("push", r.get("push_m", 0.07))),
           float(r.get("grip_force_N", D.GRIP_N)), "known" if known else "empty")
    o = r.get("observe")
    return sig + (f"obs:{o['noise']}",) if isinstance(o, dict) and o.get("noise") else sig


def obs_tag(sig):
    """'obs:<noise>' for a plan/run whose claws were aimed by the BUILT observe_pose, else None."""
    return sig[5] if len(sig) > 5 else None


def _handle_sig(r):
    if r.get("second_frac") is not None:
        h = (("L", "handle"), ("R", "handle"))
    else:
        h = (("L", "handle"),)
    return (h, float(r.get("grasp_frac", 0.6)), None, float(r.get("grip_force_N", 10.0)), "empty")


def load(extra_jsonl=()):
    out = []

    def add(sig, r, src):
        out.append(dict(sig=sig, theta=_theta(r), held=bool(r.get("held", False)), cat=category(r), src=src))

    for f in sorted(glob.glob(str(RES / "*_force100.json"))):
        r = json.load(open(f))
        add(((("L", "handle"),), 0.6, None, 10.0, "empty"), r, f)
    for pat in ("v2_*.json", "v3_*.json", "c4_*.json"):
        for f in sorted(glob.glob(str(RES / pat))):
            r = json.load(open(f))
            if r.get("grip_force_N", 10.0) == 10.0 and r.get("grip", "force") == "force":
                add(_handle_sig(r), r, f)
    for f in sorted(glob.glob(str(RES / "ur_*.json"))):
        r = json.load(open(f))
        add(_under_rim_sig(r), r, f)
    for sub in ("robust", "robust_L70", "robust_mass_known"):
        for f in sorted(glob.glob(str(RES / sub / "[AB]*.json"))):
            r = json.load(open(f))
            if sub == "robust_mass_known":
                r.setdefault("payload_mass", "true")
            add(_under_rim_sig(r), r, f)
    for f in sorted(glob.glob(str(RES.parent / "pan_observe_pose_*" / "raw" / "*.json"))):   # E2: built observe_pose, closed loop
        r = json.load(open(f))
        if r.get("observe") is not None or "observe_pose" in (r.get("error") or ""):
            r.setdefault("observe", dict(noise=r.get("noise")))
            add(_under_rim_sig(r), r, f)
    for jf in extra_jsonl:
        if Path(jf).exists():
            for line in open(jf):
                if line.strip():
                    e = json.loads(line)
                    e["sig"] = _tuplify(e["sig"])
                    out.append(e)
    return out


def _tuplify(x):
    return tuple(_tuplify(v) for v in x) if isinstance(x, list) else x


def match(db, sig, belief, failed=True):
    """recorded runs with this physical signature whose theta lies in the belief box (failed ones by default)."""
    return [e for e in db if e["sig"] == sig and belief.contains(e["theta"]) and (not e["held"]) == failed]


def match_pregrasp(db, sig, belief):
    """no-clean-pre-grasp failures generalise across push and payload: same holders + left pinch position, pose inside the box."""
    return [e for e in db if e["cat"] == "no_clean_pregrasp" and e["sig"][0] == sig[0] and e["sig"][1] == sig[1]
            and obs_tag(e["sig"]) == obs_tag(sig) and e["sig"] != sig and belief.contains(e["theta"])]
