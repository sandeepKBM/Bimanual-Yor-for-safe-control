"""vlm/spec_eval.py -- parse + compile + semantically probe a VLM-authored spec (dict) with tray_rep.sets.ir. SIMULATION / SAVED DATA ONLY.
Entities are world-frame; the state is a plain pose table. Needs numpy + the repo only (no torch / mujoco).

compile levels (per spec):
  json     the text contains a JSON object
  compile  ir.from_dict succeeds (names bound: entities of the scene; params H, lo, r_shrink, theta_max, r_ball)
  eval     margin(nominal state) is finite
  sem      nominal margin > 0 AND every 'must-violate' probe state has margin < 0
              probes: ball_off (ball well outside the tray/cup), tilt60 (container tilted 60 deg),
                      hit_post (tray overlapping a door post; door scenes only)
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from world_model.v2.tray_rep.sets import ir  # noqa: E402
from world_model.v2.tray_rep.sets.ir import Entity, Pose, Point, BoxShape, prism_xy  # noqa: E402
from world_model.v2.tray_rep.sets.tray_spec import TRAY_SPEC_FULL  # noqa: E402

HX, HY, TT = 0.15, 0.10, 0.003
PARAMS = dict(H=1, lo=0, r_shrink=0.0, theta_max=0.3, r_ball=0.03)   # K=1 state: horizon window collapses to the single tick


class WS:
    def __init__(self, poses):
        self.poses = poses


def _R(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _ent(name, shape, key=None, parent=None):
    k = key or name
    return Entity(name, shape, lambda s, k=k: Pose(*s.poses[k]), None, parent)


def entities(kind):
    I = np.eye(3)
    if kind in ("tray_ball", "tray_door"):
        e = {"tray": _ent("tray", BoxShape([HX, HY, TT])), "tray_top": _ent("tray_top", prism_xy(HX, HY), "tray"),
             "ball": _ent("ball", Point()), "target_table": _ent("target_table", prism_xy(0.2, 0.2))}
        if kind == "tray_door":
            e["door_post_left"] = _ent("door_post_left", BoxShape([0.04, 0.03, 0.55]))
            e["door_post_right"] = _ent("door_post_right", BoxShape([0.04, 0.03, 0.55]))
        return e
    if kind == "cup_beads":
        return {"cup": _ent("cup", BoxShape([0.03, 0.03, 0.04])), "cup_interior": _ent("cup_interior", prism_xy(0.0285, 0.0285), "cup"),
                "bead": _ent("bead", Point()), "target_table": _ent("target_table", prism_xy(0.12, 0.12))}
    raise KeyError(kind)


def states(kind):
    """name -> (WS, must_violate: bool)."""
    I = np.eye(3)
    z0 = np.array([0.2, 0.0, 1.0])
    if kind in ("tray_ball", "tray_door"):
        base = dict(tray=(I, z0), ball=(I, z0 + [0, 0, 0.03]), target_table=(I, np.array([0.85, 0, 0.99])),
                    door_post_left=(I, np.array([0.55, 0.25, 0.55])), door_post_right=(I, np.array([0.55, -0.25, 0.55])))
        out = {"nominal": (WS(base), False)}
        b = dict(base); b["ball"] = (I, z0 + [0.30, 0, 0.03]); out["ball_off"] = (WS(b), True)
        b = dict(base); b["tray"] = (_R("x", np.pi / 3), z0); out["tilt60"] = (WS(b), True)
        if kind == "tray_door":
            b = dict(base); b["tray"] = (I, np.array([0.50, 0.17, 1.0])); b["ball"] = (I, np.array([0.5, 0.17, 1.03]))
            out["hit_post"] = (WS(b), True)
        return out
    z0 = np.array([0.0, 0.0, 1.04])
    base = dict(cup=(I, z0), bead=(I, z0 + [0, 0, -0.02]), target_table=(I, np.array([0.7, 0, 0.998])))
    out = {"nominal": (WS(base), False)}
    b = dict(base); b["bead"] = (I, z0 + [0.08, 0, 0.0]); out["ball_off"] = (WS(b), True)
    b = dict(base); b["cup"] = (_R("x", np.pi / 3), z0); out["tilt60"] = (WS(b), True)
    return out


def extract_json(text: str):
    """LAST top-level JSON object in text (after stripping <think> blocks; repo rule 'parsers take the LAST JSON object': reasoning models
    emit drafts first; 11-03 item 12d, was the first); None if absent / unparsable.  Same scan as contents_eval.parse_last_json."""
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.rsplit("</think>", 1)[-1]
    dec, best, i = json.JSONDecoder(), None, 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            o, k = dec.raw_decode(text, j)
            if isinstance(o, dict):
                best = o
            i = k
        except ValueError:
            i = j + 1
    return best


def atoms(d, out=None):
    """multiset of (atom kind, a, b) in a spec dict (for the structural comparison with the hand spec)."""
    out = [] if out is None else out
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if k in ("and", "or"):
            for c in v:
                atoms(c, out)
        elif k == "not":
            atoms(v, out)
        elif k in ("G", "F"):
            atoms(v.get("of"), out)
        elif k in ("inside", "upright", "clear", "support"):
            def nm(x):
                return x if isinstance(x, str) else (nm(x["shrink"]["of"]) if isinstance(x, dict) and "shrink" in x else "?")
            out.append((k, nm(v.get("a")), nm(v.get("b")) if "b" in v else None))
    return out


def check(spec_dict, kind, params=None):
    """returns dict(compile, eval, sem, margins{state: m}, error)."""
    p = dict(PARAMS, **(params or {}))
    r = dict(compile=False, eval=False, sem=False, margins={}, error=None)
    try:
        ents = entities(kind)
        node = ir.from_dict(spec_dict, ents, p)
        r["compile"] = True
        sts = states(kind)
        for n, (s, must_viol) in sts.items():
            r["margins"][n] = float(ir.margin(node, s, False))
        r["eval"] = bool(all(np.isfinite(v) for v in r["margins"].values()))
        r["sem"] = bool(r["eval"] and r["margins"]["nominal"] > 0 and
                        all(r["margins"][n] < 0 for n, (_, mv) in sts.items() if mv))
        r["probes_caught"] = {n: bool(r["margins"][n] < 0) for n, (_, mv) in sts.items() if mv}
    except Exception as e:  # noqa: BLE001
        r["error"] = f"{type(e).__name__}: {str(e)[:120]}"
    return r


HAND = {"tray_ball": TRAY_SPEC_FULL}

if __name__ == "__main__":
    for kind in ("tray_ball", "tray_door", "cup_beads"):
        print(kind, list(states(kind)))
    print(check(TRAY_SPEC_FULL, "tray_ball"))
    print(check(TRAY_SPEC_FULL, "tray_door"))
    door = {"and": [TRAY_SPEC_FULL, {"clear": {"a": "tray", "b": "door_post_left", "d": 0.03}},
                    {"clear": {"a": "tray", "b": "door_post_right", "d": 0.03}}]}
    print(check(door, "tray_door"))
    print(check({"G": {"lo": 0, "hi": "H", "of": {"and": [{"inside": {"a": "bead", "b": {"shrink": {"of": "cup_interior", "by": 0.0}}}},
                                                          {"upright": {"a": "cup", "theta_max": 0.5}}]}}}, "cup_beads"))
