"""lang/grounding.py -- set-of-marks grounding: perceived zonotopes -> numbered marks on the head-camera RGB -> VLM names them.  SIMULATION / SAVED DATA ONLY.

    props = proposals_from_result(res)                 # res = a flagship perception result (res['left'] RGB, K, origin, R, zonos[...])
    g = ground(backend, res['left'], props, out_dir)   # -> Grounding(names {zono id: (name, conf)}, groups, marks image, latency)
    score_grounding(g, gt_keys={zono id: gt_key})      # EVALUATION ONLY: gt keys are never an input to the VLM

Marks: one numbered box per perceived zonotope (the projected hull of its 8..2^m vertices, uninflated Z), colour-coded, label chip at the top-left corner.
Only 'obj' zonotopes with enough points, a visible projection and >= MIN_BOX_PX pixels get a mark (support planes are the surface itself, not an entity).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from world_model.v2.vlm.spec_eval import extract_json

PALETTE = [(255, 99, 71), (65, 190, 255), (120, 230, 90), (255, 200, 40), (210, 120, 255), (60, 235, 200), (255, 130, 190), (150, 180, 255),
           (240, 150, 60), (180, 240, 90)]
MIN_PTS, MIN_BOX_PX, MAX_MARKS = 40, 12, 24
RESERVED = {"tray", "tray_top", "cup", "ball", "H", "lo", "r_shrink", "theta_max", "r_ball"}
UNKNOWN = {"unknown", "unsure", "unclear", "none", "n/a", "na", "object", "thing", "item", "unidentified", ""}


@dataclass
class Proposal:
    id: int                       # perception zonotope id (res['zonos'][i]['id'])
    c: np.ndarray
    G: np.ndarray                 # nominal zonotope (uninflated)
    ci: np.ndarray                # inflated (Zi) zonotope: what the clearance rows / monitor use
    Gi: np.ndarray
    box: tuple                    # (x0, y0, x1, y1) px, clipped to the image
    n_pts: int = 0
    range_m: float = 0.0
    kind: str = "obj"


@dataclass
class Grounding:
    names: dict                   # zono id -> (normalised name, confidence)
    raw_names: dict               # zono id -> name as the VLM wrote it
    props: list
    groups: dict = field(default_factory=dict)       # name -> [zono ids]
    marks_png: str | None = None
    texts: list = field(default_factory=list)
    seconds: float = 0.0
    wall_s: float = 0.0
    cached: bool = False
    parse_ok: bool = True
    log: list = field(default_factory=list)

    def summary(self):
        return {k: [(i, self.names[i][1]) for i in v] for k, v in self.groups.items()}


# ------------------------------------------------------------------------------------------------------------------ proposals
def proposals_from_result(res, max_marks=MAX_MARKS, min_pts=MIN_PTS, min_box_px=MIN_BOX_PX, kinds=("obj",)):
    """one Proposal per usable perceived zonotope of a flagship perception result, largest projected box first, capped at max_marks."""
    from world_model.v2.house.zed_demo import stereo as ST, perceive as PC
    from world_model.v2.tray_rep.sets.zono import Zonotope
    K, origin, R = res["K"], np.asarray(res["origin"]), np.asarray(res["R"])
    W, H = int(K["W"]), int(K["H"])
    out = []
    for z in res["zonos"]:
        if z["kind"] not in kinds or z["n_pts"] < min_pts:
            continue
        Zd = z["Z"]
        V = PC.zono_vertices(Zonotope(np.asarray(Zd["c"], float), np.asarray(Zd["G"], float)))
        uv, d = ST.project(V, K, origin, R)
        if (d < 0.05).any() or not np.isfinite(uv).all():
            vis = (d >= 0.05) & np.isfinite(uv).all(1)
            if vis.sum() < 4:
                continue
            uv = uv[vis]
        x0, y0 = np.clip(uv.min(0), 0, [W - 1, H - 1])
        x1, y1 = np.clip(uv.max(0), 0, [W - 1, H - 1])
        if x1 - x0 < min_box_px or y1 - y0 < min_box_px:
            continue
        Zi = z.get("Zi", Zd)
        out.append(Proposal(int(z["id"]), np.asarray(Zd["c"], float), np.asarray(Zd["G"], float), np.asarray(Zi["c"], float), np.asarray(Zi["G"], float),
                            (float(x0), float(y0), float(x1), float(y1)), int(z["n_pts"]), float(z.get("range_m", 0.0)), z["kind"]))
    out.sort(key=lambda p: -(p.box[2] - p.box[0]) * (p.box[3] - p.box[1]))
    return out[:max_marks]


# ------------------------------------------------------------------------------------------------------------------ marks
def render_marks(img, props, path=None):
    """numbered boxes on a copy of the RGB uint8 image; marks are numbered 1..N in proposal order (NOT the zonotope ids: the VLM sees small integers)."""
    import cv2
    im = np.ascontiguousarray(img[:, :, :3]).copy()
    for k, p in enumerate(props, 1):
        col = PALETTE[(k - 1) % len(PALETTE)]
        x0, y0, x1, y1 = [int(round(v)) for v in p.box]
        cv2.rectangle(im, (x0, y0), (x1, y1), col, 2, cv2.LINE_AA)
        txt = str(k)
        (tw, th), bl = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cx0, cy0 = x0, max(y0 - th - 6, 0)
        cv2.rectangle(im, (cx0, cy0), (cx0 + tw + 8, cy0 + th + 6), col, -1)
        cv2.putText(im, txt, (cx0 + 4, cy0 + th + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    if path is not None:
        cv2.imwrite(str(path), im[:, :, ::-1])
    return im


# ------------------------------------------------------------------------------------------------------------------ prompt / parse
SYSTEM = ("You label objects seen by the head camera of a mobile manipulator. The image has NUMBERED coloured boxes (a number chip at the top-left corner of each "
          "box). Each box was proposed by a depth-based segmenter; a box may cover a whole object, only a PART of an object or of furniture, or background.")


def user_prompt(ids):
    return ("Name what is inside each numbered box with a short common noun (examples: bottle, monitor, laptop, mug, lamp, book, box, person, desk, "
            "drawer, floor, wall, robot arm). If a box only covers part of furniture or the floor, say 'desk part' / 'floor' etc. If you cannot tell, write "
            "\"unknown\" with a low confidence.\nMarks to name: [" + ", ".join(str(i) for i in ids) + "]\n"
            "Answer with ONE LINE per mark and nothing else, in exactly this format:\n<number> | <name> | <confidence 0..1>\nfor example:\n1 | bottle | 0.9\n2 | desk part | 0.6")


def norm_name(s):
    s = re.sub(r"[^a-z0-9]+", "_", str(s).lower().strip()).strip("_")
    s = re.sub(r"^(the|a|an|my|some)_", "", s)
    if s.endswith("ies") and len(s) > 4:
        s = s[:-3] + "y"
    elif s.endswith("s") and not s.endswith(("ss", "us")) and len(s) > 3:
        s = s[:-1]
    s = "_".join(s.split("_")[:3])
    return s


def parse_marks(text, n_marks):
    """-> {mark number: (raw name, conf)}.  Accepts lines '<n> | <name> | <conf>' (preferred: fewest tokens), {"marks": [[n, name, c], ...]}, {"marks": [{"id":..,"name":..,"confidence":..}]} or {"1": "bottle", ...}."""
    out = {}
    for ln in text.splitlines():                       # line format: "<n> | <name> | <conf>"
        m = re.match(r"^\s*[-*]?\s*(?:mark\s*)?(\d+)\s*[|:,]\s*([^|]*?)\s*(?:[|,]\s*([01](?:\.\d+)?)\s*)?$", ln.strip(), re.I)
        if m and 1 <= int(m.group(1)) <= n_marks:
            out[int(m.group(1))] = (m.group(2), float(m.group(3)) if m.group(3) else 0.5)
    if out:
        return out
    j = extract_json(text)
    if j is None:
        return out
    items = j.get("marks", j) if isinstance(j, dict) else j
    if isinstance(items, dict):
        items = [[k, (v if not isinstance(v, dict) else v.get("name", "")), (v.get("confidence", 0.5) if isinstance(v, dict) else 0.5)] for k, v in items.items()]
    for it in items if isinstance(items, list) else []:
        try:
            if isinstance(it, dict):
                k, nm, c = it.get("id", it.get("mark")), it.get("name", ""), it.get("confidence", 0.5)
            else:
                k, nm, c = it[0], it[1], (it[2] if len(it) > 2 else 0.5)
            k = int(k)
            c = float(c)
            if 1 <= k <= n_marks and isinstance(nm, str):
                out[k] = (nm, min(max(c, 0.0), 1.0))
        except Exception:    # noqa: BLE001
            continue
    return out


def ground(backend, img, props, out_dir=None, tag="ground", max_new_tokens=700):
    """name the marks.  out_dir (optional) receives <tag>_marks.png.  Never raises on a bad VLM answer: unparsable marks become 'unknown' (conf 0)."""
    marks = render_marks(img, props)
    png = None
    tmp_dir = Path(out_dir) if out_dir else Path("/tmp")
    if out_dir is not None:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        png = str(tmp_dir / f"{tag}_marks.png")
        import cv2
        cv2.imwrite(png, marks[:, :, ::-1])
    else:
        import cv2, tempfile
        fd = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        png = fd.name
        fd.close()
        cv2.imwrite(png, marks[:, :, ::-1])
    n = len(props)
    o = backend.ask(user_prompt(range(1, n + 1)), images=[png], system=SYSTEM, max_new_tokens=max_new_tokens, temperature=0.0)
    txt = o["texts"][0] if o["texts"] else ""
    parsed = parse_marks(txt, n)
    names, raw, log = {}, {}, []
    for k, p in enumerate(props, 1):
        if k in parsed:
            r, c = parsed[k]
            nm = norm_name(r)
            if nm in UNKNOWN:
                nm, c = "unknown", min(c, 0.3)
            if nm in RESERVED:
                nm += "_obj"
            names[p.id], raw[p.id] = (nm, c), r
        else:
            names[p.id], raw[p.id] = ("unknown", 0.0), ""
            log.append(f"mark {k} (zono {p.id}) not in the VLM answer -> unknown")
    groups = {}
    for p in props:
        nm = names[p.id][0]
        if nm != "unknown":
            groups.setdefault(nm, []).append(p.id)
    return Grounding(names=names, raw_names=raw, props=props, groups=groups, marks_png=png, texts=[txt], seconds=float(o.get("seconds") or 0.0),
                     wall_s=float(o.get("wall_s") or 0.0), cached=bool(o.get("cached")), parse_ok=bool(parsed), log=log)


# ------------------------------------------------------------------------------------------------------------------ scoring (EVALUATION ONLY)
STRUCT = {"desk", "floor", "wall", "counter", "desk_top"}
SYN = {
    "monitor": {"monitor", "screen", "display", "tv", "television", "computer", "laptop"},
    "desk": {"desk", "table", "cabinet", "drawer", "furniture", "shelf", "dresser", "counter", "cupboard", "leg", "floor", "carpet", "rug", "surface", "bench", "wood", "panel", "part"},
    "box": {"box", "cube", "block", "package", "crate", "case", "brick"},
    "book": {"book", "binder", "notebook", "folder", "box", "case"},
    "pencup": {"cup", "mug", "pencil", "pen", "holder", "container", "can", "cylinder", "jar", "bin", "canister"},
    "mug": {"mug", "cup", "teacup", "coffee"},
    "bottle": {"bottle", "vase", "flask", "jug", "bin", "cylinder", "pillar"},
    "lamp": {"lamp", "light", "lantern"},
    "milk_jug": {"jug", "bottle", "container", "pitcher", "jar", "can", "milk", "carton"},
    "sugar_jar": {"jar", "can", "container", "bottle", "canister", "sugar", "pot"},
    "napkin_box": {"box", "tissue", "napkin", "dispenser", "container"},
    "coffee_machine": {"coffee", "machine", "maker", "espresso", "appliance", "kettle", "dispenser", "oven"},
    "cup_stack": {"cup", "stack", "glass", "tower", "cylinder", "bottle", "can"},
    "counter": {"counter", "desk", "table", "cabinet", "furniture", "shelf", "cupboard", "surface", "wall", "panel", "part", "kitchen"},
    "floor": {"floor", "carpet", "rug", "ground", "tile", "desk", "wall", "part"},
    "wall": {"wall", "curtain", "panel", "floor", "door", "window", "part"},
}


def gt_class(key):
    """gt key -> class ('desk_1' -> 'desk', 'mug_desk1' -> 'mug', 'monitor_1' -> 'monitor', 'milk_jug' -> 'milk_jug')."""
    k = str(key)
    if k in SYN:
        return k
    k = re.sub(r"_(desk|counter)?[a-z]?\d*$", "", k) if k not in SYN else k
    k = re.sub(r"_[a-z0-9]$", "", k)
    return k


def _tok(name):
    return set(re.split(r"[^a-z0-9]+", name.lower())) - {""}


def match(name, key):
    """(lenient, exact): lenient = any name token in the class synonym set; exact = the class base token itself appears in the name."""
    c = gt_class(key)
    toks = _tok(name)
    syn = SYN.get(c, set()) | set(c.split("_"))
    exact = bool(toks & set(c.split("_")))
    return bool(toks & syn), exact


def score_grounding(g: Grounding, gt_keys):
    """EVALUATION ONLY.  gt_keys: {zono id: gt key string}.  Accuracy over the marked zonotopes; 'distinct' excludes the structural classes
    (desk / floor / wall / counter fragments, whose naming is inherently fuzzy).  Unknown counts as wrong (abstentions reported separately)."""
    rows = []
    for p in g.props:
        key = gt_keys.get(p.id)
        nm, conf = g.names[p.id]
        if key is None:
            continue
        ok, ex = (False, False) if nm == "unknown" else match(nm, key)
        rows.append(dict(id=p.id, gt=key, gt_class=gt_class(key), name=nm, raw=g.raw_names.get(p.id), conf=conf, lenient=ok, exact=ex,
                         structural=gt_class(key) in STRUCT, n_pts=p.n_pts))
    def acc(rs, k):
        return (sum(r[k] for r in rs) / len(rs)) if rs else None
    dist = [r for r in rows if not r["structural"]]
    # per-OBJECT view: an object (gt key) counts as found if ANY of its fragments is named right
    objs = {}
    for r in dist:
        objs.setdefault(r["gt"], []).append(r["lenient"])
    return dict(n_marks=len(rows), n_unknown=sum(r["name"] == "unknown" for r in rows),
                acc_lenient=acc(rows, "lenient"), acc_exact=acc(rows, "exact"),
                n_distinct=len(dist), acc_distinct_lenient=acc(dist, "lenient"), acc_distinct_exact=acc(dist, "exact"),
                objects_found=sum(any(v) for v in objs.values()), objects_total=len(objs),
                objects_all_fragments_right=sum(all(v) for v in objs.values()), rows=rows)
