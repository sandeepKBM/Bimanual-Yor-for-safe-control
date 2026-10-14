"""grounding/common.py -- paths, per-model NATIVE grounding prompts + output parsers, mask/box/point metrics. SIMULATION images only.

Model families (native conventions, see docs in each fn):
  qwen   Qwen3-VL-8B-Instruct, Qwen3.5-9B : JSON {"point_2d":[x,y]} / {"bbox_2d":[x1,y1,x2,y2]}, coords RELATIVE 0-1000 of the image sent
  rb     RoboBrain2.5-4B (inference.py of FlagOpen/RoboBrain2.5): pointing "[(x, y)]", grounding "[x1, y1, x2, y2]"; coords RELATIVE 0-1000
  molmo  Molmo2-ER : "Point to ..." -> <points coords="id img x y ;..."> (Molmo2 scales x,y by 1000 -> verified empirically vs masks)
Every parser returns coords as the model wrote them; to_px() applies the convention. Scoring also evaluates the alternative convention
(absolute pixels / percent) as a sanity flag, never silently.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("GROUNDING_DATA", "/common/users/ss5772/bimanual_data/vlm/grounding"))
FAMILY = {"qwen3-vl-8b": "qwen", "qwen3.5-9b": "qwen", "robobrain2.5-4b": "rb", "robobrain2.0-7b": "rb", "molmo2-er": "molmo",
          "cosmos-reason1-7b": "qwen"}
IMG_W, IMG_H = 1280, 720


# ------------------------------------------------------------------------------------------------ prompts
def prompt(model: str, task: str, phrase: str) -> str:
    """task: 'point' | 'box'. phrase: e.g. 'red mug' or 'mug left of the bottle' (no leading article)."""
    fam = FAMILY[model]
    p = phrase
    if fam == "qwen":
        if task == "point":
            return (f"Point to the {p} in the image. Output the point coordinates in JSON format: "
                    f'[{{"point_2d": [x, y], "label": "<name>"}}]')
        return (f"Locate the {p} in the image and output its bounding box in JSON format: "
                f'[{{"bbox_2d": [x1, y1, x2, y2], "label": "<name>"}}]')
    if fam == "rb":
        if task == "point":
            return (f"the {p}. Please provide its 2D coordinates. Your answer should be formatted as a tuple, i.e. [(x, y)], "
                    f"where the tuple contains the x and y coordinates of a point satisfying the conditions above.")
        return f"Please provide the bounding box coordinate of the region this sentence describes: the {p}."
    if fam == "molmo":
        if task == "point":
            return f"Point to the {p}."
        return f"Draw a bounding box around the {p}. Give its coordinates as x1, y1, x2, y2."   # NOT a documented Molmo2 skill
    raise KeyError(model)


# ------------------------------------------------------------------------------------------------ parsers
_NUM = r"-?\d+(?:\.\d+)?"


def parse_points(model: str, text: str):
    """-> list of (x, y) floats as written (first = the answer)."""
    fam = FAMILY[model]
    out = []
    if fam == "molmo":
        for m in re.finditer(rf'<points?\s+([^>]*)>', text):
            a = m.group(1)
            mc = re.search(r'coords="([^"]*)"', a)
            if mc:
                for item in re.split(r"\s*;\s*", mc.group(1).strip()):
                    v = re.findall(_NUM, item)
                    if len(v) >= 4:                      # id img x y   (Molmo2 multi-image format)
                        out.append((float(v[-2]), float(v[-1])))
                    elif len(v) == 3:                    # id x y
                        out.append((float(v[-2]), float(v[-1])))
                    elif len(v) == 2:
                        out.append((float(v[0]), float(v[1])))
            else:
                for mx, my in re.findall(rf'x\d*="({_NUM})"\s*y\d*="({_NUM})"', a):
                    out.append((float(mx), float(my)))
        return out
    if fam == "qwen":
        for m in re.finditer(rf'"point_2d"\s*:\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*\]', text):
            out.append((float(m.group(1)), float(m.group(2))))
        if out:
            return out
    for m in re.finditer(rf'\(\s*({_NUM})\s*,\s*({_NUM})\s*\)', text):
        out.append((float(m.group(1)), float(m.group(2))))
    if not out:
        for m in re.finditer(rf'\[\s*({_NUM})\s*,\s*({_NUM})\s*\]', text):
            out.append((float(m.group(1)), float(m.group(2))))
    return out


def parse_boxes(model: str, text: str):
    """-> list of (x1, y1, x2, y2) as written."""
    out = []
    if FAMILY[model] == "molmo":                 # Molmo2: <boxes coords="id img x1 y1 x2 y2 ; id x1 y1 x2 y2 ...">; free-text variants seen in practice
        for m in re.finditer(r'<boxes\s+coords="([^"]*)"', text):
            v = [float(x) for x in re.findall(_NUM, m.group(1))]
            if len(v) >= 6:
                out.append(tuple(v[2:6]))
        for m in re.finditer(rf'x1\s+({_NUM})\s+({_NUM})\s+x2\s+({_NUM})\s+({_NUM})', text):
            out.append(tuple(float(g) for g in m.groups()))
        for m in re.finditer(rf'\[\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*,\s*\(\s*({_NUM})\s*,\s*({_NUM})\s*\)\s*\]', text):
            g = [float(x) for x in m.groups()]
            out.append(tuple(x * 1000 for x in g) if max(g) <= 1.0 else tuple(g))     # two corners, relative 0-1
        if out:
            return out
    for m in re.finditer(rf'"bbox_2d"\s*:\s*\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]', text):
        out.append(tuple(float(g) for g in m.groups()))
    if out:
        return out
    for m in re.finditer(rf'\[\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\]', text):
        out.append(tuple(float(g) for g in m.groups()))
    if not out:
        for m in re.finditer(rf'\(\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*,\s*({_NUM})\s*\)', text):
            out.append(tuple(float(g) for g in m.groups()))
    return out


def to_px(model: str, v, W=IMG_W, H=IMG_H, conv="native"):
    """v = (x, y) or (x1, y1, x2, y2) as written -> pixels. conv: 'native' | 'abs' (absolute pixels)."""
    v = np.asarray(v, float)
    if conv == "abs":
        return v
    s = np.array([W, H] * (len(v) // 2)) / 1000.0        # all three families: relative 0-1000 (molmo verified empirically)
    return v * s


# ------------------------------------------------------------------------------------------------ metrics
def mask_bbox(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def box_iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(ix2 - ix1, 0.0) * max(iy2 - iy1, 0.0)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def point_in_mask(mask, p):
    x, y = int(np.floor(p[0])), int(np.floor(p[1]))
    return bool(0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and mask[y, x])


def write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]
