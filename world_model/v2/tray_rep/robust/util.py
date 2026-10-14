"""robust/util.py -- paths, setups, small helpers shared by the robust/ modules. SIMULATION / SAVED DATA ONLY."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import env as E
from world_model.v2.tray_rep.store import read_meta

ROOT = K.ROOT                                   # /common/users/ss5772/bimanual_data/tray_rep
OUT = ROOT / "robust"
BRIDGE = ROOT / "bridge" / "main"
SETUPS = ("piper", "piper_trk")
ARMS = ("N0", "Rc3k")
SEEDS = (0, 1, 2)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def data_dir(setup):
    return ROOT / f"data_{setup}_full"


def runs_root():
    return str(ROOT / "runs")


def make_clean_env(meta):
    """env of the dataset's physics with the env-side tracker OFF: obs ball = exact [x, y, vx, vy] (the corruption is applied
    outside, by robust.corrupt). Tray part of obs is exact in every setup."""
    return E.TrayRepEnv("full", meta.get("a_scale", 1.0), bool(meta.get("lip", False)), w_scale=meta.get("w_scale"),
                        alpha=meta.get("alpha"))


def exact_ball(priv):
    """true ball columns of obs from priv rows: [x, y, vx, vy] (what the noise-free env puts in obs[9:13])."""
    p = np.asarray(priv)
    return np.concatenate([p[..., 0:2], p[..., 3:5]], -1)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(c - h), float(c + h))


def rate(k, n):
    return dict(k=int(k), n=int(n), rate=(float(k) / n if n else float("nan")), ci95=wilson(int(k), int(n)))


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass
    return out


def atomic_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    os.replace(tmp, path)


def chosen_tau(setup, arm):
    """the F3 tau the bridge VAL selection chose for (setup, arm); fallback 0.1 / 0.05."""
    p = BRIDGE / "chosen_params.json"
    try:
        ch = json.loads(p.read_text())
        spec = ch[f"{setup}/F3@{arm}"]["spec"]
        return float(spec.split("tau=")[1].split(",")[0])
    except Exception:
        return 0.1 if setup == "piper_trk" else 0.05
