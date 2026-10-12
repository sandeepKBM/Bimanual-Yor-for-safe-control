"""al/probe_voi.py -- pre-posterior table of the pickup probe for AL_transductive. SIMULATION ONLY. CPU.

build_voi.py runs the REAL probe (MuJoCo, the regime's authority, same ProbeController) on VAL carries with the hidden mu forced to a grid of levels and stores, per tilt theta
and level, the raw probe log-likelihood vector over the 40 mu nodes (v2 map), the probe duration (ticks) and whether the ball dropped. This is OFFLINE calibration data (it
uses the true mu to label the runs); the policy only ever reads the table, never a carry's mu.
ProbeVOI.entries(theta, w0): the enumeration the decision integrates over: true level j ~ the belief w0 (mass of the nodes nearest to level j) x every stored repeat at (theta, j)
-> [(weight, ll (40,), ticks, dropped)].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.sets.vlmprior import online_mu as OM


class ProbeVOI:
    def __init__(self, d):
        self.d = d
        self.levels = np.array(d["mu_levels"], float)
        self.thetas = sorted(float(t) for t in d["entries"])
        self.node_level = np.array([int(np.argmin(np.abs(np.log(self.levels) - l))) for l in OM.L_NODES])      # nearest level of each node

    @staticmethod
    def load(path):
        return ProbeVOI(json.loads(Path(path).read_text()))

    def key(self, theta):
        t = min(self.thetas, key=lambda x: abs(x - float(theta)))
        return next(k for k in self.d["entries"] if abs(float(k) - t) < 1e-12)

    def entries(self, theta, w0):
        ent = self.d["entries"][self.key(theta)]
        out = []
        w0 = np.asarray(w0, float)
        for j in range(len(self.levels)):
            wj = float(w0[self.node_level == j].sum())
            reps = ent.get(str(j), [])
            if wj <= 0 or not reps:
                continue
            for r in reps:
                out.append((wj / len(reps), None if r["ll"] is None else np.array(r["ll"], float), int(r["ticks"]), bool(r["dropped"])))
        return out
