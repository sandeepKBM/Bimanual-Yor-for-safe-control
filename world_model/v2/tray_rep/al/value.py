"""al/value.py -- the VALUE INTERFACE of AL_transductive: T(I, set-width, mu) (carry time) and drop(I, set-width, mu), plus the carry-speed rule. SIMULATION ONLY. CPU.

Interface (all a policy needs):
    V.igrid                          speeds I_rel (descending)
    V.time(I, w, mu=None)  -> s      expected time-to-goal at speed I when F5 holds a mu set of log-width w = ln(hi/lo); mu = the (hidden) true value, optional, the
                                     decision path passes a belief SAMPLE, never the truth
    V.drop(I, w, mu=None)  -> [0,1]  expected drop probability
    V.choose_I(w, eps=0.0) -> I      speed rule: among speeds with predicted drop <= eps the one with the lowest time (ties -> larger I); if none, the lowest drop
Implementations
    PlaceholderValue                 fitted to the rebaseline_v2 closed-loop F5P cells (piper_trk, TEST numbers 7.83/7.9/8.2 s ...: used only as a stand-in): time = (8.24 + 0.026 w^3.05)
                                     * (0.99 / I) [s], drop = 0 (F5 held 0 drops in every rebaseline arm). drop_mode='proxy' adds a logistic drop proxy ONLY to exercise the speed rule.
    CellsValue(cells_csv, pop, auth) the AL-0 value-of-information map: <vimap>/analysis/cells.csv (one row per pop, auth, I, set: n, drops, T_arrive, lw ...). Rows of the
                                     regime (pop, auth) with set in {oracle, narrow, medium, popr, full} give, per speed I, the points (lw -> T_arrive, drops/n); time/drop are linear
                                     interpolations in w (clamped at the ends). The 'mis' set is excluded (off-centre). mu is not used (cells average over the VAL population of the regime).
                                     voi.json (same folder) is not needed for the lookup; its path is stored for provenance.
    load_value(spec)                 'placeholder[:drop=proxy]' | 'cells:<cells.csv>[,pop=low][,auth=0.25]'  (auth and pop default to the runner's regime)
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

IGRID = (0.99, 0.75, 0.55, 0.40)
USE_SETS = ("oracle", "narrow", "medium", "popr", "full")


class ValueModel:
    igrid = IGRID
    kind = "base"

    def time(self, I, w, mu=None):
        raise NotImplementedError

    def drop(self, I, w, mu=None):
        raise NotImplementedError

    def choose_I(self, w, eps=0.0):
        d = np.array([self.drop(I, w) for I in self.igrid])
        t = np.array([self.time(I, w) for I in self.igrid])
        ok = np.flatnonzero(d <= eps + 1e-12)
        if len(ok) == 0:
            return float(self.igrid[int(np.argmin(d))])
        j = ok[np.lexsort((-np.array(self.igrid)[ok], t[ok]))[0]]       # lowest time, ties -> larger I
        return float(self.igrid[j])

    def describe(self):
        return dict(kind=self.kind)


class PlaceholderValue(ValueModel):
    kind = "placeholder"

    def __init__(self, drop_mode="zero", t0=8.24, a=0.026, p=3.05):
        self.drop_mode, self.t0, self.a, self.p = drop_mode, t0, a, p

    def time(self, I, w, mu=None):
        return (self.t0 + self.a * max(float(w), 0.0) ** self.p) * (0.99 / float(I))

    def drop(self, I, w, mu=None):
        if self.drop_mode == "zero":
            return 0.0
        # TEST FIXTURE ONLY (invented): drop rises with speed and with the set-width (a wide set is NOT the dangerous direction; this is only to exercise choose_I)
        return float(1.0 / (1.0 + np.exp(-(6.0 * (float(I) - 0.9) - 1.0 * float(w) + 1.5))))

    def describe(self):
        return dict(kind=self.kind, drop_mode=self.drop_mode, t0=self.t0, a=self.a, p=self.p)


class CellsValue(ValueModel):
    kind = "cells"

    def __init__(self, cells_csv, pop="full", auth=1.0):
        self.path, self.pop, self.auth = str(cells_csv), str(pop), float(auth)
        rows = [r for r in csv.DictReader(open(cells_csv)) if r["pop"] == self.pop and abs(float(r["auth"]) - self.auth) < 1e-9 and r["set"] in USE_SETS]
        if not rows:
            raise ValueError(f"no cells for pop={pop} auth={auth} in {cells_csv}")
        self.pts = {}
        for r in rows:
            I = float(r["I"])
            T = float(r["T_arrive"])
            if not np.isfinite(T):
                continue
            self.pts.setdefault(I, []).append((float(r["lw"]), T, int(r["drops"]) / max(int(r["n"]), 1), int(r["n"])))
        self.igrid = tuple(sorted(self.pts, reverse=True))
        self.arr = {I: np.array(sorted(v)) for I, v in self.pts.items()}
        self.voi_json = str(Path(cells_csv).with_name("voi.json"))

    def _interp(self, I, w, col):
        I = min(self.arr, key=lambda x: abs(x - float(I)))
        a = self.arr[I]
        return float(np.interp(float(w), a[:, 0], a[:, col]))

    def time(self, I, w, mu=None):
        return self._interp(I, w, 1)

    def drop(self, I, w, mu=None):
        return self._interp(I, w, 2)

    def describe(self):
        return dict(kind=self.kind, path=self.path, pop=self.pop, auth=self.auth, igrid=list(self.igrid), n_cells=sum(len(v) for v in self.pts.values()))


def load_value(spec, pop="full", auth=1.0):
    name, _, rest = (spec or "placeholder").partition(":")
    if name == "placeholder":
        kw = dict(kv.split("=") for kv in rest.split(",") if kv)
        if "drop" in kw:
            kw["drop_mode"] = kw.pop("drop")
        return PlaceholderValue(**kw)
    if name == "cells":
        parts = rest.split(",")
        path, kw = parts[0], dict(kv.split("=") for kv in parts[1:] if kv)
        return CellsValue(path, kw.get("pop", pop), float(kw.get("auth", auth)))
    raise ValueError(spec)
