"""stack/theta.py -- belief sets (unions of axis-aligned boxes), affine reading models, recovery programs with reading bins, and the
Admission container (checker design 2026-11-05 section 2 + 4.3, checker step K1).  SIMULATION ONLY.

Pure: stdlib + numpy + stack.types.  No physics.  Everything here is exact for unions of boxes and affine readings, except
`bin_sets` with a reading that depends on >= 2 fields, which returns a sound SUPERSET (per-cell bounding box) and says so (exact=False).

Bounds rule: the only numeric constant is a numerical tolerance (TOL).  Reading noise carries its own tag (`noise_tag`, `noise_src`).
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from world_model.v2.stack.types import Source, ThetaBox

TOL = 1e-12                      # ASSUMED numerical tolerance: box sides below this are measure-zero slivers and are dropped
SOURCES = ("prior", "vlm_B", "probe", "vlm_outside_B")
NOISE_TAGS = ("MEASURED", "MODEL", "ASSUMED")
_TRUTH_NAMES = ("truth", str(Source.TRUTH.value).lower())   # stack/producers.py names its sim-truth producer "truth"; Source.TRUTH = "truth"


# ----------------------------------------------------------------------------------------------------------------------- interval
@dataclass(frozen=True)
class Interval:
    lo: float
    hi: float

    def __post_init__(self):
        if not (self.lo <= self.hi):            # also rejects NaN
            raise ValueError(f"empty/invalid interval [{self.lo}, {self.hi}]")

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def contains(self, x: float) -> bool:
        return self.lo <= x <= self.hi

    def intersect(self, o: "Interval") -> "Interval | None":
        lo, hi = max(self.lo, o.lo), min(self.hi, o.hi)
        return Interval(lo, hi) if lo <= hi else None

    def to_json(self) -> list:
        return [self.lo, self.hi]

    @staticmethod
    def from_json(d) -> "Interval":
        return Interval(float(d[0]), float(d[1]))


# ----------------------------------------------------------------------------------------------------------------------- cell
@dataclass(frozen=True)
class Cell:
    """one axis-aligned box in theta coordinates."""
    bounds: dict
    source: str = "prior"
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(f"unknown cell source {self.source!r}; have {SOURCES}")
        if not self.bounds:
            raise ValueError("a Cell needs at least one field")

    @property
    def fields(self) -> tuple:
        return tuple(sorted(self.bounds))

    def contains_point(self, p: dict) -> bool:
        return all(self.bounds[f].contains(p[f]) for f in self.bounds)

    def corners(self) -> list:
        fs = self.fields
        return [dict(zip(fs, v)) for v in itertools.product(*[sorted({self.bounds[f].lo, self.bounds[f].hi}) for f in fs])]

    def intersect(self, o: "Cell") -> "Cell | None":
        if set(self.bounds) != set(o.bounds):
            raise ValueError(f"cell fields differ: {self.fields} vs {o.fields}")
        out = {}
        for f, iv in self.bounds.items():
            j = iv.intersect(o.bounds[f])
            if j is None:
                return None
            out[f] = j
        return Cell(out, self.source, dict(self.provenance))

    def with_bounds(self, bounds: dict) -> "Cell":
        return Cell(bounds, self.source, dict(self.provenance))

    def to_json(self) -> dict:
        return dict(bounds={f: iv.to_json() for f, iv in self.bounds.items()}, source=self.source, provenance=_jsonable(self.provenance))

    @staticmethod
    def from_json(d: dict) -> "Cell":
        return Cell({f: Interval.from_json(v) for f, v in d["bounds"].items()}, d.get("source", "prior"), dict(d.get("provenance", {})))


def _jsonable(x):
    try:
        json.dumps(x)
        return x
    except (TypeError, ValueError):
        return repr(x)


def _minus_cell(a: Cell, b: Cell) -> list:
    """a \\ b as disjoint boxes (a's source/provenance kept).  b overlapping a only in a sliver (< TOL) removes nothing (a kept whole)."""
    ov = {}
    for f, ia in a.bounds.items():
        j = ia.intersect(b.bounds[f])
        if j is None:
            return [a]
        if ia.lo < ia.hi and j.width <= TOL:      # proper dim: need a real overlap
            return [a]
        ov[f] = j
    pieces, cur = [], dict(a.bounds)
    for f in a.fields:
        ia, j = cur[f], ov[f]
        if j.lo - ia.lo > TOL:
            pieces.append(a.with_bounds({**cur, f: Interval(ia.lo, j.lo)}))
        if ia.hi - j.hi > TOL:
            pieces.append(a.with_bounds({**cur, f: Interval(j.hi, ia.hi)}))
        cur[f] = j
    return pieces


# ----------------------------------------------------------------------------------------------------------------------- theta set
@dataclass(frozen=True)
class ThetaSet:
    """finite union of axis-aligned boxes over the same field names."""
    cells: tuple = ()

    def __post_init__(self):
        object.__setattr__(self, "cells", tuple(self.cells))
        if self.cells:
            f0 = self.cells[0].fields
            for c in self.cells:
                if c.fields != f0:
                    raise ValueError(f"ThetaSet cells over different fields: {f0} vs {c.fields}")

    @property
    def fields(self) -> tuple:
        return self.cells[0].fields if self.cells else ()

    @staticmethod
    def box(source: str = "prior", provenance: dict | None = None, **bounds) -> "ThetaSet":
        return ThetaSet((Cell({f: Interval(float(lo), float(hi)) for f, (lo, hi) in bounds.items()}, source, dict(provenance or {})),))

    @staticmethod
    def from_thetabox(tb: ThetaBox, source: str = "prior") -> "ThetaSet":
        """ThetaBox.fields = name -> Bound(lo, hi, tag, src, unit); the tag/src/unit go to provenance."""
        c = Cell({k: Interval(float(b.lo), float(b.hi)) for k, b in tb.fields.items()}, source,
                 {"bounds": {k: dict(tag=b.tag, src=b.src, unit=b.unit) for k, b in tb.fields.items()}})
        return ThetaSet((c,))

    def is_empty(self) -> bool:
        return not self.cells

    def _check(self, o: "ThetaSet"):
        if self.cells and o.cells and self.fields != o.fields:
            raise ValueError(f"ThetaSet fields differ: {self.fields} vs {o.fields}")

    def intersect(self, o: "ThetaSet") -> "ThetaSet":
        self._check(o)
        out = []
        for a in self.cells:
            for b in o.cells:
                c = a.intersect(b)
                if c is not None and all(iv.width > TOL or (iv.width == 0 and a.bounds[f].width == 0) for f, iv in c.bounds.items()):
                    out.append(c)
        return ThetaSet(tuple(out))

    def minus(self, o: "ThetaSet") -> "ThetaSet":
        self._check(o)
        cur = list(self.cells)
        for b in o.cells:
            nxt = []
            for a in cur:
                nxt.extend(_minus_cell(a, b))
            cur = nxt
        return ThetaSet(tuple(cur))

    def union(self, o: "ThetaSet") -> "ThetaSet":
        self._check(o)
        return ThetaSet(self.cells + o.minus(self).cells)        # disjoint: self + (o \ self)

    def contains(self, o: "ThetaSet") -> bool:
        return o.minus(self).is_empty()

    def contains_point(self, p: dict) -> bool:
        return any(c.contains_point(p) for c in self.cells)

    def corners(self) -> np.ndarray:
        fs = self.fields
        rows = [[c[f] for f in fs] for cell in self.cells for c in cell.corners()]
        return np.unique(np.array(rows, float).reshape(-1, len(fs)), axis=0) if rows else np.zeros((0, len(fs)))

    def to_json(self) -> dict:
        return dict(cells=[c.to_json() for c in self.cells])

    @staticmethod
    def from_json(d: dict) -> "ThetaSet":
        return ThetaSet(tuple(Cell.from_json(c) for c in d["cells"]))


# ----------------------------------------------------------------------------------------------------------------------- reading model
@dataclass(frozen=True)
class ReadingSpec:
    """y = offset + sum_f coef[f] * theta[f] + n,  |n| <= noise (tagged noise_tag: MEASURED / MODEL / ASSUMED; noise_src = where from)."""
    producer: str
    statistic: str
    coef: dict
    offset: float
    noise: float
    noise_tag: str
    noise_src: str

    def __post_init__(self):
        if str(self.producer).lower() in _TRUTH_NAMES or str(self.producer).lower().startswith("truth"):
            raise ValueError(f"reading producer {self.producer!r} is a TRUTH source (privileged); a reading must come from a sensor")
        if self.noise_tag not in NOISE_TAGS:
            raise ValueError(f"noise_tag {self.noise_tag!r} not in {NOISE_TAGS}")
        if not (self.noise >= 0):
            raise ValueError("noise must be >= 0")

    def h_range(self, bounds: dict) -> Interval:
        lo = hi = self.offset
        for f, c in self.coef.items():
            iv = bounds[f]
            a, b = c * iv.lo, c * iv.hi
            lo, hi = lo + min(a, b), hi + max(a, b)
        return Interval(lo, hi)

    def to_json(self) -> dict:
        return dict(producer=self.producer, statistic=self.statistic, coef=dict(self.coef), offset=self.offset, noise=self.noise,
                    noise_tag=self.noise_tag, noise_src=self.noise_src)

    @staticmethod
    def from_json(d: dict) -> "ReadingSpec":
        return ReadingSpec(d["producer"], d["statistic"], {k: float(v) for k, v in d["coef"].items()}, float(d["offset"]), float(d["noise"]),
                           d["noise_tag"], d["noise_src"])


def reading_range(spec: ReadingSpec, theta: ThetaSet) -> Interval:
    """hull of every reading any theta in the set can produce: [min h, max h] widened by +-noise.  Exact (affine h over boxes)."""
    if theta.is_empty():
        raise ValueError("reading_range of an empty ThetaSet")
    rs = [spec.h_range(c.bounds) for c in theta.cells]
    return Interval(min(r.lo for r in rs) - spec.noise, max(r.hi for r in rs) + spec.noise)


# ----------------------------------------------------------------------------------------------------------------------- recovery program
@dataclass(frozen=True)
class RecoveryProgram:
    """parameters set from readings during execution, never from theta*.  prefix and bin plans are opaque (the planner's plans)."""
    name: str
    prefix: Any
    reading: "ReadingSpec | None"
    bins: tuple
    landing: str

    def __post_init__(self):
        object.__setattr__(self, "bins", tuple((iv, plan) for iv, plan in self.bins))
        if self.reading is None and len(self.bins) != 1:
            raise ValueError("a program without a reading has exactly one bin")

    @staticmethod
    def fixed(name: str, plan: Any, landing: str) -> "RecoveryProgram":
        """the fixed plan = the one-bin program (user ruling 11-16 D4)."""
        return RecoveryProgram(name, None, None, ((Interval(-math.inf, math.inf), plan),), landing)

    def to_json(self) -> dict:
        return dict(name=self.name, prefix=_jsonable(self.prefix), reading=None if self.reading is None else self.reading.to_json(),
                    bins=[[iv.to_json(), _jsonable(p)] for iv, p in self.bins], landing=self.landing)

    @staticmethod
    def from_json(d: dict) -> "RecoveryProgram":
        return RecoveryProgram(d["name"], d["prefix"], None if d["reading"] is None else ReadingSpec.from_json(d["reading"]),
                               tuple((Interval.from_json(iv), p) for iv, p in d["bins"]), d["landing"])


def _slab_cell(spec: ReadingSpec, cell: Cell, ylo: float, yhi: float):
    """(cell restricted to {theta: h(theta) in [ylo, yhi]}, exact) or (None, exact).  Exact iff <= 1 field has a nonzero coefficient."""
    nz = [f for f, c in spec.coef.items() if c != 0.0]
    if not nz:
        return (cell if ylo <= spec.offset <= yhi else None), True
    h = spec.h_range(cell.bounds)
    if h.hi < ylo or h.lo > yhi:
        return None, len(nz) == 1
    new = dict(cell.bounds)
    for f in nz:
        c = spec.coef[f]
        rest = spec.offset
        rlo = rhi = 0.0
        for g in nz:
            if g != f:
                a, b = spec.coef[g] * cell.bounds[g].lo, spec.coef[g] * cell.bounds[g].hi
                rlo, rhi = rlo + min(a, b), rhi + max(a, b)
        # c*theta_f in [ylo - offset - rhi, yhi - offset - rlo]
        t1, t2 = (ylo - spec.offset - rhi) / c, (yhi - spec.offset - rlo) / c
        j = cell.bounds[f].intersect(Interval(min(t1, t2), max(t1, t2)))
        if j is None:
            return None, len(nz) == 1
        new[f] = j
    return cell.with_bounds(new), len(nz) == 1


def bin_sets(program: RecoveryProgram, theta: ThetaSet) -> list:
    """[(j, Theta_j, exact)]: Theta_j = {theta in Theta : (h(theta) + [-nu, nu]) meets Y_j}.  Exact (a box per cell) when h has one nonzero
    coefficient; otherwise the per-cell bounding box of the slab (a sound superset) with exact=False.  Fixed program: [(0, Theta, True)]."""
    if program.reading is None:
        return [(0, theta, True)]
    spec = program.reading
    out = []
    for j, (iv, _plan) in enumerate(program.bins):
        cells, exact = [], True
        for c in theta.cells:
            cj, ex = _slab_cell(spec, c, iv.lo - spec.noise, iv.hi + spec.noise)
            exact = exact and ex
            if cj is not None:
                cells.append(cj)
        out.append((j, ThetaSet(tuple(cells)), exact))
    return out


def covers_readings(program: RecoveryProgram, theta: ThetaSet) -> tuple:
    """(True, []) iff the union of the bins' Y_j contains reading_range(spec, Theta); else (False, uncovered reading intervals)."""
    if program.reading is None or theta.is_empty():
        return True, []
    R = reading_range(program.reading, theta)
    gaps, cur = [], R.lo
    for iv in sorted((b[0] for b in program.bins), key=lambda i: (i.lo, i.hi)):
        if iv.hi < cur:
            continue
        if iv.lo > cur:
            if cur < R.hi or (R.lo == R.hi and not gaps and cur == R.lo):
                gaps.append(Interval(cur, min(iv.lo, R.hi)))
        cur = max(cur, iv.hi)
        if cur >= R.hi:
            break
    if cur < R.hi:
        gaps.append(Interval(cur, R.hi))
    return (not gaps), gaps


# ----------------------------------------------------------------------------------------------------------------------- admission
@dataclass
class Admission:
    """plain container for what the checker admitted at one step.  table: branch time -> [(recovery name, verified set)]."""
    step: int
    recovery: str
    theta_cert: ThetaSet
    table: dict = field(default_factory=dict)
    fallback_ok_on: "ThetaSet | None" = None
    rows: list = field(default_factory=list)
    assumptions: list = field(default_factory=list)

    def to_json(self) -> dict:
        return dict(step=self.step, recovery=self.recovery, theta_cert=self.theta_cert.to_json(),
                    table=[[float(t), [[n, s.to_json()] for n, s in lst]] for t, lst in self.table.items()],
                    fallback_ok_on=None if self.fallback_ok_on is None else self.fallback_ok_on.to_json(),
                    rows=_jsonable(self.rows), assumptions=_jsonable(self.assumptions))

    @staticmethod
    def from_json(d: dict) -> "Admission":
        return Admission(int(d["step"]), d["recovery"], ThetaSet.from_json(d["theta_cert"]),
                         {float(t): [(n, ThetaSet.from_json(s)) for n, s in lst] for t, lst in d["table"]},
                         None if d["fallback_ok_on"] is None else ThetaSet.from_json(d["fallback_ok_on"]),
                         list(d["rows"]), list(d["assumptions"]))
