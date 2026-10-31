"""priors.py -- loader for category_priors.json (HAND-WRITTEN real-world priors; an ASSUMPTION, order-of-magnitude, NOT sim values).  Pure python, no mujoco.

mass_range(category, vol_m3) -> [lo, hi] kg.  Without a density range: the table bounds.  With one: [vol*dlo, vol*dhi] intersected with the bounds; if the intersection is
empty (object much bigger / smaller than the bounds allow) the nearest bound wins with a +-25% band, so lo < hi always holds."""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

PRIOR_PATH = Path(__file__).with_name("category_priors.json")


@dataclass(frozen=True)
class Prior:
    category: str
    display: str
    mass_lo: float
    mass_hi: float
    fragile: bool
    has_handle: bool
    contents: str
    density: tuple | None          # (lo, hi) kg/m^3 or None


@lru_cache(maxsize=1)
def load(path=None):
    raw = json.loads(Path(path or PRIOR_PATH).read_text())
    out = {}
    for k, v in raw["categories"].items():
        d = v.get("density_kg_m3")
        out[k] = Prior(k, v["display"], float(v["mass_kg"][0]), float(v["mass_kg"][1]), bool(v["fragile"]), bool(v["has_handle"]), v.get("contents", "empty"),
                       (float(d[0]), float(d[1])) if d else None)
    return out


def _r(x):
    return round(x, 3) if x < 0.1 else round(x, 2) if x < 10 else round(x, 1)


def mass_range(category, vol_m3=None, table=None):
    p = (table or load())[category]
    lo, hi = p.mass_lo, p.mass_hi
    if p.density is not None and vol_m3 is not None and vol_m3 > 0:
        dl, dh = vol_m3 * p.density[0], vol_m3 * p.density[1]
        nlo, nhi = max(lo, dl), min(hi, dh)
        if nlo > nhi:
            if dl > hi:                      # object too big for the bounds
                nlo, nhi = 0.8 * hi, hi
            else:                            # too small (dh < lo)
                nlo, nhi = lo, 1.25 * lo
        lo, hi = nlo, nhi
    lo, hi = _r(lo), _r(hi)
    if hi <= lo:
        hi = _r(lo * 1.25) if lo > 0 else lo + 0.001
    return [lo, hi]


def display_names(table=None):
    return sorted({p.display for p in (table or load()).values()})
