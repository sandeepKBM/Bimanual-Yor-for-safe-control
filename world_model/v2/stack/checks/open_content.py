"""checks/open_content.py -- static limits of open contents (spill of a liquid in a cup, a loose lid) + the contents-set builders (ObjectSpec: adapters/contents_specs.py).  SIMULATION ONLY.

ONE spill formula for the static admission (a_max) and the ratio form below.  NOTE: the runtime spill monitor (nero/contents/carry.py:581-584, mug frame,
filtered acceleration) still computes its own ratio; switching it to `spill_ratio` here is migration step 2, not done yet.
  quasi-static liquid surface tilts by atan(a_h / (g + a_z)); it reaches the lip when (D/2) tan(phi) = freeboard = H - h
  static limit      a_max   = g * 2 (H - h) / D                 (a_z = 0)
  runtime ratio     ratio   = (D/2) tan(phi) / freeboard        (> 1 = spill)      ratio = 1  <=>  a_h = a_max  (tests check)
MODEL; slosh (dynamics) is NOT covered.  Lid on a rim (Coulomb, no tipping): a_max = mu_lid_min g, built in objects.pan_like (ASSUMED mu).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.stack.checks.row import Row

G = 9.81         # MODEL, nero/contents/objects.py G


def spill_limit(D, H, fill_h):
    """largest horizontal acceleration [m/s^2] before the liquid reaches the lip (MODEL, quasi-static)."""
    return G * 2.0 * max(H - fill_h, 0.0) / D


def spill_ratio(D, freeboard, a_h, a_z=0.0):
    """(D/2) tan(phi) / freeboard with tan(phi) = a_h / (g + a_z); > 1 = spilled.  The runtime monitor's quantity."""
    return 0.5 * D * (a_h / max(G + a_z, 1e-6)) / max(freeboard, 1e-6)


def open_content_row(a_plan, a_open, name="open_content") -> Row:
    """planned peak acceleration must not exceed the static open-content limit (proof under MODEL, slosh NOT covered)."""
    return Row(name, bool(a_plan <= a_open), margin=float(a_open - a_plan), unit="m/s^2", guarantee="proof",
               worst=dict(a_plan=float(a_plan), a_open=round(float(a_open), 4)), assumptions=dict(model="quasi-static liquid surface; slosh NOT covered"),
               why="" if a_plan <= a_open else f"planned acceleration {a_plan:.3f} exceeds the open-content limit {a_open:.3f}")


def a_open_of(spec) -> float:
    return min([ol.a_max for ol in spec.open_limits], default=np.inf)
