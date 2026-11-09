"""checks/probe_rung.py -- admission of one probe rung (push the feed-forward mass from M_lo to m_to).  SIMULATION ONLY.

  payload at rest   u(rest pose, a = 0, mass m_to) <= U_ADMIT                      checks/payload.rest_row  (proof under MODEL)
  overshoot         m_to - M_lo <= O_MAX                                           ASSUMED: largest push above the current lower bound [kg]
Row margin = min(U_ADMIT - u [utilisation], O_MAX - overshoot [kg]) -- mixed units, see worst for both.
"""
from __future__ import annotations

from world_model.v2.stack.checks import payload as PAY
from world_model.v2.stack.checks.row import Row

O_MAX = 1.0          # ASSUMED [kg]: largest push above the current lower bound


def rung_row(rest, empty_mass: float, m_to: float, M_lo: float) -> Row:
    """rest: a payload.Rest backend; empty_mass: the empty object mass [kg]."""
    r = PAY.rest_row(rest, empty_mass, m_to)
    over = float(m_to - M_lo)
    ok = bool(r.ok and over <= O_MAX + 1e-9)
    return Row("probe_rung", ok, margin=min(r.margin, O_MAX - over), unit="mixed (utilisation, kg)", guarantee="proof",
               worst=dict(u=r.worst["u"], overshoot=over), assumptions=dict(payload=r.assumptions["model"], O_MAX=f"{O_MAX} kg ASSUMED"),
               why="; ".join(x for x in (r.why, "" if over <= O_MAX + 1e-9 else f"overshoot {over:.3f} kg > {O_MAX}") if x))
