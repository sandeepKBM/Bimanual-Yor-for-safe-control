"""checks/payload.py -- arm-side torque + per-claw payload utilisation u on a path (static, proof under MODEL).  SIMULATION ONLY.

The physics backend is PASSED IN (protocol `EnvelopeLike`; the adapter adapters/payload_envelope.py builds nero/contents/envelope.Envelope):
u(wp, pi, a) = max over joints of the gravity + inertia torque / torque limit and the arm vertical load vs (payload - gripper mass), for the COMMANDED
lever split ("two_lever"); worst over the contents set (exact vertex check) x the acceleration box.
Row margin = U_ADMIT - u  (> 0 inside).  No claw rows (USER 11-02 pm).  Torque box = config.tau_limit() (MJCF 100 Nm, USER 11-05), inside the backend.
The probe's rest-pose row (u at the start pose, a = 0, carried mass m_ff) is `rest_row`, given a `Rest` backend.
"""
from __future__ import annotations

from typing import Protocol

from world_model.v2.stack.checks.row import Row

U_ADMIT = 1.0                  # MODEL: the rated limits themselves (joint torque config.tau_limit(); per-arm payload 3 kg MEASURED-spec minus gripper); USER 11-09: no extra margin (= nero/contents/envelope.py U_ADMIT; the adapter test asserts equality)
MODE = "two_lever"
GUARANTEE = ("proof under MODEL (rigid weld ASSUMED, quasi-static, NERO MJCF arms, config.tau_limit() torque, 3 kg payload, no claw rows (user 11-02), "
             f"admit u <= {U_ADMIT} MODEL, USER 11-09)")


class EnvelopeLike(Protocol):
    def worst(self, contents, wps, a=0.0, mode="two"): ...      # -> (u_max, detail dict)
    def a_max(self, contents, wps, mode="two", a_cap=5.0, tol=1e-3, limit=1.0): ...


class Rest(Protocol):
    def u(self, c: float) -> float: ...                          # u at the rest pose, a = 0, contents mass c (COM anywhere in the interior)


def torque_row(env: EnvelopeLike, contents, keys, a, name="torque_payload") -> Row:
    """u over `keys` (path samples) at peak acceleration `a` for every contents vertex; margin = U_ADMIT - u."""
    u, det = env.worst(contents, keys, a, MODE)
    u_lp, _ = env.worst(contents, keys, a, "two")
    return Row(name, bool(u <= U_ADMIT), margin=float(U_ADMIT - u), unit="utilisation", guarantee="proof",
               worst={**{k: v for k, v in det.items() if k in ("wp", "m", "c", "Fz", "twist")}, "u_max": float(u), "u_best_split": float(u_lp), "a": float(a), "n_samples": len(keys)},
               assumptions=dict(model=GUARANTEE + f" over the contents set x acceleration box at {len(keys)} path samples"),
               why="" if u <= U_ADMIT else f"utilisation {u:.3f} > {U_ADMIT}")


def a_max(env: EnvelopeLike, contents, keys, limit=U_ADMIT):
    """largest peak acceleration with u <= limit (bisection in the backend); None if even a = 0 fails."""
    return env.a_max(contents, keys, MODE, limit=limit)


def rest_row(rest: Rest, empty_mass: float, m_ff: float) -> Row:
    """u at the rest pose carrying total mass m_ff (contents m_ff - empty): MODEL."""
    u = rest.u(max(m_ff - empty_mass, 0.0))
    return Row("rest_payload", bool(u <= U_ADMIT), margin=float(U_ADMIT - u), unit="utilisation", guarantee="proof", worst=dict(u=u, m_ff=float(m_ff)),
               assumptions=dict(model=GUARANTEE + " at the rest pose, a = 0"), why="" if u <= U_ADMIT else f"rest utilisation {u:.3f} > {U_ADMIT}")
