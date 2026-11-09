"""checks/plan_shape.py -- geometry rules on a motion plan (proof under MODEL: deterministic checks on the stated supports).  SIMULATION ONLY.
Rules (from proposer/compile.py):  every motion >= MIN_MOVE_M; the object never goes below its start support (except a place);
horizontal motion needs MIN_CLEAR_M above the start support at both ends (no dragging); a place lowers >= MIN_MOVE_M and its centre is over the support;
the plan ends on the goal support within the goal tolerance of the goal xy.
steps = [(skill, args)] of the MOTION steps (lift {dz} | move {dx, dy, dz} | place {support}); the release step is not passed."""
from __future__ import annotations

import numpy as np

from world_model.v2.stack.checks.row import Row

MIN_CLEAR_M = 0.02         # ASSUMED [m]: object underside above its start support for any horizontal motion
MIN_MOVE_M = 0.005         # ASSUMED [m]: shortest motion (a shorter one is degenerate for the timing)
Z_EPS = 1e-6


def check(steps, supports, start_xy, goal_xy, goal_support, goal_tol, start_support="counter"):
    """-> (Row, deltas).  supports: {name: dict(top, x=(lo, hi), y=(lo, hi))}."""
    t0 = supports[start_support]["top"]
    sx, sy = start_xy
    pos = np.zeros(3)
    deltas, errs = [], []
    end_support = None
    for i, (sk, a) in enumerate(steps):
        if sk == "lift":
            d = np.array([0.0, 0.0, a["dz"]])
        elif sk == "move":
            d = np.array([a["dx"], a["dy"], a["dz"]])
        else:
            top = supports[a["support"]]["top"]
            d = np.array([0.0, 0.0, top - (t0 + pos[2])])
            cx, cy = sx + pos[0], sy + pos[1]
            s = supports[a["support"]]
            if not (s["x"][0] <= cx <= s["x"][1] and s["y"][0] <= cy <= s["y"][1]):
                errs.append(f"step {i} place({a['support']}): the object centre ({cx:.3f}, {cy:.3f}) is not over the support (x {s['x']}, y {s['y']})")
            if d[2] > -MIN_MOVE_M:
                errs.append(f"step {i} place({a['support']}): the object underside is at {t0 + pos[2]:.3f} m, the support top at {top:.3f} m; nothing to lower (needs >= {MIN_MOVE_M} m)")
        new = pos + d
        if np.linalg.norm(d) < MIN_MOVE_M:
            errs.append(f"step {i} {sk}: motion {np.linalg.norm(d):.4f} m shorter than {MIN_MOVE_M} m")
        horiz = float(np.hypot(d[0], d[1])) > 1e-9
        if new[2] < -Z_EPS and sk != "place":
            errs.append(f"step {i} {sk}: the object would go to {new[2]:.3f} m relative to its start support (below it)")
        if horiz and (pos[2] < MIN_CLEAR_M - Z_EPS or new[2] < MIN_CLEAR_M - Z_EPS):
            errs.append(f"step {i} {sk}: horizontal motion with the object {min(pos[2], new[2]):.3f} m above its start support; needs >= {MIN_CLEAR_M} m at both ends (no dragging)")
        deltas.append(tuple(round(float(x), 9) for x in d))
        end_support = a.get("support") if sk == "place" else a.get("support")
        pos = new
    end = (float(sx + pos[0]), float(sy + pos[1]))
    miss = float(np.hypot(end[0] - goal_xy[0], end[1] - goal_xy[1]))
    if end_support != goal_support:
        errs.append(f"the plan places on {end_support!r}, the task goal is on {goal_support!r}")
    if miss > goal_tol:
        errs.append(f"the plan ends at xy ({end[0]:.3f}, {end[1]:.3f}), {miss:.3f} m from the goal ({goal_xy[0]}, {goal_xy[1]}); tolerance {goal_tol} m")
    margin = goal_tol - miss
    if errs:
        margin = min(margin, 0.0)            # a failed row never carries a positive margin (the goal miss alone can be inside the tolerance)
    return Row("plan_shape", not errs, margin=margin, unit="m (goal miss)", guarantee="proof", worst=dict(errors=errs, end_xy=end, miss_m=miss),
               assumptions=dict(MIN_CLEAR_M=f"{MIN_CLEAR_M} m ASSUMED", MIN_MOVE_M=f"{MIN_MOVE_M} m ASSUMED"), why="; ".join(errs)), deltas
