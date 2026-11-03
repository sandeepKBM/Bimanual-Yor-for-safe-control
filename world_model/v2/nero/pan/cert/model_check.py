"""pan/cert/model_check.py -- model-level checks over the belief (the "proof" layer).  SIMULATION ONLY.

A check is exact over the contents polytope when it is linear in pi = (m, s_u, s_w): it then holds on the whole set iff it holds
at the vertices.  Pose-residual and friction corners are enumerated (exact only where the check is monotone in them; otherwise
the result is labelled sampled).  Every result carries its guarantee type and the worst vertex as a counterexample.

    support_at_rest   the total COM lies over the pan-on-stove contact area with a margin (rest set S, R_V2 "stable")
    planned_accel     peak commanded TCP acceleration of a phase list (smoothstep moves), against a bound (assumption A4)
"""
from __future__ import annotations

import numpy as np

SUPPORT_MARGIN = 0.005   # ASSUMED: COM at least 5 mm inside the contact area for "stable at rest"
DISC_N = 64              # design: the pan's bottom disc as an INSCRIBED 64-gon (a subset of the disc: conservative for support)
A_MAX = 0.5              # ASSUMED: peak commanded TCP acceleration [m/s^2] for moves (A4, quasi-static); lid retention TBD


def _clip(poly, n, h):
    """Sutherland-Hodgman: keep the part of the convex polygon with n . x <= h."""
    out = []
    for i in range(len(poly)):
        p, q = poly[i], poly[(i + 1) % len(poly)]
        fp, fq = n @ p - h, n @ q - h
        if fp <= 0:
            out.append(p)
        if fp * fq < 0:
            out.append(p + (q - p) * fp / (fp - fq))
    return out


def support_polygon(centre_xy, r_bottom, stove_xy, stove_half):
    """contact area = (pan bottom disc, inscribed DISC_N-gon) intersected with the stove-top rectangle.  Returns the vertices
    (counter-clockwise) and the half-planes (n, h) with n . x <= h inside."""
    c = np.asarray(centre_xy, float)
    a = 2 * np.pi * np.arange(DISC_N) / DISC_N
    poly = [c + r_bottom * np.array([np.cos(t), np.sin(t)]) for t in a]
    s, hx, hy = np.asarray(stove_xy, float), float(stove_half[0]), float(stove_half[1])
    for n, h in ((np.array([1.0, 0]), s[0] + hx), (np.array([-1.0, 0]), -(s[0] - hx)),
                 (np.array([0, 1.0]), s[1] + hy), (np.array([0, -1.0]), -(s[1] - hy))):
        poly = _clip(poly, n, h)
        if not poly:
            return [], []
    P = np.array(poly)
    planes = []
    for i in range(len(P)):
        p, q = P[i], P[(i + 1) % len(P)]
        e = q - p
        if np.linalg.norm(e) < 1e-12:
            continue
        n = np.array([e[1], -e[0]]) / np.linalg.norm(e)          # outward normal for a CCW polygon
        planes.append((n, float(n @ p)))
    return P, planes


def support_at_rest(contents_vertices, M0, c0_xy, centre_xy, u_xy, planes, margin=SUPPORT_MARGIN):
    """COM over the contact area for every contents vertex (exact: linear in pi).
    M0, c0_xy: mass and world COM of pan + lid (MODEL, from the MJCF); centre_xy, u_xy: pan centre and handle axis (world).
    COM = (M0 c0 + m centre + R s) / (M0 + m) with R s = s_u u + s_w w;  n . COM <= h - margin  <=>
    n . (M0 c0 + m centre + s_u u + s_w w) <= (h - margin) (M0 + m)  -- linear in (m, s_u, s_w)."""
    u = np.asarray(u_xy, float) / np.linalg.norm(u_xy)
    w = np.array([-u[1], u[0]])
    c0, cen = np.asarray(c0_xy, float), np.asarray(centre_xy, float)
    if not planes:
        return dict(ok=False, margin_m=-1.0, worst=None, guarantee="proof (MODEL; no contact area)")
    worst = (np.inf, None)
    for pi in np.asarray(contents_vertices, float):
        m, su, sw = pi
        com = (M0 * c0 + m * cen + su * u + sw * w) / (M0 + m)
        d = min(h - n @ com for n, h in planes)                 # distance inside the contact area
        if d < worst[0]:
            worst = (float(d), [float(x) for x in pi])
    return dict(ok=worst[0] >= margin, margin_m=round(worst[0], 5), required_m=margin, worst_contents=worst[1],
                guarantee="proof over the contents polytope (MODEL: rigid pan, flat stove, inscribed disc)")


def planned_accel(phases, side_dist=None):
    """peak commanded TCP acceleration of the smoothstep moves in a phase list (p(a) = d (3a^2 - 2a^3): peak 6 d / T^2).
    move_to phases need the distance (side_dist: {phase index: metres}); returns (peak, per-phase list)."""
    per = []
    for i, ph in enumerate(phases):
        if ph[0] == "move":
            d = max(float(np.linalg.norm(v)) for v in ph[1].values())
            per.append((i, 6 * d / ph[2] ** 2))
        elif ph[0] == "move_to" and side_dist and i in side_dist:
            d = side_dist[i]
            T = max(0.8, ph[2] * d / ph[3])
            per.append((i, 6 * d / T ** 2))
    return (max((a for _, a in per), default=0.0), per)
