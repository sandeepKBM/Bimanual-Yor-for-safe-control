"""nero/contents/objects.py -- carried objects with hidden contents: empty body, contents set, open-content limits.  SIM ONLY.

USER 11-01 (relayed): the grasp is perfect -> a rigid weld once the claw closes.  ASSUMED, justification given by the user: wrist
cameras + low-level control handle grasping.  Everything below is conditional on that assumption.

A contents set is  {(m, c): m in [m_lo, m_hi], c in P}  with P a 3-D polytope (object frame, origin = the object's reference point o,
x/y horizontal, z up when level).  In pi = (m, m*c) coordinates the set is convex with vertices (m_lo, m_lo v), (m_hi, m_hi v) for the
vertices v of P, so any condition that is convex in pi holds on the whole set iff it holds at those vertices (envelope.py).

Labels per AGENTS.md: MEASURED (file, n, statistic), MODEL (physics/geometry; formula given), ASSUMED (a number we chose).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

G = 9.81
WATER = 1000.0          # kg/m^3, MODEL


def circ_polygon(r, n=16):
    """CIRCUMSCRIBED n-gon of a disc of radius r (a superset: conservative for 'for all c in the disc')."""
    R = r / np.cos(np.pi / n)
    a = 2 * np.pi * (np.arange(n) + 0.5) / n
    return np.stack([R * np.cos(a), R * np.sin(a)], 1)


def prism(poly_xy, z_lo, z_hi):
    return np.array([[x, y, z] for z in (z_lo, z_hi) for x, y in poly_xy])


@dataclass
class Contents:
    m_lo: float
    m_hi: float
    P: np.ndarray                      # (k, 3) COM polytope vertices, object frame
    labels: dict = field(default_factory=dict)

    def pi_vertices(self):
        """vertices of the contents set in pi = (m, m c) coordinates: (4,) rows [m, m cx, m cy, m cz]."""
        out = []
        for mm in sorted({self.m_lo, self.m_hi}):
            for v in self.P:
                out.append(np.concatenate([[mm], mm * v]))
        return np.unique(np.round(np.array(out), 12), axis=0)

    def scaled(self, m_hi):
        return Contents(self.m_lo, m_hi, self.P, self.labels)


@dataclass
class OpenLimit:
    """quasi-static acceleration limit of an open content (liquid in a cup, a loose lid): the content stays if |a_h| <= a_max.
    MODEL: liquid surface tilts by atan(a_h / g); it reaches the lip when (D/2) tan = H - h  ->  a_max = g * 2 (H - h) / D.
    Lid on a rim (Coulomb, no tipping): a_max = mu * g.  Dynamic slosh / impacts are NOT covered (sampled in sim)."""
    name: str
    a_max: float
    formula: str
    labels: dict = field(default_factory=dict)


def cup_spill_limit(D, H, fill_h):
    return G * 2.0 * max(H - fill_h, 0.0) / D


@dataclass
class ObjectSpec:
    name: str
    empty_mass: float                  # kg (object + lid, without the hidden contents)
    empty_com: np.ndarray              # object frame
    contents: Contents
    open_limits: list
    sim_box: dict                      # the Theta box the sim currently uses (for "does the object reach the range")
    plausible: dict                    # physically plausible contents range (MODEL, e.g. water-full volume)
    labels: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------------------------------------------
# pan / pot: the PLA 3D-printed 18 cm pan of pan/scene.py (pot = the same body + a second handle, segments obj="pot")
PAN_R, PAN_WALL_H, PAN_BOTTOM_T = 0.09, 0.04, 0.0025          # MODEL (pan/scene.py ASSUMED geometry)
LID_MU = (0.5, 1.0)        # ASSUMED: lid friction scale box of hardworlds.THETA (lid_mu_scale) x MJCF friction 1.0 -> mu 0.5..1.0


def pan_like(name, empty_mass, empty_com, m_hi_sim=1.5):
    vol = np.pi * PAN_R ** 2 * PAN_WALL_H
    P = prism(circ_polygon(PAN_R), PAN_BOTTOM_T, PAN_BOTTOM_T + PAN_WALL_H)
    c = Contents(0.0, m_hi_sim, P, labels=dict(
        m="ASSUMED box 0..1.5 kg (hardworlds.THETA mass 0.02..1.5)",
        P="MODEL: any contents COM lies inside the inner wall (r <= 0.09 m, circumscribed 16-gon), 0..wall height"))
    lid = OpenLimit("lid_slide", LID_MU[0] * G, "a_max = mu_lid_min * g (Coulomb, quasi-static)",
                    labels=dict(mu="ASSUMED 0.5 (lower end of lid_mu_scale box x MJCF mu 1.0)"))
    return ObjectSpec(name, empty_mass, np.asarray(empty_com, float), c, [lid],
                      sim_box=dict(m=(0.02, 1.5), r_com=0.06, note="hardworlds.THETA (ASSUMED); c_r disc C_R = 0.06 m (MODEL puck)"),
                      plausible=dict(volume_L=vol * 1e3, water_full_kg=vol * WATER,
                                     note="MODEL: inner volume pi r^2 h; denser food (rho 1.2) -> %.2f kg" % (vol * 1200)))


# coffee tray (AgentChord's 'setup coffee tray' task, agentchord_probe/coffee_tray/build.py geometry) with hidden cup fill + placement
TRAY = dict(lx=0.22, ly=0.33, mass=0.30, rim_h=0.03)            # MODEL from build.py (their numbers were our choice, 10-28)
# 11-02: ly 0.34 -> 0.33.  At the stored grasp pose the lower finger of each claw reached ~1 mm past the base-plate edge and sat 2 mm
# inside it (90 N finger-tray contact while carried, 160 N before lift; drag at release).  Base edge 5 mm in from the finger; tabs and
# grasp sites unchanged.
CUP = dict(D=0.08, H=0.095, empty=0.30, fill_max_kg=0.25, freeboard_min=0.01)   # ASSUMED: ceramic mug 300 g, 250 ml, >= 1 cm freeboard
N_CUPS = 2                                                      # ASSUMED: the number of cups is visible (count known); 11-02 runs use 2, 4, 6


def coffee_tray(n_cups=N_CUPS):
    D, H = CUP["D"], CUP["H"]
    hx, hy = TRAY["lx"] / 2 - D / 2, TRAY["ly"] / 2 - D / 2      # cup centres anywhere inside the rim
    rect = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    m_lo = n_cups * CUP["empty"]
    m_hi = n_cups * (CUP["empty"] + CUP["fill_max_kg"])
    # contents COM height: empty mug ~H/2, full mug ~ (H/2 mug + fill/2) -> bounded by [0.02, H/2] above the tray base (MODEL bound)
    P = prism(rect, 0.008 + 0.02, 0.008 + H / 2)
    c = Contents(m_lo, m_hi, P, labels=dict(
        m=f"ASSUMED: {n_cups} mugs x (0.30 kg empty + 0..0.25 kg fill)",
        P="MODEL (conservative): combined COM inside the convex hull of possible cup centres (rect); real set is smaller (no overlap)"))
    fill_h = H - CUP["freeboard_min"]
    spill = OpenLimit("cup_spill", cup_spill_limit(D, H, fill_h), "a_max = g 2 (H - h) / D at the fullest fill (1 cm freeboard)",
                      labels=dict(geometry="ASSUMED mug D 0.08 H 0.095; fill up to 1 cm below the lip"))
    return ObjectSpec("coffee_tray", TRAY["mass"], np.array([0.0, 0.0, 0.004]), c, [spill],
                      sim_box=dict(note="no NERO tray scene yet (step 2)"),
                      plausible=dict(note=f"{n_cups} mugs: {m_lo:.2f}..{m_hi:.2f} kg"))


# closed box with hidden contents (cans etc.) for the shelf task (contents/scenes.BoxWorld geometry)
BOX = dict(lx=0.20, ly=0.30, lz=0.15, wall=0.005, mass=0.25)    # ASSUMED cardboard crate with end tabs (scenes.py)
BOX_CONTENTS_KG = (0.5, 5.0)                                     # ASSUMED (user-approved 11-02): box of cans 0.5..5 kg


def boxed(m_lo=BOX_CONTENTS_KG[0], m_hi=BOX_CONTENTS_KG[1]):
    hx, hy, w = BOX["lx"] / 2 - BOX["wall"], BOX["ly"] / 2 - BOX["wall"], BOX["wall"]
    P = prism(np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]]), w, BOX["lz"] - w)
    c = Contents(m_lo, m_hi, P, labels=dict(
        m=f"ASSUMED {m_lo}..{m_hi} kg (user-approved 11-02)",
        P="MODEL (conservative): the contents COM lies anywhere inside the box interior (the sim block is a subset)"))
    return ObjectSpec("box", BOX["mass"], np.array([0.0, 0.0, BOX["lz"] / 2]), c, [],
                      sim_box=dict(note="scenes.BoxWorld: one free block, half (0.04, 0.06, 0.05), on the box floor"),
                      plausible=dict(note="box of cans / jars: 0.5..5 kg"))
