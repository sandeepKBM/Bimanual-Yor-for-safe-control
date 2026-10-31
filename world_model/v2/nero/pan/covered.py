"""pan/covered.py -- main demo scene (USER 10-28): "covered pan to the table".  SIMULATION ONLY.

Built on pan/scene.py (same NERO robot, table, 6 cm stove block, PLA pan + handle), minus the open-pan food beads, spatula and
plate, plus:
  * a loose PLA lid (disk + 4 locating tabs inside the wall + knob), so the contents are hidden from the cameras;
  * ONE hidden weight (a puck) at one of the named placements below, or none; an overload mass for the overload case;
  * a counter block on the way (the fallback set-down spot) at (0.56, -0.02): clear of the handle grasp, carry pose reachable.
The destination table needs the mobile base; not modelled here.

Placements are in the PAN frame: u = along the handle (towards it), w = z x u (horizontal, normal to the handle).  With the handle
pointing +y (robot's left), "far" = away from the handle = robot's right; +w = -x (towards the robot).

Probe (proposal 10-28): raise the handle 1-2 cm while the far rim stays on the burner.  probe_statics() gives what that probe can
measure.  Rigid statics, exact for a pivot on the far rim and any claw wrench:
    A = moment about the pivot line (normal to u, through the far-rim contact) = W * (u_com - u_pivot)
    B = moment about the u axis through the pivot                               = W * w_com
whatever the claw/pivot split (the pitch split between claw torque and pivot force is statically indeterminate, these two are not).
A full lift needs, at the claw: vertical force W and handle moment M = W * (u_grasp - u_com), roll torque W * w_com.
Masses / COMs come from the compiled MuJoCo model (geometry x density, puck mass set explicitly).

    python -m world_model.v2.nero.pan.covered [--render out.png] [--write]
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import json

import numpy as np
import mujoco

from . import scene as S
from .. import config as C

SCENE_OUT = C.REPO / "scenes" / "yor_nero_pan_covered.mjcf"

ASSUMED_COVERED = dict(                   # every value is an assumption (no real spec yet)
    lid_t=0.0025, lid_tab=(0.008, 0.004, 0.006), lid_knob=(0.012, 0.015),   # m; tab (half x, half y, height), knob (r, h)
    puck_r=0.025, puck_h=0.02, puck_mu=0.8,
    hidden_mass=0.20,                      # kg, a normal hidden load
    overload_mass=0.60,                    # kg, the overload case (placed far-centre)
    placements={                           # (u, w) in the pan frame [m]; pan inner radius 0.09, puck radius 0.025
        "none": None,
        "near_handle": (0.05, 0.0),
        "far_centre": (-0.05, 0.0),
        "far_left": (-0.04, 0.045),
        "far_right": (-0.04, -0.045),
        "overload": (-0.05, 0.0),
    },
    counter_centre_xy=(0.56, -0.02), counter_half=(0.12, 0.10), # on-the-way counter, top at stove height; at (0.40,-0.02) it blocked the handle grasp (reach check 10-28)
)


@dataclass
class CoveredParams:
    pan: S.PanParams = dataclasses.field(default_factory=lambda: S.PanParams(food_n=0, with_spatula=False, with_plate=False))
    placement: str = "far_centre"
    hidden_mass: float = ASSUMED_COVERED["hidden_mass"]
    overload_mass: float = ASSUMED_COVERED["overload_mass"]
    lid: bool = True
    counter: bool = True


def _frame(p: S.PanParams):
    u = p.handle_axis()
    w = np.cross([0.0, 0.0, 1.0], u)
    return u, w


def build_spec(cp: CoveredParams | None = None):
    cp = cp or CoveredParams()
    p = cp.pan
    spec = S.build_spec(p)
    W = spec.worldbody
    a = ASSUMED_COVERED
    px, py = p.pan_centre_xy
    z_rim = p.pan_base_z() + p.pan_bottom_t + p.pan_wall_h
    u, w = _frame(p)
    if cp.counter:
        hx, hy = a["counter_half"]
        cz = p.pan_base_z()            # same height as the stove top
        W.add_geom(name="counter_near", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, (cz - p.table_top_z) / 2],
                   pos=[a["counter_centre_xy"][0], a["counter_centre_xy"][1], (cz + p.table_top_z) / 2],
                   contype=1, conaffinity=1, friction=[p.table_mu, 0.005, 0.0001], rgba=[0.6, 0.6, 0.62, 1])
    if cp.lid:
        R, t = p.pan_radius, p.pan_wall_t
        lid = W.add_body(name="lid", pos=[px, py, z_rim + a["lid_t"] / 2 + 1e-4])
        lid.add_freejoint(name="lid_free")
        lid.add_geom(name="lid_disk", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[R + t, a["lid_t"] / 2, 0], density=p.pla_density,
                     friction=[p.pan_mu, 0.005, 0.0001], rgba=[0.2, 0.45, 0.8, 0.95])
        tx, ty, th = a["lid_tab"]
        for i in range(4):
            ang = np.pi / 4 + i * np.pi / 2
            c = np.array([np.cos(ang), np.sin(ang)]) * (R - tx - 0.001)
            q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, [0, 0, 1], ang)
            lid.add_geom(name=f"lid_tab{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tx, ty, th / 2], density=p.pla_density,
                         pos=[c[0], c[1], -a["lid_t"] / 2 - th / 2], quat=q, rgba=[0.2, 0.45, 0.8, 1])
        kr, kh = a["lid_knob"]
        lid.add_geom(name="lid_knob", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[kr, kh / 2, 0], density=p.pla_density,
                     pos=[0, 0, a["lid_t"] / 2 + kh / 2], rgba=[0.15, 0.3, 0.6, 1])
    uw = a["placements"][cp.placement]
    if uw is not None:
        mass = cp.overload_mass if cp.placement == "overload" else cp.hidden_mass
        c = np.array([px, py]) + uw[0] * u[:2] + uw[1] * w[:2]
        puck = W.add_body(name="hidden", pos=[c[0], c[1], p.pan_base_z() + p.pan_bottom_t + a["puck_h"] / 2 + 1e-3])
        puck.add_freejoint(name="hidden_free")
        puck.add_geom(name="hidden_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[a["puck_r"], a["puck_h"] / 2, 0], mass=mass,
                      friction=[a["puck_mu"], 0.005, 0.0001], rgba=[0.55, 0.3, 0.15, 1])
    return spec


def compose(cp: CoveredParams | None = None):
    return build_spec(cp).compile()


def write_mjcf(cp: CoveredParams | None = None, path=SCENE_OUT):
    spec = build_spec(cp)
    spec.compile()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml())
    return path


def probe_statics(m, d, p: S.PanParams):
    """Probe and full-lift loads for the current (settled) state: pan + lid + hidden weight, rigid statics (see module doc)."""
    u, w = _frame(p)
    ids = [m.body(n).id for n in ("pan", "lid", "hidden") if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) >= 0]
    masses = np.array([m.body_mass[i] for i in ids])
    com = (masses[:, None] * np.array([d.xipos[i] for i in ids])).sum(0) / masses.sum()
    centre = d.site_xpos[m.site("pan_centre").id]
    grasp = d.site_xpos[m.site("pan_handle_grasp").id]
    pivot = centre - (p.pan_radius + p.pan_wall_t) * u          # far-rim bottom (horizontal coords only matter)
    Wt = masses.sum() * S.G
    uc, wc = float(np.dot(com - pivot, u)), float(np.dot(com - centre, w))
    ug = float(np.dot(grasp - pivot, u))
    return dict(mass_total_kg=round(float(masses.sum()), 4), com_u_from_centre_m=round(float(np.dot(com - centre, u)), 4),
                com_w_m=round(wc, 4),
                probe_A_pitch_about_pivot_Nm=round(Wt * uc, 4), probe_B_roll_Nm=round(Wt * wc, 4),
                lift_force_N=round(Wt, 3), lift_handle_moment_Nm=round(Wt * (ug - uc), 4), lift_roll_Nm=round(Wt * wc, 4),
                lever_grasp_to_pivot_m=round(ug, 4))


def table(cp_base: CoveredParams | None = None, settle_s=1.0):
    cp_base = cp_base or CoveredParams()
    out = {}
    for name in ASSUMED_COVERED["placements"]:
        cp = dataclasses.replace(cp_base, placement=name)
        m = compose(cp)
        d = mujoco.MjData(m)
        S.settle(m, d, settle_s)
        r = probe_statics(m, d, cp.pan)
        if name != "none":
            hp = d.xpos[m.body("hidden").id]
            r["hidden_rest_z_above_bottom_m"] = round(float(hp[2] - d.xpos[m.body("pan").id][2] - cp.pan.pan_bottom_t), 4)
        r["lid_on"] = bool(np.linalg.norm(d.xpos[m.body("lid").id][:2] - d.xpos[m.body("pan").id][:2]) < 0.01) if cp.lid else None
        out[name] = r
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--placement", default="far_centre", choices=list(ASSUMED_COVERED["placements"]))
    ap.add_argument("--render", default=None)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv)
    rep = dict(assumed=ASSUMED_COVERED, table=table())
    cp = CoveredParams(placement=a.placement)
    if a.render or a.write:
        m = compose(cp)
        d = mujoco.MjData(m)
        S.settle(m, d, 1.0)
        if a.render:
            rep["render"] = str(S.render(m, d, a.render))
        if a.write:
            rep["wrote"] = str(write_mjcf(cp))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
