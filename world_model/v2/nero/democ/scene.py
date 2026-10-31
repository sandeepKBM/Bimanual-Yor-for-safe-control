"""democ/scene.py -- Demo C scene v0: YOR-NERO + floor + table (with a central pedestal) + a free box.  SIMULATION ONLY.

Generated (never hand-edited) scenes/yor_nero_democ.mjcf = scenes/yor_nero_derived.mjcf (via nero.build_variant, read-only)
+ table + box.  Nothing existing is modified.

Grasp design (documented decision): the 'box' is a slab pinched at its two +-y END regions, one hand per end (hand-to-hand
axis = world y), jaw axis = z (fingers above/below the slab).  The slab rests on a central PEDESTAL narrower than the slab so the
lower finger of each hand can slide under the overhanging ends.  Rotating the box about y = the turnover.

    python -m world_model.v2.nero.democ.scene [--mass 0.6] [--com-x 0.0] [--mu 1.0] [--write]
Every value not taken from a real spec is listed in ASSUMED below (one place).
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import json

import numpy as np
import mujoco

from .. import config as C
from .. import adapters as A
from .. import build_variant as BV

SCENE_OUT = C.REPO / "scenes" / "yor_nero_democ.mjcf"

ASSUMED = dict(                       # EVERYTHING here is an assumption (no real spec); change only here / via CLI
    box_half=(0.10, 0.15, 0.03),      # box half extents x,y,z [m]  (20 x 30 x 6 cm slab)
    box_mass=0.6,                     # kg
    box_com_offset=(0.0, 0.0, 0.0),   # COM offset from geometric centre [m] (body frame)
    box_mu=1.0,                       # sliding friction coefficient of the box geom (MuJoCo uses max(geom1, geom2) mu)
    box_mu_torsion=0.02, box_mu_roll=0.001,
    box_solref=(0.005, 1.0), box_solimp=(0.95, 0.99, 0.001, 0.5, 2.0),
    pedestal_half=(0.09, 0.07, 0.03), # support pedestal under the box centre; narrower than the box in y so end regions overhang
    table_half=(0.30, 0.45, 0.01),    # table-top half extents
    table_centre_xy=(0.50, -0.21),    # robot-local, same y as the tray hold (ik.TRAY_LOCAL)
    box_centre=(0.45, -0.21, 0.85),   # box centre at the pre-grasp/hold pose (ik.TRAY_LOCAL height)
    table_mu=0.8,
)


@dataclass
class DemoCParams:
    box_half: tuple = ASSUMED["box_half"]
    box_mass: float = ASSUMED["box_mass"]
    box_com_offset: tuple = ASSUMED["box_com_offset"]
    box_mu: float = ASSUMED["box_mu"]
    box_mu_torsion: float = ASSUMED["box_mu_torsion"]
    box_mu_roll: float = ASSUMED["box_mu_roll"]
    pedestal_half: tuple = ASSUMED["pedestal_half"]
    table_half: tuple = ASSUMED["table_half"]
    table_centre_xy: tuple = ASSUMED["table_centre_xy"]
    box_centre: tuple = ASSUMED["box_centre"]
    table_mu: float = ASSUMED["table_mu"]
    gravcomp: bool = False
    tau_preset: str | None = None

    def table_top_z(self):
        """z of the pedestal top == box underside at rest.  The table top is lowered so the box rests on the pedestal only."""
        return self.box_centre[2] - self.box_half[2]


def build_spec(p: DemoCParams | None = None):
    """-> MjSpec (not yet compiled)."""
    p = p or DemoCParams()
    tau = C.tau_limits(p.tau_preset)
    spec = mujoco.MjSpec.from_string(BV.variant_xml(lift=False, gravcomp=p.gravcomp, tau=tau))
    W = spec.worldbody
    W.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=1, conaffinity=1)
    ztop = p.table_top_z()
    ped = np.array(p.pedestal_half, float)
    tab = np.array(p.table_half, float)
    cx, cy = p.table_centre_xy
    bx, by, bz = p.box_centre
    # table-top: top face is (pedestal height) below the pedestal top, i.e. at ztop - 2*ped_z, so the lower finger has room
    z_tab_top = ztop - 2 * ped[2]
    W.add_geom(name="democ_table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(tab), pos=[cx, cy, z_tab_top - tab[2]],
               contype=1, conaffinity=1, friction=[p.table_mu, 0.005, 0.0001], rgba=[0.55, 0.45, 0.35, 1])
    W.add_geom(name="democ_table_pedestal", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(ped), pos=[bx, cy, ztop - ped[2]],
               contype=1, conaffinity=1, friction=[p.table_mu, 0.005, 0.0001], rgba=[0.45, 0.4, 0.35, 1])
    # 4 table legs (visual only, contype 0) for plausibility
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):
        W.add_geom(name=f"democ_table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (z_tab_top - 2 * tab[2]) / 2],
                   pos=[cx + sx * (tab[0] - 0.03), cy + sy * (tab[1] - 0.03), (z_tab_top - 2 * tab[2]) / 2],
                   contype=0, conaffinity=0, rgba=[0.4, 0.35, 0.3, 1])
    box = W.add_body(name="democ_box", pos=[bx, by, bz])
    box.add_freejoint(name="democ_box_free")
    com = np.array(p.box_com_offset, float)
    hx, hy, hz = p.box_half
    Ixx = p.box_mass / 3.0 * (hy**2 + hz**2)
    Iyy = p.box_mass / 3.0 * (hx**2 + hz**2)
    Izz = p.box_mass / 3.0 * (hx**2 + hy**2)
    # inertia about the COM (parallel-axis correction removed from the uniform-box value is ignored: offset is a small perturbation)
    box.mass = p.box_mass
    box.ipos = com
    box.inertia = [Ixx, Iyy, Izz]
    box.explicitinertial = True
    box.add_geom(name="democ_box_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, hz], contype=1, conaffinity=1,
                 condim=4, friction=[p.box_mu, p.box_mu_torsion, p.box_mu_roll], solref=list(ASSUMED["box_solref"]),
                 solimp=list(ASSUMED["box_solimp"]), mass=1e-9, rgba=[0.2, 0.5, 0.9, 1])
    A.patch_collision(spec)
    return spec


def compose(p: DemoCParams | None = None, dt: float | None = None):
    """-> compiled MjModel.  Collision bits: patch_collision makes arm geoms collide with geoms whose names contain
    table/floor/leg (ARM_BITS added to conaffinity); the box and pedestal are handled below."""
    p = p or DemoCParams()
    spec = build_spec(p)
    for g in spec.worldbody.geoms:
        if g.name in ("democ_table_top", "democ_table_pedestal"):
            g.conaffinity |= A.ARM_BITS
    for b in spec.bodies:
        if b.name == "democ_box":
            for g in b.geoms:
                g.conaffinity |= A.ARM_BITS
    m = spec.compile()
    if dt is not None:
        m.opt.timestep = dt
    # keep the explicit COM offset: compile() with explicit inertial and a mass-less geom already uses body mass/ipos.
    return m


def write_mjcf(p: DemoCParams | None = None, path=SCENE_OUT):
    spec = build_spec(p)
    for g in spec.worldbody.geoms:
        if g.name in ("democ_table_top", "democ_table_pedestal"):
            g.conaffinity |= A.ARM_BITS
    for b in spec.bodies:
        if b.name == "democ_box":
            for g in b.geoms:
                g.conaffinity |= A.ARM_BITS
    spec.compile()
    xml = ("<!-- GENERATED by world_model/v2/nero/democ/scene.py from scenes/yor_nero_derived.mjcf + table + box. Do not hand-edit.\n"
           "     Assumed values: see scene.ASSUMED. -->\n" + spec.to_xml())
    from pathlib import Path
    Path(path).write_text(xml)
    return path


def add_args(ap):
    ap.add_argument("--mass", type=float, default=ASSUMED["box_mass"])
    ap.add_argument("--box-half", type=float, nargs=3, default=list(ASSUMED["box_half"]))
    ap.add_argument("--com-x", type=float, default=0.0)
    ap.add_argument("--com-y", type=float, default=0.0)
    ap.add_argument("--com-z", type=float, default=0.0)
    ap.add_argument("--mu", type=float, default=ASSUMED["box_mu"])
    ap.add_argument("--gravcomp", action="store_true")
    return ap


def params_from_args(a) -> DemoCParams:
    return DemoCParams(box_half=tuple(a.box_half), box_mass=a.mass, box_com_offset=(a.com_x, a.com_y, a.com_z), box_mu=a.mu,
                       gravcomp=a.gravcomp)


def main(argv=None):
    ap = add_args(argparse.ArgumentParser())
    ap.add_argument("--write", action="store_true", help=f"write {SCENE_OUT}")
    a = ap.parse_args(argv)
    p = params_from_args(a)
    m = compose(p)
    if a.write:
        print("wrote", write_mjcf(p))
    print(json.dumps(dict(params=dataclasses.asdict(p), ngeom=m.ngeom, nbody=m.nbody, nv=m.nv, ASSUMED=ASSUMED), default=list, indent=1))


if __name__ == "__main__":
    main()
