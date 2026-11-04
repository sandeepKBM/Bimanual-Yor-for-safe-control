"""pan/scene.py -- pan-serving scene v0 (step 1 of the pan demo): YOR-NERO + table + stove pad + a light 3D-printed PLA pan with a
designed handle + loose food beads + a PLA spatula + a serving plate.  SIMULATION ONLY.  Nothing existing is modified.

USER 10-28: the pan is a light 3D-printed plastic (PLA) pan, so the NERO claw (10 N, URDF) can plausibly hold it; the hidden food
amount then decides whether the grip holds (that is the regime the claims need).

Geometry is built from exact MuJoCo primitives (disk bottom = cylinder, wall = N box segments, handle = box), so contact is exact
(no convex decomposition).  Mass and inertia come from geometry x PLA density (MuJoCo computes them per geom).  Food = N free
spheres; amount and position are the hidden parameters.  Grasp (step 2, not here): left claw pinches the handle top/bottom
(jaw axis z), approaching along -y from the handle end; the site `pan_handle_grasp` marks the pinch point.

    python -m world_model.v2.nero.pan.scene [--food-n 8] [--food-mass 0.02] [--settle 2.0] [--reach] [--render out.png]
Every value not taken from a real spec is listed in ASSUMED below (one place).
"""
from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, field
import json

import numpy as np
import mujoco

from .. import config as C
from .. import build_variant as BV

SCENE_OUT = C.scene_path("yor_nero_pan")
G = 9.81

ASSUMED = dict(                        # EVERYTHING here is an assumption (no real spec); change only here / via the dataclass
    pla_density=1240.0,                # kg/m^3, solid PLA (100% infill); a real print with infill is lighter
    pan_radius=0.09,                   # m, inner radius of the wall (18 cm pan)
    pan_wall_h=0.04, pan_wall_t=0.0025, pan_bottom_t=0.0025, pan_wall_segments=24,
    handle_len=0.13, handle_w=0.024, handle_t=C.HANDLE_T,   # m; cross-section 24 x 14 mm (claw stroke 100 mm, so fits)
    handle_dir_deg=90.0,               # handle points along +y (towards the left arm, which approaches along -y)
    handle_grasp_frac=0.6,             # pinch point at 60% of the handle length from the wall
    pan_mu=0.6,                        # PLA vs pad/table sliding friction (MuJoCo uses max(geom1, geom2))
    pan_centre_xy=(0.38, -0.30),       # robot-local; chosen by a reach sweep (left claw on the handle, rest + lifted, no arm-table contact)
    table_top_z=0.76,                  # m (same as democ); the stove pad raises the pan so the claw's lower half clears the table
    table_half=(0.30, 0.45, 0.01), table_centre_xy=(0.50, -0.21), table_mu=0.8,
    stove_half=(0.12, 0.12, 0.03),     # 6 cm stove/hob block under the pan's start position (where set-down recoveries go)
    plate_centre_xy=(0.40, -0.02), plate_radius=0.08, plate_h=0.012,
    food_n=8, food_radius=0.012, food_mass=0.02, food_mu=0.5, food_spread=0.045,   # 8 x 20 g = 160 g nominal
    spatula_centre_xy=(0.40, -0.56), spatula_handle=(0.20, 0.022, 0.008), spatula_blade=(0.08, 0.06, 0.002),
    spatula_rest=(0.03, 0.012, 0.04),  # two narrow knife-rest blocks (x len, y width, height) so the claw fingers straddle the handle
)


@dataclass
class PanParams:
    pla_density: float = ASSUMED["pla_density"]
    pan_radius: float = ASSUMED["pan_radius"]
    pan_wall_h: float = ASSUMED["pan_wall_h"]
    pan_wall_t: float = ASSUMED["pan_wall_t"]
    pan_bottom_t: float = ASSUMED["pan_bottom_t"]
    pan_wall_segments: int = ASSUMED["pan_wall_segments"]
    handle_len: float = ASSUMED["handle_len"]
    handle_w: float = ASSUMED["handle_w"]
    handle_t: float = ASSUMED["handle_t"]
    handle_dir_deg: float = ASSUMED["handle_dir_deg"]
    handle_grasp_frac: float = ASSUMED["handle_grasp_frac"]
    handle2_len: float = 0.0           # 11-01 pot variant (USER: two side handles): second handle opposite the first; 0 = none (pan)
    handle2_grasp_frac: float = 0.7    # ASSUMED: right pinch point at 70 % of handle2 from the wall (past the stove edge, reach)
    pan_mu: float = ASSUMED["pan_mu"]
    pan_centre_xy: tuple = ASSUMED["pan_centre_xy"]
    table_top_z: float = ASSUMED["table_top_z"]
    table_half: tuple = ASSUMED["table_half"]
    table_centre_xy: tuple = ASSUMED["table_centre_xy"]
    table_mu: float = ASSUMED["table_mu"]
    stove_half: tuple = ASSUMED["stove_half"]
    plate_centre_xy: tuple = ASSUMED["plate_centre_xy"]
    plate_radius: float = ASSUMED["plate_radius"]
    plate_h: float = ASSUMED["plate_h"]
    food_n: int = ASSUMED["food_n"]
    food_radius: float = ASSUMED["food_radius"]
    food_mass: float = ASSUMED["food_mass"]
    food_mu: float = ASSUMED["food_mu"]
    food_spread: float = ASSUMED["food_spread"]
    food_seed: int = 0
    spatula_centre_xy: tuple = ASSUMED["spatula_centre_xy"]
    spatula_handle: tuple = ASSUMED["spatula_handle"]
    spatula_blade: tuple = ASSUMED["spatula_blade"]
    spatula_rest: tuple = ASSUMED["spatula_rest"]
    with_spatula: bool = True          # covered-pan scene (covered.py) turns these off
    with_plate: bool = True
    gravcomp: bool = False             # never gravity-compensate the held object (claims doc, 10-26)
    tau_preset: str | None = None

    def pan_base_z(self):
        """z of the pan bottom's underside when resting on the stove pad."""
        return self.table_top_z + 2 * self.stove_half[2]

    def handle_axis(self):
        a = np.radians(self.handle_dir_deg)
        return np.array([np.cos(a), np.sin(a), 0.0])


def _pan_local(p: PanParams):
    """Pan geometry in the pan body frame (origin = centre of the bottom underside). Returns list of geom dicts + grasp point."""
    R, t, h, tb = p.pan_radius, p.pan_wall_t, p.pan_wall_h, p.pan_bottom_t
    geoms = [dict(name="pan_bottom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[R + t, tb / 2, 0], pos=[0, 0, tb / 2],
                  quat=[1, 0, 0, 0])]
    n = p.pan_wall_segments
    seg_len = 2 * (R + t / 2) * np.tan(np.pi / n) + 1e-4          # chord length so segments meet (tiny overlap)
    for i in range(n):
        th = 2 * np.pi * i / n
        c = np.array([np.cos(th), np.sin(th)]) * (R + t / 2)
        q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, [0, 0, 1], th)
        geoms.append(dict(name=f"pan_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[t / 2, seg_len / 2, h / 2],
                          pos=[c[0], c[1], tb + h / 2], quat=q))
    ax = p.handle_axis()
    yaw = np.radians(p.handle_dir_deg)
    q = np.zeros(4); mujoco.mju_axisAngle2Quat(q, [0, 0, 1], yaw)
    hc = ax * (R + t + p.handle_len / 2)
    hz = tb + h - p.handle_t / 2                                    # handle top flush with the rim
    geoms.append(dict(name="pan_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[p.handle_len / 2, p.handle_w / 2, p.handle_t / 2],
                      pos=[hc[0], hc[1], hz], quat=q))
    grasp = ax * (R + t + p.handle_grasp_frac * p.handle_len) + np.array([0, 0, hz])
    if p.handle2_len > 0:                                           # pot: same cross-section, pointing the other way
        hc2 = -ax * (R + t + p.handle2_len / 2)
        geoms.append(dict(name="pan_handle2", type=mujoco.mjtGeom.mjGEOM_BOX, size=[p.handle2_len / 2, p.handle_w / 2, p.handle_t / 2],
                          pos=[hc2[0], hc2[1], hz], quat=q))
    return geoms, grasp


def handle2_local(p):
    """pinch point on the second handle (pot variant) in the pan body frame."""
    R, t, h, tb = p.pan_radius, p.pan_wall_t, p.pan_wall_h, p.pan_bottom_t
    return -p.handle_axis() * (R + t + p.handle2_grasp_frac * p.handle2_len) + np.array([0, 0, tb + h - p.handle_t / 2])


def build_spec(p: PanParams | None = None):
    """-> MjSpec (not yet compiled)."""
    p = p or PanParams()
    tau = C.tau_limits(p.tau_preset)
    spec = mujoco.MjSpec.from_string(BV.variant_xml(lift=False, gravcomp=p.gravcomp, tau=tau))
    W = spec.worldbody
    W.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], contype=1, conaffinity=1)
    tab = np.array(p.table_half, float)
    cx, cy = p.table_centre_xy
    W.add_geom(name="pan_table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(tab), pos=[cx, cy, p.table_top_z - tab[2]],
               contype=1, conaffinity=1, friction=[p.table_mu, 0.005, 0.0001], rgba=[0.55, 0.45, 0.35, 1])
    for i, (sx, sy) in enumerate(((1, 1), (1, -1), (-1, 1), (-1, -1))):   # visual legs
        W.add_geom(name=f"pan_table_leg{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02, 0.02, (p.table_top_z - 2 * tab[2]) / 2],
                   pos=[cx + sx * (tab[0] - 0.03), cy + sy * (tab[1] - 0.03), (p.table_top_z - 2 * tab[2]) / 2],
                   contype=0, conaffinity=0, rgba=[0.4, 0.35, 0.3, 1])
    px, py = p.pan_centre_xy
    W.add_geom(name="stove_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=list(p.stove_half),
               pos=[px, py, p.table_top_z + p.stove_half[2]], contype=1, conaffinity=1,
               friction=[p.table_mu, 0.005, 0.0001], rgba=[0.15, 0.15, 0.15, 1])
    if p.with_plate:
        W.add_geom(name="plate", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[p.plate_radius, p.plate_h / 2, 0],
                   pos=[p.plate_centre_xy[0], p.plate_centre_xy[1], p.table_top_z + p.plate_h / 2], contype=1, conaffinity=1,
                   friction=[p.table_mu, 0.005, 0.0001], rgba=[0.95, 0.95, 0.92, 1])

    # pan (free body, PLA density -> mass/inertia from geometry)
    pan = W.add_body(name="pan", pos=[px, py, p.pan_base_z() + 1e-4])
    pan.add_freejoint(name="pan_free")
    geoms, grasp = _pan_local(p)
    for g in geoms:
        pan.add_geom(name=g["name"], type=g["type"], size=g["size"], pos=g["pos"], quat=g["quat"], density=p.pla_density,
                     friction=[p.pan_mu, 0.005, 0.0001], contype=1, conaffinity=1,
                     rgba=[0.2, 0.45, 0.8, 1] if not g["name"].startswith("pan_handle") else [0.15, 0.3, 0.6, 1])
    pan.add_site(name="pan_handle_grasp", pos=list(grasp), size=[0.006, 0, 0], rgba=[1, 0.2, 0.2, 1])
    if p.handle2_len > 0:
        pan.add_site(name="pan_handle2_grasp", pos=list(handle2_local(p)), size=[0.006, 0, 0], rgba=[1, 0.2, 0.2, 1])
    pan.add_site(name="pan_centre", pos=[0, 0, p.pan_bottom_t], size=[0.004, 0, 0], rgba=[0.2, 1, 0.2, 1])

    # food beads: hidden amount and position (seeded positions inside the pan, resting on the bottom)
    rng = np.random.default_rng(p.food_seed)
    for i in range(p.food_n):
        for _ in range(200):
            r, a = p.food_spread * np.sqrt(rng.random()), 2 * np.pi * rng.random()
            xy = np.array([r * np.cos(a), r * np.sin(a)])
            if i == 0 or all(np.linalg.norm(xy - q) > 2.05 * p.food_radius for q in _placed):
                break
        _placed = ([] if i == 0 else _placed) + [xy]
        b = W.add_body(name=f"food{i}", pos=[px + xy[0], py + xy[1],
                                            p.pan_base_z() + p.pan_bottom_t + p.food_radius + 1e-3])
        b.add_freejoint(name=f"food{i}_free")
        b.add_geom(name=f"food{i}_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[p.food_radius, 0, 0], mass=p.food_mass,
                   friction=[p.food_mu, 0.005, 0.0001], contype=1, conaffinity=1, rgba=[0.85, 0.6, 0.2, 1])

    if p.with_spatula:
        _add_spatula(W, p)
    return spec


def _add_spatula(W, p):
    # spatula (PLA): handle + thin blade on two narrow rests near the right arm (top-down claw, fingers straddle the handle)
    sx, sy = p.spatula_centre_xy
    hl, hw, ht = p.spatula_handle
    bl, bw, bt = p.spatula_blade
    rl, rw, rh = p.spatula_rest
    for nm, rx in (("spatula_rest_handle", sx - hl / 2 + rl / 2 + 0.005), ("spatula_rest_blade", sx + hl / 2 + bl / 2)):
        W.add_geom(name=nm, type=mujoco.mjtGeom.mjGEOM_BOX, size=[rl / 2, rw / 2, rh / 2], pos=[rx, sy, p.table_top_z + rh / 2],
                   contype=1, conaffinity=1, friction=[p.table_mu, 0.005, 0.0001], rgba=[0.3, 0.3, 0.3, 1])
    sp = W.add_body(name="spatula", pos=[sx, sy, p.table_top_z + rh + ht / 2 + 1e-4])
    sp.add_freejoint(name="spatula_free")
    sp.add_geom(name="spatula_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hl / 2, hw / 2, ht / 2], density=p.pla_density,
                friction=[p.pan_mu, 0.005, 0.0001], rgba=[0.9, 0.4, 0.2, 1])
    sp.add_geom(name="spatula_blade", type=mujoco.mjtGeom.mjGEOM_BOX, size=[bl / 2, bw / 2, bt / 2], density=p.pla_density,
                pos=[hl / 2 + bl / 2, 0, -ht / 2 + bt / 2], friction=[p.pan_mu, 0.005, 0.0001], rgba=[0.9, 0.4, 0.2, 1])
    sp.add_site(name="spatula_grasp", pos=[0.02, 0, 0], size=[0.005, 0, 0], rgba=[1, 0.2, 0.2, 1])


def apply_sim_defaults(m):
    """11-01 USER: the pan/pot scenes run the NERO claws at config.claw_default() (40 N; MJCF says 10 N). In memory only."""
    f = float(C.claw_default())
    for s in C.SIDES:
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{s}_arm_gripper_pos")
        if a >= 0:
            m.actuator_forcerange[a] = (-f, f)
    return m


def step(m, d, nsub=None):
    """one control step of m.opt.timestep as nsub physics substeps (applied torque/ctrl held). nsub None = config.substeps_default()."""
    n = C.substeps_default() if nsub is None else int(nsub)
    if n <= 1:
        mujoco.mj_step(m, d)
        return
    dt = m.opt.timestep
    m.opt.timestep = dt / n
    try:
        for _ in range(n):
            mujoco.mj_step(m, d)
    finally:
        m.opt.timestep = dt


def compose(p: PanParams | None = None, dt: float | None = None):
    spec = build_spec(p)
    m = spec.compile()
    if dt is not None:
        m.opt.timestep = dt
    return apply_sim_defaults(m)


def write_mjcf(p: PanParams | None = None, path=SCENE_OUT):
    spec = build_spec(p)
    spec.compile()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(spec.to_xml())
    return path


def handle_moment(m, d):
    """Gravity moment [Nm] the claw must resist at the pinch point, about the horizontal axis normal to the handle, for the pan +
    every food bead whose centre is inside the pan footprint (current state).  Also returns the masses used."""
    pan = m.body("pan").id
    gp = d.site_xpos[m.site("pan_handle_grasp").id]
    hb = m.geom("pan_handle").id
    ax = d.geom_xmat[hb].reshape(3, 3)[:, 0]                                 # handle long axis (box local x)
    ax[2] = 0.0; ax /= np.linalg.norm(ax)
    centre = d.site_xpos[m.site("pan_centre").id]
    m_pan = float(m.body_mass[pan]); com_pan = d.xipos[pan]
    tot = m_pan * G * float(np.dot(gp - com_pan, ax))
    m_food, inside = 0.0, 0
    for i in range(m.nbody):
        nm = m.body(i).name
        if nm.startswith("food"):
            c = d.xpos[i]
            if np.linalg.norm((c - centre)[:2]) < 0.12:
                mf = float(m.body_mass[i]); m_food += mf; inside += 1
                tot += mf * G * float(np.dot(gp - c, ax))
    return dict(moment_Nm=tot, m_pan=m_pan, m_food_in_pan=m_food, beads_in_pan=inside,
                lever_pan_com_m=float(np.dot(gp - com_pan, ax)))


def settle(m, d, seconds=2.0):
    mujoco.mj_forward(m, d)
    pan = m.body("pan").id
    p0 = d.xpos[pan].copy()
    for _ in range(int(seconds / m.opt.timestep)):
        step(m, d)
    return float(np.linalg.norm(d.xpos[pan] - p0))


def reach_report(m, d, lift=0.08, n_sol=8, seeds=40):
    """Can the left claw reach the handle pinch (approach -y, jaw z) at rest and lifted by `lift`; right claw the spatula grasp
    (approach -z top-down, jaw y = across the handle).  Uses nero.ik.ArmIK.collect (7-DoF, posture nullspace)."""
    from ..ik import ArmIK
    out = {}
    gp = d.site_xpos[m.site("pan_handle_grasp").id].copy()
    left = ArmIK(m, "left")
    for tag, z in (("handle_rest", 0.0), ("handle_lifted", lift)):
        sols = left.collect(gp + [0, 0, z], np.array([0.0, -1.0, 0.0]), np.array([0.0, 0.0, 1.0]), n_sol=n_sol, seeds=seeds)
        out[tag] = dict(target=(gp + [0, 0, z]).round(4).tolist(), n_solutions=len(sols))
    sg = d.site_xpos[m.site("spatula_grasp").id].copy()
    right = ArmIK(m, "right")
    sols = right.collect(sg + [0, 0, 0.0], np.array([0.0, 0.0, -1.0]), np.array([0.0, 1.0, 0.0]), n_sol=n_sol, seeds=seeds)
    out["spatula"] = dict(target=sg.round(4).tolist(), n_solutions=len(sols))
    return out


def render(m, d, path, w=640, h=480):
    r = mujoco.Renderer(m, h, w)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0.45, -0.25, 0.82]; cam.distance, cam.azimuth, cam.elevation = 1.1, 200, -35
    r.update_scene(d, camera=cam)
    img = r.render()
    import imageio.v2 as iio
    iio.imwrite(path, img)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--food-n", type=int, default=ASSUMED["food_n"])
    ap.add_argument("--food-mass", type=float, default=ASSUMED["food_mass"])
    ap.add_argument("--food-seed", type=int, default=0)
    ap.add_argument("--settle", type=float, default=2.0)
    ap.add_argument("--reach", action="store_true")
    ap.add_argument("--render", default=None)
    ap.add_argument("--write", action="store_true", help=f"write {SCENE_OUT}")
    a = ap.parse_args(argv)
    p = PanParams(food_n=a.food_n, food_mass=a.food_mass, food_seed=a.food_seed)
    m = compose(p)
    d = mujoco.MjData(m)
    drift = settle(m, d, a.settle)
    rep = dict(params=dataclasses.asdict(p), nbody=m.nbody, nv=m.nv, timestep=m.opt.timestep,
               pan_settle_drift_m=drift, moment=handle_moment(m, d),
               claw_effort_N=C.GRIPPER_EFFORT)
    if a.reach:
        rep["reach"] = reach_report(m, d)
    if a.render:
        rep["render"] = str(render(m, d, a.render))
    if a.write:
        rep["wrote"] = str(write_mjcf(p))
    print(json.dumps(rep, default=lambda o: list(o) if isinstance(o, tuple) else str(o), indent=1))


if __name__ == "__main__":
    main()
