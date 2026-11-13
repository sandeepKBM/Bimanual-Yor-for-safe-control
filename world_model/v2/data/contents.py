"""contents.py -- cup CONTENTS randomization + model builders + vectorized spill oracle.

SIMULATION ONLY (MuJoCo, offline). Never edits an existing file: scenes are read
from disk and patched IN MEMORY (MjSpec / XML string).

Nominal values are read from scenes/scene_yor_cup_collect.xml (ball r=0.008 m,
density 900, friction 0.5, solref "0.004 1", 8 balls; cup mass 0.03 kg). The
RANGES below are *sim-diversity* ranges bracketing those nominal values -- they
are NOT real-world specs of any physical cup or payload (none exist in this repo).

Spill oracle == scripts/cup_task_utils.count_spilled geometry (cylinder r=0.030,
h=0.080, margin 0.002 on the bottom) for an arbitrary number of balls, vectorized
(same test as experiments/slosh/rig.py's count_spilled_vec, which was checked
bit-equal against the original).
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "experiments" / "ab_smooth_vs_bangbang"), str(REPO / "eval"),
           str(REPO / "experiments" / "slosh")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SCENE = REPO / "scenes" / "scene_yor_cup_collect.xml"
ROBOT = REPO / "scenes" / "yor_robot_derived.mjcf"
DT = 0.005
CUP_RADIUS, CUP_HEIGHT, H_MARGIN = 0.030, 0.080, 0.002     # == cup_task_utils.count_spilled defaults
CUP_WALL_HALF = 0.0015
CUP_INNER_R = CUP_RADIUS - CUP_WALL_HALF                    # 0.0285 clear radius
CUP_FLOOR_Z = 0.003                                         # top of cup_bottom geom (pos 0.0015 + half 0.0015)
RIM_EXIT_TOL = 0.012                                        # rig.py: first exit must be over the rim

# --- nominal (scene) values ----------------------------------------------------
NOMINAL = dict(n_balls=8, radius=0.008, density=900.0, friction=0.5, dampratio=1.0, cup_mass=0.03)
# --- sim-diversity ranges (bracket the nominal; NOT real-world specs) -----------
# friction capped at 0.8: at >=0.85-0.9 the beads TUNNEL through the 3 mm cup floor under dynamic
# kinematic carry (exit local z ~ -3 mm; contents_study.py); dampratio floored at 0.45: 0.3 is
# numerically unstable (a STATIC cup ejects balls).
RANGES = dict(n_balls=(2, 12), radius=(0.006, 0.010), density=(600.0, 1400.0),
              friction=(0.25, 0.8), dampratio=(0.45, 1.0), cup_mass=(0.015, 0.12))


@dataclass
class Contents:
    n_balls: int = 8
    radius: float = 0.008
    density: float = 900.0
    friction: float = 0.5
    dampratio: float = 1.0        # solref damping ratio: 1 = no bounce; 0.45 ~ restitution 0.2 (0.3 is numerically UNSTABLE: static cups eject balls)
    cup_mass: float = 0.03
    condim: int = 6
    torsional: float = 0.005
    rolling: float = 0.0003
    solref_tc: float = 0.004      # scene value (MuJoCo clamps to >= 2*dt internally)
    solimp: tuple = (0.9, 0.95, 0.001)
    drag: float = 0.0             # [1/s] per-particle drag RELATIVE to the cup (liquid proxy only)
    kind: str = "beads"           # beads | particles (liquid proxy)
    poly: float = 0.0             # polydispersity: per-ball radius = radius*(1 + poly*u), u~U(-1,1) (fixed seed)
    layout: list = field(default_factory=list)   # (n,3) cup-frame ball centres, filled by layout()

    def to_dict(self):
        d = asdict(self); d.pop("layout"); d["solimp"] = list(self.solimp); return d


def sample_contents(rng, nominal_prob=0.15) -> Contents:
    """Random contents within RANGES (a `nominal_prob` share is kept nominal)."""
    if rng.random() < nominal_prob:
        c = Contents(**NOMINAL)
    else:
        for _ in range(100):
            c = Contents(n_balls=int(rng.integers(RANGES["n_balls"][0], RANGES["n_balls"][1] + 1)),
                         radius=float(rng.uniform(*RANGES["radius"])),
                         density=float(rng.uniform(*RANGES["density"])),
                         friction=float(rng.uniform(*RANGES["friction"])),
                         dampratio=float(rng.uniform(*RANGES["dampratio"])),
                         cup_mass=float(np.exp(rng.uniform(np.log(RANGES["cup_mass"][0]),
                                                           np.log(RANGES["cup_mass"][1])))))
            if layout(c) is not None:
                break
    assert layout(c) is not None
    return c


def layout(c: Contents, freeboard=0.012, gap=0.0008):
    """Stacked-ring packing of c.n_balls spheres of c.radius inside the cup.
    Returns (n,3) cup-frame centres, or None if it does not fit below the rim
    minus `freeboard`. For the nominal contents this reproduces the scene's
    4+4 layout (ring radius 12 mm, layers at z=8.5 / 26 mm)."""
    r, n = c.radius * (1.0 + c.poly), c.n_balls
    rc_max = CUP_INNER_R - r - 0.0005
    if rc_max < 0:
        return None
    if n > 12 or c.kind == "particles":           # dense proxies: hex-packed disks, stacked
        s = 2 * r + gap
        cand = [(i * s + (j % 2) * s / 2, j * s * np.sqrt(3) / 2) for j in range(-12, 13) for i in range(-12, 13)]
        disk = sorted([p for p in cand if np.hypot(*p) <= rc_max], key=lambda p: np.hypot(*p))
        pts, z, layer = [], CUP_FLOOR_Z + r + 0.0015, 0
        while len(pts) < n:
            off = (s / 2, s * np.sqrt(3) / 6) if layer % 2 else (0.0, 0.0)
            lay_pts = [(x + off[0], y + off[1]) for x, y in disk if np.hypot(x + off[0], y + off[1]) <= rc_max]
            for x, y in lay_pts[:n - len(pts)]:
                pts.append((x, y, z))
            layer += 1; z += 2 * r * 0.9 + gap
        pts = np.round(np.array(pts), 5)
        return None if pts[:, 2].max() + r > CUP_HEIGHT - freeboard else pts
    nmax = 1
    for k in range(2, 13):
        if (r + gap / 2) / np.sin(np.pi / k) <= rc_max:
            nmax = k
    per = min(nmax, 4 if r >= 0.0075 else (6 if c.kind == "beads" else nmax))
    pts, z, layer, left = [], CUP_FLOOR_Z + r + 0.0025, 0, n
    if abs(r - 0.008) < 1e-9 and per == 4:
        z = 0.0085                                   # scene's declared first-layer height
    while left > 0:
        k = min(per, left)
        rho = 0.0 if k == 1 else max((r + gap / 2) / np.sin(np.pi / k), 0.012 if (per == 4 and r >= 0.0075) else 0.0)
        rho = min(rho, rc_max)
        ph = (np.pi / k) * (layer % 2)
        for i in range(k):
            a = ph + 2 * np.pi * i / k
            pts.append((rho * np.cos(a), rho * np.sin(a), z))
        left -= k; layer += 1; z += 2 * r + (0.0015 if c.kind == "beads" else 0.0005)
    pts = np.round(np.array(pts), 5)   # 1e-5 m grid == the scene's declared decimals
    if pts[:, 2].max() + r > CUP_HEIGHT - freeboard:
        return None
    return pts


def fill_height(c: Contents) -> float:
    p = layout(c)
    return float(p[:, 2].max() + c.radius) if p is not None else float("nan")


def ball_radii(c: Contents):
    if c.poly <= 0:
        return np.full(c.n_balls, c.radius)
    return c.radius * (1.0 + c.poly * np.random.default_rng(12345).uniform(-1, 1, c.n_balls))


def _ball_xml(i, pos, c: Contents, prefix="ball_"):
    ri = ball_radii(c)[i]
    return (f'<body name="{prefix}{i}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}">'
            f'<freejoint name="{prefix}{i}_free"/>'
            f'<geom name="{prefix}{i}" type="sphere" size="{ri:.6f}" material="ball_mat" '
            f'density="{c.density:.3f}" friction="{c.friction:.4f} {c.torsional} {c.rolling}" '
            f'condim="{c.condim}" priority="1" solref="{c.solref_tc} {c.dampratio:.4f}" '
            f'solimp="{c.solimp[0]} {c.solimp[1]} {c.solimp[2]}"/></body>')


def _extract_body(txt, name):
    m = re.search(r'<body name="%s".*?</body>' % re.escape(name), txt, re.S)
    assert m, name
    return m.group(0)


# ============================================================ cup-only rig model
def rig_model(c: Contents, cup_home=(0.0, 0.0, 1.0), extra_xml: str = ""):
    """Cup + contents only (as experiments/slosh/rig.py, but arbitrary contents).
    Cup geometry/contact params + physics <option> copied verbatim from the scene
    files at load time; cup mass scaled to c.cup_mass (inertia scaled likewise)."""
    txt = SCENE.read_text()
    opt = re.search(r"<option[^>]*/>", ROBOT.read_text()).group(0)
    cup = _extract_body(txt, "cup")
    cup = re.sub(r'<body name="cup" pos="[^"]+"', '<body name="cup" pos="%.4f %.4f %.4f"' % tuple(cup_home), cup, count=1)
    s = c.cup_mass / 0.03
    cup = re.sub(r'<inertial pos="0 0 0.03" mass="0.03" diaginertia="2.5e-5 2.5e-5 1.5e-5" />',
                 f'<inertial pos="0 0 0.03" mass="{c.cup_mass:.5f}" diaginertia="{2.5e-5*s:.3e} {2.5e-5*s:.3e} {1.5e-5*s:.3e}" />', cup)
    lay = layout(c)
    assert lay is not None, "contents do not fit"
    c.layout = lay.tolist()
    balls = "\n".join(_ball_xml(i, np.asarray(cup_home) + p, c) for i, p in enumerate(lay))
    xml = f"""<mujoco model="slosh_rig_v2">
  {opt}
  <size memory="64M"/>
  <asset>
    <material name="cup_mat" rgba="0.85 0.15 0.15 0.95"/>
    <material name="ball_mat" rgba="0.15 0.55 0.9 1"/>
  </asset>
  <worldbody>
{cup}
{balls}
{extra_xml}
  </worldbody>
</mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


class Oracle:
    """Vectorized count_spilled for any ball set (cup_task_utils geometry)."""

    def __init__(self, m, n_balls, prefix="ball_"):
        self.bq = np.array([int(m.jnt_qposadr[m.body_jntadr[m.body(f"{prefix}{i}").id]]) for i in range(n_balls)])
        self.bv = np.array([int(m.jnt_dofadr[m.body_jntadr[m.body(f"{prefix}{i}").id]]) for i in range(n_balls)])
        self.bids = np.array([m.body(f"{prefix}{i}").id for i in range(n_balls)])
        self.idx = (self.bq[:, None] + np.arange(3)[None]).ravel()
        self.vidx = (self.bv[:, None] + np.arange(3)[None]).ravel()
        self.n = n_balls

    def local(self, d, cup_pos, cup_quat):
        R = np.empty(9); mujoco.mju_quat2Mat(R, np.asarray(cup_quat, float)); R = R.reshape(3, 3)
        bp = d.qpos[self.idx].reshape(self.n, 3)
        return (bp - cup_pos) @ R

    @staticmethod
    def out_mask(loc):
        return (np.hypot(loc[:, 0], loc[:, 1]) > CUP_RADIUS) | (loc[:, 2] < -H_MARGIN) | (loc[:, 2] > CUP_HEIGHT)


# ---------------------------------------------------------------------------------------- kinematic carrier (fix 2026-10-08)
# pin_cup sets the carried body's pose and velocity BEFORE each mj_step, but the body stays DYNAMIC during the step. A light
# carrier (0.3 kg tray, 0.03 kg cup) therefore reacts to its own contents inside every step: an off-centre load tilts it, the load
# is pushed outward, and the next pin snaps the carrier back. Verified on b_tray_ball: a ball resting at x = 6 / 12 cm on a level,
# still tray drifted OUTWARD by 4.9 / 9.6 mm in 1.2 s (a static tray: 0). Gravity also made the carrier fall 0.25 mm inside every
# step. Fix: the carrier gets a mass and inertia its contents cannot move within a step, plus gravity compensation. Its pose and
# velocity still come only from the pin, so the carry kinematics (inertial carry, tilt) are unchanged.
# BIMANUAL_LEGACY_PIN=1 restores the old behaviour, to reproduce results generated before this fix.
KIN_MASS, KIN_INERTIA = 1.0e4, 1.0e2                      # [kg], [kg m^2]
LEGACY_PIN = os.environ.get("BIMANUAL_LEGACY_PIN", "0") == "1"


def carrier_body(m, cq):
    """body id of the free joint whose qpos starts at address cq."""
    j = np.flatnonzero((m.jnt_qposadr == cq) & (m.jnt_type == mujoco.mjtJoint.mjJNT_FREE))
    assert len(j) == 1, f"no free joint at qpos address {cq}"
    return int(m.jnt_bodyid[j[0]])


def make_carrier_kinematic(m, cq):
    """Make the pinned carrier immovable by its contents (idempotent). Returns True if the model was changed.
    mj_setConst runs on a SCRATCH MjData, so the caller's state is never touched."""
    if LEGACY_PIN:
        return False
    b = carrier_body(m, cq)
    if m.body_mass[b] == KIN_MASS and m.body_gravcomp[b] == 1.0 and np.all(m.body_inertia[b] == KIN_INERTIA):
        return False
    m.body_mass[b] = KIN_MASS
    m.body_inertia[b] = KIN_INERTIA
    m.body_gravcomp[b] = 1.0
    mujoco.mj_setConst(m, mujoco.MjData(m))               # invweight0 (contact impedance) and ngravcomp follow the new values
    assert m.ngravcomp >= 1, "gravity compensation did not take effect"
    return True


def pin_cup(m, d, cq, cv, pos, quat, prev_qpos):
    """Kinematic pin + INERTIAL carry (== rig.py / scene_setup.cup_velocity_from_motion).
    The carrier is made kinematic on first use (make_carrier_kinematic; BIMANUAL_LEGACY_PIN=1 skips it)."""
    make_carrier_kinematic(m, cq)
    d.qpos[cq:cq + 3] = pos
    d.qpos[cq + 3:cq + 7] = quat
    if prev_qpos is None:
        d.qvel[cv:cv + 6] = 0.0
    else:
        qv = np.zeros(m.nv)
        mujoco.mj_differentiatePos(m, qv, DT, prev_qpos, d.qpos)
        d.qvel[cv:cv + 6] = qv[cv:cv + 6]


def run_rig(m, c: Contents, pos, quat, settle_steps=80, cup_home=(0.0, 0.0, 1.0), drag=0.0, want_state=False):
    """Drive the cup-only rig along scripted (pos,quat) [T]; balls free.
    The first `settle_steps` are expected to be static (ball settle).
    Returns dict(spilled (T,) int8, exit_step, exit_z, tunnel, wall_s)."""
    import time
    d = mujoco.MjData(m)
    cup = m.body("cup").id
    cj = m.body_jntadr[cup]; cq, cv = int(m.jnt_qposadr[cj]), int(m.jnt_dofadr[cj])
    orc = Oracle(m, c.n_balls)
    mujoco.mj_resetData(m, d)
    d.qpos[orc.idx] += np.tile(pos[0] - np.asarray(cup_home), c.n_balls)
    T = len(pos)
    spilled = np.zeros(T, np.int8)
    exit_step = -np.ones(c.n_balls, int); exit_z = np.full(c.n_balls, np.nan)
    prev = None
    t0 = time.perf_counter()
    bmass = m.body_mass[orc.bids][:, None]
    for k in range(T):
        pin_cup(m, d, cq, cv, pos[k], quat[k], prev)
        prev = d.qpos.copy()
        loc = orc.local(d, pos[k], quat[k])
        out = Oracle.out_mask(loc)
        new = out & (exit_step < 0)
        exit_step[new] = k; exit_z[new] = loc[new, 2]
        spilled[k] = int(out.sum())
        if drag > 0:   # liquid proxy: viscous drag relative to the cup's (translational) velocity
            vrel = d.qvel[orc.vidx].reshape(-1, 3) - d.qvel[cv:cv + 3]
            d.xfrc_applied[orc.bids, :3] = -drag * bmass * vrel
        mujoco.mj_step(m, d)
    wall = time.perf_counter() - t0
    rim_exit = exit_z > (CUP_HEIGHT - RIM_EXIT_TOL)
    # rim-only spill curve: count only balls whose FIRST exit was over the rim (tunnels excluded)
    ok = (exit_step >= 0) & rim_exit
    spilled_rim = np.zeros(T, np.int16)
    for s_ in exit_step[ok]:
        spilled_rim[s_:] += 1
    return dict(spilled=spilled, spilled_rim=spilled_rim, exit_step=exit_step, exit_z=exit_z,
                tunnel=bool(np.any((exit_step >= 0) & ~rim_exit)), wall_s=wall, ms_per_step=1e3 * wall / T)


# ============================================================ full robot scene
def scene_model(c: Contents, keep_cubes=True):
    """Full YOR cup_collect scene (zed_cam + arm-arm collision patch, exactly as
    cup_collect_task_utils.load_yor_cup_collect_model does) with the 8 declared
    balls REPLACED by `c` (in memory). The 'home' keyframe is dropped (its qpos
    length is ball-count specific); use home_state() instead.
    Returns (model, home_qpos_by_joint dict)."""
    import sim_common
    from scripts.zed_camera import add_zed_camera
    spec = mujoco.MjSpec.from_file(str(SCENE))
    key = spec.keys[0]
    home_q = np.array(key.qpos, float).copy()
    home_ctrl = np.array(key.ctrl, float).copy()
    add_zed_camera(spec)
    sim_common._patch_real_collision_geoms(spec)
    # record joint-name -> home qpos slice from the ORIGINAL layout before editing
    m0 = mujoco.MjSpec.from_file(str(SCENE)).compile()
    home = {}
    for j in range(m0.njnt):
        nm = m0.joint(j).name
        a = int(m0.jnt_qposadr[j]); w = {0: 7, 1: 4, 2: 1, 3: 1}[int(m0.jnt_type[j])]
        home[nm] = home_q[a:a + w].copy()
    for i in range(8):
        spec.delete(spec.body(f"ball_{i}"))
    for _k in list(spec.keys):   # MjSpec may already drop keys when bodies are deleted
        spec.delete(_k)
    cupb = spec.body("cup")
    s = c.cup_mass / 0.03
    cupb.mass = c.cup_mass
    cupb.inertia = np.array([2.5e-5, 2.5e-5, 1.5e-5]) * s
    lay = layout(c); assert lay is not None
    c.layout = lay.tolist()
    wb = spec.worldbody
    radii = ball_radii(c)
    for i, p in enumerate(lay):
        b = wb.add_body(name=f"ball_{i}", pos=[0.42 + p[0], -0.21 + p[1], 0.55 + p[2]])
        b.add_freejoint(name=f"ball_{i}_free")
        g = b.add_geom(name=f"ball_{i}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(radii[i]), 0, 0],
                       density=c.density, friction=[c.friction, c.torsional, c.rolling], condim=c.condim,
                       priority=1, solref=[c.solref_tc, c.dampratio], solimp=list(c.solimp) + [0.5, 2.0])
        g.material = "ball_mat"
    if not keep_cubes:
        for i in range(6):
            try:
                spec.delete(spec.body(f"cube_{i}"))
            except Exception:
                pass
    model = spec.compile()
    return model, home, home_ctrl
