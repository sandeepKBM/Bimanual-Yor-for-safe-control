"""nyu_gripper.py -- the "NERO arm + NYU gripper (angular jaw) + wrist camera" MJCF build (the only gripper build since 11-03).

Base: the NERO arm fragments of urdf_to_mjcf.py (arm links only; its gripper_base/fingers are swapped out below).  assets_nero/nero_arm.xml is not edited.

    python -m world_model.v2.nero.nyu_gripper split    # YOR gripper_body.stl -> assets_nero/nyu_gripper/meshes/{palm,finger_a,finger_b}.stl
    python -m world_model.v2.nero.nyu_gripper write    # -> assets_nero/nero_arm_nyugrip.xml (relative mesh paths)

Source mesh: YOR/robot/arm/mujoco/assets/gripper_body.stl (YOR-robot/YOR, read-only clone; commit in assets_nero/nyu_gripper/README.md).
YOR's own MJCF freezes the fingers open (no finger joint).  The finger MECHANISM is taken from the geometry (two hinge holes, jaws
that open as a V) and from arXiv 2602.11150 Sec 3.4: "we design custom grippers with an angular jaw".  The paper gives no pivot, range,
actuator or force, so those are ASSUMED/derived below.  Every number is tagged MEASURED (from the mesh), MODEL (MJCF/physics) or ASSUMED.
Hardware safety: this file only reads STL/XML text; it never imports YOR robot code.
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import config as C
from . import urdf_to_mjcf as M

SRC_STL = Path(os.environ.get("YOR_GRIPPER_STL", "/common/users/ss5772/bimanual/YOR/robot/arm/mujoco/assets/gripper_body.stl"))
OUT_DIR = C.ASSETS / "nyu_gripper" / "meshes"
ARM_NYUGRIP_OUT = C.ASSETS / "nero_arm_nyugrip.xml"

FINGER_X_MIN_MM = -100.0   # MEASURED gripper_body.stl: of the 97 connected components (vertices merged 1e-6 m) only the 2 jaws reach x < -100 mm (palm min x = -55.7, jaws -138)
MESH_SCALE = 1e-3           # YOR MJCF: <mesh name="gripper_body" scale="0.001 0.001 0.001"> (mm -> m)  MODEL

# ---- YOR MJCF (piper_left.xml, body gripper_base) -- MODEL
YOR_MASS = 0.45                                           # kg, whole gripper_body incl. jaws and pole
YOR_COM = np.array([-0.000183807, 8.05033e-05, 0.0321437])  # in the mesh frame
YOR_QUAT_I = np.array([0.999976, 9.08444e-05, 0.00689649, 0.000788512])
YOR_DIAGI = np.array([0.000929442, 0.000714469, 0.000394318])
NYU_EULER_XYZ = (np.pi, -np.pi / 2, 0.0)    # ASSUMED idealisation of YOR's (3.14, -1.57, 0): exact pi, pi/2 (0.0008 rad difference)

# ---- hinge geometry from the mesh (MEASURED from the section of each jaw at z=29 mm: the small closed loop = pivot hole)
# filled by hinge_points() at import time from the STL when it is readable; fallback numbers = result of that fit (mm, mesh frame)
HINGE_A_MM = (-49.29, -5.59)    # MEASURED nyu_gripper.hinge_points(): centre of the smallest closed loop of jaw A at section z=29 mm (n=1 mesh)
HINGE_B_MM = (-49.33, 42.44)   # MEASURED same, jaw B
JAW_Y_MID_MM = 18.42          # MEASURED: mean of the two hinge y (mirror plane between the jaws)
JAW_Z_MID_MM = 29.0            # MEASURED: mid of the jaw z span (12..46 mm)
PALM_X_MAX_MM = 73.2           # MEASURED: max x of the mesh (mounting standoffs, the face the Piper flange touches in YOR: z=0.0798-x ~ 7 mm)

# ---- ASSUMED: joint + actuator + camera + adapter
# ---- mechanism (owner, 2026-11-03, from gripper_body.stl): XL430 -> coaxial lead screw along mesh x -> nut with a pin -> pin in a straight cam
# slot in each jaw's root plate -> each jaw = HINGE about mesh z at HINGE_A/B.  MEASURED (mesh geometry) unless tagged MODEL/ASSUMED.
# Caveat (owner): the helix could be a spring rather than a screw; the owner reads it as a lead screw + nut.
NUT_AXIS_YZ_MM = (18.2, 29.5)  # MEASURED screw axis in the mesh frame (y, z)
NUT_PIN_X_OPEN_MM = -45.6      # MODEL pin x at the open (mesh) pose, pin-in-slot solved numerically by the owner
NUT_TRAVEL_M = 0.0219          # MODEL pin travel u = x - x_open (closing pulls the pin +x toward the motor); theta(u) valid 0..NUT_TRAVEL_M
THETA_POLY = (-0.000103, 43.8231, 39.0158, -28705.2, 503607.0)   # MODEL: jaw angle [rad] = sum a_k u^k, u in m; max fit err 0.011 deg; each jaw, symmetric
THETA_MAX_RAD = 0.793          # MODEL: 45.4 deg at u = NUT_TRAVEL_M (jaw inner edges reach the centre plane)
NYU_JOINT_RANGE_RAD = 0.82     # ASSUMED: jaw hinge joint range (slightly above THETA_MAX_RAD; the equality drives the jaws, this is only a limit)
# ---- squeeze is NOT modelled (user 11-03: grasps stay the rigid weld; this gripper model is for the WRIST CAMERAS only: what they see and where they sit).
# Owner's estimate 25-44 N at the fingertip, see memory.  The nut actuator below only sets the jaw POSE kinematically.
NUT_FORCE_N = C.NUT_FORCE_N    # 231 N nut force limit, MODEL (owner 11-03): XL430 stall 1.4 N m (datasheet) x PWM cap 300/885 (USER 11-03: cap active) x
                               # lead-screw efficiency 0.48 (friction coeff 0.2 ASSUMED) x 2 pi / 6.2 mm lead (MEASURED from the mesh) = 231 N -> ~25-44 N per jaw at the tip.
                               # Replaces the 500 N pose-servo ASSUMED of the first build (11-03). Only matters for friction-pinch scenes (weld scenes unaffected).
NUT_KP = 1.0e4                 # ASSUMED N/m pose-servo gain (tuned for stability with the 1 ms step; 1e5 and kv>=300 oscillated in the smoke)
NUT_KV = 100.0                 # ASSUMED N s/m
NUT_ARMATURE = 0.01            # ASSUMED kg (slide joint reflected inertia)
NUT_MASS_KG = 0.005            # ASSUMED nut + pin mass, taken out of the 0.45 kg palm share
NYU_FINGER_DAMPING = 0.2       # ASSUMED Nm s/rad on the jaw hinges
NYU_FINGER_ARMATURE = 0.005    # ASSUMED kg m^2 on the jaw hinges
NYU_ADAPTER_Z = 0.0            # ASSUMED: flange face -> palm mounting standoffs, 0 = flush (NERO's own AgileX adapter uses 6 mm)
# ---- wrist camera = Intel RealSense D435i (USER 11-03: a RealSense on the wrist protrusion facing the fingers; not an iPhone). Only the camera IMAGES matter:
# no D435i body/mesh and no mass is added (D435i 72 g not added); the two MuJoCo cameras sit on gripper_base.
D435I_PUPIL_B = (0.03249, 0.00003, -0.00362)   # MEASURED (Menagerie d435i_7 RGB-pupil vertex mean, D435i frame [m]; used as numbers only, mesh not referenced). front = +z, width = x
D435I_IRL_B = (-0.0325, 0.0, -0.00125)         # MEASURED (Menagerie d435i_0/_3 vertex means; numbers only), imager at x<0 = the one farthest from the RGB (= left imager seen from the front)
D435I_POS_MM = (28.5, 15.0, 117.0)   # MODEL YOR piper_left.xml phone position in the gripper_base mesh frame; RGB pupil placed here. That the RealSense sits at the phone spot = ASSUMED
RGB_FOVY = 42.5                # datasheet (Intel support: RGB FOV 69.4 x 42.5 deg, 16:9), +/-3 deg per unit
DEPTH_FOVY = 57.0              # datasheet (depth FOV 86 x 57 deg), +/-3 deg per unit
CAM_RES = (640, 360)           # 16:9 default render size for both cameras (named constant)
CAM_TILT_DEG = 16.0            # MEASURED 11-03: YOR piper_left.xml iphone mesh as placed in gripper_base -- area-weighted normal of its back face (9663 mm2) = (-0.961, 0.001, -0.276) in gb:
                               # along the jaws, 16.0 deg down, 0.0 deg yaw (n = 1 assembly); YOR's own camera_body euler encodes 15 deg (on a wrong base axis); RUM/YOR papers state no tilt;
                               # RUM mount CAD = Google Drive 1OfebxMiP9Sol14l0f4rorHZhh9PptDwe, copied to /common/users/ss5772/bimanual_data/cad/rum_mount_2026-11-03 (parts not assembled,
                               # so the tilt was taken from the YOR assembly).  Supersedes the 20 deg ASSUMED of the first build.  Direction along the jaws = USER 11-03; position = YOR phone spot (USER 11-03).
# ---- grasp point (TCP, USER-approved 11-03): between the jaw pads at TCP_FRAC of the pivot-to-tip distance, measured along the approach axis.
JAW_PIVOT_YZ_M = (0.0240, 0.1225)   # MEASURED/MODEL: jaw A hinge in gripper_base (y = half the 48 mm pivot spacing, z = depth from the flange; nyu_gripper.mesh_to_base(HINGE))
JAW_TIP_Z_OPEN_M = 0.2110           # MEASURED jaw mesh (finger_a/_b.stl through mesh_to_base): max z of the open jaw tip (A 0.2110, B 0.2120)
TCP_FRAC = 0.40                     # ASSUMED (USER 11-03: "about 40 %"): TCP depth = pivot z + TCP_FRAC (tip z - pivot z)
JAW_INNER_Y160_M = 0.0425           # MEASURED jaw meshes: |y| of the inner (pad) edge of the OPEN jaw at depth z = 0.160 m (A 0.0434, B 0.0417; mean), |x| < 8 mm
JAW_INNER_ANGLE_RAD = 0.589         # MEASURED: angle of the straight inner edge from the approach axis at the open pose (A 0.598, B 0.580; fit residual <= 2.2 mm)
GRASP_CLEARANCE_M = 0.002           # ASSUMED (USER 11-03 example): clearance per side between a pad and the grasped feature at the TCP
EE_GRASP_Z = JAW_PIVOT_YZ_M[1] + TCP_FRAC * (JAW_TIP_Z_OPEN_M - JAW_PIVOT_YZ_M[1])   # 0.1579 m (was 0.165 = 50 %, ASSUMED placeholder)


def _load():
    import trimesh
    m = trimesh.load(str(SRC_STL), process=True)
    comps = sorted(m.split(only_watertight=False), key=lambda c: -len(c.faces))
    fingers = [c for c in comps if c.bounds[0, 0] < FINGER_X_MIN_MM and len(c.faces) > 1000]
    assert len(fingers) == 2, len(fingers)
    fingers.sort(key=lambda c: c.centroid[1])          # a = low y, b = high y
    ids = {id(f) for f in fingers}
    rest = [c for c in comps if id(c) not in ids]
    return m, fingers, rest


def split(out_dir=OUT_DIR):
    import trimesh
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _, fingers, rest = _load()
    palm = trimesh.util.concatenate(rest)
    info = {}
    for name, mesh in (("palm", palm), ("finger_a", fingers[0]), ("finger_b", fingers[1])):
        mm = mesh.copy()
        mm.apply_scale(MESH_SCALE)
        mm.export(str(out_dir / f"{name}.stl"))
        info[name] = (len(mm.faces), mm.bounds.round(4).tolist())
    # convex collision pieces (MuJoCo collides convex hulls; the whole jaw/palm hull would bridge the open V and the pole)
    for name, mesh, dirv, n in (("finger_a", fingers[0], _jaw_dir(0), N_JAW_PIECES), ("finger_b", fingers[1], _jaw_dir(1), N_JAW_PIECES),
                                ("palm", palm, np.array([0, 0, 1.0]), N_PALM_PIECES)):
        if name == "palm":
            proj = mesh.triangles_center @ dirv
            keep = np.ones(len(proj), bool)
        else:   # jaws: only the arm beyond the hinge; the tail behind it (slot linkage, stacked in z with the other jaw's tail) has no collision geom
            hinge = np.array(HINGE_A_MM if name == "finger_a" else HINGE_B_MM)
            proj = (mesh.triangles_center[:, :2] - hinge) @ dirv[:2]
            keep = proj >= 0.0
        edges = np.quantile(proj[keep], np.linspace(0, 1, n + 1))
        for i in range(n):
            sel = keep & (proj >= edges[i]) & (proj <= edges[i + 1])
            sub = trimesh.Trimesh(mesh.vertices, mesh.faces[sel], process=False)
            sub.remove_unreferenced_vertices()
            sub.apply_scale(MESH_SCALE)
            sub.export(str(out_dir / f"{name}_c{i}.stl"))
    return info


N_JAW_PIECES = 5        # ASSUMED: collision pieces per jaw, equal face counts along the jaw axis (hinge -> tip), each collided as its convex hull
N_PALM_PIECES = 4       # ASSUMED: collision pieces of the palm/pole, equal face counts along mesh z


def _jaw_dir(k):
    """unit vector hinge -> jaw tip (xy plane of the mesh), from the MEASURED hinge and the jaw tip centroid (x < -130 mm)."""
    _, fingers, _ = _load()
    v = fingers[k].vertices
    tip = v[v[:, 0] < -130].mean(0)
    h = np.array([*(HINGE_A_MM if k == 0 else HINGE_B_MM), 29.0])
    d = tip - h
    d[2] = 0
    return d / np.linalg.norm(d)


def hinge_points():
    """Re-measure the two pivot holes (mm, mesh frame) from the section z=29 of each jaw: the smallest closed loop."""
    _, fingers, _ = _load()
    out = []
    for f in fingers:
        s = f.section(plane_origin=[0, 0, 29.0], plane_normal=[0, 0, 1])
        loops = [e[:, :2] for e in s.discrete]
        loops.sort(key=lambda e: np.ptp(e, 0).max())
        out.append(loops[0].mean(0))
    return np.array(out)


def _R():
    from scipy.spatial.transform import Rotation as Rot
    return Rot.from_euler("XYZ", NYU_EULER_XYZ).as_matrix()      # MuJoCo euler='xyz' is intrinsic == scipy 'XYZ'


def _quat(Rm):
    from scipy.spatial.transform import Rotation as Rot
    q = Rot.from_matrix(Rm).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def mesh_to_base(p_mesh_m):
    """gripper_base frame (z = flange approach axis, origin on the jaw mirror axis at the mounting face): mesh point [m] -> base [m]."""
    R = _R()
    p0 = np.array([PALM_X_MAX_MM, JAW_Y_MID_MM, JAW_Z_MID_MM]) * 1e-3
    return R @ (np.asarray(p_mesh_m) - p0) + np.array([0, 0, NYU_ADAPTER_Z])


def _fmt(x):
    return " ".join(f"{v:.9g}" for v in np.atleast_1d(x))


def _mass_split():
    """Jaw masses = surface-area share of YOR_MASS (ASSUMED), palm = remainder; jaw inertia = convex hull scaled to that mass."""
    _, fingers, rest = _load()
    import trimesh
    area = [f.area for f in fingers]
    tot = sum(area) + sum(c.area for c in rest)
    mf = [YOR_MASS * a / tot for a in area]
    inert = []
    for f, m in zip(fingers, mf):
        h = f.convex_hull.copy(); h.apply_scale(MESH_SCALE)
        inert.append((h.center_mass, h.moment_inertia / h.volume * m))   # ASSUMED uniform density on the jaw's convex hull
    return mf, inert


def d435i_pose(R):
    """Camera-rig frame in gripper_base (pose of the D435i frame): z = lens axis = along the jaws (mesh -x) tilted down, x = image right, y = -up. RGB pupil on D435I_POS_MM."""
    t = np.deg2rad(CAM_TILT_DEG)
    f = np.array([-np.cos(t), 0.0, -np.sin(t)])
    u = np.array([-np.sin(t), 0.0, np.cos(t)])
    r = np.cross(f, u)
    Rb = R @ np.column_stack([r, -u, f])
    pos = mesh_to_base(np.array(D435I_POS_MM) * 1e-3) - Rb @ np.array(D435I_PUPIL_B)
    return pos, Rb


def d435i_body_xml(prefix: str, R):
    """The two cameras on gripper_base (no body, no mass, no geoms). Camera frame: view dir = -z_cam = lens axis (jaws, tilted down), x = right, y = up."""
    pos, Rb = d435i_pose(R)
    xy = _fmt(np.concatenate([Rb[:, 0], -Rb[:, 1]]))
    cams = (("wrist_rgb", D435I_PUPIL_B, RGB_FOVY), ("wrist_depth", D435I_IRL_B, DEPTH_FOVY))
    return [f'  <camera name="{prefix}{n}" mode="fixed" pos="{_fmt(pos + Rb @ (np.array(b) - np.array(D435I_PUPIL_B)))}" xyaxes="{xy}" fovy="{fv}" '
            f'resolution="{CAM_RES[0]} {CAM_RES[1]}" />' for n, b, fv in cams]


def gripper_body_xml(prefix: str, mesh_prefix: str = "nyu_") -> str:
    """<body gripper_base> subtree to hang under the NERO flange (replaces the AgileX gripper_base)."""
    R = _R()
    q = _quat(R)
    s = MESH_SCALE
    gpos = -R @ (np.array([PALM_X_MAX_MM, JAW_Y_MID_MM, JAW_Z_MID_MM]) * 1e-3) + np.array([0, 0, NYU_ADAPTER_Z])
    mf, inert = _mass_split()
    mp = YOR_MASS - sum(mf) - NUT_MASS_KG   # ASSUMED nut mass out of the palm share
    com = YOR_COM
    pa, pb = (np.array([*h, 0.0]) * s for h in (HINGE_A_MM, HINGE_B_MM))
    pa[2] = pb[2] = JAW_Z_MID_MM * s          # axis passes through z=29 mm (the section where it was measured), direction = mesh z
    rng = f"0 {NYU_JOINT_RANGE_RAD:g}"
    pin = mesh_to_base(np.array([NUT_PIN_X_OPEN_MM, *NUT_AXIS_YZ_MM]) * 1e-3)
    o = []
    o.append(f'<body name="{prefix}gripper_base" pos="0 0 0">')
    # palm: YOR inertial (COM, principal axes) with mass reduced by the jaws' share  # MODEL (YOR piper_left.xml) / ASSUMED split
    o.append(f'  <inertial pos="{_fmt(R @ (com - np.array([PALM_X_MAX_MM, JAW_Y_MID_MM, JAW_Z_MID_MM]) * 1e-3) + [0, 0, NYU_ADAPTER_Z])}" '
             f'mass="{mp:.6g}" fullinertia="{_fmt(np.array(_rot_inertia(R)) * mp / YOR_MASS)}" />')
    o.append(f'  <geom name="{prefix}gripper_base_vis" class="nero_visual" mesh="{mesh_prefix}palm" pos="{_fmt(gpos)}" quat="{_fmt(q)}" material="nero_grip" />')
    for i in range(N_PALM_PIECES):
        o.append(f'  <geom name="{prefix}gripper_base_col{i}" class="nero_collision" mesh="{mesh_prefix}palm_c{i}" pos="{_fmt(gpos)}" quat="{_fmt(q)}" />')
    o.append(f'  <site name="{prefix}ee_grasp" pos="0 0 {EE_GRASP_Z}" size="0.002" rgba="1 0 0 1" />')
    o.append(f'  <site name="{prefix}ee_flange" pos="0 0 0" size="0.002" rgba="0 1 0 1" />')
    o.extend(d435i_body_xml(prefix, R))
    o.append(f'  <body name="{prefix}nut" pos="{_fmt(pin)}" quat="{_fmt(q)}">')    # slide axis = mesh +x (toward the motor)
    o.append(f'    <inertial pos="0 0 0" mass="{NUT_MASS_KG}" diaginertia="1e-7 1e-7 1e-7" />')
    o.append(f'    <joint name="{prefix}gripper_nut" type="slide" axis="1 0 0" range="0 {NUT_TRAVEL_M:g}" armature="{NUT_ARMATURE}" damping="0" frictionloss="0" />')
    o.append('  </body>')
    for nm, mesh, piv, ax, k, j in (("finger1", "finger_a", pa, (0, 0, -1), 0, "gripper_joint1"), ("finger2", "finger_b", pb, (0, 0, 1), 1, "gripper_joint2")):
        bpos = mesh_to_base(piv)
        c, I = inert[k]
        # finger body frame = mesh frame rotated by R, origin at the pivot; mesh geom offset = -pivot (mesh frame)
        o.append(f'  <body name="{prefix}{nm}" pos="{_fmt(bpos)}" quat="{_fmt(q)}">')
        o.append(f'    <inertial pos="{_fmt(c - piv)}" mass="{mf[k]:.6g}" fullinertia="{_fmt([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]])}" />')
        o.append(f'    <joint name="{prefix}{j}" type="hinge" axis="{_fmt(ax)}" range="{rng}" armature="{NYU_FINGER_ARMATURE}" '
                 f'damping="{NYU_FINGER_DAMPING}" frictionloss="0" />')
        o.append(f'    <geom name="{prefix}{nm}_vis" class="nero_visual" mesh="{mesh_prefix}{mesh}" pos="{_fmt(-piv)}" material="nero_light" />')
        for i in range(N_JAW_PIECES):
            o.append(f'    <geom name="{prefix}{nm}_col{i}" class="nero_collision" mesh="{mesh_prefix}{mesh}_c{i}" pos="{_fmt(-piv)}" friction="1.5 0.01 0.01" />')
        o.append('  </body>')
    o.append('</body>')
    return "\n".join(o)


def _rot_inertia(R):
    """YOR diaginertia (principal frame given by YOR_QUAT_I in the mesh frame) -> full inertia expressed in the base frame."""
    from scipy.spatial.transform import Rotation as Rot
    P = Rot.from_quat([YOR_QUAT_I[1], YOR_QUAT_I[2], YOR_QUAT_I[3], YOR_QUAT_I[0]]).as_matrix()
    Im = P @ np.diag(YOR_DIAGI) @ P.T
    Ib = R @ Im @ R.T
    return [Ib[0, 0], Ib[1, 1], Ib[2, 2], Ib[0, 1], Ib[0, 2], Ib[1, 2]]


def assets_xml(rel_to: Path | None = None, mesh_prefix="nyu_"):
    """Mesh asset lines: NERO arm meshes (no AgileX gripper meshes) + the 3 YOR gripper meshes. rel_to -> paths relative to that dir."""
    def p(path):
        path = Path(path)
        return os.path.relpath(path, rel_to) if rel_to is not None else str(path)
    s = [l for l in M.asset_xml() if "agx_gripper" not in l]
    s = [l.replace(str(C.MESH_DIR["nero"]), p(C.MESH_DIR["nero"])) if rel_to is not None else l for l in s]
    names = ["palm", "finger_a", "finger_b"] + [f"palm_c{i}" for i in range(N_PALM_PIECES)] + \
            [f"{f}_c{i}" for f in ("finger_a", "finger_b") for i in range(N_JAW_PIECES)]
    for n in names:
        s.append(f'<mesh name="{mesh_prefix}{n}" file="{p(OUT_DIR / (n + ".stl"))}" maxhullvert="64" />')
    return s


def arm_body_xml(prefix: str, mount=None, gravcomp=False, joint_range=None) -> str:
    """NERO arm body tree (urdf_to_mjcf.body_xml) with the AgileX gripper_base swapped for the YOR gripper."""
    root = ET.fromstring(M.body_xml(prefix, mount, gravcomp, joint_range))
    fl = next(b for b in root.iter("body") if b.get("name") == f"{prefix}flange")
    for ch in list(fl):
        if ch.tag == "body" and ch.get("name") == f"{prefix}gripper_base":
            fl.remove(ch)
    fl.append(ET.fromstring(gripper_body_xml(prefix)))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")


def actuator_xml(prefix: str, tau=None):
    s = [l for l in M.actuator_xml(prefix, tau) if "gripper_pos" not in l]
    # ctrl = commanded nut position [m] = jaw closing pose (0 = open mesh pose, NUT_TRAVEL_M = closed); pose only, no force claim
    s.append(f'<position name="{prefix}gripper_pos" joint="{prefix}gripper_nut" kp="{NUT_KP:g}" kv="{NUT_KV:g}" '
             f'ctrlrange="0 {NUT_TRAVEL_M:g}" forcerange="-{NUT_FORCE_N:g} {NUT_FORCE_N:g}" />')
    return s


def equality_xml(prefix: str):
    # jaw1 (low-y) axis is -z, jaw2 +z, so closing is + for both: q_jaw = THETA_POLY(q_nut) for each
    pc = " ".join(f"{c:.9g}" for c in THETA_POLY)
    return [f'<joint joint1="{prefix}gripper_joint{k}" joint2="{prefix}gripper_nut" polycoef="{pc}" solref="0.005 1" solimp="0.99 0.999 1e-05" />' for k in (1, 2)]


def contact_excludes(prefix: str):
    ex = [e for e in M.contact_excludes(prefix) if set(e) != {f"{prefix}finger1", f"{prefix}finger2"}]   # jaws may touch each other
    return ex


# ---- fragment API (same names as urdf_to_mjcf, so build_variant.variant_xml() is the builder build_variant.variant_xml() uses)
def default_xml() -> str:
    return M.default_xml()


def asset_xml() -> list[str]:
    """absolute mesh paths, exactly like urdf_to_mjcf.asset_xml (works from scenes/ and for in-memory MjSpec.from_string)."""
    return assets_xml(None)


def body_xml(prefix: str, mount=None, gravcomp=False, joint_range=None) -> str:
    return arm_body_xml(prefix, mount, gravcomp, joint_range)


def standalone_xml(prefix: str = "", gravcomp: bool = False, tau=None, rel_to: Path | None = C.ASSETS) -> str:
    ex = "\n".join(f'    <exclude body1="{a}" body2="{b}" />' for a, b in contact_excludes(prefix))
    a = "\n".join("    " + s for s in assets_xml(rel_to))
    return f"""<mujoco model="nero_arm_nyugrip">
  <!-- GENERATED by world_model/v2/nero/nyu_gripper.py (opt-in variant; the default assets_nero/nero_arm.xml is untouched).
       NERO arm + YOR angular-jaw gripper (meshes split from YOR gripper_body.stl) + wrist camera. Mesh paths relative to this file. Do not hand-edit. -->
  <compiler angle="radian" inertiafromgeom="false" />
  <option integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81" timestep="0.001" />
  <default>
    {M.default_xml()}
  </default>
  <asset>
{a}
  </asset>
  <worldbody>
    <light pos="0 0 2.0" directional="true" />
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.3 0.3 0.35 1" contype="0" conaffinity="0" />
{arm_body_xml(prefix, None, gravcomp)}
  </worldbody>
  <contact>
{ex}
  </contact>
  <equality>
    {chr(10).join('    ' + s for s in equality_xml(prefix)).strip()}
  </equality>
  <actuator>
    {chr(10).join('    ' + s for s in actuator_xml(prefix, tau)).strip()}
  </actuator>
</mujoco>
"""


def write_standalone(path=None, **kw):
    path = Path(path or ARM_NYUGRIP_OUT)
    path.write_text(standalone_xml(**kw))
    return path


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "write"
    if cmd == "split":
        print(split())
    elif cmd == "hinge":
        print(hinge_points())
    else:
        print(write_standalone())
