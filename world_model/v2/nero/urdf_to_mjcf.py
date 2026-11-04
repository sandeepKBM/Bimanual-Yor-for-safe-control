"""urdf_to_mjcf.py -- official NERO URDF (+flange +URDF gripper) -> MJCF fragments / a standalone arm file.

11-03: this is the BASE of the NYU build: nyu_gripper.py reuses the arm links and replaces the URDF gripper_base/fingers (the URDF gripper is not used in any scene).

Kinematic tree, inertials (mass, COM, FULL tensor), joint limits, collision-mesh poses come from the URDF; everything
the URDF lacks (armature, damping, friction loss, servo gains, torque limits, contact params) comes from config.ASSUMED /
config.tau_limits() and is tagged ASSUMED in the emitted comments.

Frames: URDF link frame == MJCF body frame (so FK is a 1:1 chain; the verification is test_fk).  The fixed joints
(link7->flange->gripper_base) stay as bodies without a joint.  The URDF 'gripper' prismatic is a virtual coordinate
(its child link is empty) driving two mimic fingers (joint1 = +0.5*g, joint2 = -0.5*g): here finger1 is the DOF
(range 0..0.05) and finger2 is tied by an equality `joint` constraint (q2 = -q1); opening width = 2*q1.

    python -m world_model.v2.nero.urdf_to_mjcf            # writes assets_nero/nero_arm.xml (single arm on the world)
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import config as C
from . import urdf_fk as U

ARM_LINKS = ["base_link"] + [f"link{i}" for i in range(1, 8)]


def _f(x):
    return " ".join(f"{v:.12g}" for v in np.atleast_1d(x))


def mesh_assets(mesh_dir_override=None):
    """-> [(asset_name, abs_path)] for every collision/visual mesh used."""
    links, _ = U.parse()
    out = []
    for ln, L in links.items():
        if L.coll_mesh is None:
            continue
        base = Path(L.coll_mesh).name
        which = "agx_gripper" if "agx_gripper" in L.coll_mesh else "nero"
        out.append((f"nero_{Path(base).stem}", str(C.MESH_DIR[which] / base)))
    return out


def asset_xml():
    s = ['<material name="nero_dark" rgba="0.18 0.18 0.2 1" />', '<material name="nero_light" rgba="0.82 0.83 0.85 1" />',
         '<material name="nero_grip" rgba="0.35 0.36 0.4 1" />']
    for n, p in mesh_assets():
        s.append(f'<mesh name="{n}" file="{p}" maxhullvert="64" />')
    return s


def default_xml():
    A = C.ASSUMED
    return f"""<default class="nero">
      <!-- ASSUMED passive params (URDF has none): armature/damping/frictionloss from config.ASSUMED -->
      <joint armature="{A['armature']}" damping="{A['damping']}" frictionloss="{A['frictionloss']}" />
      <default class="nero_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" density="0" />
      </default>
      <default class="nero_collision">
        <!-- same contact params as YOR 'collision' class; contype/conaffinity 0/1 so sim_common._patch_real_collision_geoms
             style bit patching applies; convex hull of the official STL (maxhullvert 64) -->
        <geom type="mesh" condim="3" contype="0" conaffinity="1" priority="1" group="3" solref="0.005 1" friction="1 0.01 0.01" />
      </default>
      <default class="nero_finger">
        <joint type="slide" damping="{A['finger_damping']}" frictionloss="{A['finger_frictionloss']}" armature="0.001" />
      </default>
    </default>"""


def _mat(ln):
    return {"base_link": "nero_dark", "link1": "nero_light", "link2": "nero_light", "link3": "nero_dark",
            "link4": "nero_light", "link5": "nero_dark", "link6": "nero_light", "link7": "nero_dark",
            "flange_link": "nero_dark"}.get(ln, "nero_grip")


def body_xml(prefix: str, mount: dict | None = None, gravcomp: bool = False, joint_range=None) -> str:
    """The whole arm as ONE <body> subtree string (root = `{prefix}link0` == URDF base_link)."""
    links, joints = U.parse()
    A = C.ASSUMED
    lim = np.array(C.URDF_LIMITS if joint_range is None else joint_range)
    gc = ' gravcomp="1"' if gravcomp else ""
    by_parent = {}
    for j in joints.values():
        by_parent.setdefault(j.parent, []).append(j)
    mjname = {"base_link": "link0", "flange_link": "flange", "gripper_base": "gripper_base",
              "gripper_link1": "finger1", "gripper_link2": "finger2"}
    for i in range(1, 8):
        mjname[f"link{i}"] = f"link{i}"
    jname = {f"joint{i}": f"joint{i}" for i in range(1, 8)}
    jname["gripper_joint1"] = "gripper_joint1"
    jname["gripper_joint2"] = "gripper_joint2"

    def link_xml(ln, joint: U.Joint | None, indent: int, root_attrs: str = "") -> str:
        L = links[ln]
        pad = "  " * indent
        out = []
        if joint is None:
            head = f'{pad}<body name="{prefix}{mjname[ln]}"{root_attrs} childclass="nero"{gc}>'
        else:
            q = U.R_to_quat(U.rpy_to_R(*joint.rpy))
            head = f'{pad}<body name="{prefix}{mjname[ln]}" pos="{_f(joint.xyz)}" quat="{_f(q)}"{gc}>'
        out.append(head)
        if L.mass > 0:
            assert np.allclose(L.inertial_rpy, 0), f"{ln}: non-zero inertial rpy not supported"
            I = L.inertia
            out.append(f'{pad}  <inertial pos="{_f(L.com)}" mass="{L.mass:.12g}" fullinertia="{_f([I[0,0], I[1,1], I[2,2], I[0,1], I[0,2], I[1,2]])}" />')
        if joint is not None and joint.type == "revolute":
            k = int(joint.name.replace("joint", "")) - 1
            out.append(f'{pad}  <joint name="{prefix}{jname[joint.name]}" axis="{_f(joint.axis)}" range="{lim[k,0]:.9g} {lim[k,1]:.9g}" />')
        elif joint is not None and joint.type == "prismatic" and ln.startswith("gripper_link"):
            out.append(f'{pad}  <joint name="{prefix}{jname[joint.name]}" class="nero_finger" axis="{_f(joint.axis)}" range="{joint.lower:.9g} {joint.upper:.9g}" />')
        if L.coll_mesh is not None:
            stem = Path(L.coll_mesh).stem
            q = U.R_to_quat(U.rpy_to_R(*L.coll_rpy))
            gp = f'pos="{_f(L.coll_xyz)}" quat="{_f(q)}" mesh="nero_{stem}"'
            out.append(f'{pad}  <geom name="{prefix}{mjname[ln]}_vis" class="nero_visual" {gp} material="{_mat(ln)}" />')
            out.append(f'{pad}  <geom name="{prefix}{mjname[ln]}_col" class="nero_collision" {gp} />')
        if ln == "gripper_base":
            out.append(f'{pad}  <site name="{prefix}ee_grasp" pos="0 0 {A["ee_grasp_z"]}" size="0.002" rgba="1 0 0 1" />')
            out.append(f'{pad}  <site name="{prefix}ee_flange" pos="0 0 0" size="0.002" rgba="0 1 0 1" />')
        for j in by_parent.get(ln, []):
            if j.child == "gripper_link":          # virtual 'gripper' DOF: represented by finger1/finger2 below
                continue
            if j.parent == "gripper_base" and j.child in ("gripper_link1", "gripper_link2"):
                out.append(link_xml(j.child, j, indent + 1))
            elif j.parent == "world":
                continue
            else:
                out.append(link_xml(j.child, j, indent + 1))
        out.append(f"{pad}</body>")
        return "\n".join(out)

    attrs = ""
    if mount is not None:
        attrs = f' pos="{mount["pos"]}" euler="{mount["euler"]}"'
    return link_xml("base_link", None, 0, attrs)


def actuator_xml(prefix: str, tau=None) -> list[str]:
    A = C.ASSUMED
    tau = C.tau_limits() if tau is None else np.asarray(tau, float)
    s = [f'<position name="{prefix}joint{i+1}_pos" joint="{prefix}joint{i+1}" kp="{A["kp"][i]}" kv="{A["kv"][i]}" '
         f'inheritrange="1" forcerange="{-tau[i]:g} {tau[i]:g}" />' for i in range(7)]
    s.append(f'<position name="{prefix}gripper_pos" joint="{prefix}gripper_joint1" kp="{A["finger_kp"]}" '
             f'ctrlrange="0 0.05" forcerange="-{C.GRIPPER_EFFORT:g} {C.GRIPPER_EFFORT:g}" />')
    return s


def equality_xml(prefix: str) -> list[str]:
    # URDF mimic: gripper_joint2 = -0.5*g, gripper_joint1 = +0.5*g  ->  q2 = -q1
    return [f'<joint joint1="{prefix}gripper_joint2" joint2="{prefix}gripper_joint1" polycoef="0 -1 0 0 0" '
            f'solref="0.005 1" solimp="0.99 0.999 1e-05" />']


def contact_excludes(prefix: str) -> list[tuple[str, str]]:
    """Body pairs excluded from self-collision: URDF-adjacent chain links and the small overlapping wrist/gripper set."""
    n = [f"{prefix}link{i}" for i in range(0, 8)] + [f"{prefix}flange", f"{prefix}gripper_base", f"{prefix}finger1", f"{prefix}finger2"]
    ex = [(n[i], n[i + 1]) for i in range(7)]                       # link0-1 ... link6-7
    ex += [(f"{prefix}link7", f"{prefix}flange"), (f"{prefix}flange", f"{prefix}gripper_base"),
           (f"{prefix}link7", f"{prefix}gripper_base"), (f"{prefix}link6", f"{prefix}flange"),
           (f"{prefix}link6", f"{prefix}link7"),
           (f"{prefix}link5", f"{prefix}flange"), (f"{prefix}link5", f"{prefix}link7"),     # hulls overlap structurally (1500-config scan)
           (f"{prefix}finger1", f"{prefix}finger2"), (f"{prefix}gripper_base", f"{prefix}finger1"),
           (f"{prefix}gripper_base", f"{prefix}finger2"), (f"{prefix}flange", f"{prefix}finger1"),
           (f"{prefix}flange", f"{prefix}finger2")]
    return sorted(set(ex))


def standalone_xml(prefix: str = "", gravcomp: bool = False, tau=None) -> str:
    """One NERO arm on the world (base_link at the origin, z up): FK/inertia/gravity verification model."""
    ex = "\n".join(f'    <exclude body1="{a}" body2="{b}" />' for a, b in contact_excludes(prefix))
    return f"""<mujoco model="nero_arm">
  <!-- GENERATED by world_model/v2/nero/urdf_to_mjcf.py from the OFFICIAL AgileX description
       {C.URDF_REPO} @ {C.URDF_COMMIT} (nero/urdf/nero_description.urdf + flange + agx_gripper). Do not hand-edit. -->
  <compiler angle="radian" inertiafromgeom="false" />
  <option integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81" timestep="0.005" />
  <default>
    {default_xml()}
  </default>
  <asset>
    {chr(10).join('    ' + s for s in asset_xml()).strip()}
  </asset>
  <worldbody>
    <light pos="0 0 2.0" directional="true" />
    <geom name="floor" type="plane" size="2 2 0.05" rgba="0.3 0.3 0.35 1" contype="0" conaffinity="0" />
{body_xml(prefix, None, gravcomp)}
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
    path = Path(path or C.ARM_OUT)
    path.write_text(standalone_xml(**kw))
    return path


if __name__ == "__main__":
    print(write_standalone())
