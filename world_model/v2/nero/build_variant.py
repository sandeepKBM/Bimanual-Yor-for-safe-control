"""build_variant.py -- generate scenes/yor_nero_derived.mjcf: the YOR robot with the two Piper arms replaced by NERO arms.

Nothing existing is edited: scenes/yor_robot_derived.mjcf is READ, the variant is a NEW generated file.  Changes vs the
Piper derived file:
  * left_arm_link0 / right_arm_link0 (Piper subtrees) removed from Lift_Top, their meshes + position actuators + contact
    excludes removed (wrist iphone/cameras go with them);
  * NERO arms inserted as `left_arm_link0..7`, `left_arm_flange`, `left_arm_gripper_base`, `left_arm_finger1/2` (joints
    `left_arm_joint1..7`, `left_arm_gripper_joint1/2`) at the SAME mount pos/euler as the Piper link0 on Lift_Top
    (ASSUMED: NERO base_link frame == Piper link0 frame; NERO mounting flange differs/unverified);
  * actuators: `{side}_arm_joint{1..7}_pos` (position, forcerange = tau limits, ASSUMED) + `{side}_arm_gripper_pos`;
  * flags: lift=True enables Slider_1/Slider_2 (telescope class, 0..0.208 m each, +-56 N); gravcomp=True sets gravcomp=1
    on the NERO bodies (the YOR/Piper convention; default OFF so gravity is inside the torque budget).

    python -m world_model.v2.nero.build_variant [--lift] [--gravcomp]
"""
from __future__ import annotations

import argparse
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from . import config as C
from . import nyu_gripper as X      # 11-03 USER: the NYU gripper is the only build (nyu_gripper reuses the NERO arm fragments of urdf_to_mjcf)

SRC = C.REPO / "scenes" / "yor_robot_derived.mjcf"
PIPER_MESHES = {"link0", "link1", "link2", "link3", "link4", "link5", "link6", "gripper_base", "gripper_body", "iphone",
                "link7", "link8"}


def _el(s: str) -> ET.Element:
    return ET.fromstring(s)


def _arm_arm_collision(lt) -> int:
    """11-03 item 5: make the LEFT arm's collision geoms collide with the RIGHT arm's (and nothing else new).  The 'nero_collision' class ships
    contype 0 / conaffinity 1, so no arm-arm pair existed (0 collidable pairs in the contents / pan scenes).  Collide rule: (ct1 & ca2) | (ct2 & ca1).
    LEFT geoms get contype = ARM_ARM_BIT (conaffinity stays 1), RIGHT geoms get conaffinity = 1 | ARM_ARM_BIT (contype stays 0):
      L-R: ct_L & ca_R = ARM_ARM_BIT -> collide;  L-L, R-R: 0 (same-arm self-collision unchanged: none);  arm-world/object: unchanged (bit 1).
    Adjacent-link excludes of the same arm are untouched.  Returns the number of geoms patched."""
    n = 0
    for b in lt.iter("body"):
        nm = b.get("name") or ""
        side = "left" if nm.startswith("left_arm_") else "right" if nm.startswith("right_arm_") else None
        if side is None:
            continue
        for g in b.findall("geom"):
            if g.get("class") != "nero_collision":
                continue
            if side == "left":
                g.set("contype", str(C.ARM_ARM_BIT))
            else:
                g.set("conaffinity", str(1 | C.ARM_ARM_BIT))
            n += 1
    return n


def variant_xml(lift: bool = False, gravcomp: bool = False, tau=None, arm_arm_collide: bool | None = None) -> str:
    """arm_arm_collide: None = on unless BIMANUAL_LEGACY_ARMCOL=1; adapters.compose_scene / democ.scene pass False (they patch the
    bits themselves with adapters.patch_collision, which already gives arm-arm + self collision)."""
    # the source's header comment contains '--' (invalid XML, MuJoCo tolerates it): strip all comments first
    txt = re.sub(r"<!--.*?-->", "", SRC.read_text(), flags=re.S)
    root = ET.fromstring(txt)
    tree = ET.ElementTree(root)
    root.set("model", "yor-nero-derived")
    # --- defaults
    root.find("default").append(_el(X.default_xml()))
    # --- assets: drop Piper arm meshes, add NERO
    asset = root.find("asset")
    for me in list(asset.findall("mesh")):
        if me.get("name") in PIPER_MESHES:
            asset.remove(me)
    for s in X.asset_xml():
        asset.append(_el(s))
    # --- bodies: swap arms on Lift_Top
    lt = [b for b in root.iter("body") if b.get("name") == "Lift_Top"][0]
    for b in list(lt.findall("body")):
        if b.get("name") in ("left_arm_link0", "right_arm_link0"):
            lt.remove(b)
    for side in C.SIDES:
        lt.append(_el(X.body_xml(f"{side}_arm_", C.ASSUMED[f"mount_{side}"], gravcomp)))
    if (not C.LEGACY_ARMCOL) if arm_arm_collide is None else arm_arm_collide:
        _arm_arm_collision(lt)
    # --- actuators
    act = root.find("actuator")
    for a in list(act):
        if (a.get("name") or "").startswith(("left_arm_", "right_arm_")):
            act.remove(a)
    for side in C.SIDES:
        for s in X.actuator_xml(f"{side}_arm_", tau):
            act.append(_el(s))
    # --- contact excludes
    con = root.find("contact")
    for e in list(con):
        if "arm_" in (e.get("body1") or "") or "arm_" in (e.get("body2") or ""):
            con.remove(e)
    for side in C.SIDES:
        pre = f"{side}_arm_"
        con.append(_el(f'<exclude body1="Lift_Top" body2="{pre}link0" />'))
        for a, b in X.contact_excludes(pre):
            con.append(_el(f'<exclude body1="{a}" body2="{b}" />'))
    # --- equality (finger mimic) [+ lift]
    eq = ET.SubElement(root, "equality")
    for side in C.SIDES:
        for s in X.equality_xml(f"{side}_arm_"):
            eq.append(_el(s))
    if lift:
        mid = [b for b in root.iter("body") if b.get("name") == "Lift_Middle"][0]
        mid.insert(1, _el('<joint name="Slider_1" class="telescope" />'))     # after <inertial>
        lt.insert(1, _el('<joint name="Slider_2" class="telescope" />'))
        eq.append(_el('<joint joint1="Slider_1" joint2="Slider_2" />'))
        tend = ET.SubElement(root, "tendon")
        fx = ET.SubElement(tend, "fixed", name="lift")
        ET.SubElement(fx, "joint", joint="Slider_1", coef="1")
        ET.SubElement(fx, "joint", joint="Slider_2", coef="1")
        act.append(_el('<position name="Lift" tendon="lift" class="telescope" />'))
    ET.indent(tree, space="  ")
    body = ET.tostring(root, encoding="unicode")
    hdr = (f"<!-- GENERATED by world_model/v2/nero/build_variant.py from scenes/yor_robot_derived.mjcf (read-only) and the OFFICIAL\n"
           f"     AgileX description {C.URDF_REPO} @ {C.URDF_COMMIT}.  Do not hand-edit; regenerate.\n"
           f"     flags: lift={lift} gravcomp={gravcomp}.  Mount frame, armature/damping/frictionloss, servo gains and torque limits\n"
           f"     are ASSUMED (config.py).  Piper arms/wrist cameras are NOT present in this variant. -->\n")
    i = body.index(">") + 1
    return body[:i] + "\n  " + hdr + body[i:]


def write_variant(path=None, **kw) -> Path:
    p = Path(path or C.scene_path("yor_nero_derived"))
    p.write_text(variant_xml(**kw))
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--lift", action="store_true")
    ap.add_argument("--gravcomp", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    print(write_variant(a.out, lift=a.lift, gravcomp=a.gravcomp))
