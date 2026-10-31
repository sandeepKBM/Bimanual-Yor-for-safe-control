"""Build the 'setup coffee tray' task for AgentChord's sim (EmbodiChain/DexSim), outside their code.

The paper's weakest bimanual task (real robot only, 60%: Table II) is not in their release: no assets, env or prompt.
Paper steps (Appendix task table): grasp capsule (L) -> place it into the tray (L) -> grasp tray (LR) -> place tray forward (LR).
Success (paper): "the coffee capsule is correctly placed inside the tray, and the tray is moved to the designated forward position".

This script (host side, needs trimesh) writes:
  /data/local/ss5772/agentchord/data/ProbeAssets/coffee_tray/{tray,capsule}.ply   (meshes, metres, origin = body centre)
  coffee_tray/gym_config.json   (derived from THEIR rearrangement_agent config: same robot, table, cameras, randomisation style)
  coffee_tray/agent_config.json (copy of their rearrangement agent config: same agents and prompt files)
task_prompt.txt is hand-written in the style of their sim task prompts (scripted steps + numeric offsets).

Geometry (our choice; the paper gives none): tray 0.22 x 0.34 x 0.008 m base with 0.03 m rims (0.008 thick), long axis along y so
each arm reaches one short-end rim (rims at y = +-0.166, like their fork/spoon at y = +-0.21). Fingers close along the object y axis
(their top-down grasp pose, checked on the fork mesh: long axis x, 2.5 cm across y), so the y-end rims are graspable. Capsule:
cylinder r = 0.019 m, h = 0.028 m (a Nespresso-class capsule). Masses: tray 0.30 kg, capsule 0.012 kg.
"""
import copy
import json
import os

import numpy as np
import trimesh

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = "/data/local/ss5772/agentchord"
SRC = f"{ROOT}/src"
ASSET_DIR = f"{ROOT}/data/ProbeAssets/coffee_tray"
ASSET_IN_CONTAINER = "/work/data/ProbeAssets/coffee_tray"

TRAY = dict(lx=0.22, ly=0.34, base_t=0.008, rim_h=0.03, rim_t=0.008, mass=0.30)
# handles (user 10-28, after rim grasps failed: thin 8 mm rims are not graspable once CoACD fattens them): one bar per short end,
# parallel to it, 0.10 m long, 0.02 x 0.02 m section, 0.03 m gap from the rim (room for the inner finger), top flush with the rim,
# held by two 0.02 m struts at the bar ends. Grasp = top-down at the bar centre, fingers closing across the 0.02 m bar.
HANDLE = dict(len=0.10, sec=0.02, gap=0.03)
CAPSULE = dict(r=0.019, h=0.028, mass=0.012)
TRAY_POS = [0.48, 0.0]          # tray centre on the table (x forward, y left)
CAPSULE_POS = [0.58, 0.30]      # 10-28 handles reach y=0.22 (+0.03 jitter): capsule moved out, jitter cut to +-0.02. was [0.55, 0.26];       # left of the tray (rim at y=0.17), on the table (0.42,0.28 was at the table edge), left-arm reach like their fork
FORWARD = [0.10, 0.0]
TABLE_TOP_Z = 0.742             # their rearrange table: fork (1.8 cm thick) rests at z = 0.751
# spawn just above the table (their pour config spawns cup/bottle near the top; z = 1.0 left objects floating at plan time,
# because their pose randomiser does not step physics unless physics_update_step > 0) and settle 20 physics steps
SPAWN_GAP, SETTLE_STEPS = 0.003, 20           # "place the tray forward": +0.10 m in x
# per-arm grasp poses in the tray frame (top-down, their fork/spoon rotation), at the handle-bar centre
H_TOT = TRAY["base_t"] + TRAY["rim_h"]
BAR_Y = TRAY["ly"] / 2 + HANDLE["gap"] + HANDLE["sec"] / 2                 # bar centre line, outside the short-end rim
BAR_ZC = TRAY["base_t"] + TRAY["rim_h"] - HANDLE["sec"] / 2                 # bar centre height above the tray bottom
GRIP_Z = BAR_ZC - H_TOT / 2                                                # in the mesh-centre frame
RIM_Y = BAR_Y   # name kept for the grasp table below


def top_down(t):
    return [[1.0, 0.0, 0.0, t[0]], [0.0, -1.0, 0.0, t[1]], [0.0, 0.0, -1.0, t[2]], [0.0, 0.0, 0.0, 1.0]]


TRAY_GRASP = {"left_arm": top_down([0.0, +RIM_Y, GRIP_Z]), "right_arm": top_down([0.0, -RIM_Y, GRIP_Z])}


def tray_mesh():
    lx, ly, bt, rh, rt = TRAY["lx"], TRAY["ly"], TRAY["base_t"], TRAY["rim_h"], TRAY["rim_t"]
    parts = [trimesh.creation.box([lx, ly, bt], trimesh.transformations.translation_matrix([0, 0, bt / 2]))]
    for sx in (-1, 1):   # long rims (along y) at the x ends
        parts.append(trimesh.creation.box([rt, ly, rh], trimesh.transformations.translation_matrix([sx * (lx - rt) / 2, 0, bt + rh / 2])))
    for sy in (-1, 1):   # short-end rims (along x) at the y ends
        parts.append(trimesh.creation.box([lx, rt, rh], trimesh.transformations.translation_matrix([0, sy * (ly - rt) / 2, bt + rh / 2])))
    hl, hs, hg = HANDLE["len"], HANDLE["sec"], HANDLE["gap"]
    for sy in (-1, 1):   # handle bar + two struts bridging the gap
        parts.append(trimesh.creation.box([hl, hs, hs], trimesh.transformations.translation_matrix([0, sy * BAR_Y, BAR_ZC])))
        for sx in (-1, 1):
            parts.append(trimesh.creation.box([hs, hg, hs], trimesh.transformations.translation_matrix(
                [sx * (hl - hs) / 2, sy * (ly / 2 + hg / 2), BAR_ZC])))
    m = trimesh.util.concatenate(parts)
    m.apply_translation([0, 0, -(bt + rh) / 2])      # origin at the bounding-box centre, like their meshes
    return m


def capsule_mesh():
    return trimesh.creation.cylinder(radius=CAPSULE["r"], height=CAPSULE["h"], sections=32)


def obj_entry(uid, fname, pos, mass, half_h):
    return {"uid": uid, "shape": {"shape_type": "Mesh", "fpath": f"{ASSET_IN_CONTAINER}/{fname}", "compute_uv": True},
            "body_scale": [1.0, 1.0, 1.0], "init_pos": [pos[0], pos[1], TABLE_TOP_Z + half_h + SPAWN_GAP],
            "attrs": {"mass": mass, "static_friction": 0.95, "dynamic_friction": 0.9, "restitution": 0.05,
                      "min_position_iters": 16, "min_velocity_iters": 4}}


def gym_config():
    c = json.load(open(f"{SRC}/configs/gym/agent/rearrangement_agent/fast_gym_config.json"))
    c = copy.deepcopy(c)
    c["id"] = "CoffeeTrayAgent-v3"
    c["max_episode_steps"] = 600
    c["rigid_object"] = [obj_entry("tray", "tray.ply", TRAY_POS, TRAY["mass"], (TRAY["base_t"] + TRAY["rim_h"]) / 2),
                         obj_entry("capsule", "capsule.ply", CAPSULE_POS, CAPSULE["mass"], CAPSULE["h"] / 2)]
    # their default is ONE convex hull per object (cfg.max_convex_hull_num = 1): the tray became a solid block (capsule rested on
    # the rim-top plane, rim grasps closed on air; ct_gen_terra3, 10-28). Tray = base + 4 rims -> CoACD with up to 8 hulls.
    c["rigid_object"][0]["max_convex_hull_num"] = 16   # base + 4 rims + 2 bars + 4 struts = 11 boxes
    ev = c["env"]["events"]
    # randomisation in their style (fork: +-5 cm, +-45 deg); smaller here so both rims stay reachable
    ev.pop("init_fork_pose"); ev.pop("init_spoon_pose")
    ev["init_tray_pose"] = {"func": "randomize_rigid_object_pose", "mode": "reset", "params": {
        "entity_cfg": {"uid": "tray"}, "position_range": [[-0.03, -0.03, 0.0], [0.03, 0.03, 0.0]],
        "rotation_range": [[0, 0, -5], [0, 0, 5]], "relative_position": True, "relative_rotation": True}}
    ev["init_capsule_pose"] = {"func": "randomize_rigid_object_pose", "mode": "reset", "params": {
        "entity_cfg": {"uid": "capsule"}, "position_range": [[-0.02, -0.02, 0.0], [0.02, 0.02, 0.0]],
        "rotation_range": [[0, 0, 0], [0, 0, 0]], "relative_position": True, "relative_rotation": True,
        "physics_update_step": SETTLE_STEPS}}
    for a in ev["prepare_extra_attr"]["params"]["attrs"]:
        if a["name"] == "grasp_pose_object":
            a["entity_uids"] = ["capsule", "tray"]      # tray value is replaced per arm at grasp time (ac_run.patch_tray_grasp)
            a["value"] = [top_down([0.0, 0.0, 0.0])]
    reg = ev["register_info_to_env"]["params"]["registry"]
    objs = [r for r in reg if r["entity_cfg"]["uid"] in ("plate", "fork", "spoon")]
    arms = [r for r in reg if r not in objs]
    new = []
    for uid in ("tray", "capsule"):
        r = copy.deepcopy(objs[0]); r["entity_cfg"]["uid"] = uid; new.append(r)
    for r in arms:   # their arm-base registrations are relative to an object; keep the structure, point at ours
        r = copy.deepcopy(r)
        rel = r.get("pose_register_params", {}).get("compute_relative")
        if rel in ("spoon", "fork"):
            r["pose_register_params"]["compute_relative"] = "capsule" if rel == "spoon" else "tray"
        new.append(r)
    ev["register_info_to_env"]["params"]["registry"] = new
    ev.pop("random_plate_material"); ev.pop("random_fork_material"); ev.pop("random_spoon_material")
    ev["random_tray_material"] = {"func": "randomize_visual_material", "mode": "reset", "interval_step": 2, "params": {
        "entity_cfg": {"uid": "tray"}, "random_texture_prob": 1.0, "texture_path": "DexsimMaterials/Plate"}}
    ev["random_capsule_material"] = {"func": "randomize_visual_material", "mode": "reset", "interval_step": 2, "params": {
        "entity_cfg": {"uid": "capsule"}, "random_texture_prob": 1.0, "texture_path": "DexsimMaterials/Cup"}}
    # same event order as theirs: randomise poses first, then prepare_extra_attr / register_info_to_env read them
    order = ["record_camera", "validation_cameras", "init_tray_pose", "init_capsule_pose"]
    c["env"]["events"] = {k: ev[k] for k in order + [k for k in ev if k not in order]}
    s = json.dumps(c)
    for old in ("plate", "fork", "spoon"):
        assert f'"{old}"' not in s, f"leftover reference to {old}"
    return c


def write_variant(tag, x, y):
    """10-28: same scene with the tray centre moved to (x, y); writes variants/gym_config_<tag>.json only (meshes, prompts and
    success check unchanged: the check is relative to the tray's reset pose)."""
    global TRAY_POS
    TRAY_POS = [x, y]
    os.makedirs(f"{HERE}/variants", exist_ok=True)
    path = f"{HERE}/variants/gym_config_{tag}.json"
    json.dump(gym_config(), open(path, "w"), indent=2)
    return path


def main():
    import sys
    if len(sys.argv) == 5 and sys.argv[1] == "--variant":
        print(write_variant(sys.argv[2], float(sys.argv[3]), float(sys.argv[4])))
        return
    os.makedirs(ASSET_DIR, exist_ok=True)
    tray_mesh().export(f"{ASSET_DIR}/tray.ply")
    capsule_mesh().export(f"{ASSET_DIR}/capsule.ply")
    json.dump(gym_config(), open(f"{HERE}/gym_config.json", "w"), indent=2)
    ag = json.load(open(f"{SRC}/configs/gym/agent/rearrangement_agent/agent_config.json"))
    json.dump(ag, open(f"{HERE}/agent_config.json", "w"), indent=4)
    # minimal variant (user 10-28): the paper's own task line (appendix), no numbers; object name = scene uid
    ag_min = json.loads(json.dumps(ag))
    ag_min["Agent"]["prompt_kwargs"]["task_prompt"]["name"] = "task_prompt_min.txt"
    json.dump(ag_min, open(f"{HERE}/agent_config_min.json", "w"), indent=4)
    json.dump({"tray": TRAY, "handle": HANDLE, "capsule": CAPSULE, "tray_pos": TRAY_POS, "capsule_pos": CAPSULE_POS, "forward": FORWARD,
               "tray_grasp": TRAY_GRASP}, open(f"{HERE}/geometry.json", "w"), indent=2)
    print("tray extents", tray_mesh().extents.round(4), "capsule extents", capsule_mesh().extents.round(4))
    print("wrote", ASSET_DIR, "and", HERE)


if __name__ == "__main__":
    main()
