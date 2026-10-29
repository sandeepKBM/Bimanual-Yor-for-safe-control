"""agentchord_probe/ac_run.py -- run AgentChord sim episodes with scripted disturbances and full logging. Runs INSIDE the container.

Their code (/work/src, unedited) is driven from outside by monkeypatching four names in embodichain.agents.agentchord.atom_actions:
  tqdm                           -> records the step count of each atomic action (drive loops over tqdm(range(len(actions))))
  setup_interactive_error_input  -> returns our per-action injector state instead of reading the terminal
  interactive_error_requested    -> called after EVERY env.step: logs object poses, fires the scripted disturbance at its step
  inject_interactive_error       -> applies the scripted disturbance (their misplaced_object / fallen_object)
and agent_graph.drive (wrapped) to log each edge's result (which monitor fired, at which step).

Paper protocol (Sec. IV-A): "with probability p, an object held by the end-effector is randomly dropped at each atomic action",
20 trials per task. Our reading (assumptions, recorded in meta.json):
  - "held" = a rigid object lifted > 3 cm above its initial height (tabletop tasks: nothing else lifts objects);
  - the drop happens at a uniformly random step of the action; drop modes: upright (teleport down to its initial height, xy kept)
    or fallen (their fallen_object: same plus 90/90 deg rotation, i.e. lying on its side).
Their task success is is_task_success(). For pour tasks it only checks that bottle and cup are not tipped >= 45 deg, so we also
score a stricter pour criterion (pre-registered in agentchord_probe/README.md).

SAFETY: embodichain.deploy, ROS, CAN and robot SDK imports are blocked before anything else is imported (real-robot path never runs).
"""
import argparse, importlib.abc, json, os, sys, time, traceback

BLOCKED = ("embodichain.deploy", "rclpy", "rospy", "piper_sdk", "can", "pyAgxArm", "agx")


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ImportError(f"[agentchord_probe] import of {name!r} blocked (hardware path; offline sim only)")
        return None


sys.meta_path.insert(0, _Block())

import numpy as np  # noqa: E402

TASKS = {
    "SinglePourWater": ("configs/gym/agent/pour_water_agent/fast_gym_config.json", "configs/gym/agent/pour_water_agent/agent_config.json"),
    "DualPourWater": ("configs/gym/agent/pour_water_agent/fast_gym_config.json", "configs/gym/agent/pour_water_agent/agent_config_dual.json"),
    "Rearrangement": ("configs/gym/agent/rearrangement_agent/fast_gym_config.json", "configs/gym/agent/rearrangement_agent/agent_config.json"),
    # ours (10-28): the paper's real-robot 'setup coffee tray' task, built in their sim (coffee_tray/, see build.py)
    "CoffeeTray": ("/work/probe/coffee_tray/gym_config.json", "/work/probe/coffee_tray/agent_config.json"),
    # same scene; prompt = the paper's own task line, no numbers (coffee_tray/task_prompt_min.txt)
    "CoffeeTrayMin": ("/work/probe/coffee_tray/gym_config.json", "/work/probe/coffee_tray/agent_config_min.json"),
}
HF_PREFIX = "https://huggingface.co/datasets/DexForceAI/embodichain_data/resolve/main/"  # official HF (dexforce/ redirects here)
LIFT_M = 0.03
# --held-rule grip (10-28): "held" = some gripper closed below GRIP_CLOSED and the object raised >= GRIP_LIFT_M. Needed because
# their rearrange lifts fork/spoon only ~1 cm, so the pre-registered lift3cm rule never fired there (0/6 drops applied).
GRIP_CLOSED, GRIP_LIFT_M = 0.045, 0.005


def _np(x):
    try:
        return x.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(x)


class Episode:
    """Per-episode state: RNG, disturbance plan, logs."""

    def __init__(self, env, rng, p, mode, xy_jitter, held_rule="lift3cm", event=None):
        self.env, self.rng, self.p, self.mode, self.xy_jitter, self.held_rule = env, rng, p, mode, xy_jitter, held_rule
        self.event, self.event_log, self.bi_count, self.cur_bi = event, None, {}, None
        self.objs = list(env.sim.get_rigid_object_uid_list())
        self.z0 = {o: float(self.pose(o)[2, 3]) for o in self.objs}
        self.traj, self.edges, self.injections, self.step = [], [], [], 0
        self.n_next, self.cur = None, None

    def pose(self, o):
        return _np(self.env.sim.get_rigid_object(o).get_local_pose(to_matrix=True)).reshape(4, 4)

    def grip(self):
        """Gripper opening per arm, read with THEIR monitor's own function (what hold_lost compares against its threshold)."""
        from embodichain.agents.agentchord.monitor_utils import get_gripper_distance
        out = {}
        for side in ("left", "right"):
            try:
                out[side] = round(get_gripper_distance(self.env, f"{side}_arm"), 5)
            except Exception:
                out[side] = None
        return out

    # -- atom_actions hooks --
    def tqdm(self, it, *a, **k):
        try:
            self.n_next = len(it)
        except TypeError:
            self.n_next = None
        return it

    def setup(self, enabled=False):
        n = self.n_next or 0
        fire = n > 0 and self.p > 0 and self.rng.random() < self.p
        self.cur = {"n": n, "i": 0, "fire_at": int(self.rng.integers(0, n)) if fire else None}
        return self.cur

    def requested(self, tok):
        self.step += 1
        if self.step % 5 == 0:
            self.traj.append({"t": self.step, "grip": self.grip(), **{o: self.pose(o)[:3, :].round(4).tolist() for o in self.objs}})
        if tok is None:
            return False
        if self.event and self.event_log is None and tok["n"] > 0 and tok["i"] == tok["n"] // 2:
            self.maybe_event()
        hit = tok["fire_at"] is not None and tok["i"] == tok["fire_at"]
        tok["i"] += 1
        return hit

    def inject(self, env):
        from embodichain.agents.agentchord import error_functions as ef
        if self.held_rule == "grip":
            g = self.grip()
            closed = any(v is not None and v < GRIP_CLOSED for v in g.values())
            held = [o for o in self.objs if closed and float(self.pose(o)[2, 3]) > self.z0[o] + GRIP_LIFT_M]
        else:
            held = [o for o in self.objs if float(self.pose(o)[2, 3]) > self.z0[o] + LIFT_M]
        rec = {"step": self.step, "held": held, "mode": self.mode, "applied": False}
        if held and self.mode != "none":
            o = held[int(self.rng.integers(0, len(held)))]
            dz = self.z0[o] - float(self.pose(o)[2, 3]) + 0.005
            dxy = self.rng.uniform(-self.xy_jitter, self.xy_jitter, 2) if self.xy_jitter > 0 else np.zeros(2)
            fn = ef.fallen_object if self.mode == "fallen" else ef.misplaced_object
            fn(env, o, None, [float(dxy[0]), float(dxy[1]), dz])
            rec.update(obj=o, dxyz=[float(dxy[0]), float(dxy[1]), dz], applied=True)
        self.injections.append(rec)

    # -- scheduled blind-spot events (10-28, CoffeeTray only). Fired once, half-way through the targeted edge.
    #   slide_out / nudge_in: during the 2nd bimanual move_by_relative_offset (the forward carry; the 1st is the lift), the capsule
    #     is moved +0.15 m (out of the tray's front) / +0.04 m (still inside) in x. Not hand-induced; the capsule is not held.
    #   bump: during the 1st bimanual grasp (arms approaching the handles), tray AND capsule are moved -0.04 m in y.
    EVENTS = {"slide_out": (("move_by_relative_offset", 2), [("capsule", [0.15, 0.0, 0.0])]),
              "nudge_in": (("move_by_relative_offset", 2), [("capsule", [0.04, 0.0, 0.0])]),
              "bump": (("grasp", 1), [("tray", [0.0, -0.04, 0.0]), ("capsule", [0.0, -0.04, 0.0])])}

    def maybe_event(self):
        when, moves = self.EVENTS[self.event]
        if self.cur_bi != when:
            return
        from embodichain.agents.agentchord import error_functions as ef
        before = {o: self.pose(o)[:3, 3].round(4).tolist() for o, _ in moves}
        for o, d in moves:
            ef.misplaced_object(self.env, o, None, d)
        self.event_log = {"event": self.event, "step": self.step, "edge": list(when), "before": before,
                          "after": {o: self.pose(o)[:3, 3].round(4).tolist() for o, _ in moves}}

    def wrap_drive(self, drive):
        def _d(*a, **k):
            la, ra = k.get("left_arm_action"), k.get("right_arm_action")
            if la is not None and ra is not None:
                fn = _fn_name(la)
                self.bi_count[fn] = self.bi_count.get(fn, 0) + 1
                self.cur_bi = (fn, self.bi_count[fn])
            else:
                self.cur_bi = None
            t0, s0 = time.time(), self.step
            r = drive(*a, **k)
            self.edges.append({"left": _fn_name(la), "right": _fn_name(ra), "steps": self.step - s0, "wall_s": time.time() - t0,
                               "monitor_index": r.get("monitor_index") if isinstance(r, dict) else None,
                               "monitor_name": r.get("monitor_name") if isinstance(r, dict) else None,
                               "monitor_step": r.get("step_index") if isinstance(r, dict) else None,
                               "grip_end": self.grip()})
            return r
        return _d


def _fn_name(a):
    if a is None:
        return None
    if isinstance(a, dict):
        return a.get("fn")
    f = getattr(a, "func", a)
    return getattr(f, "__name__", type(a).__name__)


def patch_tray_grasp():
    """CoffeeTray only: their grasp() reads ONE object-frame grasp pose (env.obj_info[obj]["grasp_pose_obj"]); a two-arm tray
    grasp needs one per arm (left rim / right rim). Wrap grasp() so that, for the tray, the pose of the calling arm is put in
    obj_info first. Their skill, IK and planner are otherwise unchanged; the planner still writes grasp(left_arm, tray)."""
    import functools
    from embodichain.agents.agentchord import atom_actions
    import coffee_tray.build as ctb
    orig = atom_actions.grasp

    @functools.wraps(orig)
    def grasp(robot_name, obj_name, *a, env=None, **kw):
        if obj_name == "tray" and env is not None and "tray" in getattr(env, "obj_info", {}):
            import torch
            side = "left_arm" if "left" in robot_name else "right_arm"
            env.obj_info["tray"]["grasp_pose_obj"] = torch.tensor(ctb.TRAY_GRASP[side], dtype=torch.float32)
        return orig(robot_name, obj_name, *a, env=env, **kw)

    atom_actions.grasp = grasp


def patch_pk_device():
    """Fix (10-28): their create_pk_serial_chain(chain=...) builds pk.SerialChain(chain, end, root) with no device/dtype, so
    pytorch_kinematics 0.10.0 makes the sub-chain on CPU while the frames of a CUDA chain stay on CUDA -> device-mismatch crash
    with --device cuda. Pass the source chain's device and dtype. Patched in every module that imported the function by name."""
    import importlib
    su = importlib.import_module("embodichain.lab.sim.utility.solver_utils")
    orig = su.create_pk_serial_chain

    # pk 0.10.0's Chain.__init__ is itself not device-clean on CUDA (joint_type_indices stays on CPU while _dof_frame_indices
    # goes to the device), but Chain.to() moves every tensor. So: deep-copy the chain to CPU, build there, then .to(device).
    def fixed(urdf_path=None, device=None, end_link_name=None, root_link_name=None, chain=None, **kw):
        if chain is not None and urdf_path is None:
            import copy
            import pytorch_kinematics as pk
            src = copy.deepcopy(chain).to(device="cpu")
            sc = pk.SerialChain(chain=src, end_frame_name=end_link_name, root_frame_name=root_link_name or "",
                                device="cpu", dtype=chain.dtype)
            return sc.to(device=chain.device)
        return orig(urdf_path=urdf_path, device=device, end_link_name=end_link_name, root_link_name=root_link_name, chain=chain, **kw)

    for m in ("embodichain.lab.sim.utility.solver_utils", "embodichain.lab.sim.solvers.base_solver",
              "embodichain.lab.sim.objects.articulation"):
        mod = importlib.import_module(m)
        if hasattr(mod, "create_pk_serial_chain"):
            mod.create_pk_serial_chain = fixed


def patch_gym_config(path, out_dir):
    """Their released configs use texture randomizers on 'DexsimMaterials/<X>' folders that DexForce never published (release bug,
    10-28). We built substitutes under $EMBODICHAIN_DATA_ROOT/DexsimMaterials from DexForce's own BackgroundTexture pack (see its
    PROVENANCE.json). An event is dropped (into a copy in the run dir) only if its folder is missing; physics/layout events are kept."""
    cfg = json.load(open(path))
    removed = []
    root = os.environ.get("EMBODICHAIN_DATA_ROOT", "")

    def missing(tp):
        return not os.path.isdir(os.path.join(root, tp))

    def walk(d, where):
        if isinstance(d, dict):
            for k in list(d):
                v = d[k]
                tp = v.get("params", {}).get("texture_path", "") if isinstance(v, dict) else ""
                if isinstance(tp, str) and tp.startswith("DexsimMaterials") and missing(tp):
                    removed.append(f"{where}/{k}")
                    del d[k]
                else:
                    walk(v, f"{where}/{k}")
        elif isinstance(d, list):
            for i, v in enumerate(d):
                walk(v, f"{where}[{i}]")
    walk(cfg, "")
    if not removed:
        return path, removed
    out = os.path.join(out_dir, "gym_config_patched.json")
    json.dump(cfg, open(out, "w"), indent=1)
    return out, removed


def strict_pour(ep, task, tol=0.03):
    """Stricter pour success (pre-registered): bottle reached the pour offset from the cup while tilted >= 60 deg, ends upright
    within tol of its return point, and nothing tipped at the end. Offsets/targets are the ones written in their task prompts."""
    off, ret_xy = np.array([0.05, -0.10, 0.125]), np.array([0.7, -0.1])
    reached = False
    for fr in ep.traj:
        b, c = np.array(fr["bottle"]), np.array(fr["cup"])
        tilt = np.degrees(np.arccos(np.clip(b[2, 2], -1, 1)))
        if np.linalg.norm(b[:, 3] - (c[:, 3] + off)) < tol and tilt >= 60:
            reached = True
            break
    bf = ep.pose("bottle")
    returned = bool(np.linalg.norm(bf[:2, 3] - ret_xy) < tol)
    upright = all(np.degrees(np.arccos(np.clip(ep.pose(o)[2, 2], -1, 1))) < 45 for o in ("bottle", "cup"))
    out = {"reached_pour_pose": reached, "bottle_returned": returned, "upright": upright}
    if task == "DualPourWater":
        out["cup_placed"] = bool(np.linalg.norm(ep.pose("cup")[:2, 3] - np.array([0.6, 0.1])) < tol)
    out["strict_success"] = all(out.values())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p", type=float, default=0.0, help="per-atomic-action drop probability")
    ap.add_argument("--drop-mode", default="upright", choices=["upright", "fallen", "none"])
    ap.add_argument("--xy-jitter", type=float, default=0.0)
    ap.add_argument("--recovery", action="store_true")
    ap.add_argument("--regenerate", action="store_true", help="ask the LLM for a new plan (first episode only); else reuse log_dir")
    ap.add_argument("--plan-dir", required=True, help="where the 3 generated JSONs live (their log_dir)")
    ap.add_argument("--noop", action="store_true", help="control: reset, skip the plan, score success")
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="/work/src")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--save-obs", action="store_true", help="save the planner's camera image after each reset")
    ap.add_argument("--event", default=None, choices=["slide_out", "nudge_in", "bump"], help="one scheduled blind-spot event")
    ap.add_argument("--tray-mass", type=float, default=None, help="set the tray mass [kg] after each reset")
    ap.add_argument("--gym-config", default=None, help="override the task's gym config path (in-container path)")
    ap.add_argument("--held-rule", default="lift3cm", choices=["lift3cm", "grip"],
                    help="lift3cm = pre-registered (object > 3 cm above start); grip = gripper closed and object raised >= 0.5 cm")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    # Their code uses repo-relative paths (configs/..., ./outputs/videos). The clone is bound read-only, so run from a writable
    # work dir that links every top-level entry of the clone; ./outputs then lands in the run dir.
    work = os.path.join(a.out, "cwd")
    os.makedirs(os.path.join(work, "outputs"), exist_ok=True)
    for e in os.listdir(a.src):
        dst = os.path.join(work, e)
        if e != "outputs" and not os.path.lexists(dst):
            os.symlink(os.path.join(a.src, e), dst)
    os.chdir(work)
    sys.path.insert(0, a.src)
    import gymnasium
    import torch
    from embodichain.lab.gym.utils.gym_utils import add_env_launcher_args_to_parser, build_env_cfg_from_args
    from embodichain.lab.scripts import run_env
    from embodichain.agents.agentchord import atom_actions, agent_graph

    import importlib
    for m in ("w1_assets", "scene_assets", "materials", "robot_assets", "demo_assets", "eef_assets", "obj_assets"):
        importlib.import_module(f"embodichain.data.assets.{m}")  # import now so the prefix patch below reaches every asset module
    for name, mod in list(sys.modules.items()):  # assets from huggingface.co, not the hf-mirror.com default (read at call time)
        if name.startswith("embodichain.data") and hasattr(mod, "EMBODICHAIN_DOWNLOAD_PREFIX"):
            mod.EMBODICHAIN_DOWNLOAD_PREFIX = HF_PREFIX

    patch_pk_device()
    if a.task.startswith("CoffeeTray"):
        import coffee_tray.env  # noqa: F401  (registers CoffeeTrayAgent-v3)
        patch_tray_grasp()
    gym_cfg, agent_cfg = TASKS[a.task]
    if a.gym_config:   # 10-28: scene-layout variant (e.g. coffee_tray/variants/gym_config_<tag>.json); task/prompt unchanged
        gym_cfg = a.gym_config
    gym_cfg, removed = patch_gym_config(gym_cfg, a.out)
    p = argparse.ArgumentParser()
    add_env_launcher_args_to_parser(p)
    # --device cpu = their launcher default; with cuda their pk.SerialChain(chain=...) call mixes cuda/cpu tensors (pytorch_kinematics
    # 0.10.0, verified 10-28). DexSim still renders on the GPU (Vulkan, --gpu_id).
    # --filter_dataset_saving: their LeRobot recorder is for imitation-data generation, not evaluation (and needs lerobot).
    largs = p.parse_args(["--gym_config", gym_cfg, "--headless", "--device", a.device, "--gpu_id", str(a.gpu_id),
                          "--filter_dataset_saving"])
    env_cfg, gym_config, _ = build_env_cfg_from_args(largs)
    from embodichain.utils.utility import load_json
    env = gymnasium.make(id=gym_config["id"], cfg=env_cfg, agent_config=load_json(agent_cfg),  # as in their run_agent.cli
                         agent_config_path=agent_cfg, task_name=a.task)
    rng = np.random.default_rng(a.seed)
    orig_drive = agent_graph.drive
    meta = dict(vars(a), lift_threshold_m=LIFT_M, gym_config_used=gym_cfg, removed_events=removed, assumptions=__doc__.split("Our reading")[1].split("Their task")[0].strip(),
                llm_url=os.environ.get("LLM_URL"), started=time.time())
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1, default=str)

    for k in range(a.episodes):
        torch.manual_seed(a.seed * 1000 + k)
        env.reset()
        ep = Episode(env.unwrapped, rng, a.p, a.drop_mode, a.xy_jitter, a.held_rule, a.event)
        mass_log = None
        if a.tray_mass is not None:   # physics-hidden: heavier tray (planner/monitors never see mass)
            obj = env.unwrapped.sim.get_rigid_object("tray")
            obj.set_mass(torch.tensor([a.tray_mass], dtype=torch.float32))
            mass_log = {"tray_mass_set": a.tray_mass}
        if a.save_obs:  # the exact image their task agent sends to the planner model (observations["rgb"])
            try:
                import cv2
                rgb = _np(env.unwrapped.get_obs_for_agent()["rgb"]).astype(np.uint8)
                cv2.imwrite(os.path.join(a.out, f"agent_obs_{k:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            except Exception as e:  # noqa: BLE001
                print(f"[ac_run] save_obs failed: {e!r}", flush=True)
        atom_actions.tqdm, atom_actions.setup_interactive_error_input = ep.tqdm, ep.setup
        atom_actions.interactive_error_requested, atom_actions.inject_interactive_error = ep.requested, ep.inject
        atom_actions.restore_interactive_error_input = lambda tok: None
        agent_graph.drive = ep.wrap_drive(orig_drive)
        row = {"episode": k, "seed": a.seed, "error": None}
        t0 = time.time()
        try:
            if not a.noop:
                run_env.generate_and_execute_action_list(env, 0, False, regenerate=(a.regenerate and k == 0), recovery=a.recovery,
                                                         interactive_error_injection=True, log_dir=a.plan_dir)
        except Exception as e:  # noqa: BLE001 -- a crash (no recovery branch, max transitions, IK) is an outcome to record
            row["error"] = f"{type(e).__name__}: {e}"
            row["traceback"] = traceback.format_exc()[-2000:]
        row["wall_s"] = time.time() - t0
        try:
            row["their_success"] = bool(_np(env.unwrapped.is_task_success()).reshape(-1)[0])
        except Exception as e:  # noqa: BLE001
            row["their_success"], row["success_error"] = None, repr(e)
        if "PourWater" in a.task:
            row.update(strict_pour(ep, a.task))
        if a.task.startswith("CoffeeTray"):
            try:
                row["tray_check"] = env.unwrapped.check_detail()
            except Exception as e:  # noqa: BLE001
                row["tray_check"] = {"error": repr(e)}
        row.update(event=ep.event_log, event_requested=a.event, mass=mass_log)
        row.update(steps=ep.step, edges=ep.edges, injections=ep.injections,
                   n_monitor_fired=sum(e["monitor_name"] is not None for e in ep.edges),
                   n_injected=sum(i["applied"] for i in ep.injections))
        with open(os.path.join(a.out, "episodes.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        np.save(os.path.join(a.out, f"traj_{k:03d}.npy"), np.array(ep.traj, dtype=object), allow_pickle=True)
        print(f"[ac_run] ep {k} their={row['their_success']} strict={row.get('strict_success')} inj={row['n_injected']} "
              f"mon={row['n_monitor_fired']} err={row['error']}", flush=True)
    agent_graph.drive = orig_drive


if __name__ == "__main__":
    main()
