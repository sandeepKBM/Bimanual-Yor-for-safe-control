"""teleop/sim.py -- the closed simulation loop for one teleoperated episode (SIMULATION ONLY).

Control hierarchy and rates
  physics        200 Hz (dt 5 ms), MuJoCo implicitfast
  FL-MPC (arms)  every physics step (200 Hz, as phase2); encoder noise 1e-4 rad / 2e-3 rad/s on what it reads
  teleop tick    50 Hz (20 ms = 4 physics steps): operator command -> (shield) -> base twist target / lift rate / tray-frame jog
  base           kinematic mocap; slew-limited twist (base.A_HW), integrated at the physics rate
  lift           eq-joint target (locked mode) rate-limited; |v| <= 0.12 m/s, hardware slew 1.5 m/s^2
  tray jog       ObjectFrame (tray frame jog -> DLS IK -> joint references of the unmodified FL-MPC)
Grasp = weld activation (documented simplification, see TeleopSim.grasp) -- default grasp_mode="weld".
grasp_mode="friction" (OPT-IN): Cmd.grasp starts the wrist-camera-guided REAL friction grasp (house/grasp/): no weld, the pads hold the
tray; the base is dynamic (follows the kinematic target through a weld) so base acceleration reaches the tray through the contacts.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import numpy as np
import mujoco

from .. import build as B
from . import base as BS
from . import geom as GM
from . import scene as SC
from .arm import ArmCtl, ObjectFrame, ENC_Q, ENC_V

DT = 0.005
N_SUB = 4
TICK = DT * N_SUB                 # 20 ms teleop / shield rate
LIFT_V_MAX, LIFT_A_HW = 0.12, 1.5
GRASP_TOL_POS, GRASP_TOL_ANG = 0.035, math.radians(6.0)
PERSON_TORSO_R = SC.PERSON_R
PERSON_YIELD_M = 0.15        # a walker stops when the robot footprint is closer than this to its torso surface (people avoid a stationary robot)


@dataclass
class PersonPolicy:
    """per-person reaction rule of a scripted walker (people_clock="time" only).
    yield_m   : stop while the robot footprint is closer than this to the torso surface (None = OBLIVIOUS: never yields, even in contact)
    giveup_s  : after this long (sim s) of continuous yielding the person gives up waiting (None = waits forever)
    giveup_mode: "continue" = resume the script ignoring the robot (walks through); "detour" = side-step away from the robot
                 (lateral offset ramps to `detour_m` at `detour_speed` m/s, perpendicular to the track heading) and keep walking"""
    yield_m: float | None = PERSON_YIELD_M
    giveup_s: float | None = None
    giveup_mode: str = "continue"
    detour_m: float = 0.8
    detour_speed: float = 0.6


@dataclass
class Cmd:
    twist: np.ndarray = field(default_factory=lambda: np.zeros(3))      # body twist target (vx, vy, wz)
    lift_v: float = 0.0                                                 # lift velocity target [m/s]
    jog: np.ndarray = field(default_factory=lambda: np.zeros(6))        # tray-frame twist in the base frame (v3, w3)
    grasp: bool = False
    release: bool = False


class TeleopSim:
    def __init__(self, scene_name="h5", grasped=True, seed=0, with_people=True, counter=False, desk=False, tilt=True,
                 spawn=None, a_hw=None, grasp_mode="weld", people_clock="legacy", people_policy=None, people_source="truth",
                 pose_source="truth", loc_cfg=None, loc_seed=None, occ_source="true", occ_map=None):
        """pose_source="truth" (default, bit-identical): control-path consumers (shield wall keep-out) read the mocap base pose.  "est": they read
        `ctrl_pose()` = house/loc PoseEstimator (wheel odometry + ZED VIO, `loc_cfg` = LocCfg, `loc_seed` default = seed); truth stays for
        metrics/eval/operator.  occ_source="true" (default): shield wall grid = the true house grid; "perceived": `occ_map` (loc.mapper
        PerceivedMap / OccGrid / npz path; None -> built/cached by loc.mapper for this scene) = grid from a head-ZED mapping pass.
        people_clock="legacy" (default, bit-identical): a walker's track clock only runs while the CURRENT track velocity is non-zero, so a
        track that starts with a stationary segment never moves (h5 p_walker).  "time": the clock runs with sim time (every tick) except while
        the person yields; `people_policy` = PersonPolicy (all dynamic people) or {name: PersonPolicy}."""
        assert grasp_mode in ("weld", "friction"), grasp_mode
        assert people_clock in ("legacy", "time"), people_clock
        assert people_source in ("truth", "tracker"), people_source
        self.people_source = people_source        # what the SHIELD sees (make_shield); self.people_state is always the ground truth (metrics)
        self.people_clock = people_clock
        self.people_policy = people_policy
        self._yield_since, self._gave_up, self._detour = {}, {}, {}
        self.grasp_mode = grasp_mode
        if grasp_mode == "friction":
            grasped = False                       # nothing is welded; the assisted grasp establishes the hold
        self.scene = SC.get_scene(scene_name)
        wp = self.scene.waypoints
        spawn = spawn or wp["lab_start"]
        pick = wp["cafe_counter_pick"]
        counter_at = pick if counter else None
        desk_at = wp["lab_desk_place"] if desk else None
        tray_at = None
        if counter:                          # tray + cup rest on the counter at the pick pose (hold-pose position of that base pose)
            c, s = math.cos(pick[2]), math.sin(pick[2])
            tx = pick[0] + c * SC.TRAY_LOCAL[0] - s * SC.TRAY_LOCAL[1]
            ty = pick[1] + s * SC.TRAY_LOCAL[0] + c * SC.TRAY_LOCAL[1]
            tray_at = (tx, ty, pick[2] + math.pi / 2, SC.TRAY_HOLD_Z)
        self.rig = SC.new_rig(scene_name, spawn=spawn, grasped=grasped, dt=DT, with_people=with_people,
                              counter_at=counter_at, desk_at=desk_at, tray_at=tray_at,
                              dynamic_base=grasp_mode == "friction", finger_contact=grasp_mode == "friction")
        m, d = self.rig.m, self.rig.d
        self.m, self.d = m, d
        self.grasped = grasped
        self.with_people = with_people
        if grasp_mode == "friction":
            from ..grasp.ctl import FFArmCtl       # FL-MPC + base-acceleration / payload feed-forward (dynamic base)
            self.ctl = FFArmCtl(self.rig, tilt=False)
        else:
            self.ctl = ArmCtl(self.rig, tilt=tilt and grasped)     # the tray-tilt cone is only meaningful while a tray is held
        self._tilt_cfg = tilt
        self.of = ObjectFrame(self.ctl, self.rig)
        self.base = BS.BaseSim(spawn[0], spawn[1], spawn[2])
        self.a_hw = BS.A_HW if a_hw is None else np.asarray(a_hw)
        self.rng = np.random.default_rng(10_000 + seed)           # encoder noise only (operator has its own rng)
        self.dn = mujoco.MjData(m)
        self.t = 0.0
        self.k = 0
        self.lift_h = 0.0
        self.lift_v = 0.0
        self.tray, self.cup = m.body("tray").id, m.body("cup").id
        self.tray_adr = int(m.jnt_dofadr[m.joint("tray_free").id])
        self.eq = {s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"hold_{s}") for s in ("left", "right")}
        self.lift_eq = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, f"lift_lock_{i}") for i in (1, 2)]
        self.slider_q = [int(m.jnt_qposadr[m.joint(f"Slider_{i}").id]) for i in (1, 2)]
        from world_model.v2.data import contents as C
        self.C = C
        self.orc = C.Oracle(m, self.rig.n_beads)
        self.bead_out = np.zeros(self.rig.n_beads, bool)            # sticky
        self.bead_out_t = np.full(self.rig.n_beads, np.nan)
        self.cup_lost_t = None
        self.occ = SC.load_occupancy(self.scene)                       # TRUTH grid: metrics / eval only when occ_source != "true"
        assert pose_source in ("truth", "est") and occ_source in ("true", "perceived"), (pose_source, occ_source)
        self.pose_source, self.occ_source = pose_source, occ_source
        self.occ_ctrl = self.occ if occ_source == "true" else SC.load_perceived_occupancy(self.scene, occ_map)   # what the control path (shield) reads
        self.loc = None
        if pose_source == "est":
            from ..loc.estimator import PoseEstimator
            self.loc = PoseEstimator(loc_cfg, seed=seed if loc_seed is None else loc_seed, true_pose0=(spawn[0], spawn[1], spawn[2]))
        self._fp_pts = GM.boundary_pts(step=0.02)
        # house (static world) geoms for physical bump counting
        own = B._robot_tray_bodies(m)
        import re
        for b in range(m.nbody):
            nm = m.body(b).name or ""
            if nm in ("cup", "tp_counter", "tp_desk") or re.fullmatch(r"ball_\d+", nm) or nm in [p.name for p in self.scene.people if p.dynamic]:
                own.add(b)
        self.house_geom = np.array([int(m.geom_bodyid[g]) not in own and m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE for g in range(m.ngeom)])
        self.robot_geom = np.array([int(m.geom_bodyid[g]) in B._robot_tray_bodies(m) for g in range(m.ngeom)])
        self.tray_vel_prev = np.zeros(3)
        self.tray_acc = np.zeros(3)
        self.bump_contacts = 0
        self.ptime = {}
        self.set_people()
        mujoco.mj_forward(m, d)
        self.assist = None
        if grasp_mode == "friction":
            from ..grasp.assist import GraspAssist
            self.assist = GraspAssist(self)

    # ------------------------------------------------------------------ helpers
    def _dyn(self):
        return [p for p in self.scene.people if p.dynamic]

    def set_people(self, advance=False):
        """people_state = [(xy, velocity)] for ALL scene people (moving mocap capsules and static spec mannequins).  Moving people have
        their own track clock that STOPS while the robot footprint is within PERSON_YIELD_M of their torso (people wait for / avoid a
        slow or stationary robot, but a fast robot still reaches them: the yield distance is 0.30 m)."""
        if self.people_clock == "time":
            return self._set_people_time(advance)
        pts = []
        mi_of = {p.name: mi for p, mi in zip(self._dyn(), self.rig.person_mocap)}
        for p in (self.scene.people if self.with_people else []):
            tau = self.ptime.get(p.name, 0.0)
            xy, v = p.at(tau)
            if advance and p.dynamic and np.linalg.norm(v) > 0:
                dd, _, _ = GM.rect_dist_point(xy, self.pose())
                if dd - PERSON_TORSO_R >= PERSON_YIELD_M:
                    tau += TICK
                    self.ptime[p.name] = tau
                    xy, v = p.at(tau)
                else:
                    v = np.zeros(2)
            if p.name in mi_of:
                self.d.mocap_pos[mi_of[p.name]] = (xy[0], xy[1], 0.0)
            pts.append((xy, v))
        self.people_state = pts
        return pts

    def make_shield(self, cfg=None, **kw):
        """the shield for this sim's people source: "truth" -> the original Shield on people_state (bit-identical); "tracker" -> head-camera detections ->
        KF tracks -> inflated keep-out (people_track.source.make_shield; kwargs: perception, look, look_cfg, conformal, ucfg, seed)."""
        from ..people_track.source import make_shield
        return make_shield(self, self.people_source, cfg, **kw)

    def policy_of(self, name):
        pol = self.people_policy
        if isinstance(pol, dict):
            pol = pol.get(name)
        return pol if pol is not None else PersonPolicy()

    def _set_people_time(self, advance):
        """people_clock="time": track clock advances with sim time; yield / give-up per PersonPolicy.  people_state velocities are the TRUE
        velocities (script velocity + detour velocity; zero while yielding)."""
        pts = []
        mi_of = {p.name: mi for p, mi in zip(self._dyn(), self.rig.person_mocap)}
        self.person_yielding = getattr(self, "person_yielding", {})
        for p in (self.scene.people if self.with_people else []):
            tau = self.ptime.get(p.name, 0.0)
            xy, v = p.at(tau)
            off = self._detour.get(p.name, np.zeros(2))
            dv = np.zeros(2)
            if advance and p.dynamic and tau < p.pts[-1, 0]:
                pol = self.policy_of(p.name)
                dd, _, nw = GM.rect_dist_point(xy + off, self.pose())
                close = pol.yield_m is not None and dd - PERSON_TORSO_R < pol.yield_m
                gave = self._gave_up.get(p.name, False)
                if close and not gave:
                    t0 = self._yield_since.setdefault(p.name, self.t)
                    if pol.giveup_s is not None and self.t - t0 >= pol.giveup_s:
                        self._gave_up[p.name] = gave = True
                    else:
                        self.person_yielding[p.name] = True
                        v = np.zeros(2)
                if not (close and not gave):
                    self._yield_since.pop(p.name, None)
                    self.person_yielding[p.name] = False
                    tau += TICK
                    self.ptime[p.name] = tau
                    xy, v = p.at(tau)
                    if gave and pol.giveup_mode == "detour":     # side-step away from the robot, perpendicular to the heading
                        h = v if np.linalg.norm(v) > 1e-6 else np.array([1.0, 0.0])
                        perp = np.array([-h[1], h[0]]) / (np.linalg.norm(h) + 1e-12)
                        if np.dot(perp, nw) < 0:                 # nw = unit vector robot -> person: step to the side AWAY from the robot
                            perp = -perp
                        tgt = perp * pol.detour_m
                        step = pol.detour_speed * TICK
                        e = tgt - off
                        ne = np.linalg.norm(e)
                        newoff = tgt if ne <= step else off + e / ne * step
                        dv = (newoff - off) / TICK
                        self._detour[p.name] = off = newoff
            xy = xy + self._detour.get(p.name, np.zeros(2))
            v = v + dv
            if p.name in mi_of:
                self.d.mocap_pos[mi_of[p.name]] = (xy[0], xy[1], 0.0)
            pts.append((xy, v))
        self.people_state = pts
        return pts

    def set_lift_target(self, h):
        for e in self.lift_eq:
            self.m.eq_data[e, 0] = h / 2

    def lift_actual(self):
        return float(sum(self.d.qpos[q] for q in self.slider_q))

    def cup_rel(self):
        """cup base position in the tray frame (x, y, z) and tilt of the cup axis from vertical [deg]."""
        d = self.d
        Rt = d.xmat[self.tray].reshape(3, 3)
        rel = Rt.T @ (d.xpos[self.cup] - d.xpos[self.tray])
        Rc = d.xmat[self.cup].reshape(3, 3)
        return rel, math.degrees(math.acos(np.clip(Rc[2, 2], -1, 1)))

    def tray_tilt_deg(self):
        R = self.d.xmat[self.tray].reshape(3, 3)
        return math.degrees(math.acos(np.clip(R[2, 2], -1, 1)))

    def bead_check(self):
        loc = self.orc.local(self.d, self.d.xpos[self.cup], self.d.xquat[self.cup])
        out = self.C.Oracle.out_mask(loc)
        new = out & ~self.bead_out
        self.bead_out_t[new] = self.t
        self.bead_out |= out
        return int(self.bead_out.sum())

    def grasp_errors(self):
        return self.of.handle_errors(self.d, **self._tray_obs_kw())

    def _tray_obs_kw(self):
        """BIMANUAL_PROPRIO_ONLY (friction rig): the operator-visible tray pose is the wrist-camera estimate / hand FK, not the tray body."""
        a = self.assist
        if a is not None and os.environ.get("BIMANUAL_ALIGN_CUE", "") == "cam":
            # ALIGN cue without the tray body (house/grasp: external tracker + wrist-camera memory, HOLD = hand FK); None = no cue -> can_grasp False
            return dict(tray=a.operator_tray_estimate())
        if a is not None and a.cfg.proprio:
            if a.state == "HOLD":                      # grasped: the tray is where the hands + the measured grasp offset put it
                return dict(tray=a.tray_estimate_for_operator())
            if os.environ.get("BIMANUAL_PROPRIO_OPERATOR", "0") == "1":
                # STRICT: the operator cue also comes from the wrist cameras.  They are BLIND at the hold geometry (min range 0.10 m), so ALIGN never
                # sees the tray -> can_grasp stays False.  Default OFF: before the grasp the operator model keeps its (human-eye) view of the tray.
                return dict(tray=a.tray_estimate_for_operator())
        return {}

    def can_grasp(self):
        e = self.grasp_errors()
        return all(p < GRASP_TOL_POS and a < GRASP_TOL_ANG for p, a in e)

    def grasp(self):
        """SIMPLIFICATION (grasping is out of scope): activate both tray welds at the CURRENT relative pose (no snap) once the grippers
        are within GRASP_TOL_POS / GRASP_TOL_ANG of the tray handles.  Returns True on success."""
        if self.assist is not None:               # friction mode: wrist-camera-guided assisted grasp (asynchronous, see GraspAssist)
            return self.assist.start()
        if self.grasped:
            return True
        if not self.can_grasp():
            return False
        for s, b in (("left", "left_arm_gripper_base"), ("right", "right_arm_gripper_base")):
            B.capture_weld(self.m, self.d, f"hold_{s}", b, "tray")
        mujoco.mj_forward(self.m, self.d)
        self.grasped = True
        self.ctl.tilt = self._tilt_cfg
        return True

    def release(self):
        if self.assist is not None:
            self.assist.release()
            return
        for s in ("left", "right"):
            self.d.eq_active[self.eq[s]] = 0
        self.grasped = False
        self.ctl.tilt = False

    def pose(self):
        """TRUE base pose (mocap).  Metrics / evaluation / the operator model.  Control-path consumers use ctrl_pose()."""
        return np.array([self.base.x, self.base.y, self.base.yaw])

    def ctrl_pose(self):
        """base pose as the CONTROL PATH sees it: the PoseEstimator output (pose_source="est", latest tick) or the truth (default)."""
        return self.pose() if self.loc is None else self.loc.pose.copy()

    # ------------------------------------------------------------------ metrics from the geometry
    def geometry_metrics(self):
        pose = self.pose()
        wall = GM.wall_clearance(self.occ, pose, self._fp_pts)
        pdist = np.inf
        for xy, v in self.people_state:
            dd, _, _ = GM.rect_dist_point(xy, pose)
            pdist = min(pdist, dd - PERSON_TORSO_R)
        return wall, float(pdist)

    # ------------------------------------------------------------------ one teleop tick
    def tick(self, cmd: Cmd):
        m, d, ctl = self.m, self.d, self.ctl
        if self.assist is not None:
            cmd = self.assist.gate(cmd)         # base/lift/jog are held while the assisted grasp is in progress (lift only after verify)
        if cmd.grasp:
            self.grasp()
        if cmd.release:
            self.release()
        # lift: slew-limited rate toward the commanded rate, reference integrated
        dv = np.clip(cmd.lift_v - self.lift_v, -LIFT_A_HW * TICK, LIFT_A_HW * TICK)
        self.lift_v = float(np.clip(self.lift_v + dv, -LIFT_V_MAX, LIFT_V_MAX))
        new_h = float(np.clip(self.lift_h + self.lift_v * TICK, 0.0, SC.H_MAX))
        if new_h in (0.0, SC.H_MAX):
            self.lift_v = 0.0
        self.lift_h = new_h
        self.set_lift_target(self.lift_h)
        # tray-frame jog -> joint reference (FL-MPC tracks it)
        if np.any(cmd.jog) or self._jog_pending:
            self.of.jog(cmd.jog, TICK)
            self._jog_pending = bool(np.any(cmd.jog))
        if self.assist is not None:
            self.assist.tick_update()            # wrist cameras -> estimate -> hand targets (visual servo), at the camera rate
        self.set_people(advance=True)        # people move (or yield) once per tick, from the robot pose at the tick start
        v0 = self.base.v.copy()
        poses, v1, acc = self.base.advance(cmd.twist, TICK, self.a_hw, N_SUB)
        self.last_base_acc = acc
        if self.assist is not None:
            self.assist.set_feedforward(0.5 * (v0 + v1), acc, poses[N_SUB // 2][2])
        for s in range(N_SUB):
            d.mocap_pos[0] = (poses[s, 0], poses[s, 1], 0.0)
            d.mocap_quat[0] = B.yaw_quat(poses[s, 2])
            self.t += DT
            mujoco.mj_copyData(self.dn, m, d)
            self.dn.qpos[ctl.lq] += self.rng.normal(0, ENC_Q, 6); self.dn.qpos[ctl.rq] += self.rng.normal(0, ENC_Q, 6)
            self.dn.qvel[ctl.ld] += self.rng.normal(0, ENC_V, 6); self.dn.qvel[ctl.rd] += self.rng.normal(0, ENC_V, 6)
            te, ti = ctl.tau(self.dn)
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[ctl.ld] = te
            d.qfrc_applied[ctl.rd] = ti
            if self.assist is not None:
                self.assist.substep()           # finger force command (per physics step)
            mujoco.mj_step(m, d)
            if s == N_SUB - 1:
                self._count_bumps()
        self.k += 1
        if self.loc is not None:
            self.loc.update(self.t, self.pose(), TICK)          # sensors measure the true motion; the estimate is read at the NEXT tick
        v_tray = d.qvel[self.tray_adr:self.tray_adr + 3].copy()
        self.tray_acc = (v_tray - self.tray_vel_prev) / TICK
        self.tray_vel_prev = v_tray
        nb = self.bead_check()
        rel, cup_tilt = self.cup_rel()
        if self.cup_lost_t is None and (abs(rel[0]) > 0.15 + 0.01 or abs(rel[1]) > 0.10 + 0.01 or rel[2] < -0.03):
            self.cup_lost_t = self.t
        wall, pdist = self.geometry_metrics()
        return dict(t=self.t, pose=self.pose(), v=self.base.v.copy(), acc_body=acc.copy(), rel=rel, cup_tilt=cup_tilt,
                    tray_tilt=self.tray_tilt_deg(), tray_acc=self.tray_acc.copy(), beads_out=nb, wall=wall, pdist=pdist,
                    infeasible=not ctl.last_ok, lift_h=self.lift_actual(), lift_ref=self.lift_h, grasped=self.grasped,
                    bump_contacts=self.bump_contacts, **({"grasp_state": self.assist.state} if self.assist is not None else {}),
                    **({"pose_est": self.loc.pose.copy()} if self.loc is not None else {}))

    _jog_pending = False

    def _count_bumps(self):
        d = self.d
        n = 0
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = c.geom1, c.geom2
            if (self.house_geom[g1] and self.robot_geom[g2]) or (self.house_geom[g2] and self.robot_geom[g1]):
                n += 1
        self.bump_contacts += n

    def observe(self):
        """operator-visible state (the operator model delays it itself)."""
        ep, er = self.of.grasp_error_vec(self.d, **self._tray_obs_kw())
        return dict(t=self.t, pose=self.pose(), v=self.base.v.copy(), lift_h=self.lift_actual(), grasped=self.grasped,
                    grasp_err_p=ep, grasp_err_r=er, can_grasp=self.can_grasp(),
                    **({"pose_est": self.loc.pose.copy(), "v_odo": self.loc.v_odo.copy()} if self.loc is not None else {}))

    def cup_world(self):
        return self.d.xpos[self.cup].copy()
