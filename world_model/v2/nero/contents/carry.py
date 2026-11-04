"""nero/contents/carry.py -- hidden-contents carry tasks under the segment certificate (step 2, user 11-01/11-02).  SIMULATION ONLY.

Tasks (scenes.py geometry; grasp points = the pot's, so the arms' pot grasp q is reused):
  tray  coffee tray with n mugs (hidden fill 0..250 ml each, hidden placement): lift 5 cm, carry 10 cm toward the robot, place on
        the counter.  (AgentChord's "setup coffee tray" places the tray 10 cm forward; +x 10 cm is out of the right arm's reach.)
  box   closed box with hidden contents (0.5..5 kg block, anywhere on the floor): lift 18 cm, carry 20 cm toward the robot over the
        shelf, lower 3 cm onto the shelf (top 15 cm above the counter).
Grasp: rigid weld of both claws to the object from t = 0 (ASSUMED, user 11-01: wrist cameras + low-level control handle grasping).
Controller: FL-MPC with the payload model on both arms (pan/handle_grasp.TwoCtl, share 0.5), torque box 16 Nm (MIT command range,
MEASURED-spec), payload mass = empty object + the belief's contents centre (the robot does not know the contents).

Segments and their STORED backups (phases fixed at admission; the only state used at trigger time is the claws' own pose, to
retrace to poses stored when the segment was admitted):
  lift     backup: stop, lower back to the lift start (object back on the counter), let go in place (weld off, retreat 5 cm)
  carry    backup: stop, return to the carry start, lower to the lift start, let go in place
  place    backup: stop, retrace place -> carry -> lift (back to the counter), let go in place
  release  weld off, open, claws retreat 5 cm outward.  No backup (letting go is not undoable): admitted by the static support row
           (proof under MODEL) plus the sampled worlds.
Unsafe set U (truth, checked every control step): torque saturation (|tau| >= 16 Nm), object tilt > 10 deg or drop, a mug tipped (> 20 deg) or off the tray, spill
(quasi-static surface-tilt proxy from the sim mug acceleration: (D/2) tan(phi) > freeboard; slosh NOT modelled), arm contact with the
environment or the object (gripper-tab contact allowed).
Claw load (vertical force vs (3 kg - gripper) g, twist vs T_cap) is RECORDED from the weld constraint wrench but is NOT in U by
default (strict_claw=True puts it in U): under the rigid weld the two position-tracking arms hold the object with a closed-chain
internal stress set by their tracking errors (11-02 diagnosis: ~19 N pull along the grasp line and 1.2-1.7 Nm twist in the air
with the payload feedforward OFF), so the measured claw wrench reflects the weld + controller, not the hidden contents.  The
claw limits enter the certificate through the static rows (model wrench of the commanded split, proof under MODEL).
USER 11-02 pm: the grasp is not our problem -> no claw rows anywhere (twist cap dropped; T_cap/strict_claw only reproduce old runs).
The claw wrench stays a logged diagnostic.  U above is arm/object-side: joint torque, object/mug tilt, drop, spill, collisions.
R (rest set): object resting on a support (speed < 5 mm/s, tilt < 3 deg), mugs upright (< 5 deg) on the tray, no U entry.
"""
from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time

import mujoco
import numpy as np

from .. import adapters as A
from .. import config as C
from .. import gripper as G
from .. import weldcontact as WC
from ..democ import kin as K
from ..pan import scene as PS
from ..pan.handle_grasp import TwoCtl, _set_fingers
from . import envelope as E
from . import objects as OB
from . import scenes as SC
from . import rules as RU
def _hold_finger():
    from .nyu_hold import hold_finger_frac       # nyu 11-03: V-jaw opening over the whole tab depth (nyu_hold.hold_opening, theta 0.662 rad ASSUMED clearance 2 mm/side)
    return hold_finger_frac()


HOLD_FINGER = _hold_finger()

SIDES = ("left", "right")
DT = 0.005
TAU_PRESET = "mit_fw_ge_1.11"
CAP_FILE = E.CAP_FILE
PLANS = {   # ASSUMED (user-confirmed 11-02); (segment, common TCP delta [m])
    "tray": [("lift", (0, 0, 0.05)), ("carry", (-0.10, 0, 0)), ("place", (0, 0, -0.05))],
    "box": [("lift", (0, 0, 0.18)), ("carry", (-0.20, 0, 0)), ("place", (0, 0, -0.03))],
}
SEGS = ("lift", "carry", "place", "release")
A_BACKUP = 0.25            # ASSUMED backup peak acceleration [m/s^2] (half the nominal)
T_MIN = 0.6                # s, shortest move
RETREAT = 0.05             # m, release retreat
PATH_IK_TOL = 1e-4         # ASSUMED (11-03): pose-IK miss [m; rad norm] above which a commanded path counts as unreachable
PATH_MARGIN_MIN = 0.05     # ASSUMED (11-03): wanted joint-limit distance [rad] at every path waypoint (used by nyu_hold's path check)
PATH_NO_MARGIN = 9.0       # sentinel: no path planned in this window
U_TILT_OBJ, U_TILT_CUP, R_TILT_OBJ, R_TILT_CUP = 10.0, 20.0, 3.0, 5.0      # deg, ASSUMED thresholds
R_SPEED = 0.005            # m/s
SPILL_TAU = 0.025          # s, low-pass on the mug acceleration (spill proxy)


PHYS_DEFAULT = "config.carry_physics"      # sentinel: take fric_comp / weld_solref from config.carry_physics() (item 6)


class LeverCtl(TwoCtl):
    """FL-MPC with a LEVER-split payload feedforward (11-02; new variant, TwoCtl and FL-MPC unchanged).  TwoCtl's 50/50 split
    puts half the load at the COM on EACH claw, i.e. each claw is commanded the moment of half the weight about itself (an internal
    twist of the object a real claw cannot hold: 1.2 Nm per claw for a 1.3 kg tray).  Here each arm's feedforward is the wrench of
    envelope.lever_split for the robot's NOMINAL load (empty object + the belief's contents centre) at a = 0:
        bias_extra_i = J_i^T [F_i; T_i]   (J_i = 6 x 7 TCP-site Jacobian)
        M_extra_i    = clip(alpha_i, 0, 1) M_nom J_p,i^T J_p,i    (point mass share at the TCP; rotational inertia dropped)
    MODEL: the object pose comes from the sim (= the claws' FK under the weld)."""

    def __init__(self, m, hold_q, tau_vec, obj_id, grips, mass, c_rel_nom, sites, **kw):
        super().__init__(m, hold_q, tau_vec, obj_id, grips, mass, share=0.5, **kw)
        self.c_rel_nom = np.asarray(c_rel_nom, float)            # nominal total COM in the obj frame, about obj_ref
        self.sites = sites
        self.ref = m.site("obj_ref").id
        self.rules = None                                        # 11-03 opt-in rules.Rules (table + joint limit rows); None = today's rows only

    def terms(self, d):
        m = self.m
        R = d.xmat[self.pan].reshape(3, 3)
        o = d.site_xpos[self.ref].copy()
        S = self.mass * (R @ self.c_rel_nom)
        acc = -np.asarray(m.opt.gravity, float)
        p = {s: d.site_xpos[self.sites[s]].copy() for s in SIDES}
        FL, TL, FR, TR = E.lever_split(self.mass, acc, S, o, p["left"], p["right"])
        out = {}
        for s, dofs, Fi, Ti in (("left", self.ld, FL, TL), ("right", self.rd, FR, TR)):
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self.sites[s])
            Jp, Jr = jp[:, dofs], jr[:, dofs]
            al = float(np.clip(Fi[2] / max(self.mass * acc[2], 1e-9), 0.0, 1.0))
            out[s] = (al * self.mass * Jp.T @ Jp, Jp.T @ Fi + Jr.T @ Ti)
        return out

    def tau(self, d):
        if not self.on:
            if self.rules is None:
                return super().tau(d)
            mujoco.mj_forward(self.m, d)           # 11-03 rules='full': the release phase goes through the same rule set (no payload terms)
            return self._fl(d, {s: (np.zeros((7, 7)), np.zeros(7)) for s in SIDES})
        mujoco.mj_forward(self.m, d)
        return self._fl(d, self.terms(d))

    def _fl(self, d, t):
        """FL-MPC step with the payload terms t[side] = (M_extra, bias_extra); self.rules (opt-in) adds table + joint-limit rows."""
        kw = {}
        if self.rules is not None:
            Me, ce = RU.arm_dyn(self.m, d, self.ld, t["left"][0], t["left"][1])
            Mi, ci = RU.arm_dyn(self.m, d, self.rd, t["right"][0], t["right"][1])
            kw = self.rules.fl_kwargs(d, Me, ce, Mi, ci, fl=self.fl)
        te, ti, ok = self.fl.step(self.m, d, self.lq, self.ld, self.rq, self.rd, self.qe_ref, self.z, self.qi_ref, self.z,
                                  self.pairs(d), M_ego_extra=t["left"][0], bias_ego_extra=t["left"][1],
                                  M_int_extra=t["right"][0], bias_int_extra=t["right"][1], **kw)
        self.n_infeasible += int(not ok)
        self.n_fallback += int(bool(self.fl.last_fallback))
        return te, ti


FRIC_COMP = dict(f=0.3, v_s=0.02, ki=8.0, i_max=0.6, kp=5.0)     # ASSUMED gains (11-02 test: box tilt 18.7 -> 1.85 deg)
# 11-02 (new, opt-in): joint friction compensation + integral action, added AFTER FL-MPC (mpc_fl.py unchanged).
#   tau_add = f tanh(qdot_cmd / v_s) + ki * int(q_cmd - q) dt (clamped to +-i_max) + kp (q_cmd - q);  total clipped to the torque box.
#   Why: FL-MPC's model has no joint friction and the wrists have tiny inertia, so its torque per rad of wrist error is ~1 Nm/rad while
#   the MJCF frictionloss is 0.3 Nm (config.ASSUMED) -> the wrists (j5, j6) stall 0.2-0.28 rad off the reference and the welded object
#   tilts 10-15 deg (11-02 diagnosis, box 0.59 kg: commanded orientation constant; same with one claw welded or the payload ff off).
#   f = the MJCF frictionloss (MODEL of the sim; the real NERO friction is not known: ASSUMED for hardware).  These torques are not in
#   the static torque row (<= f + i_max = 0.9 Nm per joint, inside the u <= 0.9 margin of 1.6 Nm).


COOP = dict(L=0.19, lam=1e-3, gain=1.0)    # L = orientation weight [m] (~ half the claw spacing); lam = damping of J^+ (MODEL)
# 11-02 (new, opt-in): cooperative layer for the welded closed chain (FL-MPC unchanged; same 5 ms control step, same torque box).
#   Why: each arm tracks its own joint reference stiffly.  Under the weld the claws cannot move relative to each other, so any
#   DIFFERENCE between the two arms' tracking errors cannot be corrected and FL-MPC's gains (and the friction wrapper's integral) turn
#   it into internal force / twist (11-02: 1.33-2.2 Nm measured twist in admitted cells vs 1.3 Nm cap).
#   What: every step, map each claw's joint error to a 6-D claw error e_i = J_i (q_cmd,i - q_i) (encoders + FK only), fit ONE rigid
#   object motion u = (dp, dtheta) about the claw midpoint o by weighted least squares (e_i ~ A_i u, A_i = [[I, -[p_i - o]x], [0, I]],
#   orientation rows weighted by L), and remove the remainder (the internal part e_i - A_i u) from both arms' references:
#   q_ref,i <- q_ref,i - J_i^+ e_int,i (whole horizon).  The arms then correct only the shared object error, together.
#   Only while both welds are on; the joint null-space error (posture) is untouched.


def move_time(dist, a):
    """smoothstep: peak acceleration 6 d / T^2 = a."""
    return max(T_MIN, float(np.sqrt(6.0 * max(dist, 1e-9) / a)))


def commanded_accel(dist, a_plan):
    """11-03 item 10.  The move time is max(T_MIN, sqrt(6 d / a_plan)); the smoothstep TCP path has peak acceleration 6 d / T^2 (MODEL, exact for the
    straight-line smoothstep), which equals a_plan only when T_MIN does not bind.  -> dict(T, a_plan, a_cmd_peak, clamped)."""
    T = move_time(dist, a_plan)
    a_cmd = 6.0 * float(dist) / (T * T)
    return dict(T=round(T, 4), a_plan=float(a_plan), a_cmd_peak=round(a_cmd, 5), clamped=bool(T > np.sqrt(6.0 * max(dist, 1e-9) / a_plan) + 1e-12))


def _tilt_deg(R):
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


class CarryEpisode:
    def __init__(self, world, a_plan=E.A_NOM, m_hat=None, share=0.5, T_cap=None, ctl="lever", strict_claw=False,
                 fric_comp=PHYS_DEFAULT, coop=None, weld_solref=PHYS_DEFAULT, rules=None):
        # 11-03 item 6: PHYS_DEFAULT -> config.carry_physics() (fric_comp=True + stiff weld; legacy pair None/None under BIMANUAL_LEGACY_CARRY_PHYS=1);
        # an explicit None / False / value is honoured as before.
        cp = C.carry_physics()
        fric_comp = cp["fric_comp"] if isinstance(fric_comp, str) and fric_comp == PHYS_DEFAULT else fric_comp
        weld_solref = cp["weld_solref"] if isinstance(weld_solref, str) and weld_solref == PHYS_DEFAULT else weld_solref
        self.world, self.task = world, ("tray" if isinstance(world, SC.TrayWorld) else "box")
        self.a_plan, self.T_cap, self.strict_claw = float(a_plan), T_cap, bool(strict_claw)
        spec = SC.build_spec(world)
        for s in SIDES:
            spec.add_equality(name=f"weld_{s}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=f"{s}_arm_gripper_base", name2="obj", active=False)
        m = self.m = PS.apply_sim_defaults(spec.compile())
        m.opt.timestep = DT
        self.weld_solref = None if weld_solref is None else tuple(float(x) for x in weld_solref)
        if self.weld_solref is not None:      # 11-02 opt-in: stiffer weld (MuJoCo default solref 0.02 s lets the "rigid" weld bend 2-3 deg)
            for s in SIDES:
                m.eq_solref[m.equality(f"weld_{s}").id] = self.weld_solref
        d = self.d = mujoco.MjData(m)
        SC.settle(m, d, 0.5)                                   # objects at rest, arms at their initial pose
        cap = json.loads(CAP_FILE.read_text())["pot"]
        self.addr = {s: A.arm_addrs(m, s)[0] for s in SIDES}
        self.dofs = {s: A.arm_addrs(m, s)[1] for s in SIDES}
        for s in SIDES:
            d.qpos[self.addr[s]] = cap["q"]["W0_liftoff"][s]
            _set_fingers(m, d, s, HOLD_FINGER)
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)
        A.disable_arm_servos(m)
        for s in SIDES:                                        # fingers stay open (the weld carries the load)
            a = m.actuator(f"{s}_arm_gripper_pos").id
            d.ctrl[a] = G.open_frac_to_ctrl(HOLD_FINGER)
        self.obj = m.body("obj").id
        self.gb = {s: m.body(f"{s}_arm_gripper_base").id for s in SIDES}
        self.site = {s: m.site(f"{s}_arm_ee_grasp").id for s in SIDES}
        self.weld_id = {s: m.equality(f"weld_{s}").id for s in SIDES}
        self.wc = WC.WeldContacts(m, "obj", self.weld_id, SIDES)      # 11-03: gripper<->object contacts off while that gripper is welded (nyu only)
        for s in SIDES:
            self.weld(s, True)
        self.truth0 = SC.object_truth(m, d)
        self.z0 = float(d.xpos[self.obj][2])                   # resting height on the counter
        self.spec_obj = OB.coffee_tray(world.n_cups) if self.task == "tray" else OB.boxed()
        cont = self.spec_obj.contents
        self.m_hat = 0.5 * (cont.m_lo + cont.m_hi) if m_hat is None else float(m_hat)
        self.m_nom = float(self.truth0["empty_mass"] + self.m_hat)
        self.tau_lim = C.tau_limits(TAU_PRESET)
        q0 = {s: d.qpos[self.addr[s]].copy() for s in SIDES}
        hold = {s: q0[s].tolist() for s in SIDES}
        self.ctl_kind = ctl
        if ctl in ("lever", "coopimp", "coopimp_safe"):
            P = cont.P
            c_c = P.mean(0)                                        # belief contents centre (obj frame)
            c_nom = (self.truth0["empty_mass"] * np.asarray(self.truth0["empty_com"]) + self.m_hat * c_c) / self.m_nom
            if ctl == "coopimp":                                   # 11-03 opt-in: object-level impedance (coop_imp.py)
                from .coop_imp import CoopImpCtl
                self.ctl = CoopImpCtl(m, hold, self.tau_lim, self.obj, self.gb, self.m_nom, c_nom, self.site)
            elif ctl == "coopimp_safe":                            # 11-03 opt-in: coopimp wrench as FL-MPC payload feedforward (rules in the QP)
                from .coop_imp import CoopImpSafeCtl
                self.ctl = CoopImpSafeCtl(m, hold, self.tau_lim, self.obj, self.gb, self.m_nom, c_nom, self.site)
            else:
                self.ctl = LeverCtl(m, hold, self.tau_lim, self.obj, self.gb, self.m_nom, c_nom, self.site)
        else:                                                      # "share": pan/handle_grasp.TwoCtl, half the load at the COM
            self.ctl = TwoCtl(m, hold, self.tau_lim, self.obj, self.gb, self.m_nom, share=share)
        self.ctl.on = True
        self.rules_mode = rules if rules is not None else ("full" if ctl == "coopimp_safe" else "base")
        if self.rules_mode not in ("base", "full"):
            raise ValueError(f"rules must be 'base' or 'full', got {rules!r}")
        if self.rules_mode == "full":                          # 11-03: table-top + joint position/speed rows inside the FL-MPC QP (rules.py)
            if ctl not in ("lever", "coopimp_safe"):
                raise ValueError("rules='full' needs an FL-MPC controller (ctl='lever' or 'coopimp_safe')")
            self.ctl.rules = RU.Rules(m)
            self.ctl.fl._rules = self.ctl.rules               # for fl_log (row / past-limit counters)
        self.NP, self.FDT = self.ctl.fl.NP, self.ctl.fl.DT
        self.q0 = q0
        self.F_cap = (E.PAYLOAD_KG - float(m.body_subtreemass[self.gb["left"]])) * OB.G
        geom_body = lambda g: m.body(m.geom_bodyid[g]).name or ""
        self.arm_g = {g for g in range(m.ngeom) if geom_body(g).startswith(("left_arm", "right_arm"))}
        self.grip_g = {g for g in self.arm_g if any(k in geom_body(g) for k in ("gripper", "finger"))}
        self.cups = [m.body(f"cup{i}").id for i in range(world.n_cups)] if self.task == "tray" else []
        xy, fills = world.layout() if self.task == "tray" else ([], [])
        self.freeboard = [world.cup_H - (SC.CUP_BOT + f / (SC.WATER * np.pi * SC.CUP_RIN ** 2)) for f in fills]
        self.carried = {self.obj} | set(self.cups) | ({m.body("contents").id} if self.task == "box" else set())
        self.fc = dict(FRIC_COMP, **fric_comp) if isinstance(fric_comp, dict) else (dict(FRIC_COMP) if fric_comp else None)
        if self.fc is not None and ctl in ("coopimp", "coopimp_safe") and not C.LEGACY_COOP_FC:
            # USER 11-03: under coopimp drop the wrapper's joint-level kp/ki terms (they fight the object-level impedance);
            # the Coulomb term f*tanh(v/v_s) stays.  BIMANUAL_LEGACY_COOP_FC=1 = the pre-11-03 wrapper (kp/ki on).
            self.fc = dict(self.fc, kp=0.0, ki=0.0)
        self.e_int = {s: np.zeros(7) for s in SIDES}
        self.coop = dict(COOP, **coop) if isinstance(coop, dict) else (dict(COOP) if coop else None)
        self.k = 0
        self.paths = {s: [] for s in SIDES}
        self.path_log = []                                     # every _add_path: side, t0, IK miss (last / max over waypoints), min joint margin [rad]
        self.stored = {}                                       # stored TCP poses (admission-time), per segment start
        self.reset_track()
        self._acc = {b: np.zeros(3) for b in self.cups}
        self._vprev = {b: np.zeros(3) for b in self.cups}

    def physics_flags(self):
        """the physics switches this episode ran with (carried by every record, 11-03 items 5, 6, 8, 9)."""
        return dict(fric_comp=self.fc is not None, weld_solref=self.weld_solref, legacy_carry_phys=C.LEGACY_CARRY_PHYS,
                    arm_arm_collide=not C.LEGACY_ARMCOL, coop_sensor=("legacy_constraint_wrench" if C.LEGACY_COOP_SENSOR else "joint_torque_estimate"),
                    coop_fric=("legacy" if C.LEGACY_COOP_FRIC else "fixed"), gripper=C.GRIPPER,
                    fc_kp_ki=(None if self.fc is None else (self.fc["kp"], self.fc["ki"])), rules=self.rules_mode,
                    coop_f_sq=(C.coop_f_sq() if self.ctl_kind in ("coopimp", "coopimp_safe") else None))

    # ---------------- weld / state ----------------
    def weld(self, s, on):
        i, m, d = self.weld_id[s], self.m, self.d
        if on:
            b1, b2 = m.eq_obj1id[i], m.eq_obj2id[i]
            R1 = d.xmat[b1].reshape(3, 3)
            qi, q = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(qi, d.xquat[b1])
            mujoco.mju_mulQuat(q, qi, d.xquat[b2])
            m.eq_data[i, 0:3] = 0.0
            m.eq_data[i, 3:6] = R1.T @ (d.xpos[b2] - d.xpos[b1])
            m.eq_data[i, 6:10] = q
            m.eq_data[i, 10] = 1.0
        d.eq_active[i] = 1 if on else 0
        self.wc.set(s, bool(on), d)

    _SPEC = mujoco.mjtState.mjSTATE_INTEGRATION

    def snapshot(self):
        st = np.zeros(mujoco.mj_stateSize(self.m, self._SPEC))
        mujoco.mj_getState(self.m, self.d, st, self._SPEC)
        return dict(st=st, weld={s: self.m.eq_data[i].copy() for s, i in self.weld_id.items()},
                    ctl=copy.deepcopy(self.ctl, {id(self.m): self.m}), k=self.k, paths={s: list(v) for s, v in self.paths.items()},
                    track=copy.deepcopy(self.track), acc=copy.deepcopy(self._acc), vprev=copy.deepcopy(self._vprev),
                    e_int=copy.deepcopy(self.e_int), wc=self.wc.state())

    def restore(self, sn):
        for s, i in self.weld_id.items():
            self.m.eq_data[i] = sn["weld"][s]
        mujoco.mj_setState(self.m, self.d, sn["st"], self._SPEC)
        self.wc.sync(self.d, sn.get("wc"))
        self.ctl = copy.deepcopy(sn["ctl"], {id(self.m): self.m})
        self.k, self.paths = sn["k"], {s: list(v) for s, v in sn["paths"].items()}
        self.track = copy.deepcopy(sn["track"])
        self._acc, self._vprev = copy.deepcopy(sn["acc"]), copy.deepcopy(sn["vprev"])
        self.e_int = copy.deepcopy(sn["e_int"])
        mujoco.mj_forward(self.m, self.d)

    def reset_track(self):
        self.track = dict(viol=None, tau_ratio=0.0, n_sat=0, Fz_max=0.0, twist_max=0.0, F_int_max=0.0, spill_ratio=0.0,
                          cup_tilt_max=0.0, obj_tilt_max=0.0, e_obj_mm=0.0, e_obj_deg=0.0, e_int_mm=0.0, e_int_deg=0.0,
                          path_ik_err_max=0.0, path_q_margin_min=PATH_NO_MARGIN, path_unreachable=0)
        if getattr(self, "ctl", None) is not None:          # 11-03 F2: FL-MPC fallback counters + true residuals (fl_log.py), same window as the track
            self.ctl.rec.reset()
            self.track.update(self.ctl.rec.scalars())

    # ---------------- motion ----------------
    def tcp(self, s):
        return self.d.site_xpos[self.site[s]].copy()

    def _add_path(self, s, t0, t1, delta=None, to=None):
        """continuation IK from the last commanded q; `to` = absolute TCP target (stored pose) instead of a delta."""
        pik = K.PoseIK(self.m, s)
        q = self.q0[s].copy() if not self.paths[s] else self.paths[s][-1][2][-1].copy()
        p0, R0 = pik.ik.pose(q)
        delta = np.asarray(to, float) - p0 if to is not None else np.asarray(delta, float)
        wp, err = [q.copy()], 0.0
        e_max, mg_min = 0.0, pik.margin(q)[0]
        for i in range(1, 61):
            a = i / 60
            q, err = pik.solve(p0 + delta * a * a * (3 - 2 * a), R0, q, q_post=wp[0], k_post=0.1, iters=60, tol=1e-9)
            wp.append(q.copy())
            e_max, mg_min = max(e_max, float(err)), min(mg_min, pik.margin(q)[0])
        self.paths[s].append((t0, t1, wp))
        # 11-03 visibility: pose-IK miss of EVERY waypoint (not only the last) and the smallest joint-limit distance; a miss > PATH_IK_TOL is a counted event
        # in track["path_unreachable"] (NOT in U / viol: the reference is parked at a limit, the pose is not what the plan says).
        tr = self.track
        tr["path_ik_err_max"] = max(tr.get("path_ik_err_max", 0.0), e_max)
        tr["path_q_margin_min"] = min(tr.get("path_q_margin_min", PATH_NO_MARGIN), mg_min)
        if e_max > PATH_IK_TOL:
            tr["path_unreachable"] = tr.get("path_unreachable", 0) + 1
        self.path_log.append(dict(side=s, t0=round(float(t0), 3), err_last=float(err), err_max=e_max, margin_min=mg_min))
        return float(err)

    def _cmd(self, s, t):
        q = self.q0[s]
        for t0, t1, wp in self.paths[s]:
            if t < t0:
                break
            a = np.clip((t - t0) / (t1 - t0), 0, 1) * (len(wp) - 1)
            i = min(int(a), len(wp) - 2)
            q = (1 - (a - i)) * wp[i] + (a - i) * wp[i + 1]
        return q

    def _coop_split(self, t):
        """claw errors -> (shared rigid object error u, internal parts e_int, joint shifts dq).  Encoders + FK only."""
        m, d = self.m, self.d
        mujoco.mj_forward(m, d)
        L = (self.coop or COOP)["L"]
        lam = (self.coop or COOP)["lam"]
        p = {s: d.site_xpos[self.site[s]].copy() for s in SIDES}
        o = 0.5 * (p["left"] + p["right"])
        W = np.diag([1, 1, 1, L, L, L])
        J, ex, A = {}, {}, {}
        for s in SIDES:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, self.site[s])
            J[s] = np.vstack([jp[:, self.dofs[s]], jr[:, self.dofs[s]]])
            ex[s] = J[s] @ (self._cmd(s, t) - d.qpos[self.addr[s]])
            r = p[s] - o
            sk = np.array([[0, -r[2], r[1]], [r[2], 0, -r[0]], [-r[1], r[0], 0]])
            A[s] = np.block([[np.eye(3), -sk], [np.zeros((3, 3)), np.eye(3)]])
        u = np.linalg.lstsq(np.vstack([W @ A[s] for s in SIDES]), np.concatenate([W @ ex[s] for s in SIDES]), rcond=None)[0]
        e_int = {s: ex[s] - A[s] @ u for s in SIDES}
        dq = {s: J[s].T @ np.linalg.solve(J[s] @ J[s].T + lam * np.eye(6), e_int[s]) for s in SIDES}
        return u, e_int, dq

    def _step(self):
        m, d, ctl = self.m, self.d, self.ctl
        t = self.k * DT
        welded = all(d.eq_active[self.weld_id[s]] for s in SIDES)
        dq = {s: np.zeros(7) for s in SIDES}
        if welded:
            u, e_int, dq_c = self._coop_split(t)
            tr = self.track
            tr["e_obj_mm"] = max(tr["e_obj_mm"], 1e3 * float(np.linalg.norm(u[:3])))
            tr["e_obj_deg"] = max(tr["e_obj_deg"], float(np.degrees(np.linalg.norm(u[3:]))))
            tr["e_int_mm"] = max(tr["e_int_mm"], 1e3 * max(float(np.linalg.norm(e_int[s][:3])) for s in SIDES))
            tr["e_int_deg"] = max(tr["e_int_deg"], float(np.degrees(max(np.linalg.norm(e_int[s][3:]) for s in SIDES))))
            if self.coop is not None:
                dq = {s: self.coop["gain"] * dq_c[s] for s in SIDES}
        ctl.qe_ref = np.stack([self._cmd("left", t + i * self.FDT) for i in range(self.NP + 1)]) - dq["left"]
        ctl.qi_ref = np.stack([self._cmd("right", t + i * self.FDT) for i in range(self.NP + 1)]) - dq["right"]
        safe = self.ctl_kind == "coopimp_safe"
        if self.ctl_kind in ("coopimp", "coopimp_safe") and ctl.on:
            ctl.ref_in = dict(qr={s: np.stack([self._cmd(s, t + j * 0.05) - dq[s] for j in (-1, 0, 1)]) for s in SIDES},
                              h_meas=np.concatenate([np.concatenate(self.claw_wrench(s)) for s in SIDES]),           # sim truth (weld constraint wrench)
                              tau_prev={"left": d.qfrc_applied[ctl.ld].copy(), "right": d.qfrc_applied[ctl.rd].copy()})   # joint torque of the last step (11-03 item 8)
            if safe:        # Coulomb term of the friction wrapper goes INTO the QP's bias (inside the torque box); no kp/ki, no wrapper afterwards
                ctl.ref_in["fric"] = {s_: (self.fc["f"] * np.tanh(((self._cmd(s_, t + DT) - self._cmd(s_, t)) / DT) / self.fc["v_s"])
                                           if self.fc is not None else np.zeros(7)) for s_ in SIDES}
            te, ti = ctl.tau(d)
            te0, ti0 = te, ti
            te, ti = np.clip(te, -self.tau_lim, self.tau_lim), np.clip(ti, -self.tau_lim, self.tau_lim)
            if safe:        # guard only: the QP output already respects the box; count steps where the clip changed anything (> 1e-6 Nm)
                self.track["ci_postclip_steps"] = self.track.get("ci_postclip_steps", 0) + int(
                    max(float(np.max(np.abs(te - te0))), float(np.max(np.abs(ti - ti0)))) > 1e-6)
            for k, v in ctl.log.items():
                self.track[k] = max(self.track.get(k, 0.0), float(v))
        else:
            te, ti = ctl.tau(d)
        if self.fc is not None and not (safe and ctl.on):
            fc, out = self.fc, []
            for s, tau in (("left", te), ("right", ti)):
                qc = self._cmd(s, t) - dq[s]
                # 11-03 item 9: qc carries the coop shift dq, _cmd(t + DT) did not -> the old difference was v_ref + dq / DT (rad / s of a rad shift
                # over one 5 ms step: ~0.2 rad/s for dq = 1 mrad, 10 x v_s = 0.02), saturating tanh = a spurious +-f sign torque.  Commanded velocity = d/dt _cmd.
                vc = ((self._cmd(s, t + DT) - self._cmd(s, t)) / DT) if not C.LEGACY_COOP_FRIC else ((self._cmd(s, t + DT) - qc) / DT)
                e = qc - d.qpos[self.addr[s]]
                self.e_int[s] = np.clip(self.e_int[s] + e * DT, -fc["i_max"] / max(fc["ki"], 1e-9), fc["i_max"] / max(fc["ki"], 1e-9))
                add = fc["f"] * np.tanh(vc / fc["v_s"]) + fc["ki"] * self.e_int[s] + fc["kp"] * e
                out.append(np.clip(tau + add, -self.tau_lim, self.tau_lim))
            te, ti = out
        d.qfrc_applied[:] = 0.0
        d.qfrc_applied[ctl.ld] = te
        d.qfrc_applied[ctl.rd] = ti
        self.wc.tick(d)
        PS.step(m, d)
        self.k += 1
        r = max(float(np.max(np.abs(te) / self.tau_lim)), float(np.max(np.abs(ti) / self.tau_lim)))
        self.track["tau_ratio"] = max(self.track["tau_ratio"], r)
        self.track["n_sat"] += int(r >= 1.0 - 1e-6)
        self.track.update(ctl.rec.scalars())               # 11-03 F2 (logging only)
        self._check(t, r)

    # ---------------- truth: claw wrench, U, R ----------------
    def claw_wrench(self, s):
        """wrench claw s applies to the object = its weld + its finger/gripper contacts on the object:
        (F world [N], T about the claw TCP, world [Nm])."""
        m, d = self.m, self.d
        i = self.weld_id[s]
        p = self.tcp(s)
        F, T = np.zeros(3), np.zeros(3)
        if d.eq_active[i]:
            f = np.zeros(d.nefc)
            sel = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == i)
            if sel.any():
                f[sel] = d.efc_force[sel]
                res = np.zeros(m.nv)
                mujoco.mj_mulJacTVec(m, d, res, f)
                da = m.jnt_dofadr[m.body_jntadr[self.obj]]
                F = res[da:da + 3].copy()
                T = d.xmat[self.obj].reshape(3, 3) @ res[da + 3:da + 6] + np.cross(d.xpos[self.obj] - p, F)   # free joint: body-local
        side_g = {g for g in self.grip_g if (m.body(m.geom_bodyid[g]).name or "").startswith(s)}
        cf = np.zeros(6)
        for k, c in enumerate(d.contact[:d.ncon]):
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in side_g and m.geom_bodyid[g2] == self.obj:
                sg = 1.0                                     # force on geom2 (the object) along the frame
            elif g2 in side_g and m.geom_bodyid[g1] == self.obj:
                sg = -1.0
            else:
                continue
            mujoco.mj_contactForce(m, d, k, cf)
            Fc = sg * (c.frame.reshape(3, 3).T @ cf[:3])
            F = F + Fc
            T = T + np.cross(c.pos - p, Fc)
        return F, T

    def _viol(self, kind, t, **kw):
        if self.track["viol"] is None:
            self.track["viol"] = dict(kind=kind, t=round(float(t), 3), **{k: (round(float(v), 4) if isinstance(v, (int, float, np.number)) else v) for k, v in kw.items()})

    def _check(self, t, r):
        m, d, tr = self.m, self.d, self.track
        if r >= 1.0 - 1e-6:
            self._viol("tau_sat", t, ratio=r)
        for s in SIDES:
            F, T = self.claw_wrench(s)
            tr["Fz_max"] = max(tr["Fz_max"], abs(float(F[2])))
            tw = float(np.linalg.norm(T))
            tr["twist_max"] = max(tr["twist_max"], tw)
            tr["F_int_max"] = max(tr.get("F_int_max", 0.0), abs(float(F[1])))          # along the grasp line (internal)
            if self.strict_claw and abs(F[2]) > self.F_cap:
                self._viol("claw_payload", t, side=s, Fz=float(F[2]))
            if self.strict_claw and self.T_cap and tw > self.T_cap:
                self._viol("claw_twist", t, side=s, twist=tw)
        tilt = _tilt_deg(d.xmat[self.obj].reshape(3, 3))
        tr["obj_tilt_max"] = max(tr["obj_tilt_max"], tilt)
        if tilt > U_TILT_OBJ:
            self._viol("obj_tilt", t, deg=tilt)
        if d.xpos[self.obj][2] < self.z0 - 0.01:
            self._viol("obj_drop", t)
        al = DT / (SPILL_TAU + DT)
        for j, b in enumerate(self.cups):
            v = np.zeros(6)
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            acc = (v[3:] - self._vprev[b]) / DT
            self._vprev[b] = v[3:].copy()
            self._acc[b] = (1 - al) * self._acc[b] + al * acc
            Rc = d.xmat[b].reshape(3, 3)
            ct = _tilt_deg(Rc)
            tr["cup_tilt_max"] = max(tr["cup_tilt_max"], ct)
            if ct > U_TILT_CUP:
                self._viol("cup_tipped", t, cup=j, deg=ct)
            geff = self._acc[b] - np.array([0, 0, -OB.G])                      # specific force the liquid feels (world)
            gl = Rc.T @ geff                                                   # in the mug frame
            tanphi = float(np.hypot(gl[0], gl[1]) / max(gl[2], 1e-6))
            ratio = 0.5 * self.world.cup_D * tanphi / max(self.freeboard[j], 1e-6)
            if self.k > 10:                                                    # skip the filter warm-up
                tr["spill_ratio"] = max(tr["spill_ratio"], ratio)
                if ratio > 1.0:
                    self._viol("spill", t, cup=j, ratio=ratio)
            rel = d.xmat[self.obj].reshape(3, 3).T @ (d.xpos[b] - d.site_xpos[m.site("obj_ref").id])
            if abs(rel[0]) > self.world.tray_lx / 2 or abs(rel[1]) > self.world.tray_ly / 2 or rel[2] < -0.005:
                self._viol("cup_off_tray", t, cup=j)
        for c in d.contact[:d.ncon]:
            if c.dist > -1e-3:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            a1, a2 = g1 in self.arm_g, g2 in self.arm_g
            if a1 and a2:                                                      # 11-03 item 5: left<->right arm contact is a collision (it exists now)
                if (m.body(m.geom_bodyid[g1]).name or "")[:4] != (m.body(m.geom_bodyid[g2]).name or "")[:4]:
                    self._viol("collision", t, arm=m.geom(g1).name, other=m.geom(g2).name)
                continue
            if a1 == a2:
                continue
            ga, go = (g1, g2) if a1 else (g2, g1)
            if ga in self.grip_g and m.geom_bodyid[go] == self.obj:
                continue                                                       # claw on the object's tab (the grasp)
            self._viol("collision", t, arm=m.geom(ga).name, other=m.geom(go).name or m.body(m.geom_bodyid[go]).name)

    def in_R(self):
        m, d = self.m, self.d
        why = []
        v = np.zeros(6)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.obj, v, 0)
        if np.linalg.norm(v[3:]) > R_SPEED:
            why.append("moving")
        if _tilt_deg(d.xmat[self.obj].reshape(3, 3)) > R_TILT_OBJ:
            why.append("obj_tilt")
        sup = {m.geom("counter").id} | ({m.geom("shelf").id} if self.task == "box" and getattr(self.world, "shelf", False) else set())
        if not any((int(c.geom1) in sup and m.geom_bodyid[c.geom2] == self.obj) or
                   (int(c.geom2) in sup and m.geom_bodyid[c.geom1] == self.obj) for c in d.contact[:d.ncon]):
            why.append("not_supported")
        for b in self.cups:
            if _tilt_deg(d.xmat[b].reshape(3, 3)) > R_TILT_CUP:
                why.append("cup_tilt")
        if self.track["viol"]:
            why.append("U:" + self.track["viol"]["kind"])
        return not why, why

    # ---------------- phases ----------------
    def run_phases(self, phases, branch=None):
        """phases: ("move", delta, T) both claws | ("to", key, T) both claws back to stored pose `key` | ("hold", T) | ("stop", T)
        | ("weld", on) | ("retreat", r, T).  branch(ep, t_rel) -> called every 0.25 s inside a move (snapshot hooks)."""
        for ph in phases:
            t = self.k * DT
            if ph[0] == "weld":
                for s in SIDES:
                    self.weld(s, ph[1])
                self.ctl.on = bool(ph[1])                      # payload model off once the object is let go
                continue
            if ph[0] == "move":
                dur = ph[2]
                for s in SIDES:
                    self._add_path(s, t, t + dur, np.asarray(ph[1], float))
            elif ph[0] == "to":
                tgt = self.stored[ph[1]]
                dist = max(float(np.linalg.norm(tgt[s] - self.tcp(s))) for s in SIDES)
                dur = move_time(dist, A_BACKUP) if ph[2] is None else ph[2]
                for s in SIDES:
                    self._trim(s, t)
                    self._add_path(s, t, t + dur, to=tgt[s])
            elif ph[0] == "retreat":
                dur = ph[2]
                for s in SIDES:
                    self._add_path(s, t, t + dur, np.array([0, ph[1] if s == "left" else -ph[1], 0]))
            elif ph[0] == "stop":
                dur = ph[1]
                for s in SIDES:                           # decelerate the COMMANDED reference (11-02: starting from the measured
                    q = self._cmd(s, t)                   # q baked the wrist sag into the reference; repeated -> 10 deg box tilt)
                    v = (self._cmd(s, t + 1e-3) - q) / 1e-3 * dur
                    self._trim(s, t)
                    q1 = q + 0.5 * v
                    wp = [(2 * a ** 3 - 3 * a ** 2 + 1) * q + (a ** 3 - 2 * a ** 2 + a) * v + (-2 * a ** 3 + 3 * a ** 2) * q1
                          for a in np.linspace(0, 1, 31)]
                    self.paths[s].append((t, t + dur, wp))
            else:
                dur = ph[1]
            n = int(round(dur / DT))
            for i in range(n):
                if branch is not None and ph[0] == "move" and i > 0 and i % int(0.25 / DT) == 0:
                    branch(self, i * DT)
                self._step()

    def _trim(self, s, t):
        """cut the planned path at time t (a stop / backup replaces the rest of the plan): keep finished moves, freeze the
        command at its value at t."""
        q = self._cmd(s, t)
        self.paths[s] = [p for p in self.paths[s] if p[1] <= t] + [(t - 1e-9, t, [q.copy(), q.copy()])]

    def segment_phases(self, seg):
        plan = dict(PLANS[self.task])
        if seg == "release":
            return [("hold", 0.2), ("weld", False), ("retreat", RETREAT, 0.8), ("hold", 0.5)]
        dl = np.asarray(plan[seg], float)
        self.last_accel = commanded_accel(float(np.linalg.norm(dl)), self.a_plan)           # 11-03 item 10: recorded next to a_plan
        return [("move", tuple(dl), move_time(float(np.linalg.norm(dl)), self.a_plan)), ("hold", 0.3)]

    def backup_phases(self, seg):
        back = {"lift": ["lift"], "carry": ["carry", "lift"], "place": ["place", "carry", "lift"]}.get(seg)
        if back is None:
            return None
        # let go in place (11-02): back on the counter, release the weld and retreat, so the object settles flat on its own
        # (held welded, wrist sag of up to ~4 deg under load left it resting on one edge: smoke box 2 kg, 14/16 backups not in R)
        return ([("stop", 0.3)] + [("to", f"start_{s}", None) for s in back]
                + [("hold", 0.3), ("weld", False), ("retreat", RETREAT, 0.8), ("hold", 0.6)])

    def store(self, seg):
        self.stored[f"start_{seg}"] = {s: self.tcp(s) for s in SIDES}

    def set_belief(self, m_hat, a_plan=None):
        """11-02 (probe.py): new nominal contents mass after a probe narrowed the belief; the payload feedforward follows it."""
        self.m_hat = float(m_hat)
        self.m_nom = float(self.truth0["empty_mass"] + self.m_hat)
        if a_plan is not None:
            self.a_plan = float(a_plan)
        self.ctl.mass = self.m_nom
        if isinstance(self.ctl, LeverCtl):
            c_c = self.spec_obj.contents.P.mean(0)
            self.ctl.c_rel_nom = (self.truth0["empty_mass"] * np.asarray(self.truth0["empty_com"]) + self.m_hat * c_c) / self.m_nom


# ---------------------------------------------------------------------------------------------------------------------------
def world_trial(world, tag, a_plan=E.A_NOM, branch_backups=True, T_cap=None, ctl="lever", strict_claw=False, m_hat=None,
                fric_comp=PHYS_DEFAULT, coop=None, weld_solref=PHYS_DEFAULT, rules=None):
    """m_hat: the robot's nominal contents mass (belief cell centre; None = the prior's centre)."""
    """nominal run of all segments in one world + the stored backup from branch points of each segment.  -> record for admit."""
    t0 = time.time()
    ep = CarryEpisode(world, a_plan=a_plan, T_cap=T_cap, ctl=ctl, strict_claw=strict_claw, m_hat=m_hat, fric_comp=fric_comp,
                      coop=coop, weld_solref=weld_solref, rules=rules)
    ep.run_phases([("hold", 0.4)])                            # load transfer, object still on the counter
    rec = dict(tag=tag, task=ep.task, theta=_theta(world), truth=_jsonable(ep.truth0), a_plan=a_plan, m_nom=ep.m_nom, ctl=ctl,
               strict_claw=strict_claw, fric_comp=ep.fc, coop=ep.coop, weld_solref=ep.weld_solref,
               physics=ep.physics_flags(), segments=[])
    return run_segments(ep, rec, branch_backups, t0)


def run_segments(ep, rec, branch_backups=True, t0=None):
    """the segment loop of world_trial on an existing episode (11-02: probe.py runs it after the probe, same episode)."""
    t0 = time.time() if t0 is None else t0
    hold_viol = ep.track["viol"]
    for seg in SEGS:
        ep.store(seg)
        ep.reset_track()
        sn0 = ep.snapshot()
        branches = [(0.0, sn0)]
        bfun = (lambda e, tr: branches.append((tr, e.snapshot()))) if branch_backups and seg != "release" else None
        ep.run_phases(ep.segment_phases(seg), branch=bfun)
        nominal = dict(viol=ep.track["viol"], **{k: round(float(v), 4) for k, v in ep.track.items() if k not in ("viol",)})
        fl_nom = ep.ctl.rec.summary()          # 11-03 F2: nominal run of this segment (incl. the per-joint torque ratios); backups: nominal-run keys only
        okR, whyR = ep.in_R()
        nominal["end_in_R"], nominal["R_fail"] = okR, whyR
        fam = dict(n_bad=0, first_bad=None, n=0)
        bk = ep.backup_phases(seg)
        if bk is not None and branch_backups:
            end = ep.snapshot()
            for tr, sn in branches[1:] + [(None, end)]:
                ep.restore(sn)
                ep.reset_track()
                ep.run_phases(bk)
                ok, why = ep.in_R()
                fam["n"] += 1
                if not ok:
                    fam["n_bad"] += 1
                    if fam["first_bad"] is None:
                        fam["first_bad"] = dict(t=tr, viol=ep.track["viol"], R_fail=why)
            ep.restore(end)
        rec["segments"].append(dict(seg=seg, nominal=nominal, family={0.0: fam}, fl=fl_nom,
                                    accel=(dict(ep.last_accel) if seg != "release" and getattr(ep, "last_accel", None) else None)))
        if nominal["viol"]:
            break
    rec["pre_hold_viol"] = hold_viol
    rec["truth_end"] = _jsonable(SC.object_truth(ep.m, ep.d))
    rec["end_in_R"] = ep.in_R()
    rec["wall_s"] = round(time.time() - t0, 1)
    return rec


def _theta(world):
    if isinstance(world, SC.TrayWorld):
        xy, f = world.layout()
        return dict(n_cups=world.n_cups, cup_xy=[list(map(float, q)) for q in xy], fills_kg=[float(x) for x in f], cup_mu=world.cup_mu)
    return dict(contents_kg=world.contents_kg, contents_xy=list(world.contents_xy), contents_mu=world.contents_mu)


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (np.ndarray, list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return float(x)
    return x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=("tray", "box"), default="tray")
    ap.add_argument("--n-cups", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--contents-kg", type=float, default=1.0)
    ap.add_argument("--no-branch", action="store_true")
    ap.add_argument("--ctl", choices=("lever", "share"), default="lever")
    a = ap.parse_args(argv)
    if a.task == "tray":
        w = SC.sample_tray_world(a.n_cups, np.random.default_rng(a.seed))
    else:
        w = SC.BoxWorld(contents_kg=a.contents_kg)
    r = world_trial(w, f"{a.task}_s{a.seed}", branch_backups=not a.no_branch, ctl=a.ctl)
    print(json.dumps(r, indent=1, default=float))


if __name__ == "__main__":
    main()
