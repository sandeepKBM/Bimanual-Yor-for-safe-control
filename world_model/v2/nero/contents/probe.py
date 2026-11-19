"""nero/contents/probe.py -- soft-to-strong lift probe ("probe ladder"), user 11-02/03.  SIMULATION ONLY.

USER 11-02: "given any bound it should start conservatively ... poke softly and see and then learn and keep strengthening the bound".

Setting: object grasped (rigid weld, ASSUMED 11-01) and still resting on the counter; the arm reference stays at the rest pose (no
commanded motion).  Rung k ramps the payload feedforward (LeverCtl lever split, total mass m_ff,k) over T_RAMP, holds T_HOLD, and
reads the vertical load the two claws carry from joint torques (robot-observable: encoders, commanded torque, the arm's own model):

    L = sum_s  e_z . (J_s^T)^+ (tau_cmd,s - bias_s(q)) / g        J_s = 6x7 TCP Jacobian, bias_s = own-arm gravity + Coriolis (MODEL)
    (+ uniform joint-torque noise +-TAU_NOISE per joint and sample, ASSUMED; its worst case is propagated into the bound, MODEL)

What a rung teaches (belief Theta_t = interval of the total carried mass M; the object's empty mass is known):
  still on the counter: the arms carry L and the counter the rest (>= 0)      =>  M >= L - EPS_LO      (lower bound rises)
  hanging free:  L stops growing while m_ff grows (dL < FREE_RATIO dm)         =>  M in [L - EPS_FREE, L + EPS_FREE]
  EPS_LO, EPS_FREE: MEASURED by `calibrate` (max over n worlds of the read residual) + the propagated noise bound (MODEL).

Rung admission on Theta_t (checked BEFORE the rung runs; the soft-first part):
  torque     static u at the rest pose with the full feedforward m_ff,k <= U_ADMIT          proof under MODEL (quasi-static; in steady
             state the arms carry min(M, ~m_ff) so the feedforward bounds the load)
  overshoot  m_ff,k - M_lo(Theta_t) <= O_MAX                                                 ASSUMED; bounds how far and how fast a
             lighter-than-pushed object rises (rise / tilt / spill measured per rung in `calibrate`, sampled)
The first rung pushes no more than the LIGHTEST object in Theta_0 (overshoot <= 0 by construction).
Stop: free -> interval; M_lo > the heaviest carry admissible (verdict decided: refuse, never lifted); next rung not admissible ->
refuse; the static verdict is already the same for all of Theta_0 -> no probe at all.
After the probe: refuse = ramp the push down, let go in place (weld off, retreat); admit = set the belief, ramp to the new nominal
feedforward and run the carry segments (carry.run_segments, stored backups) in the SAME episode.

    python -m world_model.v2.nero.contents.probe calibrate --tasks box,tray6,tray4 --n 30 --procs 4 --out DIR
    python -m world_model.v2.nero.contents.probe e2e --tasks box,tray6 --n 30 --procs 4 --calib DIR/CALIB.json --out DIR2
    python -m world_model.v2.nero.contents.probe onepoke --tasks box,tray6 --n 10 --procs 4 --out DIR3     (ablation)
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import re
import time

import mujoco
import numpy as np

from world_model.v2.stack import tasks as TK
from world_model.v2.stack import u_groups as _UG

from . import carry as CR
from . import carry_cert as CC
from . import envelope as E
from . import objects as OB
from . import scenes as SC
from . import thresholds as TH

G = OB.G
DM = 0.25            # ASSUMED: push step between rungs [kg of feedforward]
O_MAX = 1.0          # ASSUMED: largest push above the current lower bound [kg]
T_RAMP = 0.3         # design: ramp time per rung [s]
T_HOLD = 0.4         # design: hold before reading [s]
N_READ = 40          # design: readings averaged over the last N_READ control steps of the hold (0.2 s)
FREE_RATIO = 0.2     # ASSUMED: "free" when the read grows by less than FREE_RATIO x the push step (one rung; calibrated)
TAU_NOISE = 0.05     # ASSUMED: joint-torque reading noise, uniform +-TAU_NOISE per joint [Nm] (real NERO value unknown)
PROBE_METHOD = os.environ.get("BIMANUAL_PROBE", "claw_guard")   # default "claw_guard" (USER 11-13); "poslift" | "poslift_ff"; make_probe(method=) overrides
METHODS = ("poslift", "poslift_ff", "claw_guard")               # selectable probe methods (env BIMANUAL_PROBE / CLI); claw_guard is the default since 11-13 (USER).  The push ladder (class Probe) is no longer selectable by env / CLI
_PROGRAMMATIC = ("ladder",)                                     # make_probe(method="ladder") only: probe_al (AL = push ladder) and the probe_liftoff variants subclass / patch Probe
if PROBE_METHOD not in METHODS:
    raise ValueError(f"BIMANUAL_PROBE={PROBE_METHOD!r} not in {METHODS}")
# poslift constants (option A, "lift by position, read while holding"; env PL_* kept from the 11-09 prototype runs)
DZ = float(os.environ.get("PL_DZ", 0.001))               # ASSUMED, USER-approved 11-09 (poslift): height step between rungs [m]
Z_MAX = float(os.environ.get("PL_Z_MAX", 0.020))         # ASSUMED, USER-approved 11-09 (poslift): hard cap on the commanded claw rise [m]
FREE_TOL = float(os.environ.get("PL_FREE_TOL", 0.05))    # ASSUMED, USER-approved 11-09 (poslift): "free" when N_FREE_CONFIRM consecutive |dL| < FREE_TOL [kg]
T_MOVE = 0.3                                             # design (= T_RAMP): time for one height step / feed-forward ramp [s]
T_HOLD_POSLIFT = float(os.environ.get("PL_T_HOLD", 1.2))  # ASSUMED, USER-approved 11-09 (poslift): hold per height rung [s]
N_FREE_CONFIRM = 2                                       # ASSUMED, USER-approved 11-09 (poslift): consecutive flat dL needed
DZ_MIN = float(os.environ.get("PL_DZ_MIN", 0.0001))      # ASSUMED, USER-approved 11-09: smallest adaptive height step [m]; below it the probe stops (rung_not_admissible)
POSLIFT_CALIB = pathlib.Path(__file__).resolve().parents[1] / "results" / "probe_calib_poslift_adaptive_2026-11-09" / "CALIB.json"   # MEASURED n=20/task, tracker flmpc_basic, adaptive step K_STEP (README next to it; the 11-09 fixed-step FL-MPC file is probe_calib_poslift_2026-11-09)
LADDER_CALIB = "/data/local/ss5772/contents/probe_calib_2026-11-03/CALIB.json"                                              # MEASURED 11-03 (ladder only)
# Stage 0b (11-12): two opt-in methods next to poslift (bug B6: the adaptive height step stops at headroom < K_STEP x DZ_MIN = 0.71 kg, so the
# probe cannot resolve loads above ~2.4 kg box / ~2.6 kg tray6).  New numbers below need user approval before any claim rests on them.
DM_FINE = float(os.environ.get("PL_DM_FINE", 0.1))                 # ASSUMED (proposal, needs approval): poslift_ff finish-phase feed-forward step [kg]
FF_FREE_RATIO = float(os.environ.get("PL_FF_FREE_RATIO", FREE_RATIO))   # ASSUMED (needs approval): finish-phase free test dL < FF_FREE_RATIO x DM_FINE (the 11-03 ladder rule, FREE_RATIO reused)
FF_T_HOLD = float(os.environ.get("PL_FF_T_HOLD", T_HOLD_POSLIFT))   # ASSUMED (needs approval): hold per finish rung [s] = the poslift height-rung hold (USER-approved 11-09 for height rungs), reused
_RESULTS = pathlib.Path(__file__).resolve().parents[1] / "results"
JUMP_FILE = _RESULTS / "probe_jump_2026-11-12" / "JUMP.json"       # MEASURED per-task max per-rung load jump of the height ladder (kstep_calib_2026-11-09, n=20/task); not a bound
NEW_CALIB = {"poslift_ff": _RESULTS / "probe_calib_poslift_ff_2026-11-12" / "CALIB.json",          # poslift_ff: not copied into the repo (run copy under bimanual_data/contents/probe_fix_2026-11-12/)
             "claw_guard": _RESULTS / "probe_calib_claw_guard_2026-11-12" / "CALIB.json"}   # repo copy (README next to it), MEASURED n=20/task


# ------------------------------------------------------------------------------------------------- specs and static verdicts
def n_cups(task):
    """number of mugs of a tray task key: "tray6" -> 6; a table-behind key ("tray6_behind") parses the same way (11-12 fix, B11: int('6_behind') raised).
    Keys that parsed before give the same value."""
    if not task.startswith("tray"):
        return 0
    m = re.match(r"tray(\d+)", task)
    return int(m.group(1)) if m else int(task[4:])


def empty_mass(task):
    if task == "pot":
        return SC.pot_empty_props()[0]                      # MODEL: pot + lid (geom volume x PLA density)
    return OB.TRAY["mass"] if task.startswith("tray") else OB.BOX["mass"]


def prior_contents(task):
    """contents mass interval of the full prior (tray: mugs + fill; box: the block)."""
    if task == "pot":
        return SC.POT_CONTENTS_KG
    if task.startswith("tray"):
        n = n_cups(task)
        return n * OB.CUP["empty"], n * (OB.CUP["empty"] + OB.CUP["fill_max_kg"])
    return OB.BOX_CONTENTS_KG


def interval_spec(task, c_lo, c_hi, com=None, world=None):
    """ObjectSpec for contents mass in [c_lo, c_hi].  Tray: the total fill is known only as a total, so one mug may hold all of
    it: per-mug fill <= min(fill_max, total fill hi) sets the spill limit (MODEL, conservative)."""
    if com is not None and task != "box":
        raise NotImplementedError(f"a point COM (oracle_theta) is built for the box only, not {task!r}")
    if task == "pot":
        return SC.pot_spec(c_lo, c_hi)                       # solid puck: no spill row; the loose-lid slide limit (mu_lid_min g) is the open-content row
    if not task.startswith("tray"):
        return OB.boxed(c_lo, c_hi, com=com, outer=getattr(world, "box_outer", None))      # world: the run world's drawn crate (contents prism / empty COM)
    n = n_cups(task)
    sp = OB.coffee_tray(n)
    sp.contents = OB.Contents(c_lo, c_hi, sp.contents.P, dict(sp.contents.labels, m=f"probe interval {c_lo:.3f}..{c_hi:.3f} kg"))
    f_hi = float(np.clip(c_hi - n * OB.CUP["empty"], 0.0, OB.CUP["fill_max_kg"]))
    h_fill = SC.CUP_BOT + f_hi / (SC.WATER * np.pi * SC.CUP_RIN ** 2)
    sp.open_limits = [OB.OpenLimit("cup_spill", OB.cup_spill_limit(OB.CUP["D"], OB.CUP["H"], h_fill),
                                   "a_max = g 2 (H - h) / D, one mug holding the whole fill hi")]
    return sp


def static_interval(task, c_lo, c_hi, world=None, planner=None, wp=None):
    """planner (11-05): None -> carry.resolve_planner() (env BIMANUAL_PLANNER, default "scripted").  wp: a closed_chain.WaypointPlan (the proposer's plan)."""
    world = world if world is not None else CC.world_default(task)
    return CC.static_cell(task, (round(c_lo, 4), round(c_hi, 4)), world, spec=interval_spec(task, c_lo, c_hi, world=world), planner=planner, wp=wp)


def c_admit_max(task, tol=0.01, planner=None, wp=None):
    """largest contents mass c with the point interval [c, c] admitted, on the plan of `planner` (None = the process default, CR.resolve_planner) or on wp."""
    return _c_admit_max(task, tol, "closed_chain" if wp is not None else CR.resolve_planner(planner), wp)


@functools.lru_cache(maxsize=None)
def _c_admit_max(task, tol, planner, wp=None):
    """largest contents mass c with the point interval [c, c] admitted (bisection; admission is monotone in the mass: MODEL --
    torque rows grow with m, the spill limit tightens with the fill).  None if even the lightest prior member is refused."""
    lo, hi = prior_contents(task)
    if static_interval(task, lo, lo, planner=planner, wp=wp)["action"].startswith("refuse"):
        return None
    if not static_interval(task, hi, hi, planner=planner, wp=wp)["action"].startswith("refuse"):
        return hi
    a, b = lo, hi
    while b - a > tol:
        c = 0.5 * (a + b)
        a, b = (c, b) if not static_interval(task, c, c, planner=planner, wp=wp)["action"].startswith("refuse") else (a, c)
    return a


_REST = {}


def _rest_envelope(task, planner=None, world=None):
    """rest-pose envelope of the planner's start configuration.  world (11-19): the RUN world (claws on its tabs = nyu_hold.hold_pose via the planner's inputs, object
    reference point of that world); None = the nominal scene.  Only the closed_chain planner follows the world (the scripted legacy path stays on the captured pose)."""
    from . import nyu_hold as NH
    pl = CR.resolve_planner(planner)
    k = (task, pl, None if (world is None or pl != "closed_chain") else NH._hold_key(world))
    if k not in _REST:
        _REST[k] = _rest_envelope_c(task, pl, world if pl == "closed_chain" else None)
    return _REST[k]


def _rest_envelope_c(task, planner, world=None):
    spec = interval_spec(task, *prior_contents(task))
    caps = json.loads(TH.CAP_FILE.read_text())
    if planner == "closed_chain":
        from ...stack.planners import closed_chain as CCP
        offs, _, cap = CCP.static_path(task, None, world)
    else:
        offs, _ = CC.path_offsets(task)
        cap = TH.capture_path(caps["pot"], offs)
    o_ref = TH.o_ref(world)
    return E.Envelope(spec, E.Arms(cap), o_ref, np.eye(3), ["left", "right"], grip_mass=caps["pot"]["gripper_mass"], offsets=offs)


def rung_torque_u(task, m_ff, world=None):
    """static u at the rest pose (path key p0, a = 0) carrying total mass m_ff (contents m_ff - empty, COM anywhere in P): MODEL."""
    c = max(m_ff - empty_mass(task), 0.0)
    P = interval_spec(task, *prior_contents(task)).contents.P
    u, _ = _rest_envelope(task, world=world).worst(OB.Contents(c, c, P), ["p0"], 0.0, "two_lever")
    return float(u)


# ------------------------------------------------------------------------------------------------- the probe itself
def true_mass(ep):
    return float(sum(ep.m.body_subtreemass[b] for b in ep.carried))


class Probe:
    def __init__(self, ep, task, eps_lo, eps_free, rng=None, noise=True):
        self.ep, self.task = ep, task
        self.eps_lo, self.eps_free = float(eps_lo), float(eps_free)
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.noise = bool(noise)
        self.tau = {s: np.zeros(7) for s in CR.SIDES}

        def keep_tau(_ep):                            # keep the torque the robot commanded this step (robot-side quantity)
            self.tau["left"] = ep.d.qfrc_applied[ep.ctl.ld].copy()
            self.tau["right"] = ep.d.qfrc_applied[ep.ctl.rd].copy()
        ep.step_hooks.append(keep_tau)
        m_e = empty_mass(task)
        c_lo, c_hi = prior_contents(task)
        self.M_lo, self.M_hi = m_e + c_lo, m_e + c_hi           # Theta_0 (total carried mass)
        self.m_ff = 0.0
        self.rungs = []
        self.track = dict(rise_mm=0.0, tilt=0.0, spill=0.0)

    def read(self):
        """(load L [kg], worst-case noise bound [kg]) from one control step."""
        ep, m, d = self.ep, self.ep.m, self.ep.d
        mujoco.mj_forward(m, d)
        Fz, nb = 0.0, 0.0
        for s in CR.SIDES:
            dofs = ep.dofs[s]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, ep.site[s])
            Pz = np.linalg.pinv(np.vstack([jp[:, dofs], jr[:, dofs]]).T)[2]       # row of (J^T)^+ giving F_z
            r = self.tau[s] - d.qfrc_bias[dofs]
            if self.noise:
                r = r + self.rng.uniform(-TAU_NOISE, TAU_NOISE, size=7)
            Fz += float(Pz @ r)
            nb += float(np.abs(Pz).sum()) * TAU_NOISE
        return Fz / G, nb / G

    def _ramp(self, m_to, T):
        ep = self.ep
        n = max(int(round(T / CR.DT)), 1)
        m0 = self.m_ff
        for i in range(n):
            ep.ctl.mass = m0 + (m_to - m0) * (i + 1) / n
            ep._step()
        self.m_ff = m_to

    def rung(self, m_to, t_hold=None):
        """t_hold None -> T_HOLD (the 11-03 ladder; unchanged).  poslift_ff's finish phase passes FF_T_HOLD."""
        ep = self.ep
        self._ramp(m_to, T_RAMP)
        reads, nbs = [], []
        nh = int(round((T_HOLD if t_hold is None else t_hold) / CR.DT))
        for i in range(nh):
            ep._step()
            if i >= nh - N_READ:
                L, nb = self.read()
                reads.append(L)
                nbs.append(nb)
        rise = 1e3 * (float(ep.d.xpos[ep.obj][2]) - ep.z0)
        self.track["rise_mm"] = max(self.track["rise_mm"], rise)
        self.track["tilt"] = max(self.track["tilt"], ep.track["obj_tilt_max"])
        self.track["spill"] = max(self.track["spill"], ep.track["spill_ratio"])
        return float(np.mean(reads)), float(max(nbs)), rise

    def admissible(self, m_to):
        u = rung_torque_u(self.task, m_to, getattr(getattr(self, "ep", None), "world", None))
        over = m_to - self.M_lo
        return (u <= E.U_ADMIT and over <= O_MAX + 1e-9), dict(u=round(u, 4), overshoot=round(over, 4))

    def run(self, M_stop=None):
        """ladder until free / decided / not admissible.  M_stop: refuse once M_lo > M_stop (the heaviest admissible total)."""
        self.ep.ctl.mass = 0.0
        self.ep.run_phases([("hold", 0.3)])
        prev_L, n_flat, why = None, 0, None
        m_next = self.M_lo                                   # first rung: the lightest object Theta_0 allows (overshoot 0)
        while True:
            if M_stop is not None and self.M_lo > M_stop:
                why = "decided_refuse"
                break
            ok, rows = self.admissible(m_next)
            if not ok:
                why = "rung_not_admissible"
                break
            L, nb, rise = self.rung(m_next)
            dm = m_next - (self.rungs[-1]["m_ff"] if self.rungs else 0.0)
            flat = prev_L is not None and (L - prev_L) < FREE_RATIO * dm
            n_flat = n_flat + 1 if flat else 0
            rec = dict(m_ff=round(m_next, 4), L=round(L, 4), noise_bound=round(nb, 4), rise_mm=round(rise, 3), **rows,
                       free=n_flat >= 1)
            if n_flat >= 1:                                 # hanging free: interval read
                e = self.eps_free + nb
                self.M_lo, self.M_hi = max(self.M_lo, L - e), min(self.M_hi, L + e)
            else:                                           # still (partly) on the counter: lower bound only
                self.M_lo = max(self.M_lo, L - self.eps_lo - nb)
            rec["Theta"] = [round(self.M_lo, 4), round(self.M_hi, 4)]
            self.rungs.append(rec)
            prev_L = L
            if n_flat >= 1:
                why = "free"
                break
            m_next = self.m_ff + DM
        return why


class PosLiftProbe(Probe):
    """option A (user 11-09), "lift by position, read while holding": feed-forward ctl.mass = M_lo (current lower bound of the total mass); height ladder:
    both claws move up DZ over T_MOVE, hold T_HOLD_POSLIFT, read L (mean of the last N_READ steps).  Each rung: M_lo <- max(M_lo, L - EPS_LO - nb).  FREE when
    the last N_FREE_CONFIRM dL = L_k - L_{k-1} are all < FREE_TOL in magnitude -> interval [L - EPS_FREE - nb, L + EPS_FREE + nb].  Stops: decided_refuse
    (M_lo > M_stop), rung_not_admissible (static u(M_lo) > U_ADMIT, or the adaptive step below DZ_MIN), z_cap
    (Z_MAX; lower bound only), load_guard (u(read L) > U_ADMIT after a hold; lower bound only).  Decisions use robot-side numbers only; the sim-truth claw
    payload row (ep.track["viol"]) is recorded as `sim_abort` (diagnostic, no stop).  u_read_max = max u(read L).  The claws
    are always moved back to the start height before returning.  Same Probe interface (M_lo, M_hi, m_ff, rungs, track, _ramp, admissible)."""
    method = "poslift"

    def __init__(self, *a, adaptive=True, k_step=None, dz_min=None, **k):
        """step dz_k = min(DZ, H / K_STEP), H = L_admit_max - (L_last + eps_lo + nb); stop when dz_k < dz_min.  The fixed-step probe (adaptive=False) was removed 11-19."""
        if not adaptive:
            raise ValueError("the fixed-step poslift probe (adaptive=False) was removed 11-19")
        super().__init__(*a, **k)
        self.peak_u, self.probe_time_s = 0.0, None
        self.adaptive = True
        self.k_step = poslift_k_step()[0] if k_step is None else float(k_step)       # MEASURED [kg/mm] (CALIB K_STEP)
        self.dz_min = DZ_MIN if dz_min is None else float(dz_min)
        self.summary = None
        self.guard_events, self.sim_abort, self.u_read_max = [], None, 0.0

    def _claw_z(self):
        mujoco.mj_forward(self.ep.m, self.ep.d)
        return float(np.mean([self.ep.d.site_xpos[self.ep.site[s]][2] for s in CR.SIDES]))

    def _ramp_during(self, m_to, T):
        ep = self.ep
        n = max(int(round(T / CR.DT)), 1)
        m0 = ep.ctl.mass
        m_to = m0 if m_to is None else m_to
        for i in range(n):
            ep.ctl.mass = m0 + (m_to - m0) * (i + 1) / n
            ep._step()
        self.m_ff = ep.ctl.mass

    def _move(self, dz, ramp_to=None):
        ep = self.ep
        t = ep.k * CR.DT
        if abs(dz) > 0:
            for s in CR.SIDES:
                ep._add_path(s, t, t + T_MOVE, np.array([0.0, 0.0, dz]))
        self._ramp_during(ramp_to, T_MOVE)

    def run(self, M_stop=None):
        ep = self.ep
        ep.ctl.mass = 0.0
        ep.run_phases([("hold", 0.3)])
        z_rest, k0 = self._claw_z(), ep.k
        self._z_rest = z_rest                               # read by the Stage 0b finish phase (poslift_ff); no effect on poslift
        self.m_ff = 0.0
        zc, idx, prev_L, flat_mm, why, last_nb = 0.0, 0, None, 0.0, None, 0.0
        try:
            while True:
                if M_stop is not None and self.M_lo > M_stop:
                    why = "decided_refuse"
                    break
                ok, rows = self.admissible(self.M_lo)
                if not ok:
                    why = "rung_not_admissible"
                    break
                dz_k, headroom = DZ, None
                if prev_L is not None:    # adaptive step: largest dz whose predicted load stays admissible (robot-side numbers only)
                    headroom = self._headroom(prev_L, last_nb)
                    dz_k = adaptive_dz(headroom, self.k_step)
                    if dz_k < self.dz_min - 1e-12 or self._switch(headroom):
                        self.guard_events.append(dict(kind="pre_check", headroom=round(headroom, 4), dz_k=round(dz_k, 6)))
                        why = self._after_height(prev_L, last_nb, M_stop)     # poslift: "rung_not_admissible" (unchanged); poslift_ff: the force-step finish
                        break
                if zc + dz_k > Z_MAX + 1e-9:
                    why = "z_cap"
                    break
                fz0, ep.track["Fz_max"] = ep.track["Fz_max"], 0.0
                dz_k = dz_k if idx > 0 else 0.0
                self._move(dz_k, ramp_to=self.M_lo)
                zc += dz_k
                reads, nbs, nh = [], [], int(round(T_HOLD_POSLIFT / CR.DT))
                for i in range(nh):
                    ep._step()
                    if i >= nh - N_READ:
                        L, nb = self.read()
                        reads.append(L)
                        nbs.append(nb)
                L, nb = float(np.mean(reads)), float(max(nbs))
                peak_u = ep.track["Fz_max"] / ep.F_cap
                ep.track["Fz_max"] = max(fz0, ep.track["Fz_max"])
                dL = None if prev_L is None else L - prev_L
                rate = None if dL is None else dL / (dz_k * 1e3)               # [kg/mm]
                flat_mm, free = flat_update(flat_mm, dL, dz_k)
                u_read = self._u_read(L)                                      # static u at the READ load (robot-side); claw_guard: per-claw reads
                self.u_read_max = max(self.u_read_max, u_read)
                guard = u_read > E.U_ADMIT
                if guard:                                                     # lower bound only: the claws carry more than the payload row admits
                    free = False
                last_nb = nb
                rise_obj = 1e3 * (float(ep.d.xpos[ep.obj][2]) - ep.z0)
                self.track["rise_mm"] = max(self.track["rise_mm"], rise_obj)
                self.track["tilt"] = max(self.track["tilt"], ep.track["obj_tilt_max"])
                self.track["spill"] = max(self.track["spill"], ep.track["spill_ratio"])
                rec = dict(m_ff=round(self.m_ff, 4), z_cmd_mm=round(1e3 * zc, 3), claw_rise_mm=round(1e3 * (self._claw_z() - z_rest), 3),
                           rise_mm=round(rise_obj, 3), L=round(L, 4), dL=None if dL is None else round(dL, 4), noise_bound=round(nb, 4),
                           peak_u=round(peak_u, 4), u_read=round(u_read, 4), **rows, free=free,
                           dz_mm=round(1e3 * dz_k, 5), rate_kg_per_mm=None if rate is None else round(rate, 4),
                           headroom=None if headroom is None else round(headroom, 4))
                self._rec_extra(rec)
                self.peak_u = max(self.peak_u, peak_u)
                if free:
                    e = self.eps_free + nb
                    self.M_lo, self.M_hi = max(self.M_lo, L - e), min(self.M_hi, L + e)
                else:
                    self.M_lo = max(self.M_lo, L - self.eps_lo - nb)
                rec["Theta"] = [round(self.M_lo, 4), round(self.M_hi, 4)]
                self.rungs.append(rec)
                prev_L = L
                if ep.track["viol"]:                                          # sim-truth DIAGNOSTIC only (never a decision)
                    rec["sim_abort"] = ep.track["viol"]
                    if self.sim_abort is None:
                        self.sim_abort = dict(rung=len(self.rungs) - 1, **ep.track["viol"])
                if guard:
                    self.guard_events.append(dict(kind="load_guard", rung=len(self.rungs) - 1, L=round(L, 4), u_read=round(u_read, 4)))
                    why = "load_guard"
                    break
                if free:
                    why = "free"
                    break
                idx += 1
        finally:
            if zc > 0:                                      # back to the start height (ff stays at M_lo <= M in expectation)
                t = ep.k * CR.DT
                for s in CR.SIDES:
                    ep._add_path(s, t, t + T_MOVE, np.array([0.0, 0.0, -zc]))
                self._ramp_during(None, T_MOVE)
                ep.run_phases([("hold", 0.3)])
            self.probe_time_s = round((ep.k - k0) * CR.DT, 3)
            dzs = [r["dz_mm"] for r in self.rungs[1:]]
            self.summary = dict(adaptive=self.adaptive, n_rungs=len(self.rungs), min_dz_mm=min(dzs) if dzs else None, z_used_mm=round(1e3 * zc, 4),
                                probe_time_s=self.probe_time_s, k_step=self.k_step, dz_min_mm=1e3 * self.dz_min)
            self.summary.update(self._summary_extra())
        return why

    # ---- hooks (Stage 0b, 11-12).  The poslift versions below are the expressions the loop used before (bit-identical).
    def _headroom(self, prev_L, last_nb):
        """load [kg] the next height step may add before the guard row: total load at the worst COM (MODEL l_admit_max) minus read + eps + noise bound."""
        return l_admit_max(self.task, world=getattr(getattr(self, "ep", None), "world", None)) - (prev_L + self.eps_lo + last_nb)

    def _switch(self, headroom):
        """poslift: never (the dz < DZ_MIN rule alone stops the height ladder)."""
        return False

    def _after_height(self, prev_L, last_nb, M_stop):
        """what happens when the height ladder stops for headroom.  poslift: stop, lower bound only."""
        return "rung_not_admissible"

    def _u_read(self, L):
        return rung_torque_u(self.task, L, getattr(getattr(self, "ep", None), "world", None))

    def _rec_extra(self, rec):
        pass

    def _summary_extra(self):
        return {}


def adaptive_dz(headroom, k_step, dz_max=None):
    """step [m] whose predicted load rise k_step [kg/mm] * dz stays within `headroom` [kg]: min(DZ, headroom / K_STEP).  May be negative / below DZ_MIN (caller stops)."""
    return min(DZ if dz_max is None else dz_max, headroom / (k_step * 1e3))


def rate_flat(dL, dz):
    """FREE test on the load RATE: |dL| / dz < FREE_TOL / DZ  (identical to |dL| < FREE_TOL at dz = DZ; a smaller step needs a proportionally smaller dL)."""
    return abs(dL) * (DZ / dz) < FREE_TOL


def flat_update(flat_mm, dL, dz):
    """(consecutive flat travel [m], free).  A rung is flat when rate_flat; free needs N_FREE_CONFIRM x DZ of consecutive flat travel (= N_FREE_CONFIRM flat rungs
    at dz = DZ; tiny steps inside the contact slack cannot confirm free -- the false free of the first mixed-step calibration, pot i=0)."""
    flat_mm = flat_mm + dz if (dL is not None and rate_flat(dL, dz)) else 0.0
    return flat_mm, flat_mm >= N_FREE_CONFIRM * DZ - 1e-12


_L_ADMIT = {}


def _wk(world):
    """cache key of the world's hold geometry (None = nominal)."""
    from . import nyu_hold as NH
    return None if world is None else NH._hold_key(world)


def l_admit_max(task, hi=30.0, world=None):
    """largest total load L [kg] with rung_torque_u(task, L) <= U_ADMIT (worst COM; bisection, u is nondecreasing in L): MODEL.  Cached per task."""
    k = (task, _wk(world))
    if k not in _L_ADMIT:
        lo = empty_mass(task)
        if rung_torque_u(task, hi, world) <= E.U_ADMIT:
            _L_ADMIT[k] = hi
        elif rung_torque_u(task, lo, world) > E.U_ADMIT:
            _L_ADMIT[k] = lo
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if rung_torque_u(task, mid, world) <= E.U_ADMIT else (lo, mid)
            _L_ADMIT[k] = lo
    return _L_ADMIT[k]


def _calib_path():
    return POSLIFT_CALIB


def poslift_k_step():
    """(K_STEP [kg/mm], source).  MEASURED max over still-on-counter rungs of dL/dz (CALIB.json K_STEP).  """
    c = json.loads(pathlib.Path(_calib_path()).read_text())
    if "K_STEP" in c:
        return float(c["K_STEP"]["value"]), "CALIB K_STEP"
    # no silent fallback: the adaptive probe needs a MEASURED K_STEP.
    raise ValueError(f"adaptive poslift probe needs K_STEP in {_calib_path()} (run probe_liftoff calibrate --variant poslift_mixed)")




# ------------------------------------------------------------------------------------------------- Stage 0b (11-12): poslift_ff and claw_guard
def _base_key(task):
    """calibration files are keyed by the probe's base task ("box_behind" -> "box")."""
    from world_model.v2.stack import tasks as TK
    return TK.fixed_task(task)


def jump_max(task):
    """(per-task max per-rung load jump of the height ladder [kg], label).  MEASURED, JUMP_FILE (kstep_calib_2026-11-09: n = 20 worlds/task, any dz, noise-free
    reads); max over n, NOT a bound.  A task without an entry uses the pooled max (transfer ASSUMED).  Env PL_JUMP_FILE: read-only override."""
    f = pathlib.Path(os.environ.get("PL_JUMP_FILE") or JUMP_FILE)
    c = json.loads(f.read_text())
    k = _base_key(task)
    if k in c["per_task"]:
        v = c["per_task"][k]
        return float(v["value"]), f"MEASURED {k}: max over n={v['n']} height rungs, {v['n_worlds']} worlds ({f})"
    return float(c["pooled"]["value"]), f"MEASURED pooled max over n={c['pooled']['n']} rungs ({f}); transfer to {task!r} ASSUMED"


def ff_switch(headroom, jump):
    """poslift_ff: leave the height ladder when the next height step's worst-case jump (the MEASURED per-task max, any dz) would breach the guard."""
    return headroom < jump


def ff_precheck(task, L_prev, nb_prev, eps_lo, dm):
    """finish-phase pre-check: (ok, L_pred, u_pred), L_pred = L_prev + dm + eps_lo + nb, ok = u(L_pred) <= U_ADMIT (worst COM, rest pose).
    MODEL behind it (quasi-static, rigid counter contact): while the object rests on the counter its height and so the position loop's force are fixed by the
    contact, so a feed-forward step dm raises the arms' load by dm; once it hangs, the load settles at M whatever the feed-forward.  The ramp transient is
    not covered by the model; the calibration MEASURES it (finish-phase dL - dm)."""
    L_pred = L_prev + dm + eps_lo + nb_prev
    u = rung_torque_u(task, L_pred)
    return u <= E.U_ADMIT, L_pred, u


def ff_flat(dL, dm, ratio=None):
    """finish-phase free test = the 11-03 ladder rule on the commanded step (one rung): the read grows by less than ratio x dm (ratio FF_FREE_RATIO, ASSUMED)."""
    return dL < (FF_FREE_RATIO if ratio is None else ratio) * dm


class PosLiftFFProbe(PosLiftProbe):
    """poslift_ff (Stage 0b, 11-12): hybrid finish.  SIMULATION ONLY.
    Height ladder = poslift (adaptive) unchanged, except that it also leaves the ladder when the headroom is below JUMP (MEASURED per-task max per-rung jump,
    any dz; jump_max) -- the K_STEP x dz model does not describe the per-rung jump (B6).  Then the finish phase: the claws hold their height, the feed-forward
    ramps DM_FINE above its current value (Probe.rung, the 11-03 ladder mechanism; hold FF_T_HOLD) and the load is read.  Per finish rung, in this order:
      decided_refuse      M_lo > M_stop
      rung_not_admissible ff_precheck fails (u(L_prev + DM_FINE + eps_lo + nb) > U_ADMIT), or Probe.admissible(m_ff + DM_FINE) fails (u of the ff itself, O_MAX)
      read                L; flat = L - L_prev < FF_FREE_RATIO x DM_FINE (ff_flat)
      load_guard          u(read L) > U_ADMIT -> lower bound only (as poslift)
      free                interval [L - eps_free - nb, L + eps_free + nb]; otherwise M_lo <- max(M_lo, L - eps_lo - nb) and the next rung
    On leaving, the feed-forward is ramped back to M_lo if above it (poslift's invariant ff = M_lo), then run()'s finally moves the claws down as for poslift.
    The eps must come from this method's own calibration (load_calib(.., "poslift_ff"))."""
    method = "poslift_ff"

    def __init__(self, *a, dm_fine=None, ff_free_ratio=None, ff_t_hold=None, jump=None, **k):
        super().__init__(*a, **k)
        self.dm_fine = DM_FINE if dm_fine is None else float(dm_fine)
        self.ff_free_ratio = FF_FREE_RATIO if ff_free_ratio is None else float(ff_free_ratio)
        self.ff_t_hold = FF_T_HOLD if ff_t_hold is None else float(ff_t_hold)
        self.jump, self.jump_src = jump_max(self.task) if jump is None else (float(jump), "argument")
        self.ff = None                                   # finish-phase record once it started

    def _switch(self, headroom):
        return ff_switch(headroom, self.jump)

    def _after_height(self, prev_L, last_nb, M_stop):
        ep = self.ep
        self.ff = dict(start_rung=len(self.rungs), L0=round(prev_L, 4), m_ff0=round(self.m_ff, 4), M_lo0=round(self.M_lo, 4), n=0, why=None)
        L_prev, nb_prev, why = prev_L, last_nb, None
        while True:
            if M_stop is not None and self.M_lo > M_stop:
                why = "decided_refuse"
                break
            m_next = self.m_ff + self.dm_fine
            ok_u, L_pred, u_pred = ff_precheck(self.task, L_prev, nb_prev, self.eps_lo, self.dm_fine)
            ok_a, rows = self.admissible(m_next)
            if not (ok_u and ok_a):
                self.guard_events.append(dict(kind="ff_pre_check", L_pred=round(L_pred, 4), u_pred=round(u_pred, 4), **rows))
                why = "rung_not_admissible"
                break
            fz0, ep.track["Fz_max"] = ep.track["Fz_max"], 0.0
            L, nb, _ = self.rung(m_next, t_hold=self.ff_t_hold)          # Probe.rung: ramp T_RAMP, hold, mean of the last N_READ reads; tracks rise/tilt/spill
            peak_u = ep.track["Fz_max"] / ep.F_cap
            ep.track["Fz_max"] = max(fz0, ep.track["Fz_max"])
            dL = L - L_prev
            free = ff_flat(dL, self.dm_fine, self.ff_free_ratio)
            u_read = self._u_read(L)
            self.u_read_max = max(self.u_read_max, u_read)
            guard = u_read > E.U_ADMIT
            if guard:
                free = False
            rise_obj = 1e3 * (float(ep.d.xpos[ep.obj][2]) - ep.z0)
            rec = dict(phase="ff", m_ff=round(self.m_ff, 4), dm_cmd=round(self.dm_fine, 4), z_cmd_mm=self.rungs[-1]["z_cmd_mm"] if self.rungs else 0.0,
                       claw_rise_mm=round(1e3 * (self._claw_z() - self._z_rest), 3), rise_mm=round(rise_obj, 3), L=round(L, 4), dL=round(dL, 4),
                       noise_bound=round(nb, 4), peak_u=round(peak_u, 4), u_read=round(u_read, 4), **rows, L_pred=round(L_pred, 4), u_pred=round(u_pred, 4),
                       free=free, dz_mm=0.0)
            self._rec_extra(rec)
            self.peak_u = max(self.peak_u, peak_u)
            if free:
                e = self.eps_free + nb
                self.M_lo, self.M_hi = max(self.M_lo, L - e), min(self.M_hi, L + e)
            else:
                self.M_lo = max(self.M_lo, L - self.eps_lo - nb)
            rec["Theta"] = [round(self.M_lo, 4), round(self.M_hi, 4)]
            self.rungs.append(rec)
            self.ff["n"] += 1
            if ep.track["viol"]:                                          # sim-truth DIAGNOSTIC only (never a decision)
                rec["sim_abort"] = ep.track["viol"]
                if self.sim_abort is None:
                    self.sim_abort = dict(rung=len(self.rungs) - 1, **ep.track["viol"])
            if guard:
                self.guard_events.append(dict(kind="load_guard", rung=len(self.rungs) - 1, L=round(L, 4), u_read=round(u_read, 4)))
                why = "load_guard"
                break
            if free:
                why = "free"
                break
            L_prev, nb_prev = L, nb
        if self.m_ff > self.M_lo:                                         # back to poslift's invariant ff = M_lo before the claws go down
            self._ramp_during(self.M_lo, T_RAMP)
        self.ff["why"] = why
        return why

    def _summary_extra(self):
        return dict(ff=self.ff, dm_fine=self.dm_fine, ff_free_ratio=self.ff_free_ratio, ff_t_hold=self.ff_t_hold, jump=self.jump, jump_src=self.jump_src)


def claw_cap_kg(task, world=None):
    """per-claw vertical load limit [kg] = the static rows' F_cap / g (envelope: NERO payload rating 3 kg MEASURED-spec minus the gripper mass, per arm; MODEL)."""
    return float(_rest_envelope(_base_key(task), world=world).F_cap / G)


def rung_tau_u(task, L, world=None):
    """joint-torque part only of rung_torque_u: worst over the COM set P of max_j |tau_j| / tau_lim_j at the rest pose, total mass L, lever split (MODEL;
    the envelope's tau detail is rounded to 1e-3 Nm)."""
    tk = _base_key(task)
    c = max(L - empty_mass(tk), 0.0)
    env = _rest_envelope(tk, world=world)
    best = 0.0
    for pi in OB.Contents(c, c, interval_spec(tk, *prior_contents(tk)).contents.P).pi_vertices():
        _, det = env.u("p0", pi, np.zeros(3), "two_lever")
        best = max(best, max(float(np.max(np.abs(np.asarray(v, float)) / env.tau_lim)) for v in det["tau"].values()))
    return best


_L_TAU = {}


def l_tau_admit_max(task, hi=30.0, world=None):
    """largest total load with rung_tau_u <= U_ADMIT (bisection; MODEL).  Cached."""
    tk = _base_key(task)
    k = (tk, _wk(world))
    if k not in _L_TAU:
        lo = empty_mass(tk)
        if rung_tau_u(tk, hi, world) <= E.U_ADMIT:
            _L_TAU[k] = hi
        elif rung_tau_u(tk, lo, world) > E.U_ADMIT:
            _L_TAU[k] = lo
        else:
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if rung_tau_u(tk, mid, world) <= E.U_ADMIT else (lo, mid)
            _L_TAU[k] = lo
    return _L_TAU[k]


def claw_u(L_sides, cap_kg):
    """per-claw payload utilisation of a read: max over claws of read load / limit (the static payload row applied to the reads, not to a model split)."""
    return max(float(v) for v in L_sides.values()) / cap_kg


def claw_headroom(L_sides, nb_sides, eps_claw, cap_kg):
    """load [kg] the next step may add to the most loaded claw: min over claws of cap - (read + eps_claw + noise bound)."""
    return min(cap_kg - (float(L_sides[s]) + eps_claw + float(nb_sides[s])) for s in L_sides)


def split_position(L_l, L_r, e_l, e_r, width):
    """where the arms' vertical load sits along the claw axis, by the lever rule (MODEL): fraction f = L_l / (L_l + L_r) on the left claw, offset
    (f - 1/2) width [m] from the claw midpoint, + toward the left claw.  Interval from the per-claw error bounds e (exact interval arithmetic; f is
    increasing in L_l and decreasing in L_r).  None where a denominator is not positive.  It is the arms' load split, not the contents COM: equal only when
    hanging free with no internal moment between the claws (the closed chain's split is set by the controller)."""
    tot = L_l + L_r
    if tot <= 0:
        return None
    f = L_l / tot
    d_lo = (L_l - e_l) + (L_r + e_r)
    d_hi = (L_l + e_l) + (L_r - e_r)
    f_lo = (L_l - e_l) / d_lo if d_lo > 0 else None
    f_hi = (L_l + e_l) / d_hi if d_hi > 0 else None
    off = lambda x: None if x is None else round((min(max(x, 0.0), 1.0) - 0.5) * width, 5)
    return dict(frac_left=round(f, 4), offset_m=off(f), offset_lo_m=off(f_lo), offset_hi_m=off(f_hi), width_m=round(width, 5))


def claw_load_truth(ep, s):
    """SIM TRUTH (diagnostic, never a decision input): vertical load [kg] arm s carries = its weld force on the object + every contact force between a geom
    of arm s (any body named "<s>_arm...") and a geom of any carried body (object, lid, contents, cups).
    Call it ONLY on the efc_force of a physics substep (a step hook, as carry._check does): after an mj_forward at the control DT (Probe.read) the soft weld's
    force is recomputed with the larger timestep and comes out too low (11-12 pot check, calibration world 0, hanging free: welds 1.25 kg of M = 1.686 kg,
    object z-acceleration -1.9 m/s^2 in that forward pass)."""
    m, d = ep.m, ep.d
    i = ep.weld_id[s]
    fz = 0.0
    if d.eq_active[i]:
        sel = (d.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & (d.efc_id == i)
        if sel.any():
            f = np.zeros(d.nefc)
            f[sel] = d.efc_force[sel]
            res = np.zeros(m.nv)
            mujoco.mj_mulJacTVec(m, d, res, f)
            da = m.jnt_dofadr[m.body_jntadr[ep.obj]]
            fz += float(res[da + 2])
    side_g = {g for g in ep.arm_g if (m.body(m.geom_bodyid[g]).name or "").startswith(s)}
    cf = np.zeros(6)
    for k, c in enumerate(d.contact[:d.ncon]):
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in side_g and int(m.geom_bodyid[g2]) in ep.carried:
            sg = 1.0
        elif g2 in side_g and int(m.geom_bodyid[g1]) in ep.carried:
            sg = -1.0
        else:
            continue
        mujoco.mj_contactForce(m, d, k, cf)
        fz += sg * float((c.frame.reshape(3, 3).T @ cf[:3])[2])
    return fz / G


def _claw_calib():
    """MEASURED per-claw values of the claw_guard calibration (EPS_CLAW, K_STEP_CLAW), {} when the file is absent.  Env PL_CLAW_CALIB: read-only override."""
    f = pathlib.Path(os.environ.get("PL_CLAW_CALIB") or NEW_CALIB["claw_guard"])
    if not f.exists():
        return {}
    c = json.loads(f.read_text()).get("CLAW", {})
    return {k: (float(c[k]["value"]), f"MEASURED {f} ({c[k].get('label', '')[:80]})") for k in ("EPS_CLAW", "K_STEP_CLAW") if k in c}


class ClawGuardProbe(PosLiftProbe):
    """claw_guard (Stage 0b, 11-12): the poslift height ladder, with the guard and headroom on each claw's own read load against that claw's limit
    (claw_cap_kg, the static rows' F_cap) instead of the total load at the worst COM (rung_torque_u over the whole COM set).  SIMULATION ONLY.
      read      per claw s: L_s = e_z . (J_s^T)^+ (tau_cmd,s - bias_s) / g (the two terms Probe.read sums; same noise draws), noise bound nb_s (MODEL of ASSUMED TAU_NOISE)
      guard     u_read = max(max_s L_s / cap, u_tau(L)) > U_ADMIT -> load_guard (u_tau = joint-torque rows at the worst COM, kept from today's row)
      headroom  min(min_s cap - (L_s + EPS_CLAW + nb_s), l_tau_admit_max - (L + eps_lo + nb)); step dz = min(DZ, headroom / K_STEP_CLAW); stop when dz < DZ_MIN
      admissible (top of each rung) after the first read: max_s (L_s + EPS_CLAW + nb_s) / cap and u_tau(M_lo) <= U_ADMIT, overshoot <= O_MAX; before it: today's row
    EPS_CLAW: the claw_guard CALIB's MEASURED per-claw read residual; without that file the total-load eps_lo stands in (ASSUMED); 0 when eps_lo = 0 (calibration).
    K_STEP_CLAW: MEASURED per-claw max dL_s/dz (claw_guard CALIB); without it the pooled total K_STEP (transfer ASSUMED).
    Per rung (rec["claw"]) and in the summary: the per-claw reads, noise bounds, sim-truth per-claw loads (claw_wrench; DIAGNOSTIC, never a decision input)
    and split_position (the arms' load position along the claw axis + its interval).  The checker does not use the split yet."""
    method = "claw_guard"

    def __init__(self, *a, eps_claw=None, k_step_claw=None, **k):
        super().__init__(*a, **k)
        cal = _claw_calib()
        if eps_claw is not None:
            self.eps_claw, self.eps_claw_src = float(eps_claw), "argument"
        elif self.eps_lo == 0.0:
            self.eps_claw, self.eps_claw_src = 0.0, "0 (eps_lo = 0: calibration run)"
        elif "EPS_CLAW" in cal:
            self.eps_claw, self.eps_claw_src = cal["EPS_CLAW"]
        else:
            self.eps_claw, self.eps_claw_src = self.eps_lo, "ASSUMED = eps_lo (total-load residual; per-claw residual not yet MEASURED)"
        if k_step_claw is not None:
            self.k_step, self.k_step_src = float(k_step_claw), "argument"
        elif "K_STEP_CLAW" in cal:
            self.k_step, self.k_step_src = cal["K_STEP_CLAW"]
        else:
            self.k_step_src = "pooled total K_STEP (CALIB of poslift); transfer to the per-claw load ASSUMED"
        self.cap_kg = claw_cap_kg(self.task, getattr(getattr(self, "ep", None), "world", None))
        self._buf, self._last_sides, self._last_u, self._last_h = [], None, None, None
        self._truth_now = {s: 0.0 for s in CR.SIDES}

        def keep_truth(_ep):                          # sim truth from the substep's own constraint forces (diagnostic; see claw_load_truth)
            self._truth_now = {s: claw_load_truth(_ep, s) for s in CR.SIDES}
        self.ep.step_hooks.append(keep_truth)

    def read(self):
        """Probe.read with the per-claw terms kept (same rng draws, same summation order -> the same total); + sim-truth per-claw loads (diagnostic)."""
        ep, m, d = self.ep, self.ep.m, self.ep.d
        mujoco.mj_forward(m, d)
        Fz, nb, parts = 0.0, 0.0, {}
        for s in CR.SIDES:
            dofs = ep.dofs[s]
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mujoco.mj_jacSite(m, d, jp, jr, ep.site[s])
            Pz = np.linalg.pinv(np.vstack([jp[:, dofs], jr[:, dofs]]).T)[2]
            r = self.tau[s] - d.qfrc_bias[dofs]
            if self.noise:
                r = r + self.rng.uniform(-TAU_NOISE, TAU_NOISE, size=7)
            f, b = float(Pz @ r), float(np.abs(Pz).sum()) * TAU_NOISE
            Fz += f
            nb += b
            parts[s] = (f / G, b / G)
        self._buf.append((parts, dict(self._truth_now)))
        return Fz / G, nb / G

    def _consume(self):
        buf, self._buf = self._buf, []
        if not buf:
            return self._last_sides
        self._last_sides = dict(L={s: float(np.mean([p[s][0] for p, _ in buf])) for s in CR.SIDES},
                                nb={s: float(max(p[s][1] for p, _ in buf)) for s in CR.SIDES},
                                truth={s: float(np.mean([t[s] for _, t in buf])) for s in CR.SIDES})
        return self._last_sides

    def _u_read(self, L):
        sd = self._consume()
        u_c, u_t = claw_u(sd["L"], self.cap_kg), rung_tau_u(self.task, L, getattr(getattr(self, "ep", None), "world", None))
        self._last_u = (u_c, u_t)
        return max(u_c, u_t)

    def _headroom(self, prev_L, last_nb):
        sd = self._last_sides
        h_c = claw_headroom(sd["L"], sd["nb"], self.eps_claw, self.cap_kg)
        h_t = l_tau_admit_max(self.task, world=getattr(getattr(self, "ep", None), "world", None)) - (prev_L + self.eps_lo + last_nb)
        self._last_h = (h_c, h_t)
        return min(h_c, h_t)

    def admissible(self, m_to):
        if self._last_sides is None:
            return super().admissible(m_to)                  # before the first read: today's row (worst COM)
        sd = self._last_sides
        u_c = max((sd["L"][s] + self.eps_claw + sd["nb"][s]) / self.cap_kg for s in CR.SIDES)
        u_t = rung_tau_u(self.task, m_to, getattr(getattr(self, "ep", None), "world", None))
        over = m_to - self.M_lo
        u = max(u_c, u_t)
        return (u <= E.U_ADMIT and over <= O_MAX + 1e-9), dict(u=round(u, 4), overshoot=round(over, 4), u_claw=round(u_c, 4), u_tau=round(u_t, 4))

    def _width(self):
        mujoco.mj_forward(self.ep.m, self.ep.d)
        pl, pr = (self.ep.d.site_xpos[self.ep.site[s]].copy() for s in ("left", "right"))
        return pl, pr

    def _split(self, sd):
        pl, pr = self._width()
        e = {s: self.eps_claw + sd["nb"][s] for s in CR.SIDES}
        return split_position(sd["L"]["left"], sd["L"]["right"], e["left"], e["right"], float(np.linalg.norm(pl - pr)))

    def _rec_extra(self, rec):
        sd = self._last_sides
        if sd is None:
            return
        rec["claw"] = dict(L={s: round(v, 4) for s, v in sd["L"].items()}, nb={s: round(v, 4) for s, v in sd["nb"].items()},
                           truth={s: round(v, 4) for s, v in sd["truth"].items()},
                           u_claw=None if self._last_u is None else round(self._last_u[0], 4), u_tau=None if self._last_u is None else round(self._last_u[1], 4),
                           headroom_claw=None if self._last_h is None else round(self._last_h[0], 4),
                           headroom_tau=None if self._last_h is None else round(self._last_h[1], 4), split=self._split(sd))

    def _com_truth_offset(self):
        """sim-truth offset [m] of the carried bodies' COM from the claw midpoint along the claw axis (+ toward left).  DIAGNOSTIC only."""
        ep, d = self.ep, self.ep.d
        ms = np.array([ep.m.body_mass[b] for b in ep.carried])
        com = (ms[:, None] * np.array([d.xipos[b] for b in ep.carried])).sum(0) / ms.sum()
        pl, pr = self._width()
        ax = (pl - pr) / np.linalg.norm(pl - pr)
        return round(float(np.dot(com - 0.5 * (pl + pr), ax)), 5)

    def _summary_extra(self):
        last = self.rungs[-1].get("claw") if self.rungs else None
        return dict(claw=dict(cap_kg=round(self.cap_kg, 4), eps_claw=self.eps_claw, eps_claw_src=self.eps_claw_src, k_step=self.k_step,
                              k_step_src=getattr(self, "k_step_src", None), l_tau_admit_max=round(l_tau_admit_max(self.task, world=getattr(getattr(self, "ep", None), "world", None)), 4),
                              split_last=None if last is None else last["split"], split_last_free=bool(self.rungs and self.rungs[-1]["free"]),
                              com_truth_offset_m=self._com_truth_offset()))


def _new_method_assumptions(method):
    a = method_assumptions("poslift")
    a["probe_method"] = method
    a["U_ADMIT"] = f"{E.U_ADMIT} MODEL (rated limits, USER 11-09)"
    if method == "poslift_ff":
        try:
            j = {t: jump_max(t)[0] for t in ("box", "tray6", "pot")}
        except Exception as e:                       # noqa: BLE001 -- report, do not invent
            j = f"unavailable ({e})"
        a.update(JUMP=f"MEASURED per-task max per-rung height-ladder jump {j} kg ({JUMP_FILE}); max over n=20 worlds/task, not a bound; switch to the finish when headroom < JUMP",
                 DM_FINE=f"{DM_FINE} kg ASSUMED (proposal 11-12, needs user approval): finish-phase feed-forward step",
                 FF_FREE_RATIO=f"{FF_FREE_RATIO} ASSUMED (needs approval): finish-phase free test dL < ratio x DM_FINE (11-03 ladder rule, FREE_RATIO reused)",
                 FF_T_HOLD=f"{FF_T_HOLD} s ASSUMED (needs approval): hold per finish rung (= poslift hold, reused)",
                 T_RAMP=f"{T_RAMP} s design (Probe.rung)", O_MAX=f"{O_MAX} kg ASSUMED (11-03 ladder, Probe.admissible)",
                 ff_precheck="MODEL quasi-static: a feed-forward step dm adds <= dm to the arms' load while the object rests on the counter; ramp transient MEASURED in calibration only")
    else:
        cal = _claw_calib()
        a.update(claw_cap="MODEL: per-claw F_cap / g of the static rows ((3 kg NERO payload rating, MEASURED-spec) - gripper mass)",
                 EPS_CLAW=(f"{cal['EPS_CLAW'][0]} kg {cal['EPS_CLAW'][1]}" if "EPS_CLAW" in cal else "ASSUMED = eps_lo until the claw_guard calibration MEASURES the per-claw read residual"),
                 K_STEP_CLAW=(f"{cal['K_STEP_CLAW'][0]} kg/mm {cal['K_STEP_CLAW'][1]}" if "K_STEP_CLAW" in cal else "pooled total K_STEP; transfer to per-claw ASSUMED until MEASURED"),
                 torque_rows="MODEL: joint-torque rows kept at the worst COM (rung_tau_u)",
                 split="MODEL lever rule on the per-claw reads; interval from EPS_CLAW + per-claw noise bound (ASSUMED TAU_NOISE); the arms' load split, not the COM")
    return a


def make_probe(ep, task, eps_lo, eps_free, rng=None, noise=True, method=None, adaptive=True):
    """the one place a Probe is built.  method None -> PROBE_METHOD (env BIMANUAL_PROBE, default "claw_guard" since 11-13).  method="ladder" (programmatic only, not in METHODS) returns the module's
    `Probe` (looked up now, so probe_liftoff, which patches P.Probe, and probe_al keep working)."""
    method = PROBE_METHOD if method is None else method
    if method not in METHODS + _PROGRAMMATIC:
        raise ValueError(f"probe method {method!r} not in {METHODS}")
    cls = {"poslift": PosLiftProbe, "poslift_ff": PosLiftFFProbe, "claw_guard": ClawGuardProbe}.get(method) or Probe
    pr = cls(ep, task, eps_lo, eps_free, rng=rng, noise=noise, **(dict(adaptive=adaptive) if method != "ladder" else {}))
    pr.method = method
    return pr


def method_of(pr):
    return getattr(pr, "method", "ladder")


def default_calib(method=None):
    """CALIB.json of the method's eps (the eps must match the method): poslift -> the repo copy; ladder -> the 11-03 file; poslift_ff / claw_guard ->
    NEW_CALIB (the repo copy of the PROBE_FIX_RUN.md calibration; absent until that run is done and copied)."""
    m = PROBE_METHOD if method is None else method
    if m == "poslift":
        return str(POSLIFT_CALIB)
    return str(NEW_CALIB[m]) if m in NEW_CALIB else LADDER_CALIB


def _k_step_note():
    try:
        k, src = poslift_k_step()
    except ValueError as e:                    # a CALIB without K_STEP: say so, do not invent a value
        return f"not in CALIB ({e}); adaptive step unavailable"
    return f"MEASURED {k:.4g} kg/mm, {src} (adaptive step dz = min(DZ, H/K_STEP))"


def method_assumptions(method=None):
    method = PROBE_METHOD if method is None else method
    if method in ("poslift_ff", "claw_guard"):
        return _new_method_assumptions(method)
    if method == "poslift":
        return dict(probe_method="poslift", DZ=f"{DZ} m ASSUMED, USER-approved 11-09", Z_MAX=f"{Z_MAX} m ASSUMED, USER-approved 11-09",
                    FREE_TOL=f"{FREE_TOL} kg ASSUMED, USER-approved 11-09", T_HOLD=f"{T_HOLD_POSLIFT} s ASSUMED, USER-approved 11-09",
                    N_FREE_CONFIRM=f"{N_FREE_CONFIRM} ASSUMED, USER-approved 11-09", T_MOVE=f"{T_MOVE} s design",
                    TAU_NOISE=f"+-{TAU_NOISE} Nm/joint ASSUMED uniform", tracker="carry.LeverCtl (FL-MPC, pre-11-07)",
                    DZ_MIN=f"{DZ_MIN} m ASSUMED, USER-approved 11-09", K_STEP=_k_step_note(),
                    free_test="rate |dL|/dz < FREE_TOL/DZ",
                    guard="u(read L) > U_ADMIT (1.0 MODEL: rated limits, USER 11-09)")
    return dict(probe_method="ladder", DM=f"{DM} kg ASSUMED", O_MAX=f"{O_MAX} kg ASSUMED", FREE_RATIO=f"{FREE_RATIO} ASSUMED",
                TAU_NOISE=f"+-{TAU_NOISE} Nm/joint ASSUMED uniform", tracker="carry.LeverCtl (FL-MPC, pre-11-07)")


def load_calib(path, method):
    """CALIB.json of the probe method `method` (its eps must match the method).  Asserts the file's "method" field; the 11-03 ladder file predates the
    field (no "method" key) and is accepted for "ladder" only.  Returns the whole dict."""
    c = json.loads(pathlib.Path(path).read_text())
    got = c.get("method", "ladder" if "method" not in c and "variant" not in c else None)
    assert got == method, f"{path}: CALIB method {got!r} != probe method {method!r} (eps must match the method)"
    return c


# ------------------------------------------------------------------------------------------------- jobs
def _seed(kind, task, i):
    return int(np.random.SeedSequence([{"calib": 11, "e2e": 22, "onepoke": 33}[kind], CC.TASK_ID[task], i]).generate_state(1)[0])


def _world(task, seed):
    return CC.sample_world(task, TK.prior_box(task), np.random.default_rng(seed))


def _episode(world, wp=None):
    return CR.CarryEpisode(world, m_hat=0.0, fric_comp=True, weld_solref=CC.WELD_STIFF, wp=wp)


def _calib_job(args, method=None):
    """noise-free ladder to free (no stop rule): record per-rung read vs truth.  method None -> PROBE_METHOD (make_probe)."""
    task, i = args
    seed = _seed("calib", task, i)
    t0 = time.time()
    try:
        w = _world(task, seed)
        ep = _episode(w)
        M = true_mass(ep)
        pr = make_probe(ep, task, eps_lo=0.0, eps_free=0.0, noise=False, method=method)
        pr.M_lo = empty_mass(task)                          # calibration: start soft from the empty object, ignore the stop rule
        why = pr.run(M_stop=None)
        free = [r for r in pr.rungs if r["free"]]
        on = [r for r in pr.rungs if not r["free"]]
        return dict(task=task, i=i, seed=seed, probe_method=method_of(pr), M_true=M, why=why, rungs=pr.rungs,
                    res_free=(free[-1]["L"] - M) if free else None,
                    res_on_max=max((r["L"] - M for r in on), default=None),
                    viol=ep.track["viol"], **pr.track, wall_s=round(time.time() - t0, 1),
                    **({} if method_of(pr) in ("poslift", "ladder") else      # Stage 0b methods only (additive; poslift/ladder records unchanged)
                       dict(probe_time_s=getattr(pr, "probe_time_s", None), peak_u=round(getattr(pr, "peak_u", 0.0), 4), interval=[pr.M_lo, pr.M_hi],
                            truth_inside=bool(pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9), probe_summary=getattr(pr, "summary", None),
                            guard_events=list(getattr(pr, "guard_events", [])), u_read_max=round(float(getattr(pr, "u_read_max", 0.0)), 4))))
    except Exception as e:
        import traceback
        return dict(task=task, i=i, seed=seed, error=repr(e), tb=traceback.format_exc()[-1500:])


def calibrate_summary(rows):
    out = {}
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task and "error" not in r]
        fr = [r["res_free"] for r in rs if r["res_free"] is not None]
        on = [r["res_on_max"] for r in rs if r["res_on_max"] is not None]
        out[task] = dict(n=len(rs), n_free=len(fr), n_error=sum("error" in r for r in rows if r["task"] == task),
                         free_res_max_abs=round(max(map(abs, fr)), 4) if fr else None,
                         free_res_p50=round(float(np.median(fr)), 4) if fr else None,
                         on_res_max=round(max(on), 4) if on else None,
                         n_viol=sum(bool(r["viol"]) for r in rs), rise_mm_max=round(max(r["rise_mm"] for r in rs), 2) if rs else None,
                         tilt_max=round(max(r["tilt"] for r in rs), 2) if rs else None,
                         spill_max=round(max(r["spill"] for r in rs), 3) if rs else None,
                         not_free=[r["why"] for r in rs if r["why"] != "free"])
    fr_all = [abs(r["res_free"]) for r in rows if r.get("res_free") is not None]
    on_all = [r["res_on_max"] for r in rows if r.get("res_on_max") is not None]
    eps = dict(EPS_FREE=round(max(fr_all), 4) if fr_all else None, EPS_LO=round(max(max(on_all), 0.0), 4) if on_all else None,
               label="MEASURED: max over n calibration worlds (noise-free reads; the ASSUMED torque-noise bound is added per read)")
    return dict(per_task=out, eps=eps)


def _e2e_job(args, method=None, adaptive=True):
    """args = (task, i, eps_lo, eps_free[, plan]).  plan (11-05, same_worlds --plan-source codex): dict(wp = closed_chain.WaypointPlan, fallback = tuple of recovery
    skills, belief = {param: [lo, hi]} | None, outside_b = "refuse" | "carry"): the proposer's compiled plan replaces the default one in the static admission, the
    carry (its segments and recoveries) and the refuse path (its fallback).  Without it: unchanged.  method: probe method (None -> PROBE_METHOD).
    A schema-v2 plan (plan["plan_dict"] with phenomena + probes) runs probe_select.fixed_job instead; any other plan dict (wp / fallback / belief, e.g. coffee_run's compiled job_plan) runs the code below."""
    task, i, eps_lo, eps_free, *rest = args
    plan = rest[0] if rest else None
    from . import probe_select as PS
    if PS.is_v2(plan):                                       # schema-v2 plan (11-09): the LLM's probes -> coverage -> Rule 1 -> Rule 2 re-ask -> carry (probe_select.run_world)
        return PS.fixed_job((task, i, eps_lo, eps_free, plan), method=method, adaptive=adaptive)
    wp = plan["wp"] if plan else None
    seed = _seed("e2e", task, i)
    t0 = time.time()
    try:
        w = _world(task, seed)
        ep = _episode(w, wp)
        M = true_mass(ep)
        cmax = c_admit_max(task, wp=wp)
        M_stop = None if cmax is None else empty_mass(task) + cmax
        pr = make_probe(ep, task, eps_lo, eps_free, rng=np.random.default_rng(seed), noise=True, method=method, adaptive=adaptive)
        if not static_interval(task, *prior_contents(task), w, wp=wp)["action"].startswith("refuse"):
            why = "no_probe_needed"                          # verdict already "carry" on all of Theta_0: the probe runs 0 rungs (11-09: was labelled "free")
            pr.M_lo, pr.M_hi = empty_mass(task) + prior_contents(task)[0], empty_mass(task) + prior_contents(task)[1]
        else:
            why = pr.run(M_stop=M_stop)
        probe_viol = ep.track["viol"]
        sound = bool(pr.M_lo - 1e-9 <= M <= pr.M_hi + 1e-9)
        rec = dict(task=task, i=i, seed=seed, probe_method=method_of(pr), probe_assumptions=method_assumptions(method_of(pr)), M_true=M, why=why,
                   rungs=pr.rungs, Theta=[pr.M_lo, pr.M_hi], theta_sound=sound, peak_u=round(getattr(pr, 'peak_u', 0.0), 4), probe_time_s=getattr(pr, 'probe_time_s', None),
                   probe_viol=probe_viol, probe_track=dict(pr.track), theta_world=CR._theta(w),
                   u_read_max=round(float(getattr(pr, 'u_read_max', 0.0)), 4), guard_events=list(getattr(pr, 'guard_events', [])), probe_summary=getattr(pr, 'summary', None), sim_abort=getattr(pr, 'sim_abort', None))
        decision = None
        if why in ("free", "no_probe_needed"):
            c_lo, c_hi = pr.M_lo - empty_mass(task), pr.M_hi - empty_mass(task)
            st = static_interval(task, c_lo, c_hi, w, wp=wp)
            decision = dict(action=st["action"], a_admit=st["a_admit"], u_nom=st["u_nom"], contents=[c_lo, c_hi])
            if plan and plan.get("belief"):                  # v1 Rule 2 (probe set inside the proposer's belief B) removed with the v1 schema
                raise ValueError("v1 plan with a belief B: the v1 proposer schema is retired (USER 11-19); send a schema-v2 plan")
        rec["decision"] = decision
        if decision and decision["a_admit"] is not None:
            ep.set_belief(0.5 * (decision["contents"][0] + decision["contents"][1]), a_plan=decision["a_admit"])
            m_nom = ep.m_nom
            ep.ctl.mass = pr.m_ff
            pr._ramp(m_nom, 0.3)
            ep.run_phases([("hold", 0.4)])
            ep.ctl.mass = m_nom
            carry = CR.run_segments(ep, dict(segments=[]), branch_backups=True)
            rec["carry"] = dict(segments=carry["segments"], end_in_R=carry["end_in_R"], pre_hold_viol=carry["pre_hold_viol"])
            rec["outcome"] = "carried" if (carry["end_in_R"][0] and not any(s["nominal"]["viol"] for s in carry["segments"])) else "carry_failed"
        else:
            pr._ramp(0.0, 0.5)
            if plan and plan.get("fallback"):               # the proposer's fallback (hold / let_go / set_down: the object is still on its start support)
                from world_model.v2.stack.proposer import compile as PC
                ep.run_phases(PC.fallback_phases(plan["fallback"], ep))
                rec["fallback"] = list(plan["fallback"])
            else:
                ep.run_phases([("hold", 0.3), ("weld", False), ep.retreat_phase(), ("hold", 0.5)])
            okR, whyR = ep.in_R()
            rec["refuse_end_in_R"] = [okR, whyR]
            rec["outcome"] = "refused_in_R" if okR and not ep.track["viol"] else "refused_bad"
        rec["any_U"] = bool(probe_viol or ep.track["viol"] or any(s["nominal"]["viol"] for s in rec.get("carry", {}).get("segments", []))
                            or rec.get("carry", {}).get("pre_hold_viol"))            # 11-12 B12: the ramp / hold between probe and carry
        rec["recovery"] = _UG.recovery_of(rec.get("carry", {}).get("segments"))                 # 11-12 B4 step 1: recorded, not in the outcome
        rec["recovery_bad"] = _UG.recovery_bad(rec["recovery"])
        from world_model.v2.adapters import outcome_carry as OCC      # the ONE goal + success rule (stack.goal / stack.outcome), same as AgentChord's records
        carried = "carry" in rec
        rec.update(OCC.measure(ep, task, carried, rec["any_U"], rec["carry"]["pre_hold_viol"] if carried else None, refused=not carried, reason=rec["outcome"],
                               end_in_R=tuple(rec["carry"]["end_in_R"] if carried else rec["refuse_end_in_R"])))
        rec["wall_s"] = round(time.time() - t0, 1)
        return rec
    except Exception as e:
        import traceback
        return dict(task=task, i=i, seed=seed, error=repr(e), tb=traceback.format_exc()[-800:])


def _onepoke_job(args):
    """ablation: one push straight to the strongest torque-admissible feedforward (no ladder), on the same object."""
    task, i = args
    seed = _seed("onepoke", task, i)
    try:
        w = _world(task, seed)
        ep = _episode(w)
        M = true_mass(ep)
        m_top = empty_mass(task)
        while rung_torque_u(task, m_top + DM) <= E.U_ADMIT and m_top < 8.0:
            m_top += DM
        pr = Probe(ep, task, 0.0, 0.0, noise=False)           # onepoke ablation = ladder rung by definition
        ep.ctl.mass = 0.0
        ep.run_phases([("hold", 0.3)])
        L, nb, rise = pr.rung(m_top)
        return dict(task=task, i=i, seed=seed, M_true=M, m_push=m_top, L=L, rise_mm=pr.track["rise_mm"], tilt=pr.track["tilt"],
                    spill=pr.track["spill"], viol=ep.track["viol"])
    except Exception as e:
        return dict(task=task, i=i, seed=seed, error=repr(e))


def e2e_summary(rows):
    L = ["task | n | probe stop: free / decided refuse / rung refused | Theta contains truth | outcome: carried / carry_failed / "
         "refused_in_R / refused_bad | any U | rungs p50/max | final width kg p50/max | probe rise mm max | errors"]
    for task in sorted({r["task"] for r in rows}):
        rs = [r for r in rows if r["task"] == task and "error" not in r]
        if not rs:
            continue
        c = lambda k, v: sum(r.get(k) == v for r in rs)
        nr = [len(r["rungs"]) for r in rs]
        wid = [r["Theta"][1] - r["Theta"][0] for r in rs]
        L.append(f"{task} | {len(rs)} | {c('why', 'free')} (+{c('why', 'no_probe_needed')} no probe needed) / {c('why', 'decided_refuse')} / {c('why', 'rung_not_admissible')} | "
                 f"{sum(r['theta_sound'] for r in rs)}/{len(rs)} | {c('outcome', 'carried')} / {c('outcome', 'carry_failed')} / "
                 f"{c('outcome', 'refused_in_R')} / {c('outcome', 'refused_bad')} | {sum(r['any_U'] for r in rs)} | "
                 f"{int(np.median(nr))}/{max(nr)} | {np.median(wid):.3f}/{max(wid):.3f} | "
                 f"{max(r['probe_track']['rise_mm'] for r in rs):.2f} | {sum('error' in r for r in rows if r['task'] == task)}")
    return "\n".join(L)


def _pool_run(fn, jobs, procs, out_file):
    import multiprocessing as mp
    rows = []
    with mp.get_context("spawn").Pool(procs, maxtasksperchild=3) as pool, open(out_file, "a") as f:
        for r in pool.imap_unordered(fn, jobs):
            rows.append(r)
            f.write(json.dumps(r, default=float) + "\n")
            f.flush()
            print(f"{r.get('task')} {r.get('i')} {r.get('why', '')} {r.get('outcome', '')} {r.get('error', '')[:120]}", flush=True)
    return rows


ASSUMPTIONS = dict(probe_method="ladder (see method_assumptions() for the active method)", DM=f"{DM} kg ASSUMED push step", O_MAX=f"{O_MAX} kg ASSUMED max push above the lower bound",
                   FREE_RATIO=f"{FREE_RATIO} ASSUMED free test (one rung)", TAU_NOISE=f"+-{TAU_NOISE} Nm/joint ASSUMED uniform",
                   grasp="ASSUMED rigid weld (user 11-01), stiff solref (0.004, 1)", controller="LeverCtl + FRIC_COMP (ASSUMED gains)",
                   rows="arm-side static rows (user 11-02 pm), U_ADMIT 1.0 MODEL (rated limits, USER 11-09)", worlds="uniform on the full prior per task")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "e2e", "onepoke"])
    ap.add_argument("--tasks", default="box,tray6")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--calib", help="default: the CALIB of the active method (default_calib)")
    ap.add_argument("--method", choices=METHODS, default=None, help="probe method (default: env BIMANUAL_PROBE, else claw_guard)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--planner", choices=CR.PLANNERS, default=None,
                    help="plan the static admission uses and the carry executes (default: env BIMANUAL_PLANNER, else scripted = legacy PLANS)")
    a = ap.parse_args(argv)
    if a.planner is not None:
        import os
        os.environ["BIMANUAL_PLANNER"] = a.planner                  # inherited by the spawned workers
    global PROBE_METHOD
    if a.method is not None:
        PROBE_METHOD = a.method
        os.environ["BIMANUAL_PROBE"] = a.method                    # inherited by the spawned workers
    a.calib = a.calib or default_calib()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tasks = a.tasks.split(",")
    t0 = time.time()
    if PROBE_METHOD == "poslift":                                   # drop the ladder-only fields
        for k in ("DM", "O_MAX", "FREE_RATIO"):
            ASSUMPTIONS.pop(k, None)
    ASSUMPTIONS.update(method_assumptions())
    if a.cmd == "calibrate":
        rows = _pool_run(functools.partial(_calib_job, method=PROBE_METHOD), [(t, i) for t in tasks for i in range(a.n)], a.procs, out / "calib.jsonl")
        s = calibrate_summary(rows)
        s.update(method=PROBE_METHOD, date=time.strftime("%Y-%m-%d"), assumptions=ASSUMPTIONS, n_per_task=a.n, wall_s=round(time.time() - t0))
        (out / "CALIB.json").write_text(json.dumps(s, indent=1, default=float))
        print(json.dumps(s, indent=1, default=float))
    elif a.cmd == "e2e":
        cal = load_calib(a.calib, PROBE_METHOD)["eps"]
        jobs = [(t, i, cal["EPS_LO"], cal["EPS_FREE"]) for t in tasks for i in range(a.n)]
        rows = _pool_run(functools.partial(_e2e_job, method=PROBE_METHOD), jobs, a.procs, out / "e2e.jsonl")
        s = e2e_summary(rows)
        hdr = (f"probe ({PROBE_METHOD}) end-to-end ({time.strftime('%Y-%m-%d')}), worlds uniform on the full prior, n = {a.n} per task, held-out "
               f"seeds (calibration used different worlds)\neps (MEASURED, {a.calib}): {cal}\nassumptions: {ASSUMPTIONS}\n"
               f"static admission ceiling (MODEL): " + ", ".join(f"{t} contents <= {c_admit_max(t)}" for t in tasks) + "\n")
        (out / "SUMMARY.txt").write_text(hdr + s + f"\nwall {round(time.time() - t0)} s\n")
        print(hdr + s, flush=True)
    else:
        rows = _pool_run(_onepoke_job, [(t, i) for t in tasks for i in range(a.n)], a.procs, out / "onepoke.jsonl")
        (out / "SUMMARY.txt").write_text("\n".join(json.dumps({k: r.get(k) for k in ("task", "i", "M_true", "m_push", "rise_mm", "tilt",
                                                                                    "spill", "viol", "error")}, default=float)
                                                   for r in rows) + "\n")


if __name__ == "__main__":
    main()
