"""Symbolic covered-pan domain + belief box.  SIMULATION ONLY.

Symbolic semantics are deliberately optimistic (what a task-level planner / AgentChord-style graph assumes): any held part
of the pan + `lift` = pan lifted.  Physics (one claw cannot carry it, the lid is a loose body, ...) is the certificate's job.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

# --- bounds, labelled per AGENTS.md 'Bounds and assumptions': ASSUMED / MEASURED <source> / MODEL; 'design' = a choice, not a bound ---------------------------------------------------------------
OBS_DXY = 0.005        # ASSUMED (10-28). m residual pose error after observe_pose; used only in OBS_MODE='assumed'. MEASURED 10-29
                       # (results/pan_observe_pose_2026-10-29, n=200, max over n): zed p95 2.3 / max 3.4 mm; zed_calib p95 6.2 / max 8.1 mm
OBS_YAW = 2.0          # ASSUMED (10-28). deg; MEASURED 10-29 (same source): zed p95 4.7 / max 9.4 deg -> the 2 deg assumption does not hold
OBS_MODE = "assumed"   # "assumed" (10-28): observe_pose shrinks the box to OBS_DXY/OBS_YAW and Stage B aims at the nominal pose.
                       # "built" (10-29): pan/observe_pose.py runs inside the Stage-B sim (depth fit at noise OBS_NOISE, claws re-aimed
                       # at the estimate); the box is NOT shrunk, the plan is tested on the full pose box.
OBS_NOISE = "zed"      # observe_pose.NOISE level used in "built" mode
L_FRACS = (0.6, 0.65, 0.7, 0.75)   # design grid of left pinch positions; lower limit MEASURED 10-28 (<0.6: no clean pre-grasp at nominal)
R_HANDLE_FRAC = 0.5    # design: right claw on the handle (side grip), as in the v2/v3 sweeps
PUSHES = (0.07, 0.08)  # design: m push along -u before the rim clamp (0.07 = the plan measured in the 10-28 sweeps)
GRIP_N = 10.0          # MODEL: squeeze = MJCF gripper force limit (force mode)


@dataclass(frozen=True)
class Belief:
    """Box of true-world parameters theta the plan must work for (truth vs believed nominal pose)."""
    dxy: float = 0.015                 # ASSUMED prior (pre-registered, robust/PREREG.txt): |dx|, |dy| <= dxy [m] -- a WORLD bound
    yaw: float = 8.0                   # ASSUMED prior: |yaw| <= yaw [deg] -- a WORLD bound
    mu: tuple = (0.6, 1.2)             # ASSUMED prior: friction scale interval (all geoms)
    mass: tuple = (0.0, 0.4)           # ASSUMED prior: hidden mass interval [kg], anywhere in the pan
    payload: str = "empty"             # what FL-MPC is told: "empty" (pan + lid) | "known" (probe-measured total)
    assumptions: tuple = ()            # assumed (unverified) steps that shaped this box
    observed: str | None = None        # "built" observe_pose ran before this point: its noise level (None = not observed)

    def contains(self, th, tol=1e-9):
        return (abs(th.get("dx", 0.0)) <= self.dxy + tol and abs(th.get("dy", 0.0)) <= self.dxy + tol
                and abs(th.get("yaw_deg", 0.0)) <= self.yaw + tol
                and self.mu[0] - tol <= th.get("mu_scale", 1.0) <= self.mu[1] + tol
                and self.mass[0] - tol <= th.get("hidden_mass", 0.0) <= self.mass[1] + tol)

    def vertices(self):
        """2^5 corners; hidden mass at the far centre (largest moment about the left claw), none when the mass is 0."""
        out = []
        for sx, sy, sw, mu, mh in itertools.product((-1, 1), (-1, 1), (-1, 1), self.mu, self.mass):
            out.append(dict(dx=sx * self.dxy, dy=sy * self.dxy, yaw_deg=sw * self.yaw, mu_scale=mu, hidden_mass=mh,
                            placement="far_centre" if mh > 0 else "none"))
        return out

    def widen(self, dxy=None, yaw=None, mu=None, mass=None):
        """union with a proposed box: may only grow (narrowing proposals are ignored by the caller and logged)."""
        b = self
        if dxy is not None:
            b = replace(b, dxy=max(b.dxy, float(dxy)))
        if yaw is not None:
            b = replace(b, yaw=max(b.yaw, float(yaw)))
        if mu is not None:
            b = replace(b, mu=(min(b.mu[0], float(mu[0])), max(b.mu[1], float(mu[1]))))
        if mass is not None:
            b = replace(b, mass=(min(b.mass[0], float(mass[0])), max(b.mass[1], float(mass[1]))))
        return b

    def describe(self):
        return dict(dxy_m=self.dxy, yaw_deg=self.yaw, mu_scale=list(self.mu), hidden_mass_kg=list(self.mass),
                    payload=self.payload, assumptions=list(self.assumptions), observed=self.observed)


PREREG = Belief()                                   # the pre-registered robustness set (robust/PREREG.txt)
OVERLOAD = Belief(mass=(0.0, 0.8))                  # ASSUMED prior: contents up to 0.8 kg


@dataclass(frozen=True)
class Act:
    name: str
    arm: str | None = None
    part: str | None = None
    frac: float | None = None
    push: float | None = None

    def __str__(self):
        a = [x for x in (self.arm, self.part) if x]
        if self.frac is not None:
            a.append(f"frac={self.frac}")
        if self.push is not None:
            a.append(f"push={self.push}")
        return f"{self.name}({', '.join(a)})"

    def params(self):
        return {k: v for k, v in dict(arm=self.arm, part=self.part, frac=self.frac, push=self.push).items() if v is not None}


SKILLS = {   # name -> (signature, precondition text, effect text): also the skill library shown to the proposer
    "observe_pose": ("observe_pose()", "pan on stove, nothing lifted", "pose_refined (smaller pose error)"),
    "pinch": ("pinch(arm in {L,R}, part in {handle, lid}, frac)", "arm free, pan on stove", "held_<arm>_<part>"),
    "push": ("push(dist)", "L holds the handle, R free, pan on stove", "overhang (far rim hangs over the stove edge)"),
    "clamp_rim": ("clamp_rim()", "overhang, R free", "held_R_rim, lid_secured (R clamps lid + wall + bottom)"),
    "probe_mass": ("probe_mass()", "L holds the handle, R free, pan on stove", "mass_known (tip-up about the far rim)"),
    "lift": ("lift()", "at least one arm holds a part of the pan", "lifted"),
}
ATOMS = ("on_stove", "free_L", "free_R", "held_L_handle", "held_L_lid", "held_R_handle", "held_R_lid", "held_R_rim",
         "overhang", "lid_secured", "pose_refined", "mass_known", "lifted")
PARTS = ("handle", "lid")
GOAL = "lifted"
INIT = frozenset({"on_stove", "free_L", "free_R"})


def actions(state):
    """applicable ground actions (symbolic preconditions only), in a fixed prior order."""
    s = state
    if "lifted" in s:
        return []
    out = []
    if "pose_refined" not in s:
        out.append(Act("observe_pose"))
    for arm in ("L", "R"):
        if f"free_{arm}" in s:
            if arm == "L":
                out += [Act("pinch", "L", "handle", frac=f) for f in L_FRACS]
            else:
                out.append(Act("pinch", "R", "handle", frac=R_HANDLE_FRAC))
            out.append(Act("pinch", arm, "lid"))
    if "held_L_handle" in s and "free_R" in s and "overhang" not in s:
        out += [Act("push", push=d) for d in PUSHES]
    if "overhang" in s and "free_R" in s:
        out.append(Act("clamp_rim", "R", "rim"))
    if "held_L_handle" in s and "free_R" in s and "mass_known" not in s:
        out.append(Act("probe_mass"))
    if any(a.startswith("held_") for a in s):
        out.append(Act("lift"))
    return out


def apply(state, belief, act):
    """symbolic effect + belief update.  Returns (state', belief')."""
    s = set(state)
    b = belief
    s.add(f"did_{act.name}")
    if act.name == "observe_pose":
        s.add("pose_refined")
        if OBS_MODE == "built":
            if b.observed is None:
                b = replace(b, observed=OBS_NOISE, assumptions=b.assumptions + (
                    f"observe_pose: BUILT (depth fit, noise '{OBS_NOISE}'); claws re-aimed at the estimate; pose box not shrunk",))
        elif b.dxy > OBS_DXY or b.yaw > OBS_YAW:
            b = replace(b, dxy=min(b.dxy, OBS_DXY), yaw=min(b.yaw, OBS_YAW),
                        assumptions=b.assumptions + (f"observe_pose: residual pose error <= {OBS_DXY*1e3:.0f} mm / {OBS_YAW:.0f} deg (ASSUMED, not built)",))
    elif act.name == "pinch":
        s.discard(f"free_{act.arm}")
        s.add(f"held_{act.arm}_{act.part}")
    elif act.name == "push":
        s.add("overhang")
    elif act.name == "clamp_rim":
        s.discard("free_R")
        s |= {"held_R_rim", "lid_secured"}
    elif act.name == "probe_mass":
        s.add("mass_known")
        b = replace(b, payload="known", assumptions=b.assumptions + ("probe_mass: total mass measured exactly (ASSUMED; probe not built for the pan)",))
    elif act.name == "lift":
        s.discard("on_stove")
        s.add("lifted")
    return frozenset(s), b


@dataclass
class Plan:
    acts: list = field(default_factory=list)
    beliefs: list = field(default_factory=list)     # belief BEFORE each act (beliefs[i] is what act i was planned with)
    final: Belief | None = None
    state: frozenset = INIT

    def __str__(self):
        return " -> ".join(str(a) for a in self.acts)

    def holders(self):
        """arm -> part held at the lift."""
        return {a[5]: a[7:] for a in self.state if a.startswith("held_")}

    def act(self, name, **kw):
        for i, a in enumerate(self.acts):
            if a.name == name and all(getattr(a, k) == v for k, v in kw.items()):
                return i, a
        return None, None

    def signature(self):
        """the physical configuration the certificate/sim judge (the symbolic prefix order does not enter)."""
        h = self.holders()
        i, pl = self.act("pinch", arm="L", part="handle")
        _, pu = self.act("push")
        sig = (tuple(sorted(h.items())), pl.frac if pl else None, pu.push if pu else None, GRIP_N,
               self.final.payload if self.final else "empty")
        idx = [k for k, a in enumerate(self.acts) if a.name in ("pinch", "clamp_rim", "push")]
        obs = {self.beliefs[k].observed for k in idx} if idx else {None}
        return sig + (f"obs:{obs.pop()}",) if obs and None not in obs else sig      # 6th field only when every grasp step was observed

    def aim_belief(self):
        """pose box the claws were aimed with: the WIDEST pose belief over the grasp/clamp steps; payload/mass from the lift."""
        idx = [i for i, a in enumerate(self.acts) if a.name in ("pinch", "clamp_rim", "push")]
        dxy = max([self.beliefs[i].dxy for i in idx] or [self.final.dxy])
        yaw = max([self.beliefs[i].yaw for i in idx] or [self.final.yaw])
        return replace(self.final, dxy=dxy, yaw=yaw)
