"""Analytic spill-predictor LADDER for the WM-requirement study (task #28).

QUESTION (user, 2026-09-10): "before [building the] WM I want to explore what is
the limit and how powerful the WM needs to be and what degree of dynamics should
it capture."

METHOD: the WM only owes us the dynamics the ANALYTIC model can't capture. So we
don't guess the WM's power -- we MEASURE the gap between a ladder of increasingly
dynamic analytic spill predictors and the GROUND-TRUTH spill oracle (the 8 loose
balls in the cup, scenes/scene_yor_cup*.xml -> scripts.cup_task_utils.count_spilled).
The rung at which prediction accuracy SATURATES is exactly the "degree of dynamics"
the WM must capture, and no more.

THE LADDER (each rung adds one order of dynamics; all take the CUP kinematics only
-- pose + linear acceleration of the cup body over time -- which is all the robot
could ever know analytically, since it has no fluid/ball sensor):

  rung 0  STATIC TILT      : spill risk from cup-axis tilt-from-vertical alone.
  rung 1  EFFECTIVE-GRAVITY: spill risk from cup-axis tilt vs the EFFECTIVE gravity
                             up-direction  up_eff = (a_cup - g)/||a_cup - g||
                             (in the cup's accelerating frame the liquid/ball pile
                             feels g_eff = g - a_cup; "up" is -g_eff). Quasi-static:
                             assumes the free surface instantly aligns to up_eff.
  rung 2  + 1st SLOSH MODE : the surface does NOT align instantly -- it is a damped
                             2nd-order oscillator that LAGS and OVERSHOOTS the
                             quasi-static target, so a fast accel transient flings
                             the surface (a ball) past the rim AFTER the motion.
                             sigma'' + 2*zeta*w*sigma' + w^2*sigma = w^2 * sigma_qs
                             sigma = surface-tilt 2-vector (cup frame), sigma_qs =
                             rung-1 quasi-static target as a 2-vector.
                             w^2 = (g/R)*1.84*tanh(1.84*H/R)  (Abramson / NASA SP-106
                             first antisymmetric mode of a cylindrical container).

Rungs 0/1 are MEMORYLESS (functions of the instantaneous pose/accel); rung 2 needs
a 2nd-order latent (the oscillator state). So if rung 2 -- and not rung 1 -- is what
closes the gap to the ball oracle, the WM PROVABLY needs at least a 2nd-order
oscillatory latent driven by cup acceleration, and a purely feedforward / Markov-on-
pose model is insufficient; but it does NOT need a large high-dimensional fluid state
(a full DreamerV3 would be overkill for spill). If rung 2 does NOT close it, the balls
are genuinely higher-D/nonlinear and the WM must be correspondingly richer. Either way
we get a NUMBER, not a vibe.

Everything here is numpy-only and consumes logged trajectories (see collect_ladder.py);
no MuJoCo dependency, so it is cheap to sweep predictor hyper-params (w, zeta, theta).

CAVEAT (honest): the balls are 8 discrete 8mm spheres, NOT a continuum fluid, so the
slosh w/zeta from the fluid formula are ANALOGS/priors, not exact -- collect_ladder.py
scans w and zeta around these priors and reports the best, and we report that these
are surrogate constants. The point of the study is the SHAPE of the accuracy-vs-rung
curve (does adding oscillator dynamics help?), which is robust to the exact w/zeta.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

G_WORLD = np.array([0.0, 0.0, -9.81])   # gravity vector, world frame


@dataclass
class CupConfig:
    """Cup + fill geometry, from scenes/scene_yor_cup.xml (2026-09-10 read):
    cup_bottom cylinder radius 0.030 (minus ~0.0015 wall -> inner ~0.0285);
    12 wall boxes half-height 0.040 at z=0.040 -> rim at z=0.080 above the cup
    body origin (which is the cup's BOTTOM face); 8 balls initially span
    z ~ 0.0085..0.026 above the bottom -> fill height ~0.026."""
    R: float = 0.028          # cup inner radius [m]
    h_rim: float = 0.080      # rim height above cup bottom [m]
    H_fill: float = 0.026     # fill height (top of the settled ball pile) [m]
    zeta: float = 0.15        # slosh damping ratio (loose-ball analog; scanned)
    omega: float | None = None  # slosh natural freq [rad/s]; None -> from formula

    @property
    def theta_max(self) -> float:
        """Static overflow tilt: the tilt at which the free surface on one side
        first reaches the rim. atan((h_rim - H_fill)/R). For this tall,
        ~1/3-full cup this is LARGE (~63 deg), which is itself a finding: pure
        static tilt almost never spills -> spills are dynamic."""
        return np.arctan2(self.h_rim - self.H_fill, self.R)

    def omega_slosh(self) -> float:
        if self.omega is not None:
            return self.omega
        # Abramson/NASA SP-106 first antisymmetric mode, cylindrical container.
        k = 1.84
        return float(np.sqrt((9.81 / self.R) * k * np.tanh(k * self.H_fill / self.R)))


def _axis_from_R(Rmat: np.ndarray) -> np.ndarray:
    """Cup axis (world frame) = the cup's local +z mapped to world. Rmat is the
    cup body's 3x3 world rotation (row-major, as data.xmat.reshape(3,3))."""
    return Rmat @ np.array([0.0, 0.0, 1.0])


def _tilt_vec_between(n_cup: np.ndarray, up_ref: np.ndarray) -> np.ndarray:
    """Small-rotation 2-vector (about world x,y) taking up_ref onto n_cup, i.e.
    the tilt of the cup axis away from the reference "up" direction. Returns
    (tx, ty); its norm ~ the tilt ANGLE for small tilts, and it is exact in the
    sense that arcsin(norm of the full cross product) is the true angle -- we
    return both via tilt_angle() below. Sign/axis convention matches
    mpc_fl.FLMPC.step's cup-upright rows (omega_x=-b_y, omega_y=b_x)."""
    n = n_cup / (np.linalg.norm(n_cup) + 1e-12)
    u = up_ref / (np.linalg.norm(up_ref) + 1e-12)
    # rotation vector (world) that maps u onto n is ~ u x n (small angle); take
    # its world x,y components as the 2-vector tilt.
    w = np.cross(u, n)
    return w[:2]


def tilt_angle(n_cup: np.ndarray, up_ref: np.ndarray) -> float:
    n = n_cup / (np.linalg.norm(n_cup) + 1e-12)
    u = up_ref / (np.linalg.norm(up_ref) + 1e-12)
    return float(np.arccos(np.clip(n @ u, -1.0, 1.0)))


# --------------------------------------------------------------------------
# Trajectory-level predictors. Each takes:
#   n_cup_traj : (T,3) cup axis (world) per step  (from cup xmat)
#   a_cup_traj : (T,3) cup body LINEAR acceleration (world) per step [m/s^2]
#   dt         : timestep [s]
# and returns a per-step SPILL-RISK score in [0, inf) where higher = more likely
# to spill. We use "predicted surface tilt / theta_max" so that score >= 1.0 is
# the analytic spill threshold; the ROC sweep (score_predictor) does not rely on
# the exact threshold, only on the ranking.
# --------------------------------------------------------------------------

def rung0_static(n_cup_traj: np.ndarray, a_cup_traj: np.ndarray, dt: float,
                 cfg: CupConfig) -> np.ndarray:
    up = np.array([0.0, 0.0, 1.0])
    ang = np.array([tilt_angle(n, up) for n in n_cup_traj])
    return ang / cfg.theta_max


def rung1_effgrav(n_cup_traj: np.ndarray, a_cup_traj: np.ndarray, dt: float,
                  cfg: CupConfig) -> np.ndarray:
    scores = np.empty(len(n_cup_traj))
    for t, (n, a) in enumerate(zip(n_cup_traj, a_cup_traj)):
        up_eff = a - G_WORLD                       # = -(g - a) = effective "up"
        scores[t] = tilt_angle(n, up_eff) / cfg.theta_max
    return scores


def rung2_slosh(n_cup_traj: np.ndarray, a_cup_traj: np.ndarray, dt: float,
                cfg: CupConfig) -> np.ndarray:
    """Drive the 1st slosh mode (a damped 2-D oscillator, in the WORLD x,y tilt
    2-vector for simplicity -- the cup stays near-upright so world-xy tilt ~
    cup-frame tilt) toward the rung-1 quasi-static target and read the surface's
    ACTUAL tilt magnitude, which lags+overshoots. score = ||sigma|| / theta_max."""
    w = cfg.omega_slosh()
    z = cfg.zeta
    T = len(n_cup_traj)
    sigma = np.zeros(2)      # surface tilt 2-vector (world x,y)
    sigd = np.zeros(2)
    scores = np.empty(T)
    for t in range(T):
        n = n_cup_traj[t]
        up_eff = a_cup_traj[t] - G_WORLD
        sigma_qs = _tilt_vec_between(n, up_eff)     # quasi-static target (rung 1)
        # semi-implicit Euler on  sig'' = w^2 (sigma_qs - sigma) - 2 z w sig'
        acc = w * w * (sigma_qs - sigma) - 2.0 * z * w * sigd
        sigd = sigd + dt * acc
        sigma = sigma + dt * sigd
        scores[t] = np.linalg.norm(sigma) / cfg.theta_max
    return scores


LADDER = {"rung0_static": rung0_static,
          "rung1_effgrav": rung1_effgrav,
          "rung2_slosh": rung2_slosh}


# --------------------------------------------------------------------------
# Scoring against the ball oracle.
# --------------------------------------------------------------------------

def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank (Mann-Whitney U) formula; robust to ties. labels in
    {0,1}. Returns 0.5 if a class is empty (undefined)."""
    pos = labels == 1
    neg = ~pos
    npos, nneg = int(pos.sum()), int(neg.sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = {}
    start = 0
    for i, c in enumerate(counts):
        avg[i] = (start + 1 + start + c) / 2.0
        start += c
    ranks = np.array([avg[i] for i in inv])
    auc = (ranks[pos].sum() - npos * (npos + 1) / 2.0) / (npos * nneg)
    return float(auc)


def score_predictor(scores: np.ndarray, spill_onset_step: int | None, T: int,
                    lead_window_steps: int) -> dict:
    """Frame the prediction as: 'will a spill BEGIN within the next
    lead_window_steps?'. Build per-step labels = 1 for the lead_window_steps
    steps BEFORE spill onset (the window in which an early warning is useful),
    0 elsewhere; if no spill in the episode, all 0. Report per-episode AUROC and
    the LEAD TIME = how many steps before onset the score first crosses 1.0
    (the analytic threshold), or None if it never does before onset."""
    labels = np.zeros(T, dtype=int)
    lead = None
    if spill_onset_step is not None and spill_onset_step > 0:
        lo = max(0, spill_onset_step - lead_window_steps)
        labels[lo:spill_onset_step] = 1
        crossed = np.where(scores[:spill_onset_step] >= 1.0)[0]
        if len(crossed):
            lead = int(spill_onset_step - crossed[0])
    return dict(auroc=_auroc(scores, labels),
                lead_steps=lead,
                spilled=spill_onset_step is not None,
                max_score=float(np.max(scores)) if T else 0.0)


if __name__ == "__main__":
    # ---- self-test: synthetic trajectories exercising each mechanism ----
    cfg = CupConfig()
    dt = 0.005
    print(f"theta_max = {np.degrees(cfg.theta_max):.1f} deg  "
          f"omega_slosh = {cfg.omega_slosh():.1f} rad/s "
          f"({cfg.omega_slosh()/(2*np.pi):.2f} Hz, period {2*np.pi/cfg.omega_slosh()*1e3:.0f} ms)")
    T = 400
    up = np.tile(np.array([0.0, 0.0, 1.0]), (T, 1))

    # case A: pure static tilt to 40 deg, no accel -> rung0/1 see it, rung2 tracks.
    ang = np.radians(40.0)
    nA = np.tile(np.array([np.sin(ang), 0.0, np.cos(ang)]), (T, 1))
    aA = np.zeros((T, 3))
    # case B: upright cup, a sharp lateral ACCELERATION PULSE then stop -> rung0
    # blind (cup never tilts), rung1 sees only during the pulse, rung2 keeps
    # ringing AFTER the pulse (the transient that flings a ball).
    aB = np.zeros((T, 3))
    aB[100:110, 0] = 30.0      # ~10-step lateral accel spike
    nB = up
    for name, fn in LADDER.items():
        sA = fn(nA, aA, dt, cfg)
        sB = fn(nB, aB, dt, cfg)
        print(f"{name:16s}  A(static40): max {sA.max():.2f} @step{sA.argmax():3d} | "
              f"B(accel-pulse): max {sB.max():.2f} @step{sB.argmax():3d} "
              f"post-pulse(>step110) max {sB[110:].max():.2f}")
