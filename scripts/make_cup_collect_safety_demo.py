#!/usr/bin/env python
"""scripts/make_cup_collect_safety_demo.py

Minimal flagship controller + real demonstration video for the
cube-collection-into-held-cup task on scenes/scene_yor_cup_collect.xml.

Episode (one continuous rollout, REAL torque-controlled dynamics for BOTH
arms the whole time -- see "WHAT IS REAL vs SCRIPTED" below):
  1. LEFT arm grasps the cup at t=0 (eval/scene_setup.init_cup_grasped_state,
     unmodified, called ONCE), then LIFTS it ~11cm off the table via a
     computed-torque joint-space PD reference ramp (own IK-solved lift pose)
     and holds there for the rest of the episode.
  2. RIGHT arm runs one pick -> carry -> deposit cycle for cube_4 (chosen
     because it sits north of the cup, inside the left arm's territory --
     see scenes/scene_yor_cup_collect.xml's own docstring -- so reaching AND
     carrying it both sweep close to the held cup), via an IK-solved
     joint-space waypoint reference (approach/descend/grasp/lift/carry/
     release/retreat), torque-tracked exactly like the left arm.
  3. BOTH arms' torque commands are filtered every physics step through
     their OWN decentralized HOCBF-QP safety filter -- byte-identical
     construction to experiments/ab_smooth_vs_bangbang/run_ab4.py's
     condition "C" (v4-C*, cooperative, ego-priority: LEFT=ego/protected,
     RIGHT=intruder with INTRUDER_EXTRA_MARGIN["C"]=0.03 extra static
     margin, same ALPHA1=ALPHA2=30 class-K gains) -- reusing sim_common.py's
     all_pair_barriers/solve_constrained_box_qp/build_weighted_least_squares_qp
     and run_ab4.py's `_resolve_extra_margin` UNCHANGED. The LEFT (cup) arm's
     own QP is what makes it visibly yield when the RIGHT arm sweeps close --
     that automatic reaction is what this video demonstrates.

WHAT IS REAL vs SCRIPTED (see also the on-screen caption)
  REAL (genuinely computed every step, not scripted):
    - both arms' torque commands (computed-torque PD toward a joint-space
      reference + the cooperative safety filter [v4c: decentralized HOCBF-QP;
      v7c: cooperative MPC-HOCBF, see CHANGE 2 below] + a rate-limited
      low-pass command filter, exactly the run_ab4.py/mpc7.py machinery);
    - the arm-arm clearance h (sim_common.min_pair_distance, the SAME
      corrected ~25-sphere/arm keypoint proxy used to build the QP/MPC
      constraints, not a simplified stand-in for display);
    - count_spilled / count_deposited_cubes (ground-truth geometric
      detectors, scripts/cup_task_utils.py / scripts/cup_collect_task_utils.py);
    - the cube's physical PICKUP and drop: a WELD "magnet" attach (CHANGE 1,
      cup_collect_task_utils.add_cube_weld/activate_weld/deactivate_weld) --
      a real mjEQ_WELD equality constraint (real constraint force, not a
      per-step kinematic teleport) activated when the right gripper reaches
      the cube and deactivated at release, so the cube physically follows
      the gripper (and drops under real gravity/contact once released);
    - whether the safety filter is "deflecting" the cup arm (measured as
      ||tau_applied - clip(tau_task)||, i.e. how much the filter actually
      changed the commanded torque this step, for either controller).
  ALLOWED-SCRIPTED (repo-standard, documented at the top of this task):
    - the RIGHT arm's pick/carry/deposit path is an IK-solved joint-space
      REFERENCE trajectory (not a closed-loop vision policy) -- the arm's
      actual motion tracking that reference is still real torque control +
      safety filtering, exactly like run_ab4.py's intruder already does for
      its own scripted nominal sweep;
    - the cup is kinematically pinned to the left EE every step (the SAME
      reattach_cup_rigidly call eval/scene_setup.py's own harness uses) --
      this is NOT a simplification introduced here, it is how this repo's
      existing cup-grasp technique already works (see reattach_cup_rigidly's
      own docstring: the teleport-to-fingers grasp never makes real contact,
      so without this the cup would sit inert on the table regardless of arm
      motion). The cube, by contrast, is now a real WELD attach (see REAL
      above) -- CHANGE 1 replaced its old scripted kinematic pin.

CHANGE 2 -- controller axis (--controller {v4c,v7c}, default v4c unchanged):
  v4c: the original decentralized cooperative HOCBF-QP (v4-C*, unchanged).
  v7c: once the crossing scenario begins (right arm's track started, left
    arm already settled at its lift-hold pose -- the LEFT arm's own initial
    lift stays on the v4c ramp+QP path; see run_episode()'s INTERFACE NOTE #2
    for the measured reason), BOTH arms are driven every step by ONE call to
    JointMPCv7.step() (experiments/ab_smooth_vs_bangbang/mpc7.py, reused
    UNMODIFIED -- same solver + all_pair_barriers collision barrier as
    run_ab7.py's "MPC-v7"), LEFT=ego/protected, RIGHT=intruder/yields via
    mpc7's own grasp-fragility-scaled priority weights (the MPC-native analog
    of v4-C's fixed INTRUDER_EXTRA_MARGIN). See run_episode()'s docstring for
    the two real interface caveats (#1: mpc7's contact-based fragility signal
    saturates at max-caution in this scene, since the cup is a kinematic
    reattach, not a real finger grasp; #2 above) -- documented, not worked
    around.

Run (GPU 5 only -- GPUs 1-4 busy):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=5 \\
        /common/users/ss5772/miniforge3/envs/mujoco_ur5e/bin/python \\
        /common/users/ss5772/bimanual/scripts/make_cup_collect_safety_demo.py \\
        [--config {geometric_only,geometric_spill}] [--controller {v4c,v7c}] \\
        [--scenario {crossing,spill}] [--no-video] [--baseline-only] [--max-steps N]

--scenario crossing (default, unchanged): the RIGHT-arm pick/carry/deposit crossing described above.
--scenario spill: RIGHT arm just holds home; LEFT arm (holding the cup) does a FAST wrist tilt/transfer
    (eval/scene_setup.transfer_tilt_offset + left_ee_jac_rot_pinv, reused unmodified) that spills the
    cup -- a failure the geometric HOCBF-QP is structurally blind to (no arm-arm proximity involved at
    all: the RIGHT arm never moves). --config geometric_spill's WM veto freezes the LEFT arm's OWN tilt
    clock instead of the right-arm deposit clock (same freeze-clock mechanism, gated on the left tilt).

--config geometric_only  : the original v4-C* HOCBF-QP-only run (measured failure above, unchanged).
--config geometric_spill : (default) + the REAL trained WM spill monitor
    (world_model/wm_semantic_monitor.RealWMSpillMonitor, world_model/logdir/wm_cup_full/{best.pt,
    monitor_head_cup_full.pt}) consulted every --vision-cadence-steps physics steps; its veto FREEZES
    the right-arm deposit reference clock (freeze-clock pattern from eval/run_eval.py) so the cup
    (LEFT) arm's QP never has to deflect violently to dodge a fast deposit crossing. --controller v7c
    forces this OFF (geometric run only, see CHANGE 2).
--controller v4c (default) | v7c : see CHANGE 2 above.

--grasp-mode faked (default, byte-identical to before) | real : REAL friction grasp of the cup (SIM ONLY):
    the left pads physically pinch the cup under grip-force control (control/realgrasp_flmpc.py +
    control/grasp_routines.py), NO reattach_cup_rigidly / carry_balls_with_cup / cup-velocity overwrite;
    the balls ride in the cup under physics from t=0; FL-MPC (--controller flmpc|flmpc_decentB required)
    also drives the LIFT (ego_hold=True, orientation-holding IK lift reference, joint dry-friction
    feedforward); cup barrier keypoints + the cup-upright tilt anchor are recomputed from the ACTUAL cup
    body pose every step; the RIGHT arm's cubes stay WELDED (unchanged). Adds cup_dropped /
    true_cup_tilt_max_deg / grip_slip_max_m / pad_contacts_min metrics. Output files get a "_realgrasp"
    suffix so faked artifacts are never overwritten.

Writes artifacts/cup_collect_safety_demo_{config}[_v7c].mp4 + artifacts/cup_collect_safety_demo_meta_{config}[_v7c].json
(the "_v7c" suffix is only appended for --controller v7c; v4c keeps the original filenames unchanged).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
AB4_DIR = REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
EVAL_DIR = REPO_ROOT / "eval"
for _p in (str(REPO_ROOT), str(AB4_DIR), str(EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sim_common as sc  # noqa: E402  (read-only import)
import run_ab4  # noqa: E402  (read-only import; reuses the v4-C* intruder-margin law)
import mpc7 as mpc7_mod  # noqa: E402  (read-only import; --controller v7c reuses this UNMODIFIED)
import mpc_fl  # noqa: E402  (feedback-linearized cooperative MPC; --controller flmpc)
import mpc_fl_decentralized  # noqa: E402  (Option B: decentralized worst-case FL-MPC; --controller flmpc_decentB)
import ee_speed_limit as ee_speed_limit_mod  # noqa: E402  (EE-SPEED-LIMIT FIX; module aliased to avoid the `ee_speed_limit` bool)
import scene_setup  # noqa: E402  (eval/, read-only import)
import safety_module as safety_mod  # noqa: E402  (eval/, read-only import; SemanticObservation only)
from scripts import cup_collect_task_utils as cct  # noqa: E402  (read-only import)
from teleop import bimanual_teleop as bt  # noqa: E402  (read-only import)
from control import realgrasp_flmpc as rg  # noqa: E402  (--grasp-mode real helpers; measurement-only in faked mode)
from world_model.wm_semantic_monitor import RealWMSpillMonitor, LatentValueShield  # noqa: E402  (real WM; LatentValueShield = the learned-HJ-in-the-loop shield)

ART_DIR = REPO_ROOT / "artifacts"
ART_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Task constants
# ---------------------------------------------------------------------------
# MULTI-CUBE pick-and-place (crossing scenario only -- "spill" never touches
# cubes): N_CUBES cubes are picked from cube_body_ids(model) (cube_0..cube_5),
# ranked farthest-from-the-cup first (same reasoning as the old single-cube
# CUBE_NAME="cube_2" choice: "far on the right arm's own side -- the PICK is
# clear of the cup arm; the CARRY toward p_cross is the transient, dodgeable
# near-crossing" -- cubes north of the cup, e.g. cube_4, sit too close and can
# collide during the PICK itself, not just the carry), kept only if both their
# hover+grasp IK solves are clean (< CUBE_IK_ERR_MAX_M). See
# build_episode/select_crossing_cubes below.
N_CUBES = 6                   # ALL 6 cubes (cube_0..cube_5) -- select_crossing_cubes below IK-filters
                              # and returns however many are cleanly reachable (<=6), and build_episode's
                              # per-cube waypoint/weld/phase generation is a generic loop over whatever it
                              # returns, so raising this from the old N_CUBES=3 is the whole change needed
                              # to make the demo collect the full scatter instead of a subset.
CUBE_IK_ERR_MAX_M = 0.006     # 6mm -- IK-reachability cutoff for cube selection
_CUBE_OVERRIDE = None         # set by --cubes (list of names) to force a specific cube set,
                              # e.g. ["cube_3","cube_4","cube_5"] -- still IK-filtered, order kept.
CUBE_PLACE_SPACING_M = 0.08   # spacing [m] between adjacent place spots in the center cluster grid (see
                              # _place_cluster_offsets below) -- lowered from the old single-row 0.06 so a
                              # full 6-cube 3x2 grid still fits comfortably inside KEEP_ZONE_RADIUS_M.
KEEP_ZONE_RADIUS_M = 0.16     # [m] radius of the center "keep zone": (a) the visual reference circle
                              # drawn on the table (cup_collect_task_utils.add_keep_zone_marker, via
                              # load_yor_cup_collect_model's keep_zone_radius_m below) and (b) the exact
                              # same radius passed into count_placed_cubes' `radius` below, so the drawn
                              # circle and the "is this cube placed" ground truth always agree. Sized to
                              # comfortably contain the full 6-cube place cluster (built around
                              # `place_center`, itself offset ~6cm from the cup's own ground xy -- see
                              # build_episode) even though the marker itself is centered at the cup's
                              # ground xy (an approximation -- see add_keep_zone_marker's docstring), NOT
                              # at place_center's own IK-solved position, which isn't known until after the
                              # real model load. Larger than cup_collect_task_utils.KEEP_ZONE_RADIUS_
                              # DEFAULT_M (0.10) for exactly that slack; count_placed_cubes' own default
                              # (0.14, unrelated/pre-existing) is left untouched.
LIFT_HEIGHT_M = 0.11          # cup lift off the table (in [0.10, 0.12] per spec)
HOVER = 0.14                  # pre-grasp/carry hover clearance
GRASP_Z_OFFSET = 0.020        # STANDOFF: grasp target sits ~2cm above the cube center. The grasp is a
                              # magnet weld (no physical finger pinch needed), so the cube is captured
                              # dangling just below the gripper -- this keeps the whole gripper (base +
                              # fingertips) ABOVE the table surface instead of the wrist body sinking into
                              # it (measured: at the old 0.006 offset the gripper BASE reached z=0.528, ~2cm
                              # below the 0.55 tabletop, during place). See also PLACE_STANDOFF_M below.
PLACE_STANDOFF_M = 0.020      # place target z = table_top_z + this. GENTLER-DROP FIX: lowered from 0.070
                              # (was raised from an even older 0.03 -- see the stale comment this replaces --
                              # back before the TABLE-FLOOR BARRIER FIX below existed, when nothing kept a
                              # low target from actually sinking the gripper). At 0.070 the cube was
                              # RELEASED (see cube_pin_active/REACH_TOL below) ~7cm above the table and
                              # free-fell/bounced/rolled out of the keep zone -- widening cube spacing did
                              # NOT fix this (ruling out crowding), confirming it was drop height + release
                              # timing precision, not collisions between placed cubes. 0.020 sets the
                              # gripper BASE (right_ee_body) down to ~2cm above the table at release --
                              # a set-down, not a drop -- while the TABLE-FLOOR BARRIER FIX's real
                              # collision-proxy-keypoint rows (table_floor=True, default on, wired into
                              # v4c/v7c/flmpc alike -- see run_episode) still hard-stop the actual gripper
                              # (base + fingertip keypoints) from sinking below TABLE_TOP_Z -
                              # TABLE_FLOOR_MARGIN_M regardless of this target, so lowering the target alone
                              # cannot reintroduce the old table-sinking problem. GRASP_Z_OFFSET (the PICKUP
                              # standoff) is untouched -- only the RELEASE waypoint (place_xyzs[i], the sole
                              # consumer of this constant) is lowered.
CUP_HOVER_CLEARANCE = 0.11    # above the cup rim, for the carry-in hover waypoint
CUP_RELEASE_CLEARANCE = 0.045 # above the cup rim, for the release waypoint (real drop from here)

# PLACEMENT-PRECISION FIX (position-triggered release, replacing a pure phase
# clock -- see cube_pin_active/right_gripper_opening below): the cube weld
# used to deactivate at a FIXED TIME (the instant the release_i phase's MOVE
# began -- i.e. before the arm had even started heading toward place_xyzs[i]
# at all), regardless of whether the right EE had actually reached the place
# spot. Under the FL-MPC's safety-constraint deflection the arm is often
# short/wide of the target at that instant, so the cube got released nowhere
# near its place spot and then (compounded by the old 7cm PLACE_STANDOFF_M
# above) bounced/rolled out of the keep zone. Measured: only 3-4/6 cubes
# ended counted in-zone.
REACH_TOL = 0.03               # [m] release once ||right_ee_xy - place_xy_i|| < this (xy only -- z is
                                # governed by PLACE_STANDOFF_M/the table-floor barrier, not this gate)
RELEASE_SPEED_TOL_MS = 0.05    # [m/s] "settling" -- measured (finite-difference) right-EE speed must also
                                # be under this before releasing, so a cube isn't dropped while still
                                # sweeping through the target on its way past it
GRIP_OPEN_RAMP_S = 0.3         # [s] visual gripper-open ramp duration, timed from the ACTUAL release
                                # instant (not a fixed phase time) -- fits inside release_i's 0.35s hold

# left-track (cup lift) timing
T_LIFT_MOVE = 1.2
T_LIFT_HOLD = 0.3
RIGHT_START_T = T_LIFT_MOVE + T_LIFT_HOLD + 0.1   # 1.6s: right arm waits for the cup to be lifted+settled

# right-track (pick/carry/deposit) phase durations: (name, duration, hold)
RIGHT_PHASES = [
    ("hover_cube", 1.2, 0.10),
    ("at_cube",    1.0, 0.20),
    ("lift",       1.0, 0.10),
    ("carry",      2.0, 0.20),   # the crossing -- AGGRESSIVE pace for the WM-helps contrast: fast
                                 # enough that the geometric-only cup-arm dodge jostles/spills the cup,
                                 # so the WM (which slows the approach) has something to prevent
    ("release",    0.7, 0.35),   # RELIABLE-PLACEMENT FIX: hold raised from 0.20 -> 0.35s so a just-
                                 # dropped cube has more real-physics time to settle (stop bouncing/
                                 # sliding) INSIDE the keep zone before the arm moves on to the next
                                 # cube -- see count_placed_cubes' own vel_tol/settled check. Pure
                                 # timing change; the release WAYPOINT itself (place_xyzs[i], standoff
                                 # etc.) is unchanged.
    ("park",       1.0, 0.30),
]
SETTLE_TAIL_S = 1.2   # extra time at the end to let the drop/settle play out

DURATION_S = RIGHT_START_T + sum(d + h for _, d, h in RIGHT_PHASES) + SETTLE_TAIL_S
N_STEPS = int(round(DURATION_S / sc.DT))

SCENARIO_CHOICES = ("crossing", "spill")

# --- "spill" scenario timing (--scenario spill; "crossing" above is unchanged) ---
# LEFT arm: grasp+lift (same track_l ramp+hold as "crossing"), then a FAST
# TILT/TRANSFER of the cup EE -- reusing eval/scene_setup.py's
# transfer_tilt_offset/left_ee_jac_rot_pinv (the SAME validated
# tilt-through-the-grasp mechanism eval/run_eval.py's spill scenario uses,
# just parked on the LEFT arm here instead of run_eval's single-arm setup).
# RIGHT arm just holds home the whole episode (no pick/carry/deposit) --
# this scenario is about a payload failure the geometric HOCBF-QP cannot see
# at all (no arm-arm proximity is involved), not about a crossing.
LEFT_TILT_START_T = RIGHT_START_T   # tilt begins once the lift is settled (same settle time as "crossing")
SPILL_TILT_DUR_S = 10.0             # quintic ramp 0->amp, UN-VETOED elapsed time to complete. TUNED
                                     # empirically (see scripts/make_cup_collect_safety_demo.py's own
                                     # tuning notes / task report): this scene's cup tips at a FIXED
                                     # physical wrist angle of ~55 deg regardless of amp/dur_s (measured
                                     # directly by sweeping amp*quintic(s) at the moment count_spilled
                                     # first fires). geometric_only never vetoes -> reaches the full
                                     # SPILL_TILT_AMP_DEG (well past 55 deg) comfortably inside the
                                     # episode -> spills all 8 balls. The retrained WM vetoes ~35-40% of
                                     # steps once risk rises (empirically, on THIS trajectory) -> the
                                     # un-vetoed elapsed clock only reaches ~75-80% of SPILL_TILT_DUR_S by
                                     # episode end -- at SPILL_TILT_AMP_DEG=65 that keeps the ACTUAL wrist
                                     # angle mostly under the ~55 deg tip-over point, so most balls never
                                     # spill (a short dur_s does NOT work: the veto only engages once risk
                                     # has already risen, so a fast ramp blows through the tip-over angle
                                     # before enough un-vetoed-time accounting can matter -- see the sweep
                                     # in the task report for the amp/dur_s grid that motivated this).
SPILL_TILT_HOLD_S = 1.0             # hold at full tilt (unvetoed) so spilled beads finish leaving/landing
SPILL_TILT_AMP_DEG = 65.0           # amplitude [deg] about TRANSFER_ROT_AXIS (world-X) -- comfortably past
                                     # this scene's measured ~55 deg tip-over angle when the ramp completes
                                     # (geometric_only), but under it for the WM-vetoed partial ramp above.
SPILL_SETTLE_TAIL_S = 1.5           # extra tail so a late/partial spill finishes landing before video ends
SPILL_DURATION_S = LEFT_TILT_START_T + SPILL_TILT_DUR_S + SPILL_TILT_HOLD_S + SPILL_SETTLE_TAIL_S
SPILL_N_STEPS = int(round(SPILL_DURATION_S / sc.DT))

ENGAGE_H_M = 0.05      # HUD "engaged" threshold, chosen for visualization (see make_safety_demo.py precedent)
QP_DEFLECT_NM = 0.5    # || tau_qp - clip(tau_task) || threshold [Nm] to call the left QP "actively deflecting"
INFEAS_BRAKE_KD = 8.0  # velocity-damping gain for the QP-infeasibility recovery brake (Nm per rad/s)

# V4C RETUNE KNOBS (2026-09-25, --controller v4c ONLY; all None/"torque" = the
# original v4-C* reactive branch, byte-identical). Set per job by
# experiments/ab_smooth_vs_bangbang/comprehensive_eval.py (--v4c-* flags / V4C_* env).
# The ego (left) class-K gains are sim_common.ALPHA1/ALPHA2 and the keep-out margin
# sim_common.D_MIN (both read live); these cover what those do not reach:
SELECT_D_MIN_REF = 0.03     # [m] nominal keep-out margin the F2 cube SELECTION uses (== sc.D_MIN default)
V4C_INT_ALPHA = None       # intruder (right) HOCBF class-K gain; None -> run_ab4.INTRUDER_ALPHA["C"] (30)
V4C_TABLE_ALPHA = None      # table-floor HOCBF class-K gain (v4c rows); None -> the def-time sc.ALPHA1/2 (30)
V4C_QP_METRIC = "torque"    # "torque": min||tau - tau_task||^2 (original) | "accel": min||M^-1(tau - tau_task)||^2
V4C_ANTIWINDUP_M = None     # [m] clamp the right-arm PD reference so its EE error <= this (F1 analog); None = off
V4C_INFEAS_BRAKE_KD = None  # QP-infeasibility brake damping for v4c; None -> INFEAS_BRAKE_KD
V4C_SPEEDCAP_DIR = False    # True: add FL-MPC's motion-direction speed row (u . v_next <= V_EE_MAX_MS) to v4c's
                            # per-axis EE speed box (which alone lets the speed NORM reach sqrt(3)*V)
V4C_FILTER_MODE = "post"    # "post": torque filter AFTER the QP (original) | "pre": low-pass the NOMINAL,
                            # QP last with the slew limit as a box around the last applied torque
V4C_TORQUE_FILTER = None    # (lowpass_alpha, rate_limit_nm_s) for v4c's torque command filter; None -> sc.FILTER_*

# EE-SPEED-LIMIT FIX (--ee-speed-limit {on,off}, default on): neither the
# reactive HOCBF-QP (v4c) nor the MPC-HOCBF (v7c) bounded how fast an
# end-effector may move -- only arm-arm DISTANCE. MEASURED on this crossing:
# the RIGHT arm's scripted return-to-home ("park") reaches ~0.79 m/s at the
# EE (release ~0.40, hover ~0.36), while the actual crossing ("carry") is only
# ~0.24 m/s -- i.e. the "EE moves too fast" complaint is the fast release->park
# return, the SAME segment where the arm sinks ~3.4cm into the table. Two
# coordinated parts, both keyed off ONE Cartesian speed cap V_EE_MAX_MS:
#   (1) REFERENCE time-scaling (build_episode): each right-arm phase's move
#       duration is stretched so the scripted EE speed never exceeds the cap
#       -- computed from the phase's actual Cartesian distance, NOT hand-picked
#       durations. Phases already under the cap (carry) are UNCHANGED, so the
#       crossing dynamics/spill-contrast are preserved; only park/release/hover
#       are calmed.
#   (2) HARD in-loop cap (ee_speed_limit.ee_velocity_cap_rows): 6 extra box-QP
#       rows bounding the next-step |v_ee_axis| <= V_EE_MAX_MS on the arm doing
#       the fast motion (the RIGHT/intruder arm), so even a violent safety
#       dodge cannot exceed the cap. Wired into BOTH controllers' RIGHT-arm QP
#       (v4c reactive rows; v7c/mpc7 via JointMPCv7's v_ee_max). part (1) keeps
#       the nominal motion off the constraint; part (2) is the guarantee.
V_EE_MAX_MS = 0.28     # [m/s] Cartesian EE speed cap (leaves carry ~0.24 untouched; tames park 0.79)

# FL-MPC ABLATION OVERRIDES (workshop-paper knobs, default-OFF -- None means
# byte-identical to the prior fl_kwargs = dict(NC=4, L=8, W_POS_EGO=800.0)
# behavior). Set by experiments/ab_smooth_vs_bangbang/comprehensive_eval.py's
# --variants BEFORE build_episode, read once at FLMPC construction below.
#   FL_W_JERK_OVERRIDE:    "bang-bang" ablation -- 0.0 disables mpc_fl.FLMPC's
#                          jerk regularization (W_JERK, isolates the SMOOTH claim).
#   FL_W_POS_INT_OVERRIDE: "unilateral" ablation -- a large value (e.g. 1e6)
#                          makes the intruder/right arm rigidly track its task
#                          reference, so it will NOT deviate to help avoid
#                          collisions -- only the ego/left arm can yield
#                          (isolates the COOPERATIVE claim: the shared QP can
#                          no longer spend the intruder's accel on avoidance).
FL_W_JERK_OVERRIDE = None
FL_W_POS_INT_OVERRIDE = None

# COOPERATIVE YIELD BALANCE (both arms move): v4-C* gives the collecting arm a
# +0.03m extra collision margin (run_ab4 INTRUDER_EXTRA_MARGIN["C"]) so it
# yields FIRST and FULLY -- which is why the cup arm never had to move (its own
# collision constraint never activated: the collecting arm resolved the crossing
# before h got tight enough for the ego). Setting the demo's intruder margin to
# 0 makes the balance SYMMETRIC: both arms see the same clearance, so both
# constraints activate together and BOTH visibly yield around each other (the
# cooperative counter-planning we want to show). This is a DEMO-LOCAL override
# (run_ab4 / LEG A's fixed 0.03 is untouched). 0.0 = symmetric; raise toward
# 0.03 to shift more of the dodge onto the collecting arm again.
COOP_INTRUDER_MARGIN = 0.0

# ---------------------------------------------------------------------------
# CUP-IN-BARRIER FIX: the cooperative HOCBF-QP barrier (sim_common.
# all_pair_barriers) is arm-arm ONLY -- LEFT-arm-link keypoints vs
# RIGHT-arm-link keypoints. The held cup is rigidly carried by the LEFT
# gripper but was never itself a keypoint, and the arm collision-PROXY geoms
# are deliberately masked off from the cup geoms (arm<->cup PHYSICS must stay
# off -- a real contact would knock/spill the cup), so the RIGHT arm's return/
# park leg had no avoidance signal at all w.r.t. the cup and could sweep
# straight through it.
#
# Fix (additive, no new barrier type): add a few CUP keypoints -- a rim ring
# + 2 body points, defined ONCE in the cup's own local frame (bottom-center
# at local z=0, per cup_collect_task_utils.CUP_RADIUS/CUP_HEIGHT) and
# reprojected into the LEFT EE's local frame via the SAME fixed rigid
# cup<-EE offset (cup_rel_pos0/cup_rel_quat0, scene_setup.cup_rel_pose) that
# scene_setup.reattach_cup_rigidly already re-applies every step -- to the
# LEFT arm's barrier keypoint set, via sim_common.all_pair_barriers'
# `extra_ego_keypoints` (see sim_common.py diff). all_pair_barriers'
# EXISTING left-keypoints-vs-right-keypoints pairing then AUTOMATICALLY
# produces "right arm vs cup" HOCBF rows, on both arms' QPs (and mpc7, which
# consumes the same `pairs`), with zero new physics and zero new QP math.
# ---------------------------------------------------------------------------
CUP_KP_RADIUS = 0.004   # [m] proxy sphere radius per cup keypoint -- close to the
                        # cup wall's own physical half-thickness (~0.00075m) + a
                        # small buffer, NOT inflated further: sim_common.D_MIN
                        # (0.03m, unchanged/shared with every other keypoint
                        # pair in the barrier) already supplies the intended
                        # safety margin on top of this, so this radius should
                        # track the wall's real geometry, not add a second,
                        # redundant buffer that would needlessly force a
                        # harder avoidance reaction than the actual physical
                        # geometry requires.
CUP_RIM_N = 6           # keypoints per WALL ring, evenly spaced azimuthally.
# THREE wall rings (bottom/mid/rim), all at the cup's own wall radius
# (cup_task_utils.CUP_RADIUS), not on-axis: an on-axis-only "body point" (as
# first tried here) sits at radius 0, so it under-represents exactly the
# case that matters -- the demo's own p_cross crossing target aims ~4cm
# laterally from the cup's AXIS near its BOTTOM height (see build_episode),
# i.e. close to the WALL there, not the axis -- an on-axis point measures
# that as a full 4cm clearance when the real wall is only ~1cm away. Rings
# at 3 heights (not just the top rim) keep the whole cylindrical wall height
# covered by at least one nearby keypoint.
CUP_RING_Z_FRACS = (0.0125, 0.5, 1.0)  # bottom / mid / rim, as a fraction of cct.CUP_HEIGHT (resolved below,
                                        # not hardcoded, so it always tracks cup_collect_task_utils' own constant)

# ---------------------------------------------------------------------------
# TABLE-FLOOR BARRIER FIX (--table-floor {on,off}, default on): arm<->table
# COLLISION PHYSICS is deliberately masked off (sim_common._patch_real_
# collision_geoms only ever turns on left<->right bits, scenes/
# scene_yor_cup_collect.xml's table_top geom keeps its shipped contype/
# conaffinity), and the cooperative HOCBF-QP barrier (sim_common.
# all_pair_barriers) is arm-arm (+held-cup, see the CUP-IN-BARRIER FIX above)
# ONLY -- nothing in either arm's QP has ever known the tabletop exists, so a
# link/gripper commanded low enough (e.g. the right arm's cube descend/grasp)
# can sink its collision-proxy keypoints below the table surface with zero
# resistance. MEASURED (see the task's measurement run, --table-floor off):
# a real, non-trivial dip during the RIGHT arm's "at_cube" (descend/grasp)
# phase -- see the task report for the exact minimum and magnitude.
#
# Fix: for EACH arm, add one HOCBF-QP row per that arm's OWN real
# collision-proxy keypoint (sim_common.barrier_keypoints(model, EGO_PREFIX /
# INT_PREFIX) -- the SAME ~25-keypoint set already used for the arm-arm
# barrier, group==3 fitted-capsule/box spheres) against a STATIC horizontal
# half-space floor at world z = TABLE_TOP_Z - TABLE_FLOOR_MARGIN_M, via
# sim_common.table_plane_barrier_rows (SAME relative-degree-2 HOCBF cascade +
# box-QP row convention as the arm-arm rows below, just simpler since the
# plane's normal is constant and the plane itself is static -- see that
# function's own docstring). Purely additive: appended as EXTRA rows onto
# each arm's EXISTING a_ineq/b_ineq (built from `pairs` below) right before
# the box-QP solve -- the arm-arm row math itself is untouched. Only wired
# into the v4c (decentralized box-QP) branch below, since that is the branch
# this fix's task targets and the one LEG A (run_ab4.py/run_sweep.py) never
# calls; --controller v7c (mpc7.JointMPCv7, reused unmodified) does not
# consume this and is unaffected either way.
# ---------------------------------------------------------------------------
TABLE_TOP_GEOM_NAME = "table_top"
TABLE_FLOOR_MARGIN_M = 0.006   # [m] enforced threshold = TABLE_TOP_Z - this margin, i.e. a keypoint may sit
                                # up to 6mm below the table's own z=0.55 surface before the barrier engages --
                                # tuned so the right gripper's finger-pad keypoints (smallest radius, ~0.008m,
                                # see sim_common.barrier_keypoints' fitted radii) can still reach cube-grasp
                                # depth (cube centers ~z=0.56, so the fingertip/pad keypoints legitimately
                                # travel to within ~1cm of the table surface during "at_cube") while every
                                # OTHER (larger) link keypoint -- which never needs to get anywhere near this
                                # low -- stays comfortably clear; see the task report's ON-vs-OFF min-z numbers.


def _rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate 3-vector v by quaternion q=[w,x,y,z] (MuJoCo convention), via
    the sandwich product q*(0,v)*conj(q) using the quat_mul/quat_conj already
    defined above (no new quaternion machinery)."""
    qv = np.array([0.0, v[0], v[1], v[2]])
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def build_cup_barrier_keypoints(left_ee_body: int, cup_rel_pos0: np.ndarray,
                                 cup_rel_quat0: np.ndarray) -> list:
    """CUP keypoints (3 wall rings -- bottom/mid/rim -- + a bottom-center
    point for the solid base disc) as (body_id, local_offset(3,), radius)
    tuples -- the EXACT format sim_common.barrier_keypoints/
    _body_keypoints already produce for ordinary arm-link keypoints, so
    sim_common.all_pair_barriers' `extra_ego_keypoints` can append them with
    ZERO new Jacobian/assembly code.

    body_id is the LEFT EE (`left_ee_body`), NOT the cup body: the cup is
    only KINEMATICALLY re-pinned to the EE each step (scene_setup.
    reattach_cup_rigidly), not a real weld/joint, so its own free-joint dofs
    are not among left_dofadr -- a Jacobian taken at the cup body would be
    structurally zero w.r.t. the left arm's dofs (mj_jac only propagates
    through the actual kinematic tree). Since the cup is kept rigidly
    pinned to the EE frame at the FIXED offset (cup_rel_pos0, cup_rel_quat0)
    every step, a point given as a fixed EE-local offset moves EXACTLY like
    the corresponding cup point, and mj_jac at the EE correctly propagates
    the left arm's joint motion into that point's world velocity for the
    HOCBF psi1/psi2 terms -- unlike a cup-body Jacobian, this is actually
    correct, not just convenient.
    """
    pts_local = []
    for z_frac in CUP_RING_Z_FRACS:
        z = z_frac * cct.CUP_HEIGHT
        for i in range(CUP_RIM_N):
            ang = 2.0 * np.pi * i / CUP_RIM_N
            pts_local.append(np.array([cct.CUP_RADIUS * np.cos(ang),
                                        cct.CUP_RADIUS * np.sin(ang),
                                        z]))
    pts_local.append(np.array([0.0, 0.0, 0.0]))   # bottom-center (solid base disc)

    kps = []
    for p_local in pts_local:
        offset_ee = cup_rel_pos0 + _rotate_vec(cup_rel_quat0, p_local)
        kps.append((left_ee_body, offset_ee, CUP_KP_RADIUS))
    return kps


def _kp_world_z(data, body_id: int, offset: np.ndarray) -> float:
    """World-z of a single (body_id, local_offset) collision-proxy keypoint
    -- the SAME body_id/xmat/xpos convention sim_common._kp_world uses, just
    z-only (this module's own measurement/verification metric, TABLE-FLOOR
    BARRIER FIX)."""
    return float(data.xpos[body_id][2] + (data.xmat[body_id].reshape(3, 3) @ offset)[2])


def right_arm_to_cup_clearance(model, data, cup_kps: list, right_kps: list, left_ee_body: int) -> float:
    """Min surface-to-surface clearance (dist - r_cup - r_right) over every
    (cup keypoint, right-arm keypoint) pair, at the CURRENT data. Verification/
    logging helper only -- NOT used by the QP itself (that reads `pairs` from
    sim_common.all_pair_barriers, which already includes the cup keypoints
    when extra_ego_keypoints is passed); this exists so the fix's effect can
    be measured directly (min right-arm<->cup distance) independent of the
    QP's own D_MIN-shifted `h`."""
    R_ee = data.xmat[left_ee_body].reshape(3, 3)
    p_ee = data.xpos[left_ee_body]
    cup_pts = np.array([p_ee + R_ee @ off for (_bid, off, _r) in cup_kps])
    cup_r = np.array([r for (_bid, _off, r) in cup_kps])
    best = np.inf
    for bid, off, r in right_kps:
        p = data.xpos[bid] + data.xmat[bid].reshape(3, 3) @ off
        d = np.linalg.norm(cup_pts - p[None, :], axis=1) - cup_r - r
        m = float(d.min())
        if m < best:
            best = m
    return best

# ---------------------------------------------------------------------------
# Spill-aware semantic layer (config="geometric_spill"): the REAL trained WM
# spill monitor (world_model/wm_semantic_monitor.RealWMSpillMonitor -- same
# checkpoints/threshold eval/'s run_wm_eval_harness.py and
# scripts/make_safety_demo.py already use), consulted at a VISION CADENCE
# (camera rate << 200 Hz control rate) and used to FREEZE the right-arm
# deposit reference clock while spill risk is high (freeze-clock pattern
# from eval/run_eval.py's `transfer_t_elapsed`, avoids a catch-up snap).
# ---------------------------------------------------------------------------
# Retrained ON the cup-collect scene (was wm_cup_full, which was OOD here and
# saturated risk~1.0). cup_collect holdout: spill AUC 0.999, collision AUC 0.992.
WM_CHECKPOINT = str(REPO_ROOT / "world_model" / "logdir" / "wm_cup_collect" / "best.pt")
HEAD_CHECKPOINT = str(REPO_ROOT / "world_model" / "logdir" / "wm_cup_collect" / "monitor_head_cup_collect.pt")
SPILL_RISK_THRESHOLD = 0.5
VISION_CADENCE_STEPS = 20   # WM/monitor consulted every 20 torque steps (~10 Hz @ DT=0.005s / 200 Hz
                            # control) -- the veto is held between updates (see run_episode below).
CONFIG_CHOICES = ("geometric_only", "geometric_spill")

# --monitor axis (additive, orthogonal to --config): which SOURCE backs the
# geometric_spill config's veto. "classifier" (default, UNCHANGED) = the
# original RealWMSpillMonitor instantaneous P(spill) head. "value" = the
# NEW LatentValueShield -- world_model/wm_semantic_monitor.LatentValueShield,
# the trained latent HJ value V_theta(z) (world_model/
# train_latent_safety_value.py), veto'ing anticipatorily at V(z) < V_THRESH
# (a small POSITIVE margin above the V=0 safe/unsafe boundary) instead of
# waiting for the instantaneous classifier to cross risk_threshold.
VALUE_CHECKPOINT = str(REPO_ROOT / "world_model" / "logdir" / "wm_cup_collect" / "latent_safety_value.pt")
VALUE_V_THRESH = 0.1
MONITOR_CHOICES = ("classifier", "value")

# ---------------------------------------------------------------------------
# Controller axis (CHANGE 2): v4c = the original decentralized cooperative
# HOCBF-QP (v4-C*, unchanged, default). v7c = BOTH arms driven by the
# cooperative MPC-HOCBF from experiments/ab_smooth_vs_bangbang/mpc7.py
# (JointMPCv7, reused UNMODIFIED -- see run_ab7.py's "MPC-v7" wiring), with
# LEFT (cup) = ego/protected, RIGHT = intruder/yields, exactly mpc7's own
# grasp-fragility-scaled ego-priority mechanism (the MPC-native analog of
# v4-C's fixed INTRUDER_EXTRA_MARGIN -- see the --controller v7c interface
# note in run_episode below for the one real caveat: this scene's cup is
# kinematically reattached, not really gripped, so mpc7's contact-based
# fragility signal saturates at 1.0/max-caution permanently rather than
# being state-adaptive).
CONTROLLER_CHOICES = ("v4c", "v7c", "flmpc", "flmpc_decentB")

# --controller-impl {py,rt,rt_pipe} (2026-09-29; experiments/realtime/rt_flmpc.py, rt_pipe.py).
# LIBRARY default "py" = mpc_fl.FLMPC exactly as before (byte-identical). --controller flmpc only:
#   "rt"      = rt_flmpc.RTFLMPC: the SAME QP (rows/costs/limits/fallback semantics), built and
#               solved by ONE compiled, allocation-free, iteration-capped Numba entry point
#               (Goldfarb-Idnani dual active set) on a ROBOT-ONLY dynamics source (M, c,
#               Jacobians, keypoints from a robot-only MjModel synced from the joint state).
#   "rt_pipe" = rt + a pipelined/anytime solver thread (plan for step k+1 computed during step k
#               from the FL-predicted state; deadline misses reuse the shifted previous plan with
#               execution-time torque-box + barrier re-checks and a plan-age cap).
CONTROLLER_IMPL_CHOICES = ("py", "rt", "rt_pipe")
CONTROLLER_IMPL = "py"
RT_PIPE_OPTS: dict = {}   # rt_pipe.PipelinedRTFLMPC kwargs (budget_ms, age_max, inject_p, inject_ms, ...)


def _make_rt_flmpc(model, fl_kwargs, impl):
    """--controller-impl rt / rt_pipe controller (lazy import: py never touches experiments/realtime)."""
    _rt_dir = str(Path(__file__).resolve().parents[1] / "experiments" / "realtime")
    if _rt_dir not in sys.path:
        sys.path.insert(0, _rt_dir)
    if impl == "rt_pipe":
        import rt_pipe  # noqa: E402
        return rt_pipe.PipelinedRTFLMPC(model, **dict(fl_kwargs, **RT_PIPE_OPTS))
    import rt_flmpc  # noqa: E402
    return rt_flmpc.RTFLMPC(model, **fl_kwargs)

# --controller flmpc CUP-ARM YIELD REFINEMENT (see mpc_fl.py's cup-upright
# tilt rows + new cup-arm displacement-leash rows): rather than gluing the
# ego (cup) arm to its hold pose via a crude high W_POS_EGO, the flmpc branch
# below lowers W_POS_EGO and instead bounds the cup EE's Cartesian
# displacement from the hold pose with an explicit leash CONSTRAINT (so the
# arm can cooperate with collision avoidance -- yield -- within a bounded
# envelope) plus the existing cup-upright tilt constraint (so the cup stays
# vertical while it yields); smoothness stays via W_VEL_EGO/W_JERK, unchanged.
CUP_LEASH_M = 0.08       # [m] cup EE Cartesian-displacement leash radius (per world axis) from the hold pose
CUP_TILT_MAX_RAD = 0.10  # [rad] cup-upright tilt bound (~5.7 deg) while yielding
CUP_CONE_FACETS = 0      # 0 => per-axis small-angle tilt BOX (the demo default). The
                         # EFFECTIVE-GRAVITY CONE (mpc_fl block 3, set >0 e.g. 8) is
                         # IMPLEMENTED and correct but OFF here: a 10-seed cone-vs-box
                         # comparison (2026-09-11) found IDENTICAL spill (4/10 both) and
                         # the cone allows MORE absolute tilt (16 vs 11 deg). Reason: the
                         # cup is kinematically pinned with ZEROED velocity
                         # (scene_setup.reattach_cup_rigidly), so the balls feel geometric
                         # wall-sweep, NOT the cup's true inertial forces -- an effective-
                         # gravity constraint can't help when spill isn't inertial. Enable
                         # the cone once the cup-carry has faithful inertial dynamics (or on
                         # hardware). See memory spill-oracle-lift-loss (fidelity note).


# ---------------------------------------------------------------------------
# --grasp-mode {faked,real} (default faked = byte-identical to before). See the
# module docstring + build_episode/run_episode. REAL = genuine two-pad friction
# pinch of the cup (SIMULATION ONLY), no teleport/weld/velocity-zeroing on the cup.
# ---------------------------------------------------------------------------
GRASP_MODE_CHOICES = ("faked", "real")
REAL_GRIP_FORCE_N = 8.0            # grip-force target [N] (finger forcerange stays +-20 N, never exceeded)
REAL_GRASP_LEVELED = True          # level the jaw axis before closing (LEFT_GRASP_Q's jaw is ~6 deg off
                                   # horizontal -> Stage-1 grasp leaves the cup ~5.4 deg tilted; see
                                   # control/realgrasp_flmpc.grasp_cup_real_leveled). False = Stage-1 exact.
REAL_FRICTION_FF = True            # joint dry-friction feedforward on FL-MPC's LIFT torque (frictionloss
                                   # 0.3 Nm/joint is NOT in FL's c term -> the wrist stalls, EE rolls ~12 deg)
REAL_BALL_JITTER_MAX_M = 0.003     # per-seed ball-layout jitter [m] inside the cup (real mode's seed variation;
                                   # scaled by cup_jitter_m>0, capped here so balls start inside the cup)
REAL_CUP_DROP_NO_CONTACT_STEPS = 20  # cup_dropped if pads lose ALL cup contact for >= this many steps (0.1 s)...
REAL_CUP_DROP_RISE_MIN_M = 0.03      # ...or, once the crossing starts, the cup sits < this above its grasp height
REAL_CUP_CONE_FACETS = None        # real mode only: None => same tilt rows as faked (CUP_CONE_FACETS, the per-axis
                                   # BOX: each world-x/y small-angle component <= CUP_TILT_MAX_RAD, so the L2 tilt can
                                   # reach ~sqrt(2)*5.73 deg). K>0 => mpc_fl's K-facet inner L2 cone (effective-gravity
                                   # up), a TIGHTER bound on the combined tilt. Default keeps faked-vs-real apples-to-apples.


# ---------------------------------------------------------------------------
# --bugfixes {on,off} (CHANGE 2, 2026-09-24). CLI default ON (this script's
# main() and comprehensive_eval.py); LIBRARY default OFF (module BUGFIXES=False,
# so other build_episode/run_episode callers -- collect_ladder, realtime,
# imitation, teleop -- stay byte-identical unless they opt in with
# bugfixes=True). Behavior-changing fixes for verified harness bugs; `off`
# reproduces the pre-fix behavior byte-for-byte (the new METRICS below are
# logged either way):
#   (1) SIDEWAYS CUP INIT (faked mode). Root cause (owner-verified):
#       eval/scene_setup.init_cup_grasped_state applies cup_jitter_m ONLY to
#       old_cup_pos, while the cup always goes to the pad midpoint -- so the
#       balls are shifted by (pad_mid - JITTERED old pos), landing 8-43 mm
#       off-centre inside the cup walls; their push topples the free
#       (unheld) cup, and cup_rel_quat0 then pins it ON ITS SIDE for the whole
#       episode (10/24 seeds at jitter 0.04: 83-97 deg). Fix (demo-local copy,
#       scene_setup.py is not edited): _init_cup_grasped_state_fixed shifts the
#       balls by (pad_mid - TRUE old pos) and keeps per-seed variation as an
#       in-cup ball-layout jitter of <= FAKED_BALL_JITTER_MAX_M; then ASSERTS
#       the pinned cup tilt < CUP_INIT_TILT_ASSERT_DEG. Real mode: a toppled
#       cup (> CUP_INIT_TOPPLED_DEG) is a reported grasp FAILURE.
#       Metric: cup_init_tilt_deg (both modes, fixes on or off).
#   (3) GRIPPER CLOCK: right_gripper_opening() was called with ABSOLUTE t
#       while its `bounds` live on the right-arm track clock t_r (=t-RIGHT_START_T
#       when unfrozen) -> the gripper animation fired RIGHT_START_T (1.6 s) early.
#       Fix: pass t_r (and store release_t on the same clock).
#   (4) WELD PROXIMITY -- MEASURED + GATE AVAILABLE, but NOT part of --bugfixes
#       (off by default, enable with --weld-gate-m 0.02). Measured (flmpc, seeds
#       0/1/2/5, 3 cubes, jitter 0.04): the clock-timed weld attaches with the
#       right grasp site 10.0-12.2 cm from its grasp target (11.6-15.1 cm from the
#       cube). Root cause is NOT the weld: the right arm cannot reach the IK grasp
#       pose -- at that reference pose the gripper-base collision proxy (r=5 cm)
#       sits inside the table-floor barrier and the gripper-base geom physically
#       touches the table/cube, so the arm stops ~12 cm short. Gating at 2 cm
#       therefore turns EVERY pick into a miss (12/12) -- the grasp reference (or
#       the real cube grasp, #51) must be fixed first. With the gate on, the weld
#       only attaches once the site is within the gate of p_at_i, else the cube is
#       a MISSED pick (no long-range weld).
#   (6) FL-MPC JOINT-FRICTION STALL (found while measuring (4)): every arm joint
#       has frictionloss=0.3 Nm (unmodified YOR model), a MuJoCo CONSTRAINT force
#       that FL-MPC's tau = M a + c never cancels. With the arms' built-in gravcomp
#       the planned M a on the light wrist joints is < 0.3 Nm, so under --controller
#       flmpc the RIGHT wrist (joints 4-6) essentially never leaves q_home: the
#       gripper arrives mis-oriented, its base jams on the table/cube 10-12 cm from
#       the grasp target, and the clock-timed weld then yanks the cube in from
#       there (measured, seeds 0/1/2/5: onset 10.0-12.2 cm from target). Fix: the
#       same Coulomb feedforward the real-grasp lift uses
#       (control/realgrasp_flmpc.friction_feedforward, along FL-MPC's commanded
#       motion; re-clipped to TAU_MAX) on BOTH arms in the flmpc crossing branch.
#   (2)/(5) are metrics only (placed_strict, true cup-body tilt), no behavior change.
# ---------------------------------------------------------------------------
BUGFIXES = False                   # LIBRARY default (see above); the CLIs default to --bugfixes on
FAKED_BALL_JITTER_MAX_M = 0.003    # (1) faked+fixes: per-seed in-cup ball-layout jitter cap [m]
CUP_INIT_TILT_ASSERT_DEG = 15.0    # (1) faked+fixes: build_episode raises if the pinned cup is tilted more than this
CUP_INIT_TOPPLED_DEG = 30.0        # real: grasp FAILURE if the gripped cup is tilted more than this after the close
WELD_ATTACH_GATE_M = None          # (4): None = off (default). e.g. 0.02 = weld attaches only within 2 cm of p_at_i
PLACED_STRICT_MIN_MOVE_M = 0.05    # (2): a selected cube must have moved > this from its t=0 position


# ---------------------------------------------------------------------------
# --ctrlfixes {on,off} (2026-09-25). Same pattern as --bugfixes: LIBRARY default
# OFF (module CTRLFIXES=False -> every existing build_episode/run_episode caller
# is byte-identical), CLI default ON (this script's main() and
# comprehensive_eval.py pass it explicitly per job). Controller fixes:
#   F3 "gravity": the computed-torque branch (lift / v4c task torque / v4c HOCBF
#       + table rows / infeasible brake) used bias = qfrc_bias, but the arm bodies
#       have gravcomp=1, so gravity is ALSO in qfrc_passive -> double gravity
#       compensation. Fix: bias = qfrc_bias - qfrc_passive (as mpc_fl.FLMPC does).
#       NOTE: this also corrects the TABLE rows' dynamics model under flmpc (they
#       are built from bias_l/bias_r and mapped into accel space by FLMPC).
#   F1 "speedcap": right-EE speed-cap rows inside FL-MPC over the horizon
#       (|Jp_r qd_r(t)| <= V_EE_MAX_MS per axis + along the current motion
#       direction), slackable like the collision rows on an infeasible step.
#   F1 "antiwindup": the right-arm reference clock t_r is frozen while the
#       right-EE tracking error exceeds CTRLFIX_AW_ERR_M (flmpc only); the
#       episode is extended by at most CTRLFIX_AW_MAX_FREEZE_S so the task can finish.
#   F2 "waypoints": top-down (fingers down), table-aware right-arm cube
#       waypoints (control/cube_grasp.plan_top_down_pregrasp) for cubes where it
#       IK-solves cleanly (else the old position-only IK is kept for that cube).
#   F2 "tableradius": table-floor rows include each keypoint's radius
#       (sim_common.table_plane_barrier_rows(kp_radius=True)).
#   F5 "delaycomp": FL-MPC forward-predicts the state over CTRLFIX_DELAY_STEPS
#       using the already-committed accelerations (default 0 -> no-op).
#   F6 "taumax": FL-MPC per-joint torque limits CTRLFIX_TAU_MAX_VEC (default
#       None -> scalar sc.TAU_MAX); |c_i| > tau_max_i is handled explicitly.
#   F4 "cone": FL-MPC tilt rows as a K-facet L2 cone on the TRUE tilt (no
#       effective-gravity term), see CTRLFIX_CONE_FACETS.
#   F2b "realsel" (--grasp-mode real + "waypoints" only): the cube-selection
#       pre-pass (select_crossing_cubes_topdown) checks the right arm's hover/grasp
#       poses against the ACTUAL real-grasp cup-arm lift pose + cup pose
#       (_real_grasp_prepass) instead of the unfixed-init pose (whose cup is ~17 cm
#       off). No effect in faked mode (whose pre-pass has the same stale cup pose).
#   F2c "fakedsel" (--grasp-mode faked + --bugfixes on + "waypoints" only): the same
#       fix for FAKED mode. The pre-pass built its state with the UNFIXED
#       scene_setup.init_cup_grasped_state (sideways cup, ~17 cm off) while the episode
#       holds the cup with _init_cup_grasped_state_fixed, so reachability/clearance
#       were checked against the wrong cup + cup-arm pose. fakedsel runs the SAME fixed
#       init + lift IK + cup_rel as the episode (_faked_grasp_prepass). Old
#       `--ctrlfixes on` results reproduce with an explicit parts list omitting "fakedsel".
# CTRLFIX_PARTS selects which of the parts are active when ctrlfixes is on
# (default: all). --ctrlfix-parts overrides it (for per-fix ablations).
# ---------------------------------------------------------------------------
CTRLFIXES = False                  # LIBRARY default; the CLIs default to --ctrlfixes on
CTRLFIX_ALL_PARTS = ("gravity", "speedcap", "antiwindup", "waypoints", "tableradius",
                     "delaycomp", "taumax", "cone", "realsel", "fakedsel")
CTRLFIX_PARTS = set(CTRLFIX_ALL_PARTS)
# OPT-IN parts: accepted by --ctrlfix-parts (e.g. "all,heldcube") but NOT in the default "all",
# so the default ctrlfixes-on behavior (and its owner-verified numbers) is unchanged.
#   "heldcube" (F8): while a cube is welded, the right arm's table-floor rows get one extra
#       keypoint at the held cube (radius CUBE_HALF + TABLE_FLOOR_MARGIN_M + HELDCUBE_TABLE_CLEAR_M,
#       needs "tableradius"), so the carried cube is not dragged along / into the table; and on
#       release the cube's arm-collision is re-enabled only once the grasp site is
#       HELDCUBE_REENABLE_DIST_M away from it (re-enabling while the lagging cube overlaps the
#       fingers made the contact solver kick the gripper up -- the faked-mode 0.458 m/s "fling" --
#       or eject the cube ~20 cm out of the keep zone -- faked seed 3's placed_strict=0).
#   "retreat" (F9): between cubes the right arm transits via q_home (see build_episode).
CTRLFIX_OPTIONAL_PARTS = ("heldcube", "retreat")
HELDCUBE_TABLE_CLEAR_M = 0.002
HELDCUBE_REENABLE_DIST_M = 0.06
CTRLFIX_REF_SPEED_FRAC = 0.9       # F1: scripted right-EE PEAK speed <= this * V_EE_MAX_MS (FK-sampled path)
CTRLFIX_AW_MODE = "clamp"          # F1 anti-windup: "clamp" (default) = the right-arm reference that enters the
                                   # FL-MPC cost is pulled toward the current q so the EE error is <= CTRLFIX_AW_ERR_M
                                   # (the clock keeps running); "freeze" = freeze t_r while the error exceeds it.
                                   # "freeze" deadlocks whenever the reference passes through the collision barrier
                                   # (the intended crossing dodge): the clock stops at a point the arm may never reach.
CTRLFIX_AW_ERR_M = 0.03            # F1 anti-windup: EE tracking-error threshold [m]
CTRLFIX_AW_GRASP_FREEZE = True     # F1 ("clamp" mode): ALSO freeze t_r inside the at_i grasp windows while the
                                   # EE error > CTRLFIX_AW_ERR_M (budget CTRLFIX_AW_MAX_FREEZE_S), so the timed weld
                                   # does not attach from afar when the arm lags
CTRLFIX_AW_MAX_FREEZE_S = 3.0      # F1 anti-windup ("freeze" mode): total freeze budget [s] (episode extended by this)
CTRLFIX_DELAY_STEPS = 0            # F5: known actuation delay [control steps] to compensate (0 = none)
CTRLFIX_TAU_MAX_VEC = None         # F6: per-joint |tau| limit (6,) for FL-MPC; None = scalar sc.TAU_MAX
CTRLFIX_CONE_ANCHOR = "once"       # F4 (real grasp): "once" = anchor the cone on the true cup axis at the first
                                   # crossing step; "per_step" = re-anchor every step (the pre-existing behavior)
CTRLFIX_CONE_FACETS = 8            # F4: >0 -> true-tilt K-facet INSCRIBED cone (mpc_fl cone_mode="true", ||tilt||
                                   # <= CUP_TILT_MAX_RAD), anchored per CTRLFIX_CONE_ANCHOR; 0 = keep the per-axis box
CTRLFIX_WP_HOVER_M = 0.10          # F2: top-down hover height above the grasp depth [m]
CTRLFIX_WP_IK_ERR_MAX_M = 0.006    # F2: use the top-down waypoints only if both IK solves are < this
CTRLFIX_WP_MAX_TILT_DEG = 20.0     # F2: near-top-down fallback: approach axis within this of vertical
CTRLFIX_PLACE_DROP_M = 0.010       # F2: place the welded cube this much higher than it was grasped [m]
CTRLFIX_DROP_SLIP_M = 0.05         # F7: cup "dropped" if its EE-frame offset exceeds this (sustained)
CTRLFIX_PAD_FLICKER_STEPS = 5      # F7: pad_contacts_min ignores no-contact flicker shorter than this


def parse_ctrlfix_parts(spec: str | None) -> set:
    """'all' / None -> every part; else a comma list (validated)."""
    if spec is None or spec.strip() in ("", "all"):
        return set(CTRLFIX_ALL_PARTS)
    parts = {p.strip() for p in spec.split(",") if p.strip()}
    if "all" in parts:
        parts = (parts - {"all"}) | set(CTRLFIX_ALL_PARTS)
    bad = parts - set(CTRLFIX_ALL_PARTS) - set(CTRLFIX_OPTIONAL_PARTS)
    assert not bad, (f"unknown --ctrlfix-parts {sorted(bad)}; expected a subset of "
                     f"{CTRLFIX_ALL_PARTS + CTRLFIX_OPTIONAL_PARTS}")
    return parts


def _cf(info_or_flag, part: str) -> bool:
    """True iff ctrlfixes is on (bool or info dict) AND `part` is enabled."""
    on = bool(info_or_flag.get("ctrlfixes", False)) if isinstance(info_or_flag, dict) else bool(info_or_flag)
    return on and (part in CTRLFIX_PARTS)


# ---------------------------------------------------------------------------
# Joint-space waypoint reference (smoothstep-eased, C1-continuous -- same
# easing sim_common._smoothstep already uses for the v4-C* intruder ramp).
# ---------------------------------------------------------------------------
class JointWaypointTrack:
    def __init__(self, start_q: np.ndarray):
        self.start_q = np.asarray(start_q, dtype=float).copy()
        self.waypoints: list[tuple[np.ndarray, float, float]] = []
        # per-waypoint phase name (parallel list, same length as self.waypoints) --
        # used by bounds() below. MULTI-CUBE GENERALIZATION: this used to be derived
        # implicitly by zipping self.waypoints against the module-level RIGHT_PHASES
        # list (assumed a fixed 1:1 correspondence), which breaks once a track can
        # have a variable, per-cube-indexed phase list (hover_0/at_0/.../hover_1/...).
        # Storing the name alongside each waypoint at add()-time removes that
        # assumption entirely -- bounds() now works for ANY phase list, not just
        # RIGHT_PHASES's fixed 6 entries.
        self.names: list[str | None] = []

    def add(self, q_target, duration, hold, name: str | None = None):
        self.waypoints.append((np.asarray(q_target, dtype=float).copy(), float(duration), float(hold)))
        self.names.append(name)

    def sample(self, t: float) -> np.ndarray:
        q = self.start_q
        t0 = 0.0
        for q_target, duration, hold in self.waypoints:
            t_end_move = t0 + duration
            t_end_hold = t_end_move + hold
            if t <= t_end_move:
                s = 0.0 if duration <= 0 else float(np.clip((t - t0) / duration, 0.0, 1.0))
                s = s * s * (3.0 - 2.0 * s)
                return q + s * (q_target - q)
            elif t <= t_end_hold:
                return q_target
            q = q_target
            t0 = t_end_hold
        return q

    def sample_dot(self, t: float, h: float = 1e-3) -> np.ndarray:
        return (self.sample(t + h) - self.sample(t - h)) / (2.0 * h)

    def bounds(self) -> dict:
        """name -> (move_start, move_end, hold_end), absolute (track-local) times.
        Only waypoints added with a non-None `name` appear in the result."""
        out = {}
        t0 = 0.0
        for (q_target, duration, hold), name in zip(self.waypoints, self.names):
            move_end = t0 + duration
            hold_end = move_end + hold
            if name is not None:
                out[name] = (t0, move_end, hold_end)
            t0 = hold_end
        return out


class CupCollectMpcCtrl:
    """Minimal ctrl adapter satisfying mpc7.JointMPCv7's duck-typed interface
    (left/right_qposadr, left/right_dofadr, ee_body_id, payload_terms(data))
    for THIS scene -- reflects the HELD CUP's mass/inertia (not a
    canonical-scene 'test_box') into the LEFT arm's dynamics, via the SAME
    sc.payload_reflected_terms call the v4c path above already uses. Does
    NOT touch mpc7.py or sim_common.EgoController (which is hardcoded to the
    canonical scene's box/EE body names and would not resolve in this scene)."""

    def __init__(self, model, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
                 left_ee_body, cup_body, right_ee_body=None):
        self.model = model
        self.left_qposadr = left_qposadr
        self.left_dofadr = left_dofadr
        self.right_qposadr = right_qposadr
        self.right_dofadr = right_dofadr
        self.ee_body_id = left_ee_body
        # right (intruder) EE body id -- consumed by JointMPCv7's EE-speed cap
        # (v_ee_max/ee_body_id_int); optional so the pre-EE-cap ctor call stays valid.
        self.ee_body_id_int = right_ee_body
        self._cup_body = cup_body

    def payload_terms(self, data):
        return sc.payload_reflected_terms(self.model, data, self.ee_body_id,
                                           self._cup_body, self.left_dofadr)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])


def quat_angle(q):
    w = np.clip(abs(q[0]), -1.0, 1.0)
    return 2.0 * np.arccos(w)


def _tilt_offset_dot(t_clock: float, dur_s: float, amp_rad: float, h: float = 1e-3) -> float:
    """Finite-difference d/dt of scene_setup.transfer_tilt_offset(t_clock,
    start_s=0.0, dur_s, amp_rad) -- velocity feedforward for the LEFT arm's
    "spill" scenario tilt PD target (see run_episode). Only ever evaluated
    on un-vetoed steps (the caller zeroes it during a freeze), matching
    track_l/track_r's own sample_dot pattern above."""
    t_lo = max(0.0, t_clock - h)
    return (scene_setup.transfer_tilt_offset(t_clock + h, 0.0, dur_s, amp_rad)
            - scene_setup.transfer_tilt_offset(t_lo, 0.0, dur_s, amp_rad)) / (t_clock + h - t_lo)


# ---------------------------------------------------------------------------
# MULTI-CUBE cube selection (scenario="crossing" only): pick N_CUBES of the 6
# scattered cubes (cube_0..cube_5) that are (a) cleanly IK-reachable at both
# their hover and grasp pose (< CUBE_IK_ERR_MAX_M) and (b) farthest from the
# cup -- same reasoning the old single-cube CUBE_NAME="cube_2" choice used
# ("far on the right arm's own side" keeps the PICK itself clear of the cup
# arm, so the CARRY toward p_cross is the one genuine, dodgeable near-cup
# crossing per cube; a cube already sitting close to the cup, e.g. cube_4,
# risks colliding during the PICK itself, not just the carry).
#
# Run on a THROWAWAY pre-pass model/data (no weld equalities -- those must be
# baked in at spec.compile() time in build_episode, which needs to already
# know the selected cube NAMES first -- hence this separate, disposable load).
# Cube xy positions are static (scene_setup.init_cup_grasped_state does not
# jitter/randomize the cube scatter, only the cup -- cup_jitter_m=0.0 here
# regardless), so this pre-pass and the real build_episode load agree exactly
# for the same seed.
# ---------------------------------------------------------------------------
def _place_cluster_offsets(n: int, spacing: float) -> list:
    """Non-overlapping 2D (dx, dy) offsets, centered at the origin, for `n`
    place spots arranged as a compact grid (cols = ceil(sqrt(n)), rows =
    ceil(n/cols)) with `spacing` between adjacent columns AND rows.

    Replaces the old single-row layout (`place_center + [(i - (n-1)/2)*
    spacing, 0, 0]`), which only spread cubes apart along ONE axis -- fine
    for the old N_CUBES=3, but for up to 6 cubes a single row either gets
    too long (place spots pushed out of easy IK reach / outside the keep
    zone) or, if spacing is shrunk to compensate, too tight (adjacent cubes
    close enough to nudge each other out of the counted zone on landing). A
    2D grid keeps every pair of spots >= `spacing` apart in x AND y for any
    n up to 6 while staying compact -- see KEEP_ZONE_RADIUS_M's own comment
    for the margin this leaves inside the keep zone."""
    if n <= 0:
        return []
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    offsets = []
    for i in range(n):
        row, col = divmod(i, cols)
        dx = (col - (cols - 1) / 2.0) * spacing
        dy = (row - (rows - 1) / 2.0) * spacing
        offsets.append(np.array([dx, dy]))
    return offsets


def select_crossing_cubes(model, data, n_cubes: int = N_CUBES,
                          ik_err_max: float = CUBE_IK_ERR_MAX_M) -> list[str]:
    _, right_qposadr, _ = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    right_ik = bt.ArmIK(model, "right")
    q_home = data.qpos[right_qposadr].copy()
    cup_body = model.body(cct.CUP_BODY_NAME).id
    cup_xy = data.xpos[cup_body][:2].copy()

    cand = []
    for bid in cct.cube_body_ids(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        cxyz = data.xpos[bid].copy()
        dist = float(np.linalg.norm(cxyz[:2] - cup_xy))
        cand.append((dist, name, cxyz))
    if _CUBE_OVERRIDE is not None:
        # forced cube set (--cubes): restrict to these names, keep the given
        # order (still IK-filtered below), skip the distance sort.
        by_name = {name: (dist, name, cxyz) for dist, name, cxyz in cand}
        cand = [by_name[n] for n in _CUBE_OVERRIDE if n in by_name]
    else:
        cand.sort(key=lambda c: -c[0])   # farthest-from-cup first

    selected = []
    for dist, name, cxyz in cand:
        right_ik.last_q = q_home.copy()
        p_hover = cxyz + np.array([0.0, 0.0, HOVER])
        p_at = cxyz + np.array([0.0, 0.0, GRASP_Z_OFFSET])
        _, e_hover = right_ik.solve(data, p_hover)
        _, e_at = right_ik.solve(data, p_at)
        if max(e_hover, e_at) < ik_err_max:
            selected.append(name)
        if len(selected) >= n_cubes:
            break

    assert len(selected) >= 2, (
        f"multi-cube crossing needs >=2 cleanly IK-reachable cubes "
        f"(err<{ik_err_max * 1000:.0f}mm), only found {len(selected)}")
    if len(selected) < n_cubes:
        print(f"[build_episode] only {len(selected)}/{n_cubes} cubes cleanly "
              f"IK-reachable (err<{ik_err_max * 1000:.0f}mm) -- using {len(selected)}: {selected}")
    return selected


def _ctrlfix_down_grasp(model, data, bid: int, ik, q_home: np.ndarray, table_thr: float | None = None,
                        table_kps: list | None = None) -> dict | None:
    """--ctrlfixes F2: fingers-DOWN grasp waypoints for one cube (hover + grasp depth).
    First the exact top-down plan (control/cube_grasp.plan_top_down_pregrasp); if its
    IK misses (> CTRLFIX_WP_IK_ERR_MAX_M, e.g. cubes near the reach boundary), a
    NEAR-top-down fallback: the same site targets with a down-pointing orientation
    target at reduced orientation weight (4 face-aligned yaws), accepted only if the
    approach axis is within CTRLFIX_WP_MAX_TILT_DEG of vertical. If table_thr/
    table_kps are given, the grasp pose is raised until every keypoint SURFACE
    (z - r) clears table_thr. Returns None if the cube is not reachable this way."""
    from control import cube_grasp as cg
    tol = CTRLFIX_WP_IK_ERR_MAX_M
    pg = cg.plan_top_down_pregrasp(model, data, bid, "right", hover=CTRLFIX_WP_HOVER_M, ik=ik, prefer_q=q_home)
    ee_b = model.body("right_arm_gripper_base").id
    _, rqa, _ = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    scr = mujoco.MjData(model)
    scr.qpos[:] = data.qpos

    def _fk(q):
        scr.qpos[rqa] = q
        mujoco.mj_kinematics(model, scr)
        return scr.xpos[ee_b].copy(), scr.xmat[ee_b].reshape(3, 3).copy()

    def _tilt_deg(q):
        return float(np.degrees(np.arccos(np.clip(-_fk(q)[1][2, 2], -1.0, 1.0))))

    def _table_clr(q):
        if table_thr is None or table_kps is None:
            return np.inf
        scr.qpos[rqa] = q
        mujoco.mj_kinematics(model, scr)
        return min(float((scr.xpos[b_] + scr.xmat[b_].reshape(3, 3) @ o_)[2] - r_ - table_thr)
                   for b_, o_, r_ in table_kps)

    cand = None
    if max(pg.ik_err_hover, pg.ik_err_at) < tol:
        cand = dict(q_hover=pg.q_hover.copy(), q_at=pg.q_at.copy(), e_h=float(pg.ik_err_hover),
                    e_a=float(pg.ik_err_at), site_at=pg.site_pos_at.copy(), site_quat=pg.site_quat.copy(),
                    mode="topdown")
    else:
        for w in (0.3, 0.1, 0.03):
            for kk in range(4):
                psi = pg.yaw + kk * 0.5 * np.pi
                zb = np.array([0.0, 0.0, -1.0])
                yb = np.array([np.cos(psi), np.sin(psi), 0.0])
                Rk = np.column_stack([np.cross(yb, zb), yb, zb])
                qb = np.zeros(4)
                mujoco.mju_mat2Quat(qb, Rk.reshape(9))
                qs = np.zeros(4)
                mujoco.mju_mulQuat(qs, qb, model.site_quat[ik.site_id])
                ik.last_q = q_home.copy()
                q_h, e_h = ik.solve(data, pg.site_pos_hover, qs, ori_weight=w, iters=300)
                ik.last_q = q_h.copy()
                q_a, e_a = ik.solve(data, pg.site_pos_at, qs, ori_weight=w, iters=300)
                if max(e_h, e_a) < tol and max(_tilt_deg(q_h), _tilt_deg(q_a)) <= CTRLFIX_WP_MAX_TILT_DEG:
                    cand = dict(q_hover=q_h, q_at=q_a, e_h=float(e_h), e_a=float(e_a), site_at=pg.site_pos_at.copy(),
                                site_quat=qs, ori_w=w, mode=f"near_topdown_w{w}")
                    break
            if cand is not None:
                break
    if cand is None:
        return None
    # table-aware: raise the grasp pose until every keypoint surface clears the floor threshold
    for _ in range(4):
        clr = _table_clr(cand["q_at"])
        if clr >= 0.0:
            break
        cand["site_at"] = cand["site_at"] + np.array([0.0, 0.0, -clr + 0.002])
        ik.last_q = cand["q_at"].copy()
        cand["q_at"], cand["e_a"] = ik.solve(data, cand["site_at"], cand["site_quat"],
                                             ori_weight=cand.get("ori_w", 1.0), iters=300)
        cand["mode"] += "+raised"
    cand["table_clr_at"] = _table_clr(cand["q_at"])
    ee_at, R_at = _fk(cand["q_at"])
    cand.update(R_at=R_at, tip_local=pg.tip_local.copy(), tip_at=ee_at + R_at @ pg.tip_local, yaw=float(pg.yaw),
                tilt_at_deg=_tilt_deg(cand["q_at"]))
    return cand


def _ctrlfix_table_safe_segments(model, data, site_id, qposadr, q_start, targets, phases, table_kps, table_thr,
                                 n_sample: int = 41, n_via: int = 3, max_iter: int = 4):
    """--ctrlfixes F2: split right-arm reference segments that dip below the table.
    For each move q_prev -> q_t (smoothstep joint interpolation, as JointWaypointTrack
    samples it), FK-sample the min keypoint-surface clearance; if < 0, replace the move
    by n_via+1 sub-moves through IK via-points on the straight Cartesian site line
    (orientation slerped), raising the via-points until every sub-move clears. The
    original waypoint keeps its (name, hold); vias are unnamed with zero hold and an
    equal share of the move duration. Returns (targets, phases, n_vias_added)."""
    scr = mujoco.MjData(model)
    scr.qpos[:] = data.qpos

    def _clr_path(qa, qb):
        worst = np.inf
        for sft in np.linspace(0.0, 1.0, n_sample):
            scr.qpos[qposadr] = qa + sft * (qb - qa)
            mujoco.mj_kinematics(model, scr)
            for b_, o_, r_ in table_kps:
                worst = min(worst, float((scr.xpos[b_] + scr.xmat[b_].reshape(3, 3) @ o_)[2] - r_ - table_thr))
        return worst

    def _site_pose(q):
        scr.qpos[qposadr] = q
        mujoco.mj_kinematics(model, scr)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, scr.site_xmat[site_id].copy())
        return scr.site_xpos[site_id].copy(), quat

    def _slerp(q0, q1, u):
        q0 = q0 / np.linalg.norm(q0); q1 = q1 / np.linalg.norm(q1)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1, dot = -q1, -dot
        if dot > 0.9995:
            q = q0 + u * (q1 - q0)
            return q / np.linalg.norm(q)
        th = np.arccos(dot)
        return (np.sin((1 - u) * th) * q0 + np.sin(u * th) * q1) / np.sin(th)

    ik = bt.ArmIK(model, "right")
    out_t, out_p, added = [], [], 0
    q_prev = np.asarray(q_start, float)
    for qt, (name, dur, hold) in zip(targets, phases):
        if _clr_path(q_prev, qt) >= 0.0:
            out_t.append(qt); out_p.append((name, dur, hold)); q_prev = qt
            continue
        pa, qa_ = _site_pose(q_prev)
        pb, qb_ = _site_pose(qt)
        lift = 0.0
        vias = None
        for _ in range(max_iter):
            vias, qs = [], q_prev
            for k in range(1, n_via + 1):
                u = k / (n_via + 1)
                ik.last_q = qs.copy()
                qv, _e = ik.solve(data, pa + u * (pb - pa) + np.array([0.0, 0.0, lift]), _slerp(qa_, qb_, u),
                                  ori_weight=0.3, iters=200)
                vias.append(qv); qs = qv
            chain = [q_prev] + vias + [qt]
            worst = min(_clr_path(chain[j], chain[j + 1]) for j in range(len(chain) - 1))
            if worst >= 0.0:
                break
            lift += -worst + 0.01
        for qv in vias:
            out_t.append(qv); out_p.append((None, dur / (n_via + 1), 0.0)); added += 1
        out_t.append(qt); out_p.append((name, dur / (n_via + 1), hold))
        q_prev = qt
    return out_t, out_p, added


def select_crossing_cubes_topdown(model, data, n_cubes: int = N_CUBES, *, left_q_lift: np.ndarray | None = None,
                                  cup_rel: tuple | None = None) -> list[str]:
    """--ctrlfixes F2: like select_crossing_cubes (farthest-from-cup first, or the
    --cubes override order), but a cube qualifies only if (a) a fingers-DOWN grasp
    (_ctrlfix_down_grasp) IK-solves at hover and grasp depth and (b) those two poses
    clear the LEFT arm + held cup (arm-arm barrier h >= 0 and right-arm-to-cup
    clearance >= 0) with the left arm at its lift pose -- a cube right next to the
    lifted cup (e.g. cube_4) cannot be picked without the barrier blocking it.

    left_q_lift / cup_rel (default None = the FAKED-grasp lift pose, IK-solved here,
    and the cup pose in `data`): the --ctrlfix-parts "realsel" pre-pass passes the
    ACTUAL real-grasp lift pose and cup-in-EE pose (_real_grasp_prepass) instead."""
    _, right_qposadr, _ = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    _, left_qposadr, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    q_home = data.qpos[right_qposadr].copy()
    cup_body = model.body(cct.CUP_BODY_NAME).id
    cup_xy = data.xpos[cup_body][:2].copy()
    left_ee_body = model.body(scene_setup.EE_BODY_NAME).id
    # left lift pose (same IK as build_episode's faked path) + cup keypoints
    if left_q_lift is None:
        left_ik = bt.ArmIK(model, "left")
        left_ik.last_q = data.qpos[left_qposadr].copy()
        q_lift, _ = left_ik.solve(data, data.site_xpos[left_ik.site_id].copy() + np.array([0.0, 0.0, LIFT_HEIGHT_M]))
    else:
        q_lift = np.asarray(left_q_lift, dtype=float).copy()
    if cup_rel is None:
        rel_p, rel_q = scene_setup.cup_rel_pose(model, data, left_ee_body, cup_body)
    else:
        rel_p, rel_q = cup_rel
    cup_kps = build_cup_barrier_keypoints(left_ee_body, rel_p, rel_q)
    right_kps = sc.barrier_keypoints(model, sc.INT_PREFIX)
    scr = mujoco.MjData(model)
    scr.qpos[:] = data.qpos
    scr.qpos[left_qposadr] = q_lift
    cand = []
    for bid in cct.cube_body_ids(model):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        cand.append((float(np.linalg.norm(data.xpos[bid][:2] - cup_xy)), name, bid))
    if _CUBE_OVERRIDE is not None:
        by_name = {nm: (dd, nm, bb) for dd, nm, bb in cand}
        cand = [by_name[n] for n in _CUBE_OVERRIDE if n in by_name]
    else:
        cand.sort(key=lambda c: -c[0])
    ik = bt.ArmIK(model, "right")
    selected = []
    for _d, name, bid in cand:
        g = _ctrlfix_down_grasp(model, data, bid, ik, q_home)
        if g is None:
            continue
        clear = True
        for q in (g["q_hover"], g["q_at"]):
            scr.qpos[right_qposadr] = q
            mujoco.mj_kinematics(model, scr)
            # TASK definition is independent of the controller's keep-out margin: the
            # clearance test uses the nominal SELECT_D_MIN_REF, not the live sc.D_MIN
            # (a v4c D_MIN retune must not change WHICH cubes are picked). Exactly h
            # when sc.D_MIN == SELECT_D_MIN_REF (the default).
            h = sc.min_pair_distance(model, scr)[0] + (sc.D_MIN - SELECT_D_MIN_REF)
            c = right_arm_to_cup_clearance(model, scr, cup_kps, right_kps, left_ee_body)
            if h < 0.0 or c < 0.0:
                clear = False
        if clear:
            selected.append(name)
        if len(selected) >= n_cubes:
            break
    assert len(selected) >= 2, f"ctrlfixes F2: need >=2 fingers-down reachable, clear cubes, found {selected}"
    if len(selected) < n_cubes:
        print(f"[build_episode] ctrlfixes F2: only {len(selected)}/{n_cubes} cubes are fingers-down reachable and "
              f"clear of the cup arm -- using {selected}")
    return selected


def _real_grasp_prepass(model, seed: int, cup_jitter_m: float):
    """--ctrlfixes F2b "realsel" (real grasp only): the cube-selection pre-pass
    runs the SAME real friction grasp + dense lift IK that build_episode's real
    branch runs (same seed/rng, same constants) on the throwaway pre-pass model and
    returns (data, q_lift, (cup_rel_pos, cup_rel_quat)) -- the ACTUAL real-mode
    cup-arm lift pose and cup-in-EE pose, so select_crossing_cubes_topdown checks
    the right arm's hover/grasp poses against where the cup arm and cup really are.
    MEASURED (seed 0): the default pre-pass takes the cup pose from the UNFIXED
    scene_setup.init_cup_grasped_state (the sideways-cup bug, --bugfixes (1)), which
    puts the lifted cup ~17 cm from where it actually is at lift ([0.49,-0.37,0.68]
    vs [0.42,-0.21,0.66]); the actual real-mode and faked(+bugfixes) lifted cup
    agree within ~1 cm, and this pre-pass reproduces them. SIMULATION ONLY."""
    rg.restore_left_pad_object_collision(model)
    data = mujoco.MjData(model)
    ball_jit = min(REAL_BALL_JITTER_MAX_M, float(cup_jitter_m)) if cup_jitter_m > 0.0 else 0.0
    _arm, gres, _gi = rg.grasp_cup_real_leveled(
        model, data, grip_force=REAL_GRIP_FORCE_N, level=REAL_GRASP_LEVELED,
        ball_jitter_m=ball_jit, rng=np.random.default_rng(seed))
    if not gres.held:
        raise RuntimeError(f"real grasp failed (selection pre-pass): {gres}")
    _, lqa, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    q_grasp = data.qpos[lqa].copy()
    left_ik = bt.ArmIK(model, "left")
    left_ik.last_q = q_grasp.copy()
    _trk, q_lift, _e, _o = rg.build_dense_lift_track(
        model, data, left_ik, q_grasp, np.array([0.0, 0.0, LIFT_HEIGHT_M]), T_LIFT_MOVE, T_LIFT_HOLD)
    cup_rel = rg.cup_rel_pose_in_ee(data, model.body(scene_setup.EE_BODY_NAME).id, model.body(cct.CUP_BODY_NAME).id)
    return data, q_lift, cup_rel


def _faked_grasp_prepass(model, seed: int, cup_jitter_m: float):
    """--ctrlfixes F2c "fakedsel" (faked grasp + bugfixes only): the cube-selection
    pre-pass state = the SAME upright (_init_cup_grasped_state_fixed) init that the
    faked episode uses, plus the same single-IK lift pose (left_ik.solve from the
    settled grasp, +LIFT_HEIGHT_M in z) and the same cup-in-EE pose the episode
    computes in build_episode. Returns (data, q_lift, (cup_rel_pos, cup_rel_quat)).
    SIMULATION ONLY."""
    data = _init_cup_grasped_state_fixed(model, seed=seed, cup_jitter_m=cup_jitter_m)
    _, lqa, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    left_ik = bt.ArmIK(model, "left")
    left_ik.last_q = data.qpos[lqa].copy()
    q_lift, _err = left_ik.solve(data, data.site_xpos[left_ik.site_id].copy() + np.array([0.0, 0.0, LIFT_HEIGHT_M]))
    cup_rel = scene_setup.cup_rel_pose(model, data, model.body(scene_setup.EE_BODY_NAME).id,
                                       model.body(cct.CUP_BODY_NAME).id)
    return data, q_lift, cup_rel


def _init_cup_grasped_state_fixed(model, seed: int = 0, cup_jitter_m: float = 0.0):
    """(1) SIDEWAYS-CUP FIX: a demo-local copy of eval/scene_setup.
    init_cup_grasped_state (same reset, LEFT_GRASP_Q teleport, cup-to-pad-
    midpoint placement, gravity-off close, 500+300 settle) with ONE change:
    the balls are translated by (new_cup_pos - TRUE old_cup_pos), so they stay
    in the canonical in-cup layout; per-seed variation (when cup_jitter_m>0)
    is an in-cup ball-layout xy jitter of at most FAKED_BALL_JITTER_MAX_M. With
    cup_jitter_m == 0 this is step-for-step identical to the original."""
    rng = np.random.default_rng(seed)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    mujoco.mj_forward(model, data)
    _, lqa, _ = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    data.qpos[lqa] = sc.LEFT_GRASP_Q
    left_act_ids = [model.actuator(f"left_arm_joint{i}_pos").id for i in range(1, 7)]
    data.ctrl[left_act_ids] = sc.LEFT_GRASP_Q
    mujoco.mj_forward(model, data)
    pad7 = model.geom("left_arm_link7_pad").id
    pad8 = model.geom("left_arm_link8_pad").id
    pad_mid = 0.5 * (data.geom_xpos[pad7] + data.geom_xpos[pad8])
    cup_jid = model.joint("cup_free").id
    cqa, cda = int(model.jnt_qposadr[cup_jid]), int(model.jnt_dofadr[cup_jid])
    old_cup_pos = data.qpos[cqa:cqa + 3].copy()          # TRUE old position (no jitter here)
    new_cup_pos = pad_mid - np.array([0.0, 0.0, scene_setup.CUP_GRASP_Z_FRAC * cct.CUP_HEIGHT])
    delta = new_cup_pos - old_cup_pos
    data.qpos[cqa:cqa + 3] = new_cup_pos
    data.qpos[cqa + 3:cqa + 7] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qvel[cda:cda + 6] = 0.0
    ball_jit = min(FAKED_BALL_JITTER_MAX_M, float(cup_jitter_m)) if cup_jitter_m > 0.0 else 0.0
    for bid in cct.ball_body_ids(model):
        jadr = model.body_jntadr[bid]
        bqa, bda = int(model.jnt_qposadr[jadr]), int(model.jnt_dofadr[jadr])
        data.qpos[bqa:bqa + 3] += delta
        if ball_jit > 0.0:
            data.qpos[bqa:bqa + 2] += rng.uniform(-ball_jit, ball_jit, size=2)
        data.qvel[bda:bda + 6] = 0.0
    mujoco.mj_forward(model, data)
    g0 = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    data.ctrl[model.actuator("left_arm_joint7_pos").id] = 0.0
    data.ctrl[model.actuator("left_arm_joint8_pos").id] = 0.0
    for _ in range(500):
        mujoco.mj_step(model, data)
    model.opt.gravity[:] = g0
    for _ in range(300):
        mujoco.mj_step(model, data)
    return data


# ---------------------------------------------------------------------------
# Setup: model + grasp + IK-solved references (done ONCE, before the loop).
# ---------------------------------------------------------------------------
def build_episode(seed: int = 0, scenario: str = "crossing", ee_speed_limit: bool = True, cup_jitter_m: float = 0.0,
                   n_cubes: int | None = None, grasp_mode: str = "faked", bugfixes: bool | None = None,
                   weld_gate_m: float | None = None, ctrlfixes: bool | None = None):
    """n_cubes: SMOKE-TEST ONLY override for the number of crossing cubes
    (default None -> module constant N_CUBES=6, unchanged). Passed straight
    through to select_crossing_cubes below; a smaller n_cubes (e.g. 3) makes
    for a shorter/harder-to-clear crossing scenario in a quick sweep."""
    assert scenario in SCENARIO_CHOICES, f"unknown scenario {scenario!r}"
    assert grasp_mode in GRASP_MODE_CHOICES, f"unknown grasp_mode {grasp_mode!r}"
    bugfixes = BUGFIXES if bugfixes is None else bool(bugfixes)
    ctrlfixes = CTRLFIXES if ctrlfixes is None else bool(ctrlfixes)
    if grasp_mode == "real" and scenario != "crossing":
        raise ValueError("--grasp-mode real is implemented for --scenario crossing only")
    n_cubes_eff = N_CUBES if n_cubes is None else int(n_cubes)

    selected_cube_names = None
    if scenario == "crossing":
        # Pre-pass (throwaway model, no welds) -- see select_crossing_cubes' own
        # docstring for why cube selection must happen BEFORE the real model load.
        _pre_model, _pre_data0 = cct.load_yor_cup_collect_model(keep_zone_marker=False)  # throwaway; skip the marker geom
        _use_fakedsel = (_cf(ctrlfixes, "waypoints") and grasp_mode == "faked" and bugfixes
                         and _cf(ctrlfixes, "fakedsel"))
        _pre_data = None if _use_fakedsel else scene_setup.init_cup_grasped_state(
            _pre_model, seed=seed, cup_jitter_m=cup_jitter_m)
        if _use_fakedsel:
            # F2c: the SAME upright init + lift pose + cup_rel as the faked episode below.
            _pre_data, _q_lift_f, _cup_rel_f = _faked_grasp_prepass(_pre_model, seed, cup_jitter_m)
            selected_cube_names = select_crossing_cubes_topdown(_pre_model, _pre_data, n_cubes=n_cubes_eff,
                                                                left_q_lift=_q_lift_f, cup_rel=_cup_rel_f)
        elif _cf(ctrlfixes, "waypoints") and grasp_mode == "real" and _cf(ctrlfixes, "realsel"):
            # F2b: check reachability/clearance against the ACTUAL real-grasp cup-arm
            # lift pose (the faked-init pose above is ~cm off -- see _real_grasp_prepass).
            _pre_data, _q_lift_real, _cup_rel_real = _real_grasp_prepass(_pre_model, seed, cup_jitter_m)
            selected_cube_names = select_crossing_cubes_topdown(_pre_model, _pre_data, n_cubes=n_cubes_eff,
                                                                left_q_lift=_q_lift_real, cup_rel=_cup_rel_real)
        elif _cf(ctrlfixes, "waypoints"):
            selected_cube_names = select_crossing_cubes_topdown(_pre_model, _pre_data, n_cubes=n_cubes_eff)
        else:
            selected_cube_names = select_crossing_cubes(_pre_model, _pre_data, n_cubes=n_cubes_eff)
        del _pre_model, _pre_data0, _pre_data

    # CHANGE 1: the cube(s) are picked up via a WELD "magnet" attach (real
    # constraint force, activated/deactivated in run_episode below) instead
    # of the scripted-pin reattach_cup_rigidly path -- add the (inactive)
    # equalities here, at load time, via cup_collect_task_utils. "spill" never
    # touches cubes, so no weld equality is added at all for that scenario.
    # KEEP-ZONE MARKER: keep_zone_radius_m=KEEP_ZONE_RADIUS_M threads this
    # module's own constant through to the marker geom's size AND (via the
    # count_placed_cubes call in run_episode below) the placement ground
    # truth, so the drawn circle and the counted "placed" zone always match.
    if scenario == "crossing":
        model, _ = cct.load_yor_cup_collect_model(weld_cube_names=selected_cube_names,
                                                   weld_ee_body_name="right_arm_gripper_base",
                                                   keep_zone_radius_m=KEEP_ZONE_RADIUS_M)
    else:
        # "spill": no cube collection at all in this scenario -- skip the keep-zone marker.
        model, _ = cct.load_yor_cup_collect_model(keep_zone_marker=False)
    grip_ctrl = None
    grasp_info = None
    if grasp_mode == "real":
        # REAL friction grasp (SIM ONLY): LEFT pads only get object-contact bits
        # (right fingers untouched -> the cube-weld path is identical to faked),
        # then a genuine grip-force-controlled pinch with gravity ON. No pin.
        rg.restore_left_pad_object_collision(model)
        data = mujoco.MjData(model)
        ball_jit = min(REAL_BALL_JITTER_MAX_M, float(cup_jitter_m)) if cup_jitter_m > 0.0 else 0.0
        _rg_arm, gres, grasp_info = rg.grasp_cup_real_leveled(
            model, data, grip_force=REAL_GRIP_FORCE_N, level=REAL_GRASP_LEVELED,
            ball_jitter_m=ball_jit, rng=np.random.default_rng(seed))
        grasp_info["held"] = bool(gres.held)
        grasp_info["n_pad_contacts"] = int(gres.n_pad_contacts)
        grasp_info["grip_force_measured_n"] = float(gres.grip_force_measured)
        if not gres.held:
            raise RuntimeError(f"real grasp failed: {gres}")
        grip_ctrl = _rg_arm.grip          # keep ONLY the grip-force controller; the OSC arm controller is retired
        grip_ctrl.set_target_force(REAL_GRIP_FORCE_N)
    elif bugfixes:
        data = _init_cup_grasped_state_fixed(model, seed=seed, cup_jitter_m=cup_jitter_m)   # (1)
    else:
        data = scene_setup.init_cup_grasped_state(model, seed=seed, cup_jitter_m=cup_jitter_m)

    _, left_qposadr, left_dofadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, right_qposadr, right_dofadr = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)

    left_ee_body = model.body(scene_setup.EE_BODY_NAME).id
    right_ee_body = model.body("right_arm_gripper_base").id
    cup_body = model.body(cct.CUP_BODY_NAME).id

    # --controller v7c only (CHANGE 2): geoms mpc7.grasp_fragility needs to
    # measure the LEFT arm's (real, contact-based) grasp margin -- see the
    # interface note in run_episode for why this scene's cup makes the
    # contact-force term degenerate (no real finger/cup contact).
    left_pad7_geom = model.geom("left_arm_link7_pad").id
    left_pad8_geom = model.geom("left_arm_link8_pad").id
    cup_frag_geom = model.geom("cup_bottom").id

    q_grasp = data.qpos[left_qposadr].copy()

    # Cup rigid-attach offset, captured ONCE right after grasp settle (same
    # technique eval/scene_setup.py's own harness uses).
    # (1) cup orientation at PIN time (world tilt of the cup body's own +z).
    cup_init_tilt_deg = float(np.degrees(rg.cup_true_tilt_rad(data, cup_body)))
    if grasp_mode == "real":
        if cup_init_tilt_deg > CUP_INIT_TOPPLED_DEG:
            raise RuntimeError(f"real grasp failed: cup toppled in the grip "
                               f"(tilt {cup_init_tilt_deg:.1f} deg > {CUP_INIT_TOPPLED_DEG:.0f})")
    elif bugfixes and cup_init_tilt_deg > CUP_INIT_TILT_ASSERT_DEG:
        raise RuntimeError(f"faked init: pinned cup tilted {cup_init_tilt_deg:.1f} deg "
                           f"(> {CUP_INIT_TILT_ASSERT_DEG:.0f}) even with the fixed init (seed {seed})")

    cup_rel_pos0, cup_rel_quat0 = scene_setup.cup_rel_pose(model, data, left_ee_body, cup_body)

    # CUP-IN-BARRIER FIX: cup keypoints (rim ring + body points), in the LEFT
    # EE's local frame, for sim_common.all_pair_barriers' extra_ego_keypoints
    # (see that module's diff + the constants/docstring above) -- built ONCE
    # here since cup_rel_pos0/cup_rel_quat0 are themselves fixed for the
    # whole episode.
    cup_barrier_keypoints = build_cup_barrier_keypoints(left_ee_body, cup_rel_pos0, cup_rel_quat0)

    # TABLE-FLOOR BARRIER FIX: the tabletop's real world-z surface height,
    # read directly off the compiled model/settled data (table_top is an
    # axis-aligned box, geom_pos z + half-height = top surface), NOT
    # hardcoded -- always tracks scenes/scene_yor_cup_collect.xml's own
    # table geom regardless of any future scene edit.
    table_top_gid = model.geom(TABLE_TOP_GEOM_NAME).id
    table_top_z = float(data.geom_xpos[table_top_gid][2] + model.geom_size[table_top_gid][2])

    # --- LEFT arm: IK-solve the lift target (vertical, from the settled grasp) ---
    left_ik = bt.ArmIK(model, "left")
    left_ik.last_q = q_grasp.copy()
    site_pos0 = data.site_xpos[left_ik.site_id].copy()
    if grasp_mode == "real":
        # straight vertical lift with the EE (hence the really-gripped cup)
        # orientation HELD: dense orientation-weighted IK knots, smoothstep-timed.
        track_l, q_lift, lift_err, _lift_ori = rg.build_dense_lift_track(
            model, data, left_ik, q_grasp, np.array([0.0, 0.0, LIFT_HEIGHT_M]), T_LIFT_MOVE, T_LIFT_HOLD)
    else:
        q_lift, lift_err = left_ik.solve(data, site_pos0 + np.array([0.0, 0.0, LIFT_HEIGHT_M]))

        track_l = JointWaypointTrack(q_grasp)
        track_l.add(q_lift, T_LIFT_MOVE, T_LIFT_HOLD)

    # Nominal (undisturbed) lifted cup position, via forward kinematics on a
    # scratch copy at q_lift -- used only to plan the RIGHT arm's IK targets;
    # the actual cup position at runtime is whatever the real (QP-filtered)
    # left-arm dynamics produce.
    scratch = mujoco.MjData(model)
    scratch.qpos[:] = data.qpos
    scratch.qpos[left_qposadr] = q_lift
    mujoco.mj_forward(model, scratch)
    ee_lift_pos = scratch.xpos[left_ee_body].copy()
    ee_lift_mat = scratch.xmat[left_ee_body].reshape(3, 3).copy()
    cup_lift_pos = ee_lift_pos + ee_lift_mat @ cup_rel_pos0
    cup_rim_z = cup_lift_pos[2] + cct.CUP_HEIGHT

    right_ik = bt.ArmIK(model, "right")
    q_home = data.qpos[right_qposadr].copy()
    right_ik.last_q = q_home.copy()

    # --- LEFT arm ("spill" scenario only): joint-space pinv of the LEFT EE's
    # rotational Jacobian, evaluated at the SETTLED LIFT pose (`scratch`,
    # already at q_lift above -- the tilt happens after the lift, unlike
    # eval/run_eval.py's single-phase scene where grasp pose == working
    # pose) -- reused UNMODIFIED from eval/scene_setup.py, same technique
    # eval/run_eval.py's transfer/tilt uses (see that module's docstrings
    # for why a wrist TILT, not a lateral translation, is what transmits
    # through this contact-free finger grasp). Converts the scalar
    # transfer_tilt_offset(t) angle (about world-X, TRANSFER_ROT_AXIS) into
    # a joint-space PD-target offset added on top of track_l's lift-hold. --
    q_tilt_dir = None
    if scenario == "spill":
        Jr_pinv = scene_setup.left_ee_jac_rot_pinv(model, scratch, left_ee_body, left_dofadr)
        q_tilt_dir = Jr_pinv @ scene_setup.TRANSFER_ROT_AXIS

    cube_plan = []
    if scenario == "crossing":
        # --- RIGHT arm: IK-solve the MULTI-CUBE pick/carry/place waypoint chain ---
        # `p_cross` is the SAME near-cup waypoint for every cube (reaches up beside
        # the lifted cup arm -- genuine arm-arm collision risk the cooperative QP
        # must dodge without spilling); each cube gets its OWN place spot, a short
        # row spread CUBE_PLACE_SPACING_M apart, centered near the cup's own ground
        # xy (so placed cubes don't stack/collide with each other).
        p_cross = cup_lift_pos + np.array([0.0, 0.04, -0.01])   # DEEPER: aim right EE close to the cup so the
                                                                 # cup arm is forced to dodge (WM-helps stress test)
        place_center = np.array([cup_lift_pos[0] - 0.02, cup_lift_pos[1] - 0.06, table_top_z + PLACE_STANDOFF_M])
        n_sel = len(selected_cube_names)
        # RELIABLE-PLACEMENT FIX: a compact 2D grid (not a single row -- see
        # _place_cluster_offsets' own docstring) of non-overlapping spots
        # centered on place_center, sized (via CUBE_PLACE_SPACING_M) to fit
        # inside KEEP_ZONE_RADIUS_M for any n_sel up to 6.
        place_xyzs = [place_center + np.array([dx, dy, 0.0])
                      for dx, dy in _place_cluster_offsets(n_sel, CUBE_PLACE_SPACING_M)]

        cube_ids_by_name = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid): bid
                             for bid in cct.cube_body_ids(model)}

        right_targets = []
        phases = []
        ik_errs = {}
        wp_topdown = _cf(ctrlfixes, "waypoints")
        wp_modes = []
        if wp_topdown:
            from control import cube_grasp as cg
            _td_ik = bt.ArmIK(model, "right")
            _site_local = model.site_pos[right_ik.site_id].copy()
            _td_used_slots = []
            # table-floor threshold + right keypoints (incl. fingertip/wrist extras, same as run_episode)
            _td_table_thr = table_top_z - TABLE_FLOOR_MARGIN_M
            _td_table_kps = list(sc.barrier_keypoints(model, sc.INT_PREFIX))
            for _gname, _r in (("right_arm_link7_pad", 0.006), ("right_arm_link8_pad", 0.006)):
                _gid = model.geom(_gname).id
                _td_table_kps.append((int(model.geom_bodyid[_gid]), model.geom_pos[_gid].copy(), _r))
            _td_table_kps.append((model.body("right_arm_gripper_base").id, np.zeros(3), 0.01))
        for i, name in enumerate(selected_cube_names):
            bid = cube_ids_by_name[name]
            cxyz = data.xpos[bid].copy()
            p_hover_i = cxyz + np.array([0.0, 0.0, HOVER])
            p_at_i = cxyz + np.array([0.0, 0.0, GRASP_Z_OFFSET])

            if wp_topdown:
                # F2 FINGERS-DOWN, TABLE-AWARE waypoints (_ctrlfix_down_grasp): top-down
                # (or near-top-down, <= CTRLFIX_WP_MAX_TILT_DEG) hover + grasp with the
                # jaws straddling the cube and every keypoint surface above the table
                # threshold; lift = hover; the carry (p_cross) and place keep a
                # top-down orientation (best of the 4 face yaws); the place puts the
                # fingertips (hence the welded cube's bottom) CTRLFIX_PLACE_DROP_M above
                # their grasp height. An orientation-IK miss (> CTRLFIX_WP_IK_ERR_MAX_M)
                # falls back to the old position-only solve for that waypoint.
                g = _ctrlfix_down_grasp(model, data, bid, _td_ik, q_home, table_thr=_td_table_thr,
                                        table_kps=_td_table_kps)
                assert g is not None, f"ctrlfixes F2: {name} not fingers-down reachable (selection bug)"
                q_hover_i, e_h = g["q_hover"].copy(), g["e_h"]
                q_at_i, e_a = g["q_at"].copy(), g["e_a"]
                q_lift_i, e_l = q_hover_i.copy(), e_h
                p_at_i = g["site_at"].copy()
                wp_modes.append(g["mode"])
                tip_at_z = float(g["tip_at"][2])

                class pg:   # the two fields _td_best / the place solve read
                    yaw = g["yaw"]
                    tip_local = g["tip_local"]

                def _td_best(site_target_fn, q_seed):
                    # the cube is square and welded in the jaws -> any of the 4 top-down
                    # yaws (pg.yaw + k*pi/2) is a valid carry/place orientation; pick the
                    # best IK (then least joint travel from q_seed).
                    best = None
                    for kk in range(4):
                        psi = pg.yaw + kk * 0.5 * np.pi
                        zb = np.array([0.0, 0.0, -1.0])
                        yb = np.array([np.cos(psi), np.sin(psi), 0.0])
                        Rk = np.column_stack([np.cross(yb, zb), yb, zb])
                        qb = np.zeros(4)
                        mujoco.mju_mat2Quat(qb, Rk.reshape(9))
                        qsk = np.zeros(4)
                        mujoco.mju_mulQuat(qsk, qb, model.site_quat[right_ik.site_id])
                        _td_ik.last_q = q_seed.copy()
                        qk, ek = _td_ik.solve(data, site_target_fn(Rk), qsk, ori_weight=1.0, iters=200)
                        sc_k = (round(ek, 3), float(np.linalg.norm(qk - q_seed)))
                        if best is None or sc_k < best[0]:
                            best = (sc_k, qk, ek)
                    return best[1], best[2]

                q_cross_i, e_c = _td_best(lambda Rk: p_cross, q_lift_i)
                if e_c > CTRLFIX_WP_IK_ERR_MAX_M:
                    right_ik.last_q = q_lift_i.copy()
                    q_cross_i, e_c = right_ik.solve(data, p_cross)
                # place slot: this cube's own grid slot if the top-down place IK-solves,
                # else the nearest still-free slot of the 6-slot grid that does (the
                # right arm cannot reach every slot fingers-down). place_xyzs[i] is
                # updated so the release gate / placed metrics use the real slot.
                _cands = [place_xyzs[i]] + sorted(
                    [place_center + np.array([dx, dy, 0.0])
                     for dx, dy in _place_cluster_offsets(6, CUBE_PLACE_SPACING_M)],
                    key=lambda p_: float(np.linalg.norm(p_[:2] - place_xyzs[i][:2])))
                q_place_i, e_p = None, np.inf
                for _pc in _cands:
                    if any(np.linalg.norm(_pc[:2] - u[:2]) < 0.01 for u in _td_used_slots):
                        continue
                    _tip_pc = np.array([_pc[0], _pc[1], tip_at_z + CTRLFIX_PLACE_DROP_M])
                    _q, _e = _td_best(
                        lambda Rk: cg.ee_targets_for_tip(_tip_pc, Rk, pg.tip_local, _site_local)[1], q_cross_i)
                    if _e <= CTRLFIX_WP_IK_ERR_MAX_M:
                        q_place_i, e_p = _q, _e
                        place_xyzs[i] = _pc.copy()
                        break
                if q_place_i is None:
                    right_ik.last_q = q_cross_i.copy()
                    q_place_i, e_p = right_ik.solve(data, place_xyzs[i])
                _td_used_slots.append(place_xyzs[i].copy())
                right_ik.last_q = q_place_i.copy()
            else:
                q_hover_i, e_h = right_ik.solve(data, p_hover_i)
                q_at_i, e_a = right_ik.solve(data, p_at_i)
                q_lift_i, e_l = right_ik.solve(data, p_hover_i)   # reuse hover pose for the lift-back waypoint
                q_cross_i, e_c = right_ik.solve(data, p_cross)
                q_place_i, e_p = right_ik.solve(data, place_xyzs[i])
            ik_errs[f"hover_{i}"], ik_errs[f"at_{i}"], ik_errs[f"lift_{i}"] = e_h, e_a, e_l
            ik_errs[f"cross_{i}"], ik_errs[f"place_{i}"] = e_c, e_p

            eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, f"{cct.CUBE_WELD_EQ_NAME}_{i}")
            assert eq_id >= 0, f"cube weld equality '{cct.CUBE_WELD_EQ_NAME}_{i}' not found -- wiring broken"
            cube_plan.append(dict(body_id=bid, eq_id=eq_id, name=name, place_xyz=place_xyzs[i],
                                  grasp_xyz=p_at_i.copy()))

            if i > 0 and _cf(ctrlfixes, "retreat"):
                # F9 "retreat" (opt-in): transit between cubes via the home pose instead of the
                # straight joint-space move from the place zone (under the cup arm) to the next
                # hover -- that move can dead-lock against the arm-arm barrier (real grasp, a
                # cube on the far side of the cup arm: the weld then attaches from ~34 cm).
                right_targets.append(q_home.copy())
                phases.append((None, RIGHT_PHASES[5][1], 0.0))
            right_targets += [q_hover_i, q_at_i, q_lift_i, q_cross_i, q_place_i]
            # phase NAMES suffixed by cube index; SAME per-phase (move_dur, hold)
            # template as RIGHT_PHASES' first 5 entries (hover/at/lift/carry/release).
            phases += [
                (f"hover_{i}", RIGHT_PHASES[0][1], RIGHT_PHASES[0][2]),
                (f"at_{i}", RIGHT_PHASES[1][1], RIGHT_PHASES[1][2]),
                (f"lift_{i}", RIGHT_PHASES[2][1], RIGHT_PHASES[2][2]),
                (f"carry_{i}", RIGHT_PHASES[3][1], RIGHT_PHASES[3][2]),
                (f"release_{i}", RIGHT_PHASES[4][1], RIGHT_PHASES[4][2]),
            ]
        right_targets.append(q_home)
        phases.append(RIGHT_PHASES[5])   # ONE final ("park", 1.0, 0.30) home, after every cube
        if wp_topdown:
            # F2 TABLE-AWARE PATHS: a joint-space move between two table-clear waypoints
            # can still sweep a keypoint below the table (big wrist re-orientations).
            # Segments whose FK-sampled keypoint-surface clearance dips below 0 are split
            # into Cartesian via-points (site lerp + orientation slerp, raised as needed).
            right_targets, phases, n_vias = _ctrlfix_table_safe_segments(
                model, data, right_ik.site_id, right_qposadr, q_home, right_targets, phases,
                _td_table_kps, _td_table_thr)
            ik_errs["table_vias"] = n_vias

        track_r = JointWaypointTrack(q_home)
        # EE-SPEED-LIMIT FIX part (1): stretch each phase's MOVE duration so the
        # scripted right-EE Cartesian speed never exceeds V_EE_MAX_MS -- computed
        # from the phase's real EE travel distance (FK on a scratch qpos, base
        # restored), NOT hand-picked. Phases already under the cap (e.g. carry
        # ~0.24 m/s) keep their nominal duration, so the crossing/spill-contrast
        # dynamics are preserved; only the genuinely-fast, task-noncritical
        # phases (park ~0.79, release ~0.40, hover ~0.36 m/s) are calmed. Applies
        # per-phase over the FULL multi-cube phase list, not just RIGHT_PHASES'
        # fixed 6 entries.
        move_durs = [d for _, d, _ in phases]
        if ee_speed_limit:
            _qpos_save = data.qpos.copy()
            def _right_ee(qcfg):
                data.qpos[right_qposadr] = qcfg
                mujoco.mj_forward(model, data)
                return data.xpos[right_ee_body].copy()
            prev_ee = _right_ee(q_home)
            scaled = []
            for (name, dur, _hold), qt in zip(phases, right_targets):
                ee = _right_ee(qt)
                dist = float(np.linalg.norm(ee - prev_ee))
                scaled.append(max(dur, dist / V_EE_MAX_MS))
                prev_ee = ee
            move_durs = scaled
            data.qpos[:] = _qpos_save
            mujoco.mj_forward(model, data)
            if _cf(ctrlfixes, "speedcap"):
                # F1 (reference side): the smoothstep joint-space move peaks at ~1.5x
                # its average speed, and its EE path is not a straight line, so the
                # straight-line dist/V scaling above still commands up to ~0.4 m/s at
                # the peak -- faster than the in-QP cap allows -> the arm lags and the
                # anti-windup freezes the clock all the time. Scale each move so the
                # PEAK EE speed along the actual joint-space path (FK-sampled) is
                # <= CTRLFIX_REF_SPEED_FRAC * V_EE_MAX_MS.
                _scr = mujoco.MjData(model)
                _scr.qpos[:] = data.qpos
                _ss = np.linspace(0.0, 1.0, 81)
                _ease = _ss * _ss * (3.0 - 2.0 * _ss)
                q_prev = q_home
                peak_scaled = []
                for dur, qt in zip(move_durs, right_targets):
                    pts = []
                    for e in _ease:
                        _scr.qpos[right_qposadr] = q_prev + e * (qt - q_prev)
                        mujoco.mj_kinematics(model, _scr)
                        pts.append(_scr.xpos[right_ee_body].copy())
                    pts = np.asarray(pts)
                    # EE path speed for a unit-duration move: |dp| / ds (s in [0,1])
                    v_unit = float(np.max(np.linalg.norm(np.diff(pts, axis=0), axis=1)) / (_ss[1] - _ss[0]))
                    peak_scaled.append(max(dur, v_unit / (CTRLFIX_REF_SPEED_FRAC * V_EE_MAX_MS)))
                    q_prev = qt
                move_durs = peak_scaled
        for (name, _dur, hold), qt, dur in zip(phases, right_targets, move_durs):
            track_r.add(qt, dur, hold, name=name)
        bounds = track_r.bounds()
        # Episode length must follow the ACTUAL (possibly speed-scaled) track,
        # else the stretched park phase gets truncated before the arm gets home.
        # Recomputed from the ACTUAL (scaled) per-phase durations (move_durs),
        # generalized over the full multi-cube phase list.
        duration_s_total = RIGHT_START_T + sum(d + h for d, (_, _, h) in zip(move_durs, phases)) + SETTLE_TAIL_S
        n_steps_total = int(round(duration_s_total / sc.DT))
    else:
        # "spill" scenario: RIGHT arm just holds home for the whole episode
        # (secondary -- no pick/carry/deposit, keeps the demo focused on the
        # LEFT arm's own tilt/spill, no arm-arm proximity involved at all).
        ik_errs = dict()
        track_r = JointWaypointTrack(q_home)
        track_r.add(q_home, 0.0, SPILL_DURATION_S)
        bounds = None
        n_steps_total, duration_s_total = SPILL_N_STEPS, SPILL_DURATION_S

    info = dict(
        model=model, data=data, scenario=scenario,
        left_qposadr=left_qposadr, left_dofadr=left_dofadr,
        right_qposadr=right_qposadr, right_dofadr=right_dofadr,
        left_ee_body=left_ee_body, right_ee_body=right_ee_body,
        cup_body=cup_body, cube_plan=cube_plan,
        left_pad7_geom=left_pad7_geom, left_pad8_geom=left_pad8_geom,
        cup_frag_geom=cup_frag_geom,
        cup_rel_pos0=cup_rel_pos0, cup_rel_quat0=cup_rel_quat0,
        cup_barrier_keypoints=cup_barrier_keypoints,
        table_top_z=table_top_z,
        track_l=track_l, track_r=track_r, bounds=bounds,
        q_tilt_dir=q_tilt_dir,
        q_grasp=q_grasp, q_lift=q_lift, q_home=q_home,
        cup_lift_pos=cup_lift_pos,
        ik_errs=ik_errs, lift_err=lift_err,
        n_steps_total=n_steps_total, duration_s_total=duration_s_total,
        grasp_mode=grasp_mode, grip_ctrl=grip_ctrl, grasp_info=grasp_info,
        bugfixes=bugfixes, cup_init_tilt_deg=cup_init_tilt_deg,
        ctrlfixes=ctrlfixes, ctrlfix_parts=sorted(CTRLFIX_PARTS) if ctrlfixes else [],
        wp_modes=(list(wp_modes) if scenario == "crossing" else []),
        weld_gate_m=(WELD_ATTACH_GATE_M if weld_gate_m is None else float(weld_gate_m)),
    )
    return info


def right_gripper_opening(t: float, bounds: dict, n_cubes: int,
                           release_t: np.ndarray | None = None) -> float:
    """Open (1.0) by default; closed (0.0) from cube i's at_i grasp-close ramp
    through to its ACTUAL release. PLACEMENT-PRECISION FIX: the open ramp now
    follows the REAL (position-triggered or timeout) release instant,
    `release_t[i]` -- set by run_episode's main loop on the weld's falling
    edge, np.nan while still held -- instead of a fixed phase-clock time:
    the gripper stays shut through carry_i AND the move toward place_xyzs[i]
    until cube_pin_active (below) actually lets go, then ramps open over
    GRIP_OPEN_RAMP_S from that real instant. Falls back to the OLD fixed
    release_i-hold ramp when `release_t` is omitted (kept for callers that
    don't wire the new state). Generalized from the old single-cube version
    by looping the SAME per-cube logic over every selected cube's
    `at_i`/`release_i` phase-name bounds (only one cube's window can be
    active at a given t, since the phases are sequential/non-overlapping)."""
    for i in range(n_cubes):
        at_key, rel_key = f"at_{i}", f"release_{i}"
        if at_key not in bounds or rel_key not in bounds:
            break
        at_move_end, at_hold_end = bounds[at_key][1], bounds[at_key][2]
        rel_move_end, rel_hold_end = bounds[rel_key][1], bounds[rel_key][2]
        if at_move_end <= t < at_hold_end:
            s = (t - at_move_end) / max(at_hold_end - at_move_end, 1e-6)
            return float(1.0 - np.clip(s, 0.0, 1.0))
        if release_t is not None:
            if at_hold_end <= t < rel_hold_end:
                rt = release_t[i]
                if not np.isnan(rt) and t >= rt:
                    s = (t - rt) / GRIP_OPEN_RAMP_S
                    return float(np.clip(s, 0.0, 1.0))
                return 0.0
        else:
            if at_hold_end <= t < rel_move_end:
                return 0.0
            if rel_move_end <= t < rel_hold_end:
                s = (t - rel_move_end) / max(rel_hold_end - rel_move_end, 1e-6)
                return float(np.clip(s, 0.0, 1.0))
    return 1.0


def cube_pin_active(t: float, bounds: dict, n_cubes: int, *,
                     right_ee_xy: np.ndarray | None = None,
                     place_xy_by_idx: list | None = None,
                     right_ee_speed: float | None = None,
                     released: np.ndarray | None = None,
                     reach_tol: float = REACH_TOL,
                     speed_tol: float = RELEASE_SPEED_TOL_MS) -> tuple[bool, int | None]:
    """Returns (active, cube_index): cube i's weld is active from its at_i
    hold-end until release. Generalized from the old single-cube bool-only
    version -- at most one cube can be active at a given t (phases are
    sequential), so the first match wins.

    PLACEMENT-PRECISION FIX: release is now POSITION-TRIGGERED, not a fixed
    phase-clock instant. The OLD code deactivated the weld at exactly
    bounds[f"release_{i}"][0] -- the release_i phase's MOVE-START, i.e.
    before the arm had even begun heading toward place_xyzs[i] -- regardless
    of whether the EE ever got there. The candidate window is now extended
    through the WHOLE release_i phase (at_i hold-end -> release_i hold-end,
    covering both the move toward place_xyzs[i] and its hold), and within
    that window (once the move has actually started, t >= release_i's own
    move-start) the weld deactivates (this function returns False for cube
    i) as soon as BOTH right_ee_xy is within `reach_tol` of
    place_xy_by_idx[i] (xy-only) AND right_ee_speed < speed_tol (settling,
    not still sweeping through the target). If that condition never holds,
    the window itself still closes at release_i's own hold_end -- a TIMEOUT
    fallback, so a target the arm can't quite reach under safety-QP
    deflection still releases eventually and can never hang the demo.
    `released[i]` (set True by the caller on the actual falling edge, i.e.
    once this function has returned False for cube i) permanently disables
    cube i's window afterward, so the EE resting near an already-placed
    cube's spot can't re-trigger its weld. When right_ee_xy/place_xy_by_idx
    are omitted, falls back to the ORIGINAL fixed release_i-move-start
    cutoff (kept for callers that don't wire the new kwargs)."""
    for i in range(n_cubes):
        at_key, rel_key = f"at_{i}", f"release_{i}"
        if at_key not in bounds or rel_key not in bounds:
            break
        if released is not None and released[i]:
            continue
        at_hold_end = bounds[at_key][2]
        rel_move_start, _rel_move_end, rel_hold_end = bounds[rel_key]
        if right_ee_xy is None or place_xy_by_idx is None:
            if at_hold_end <= t < rel_move_start:
                return True, i
            continue
        if at_hold_end <= t < rel_hold_end:
            if t >= rel_move_start:
                dist = float(np.linalg.norm(right_ee_xy - place_xy_by_idx[i]))
                slow = True if right_ee_speed is None else right_ee_speed < speed_tol
                if dist < reach_tol and slow:
                    return False, i   # reached + settled -> release now
            return True, i   # still in-window, not yet released (timeout at rel_hold_end)
    return False, None


def count_placed_strict(model, data, cube_plan, cube_init_pos, center_xy, table_z,
                        radius: float = None, z_tol: float = 0.05, vel_tol: float = 0.05,
                        min_move_m: float = None, require_start_outside: bool = False) -> int:
    """(2) STRICT placed metric: only the SELECTED cubes (cube_plan), each must be
    in the keep zone (same radius/height/settled tests as count_placed_cubes)
    AND have moved > min_move_m from its t=0 position -- the stock metric also
    counts un-selected cubes that START inside the zone (seed 0: placed=1 at
    t=0). Same rule as experiments/imitation/il_common.placed_selected."""
    radius = KEEP_ZONE_RADIUS_M if radius is None else radius
    min_move_m = PLACED_STRICT_MIN_MOVE_M if min_move_m is None else min_move_m
    n = 0
    for c in cube_plan:
        b = c["body_id"]
        p = data.xpos[b]
        # require_start_outside (2026-09-29, metric only): cube_3 STARTS inside the keep zone and is selected on every
        # seed, so placed_strict counts its in-zone re-placement. This variant counts only cubes carried IN.
        if require_start_outside and np.hypot(cube_init_pos[b][0] - center_xy[0],
                                              cube_init_pos[b][1] - center_xy[1]) <= radius:
            continue
        if np.hypot(p[0] - center_xy[0], p[1] - center_xy[1]) > radius:
            continue
        if abs(p[2] - (table_z + cct.CUBE_HALF)) > z_tol:
            continue
        if np.linalg.norm(p[:2] - cube_init_pos[b][:2]) < min_move_m:
            continue
        da = model.body_dofadr[b]
        if np.linalg.norm(data.qvel[da:da + 3]) > vel_tol:
            continue
        n += 1
    return n


# ---------------------------------------------------------------------------
# Main per-step control loop. qp_enabled=False => pure computed-torque PD,
# no HOCBF-QP safety filter on EITHER arm (used only for the baseline
# deflection comparison -- never for the rendered video).
# ---------------------------------------------------------------------------
def run_episode(info: dict, *, controller: str = "v4c", qp_enabled: bool = True, capture: bool = False,
                capture_stride: int = 5, cam_main=None, cam_zed_size: int = 96,
                spill_monitor=None, vision_cadence_steps: int = VISION_CADENCE_STEPS,
                max_steps: int | None = None, cup_in_barrier: bool = True, table_floor: bool = True,
                ee_speed_limit: bool = True, adaptive_yield: bool = False,
                inertial_cup: bool = True, controller_impl: str | None = None):
    """spill_monitor=None -> byte-identical to the original (geometric_only)
    loop: t_r is the plain wall-clock-relative right-arm time, no freeze.
    spill_monitor=<RealWMSpillMonitor> -> config="geometric_spill": the WM
    is consulted every `vision_cadence_steps` physics steps (veto held
    between updates); the right-arm deposit reference clock (t_r, driving
    track_r + the cube weld window) advances ONLY on un-vetoed steps.
    max_steps: truncate the episode early (SHORT foreground smoke only --
    the full DURATION_S/N_STEPS episode is unaffected when None).

    cup_in_barrier (default True, the FIX): when True, info["cup_barrier_
    keypoints"] (rim ring + body points, rigidly tracked on the LEFT EE) is
    passed as sim_common.all_pair_barriers' extra_ego_keypoints every step,
    so BOTH arms' QPs (and mpc7 under --controller v7c, which consumes the
    same `pairs`) gain "avoid the cup too" HOCBF rows -- this is what makes
    the RIGHT arm's return/park leg stay clear of the held cup. False
    reproduces the pre-fix hole (cup absent from the barrier; used only for
    the ON-vs-OFF verification comparison, never for the real demo).

    table_floor (default True, the TABLE-FLOOR BARRIER FIX -- see the module
    constants above): when True, EACH arm's own real collision-proxy
    keypoints (sim_common.barrier_keypoints(model, EGO_PREFIX / INT_PREFIX),
    the SAME set already used for the arm-arm barrier) get extra HOCBF-QP
    rows (sim_common.table_plane_barrier_rows) keeping them above the
    tabletop (world z >= TABLE_TOP_Z - TABLE_FLOOR_MARGIN_M), appended
    alongside that arm's existing arm-arm rows in the v4c decentralized
    box-QP branch below ONLY (--controller v7c's mpc7.JointMPCv7 is reused
    unmodified and does not consume these rows). False reproduces the
    pre-fix hole (no table awareness at all in either QP) -- used only for
    the OFF-vs-ON measurement/verification, never for the real demo.
    left_min_arm_z/right_min_arm_z are logged EVERY step regardless of this
    flag (unconditional, like right_cup_clearance_m above), so the ON/OFF
    comparison is apples-to-apples.

    controller="v4c" (default, CHANGE 2): unchanged decentralized cooperative
    HOCBF-QP (v4-C*) on both arms, exactly as before -- byte-identical when
    qp_enabled=True, no MPC code touched.
    controller="v7c": once qp_enabled (the no-QP `not qp_enabled` baseline
    ALWAYS takes the plain-clip path below regardless of controller -- that
    baseline is deliberately safety-filter-agnostic) AND t_r>=0 (the crossing
    scenario has started -- see INTERFACE NOTE below for why the initial lift
    itself stays on the v4c ramp+QP path), BOTH arms' torques come from a
    SINGLE JointMPCv7.step() call (experiments/ab_smooth_vs_bangbang/mpc7.py,
    reused unmodified) each step, replacing the two decentralized box-QP
    solves. tau_task_l/tau_task_r (the v4c computed-torque PD-toward-track
    reference) are still computed every step under v7c too -- NOT fed into
    the MPC, only used as the common "naive" baseline for the
    left/right_qp_deflect_nm comparison metric so both controllers are
    judged against the identical reference law.

    INTERFACE NOTE (v7c): mpc7.grasp_fragility's ego-priority signal is
    derived partly from REAL pad<->box contact forces (this scene's
    analog: left_arm_link7/8_pad <-> the cup). This scene's cup is held via
    scene_setup.reattach_cup_rigidly (a kinematic reattach), not a real
    finger pinch, so that contact is never registered -> found_contact=False
    every step -> slip_ratio defaults to 1.0 -> fragility saturates at 1.0
    (max caution) for the whole episode. So under v7c, the ego (left/cup arm)
    priority weights (REG_EFFORT_EGO/W_JERK_EGO/W_EE_HOLD) sit at mpc7's
    fixed ceiling throughout, rather than being state-adaptive -- structurally
    similar to v4-C's own fixed INTRUDER_EXTRA_MARGIN (also a constant, not
    state-adaptive, margin), just via mpc7's cost weights instead of a QP
    constraint margin. mpc7.py itself is NOT modified to work around this.

    INTERFACE NOTE #2 (v7c): mpc7's JointMPCv7 regulates the ego arm toward a
    SINGLE FIXED q0_left (set at construction) via a short (NP=32, 160ms)
    predictive horizon -- it was designed/validated (run_ab7.py) for an
    ALREADY-SETTLED static hold pose reacting to an approaching intruder, not
    for performing a large motion itself. Measured directly: routing this
    scene's own ~11cm LEFT-arm LIFT through the MPC from t=0 (q0_left=q_lift
    far from the arm's actual t=0/grasp pose) produced ~83% infeasible
    steps, some multi-second solves, and a large uncontrolled cup overshoot
    in a --max-steps smoke test. FIX (not a workaround to mpc7.py itself):
    the LEFT arm's initial lift-to-q_lift stays on the existing v4c
    ramp+decentralized-QP path (t_r<0, right arm still at home -- no
    cooperative interaction needed yet anyway); v7c's JointMPCv7 only takes
    over BOTH arms once t_r>=0 (the crossing scenario begins, left arm
    already settled near q_lift -- a small initial error, matching mpc7's
    validated regime).
    """
    model, data = info["model"], info["data"]
    scenario = info["scenario"]
    left_dofadr, right_dofadr = info["left_dofadr"], info["right_dofadr"]
    left_qposadr, right_qposadr = info["left_qposadr"], info["right_qposadr"]
    left_ee_body, right_ee_body = info["left_ee_body"], info["right_ee_body"]
    cup_body = info["cup_body"]
    # --grasp-mode (build_episode decides; default faked => every `real` branch
    # below is skipped and the loop is byte-identical to before).
    grasp_mode = info.get("grasp_mode", "faked")
    real_grasp = grasp_mode == "real"
    grip_ctrl = info.get("grip_ctrl")
    if real_grasp:
        if qp_enabled and controller not in ("flmpc", "flmpc_decentB"):
            raise ValueError("--grasp-mode real requires --controller flmpc|flmpc_decentB "
                             "(FL-MPC drives the lift of the really-gripped cup)")
        if scenario != "crossing":
            raise ValueError("--grasp-mode real is implemented for --scenario crossing only")
        assert grip_ctrl is not None, "real grasp_mode needs info['grip_ctrl'] (build_episode)"
    left_pad_ids = (info["left_pad7_geom"], info["left_pad8_geom"])
    bugfixes = bool(info.get("bugfixes", False))
    weld_gate_m = info.get("weld_gate_m")
    # --ctrlfixes parts (all False when ctrlfixes is off -> byte-identical)
    cf_gravity = _cf(info, "gravity")
    _is_fl = controller in ("flmpc", "flmpc_decentB") and qp_enabled
    # --controller-impl (None -> module CONTROLLER_IMPL, library default "py" = unchanged)
    _impl = CONTROLLER_IMPL if controller_impl is None else controller_impl
    assert _impl in CONTROLLER_IMPL_CHOICES, f"unknown controller_impl {_impl!r}"
    if _impl != "py" and controller == "flmpc_decentB":
        raise ValueError("--controller-impl rt/rt_pipe is implemented for --controller flmpc only")
    _rt_impl = _impl != "py" and controller == "flmpc" and qp_enabled
    cf_speedcap = _cf(info, "speedcap") and _is_fl and ee_speed_limit
    cf_aw = _cf(info, "antiwindup") and _is_fl and scenario == "crossing"
    cf_aw_clamp = cf_aw and CTRLFIX_AW_MODE == "clamp"
    # the clock-freeze machinery: everywhere ("freeze" mode) or, in "clamp" mode, only
    # inside the grasp windows (CTRLFIX_AW_GRASP_FREEZE) so the clock-timed weld waits
    # for the gripper to actually reach the cube (those poses are barrier-clear by the
    # F2 selection, so the freeze there cannot deadlock on the crossing dodge).
    cf_aw_grasp_only = cf_aw_clamp and CTRLFIX_AW_GRASP_FREEZE
    cf_aw = cf_aw and (CTRLFIX_AW_MODE == "freeze" or cf_aw_grasp_only)
    _aw_grasp_windows = ([(b_[0], b_[2]) for n_, b_ in info["bounds"].items() if n_.startswith("at_")]
                         if (cf_aw_grasp_only and info.get("bounds")) else [])
    aw_clamped = 0
    _cone_anchored = False
    cf_tabler = _cf(info, "tableradius")
    cf_delay = int(CTRLFIX_DELAY_STEPS) if (_cf(info, "delaycomp") and _is_fl) else 0
    if _rt_impl and _impl == "rt_pipe":
        if cf_delay > 0:
            raise ValueError("--controller-impl rt_pipe with ctrlfix delaycomp > 0 is not supported")
        cf_delay = 1   # rt_pipe: the plan computed now is applied at the NEXT step -> refs one step ahead
    cf_taumax = (np.asarray(CTRLFIX_TAU_MAX_VEC, dtype=np.float64)
                 if (_cf(info, "taumax") and CTRLFIX_TAU_MAX_VEC is not None) else None)
    cf_cone = int(CTRLFIX_CONE_FACETS) if _cf(info, "cone") else 0
    _tab_kw = dict(kp_radius=True) if cf_tabler else {}
    right_site_id = model.site("right_arm_ee_grasp").id
    # free-joint dof address of the cup, for the INERTIAL CUP-CARRY fix (see loop).
    _cup_dofadr = int(model.jnt_dofadr[model.body_jntadr[cup_body]])

    # CUP-TILT MEASUREMENT (logging only, not a constraint): world-"up"
    # expressed in the LEFT EE's local frame at the SETTLED q_lift hold pose
    # (via a throwaway scratch mj_forward -- SAME "vertical" definition
    # mpc_fl.FLMPC's own cup-upright rows lazily capture at their first
    # step()). Used below to log log["left_cup_tilt_rad"] = the cup's actual
    # tilt off vertical every step (arccos of the world-z component of the
    # rotated axis), independent of controller -- so run_episode's smoke/eval
    # callers can verify the cup-upright behavior regardless of whether the
    # mpc_fl tilt CONSTRAINT is engaged for this run.
    _tilt_scratch = mujoco.MjData(model)
    _tilt_scratch.qpos[:] = data.qpos
    _tilt_scratch.qpos[left_qposadr] = info["q_lift"]
    mujoco.mj_forward(model, _tilt_scratch)
    _cup_axis_local_log = _tilt_scratch.xmat[left_ee_body].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
    cube_plan = info["cube_plan"]   # list of {body_id, eq_id, name, place_xyz}, one per selected cube (crossing only)
    n_cubes = len(cube_plan)
    track_l, track_r, bounds = info["track_l"], info["track_r"], info["bounds"]

    assert controller in CONTROLLER_CHOICES, f"unknown controller {controller!r}"

    left_j7 = model.actuator("left_arm_joint7_pos").id
    left_j8 = model.actuator("left_arm_joint8_pos").id
    right_finger_act = bt.finger_actuators(model, "right")

    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)

    _flt_a, _flt_r = sc.FILTER_LOWPASS_ALPHA, sc.FILTER_RATE_LIMIT_NM_S
    if controller == "v4c" and qp_enabled and V4C_TORQUE_FILTER is not None:
        _flt_a, _flt_r = float(V4C_TORQUE_FILTER[0]), float(V4C_TORQUE_FILTER[1])
    tau_filter_l = sc.TorqueCommandFilter(6, _flt_a, np.full(6, _flt_r))
    tau_filter_r = sc.TorqueCommandFilter(6, _flt_a, np.full(6, _flt_r))

    alpha1_int = alpha2_int = run_ab4.INTRUDER_ALPHA["C"]
    # V4C RETUNE KNOBS (v4c only; every one None/default -> byte-identical, see V4C_* above)
    _v4c = (controller == "v4c") and qp_enabled
    if _v4c and V4C_INT_ALPHA is not None:
        alpha1_int = alpha2_int = float(V4C_INT_ALPHA)
    _v4c_tab_kw = {}
    if _v4c and V4C_TABLE_ALPHA is not None:
        _v4c_tab_kw = dict(alpha1=float(V4C_TABLE_ALPHA), alpha2=float(V4C_TABLE_ALPHA))
    _v4c_metric = V4C_QP_METRIC if _v4c else "torque"
    _v4c_aw = _v4c and V4C_ANTIWINDUP_M is not None and scenario == "crossing"
    _v4c_pre = _v4c and V4C_FILTER_MODE == "pre"
    _v4c_y_l = _v4c_y_r = None          # previously APPLIED torques (V4C_FILTER_MODE="pre")
    _v4c_brake_kd = (float(V4C_INFEAS_BRAKE_KD) if (_v4c and V4C_INFEAS_BRAKE_KD is not None)
                     else INFEAS_BRAKE_KD)

    left_geoms, right_geoms = run_ab4.left_right_geom_sets(model)

    # --controller v7c setup (CHANGE 2): build the cooperative MPC-HOCBF ONCE,
    # only when it will actually be used (qp_enabled=True -- the no-QP
    # baseline never touches it). Ego (left/cup arm) is regulated toward
    # info["q_lift"] -- the settled lift-and-hold pose -- since mpc7's ego
    # target is a single fixed hold config (unlike v4c's smoothstep-eased
    # track_l lift ramp); see the INTERFACE NOTE above.
    joint_mpc = mpc_ctrl = None
    if controller == "v7c" and qp_enabled:
        mpc_ctrl = CupCollectMpcCtrl(model, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
                                      left_ee_body, cup_body, right_ee_body=info["right_ee_body"])
        # EE-SPEED-LIMIT FIX part (2): hard in-loop cap on the intruder EE
        # (v_ee_max=V_EE_MAX_MS), enforced inside JointMPCv7.step on the applied
        # (first-block) intruder torque. None when the fix is off -> byte-identical v7.
        joint_mpc = mpc7_mod.JointMPCv7(
            model, mpc_ctrl, info["q_lift"],
            v_ee_max=(V_EE_MAX_MS if ee_speed_limit else None),
            ee_body_id_int=info["right_ee_body"])
        mpc7_mod.warmup()
        from controller_core.constrained_box_qp import numba_warmup as _box_qp_warmup  # noqa: E402
        _box_qp_warmup()
        left_pad7_geom, left_pad8_geom = info["left_pad7_geom"], info["left_pad8_geom"]
        cup_frag_geom = info["cup_frag_geom"]
        cup_rel_pos_frag0 = info["cup_rel_pos0"].copy()
        cup_rel_quat_frag0 = info["cup_rel_quat0"].copy()

    # --controller flmpc setup: the feedback-linearized cooperative MPC
    # (experiments/ab_smooth_vs_bangbang/mpc_fl.py's FLMPC), built ONCE, only
    # when it will actually be used. Same LEFT=ego (holds info["q_lift"]) /
    # RIGHT=int (tracks the moving track_r) roles as v7c above -- see the
    # flmpc branch in the control loop below. NC=4,L=8 => NP=32 steps =
    # 160ms horizon @ DT=0.005 (v7c's own horizon length).
    fl = None
    if controller in ("flmpc", "flmpc_decentB") and qp_enabled:
        # W_POS_EGO lowered from the mpc_fl.FLMPC default (3000 -> 800): the
        # cup-upright tilt constraint + new cup-arm displacement-leash
        # constraint (both wired into the fl.step() call below) now do the
        # job of keeping the cup arm near its hold pose / upright -- via
        # CONSTRAINTS, not a crude high hold-weight -- so the tracking cost
        # can be loosened enough for the ego arm to actually cooperate
        # (yield) with collision avoidance inside the leash. W_VEL_EGO/
        # W_JERK (smoothness) are left at their FLMPC defaults.
        #
        # ADAPTIVE COOPERATIVE YIELDING (this task, default OFF -- see
        # mpc_fl.FLMPC's __init__ docstring for the full derivation): when
        # adaptive_yield=True, the anticipatory clearance margin widens
        # (0.02 -> up to 0.06 m) as the worst pair closes fast, so both arms
        # yield EARLIER and the anticipatory-clearance QP rows stay feasible
        # more often (the actual rate-lever -- avoiding the QP-infeasibility
        # cliff, not a fallback that only softens it); the cup(ego) arm's
        # share of that yielding is separately modulated by the SAME urgency
        # signal AND the live cup tilt (W_POS_EGO_MIN/LEASH_MAX_URGENT_M
        # loosen the cup arm's hold weight/leash toward these bounds when
        # urgency is high and the cup is upright, and revert toward the
        # 800.0/CUP_LEASH_M baseline above as tilt approaches
        # CUP_TILT_SAFE_FRAC of CUP_TILT_MAX_RAD). Off by default -> exactly
        # the prior fl = mpc_fl.FLMPC(model, NC=4, L=8, W_POS_EGO=800.0)
        # behavior (adaptive_yield=False is FLMPC's own default too).
        fl_kwargs = dict(NC=4, L=8, W_POS_EGO=800.0)
        if adaptive_yield:
            fl_kwargs.update(adaptive_yield=True, H_MARGIN_URGENT=0.06, URGENCY_HDOT_REF=0.15,
                              W_POS_EGO_MIN=200.0, LEASH_MAX_URGENT_M=0.14, CUP_TILT_SAFE_FRAC=0.6)
        # ABLATION OVERRIDES (see FL_W_JERK_OVERRIDE/FL_W_POS_INT_OVERRIDE
        # module-level docstring above) -- both default None -> unchanged.
        if FL_W_JERK_OVERRIDE is not None:
            fl_kwargs["W_JERK"] = FL_W_JERK_OVERRIDE
        if FL_W_POS_INT_OVERRIDE is not None:
            fl_kwargs["W_POS_INT"] = FL_W_POS_INT_OVERRIDE
        # Option B: decentralized worst-case variant reuses the SAME kwargs/roles
        # (mpc_fl_decentralized.FLMPCDecentralized subclasses FLMPC; identical
        # .NP and .step() signature) so the control-loop branch below is shared.
        # --ctrlfixes FL-MPC options (none added when ctrlfixes is off -> identical ctor)
        if cf_speedcap:
            fl_kwargs.update(ee_speed_cap=V_EE_MAX_MS, ee_speed_body_int=info["right_ee_body"])
        if cf_delay > 0:
            fl_kwargs.update(delay_steps=cf_delay)
        if cf_taumax is not None:
            fl_kwargs.update(tau_max_vec=cf_taumax)
        if cf_cone > 0:
            fl_kwargs.update(cone_mode="true")
        if controller == "flmpc_decentB":
            fl = mpc_fl_decentralized.FLMPCDecentralized(model, **fl_kwargs)
        elif _rt_impl:
            fl = _make_rt_flmpc(model, fl_kwargs, _impl)
        else:
            fl = mpc_fl.FLMPC(model, **fl_kwargs)

    # CUP-IN-BARRIER FIX: right-arm-vs-cup keypoints, resolved ONCE outside the
    # loop (both are static lists; sim_common.barrier_keypoints is itself
    # cached per-model). `cup_kps_for_barrier` is what actually gets fed into
    # the QP (via all_pair_barriers' extra_ego_keypoints, gated on
    # cup_in_barrier); `cup_kps_for_log`/`right_kps_for_log` are used
    # UNCONDITIONALLY below to log the verification metric (right-arm<->cup
    # clearance) regardless of whether the fix is ON or OFF, so the ON/OFF
    # comparison is apples-to-apples.
    cup_kps_for_log = info["cup_barrier_keypoints"]
    right_kps_for_log = sc.barrier_keypoints(model, sc.INT_PREFIX)
    cup_kps_for_barrier = cup_kps_for_log if cup_in_barrier else None

    # TABLE-FLOOR BARRIER FIX: each arm's own real collision-proxy keypoints
    # (SAME set the arm-arm barrier already uses) -- resolved ONCE here
    # (barrier_keypoints is itself cached per model), used both for the extra
    # QP rows below (gated on `table_floor`) AND for the UNCONDITIONAL
    # left/right_min_arm_z logging (so OFF-vs-ON is apples-to-apples, same
    # pattern as cup_kps_for_log/right_kps_for_log above).
    left_table_kps = sc.barrier_keypoints(model, sc.EGO_PREFIX)
    # FINGERTIP KEYPOINTS: the INT_PREFIX collision-proxy set covers the arm
    # LINKS but not the actual gripper FINGERTIPS/wrist, so the tip mesh could
    # still dip below the table even while the link-proxy stayed above. Append
    # the right gripper's finger pads + wrist base as extra table-floor
    # keypoints (body_id, local_offset, radius) -- SAME format barrier_keypoints
    # uses -- so the barrier holds the ACTUAL end-effector above the surface,
    # and right_min_arm_z (logged over right_table_kps) reflects the tip. These
    # go ONLY into the table-floor set, NOT the arm-arm barrier or the cup
    # clearance metric (which stay on right_kps_for_log, untouched).
    _tip_specs = [("right_arm_link7_pad", 0.006), ("right_arm_link8_pad", 0.006)]
    fingertip_kps = []
    for _gname, _r in _tip_specs:
        _gid = model.geom(_gname).id
        fingertip_kps.append((int(model.geom_bodyid[_gid]), model.geom_pos[_gid].copy(), _r))
    fingertip_kps.append((model.body("right_arm_gripper_base").id, np.zeros(3), 0.01))
    right_table_kps = list(right_kps_for_log) + fingertip_kps   # new list; right_kps_for_log untouched
    table_top_z = info["table_top_z"]
    if _rt_impl:
        # robot-only dynamics source + static keypoint/table wiring for rt_flmpc (scene ids are
        # mapped by name; the lean model's parameters are checked bit-for-bit against `model`).
        import rt_dynamics  # noqa: E402  (experiments/realtime, on sys.path via _make_rt_flmpc)
        fl.configure_lean(rt_dynamics.LeanRobotDynamics(model, xml_path=cct.YOR_CUP_COLLECT_XML), model,
                          left_qposadr, left_dofadr, right_qposadr, right_dofadr, left_ee_body,
                          cup_body=cup_body, right_ee_body=right_ee_body, speed_body=info["right_ee_body"],
                          ego_extra_kps=cup_kps_for_barrier, left_table_kps=left_table_kps,
                          right_table_kps=right_table_kps, table_z=table_top_z,
                          table_margin=TABLE_FLOOR_MARGIN_M, table_kp_radius=cf_tabler, table_gravfix=cf_gravity)

    n_steps_total = info["n_steps_total"]
    n_steps = n_steps_total if max_steps is None else int(min(n_steps_total, max_steps))
    # F1 ANTI-WINDUP (ctrlfixes, flmpc crossing only): the right-arm reference
    # clock is frozen while the right-EE tracking error exceeds CTRLFIX_AW_ERR_M,
    # for at most CTRLFIX_AW_MAX_FREEZE_S in total; the episode runs n_steps +
    # (frozen steps) so the (delayed) task still finishes. Buffers are allocated
    # for the worst case and truncated after the loop. Off -> n_alloc == n_steps.
    n_steps_run = n_steps
    aw_budget = int(round(CTRLFIX_AW_MAX_FREEZE_S / sc.DT)) if cf_aw else 0
    aw_frozen = 0
    aw_clock = 0.0
    n_steps = n_steps + aw_budget          # allocation size (== n_steps_run when off)
    log = {k: np.zeros(n_steps) for k in
           ["t", "h_min", "dist_min", "collision", "spilled", "deposited", "placed",
            "qp_feasible_left", "qp_feasible_right", "left_qp_deflect_nm",
            "right_qp_deflect_nm", "cup_z", "cube_pinned", "solve_time",
            "spill_risk", "spill_veto", "right_track_t", "left_tilt_t",
            "right_cup_clearance_m", "left_min_arm_z", "right_min_arm_z",
            "left_cup_tilt_rad"]}
    log["left_q"] = np.zeros((n_steps, 6))
    log["right_q"] = np.zeros((n_steps, 6))
    log["left_ee_pos"] = np.zeros((n_steps, 3))
    log["right_ee_pos"] = np.zeros((n_steps, 3))
    log["cup_pos"] = np.zeros((n_steps, 3))
    # WM-REQUIREMENT LADDER (experiments/wm_requirement/): the cup body's own
    # world orientation axis, per step -- additive, least-invasive (one new
    # log key + one line below); nothing existing renamed/removed. cup_pos
    # above already proved data.xpos[cup_body] is valid every step (the cup
    # is kinematically re-pinned via scene_setup.reattach_cup_rigidly before
    # this point in the loop), so data.xmat[cup_body] is equally valid here.
    log["cup_n_axis"] = np.zeros((n_steps, 3))
    # --grasp-mode metrics (logged in BOTH modes, read-only measurements; in
    # faked mode the pads never touch the cup, so pad_* are 0 there).
    for _key in ("true_cup_tilt_rad", "grip_slip_m", "pad_contacts", "pads_touching",
                 "finger_force_max_n", "pad_normal_force_n"):
        log[_key] = np.zeros(n_steps)
    log["grasp_mode_real"] = np.array(1.0 if real_grasp else 0.0)
    # CHANGE 2 metrics (logged regardless of --bugfixes)
    log["placed_strict"] = np.zeros(n_steps)
    log["placed_from_outside"] = np.zeros(n_steps)   # metric only: selected cubes that STARTED outside the zone
    log["cup_init_tilt_deg"] = np.array(float(info.get("cup_init_tilt_deg", np.nan)))
    log["bugfixes"] = np.array(1.0 if bugfixes else 0.0)
    log["ctrlfixes"] = np.array(1.0 if bool(info.get("ctrlfixes", False)) else 0.0)
    _n_plan = len(info["cube_plan"])
    log["weld_onset_t"] = np.full(_n_plan, np.nan)
    log["weld_onset_site_cube_dist_m"] = np.full(_n_plan, np.nan)
    log["weld_onset_site_target_dist_m"] = np.full(_n_plan, np.nan)
    log["weld_deferred_steps"] = np.zeros(_n_plan)
    # selected cubes' t=0 positions (free-joint qpos = world position; no kinematics call)
    cube_init_pos = {}
    for _c in info["cube_plan"]:
        _qa = int(model.jnt_qposadr[model.body_jntadr[_c["body_id"]]])
        cube_init_pos[_c["body_id"]] = data.qpos[_qa:_qa + 3].copy()

    frames = []
    renderer_main = renderer_zed = None
    if capture:
        renderer_main = mujoco.Renderer(model, height=720, width=1280)
        renderer_zed = mujoco.Renderer(model, height=cam_zed_size, width=cam_zed_size)
        zed_cam_id = model.camera("zed_cam").id

    weld_active_prev = False  # tracks the cube weld's ACTIVE/INACTIVE edge (see CHANGE 1 below)
    active_cube_idx_prev = None  # which cube_plan[] index the (currently or most-recently) active weld belongs to

    # PLACEMENT-PRECISION FIX state (position-triggered release -- see
    # cube_pin_active/right_gripper_opening): per-cube xy place targets (for
    # the reach-tolerance check), a "permanently released" latch (so a cube
    # already set down can never be re-pinned by a later position match),
    # the actual release time per cube (drives the gripper-open ramp off the
    # REAL release instant instead of a fixed phase time), and the previous
    # step's right-EE position (finite-difference -> measured settling speed).
    place_xy_by_idx = [c["place_xyz"][:2].copy() for c in cube_plan]
    cf_heldcube = _cf(info, "heldcube") and scenario == "crossing"
    _held_kp = None            # F8: (right_ee_body, local offset, radius) of the welded cube
    _reenable_pending = []     # F8: released cube indices whose arm-collision is not re-enabled yet
    released = np.zeros(n_cubes, dtype=bool)
    release_t = np.full(n_cubes, np.nan)
    prev_right_ee_pos = None

    # MULTI-CUBE placed-cube metric target (crossing only): the center of the
    # place-spot row (see build_episode) -- table_top_z + cube half-extent is
    # handled inside count_placed_cubes itself.
    placed_center_xy = None
    if cube_plan:
        placed_center_xy = np.mean(np.stack([c["place_xyz"][:2] for c in cube_plan]), axis=0)

    # Freeze-clock state for the spill-aware right-arm deposit reference
    # (config="geometric_spill" only; unused/inert when spill_monitor=None).
    right_elapsed = 0.0     # right-arm reference clock; advances only on un-vetoed steps
    # Freeze-clock state for the LEFT arm's own tilt/transfer reference
    # (scenario="spill" only) -- SAME mechanism, gated on the LEFT tilt
    # instead of the right-arm deposit clock (see run_episode's docstring).
    left_tilt_elapsed = 0.0
    held_veto = False
    held_risk = 0.0
    # CANONICAL ball-in-cup layout (home-keyframe offsets), used to carry the
    # balls through the lift; NOT the live post-init positions, which are
    # already scattered because init's grasp doesn't hold the cup (see
    # scene_setup.canonical_ball_rel + the SPILL-ORACLE FIX note below).
    _ball_rel = scene_setup.canonical_ball_rel(model, cup_body)
    _prev_qpos = None   # previous-step full qpos, for the inertial cup-carry velocity (see loop)

    for k in range(n_steps):
        if cf_aw and k >= n_steps_run + aw_frozen:
            break
        t = k * sc.DT

        # -- gripper ctrls (position-servo, real; always applied) --
        if real_grasp:
            # REAL grasp: grip-force control (servo-emulated, finger forcerange
            # +-20 N) maintained EVERY step for the whole episode.
            grip_ctrl.apply(data)
        else:
            data.ctrl[left_j7] = 0.0
            data.ctrl[left_j8] = 0.0
        if scenario == "crossing":
            if bugfixes:
                # (3) GRIPPER CLOCK FIX: bounds are on the right-track clock t_r
                # (right_elapsed when the WM freeze-clock is active).
                _t_r_grip = right_elapsed if spill_monitor is not None else t - RIGHT_START_T
                if cf_aw and t >= RIGHT_START_T:
                    _t_r_grip = aw_clock   # F1 anti-windup clock (== the t_r used this step)
                grip_r = right_gripper_opening(_t_r_grip, bounds, n_cubes, release_t) if t >= RIGHT_START_T else 1.0
            else:
                grip_r = right_gripper_opening(t, bounds, n_cubes, release_t) if t >= RIGHT_START_T else 1.0
        else:
            grip_r = 1.0   # "spill": right arm just holds home, gripper stays open the whole episode
        bt.set_gripper(data, right_finger_act, grip_r)

        mujoco.mj_forward(model, data)

        # -- kinematic reattachment (real cup carry technique; cube only while pinned) --
        # (--grasp-mode real: SKIPPED -- the cup is held only by the pad pinch)
        if not real_grasp:
            scene_setup.reattach_cup_rigidly(model, data, left_ee_body, cup_body,
                                              info["cup_rel_pos0"], info["cup_rel_quat0"])

        # -- SPILL-ORACLE FIX (2026-09-10): carry the balls rigidly WITH the cup
        # through the initial LIFT phase (t < RIGHT_START_T), then release them.
        # reattach_cup_rigidly teleports the cup with ZEROED velocity, so a
        # moving cup sweeps its walls through the loose balls and scatters them
        # out ("lift-loss") -- which made count_spilled read 6-8/8 balls out
        # from step 0, corrupting the spill metric (it measured the setup lift,
        # not the arm-arm dodges). Freezing the balls in the cup through the lift
        # and releasing them exactly when the crossing/tilt begins
        # (RIGHT_START_T == LEFT_TILT_START_T) makes count_spilled ~0 at the
        # start of the measured window, so the spill signal reflects only
        # DODGE/TILT-induced spill. Balls are ordinary free bodies from
        # RIGHT_START_T onward. Uses the CANONICAL home-keyframe layout (see
        # _ball_rel above) because init's grasp doesn't actually hold the cup,
        # so the balls have already scattered by the time the loop starts --
        # carrying with the canonical layout forces them back inside.
        if t < RIGHT_START_T and not real_grasp:   # real: balls ride in the gripped cup under physics
            scene_setup.carry_balls_with_cup(model, data, cup_body, _ball_rel)

        # -- INERTIAL CUP-CARRY (2026-09-11, #31): reattach_cup_rigidly pinned the
        # cup with ZEROED velocity, so its walls had no velocity in contact
        # resolution and the loose balls felt only geometric wall-sweep, not the
        # cup's true inertial forces (empirically: the effective-gravity cone gave
        # zero spill benefit because spill wasn't inertial). Set the cup's
        # free-joint qvel to the finite-difference of its pinned pose (exact
        # free-joint convention via mj_differentiatePos) BEFORE mj_step, so the
        # moving walls impart real momentum to the balls = faithful sloshing. The
        # cup POSITION is still re-pinned to the EE each step (kinematic carry
        # unchanged); only its per-step velocity is corrected. Cup dofs only --
        # the arms' physics qvel is untouched. inertial_cup=False restores the old
        # zeroed-velocity behavior.
        # (--grasp-mode real: SKIPPED -- the cup's velocity is its own physics)
        if inertial_cup and _prev_qpos is not None and not real_grasp:
            _qvel_fd = scene_setup.cup_velocity_from_motion(model, _prev_qpos, data.qpos, sc.DT)
            data.qvel[_cup_dofadr:_cup_dofadr + 6] = _qvel_fd[_cup_dofadr:_cup_dofadr + 6]
        _prev_qpos = data.qpos.copy()

        # -- spill-aware semantic layer (config="geometric_spill" only): consult the
        # REAL WM monitor at a vision cadence (camera rate << control rate), hold its
        # veto/risk between updates, and use it to freeze the right-arm deposit clock
        # below -- byte-identical to the original (no freeze) when spill_monitor=None. --
        if spill_monitor is not None and k % vision_cadence_steps == 0:
            verdict = spill_monitor.assess(safety_mod.SemanticObservation(model=model, data=data, t=t))
            held_veto = bool(verdict.veto)
            held_risk = float(verdict.risk)

        if scenario == "crossing":
            t_r_wall = t - RIGHT_START_T   # plain wall-clock right-arm time (unfrozen)
            if spill_monitor is not None:
                if t_r_wall >= 0 and not held_veto:
                    right_elapsed += sc.DT
                t_r = right_elapsed if t_r_wall >= 0 else t_r_wall
            else:
                t_r = t_r_wall
            if cf_aw and t_r_wall >= 0:
                # F1 ANTI-WINDUP: use the (possibly frozen) clock; advance it for the
                # NEXT step only if the right EE is within CTRLFIX_AW_ERR_M of the
                # reference at the current clock (linearized: ||Jp_r (q_ref - q)||).
                t_r = aw_clock
                _Jp_r_aw = sc.body_jac_full(model, data, right_ee_body)[0][:, right_dofadr]
                _e_aw = float(np.linalg.norm(_Jp_r_aw @ (track_r.sample(t_r) - data.qpos[right_qposadr])))
                _aw_in_win = (not cf_aw_grasp_only) or any(a_ <= t_r < c_ for a_, c_ in _aw_grasp_windows)
                if _e_aw > CTRLFIX_AW_ERR_M and aw_frozen < aw_budget and _aw_in_win:
                    aw_frozen += 1
                else:
                    aw_clock += sc.DT
            # PLACEMENT-PRECISION FIX: measured EE speed (finite-difference of
            # the ACTUAL simulated right-EE position across control steps, not
            # the scripted reference) -- the "settling" half of the
            # position-triggered release gate below.
            right_ee_pos_now = data.xpos[right_ee_body].copy()
            right_ee_speed_now = (float(np.linalg.norm(right_ee_pos_now - prev_right_ee_pos) / sc.DT)
                                   if prev_right_ee_pos is not None else 0.0)
            prev_right_ee_pos = right_ee_pos_now
            # -- CHANGE 1: cube "magnet" WELD attach (real constraint force),
            # replacing the old scripted-pin reattach_cup_rigidly(...cube...)
            # call. Activate exactly on the rising edge (captures the CURRENT
            # gripper-cube relative pose -- see cup_collect_task_utils.activate_weld)
            # and deactivate exactly on the falling edge; while active the weld
            # constraint holds every step via real physics, no per-step teleport.
            # MULTI-CUBE: pin_idx says WHICH cube (cube_plan[] index) is
            # currently held -- at most one at a time, phases are sequential --
            # so each cube gets its OWN weld eq activated/deactivated in turn.
            #
            # PLACEMENT-PRECISION FIX (release gate): the OLD deactivation was
            # a pure PHASE CLOCK (release_i's own move-START -- before the arm
            # had even begun heading toward place_xyzs[i]), regardless of
            # whether the right EE actually got there -- under the FL-MPC's
            # safety-constraint deflection the arm is often short/wide of the
            # target, so the cube got released nowhere near its place spot.
            # NOW: cube_pin_active only lets go once the right EE is within
            # REACH_TOL of that cube's place_xy AND settling (speed <
            # RELEASE_SPEED_TOL_MS), with a TIMEOUT fallback at release_i's
            # own hold_end so a target the arm can't quite reach still
            # releases (never hangs). --
            if t_r >= 0:
                pin_now, pin_idx = cube_pin_active(
                    t_r, bounds, n_cubes, right_ee_xy=right_ee_pos_now[:2],
                    place_xy_by_idx=place_xy_by_idx, right_ee_speed=right_ee_speed_now,
                    released=released)
            else:
                pin_now, pin_idx = False, None
            if pin_now and not weld_active_prev:
                _site = data.site_xpos[right_site_id]
                _d_tgt = float(np.linalg.norm(_site - cube_plan[pin_idx]["grasp_xyz"]))
                if weld_gate_m is not None and _d_tgt > weld_gate_m:
                    # (4) WELD PROXIMITY GATE: not at the grasp target yet -> defer
                    # (re-checked every step inside the pin window; never attaches
                    # from afar -> a MISSED pick if the site never gets there).
                    log["weld_deferred_steps"][pin_idx] += 1
                    pin_now, pin_idx = False, None
            if pin_now and not weld_active_prev:
                log["weld_onset_t"][pin_idx] = t
                log["weld_onset_site_cube_dist_m"][pin_idx] = float(np.linalg.norm(
                    data.site_xpos[right_site_id] - data.xpos[cube_plan[pin_idx]["body_id"]]))
                log["weld_onset_site_target_dist_m"][pin_idx] = float(np.linalg.norm(
                    data.site_xpos[right_site_id] - cube_plan[pin_idx]["grasp_xyz"]))
                cct.activate_weld(model, data, cube_plan[pin_idx]["eq_id"],
                                   right_ee_body, cube_plan[pin_idx]["body_id"])
                # COLLISION-FIDELITY: while this cube is welded to the gripper,
                # disable its arm-collision bits so the moving held cube does not
                # fight the weld (un-held cubes still collide with both arms).
                sc.set_cube_arm_collision(model, cube_plan[pin_idx]["body_id"], enabled=False)
                active_cube_idx_prev = pin_idx
                if cf_heldcube:
                    _Ree = data.xmat[right_ee_body].reshape(3, 3)
                    _held_kp = (right_ee_body,
                                _Ree.T @ (data.xpos[cube_plan[pin_idx]["body_id"]] - data.xpos[right_ee_body]),
                                cct.CUBE_HALF + TABLE_FLOOR_MARGIN_M + HELDCUBE_TABLE_CLEAR_M)
            elif not pin_now and weld_active_prev:
                cct.deactivate_weld(data, cube_plan[active_cube_idx_prev]["eq_id"])
                # restore the released cube's arm-collision (now a free obstacle again)
                if cf_heldcube:
                    _held_kp = None
                    _reenable_pending.append(active_cube_idx_prev)   # F8: deferred (see below)
                else:
                    sc.set_cube_arm_collision(model, cube_plan[active_cube_idx_prev]["body_id"], enabled=True)
                released[active_cube_idx_prev] = True   # latch: never re-pin this cube (see cube_pin_active)
                release_t[active_cube_idx_prev] = t_r if bugfixes else t   # ACTUAL release time (gripper-open ramp clock; (3))
            weld_active_prev = pin_now
            if _reenable_pending:
                # F8: re-enable a released cube's arm-collision once the grasp site is clear of it
                for _ci in list(_reenable_pending):
                    if (np.linalg.norm(data.site_xpos[right_site_id] - data.xpos[cube_plan[_ci]["body_id"]])
                            > HELDCUBE_REENABLE_DIST_M):
                        sc.set_cube_arm_collision(model, cube_plan[_ci]["body_id"], enabled=True)
                        _reenable_pending.remove(_ci)
            mujoco.mj_forward(model, data)
        else:
            # "spill" scenario: no cube pick, right arm never leaves home --
            # see the LEFT-arm tilt clock below for the freeze-gated motion.
            t_r = 0.0
            pin_now, pin_idx = False, None

        # -- joint-space references --
        q_left_des = track_l.sample(t)
        qd_left_des = track_l.sample_dot(t)

        # -- LEFT arm tilt/transfer (scenario="spill" only): FAST wrist tilt
        # added on top of the settled lift-hold, via eval/scene_setup.py's
        # validated transfer_tilt_offset/left_ee_jac_rot_pinv (see build_episode).
        # Driven by its OWN freeze-clock (`left_tilt_elapsed`) that advances
        # ONLY on un-vetoed steps -- the SAME freeze-clock mechanism the
        # "crossing" scenario uses on the right-arm deposit clock above
        # (`right_elapsed`), gated here on the LEFT tilt instead: on veto,
        # the tilt PAUSES (holds at its current angle) rather than snapping
        # ahead on release -- this is the whole point of the demo. --
        if scenario == "spill":
            t_tilt_wall = t - LEFT_TILT_START_T
            tilt_unfrozen = t_tilt_wall >= 0 and not held_veto
            if tilt_unfrozen:
                left_tilt_elapsed += sc.DT
            tilt_off = scene_setup.transfer_tilt_offset(
                left_tilt_elapsed, start_s=0.0, dur_s=SPILL_TILT_DUR_S,
                amp_rad=np.deg2rad(SPILL_TILT_AMP_DEG))
            tilt_off_dot = _tilt_offset_dot(left_tilt_elapsed, SPILL_TILT_DUR_S,
                                             np.deg2rad(SPILL_TILT_AMP_DEG)) if tilt_unfrozen else 0.0
            q_left_des = q_left_des + info["q_tilt_dir"] * tilt_off
            qd_left_des = qd_left_des + info["q_tilt_dir"] * tilt_off_dot

        if scenario == "crossing" and t_r >= 0:
            q_right_des = track_r.sample(t_r)
            qd_right_des = track_r.sample_dot(t_r)
            if _v4c_aw:
                # V4C_ANTIWINDUP_M (v4c analog of ctrlfix F1 "antiwindup" clamp): pull the PD
                # reference toward the current q so the right-EE error it commands is <= this.
                _Jp_v4aw = sc.body_jac_full(model, data, right_ee_body)[0][:, right_dofadr]
                _q_r_now = data.qpos[right_qposadr]
                _e_v4aw = float(np.linalg.norm(_Jp_v4aw @ (q_right_des - _q_r_now)))
                if _e_v4aw > V4C_ANTIWINDUP_M:
                    _s_v4aw = V4C_ANTIWINDUP_M / _e_v4aw
                    q_right_des = _q_r_now + _s_v4aw * (q_right_des - _q_r_now)
                    qd_right_des = _s_v4aw * qd_right_des
                    aw_clamped += 1
        else:
            q_right_des = info["q_home"]
            qd_right_des = np.zeros(6)

        # -- task torques (computed-torque PD + payload reflection) --
        M_full = sc.full_mass_matrix(model, data)
        M_ll = M_full[np.ix_(left_dofadr, left_dofadr)]
        bias_l = data.qfrc_bias[left_dofadr]
        if cf_gravity:
            # F3: gravcomp=1 puts gravity in qfrc_passive -> subtract it (and the
            # joint damping) so tau = M a + bias realizes q'' = a (no double gravity).
            bias_l = bias_l - data.qfrc_passive[left_dofadr]
        M_box, bias_box = sc.payload_reflected_terms(model, data, left_ee_body, cup_body, left_dofadr)
        M_ll = M_ll + M_box
        bias_l = bias_l + bias_box
        q_l = data.qpos[left_qposadr]
        qd_l = data.qvel[left_dofadr]
        qacc_des_l = sc.KP_TASK * (q_left_des - q_l) + sc.KD_TASK * (qd_left_des - qd_l)
        tau_task_l = M_ll @ qacc_des_l + bias_l
        M_ll_inv = np.linalg.inv(M_ll)

        M_rr = M_full[np.ix_(right_dofadr, right_dofadr)]
        bias_r = data.qfrc_bias[right_dofadr]
        if cf_gravity:
            bias_r = bias_r - data.qfrc_passive[right_dofadr]   # F3 (see bias_l above)
        if pin_now:
            # MULTI-CUBE: reflect the CURRENTLY-held cube's own body (pin_idx),
            # not a fixed single cube -- different cubes in cube_plan may have
            # (very slightly) different mass/inertia in principle, and this is
            # simply correct regardless.
            held_cube_body = cube_plan[pin_idx]["body_id"]
            M_cube, bias_cube = sc.payload_reflected_terms(model, data, right_ee_body, held_cube_body, right_dofadr)
            M_rr = M_rr + M_cube
            bias_r = bias_r + bias_cube
        q_r = data.qpos[right_qposadr]
        qd_r = data.qvel[right_dofadr]
        qacc_des_r = sc.KP_TASK * (q_right_des - q_r) + sc.KD_TASK * (qd_right_des - qd_r)
        tau_task_r = M_rr @ qacc_des_r + bias_r
        M_rr_inv = np.linalg.inv(M_rr)

        if real_grasp:
            # CUP-IN-BARRIER from the ACTUAL cup body pose this step (the cup may
            # slip/rotate in the grip): same rings/radius as build_cup_barrier_
            # keypoints, re-expressed as EE-local offsets (Jacobian via the arm).
            _rel_p, _rel_q = rg.cup_rel_pose_in_ee(data, left_ee_body, cup_body)
            cup_kps_for_log = build_cup_barrier_keypoints(left_ee_body, _rel_p, _rel_q)
            cup_kps_for_barrier = cup_kps_for_log if cup_in_barrier else None
        pairs = sc.all_pair_barriers(model, data, left_dofadr, right_dofadr,
                                      extra_ego_keypoints=cup_kps_for_barrier)
        # F8 "heldcube": the welded cube joins the right arm's TABLE-floor keypoints only
        _rtk_eff = right_table_kps if _held_kp is None else (right_table_kps + [_held_kp])
        qacc_left_known = data.qacc[left_dofadr]
        qacc_right_known = data.qacc[right_dofadr]

        lower = -sc.TAU_MAX * np.ones(6)
        upper = sc.TAU_MAX * np.ones(6)

        t_solve0 = time.perf_counter()
        _fl_stepped_this_k = False
        if not qp_enabled:
            # Safety-filter-agnostic baseline (no HOCBF-QP, no MPC): plain
            # clipped task torque -- used for the deflection-proof comparison
            # under EITHER controller.
            tau_l_qp = np.clip(tau_task_l, -sc.TAU_MAX, sc.TAU_MAX)
            tau_r_qp = np.clip(tau_task_r, -sc.TAU_MAX, sc.TAU_MAX)
            feasible_l = feasible_r = True
            solve_time = 0.0
        elif controller == "v7c" and scenario == "crossing" and t_r >= 0:
            # -- CHANGE 2: BOTH arms via the cooperative MPC-HOCBF (mpc7.py,
            # reused unmodified -- same solver + collision barrier `pairs`
            # already computed above). LEFT=ego/protected, RIGHT=intruder/
            # yields via mpc7's own grasp-fragility-scaled priority (see the
            # INTERFACE NOTE in this function's docstring).
            #
            # Gated on t_r>=0 (the crossing scenario has started, i.e. the
            # LEFT arm has already reached+settled at its q_lift hold pose
            # via the v4c-style ramp+QP branch below): mpc7's ego cost
            # regulates toward a SINGLE FIXED target (q0_left=info["q_lift"])
            # over a short 160ms horizon -- it assumes a small/near-zero
            # initial error (an already-settled hold, exactly run_ab7's own
            # validated regime), not a large jump target. Measured: routing
            # the initial ~11cm LIFT itself through the MPC from t=0 (q0_left
            # far from the arm's actual t=0 pose) produced ~83% infeasible
            # steps, multi-second worst-case solves, and a large uncontrolled
            # cup overshoot in a --max-steps smoke test -- a genuine mpc7
            # interface mismatch (static-hold-target assumption), NOT
            # something patched around here; see run_episode's docstring. --
            q_int_des_traj = np.stack([track_r.sample(t_r + tt * sc.DT) for tt in range(mpc7_mod.NP + 1)])
            qd_int_des_traj = np.stack([track_r.sample_dot(t_r + tt * sc.DT) for tt in range(mpc7_mod.NP + 1)])
            R_left = data.xmat[left_ee_body].reshape(3, 3)
            cup_rel_pos_now = R_left.T @ (data.xpos[cup_body] - data.xpos[left_ee_body])
            cup_rel_quat_now = quat_mul(quat_conj(data.xquat[left_ee_body]), data.xquat[cup_body])
            box_drift_trans = float(np.linalg.norm(cup_rel_pos_now - cup_rel_pos_frag0))
            box_drift_rot = float(quat_angle(quat_mul(quat_conj(cup_rel_quat_frag0), cup_rel_quat_now)))
            fragility = mpc7_mod.grasp_fragility(model, data, cup_frag_geom, left_pad7_geom, left_pad8_geom,
                                                  box_drift_trans, box_drift_rot, mujoco)
            # TABLE-FLOOR into the MPC too (the subagent's table_floor fix only
            # touched the v4c reactive branch): build the SAME per-arm HOCBF
            # floor rows here and hand them to JointMPCv7.step, which embeds
            # them onto the first-block ego/intruder torque. None when off.
            mpc_table_rows = None
            if table_floor:
                a_te, b_te = sc.table_plane_barrier_rows(
                    model, data, left_table_kps, left_dofadr, M_ll_inv, bias_l,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw)
                a_ti, b_ti = sc.table_plane_barrier_rows(
                    model, data, _rtk_eff, right_dofadr, M_rr_inv, bias_r,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw)
                mpc_table_rows = (a_te, b_te, a_ti, b_ti)
            tau_l_qp, tau_r_qp, feasible_mpc, mpc_solve_time = joint_mpc.step(
                model, data, q_int_des_traj, qd_int_des_traj, pairs, fragility,
                table_rows=mpc_table_rows)
            feasible_l = feasible_r = feasible_mpc
            solve_time = mpc_solve_time
        elif controller in ("flmpc", "flmpc_decentB") and scenario == "crossing" and t_r >= 0:
            # -- feedback-linearized cooperative MPC (mpc_fl.FLMPC): BOTH arms
            # via a joint QP over PLANNED ACCELERATIONS (tau = M*a+c realizes
            # q̈=a EXACTLY every step, so the horizon prediction is exact/
            # linear -- no time-varying-linearization rollout needed, unlike
            # mpc7/v7c). LEFT=ego holds the settled info["q_lift"] hold pose
            # (SAME gating/rationale as v7c above: t_r>=0, i.e. after the
            # LEFT arm's initial lift has already settled on the v4c ramp
            # path); RIGHT=int tracks the SAME track_r used by v7c/v4c.
            if cf_delay > 0:
                # F5: index 0 of the reference = when this command takes effect (t + d*DT)
                q_int_ref = np.stack([track_r.sample(t_r + (tt + cf_delay) * sc.DT) for tt in range(fl.NP + 1)])
                qd_int_ref = np.stack([track_r.sample_dot(t_r + (tt + cf_delay) * sc.DT)
                                       for tt in range(fl.NP + 1)])
            else:
                q_int_ref = np.stack([track_r.sample(t_r + tt * sc.DT) for tt in range(fl.NP + 1)])
                qd_int_ref = np.stack([track_r.sample_dot(t_r + tt * sc.DT) for tt in range(fl.NP + 1)])
            if cf_aw_clamp:
                # F1 ANTI-WINDUP (clamp): scale the int reference's deviation from the
                # current q (and its velocity) so the EE error entering the cost is
                # <= CTRLFIX_AW_ERR_M -- no wound-up pull when the arm is held back
                # (jam / barrier), and no clock stall (the task keeps its timing).
                _Jp_aw = sc.body_jac_full(model, data, right_ee_body)[0][:, right_dofadr]
                _e_aw = float(np.linalg.norm(_Jp_aw @ (q_int_ref[0] - q_r)))
                if _e_aw > CTRLFIX_AW_ERR_M:
                    _s_aw = CTRLFIX_AW_ERR_M / _e_aw
                    q_int_ref = q_r[None, :] + _s_aw * (q_int_ref - q_r[None, :])
                    qd_int_ref = _s_aw * qd_int_ref
                    aw_clamped += 1
            q_ego_ref = np.tile(info["q_lift"], (fl.NP + 1, 1))
            qd_ego_ref = np.zeros((fl.NP + 1, 6))
            # TABLE FLOOR (optional, same per-arm HOCBF rows v7c builds above,
            # in the usual TORQUE convention) -- FLMPC.step's `table_rows`
            # kwarg maps them into the equivalent ACCELERATION-space rows on
            # the first-block accel (see mpc_fl.FLMPC.step's docstring).
            fl_table_rows = None
            if table_floor:
                a_te, b_te = sc.table_plane_barrier_rows(
                    model, data, left_table_kps, left_dofadr, M_ll_inv, bias_l,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw)
                a_ti, b_ti = sc.table_plane_barrier_rows(
                    model, data, _rtk_eff, right_dofadr, M_rr_inv, bias_r,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw)
                fl_table_rows = (a_te, b_te, a_ti, b_ti)
            # CUP-UPRIGHT + CUP-ARM DISPLACEMENT LEASH (both optional
            # mpc_fl.FLMPC constraint rows -- see that module): keep the cup
            # vertical (tilt_max_rad) and keep the cup arm's Cartesian
            # excursion from its hold pose bounded (leash_max_m) while it
            # yields, instead of relying on a crude high hold-weight (see the
            # lowered W_POS_EGO at the fl = mpc_fl.FLMPC(...) construction
            # above).
            if real_grasp and cf_cone > 0 and CTRLFIX_CONE_ANCHOR == "once":
                # F4: anchor the cone ONCE (true cup axis in the EE frame at the first
                # crossing step), then bound the GRIPPER's rotation from there. Re-anchoring
                # every step on the live cup axis makes the tight cone CHASE a cup that
                # pivots in the hinge-like two-pad pinch: the EE rotates to right the
                # cup, the pinch transmits little of that rotation, the gripper tilts
                # further, and the grip slips (smoke: seed 26 gripper 38 deg, slip 25 cm).
                if not _cone_anchored:
                    fl._cup_axis_local = rg.cup_axis_in_ee(data, left_ee_body, cup_body)
                    _cone_anchored = True
            elif real_grasp:
                # cup-upright rows anchored on the TRUE cup axis (cup body pose), not a captured EE axis
                fl._cup_axis_local = rg.cup_axis_in_ee(data, left_ee_body, cup_body)
            if _rt_impl:
                # --controller-impl rt/rt_pipe: same options; M/c/Jacobians/keypoints/table rows
                # from the robot-only model inside ONE compiled call (sensing: joint state + the
                # held cup / welded cube pose).
                tau_l_qp, tau_r_qp, feasible_fl = fl.step_lean(
                    data, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
                    left_ee_body=left_ee_body, tilt_max_rad=CUP_TILT_MAX_RAD,
                    leash_ee_body=left_ee_body, leash_max_m=CUP_LEASH_M,
                    cup_cone_facets=(cf_cone if cf_cone > 0 else
                                     (CUP_CONE_FACETS if (not real_grasp or REAL_CUP_CONE_FACETS is None)
                                      else REAL_CUP_CONE_FACETS)),
                    ego_hold=False, table_floor=table_floor,
                    cup_pose=(data.xpos[cup_body].copy(), data.xmat[cup_body].reshape(3, 3).copy()),
                    cube=((cube_plan[pin_idx]["body_id"], data.xpos[cube_plan[pin_idx]["body_id"]].copy(),
                           data.xmat[cube_plan[pin_idx]["body_id"]].reshape(3, 3).copy()) if pin_now else None),
                    ego_extra_kps=cup_kps_for_barrier, right_table_kps=_rtk_eff)
            else:
                tau_l_qp, tau_r_qp, feasible_fl = fl.step(
                model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
                q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs,
                M_ego_extra=M_box, bias_ego_extra=bias_box,
                table_rows=fl_table_rows,
                left_ee_body=left_ee_body, tilt_max_rad=CUP_TILT_MAX_RAD,
                leash_ee_body=left_ee_body, leash_max_m=CUP_LEASH_M,
                cup_cone_facets=(cf_cone if cf_cone > 0 else
                                 (CUP_CONE_FACETS if (not real_grasp or REAL_CUP_CONE_FACETS is None)
                                  else REAL_CUP_CONE_FACETS)),
                # EGO POSITION-HOLD RELEASE (always False here). This flmpc branch
                # only runs once t_r>=0 (i.e. t>=RIGHT_START_T, the crossing): the
                # initial lift is driven by the else/computed-torque ramp branch,
                # NOT fl.step. So every time fl.step runs we are already past the
                # lift and want the LEFT arm's position-hold RELEASED -- it moves
                # AWAY from the approaching right arm and stays there (no spring-back
                # to its initial pose); velocity-damping + collision-avoidance + the
                # leash keep it at rest, safe, and bounded. (ego_hold=True is kept in
                # mpc_fl.FLMPC.step for other callers, e.g. a future teleop mode where
                # the protected arm must hold a commanded pose between dodges.)
                ego_hold=False)
            _fl_stepped_this_k = True
            if bugfixes:
                # (6) joint dry-friction feedforward (see the --bugfixes block comment)
                tau_l_qp = tau_l_qp + rg.friction_feedforward(model, left_dofadr, data.qvel[left_dofadr],
                                                              fl.a_ego_prev, sc.DT)
                tau_r_qp = tau_r_qp + rg.friction_feedforward(model, right_dofadr, data.qvel[right_dofadr],
                                                              fl.a_int_prev, sc.DT)
            feasible_l = feasible_r = feasible_fl
            solve_time = time.perf_counter() - t_solve0
        elif real_grasp and controller in ("flmpc", "flmpc_decentB") and scenario == "crossing" and t_r < 0:
            # -- --grasp-mode real LIFT: FL-MPC (not v4c) lifts the really-gripped
            # cup. Ego (left) TRACKS the orientation-holding lift reference
            # track_l over the horizon (ego_hold=True); int (right) holds q_home.
            # ALL hard rows active: arm-arm HOCBF (cup keypoints from the actual
            # cup pose, above), torque box, cup-upright tilt (anchored on the
            # TRUE cup axis), leash, table floor; real cup payload via M_box/bias_box.
            if cf_delay > 0:
                q_ego_ref = np.stack([track_l.sample(t + (tt + cf_delay) * sc.DT) for tt in range(fl.NP + 1)])
                qd_ego_ref = np.stack([track_l.sample_dot(t + (tt + cf_delay) * sc.DT) for tt in range(fl.NP + 1)])
            else:
                q_ego_ref = np.stack([track_l.sample(t + tt * sc.DT) for tt in range(fl.NP + 1)])
                qd_ego_ref = np.stack([track_l.sample_dot(t + tt * sc.DT) for tt in range(fl.NP + 1)])
            q_int_ref = np.tile(info["q_home"], (fl.NP + 1, 1))
            qd_int_ref = np.zeros((fl.NP + 1, 6))
            fl_table_rows = None
            if table_floor:
                a_te, b_te = sc.table_plane_barrier_rows(
                    model, data, left_table_kps, left_dofadr, M_ll_inv, bias_l,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw)
                a_ti, b_ti = sc.table_plane_barrier_rows(
                    model, data, _rtk_eff, right_dofadr, M_rr_inv, bias_r,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw)
                fl_table_rows = (a_te, b_te, a_ti, b_ti)
            fl._cup_axis_local = rg.cup_axis_in_ee(data, left_ee_body, cup_body)
            if _rt_impl:
                tau_l_qp, tau_r_qp, feasible_fl = fl.step_lean(
                    data, q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref,
                    left_ee_body=left_ee_body, tilt_max_rad=CUP_TILT_MAX_RAD,
                    leash_ee_body=left_ee_body, leash_max_m=CUP_LEASH_M,
                    cup_cone_facets=(cf_cone if cf_cone > 0 else
                                     (CUP_CONE_FACETS if REAL_CUP_CONE_FACETS is None else REAL_CUP_CONE_FACETS)),
                    ego_hold=True, table_floor=table_floor,
                    cup_pose=(data.xpos[cup_body].copy(), data.xmat[cup_body].reshape(3, 3).copy()),
                    cube=None, ego_extra_kps=cup_kps_for_barrier, right_table_kps=_rtk_eff)
            else:
                tau_l_qp, tau_r_qp, feasible_fl = fl.step(
                model, data, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
                q_ego_ref, qd_ego_ref, q_int_ref, qd_int_ref, pairs,
                M_ego_extra=M_box, bias_ego_extra=bias_box,
                table_rows=fl_table_rows,
                left_ee_body=left_ee_body, tilt_max_rad=CUP_TILT_MAX_RAD,
                leash_ee_body=left_ee_body, leash_max_m=CUP_LEASH_M,
                cup_cone_facets=(cf_cone if cf_cone > 0 else
                                 (CUP_CONE_FACETS if REAL_CUP_CONE_FACETS is None else REAL_CUP_CONE_FACETS)),
                ego_hold=True)
            _fl_stepped_this_k = True
            if REAL_FRICTION_FF:
                # joint dry-friction feedforward (frictionloss is a constraint force,
                # not in FL's c) along FL-MPC's commanded motion -- see
                # control/realgrasp_flmpc.friction_feedforward. Final clip below keeps TAU_MAX.
                tau_l_qp = tau_l_qp + rg.friction_feedforward(model, left_dofadr, data.qvel[left_dofadr],
                                                              fl.a_ego_prev, sc.DT)
                tau_r_qp = tau_r_qp + rg.friction_feedforward(model, right_dofadr, data.qvel[right_dofadr],
                                                              fl.a_int_prev, sc.DT)
            feasible_l = feasible_r = feasible_fl
            solve_time = time.perf_counter() - t_solve0
        else:
            # nominal torque + torque box of the reactive QP. Default (post-filter): the
            # raw task torque and the static +-TAU_MAX box (original). V4C_FILTER_MODE=
            # "pre" (v4c only): the SAFETY QP IS LAST -- the nominal is low-passed toward
            # the previously APPLIED torque and the box also carries the actuator slew
            # limit around it, so the CBF rows constrain the torque the plant actually gets.
            if _v4c_pre and _v4c_y_l is not None:
                _nom_l = _flt_a * tau_task_l + (1.0 - _flt_a) * _v4c_y_l
                _nom_r = _flt_a * tau_task_r + (1.0 - _flt_a) * _v4c_y_r
                _lo_l = np.maximum(lower, _v4c_y_l - _flt_r * sc.DT); _hi_l = np.minimum(upper, _v4c_y_l + _flt_r * sc.DT)
                _lo_r = np.maximum(lower, _v4c_y_r - _flt_r * sc.DT); _hi_r = np.minimum(upper, _v4c_y_r + _flt_r * sc.DT)
            else:
                _nom_l, _nom_r, _lo_l, _hi_l, _lo_r, _hi_r = tau_task_l, tau_task_r, lower, upper, lower, upper
            a_rows, b_rows = [], []
            for p in pairs:
                n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
                JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
                v_ego_p, v_int_p = p["v_ego"], p["v_int"]
                h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
                a_int_p = Ji_p @ qacc_right_known + JiDot_p @ data.qvel[right_dofadr]
                ndot_p = (np.eye(3) - np.outer(n_p, n_p)) @ (v_ego_p - v_int_p) / max(dist_p, 1e-6)
                hddot_rest_p = float(ndot_p @ (v_ego_p - v_int_p)
                                      + n_p @ (JeDot_p @ data.qvel[left_dofadr])
                                      - n_p @ a_int_p)
                psi1_p = hdot_p + sc.ALPHA1 * h_p
                A_h_p = n_p @ Je_p @ M_ll_inv
                const_term_p = (-float(n_p @ Je_p @ M_ll_inv @ bias_l) + hddot_rest_p
                                + sc.ALPHA1 * hdot_p + sc.ALPHA2 * psi1_p)
                a_rows.append(-A_h_p)
                b_rows.append(const_term_p)
            a_ineq = np.stack(a_rows, axis=0)
            b_ineq = np.array(b_rows)
            # TABLE-FLOOR BARRIER FIX: extra rows keeping the LEFT arm's own
            # collision-proxy keypoints above the tabletop -- appended onto
            # the arm-arm rows just built above; the arm-arm row math itself
            # (a_rows/b_rows) is untouched.
            if table_floor:
                a_table_l, b_table_l = sc.table_plane_barrier_rows(
                    model, data, left_table_kps, left_dofadr, M_ll_inv, bias_l,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw, **_v4c_tab_kw)
                a_ineq = np.vstack([a_ineq, a_table_l])
                b_ineq = np.concatenate([b_ineq, b_table_l])
            if _v4c_metric == "accel":
                # V4C_QP_METRIC="accel": min ||M^-1 (tau - tau_task)|| (joint-ACCELERATION deviation)
                # instead of the torque-space ||tau - tau_task||, which deflects mostly through
                # the low-inertia wrist joints (cheap torque, huge acceleration -> cup whipped).
                _Wl = M_ll_inv / float(np.max(np.abs(np.diag(M_ll_inv))))
                H_qp, f_qp = sc.build_weighted_least_squares_qp([(_Wl, _Wl @ _nom_l, 1.0)])
            else:
                H_qp, f_qp = sc.build_weighted_least_squares_qp([(np.eye(6), _nom_l, 1.0)])
            tau_l_qp, _, feasible_l = sc.solve_constrained_box_qp(H_qp, f_qp, _lo_l, _hi_l, a_ineq, b_ineq)

            ai_rows, bi_rows = [], []
            for p in pairs:
                n_p, Je_p, Ji_p = p["n"], p["Je"], p["Ji"]
                JeDot_p, JiDot_p = p["JeDot"], p["JiDot"]
                v_ego_p, v_int_p = p["v_ego"], p["v_int"]
                h_p, hdot_p, dist_p = p["h"], p["hdot"], p["dist"]
                h_p_int = h_p - COOP_INTRUDER_MARGIN   # SYMMETRIC yield (demo-local; see COOP_INTRUDER_MARGIN)
                a_ego_p = Je_p @ qacc_left_known + JeDot_p @ data.qvel[left_dofadr]
                ndot_p = (np.eye(3) - np.outer(n_p, n_p)) @ (v_ego_p - v_int_p) / max(dist_p, 1e-6)
                hddot_rest_int_p = float(ndot_p @ (v_ego_p - v_int_p)
                                          + n_p @ a_ego_p
                                          - n_p @ (JiDot_p @ qd_r))
                psi1_p_int = hdot_p + alpha1_int * h_p_int
                A_h_int_p = n_p @ Ji_p @ M_rr_inv
                const_term_int_p = (float(n_p @ Ji_p @ M_rr_inv @ bias_r) + hddot_rest_int_p
                                     + alpha1_int * hdot_p + alpha2_int * psi1_p_int)
                ai_rows.append(A_h_int_p)
                bi_rows.append(const_term_int_p)
            ai_ineq = np.stack(ai_rows, axis=0)
            bi_ineq = np.array(bi_rows)
            # TABLE-FLOOR BARRIER FIX: same as the LEFT arm above, for the
            # RIGHT arm's own collision-proxy keypoints (the arm actually
            # doing the cube pick/carry/deposit, so the one that matters
            # most here -- see the task's measurement).
            if table_floor:
                a_table_r, b_table_r = sc.table_plane_barrier_rows(
                    model, data, _rtk_eff, right_dofadr, M_rr_inv, bias_r,
                    table_top_z, margin=TABLE_FLOOR_MARGIN_M, **_tab_kw, **_v4c_tab_kw)
                ai_ineq = np.vstack([ai_ineq, a_table_r])
                bi_ineq = np.concatenate([bi_ineq, b_table_r])
            # EE-SPEED-LIMIT FIX part (2), reactive branch: hard-cap the RIGHT
            # (intruder) EE's next-step Cartesian speed at V_EE_MAX_MS -- the arm
            # doing the fast release->park return. SAME 6 rows the MPC uses, on
            # the applied torque; keeps the guarantee even during a safety dodge.
            if ee_speed_limit:
                Jp_r_full, _ = sc.body_jac_full(model, data, right_ee_body)
                Jp_r_ee = Jp_r_full[:, right_dofadr]
                a_vcap, b_vcap = ee_speed_limit_mod.ee_velocity_cap_rows(
                    Jp_r_ee, qd_r, M_rr_inv, bias_r, V_EE_MAX_MS, sc.DT)
                ai_ineq = np.vstack([ai_ineq, a_vcap])
                bi_ineq = np.concatenate([bi_ineq, b_vcap])
                if _v4c and V4C_SPEEDCAP_DIR:
                    # V4C_SPEEDCAP_DIR (ctrlfix F1 analog): the per-axis box above lets the
                    # Cartesian NORM reach sqrt(3)*V; add FL-MPC's row along the current EE
                    # motion direction u: u . v_next <= V.
                    _v_now = Jp_r_ee @ qd_r
                    _vn = float(np.linalg.norm(_v_now))
                    if _vn > 1e-3:
                        _u = _v_now / _vn
                        _G = sc.DT * (Jp_r_ee @ M_rr_inv)
                        _c = _v_now - sc.DT * (Jp_r_ee @ (M_rr_inv @ bias_r))
                        ai_ineq = np.vstack([ai_ineq, (_u @ _G)[None, :]])
                        bi_ineq = np.concatenate([bi_ineq, [V_EE_MAX_MS - float(_u @ _c)]])
            if _v4c_metric == "accel":
                _Wr = M_rr_inv / float(np.max(np.abs(np.diag(M_rr_inv))))
                Hi_qp, fi_qp = sc.build_weighted_least_squares_qp([(_Wr, _Wr @ _nom_r, 1.0)])
            else:
                Hi_qp, fi_qp = sc.build_weighted_least_squares_qp([(np.eye(6), _nom_r, 1.0)])
            tau_r_qp, _, feasible_r = sc.solve_constrained_box_qp(Hi_qp, fi_qp, _lo_r, _hi_r, ai_ineq, bi_ineq)
            solve_time = time.perf_counter() - t_solve0

        # QP-infeasibility recovery (safety net): if no safe torque exists within
        # the box, brake the arm (gravity/coriolis comp + velocity damping) rather
        # than apply the least-violating QP torque, which otherwise destabilizes
        # the arm and flings the held cup. With the slowed crossing this rarely fires.
        #
        # IMPORTANT (2026-09-09): the FL-MPC controller (`flmpc`) already runs its
        # OWN graceful QP-infeasibility fallback INTERNALLY -- mpc_fl.FLMPC.step's
        # `_relaxed_fallback` (slack-softened collision rows, torque box kept HARD)
        # + `_torque_safe_scale` closed-form backstop -- and RETURNS a
        # least-clearance-violating, torque-box-consistent tau on an infeasible
        # step (see mpc_fl.py). Overwriting that here with the crude
        # `bias - INFEAS_BRAKE_KD*qvel` brake would THROW AWAY that graceful
        # fallback (it was doing exactly that in every prior sweep -- the reason
        # the graceful fallback appeared not to move the collision rate). So for
        # `flmpc` we KEEP the controller's own returned torque. The reactive v4c
        # (`v7c`/default) branch has no internal graceful handler, so it still
        # uses this brake.
        _ctrl_has_graceful_fallback = (controller in ("flmpc", "flmpc_decentB"))
        if not feasible_l and not _ctrl_has_graceful_fallback:
            tau_l_qp = bias_l - _v4c_brake_kd * data.qvel[left_dofadr]
        if not feasible_r and not _ctrl_has_graceful_fallback:
            tau_r_qp = bias_r - _v4c_brake_kd * data.qvel[right_dofadr]

        left_deflect = float(np.linalg.norm(tau_l_qp - np.clip(tau_task_l, -sc.TAU_MAX, sc.TAU_MAX)))
        right_deflect = float(np.linalg.norm(tau_r_qp - np.clip(tau_task_r, -sc.TAU_MAX, sc.TAU_MAX)))

        if _v4c_pre:
            # V4C_FILTER_MODE="pre": the QP output IS the applied torque (already inside the
            # slew box); the infeasibility brake is clipped into the same box.
            if _v4c_y_l is not None:
                tau_l_qp = np.clip(tau_l_qp, _v4c_y_l - _flt_r * sc.DT, _v4c_y_l + _flt_r * sc.DT)
                tau_r_qp = np.clip(tau_r_qp, _v4c_y_r - _flt_r * sc.DT, _v4c_y_r + _flt_r * sc.DT)
            tau_l_cmd = np.clip(tau_l_qp, -sc.TAU_MAX, sc.TAU_MAX)
            tau_r_cmd = np.clip(tau_r_qp, -sc.TAU_MAX, sc.TAU_MAX)
            _v4c_y_l, _v4c_y_r = tau_l_cmd.copy(), tau_r_cmd.copy()
        elif cf_taumax is None:
            tau_l_cmd = np.clip(tau_filter_l.apply(tau_l_qp, sc.DT), -sc.TAU_MAX, sc.TAU_MAX)
            tau_r_cmd = np.clip(tau_filter_r.apply(tau_r_qp, sc.DT), -sc.TAU_MAX, sc.TAU_MAX)
        else:
            # F6: per-joint actuator limit on the final command too
            tau_l_cmd = np.clip(tau_filter_l.apply(tau_l_qp, sc.DT), -cf_taumax, cf_taumax)
            tau_r_cmd = np.clip(tau_filter_r.apply(tau_r_qp, sc.DT), -cf_taumax, cf_taumax)
        if cf_delay > 0 and fl is not None:
            # F5: the plant will see the FILTERED/clipped torque, not FL-MPC's raw
            # tau = M a + c -> record the acceleration it actually realizes
            # (a = M^-1 (tau_cmd - c), c = bias - passive (+ cup payload on the ego)),
            # so the in-flight prediction is exact w.r.t. what was sent.
            _c_l = data.qfrc_bias[left_dofadr] - data.qfrc_passive[left_dofadr] + bias_box
            _c_r = data.qfrc_bias[right_dofadr] - data.qfrc_passive[right_dofadr]
            _M_l = M_full[np.ix_(left_dofadr, left_dofadr)] + M_box
            _M_r = M_full[np.ix_(right_dofadr, right_dofadr)]
            # joint dry friction (frictionloss constraint) opposes the motion: same
            # Coulomb model the friction feedforward uses, along the planned motion
            _fr_l = rg.friction_feedforward(model, left_dofadr, data.qvel[left_dofadr], fl.a_ego_prev, sc.DT)
            _fr_r = rg.friction_feedforward(model, right_dofadr, data.qvel[right_dofadr], fl.a_int_prev, sc.DT)
            _a_l = np.linalg.solve(_M_l, tau_l_cmd - _c_l - _fr_l)
            _a_r = np.linalg.solve(_M_r, tau_r_cmd - _c_r - _fr_r)
            if _fl_stepped_this_k:
                fl.replace_last_applied(_a_l, _a_r)
            else:
                fl.push_applied(_a_l, _a_r)
        data.qfrc_applied[left_dofadr] = tau_l_cmd
        data.qfrc_applied[right_dofadr] = tau_r_cmd

        # -- ground-truth signals (real, every step) --
        h_min, dist_min, *_ = sc.min_pair_distance(model, data)
        right_cup_clearance = right_arm_to_cup_clearance(model, data, cup_kps_for_log,
                                                          right_kps_for_log, left_ee_body)
        # TABLE-FLOOR BARRIER FIX: min world-z over each arm's own real
        # collision-proxy keypoints, logged UNCONDITIONALLY (independent of
        # `table_floor`) -- this IS the measurement/verification metric.
        left_min_arm_z = min(_kp_world_z(data, bid, off) for bid, off, _r in left_table_kps)
        right_min_arm_z = min(_kp_world_z(data, bid, off) for bid, off, _r in right_table_kps)
        spilled = cct.count_spilled(model, data)
        deposited = cct.count_deposited_cubes(model, data)
        # MULTI-CUBE metric: how many of the selected cubes are RESTING near the
        # place-spot cluster (see cup_collect_task_utils.count_placed_cubes) --
        # 0 for "spill" (placed_center_xy is None, no cubes in that scenario).
        placed = (cct.count_placed_cubes(model, data, placed_center_xy, table_top_z,
                                          radius=KEEP_ZONE_RADIUS_M)
                  if placed_center_xy is not None else 0)
        collision = False
        for i in range(data.ncon):
            con = data.contact[i]
            g = {con.geom1, con.geom2}
            if (con.geom1 in left_geoms and con.geom2 in right_geoms) or \
               (con.geom2 in left_geoms and con.geom1 in right_geoms):
                collision = True
                break

        log["t"][k] = t
        log["h_min"][k] = h_min
        log["dist_min"][k] = dist_min
        log["right_cup_clearance_m"][k] = right_cup_clearance
        log["left_min_arm_z"][k] = left_min_arm_z
        log["right_min_arm_z"][k] = right_min_arm_z
        log["collision"][k] = float(collision)
        log["spilled"][k] = spilled
        log["deposited"][k] = deposited
        log["placed"][k] = placed
        log["placed_strict"][k] = (count_placed_strict(model, data, cube_plan, cube_init_pos, placed_center_xy,
                                                        table_top_z) if placed_center_xy is not None else 0)
        log["placed_from_outside"][k] = (count_placed_strict(model, data, cube_plan, cube_init_pos, placed_center_xy,
                                                              table_top_z, require_start_outside=True)
                                         if placed_center_xy is not None else 0)
        log["qp_feasible_left"][k] = float(feasible_l)
        log["qp_feasible_right"][k] = float(feasible_r)
        log["left_qp_deflect_nm"][k] = left_deflect
        log["right_qp_deflect_nm"][k] = right_deflect
        log["solve_time"][k] = solve_time
        log["cup_z"][k] = data.xpos[cup_body][2]
        log["cube_pinned"][k] = float(pin_now)
        log["spill_risk"][k] = held_risk
        log["spill_veto"][k] = float(held_veto)
        log["right_track_t"][k] = t_r
        log["left_tilt_t"][k] = left_tilt_elapsed
        log["left_q"][k] = q_l
        log["right_q"][k] = q_r
        log["left_ee_pos"][k] = data.xpos[left_ee_body]
        log["right_ee_pos"][k] = data.xpos[right_ee_body]
        log["cup_pos"][k] = data.xpos[cup_body]
        log["cup_n_axis"][k] = data.xmat[cup_body].reshape(3, 3) @ np.array([0.0, 0.0, 1.0])
        _cup_axis_world_k = data.xmat[left_ee_body].reshape(3, 3) @ _cup_axis_local_log
        log["left_cup_tilt_rad"][k] = float(np.arccos(np.clip(_cup_axis_world_k[2], -1.0, 1.0)))
        # --grasp-mode metrics (read-only; both modes)
        _rel_now, _ = rg.cup_rel_pose_in_ee(data, left_ee_body, cup_body)
        log["true_cup_tilt_rad"][k] = rg.cup_true_tilt_rad(data, cup_body)
        log["grip_slip_m"][k] = float(np.linalg.norm(_rel_now - info["cup_rel_pos0"]))
        log["pad_contacts"][k] = rg.pad_cup_contacts(model, data, left_pad_ids, cup_body)
        log["pads_touching"][k] = rg.pads_touching_cup(model, data, left_pad_ids, cup_body)
        log["finger_force_max_n"][k] = float(np.max(np.abs(rg.finger_actuator_forces(model, data))))
        log["pad_normal_force_n"][k] = rg.pad_normal_force_sum(model, data, left_pad_ids, cup_body)

        if capture and k % capture_stride == 0:
            renderer_main.update_scene(data, camera=cam_main)
            rgb = renderer_main.render().copy()
            renderer_zed.update_scene(data, camera=zed_cam_id)
            zed_rgb = renderer_zed.render().copy()
            frames.append((k, rgb, zed_rgb))

        mujoco.mj_step(model, data)

    if renderer_main is not None:
        renderer_main.close()
        renderer_zed.close()

    if cf_aw:
        # truncate the worst-case buffers to the steps actually run
        n_done = min(n_steps, n_steps_run + aw_frozen)
        for _k, _v in list(log.items()):
            if isinstance(_v, np.ndarray) and _v.ndim >= 1 and _v.shape[0] == n_steps:
                log[_k] = _v[:n_done]
    # --ctrlfixes bookkeeping (scalars; additive keys)
    log["aw_frozen_steps"] = np.array(float(aw_frozen))
    log["aw_clamped_steps"] = np.array(float(aw_clamped))
    log["fl_relaxed_steps"] = np.array(float(fl.n_relaxed if fl is not None else 0))
    log["fl_zero_fallback_steps"] = np.array(float(fl.n_zero_fallback if fl is not None else 0))
    log["fl_taumax_relaxed_steps"] = np.array(float(fl.n_taumax_relaxed if fl is not None else 0))
    if _rt_impl:
        # --controller-impl rt/rt_pipe counters (cap hits, iterations, pipeline misses/ages; never silent)
        for _k, _v in fl.rt_log_fields().items():
            log[_k] = np.asarray(_v, dtype=float)
    return log, frames


# ---------------------------------------------------------------------------
# HUD + video encoding (pattern reused from scripts/make_safety_demo.py).
# ---------------------------------------------------------------------------
def build_hud_and_encode(log: dict, frames: list, out_path: Path, config: str = "geometric_only",
                         controller: str = "v4c", scenario: str = "crossing", monitor: str = "classifier"):
    from PIL import Image, ImageDraw, ImageFont

    W, H = 1280, 720
    INSET = 220

    def _font(size):
        try:
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
        except Exception:
            return ImageFont.load_default()

    F_TITLE = _font(28)
    F_LABEL = _font(20)
    F_BANNER = _font(26)
    F_OUTCOME = _font(30)
    F_CAPTION = _font(15)

    GREEN = (60, 200, 90)
    YELLOW = (230, 200, 40)
    RED = (220, 55, 55)
    WHITE = (255, 255, 255)
    BLACK = (20, 20, 20)

    def meter_color(frac):
        if frac > 0.6:
            return GREEN
        if frac > 0.4:
            return YELLOW
        return RED

    ever_collision = False
    ever_spilled = False
    ever_deposited = False

    out_frames = []
    for k, rgb, zed_rgb in frames:
        t = float(log["t"][k])
        h_val = float(log["h_min"][k])
        ever_collision = ever_collision or bool(log["collision"][k] > 0.5)
        ever_spilled = ever_spilled or bool(log["spilled"][k] > 0.5)
        ever_deposited = ever_deposited or bool(log["deposited"][k] > 0.5)
        left_deflecting = bool(log["left_qp_deflect_nm"][k] > QP_DEFLECT_NM)
        engaged = (h_val < ENGAGE_H_M) and left_deflecting

        img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(img, "RGBA")

        if controller == "v7c":
            title = "CUP-COLLECT SAFETY DEMO -- V7-C cooperative MPC-HOCBF (both arms)"
        elif controller == "flmpc":
            title = "CUP-COLLECT SAFETY DEMO -- feedback-linearized cooperative MPC (both arms)"
        else:
            title = "CUP-COLLECT SAFETY DEMO -- v4-C* cooperative HOCBF-QP (both arms)"
        if scenario == "spill":
            title += " -- SPILL scenario (LEFT-arm fast tilt/transfer)"
        if config == "geometric_spill":
            title += " + latent HJ value shield V(z)" if monitor == "value" else " + REAL WM spill monitor"
        draw.rectangle([0, 0, W, 46], fill=(0, 0, 0, 150))
        draw.text((14, 8), title, font=F_TITLE, fill=WHITE)

        # arm-arm clearance meter (bottom-left)
        bar_x0, bar_y0, bar_w, bar_h = 14, H - 100, 360, 26
        draw.rectangle([bar_x0 - 6, bar_y0 - 30, bar_x0 + bar_w + 6, bar_y0 + bar_h + 60],
                       fill=(0, 0, 0, 150))
        draw.text((bar_x0, bar_y0 - 26), "arm-arm clearance h (measured, QP constraint value)", font=F_LABEL, fill=WHITE)
        draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h], outline=WHITE, width=2)
        frac = float(np.clip((h_val + 0.05) / 0.20, 0.0, 1.0))
        color = meter_color(frac)
        if frac > 0.0:
            draw.rectangle([bar_x0 + 2, bar_y0 + 2, bar_x0 + 2 + frac * (bar_w - 4), bar_y0 + bar_h - 2],
                           fill=color)
        draw.text((bar_x0 + bar_w + 12, bar_y0 + 2), f"h={h_val:+.3f} m", font=F_LABEL, fill=WHITE)

        spill_line = f"spilled balls: {int(log['spilled'][k])}   cubes placed: {int(log['placed'][k])}"
        draw.text((bar_x0, bar_y0 + 34), spill_line, font=F_LABEL, fill=WHITE)

        spill_veto = bool(log["spill_veto"][k] > 0.5) if config == "geometric_spill" else False
        if config == "geometric_spill":
            risk_val = float(log["spill_risk"][k])
            clock_label = "left-arm tilt clock" if scenario == "spill" else "right-arm deposit clock"
            clock_val = float(log["left_tilt_t"][k]) if scenario == "spill" else float(log["right_track_t"][k])
            risk_label = f"latent HJ value risk (mapped from V)={risk_val:.2f}" if monitor == "value" \
                else f"REAL WM spill risk P(spill)={risk_val:.2f}"
            spill_line2 = f"{risk_label}   {clock_label}={clock_val:.2f}s"
            draw.text((bar_x0, bar_y0 + 56), spill_line2, font=F_LABEL, fill=WHITE)

        if spill_veto:
            held_what = "tilt" if scenario == "spill" else "deposit"
            monitor_label = "LATENT HJ VALUE SHIELD" if monitor == "value" else "REAL WM monitor"
            banner = f"⚠ SPILL RISK — {monitor_label} HOLDING {held_what}"
            tw = draw.textlength(banner, font=F_BANNER)
            bx0, by0 = W / 2 - tw / 2 - 18, 58
            draw.rectangle([bx0, by0, bx0 + tw + 36, by0 + 44], fill=(200, 130, 20, 210))
            draw.text((bx0 + 18, by0 + 7), banner, font=F_BANNER, fill=WHITE)
        elif engaged and scenario == "crossing":
            banner = "⚠ COOPERATIVE AVOID — cup arm yielding to stay safe"
            tw = draw.textlength(banner, font=F_BANNER)
            bx0, by0 = W / 2 - tw / 2 - 18, 58
            draw.rectangle([bx0, by0, bx0 + tw + 36, by0 + 44], fill=(200, 30, 30, 210))
            draw.text((bx0 + 18, by0 + 7), banner, font=F_BANNER, fill=WHITE)

        outcome_text, outcome_color = None, BLACK
        if ever_collision:
            outcome_text, outcome_color = "COLLISION", RED
        elif ever_spilled:
            outcome_text, outcome_color = "SPILLED", RED
        elif ever_deposited:
            outcome_text, outcome_color = "CUBE DEPOSITED", GREEN
        else:
            outcome_text, outcome_color = "SAFE", GREEN
        tw = draw.textlength(outcome_text, font=F_OUTCOME)
        ox0, oy0 = W - tw - 40, 56
        draw.rectangle([ox0 - 12, oy0 - 6, ox0 + tw + 12, oy0 + 40], fill=outcome_color + (215,))
        draw.text((ox0, oy0), outcome_text, font=F_OUTCOME, fill=WHITE)

        ins = Image.fromarray(zed_rgb).resize((INSET, INSET), Image.NEAREST)
        ix0, iy0 = W - INSET - 14, H - INSET - 14
        draw.rectangle([ix0 - 4, iy0 - 22, ix0 + INSET + 4, iy0 + INSET + 4], fill=(0, 0, 0, 150))
        draw.text((ix0, iy0 - 20), "ZED workspace camera (real sensor view)", font=ImageFont.load_default(), fill=WHITE)
        img.paste(ins, (ix0, iy0))

        if scenario == "spill":
            if config == "geometric_spill" and monitor == "value":
                caption = ("MEASURED: h, spilled, deposited, QP deflection, shield risk/veto, LEFT-arm "
                           "tilt-clock freeze.  REAL: the risk + veto come from V_theta(z), the trained "
                           "latent HJ safety value (world_model/wm_semantic_monitor.LatentValueShield, "
                           "world_model/train_latent_safety_value.py), veto'ing anticipatorily at "
                           "V(z) < v_thresh, consulted at a vision cadence -- not a scripted signal; the "
                           "geometric HOCBF-QP CANNOT see this failure (no arm-arm proximity involved) -- "
                           "only the shield prevents the spill.  RIGHT arm holds home.")
            elif config == "geometric_spill":
                caption = ("MEASURED: h, spilled, deposited, QP deflection, WM spill risk/veto, LEFT-arm "
                           "tilt-clock freeze.  REAL: the P(spill) risk + veto come from the frozen trained "
                           "WorldModel + monitor head (world_model/wm_semantic_monitor.RealWMSpillMonitor, "
                           "retrained ON this scene: wm_cup_collect), consulted at a vision cadence -- not a "
                           "scripted signal; the geometric HOCBF-QP CANNOT see this failure (no arm-arm "
                           "proximity involved) -- only the WM prevents the spill.  RIGHT arm holds home.")
            else:
                caption = ("MEASURED: h, spilled, deposited, QP deflection.  The LEFT arm's fast wrist tilt/"
                           "transfer is a REAL torque-tracked reference (freeze-clock gated, but no monitor "
                           "here -- geometric HOCBF-QP only, which cannot see a spill: no arm-arm proximity "
                           "involved).  RIGHT arm holds home.")
        elif config == "geometric_spill" and monitor == "value":
            caption = ("MEASURED: h, spilled, deposited, QP deflection, shield risk/veto, deposit-clock "
                       "freeze.  REAL: the risk + veto come from V_theta(z), the trained latent HJ safety "
                       "value (world_model/wm_semantic_monitor.LatentValueShield, world_model/"
                       "train_latent_safety_value.py), veto'ing anticipatorily at V(z) < v_thresh, "
                       "consulted at a vision cadence -- not a scripted signal.  SCRIPTED (allowed): "
                       "right-arm pick/carry/deposit path is an IK reference (its clock freezes under "
                       "veto); cube WELD-attached (real constraint force, magnet-style pickup) in transit, "
                       "released above the cup for a real physics drop.")
        elif config == "geometric_spill":
            caption = ("MEASURED: h, spilled, deposited, QP deflection, WM spill risk/veto, deposit-clock "
                       "freeze.  REAL: the P(spill) risk + veto come from the frozen trained WorldModel + "
                       "monitor head (world_model/wm_semantic_monitor.RealWMSpillMonitor), consulted at a "
                       "vision cadence -- not a scripted signal.  SCRIPTED (allowed): right-arm pick/carry/"
                       "deposit path is an IK reference (its clock freezes under veto); cube WELD-attached "
                       "(real constraint force, magnet-style pickup) in transit, released above the cup for "
                       "a real physics drop.")
        else:
            caption = ("MEASURED: h, spilled, deposited, QP deflection.  SCRIPTED (allowed): right-arm "
                       "pick/carry/deposit path is an IK reference; cube WELD-attached (real constraint "
                       "force, magnet-style pickup) in transit, released above the cup for a real physics drop.")
        draw.rectangle([0, H - 22, W, H], fill=(0, 0, 0, 150))
        draw.text((10, H - 20), caption, font=F_CAPTION, fill=WHITE)

        out_frames.append(np.asarray(img.convert("RGB")))

    import imageio.v2 as imageio
    fps = 1.0 / (sc.DT * 5)
    writer = imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8, macro_block_size=None)
    for f in out_frames:
        writer.append_data(f)
    writer.close()
    return dict(n_frames=len(out_frames), fps=fps, ever_collision=ever_collision,
                ever_spilled=ever_spilled, ever_deposited=ever_deposited)


def _longest_true_run(mask: np.ndarray) -> int:
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def summarize(log: dict) -> dict:
    out = _summarize_core(log)
    if "true_cup_tilt_rad" in log:
        real = bool(float(np.asarray(log.get("grasp_mode_real", 0.0))) > 0.5)
        n = len(log["t"])
        idx_cross = int(np.searchsorted(log["t"], RIGHT_START_T))
        z0 = float(log["cup_z"][0])
        cup_low = bool(idx_cross < n and np.any(log["cup_z"][idx_cross:] < z0 + REAL_CUP_DROP_RISE_MIN_M))
        no_contact_run = _longest_true_run(log["pad_contacts"] < 0.5)
        cf_on = bool(float(np.asarray(log.get("ctrlfixes", 0.0))) > 0.5)
        if cf_on:
            # F7 (ctrlfixes): a DROP is gripper-relative -- the cup's offset in the
            # EE frame (grip_slip_m) beyond CTRLFIX_DROP_SLIP_M for >=
            # REAL_CUP_DROP_NO_CONTACT_STEPS consecutive steps, or (real mode) a
            # SUSTAINED no-contact run. The legacy `cup_low` (cup_z < z0+3 cm after
            # the crossing starts) false-alarms when the cooperative yield merely
            # lowers the held cup; it is kept as cup_low_legacy for reference.
            slip_run = _longest_true_run(np.asarray(log["grip_slip_m"]) > CTRLFIX_DROP_SLIP_M)
            cup_dropped_v = bool(slip_run >= REAL_CUP_DROP_NO_CONTACT_STEPS
                                 or (real and no_contact_run >= REAL_CUP_DROP_NO_CONTACT_STEPS))
        else:
            cup_dropped_v = bool(cup_low or (real and no_contact_run >= REAL_CUP_DROP_NO_CONTACT_STEPS))
        pc = np.asarray(log["pad_contacts"])
        w = CTRLFIX_PAD_FLICKER_STEPS
        # robust min: min over k of max(pc[k:k+w]) -> ignores dropouts shorter than w steps
        pc_robust = (int(np.min(np.lib.stride_tricks.sliding_window_view(pc, w).max(axis=1)))
                     if pc.shape[0] >= w else int(np.min(pc)))
        out.update(
            grasp_mode="real" if real else "faked",
            # pad-contact loss only counts in real mode (faked pads never touch the cup by design)
            cup_dropped=cup_dropped_v,
            cup_low_legacy=bool(cup_low),
            true_cup_tilt_max_deg=float(np.degrees(np.max(log["true_cup_tilt_rad"]))),
            true_cup_tilt_max_deg_after_lift=(float(np.degrees(np.max(log["true_cup_tilt_rad"][idx_cross:])))
                                              if idx_cross < n else float("nan")),
            grip_slip_max_m=float(np.max(log["grip_slip_m"])),
            pad_contacts_min=(pc_robust if cf_on else int(np.min(log["pad_contacts"]))),
            pad_contacts_min_raw=int(np.min(log["pad_contacts"])),
            pad_contacts_min_robust=pc_robust,
            pads_touching_min=int(np.min(log["pads_touching"])),
            pad_no_contact_longest_steps=int(no_contact_run),
            finger_force_max_n=float(np.max(log["finger_force_max_n"])),
            pad_normal_force_mean_n=float(np.mean(log["pad_normal_force_n"])),
            left_ee_tilt_max_deg=float(np.degrees(np.max(log["left_cup_tilt_rad"]))),
        )
    if "aw_frozen_steps" in log:
        idx_x = int(np.searchsorted(log["t"], RIGHT_START_T))
        rp = np.asarray(log["right_ee_pos"])
        sp = np.linalg.norm(np.diff(rp, axis=0), axis=1) / sc.DT if rp.shape[0] > 1 else np.zeros(1)
        out.update(
            aw_frozen_steps=int(float(np.asarray(log["aw_frozen_steps"]))),
            aw_clamped_steps=int(float(np.asarray(log.get("aw_clamped_steps", 0.0)))),
            fl_relaxed_steps=int(float(np.asarray(log["fl_relaxed_steps"]))),
            fl_zero_fallback_steps=int(float(np.asarray(log["fl_zero_fallback_steps"]))),
            fl_taumax_relaxed_steps=int(float(np.asarray(log["fl_taumax_relaxed_steps"]))),
            right_ee_speed_max_cross_ms=(float(np.max(sp[idx_x:])) if idx_x < sp.shape[0] else float("nan")),
        )
    if "placed_strict" in log:
        wd = np.asarray(log["weld_onset_site_cube_dist_m"])
        wt = np.asarray(log["weld_onset_site_target_dist_m"])
        out.update(
            bugfixes=bool(float(np.asarray(log["bugfixes"])) > 0.5),
            ctrlfixes=bool(float(np.asarray(log.get("ctrlfixes", 0.0))) > 0.5),
            placed_strict_final=int(log["placed_strict"][-1]),
            placed_strict_max=int(np.max(log["placed_strict"])),
            placed_from_outside_final=(int(log["placed_from_outside"][-1]) if "placed_from_outside" in log else -1),
            placed_from_outside_max=(int(np.max(log["placed_from_outside"])) if "placed_from_outside" in log else -1),
            cup_init_tilt_deg=float(np.asarray(log["cup_init_tilt_deg"])),
            n_welds=int(np.sum(np.isfinite(wd))),
            weld_missed=int(np.sum(~np.isfinite(wd) & (np.asarray(log["weld_deferred_steps"]) > 0))),
            weld_onset_site_cube_dist_m=[float(x) for x in wd],
            weld_onset_site_target_dist_m=[float(x) for x in wt],
            weld_onset_site_target_dist_max_m=(float(np.nanmax(wt)) if np.any(np.isfinite(wt)) else float("nan")),
        )
    return out


def _summarize_core(log: dict) -> dict:
    return dict(
        h_min=float(np.min(log["h_min"])),
        h_final=float(log["h_min"][-1]),
        dist_min=float(np.min(log["dist_min"])),
        collision=bool(np.any(log["collision"] > 0.5)),
        spilled_max=int(np.max(log["spilled"])),
        deposited_final=int(log["deposited"][-1]),
        deposited_max=int(np.max(log["deposited"])),
        placed_final=int(log["placed"][-1]),
        placed_max=int(np.max(log["placed"])),
        qp_infeasible_frac_left=float(np.mean(1.0 - log["qp_feasible_left"])),
        qp_infeasible_frac_right=float(np.mean(1.0 - log["qp_feasible_right"])),
        left_qp_deflect_max_nm=float(np.max(log["left_qp_deflect_nm"])),
        left_qp_deflect_mean_nm=float(np.mean(log["left_qp_deflect_nm"])),
        frac_steps_left_deflecting=float(np.mean(log["left_qp_deflect_nm"] > QP_DEFLECT_NM)),
        cup_z_min=float(np.min(log["cup_z"])),
        cup_z_final=float(log["cup_z"][-1]),
        spill_risk_max=float(np.max(log["spill_risk"])),
        spill_veto_frac=float(np.mean(log["spill_veto"])),
        right_track_t_final=float(log["right_track_t"][-1]),
        left_tilt_t_final=float(log["left_tilt_t"][-1]),
        solve_time_mean_ms=float(np.mean(log["solve_time"]) * 1000.0),
        solve_time_max_ms=float(np.max(log["solve_time"]) * 1000.0),
        right_cup_clearance_min_m=float(np.min(log["right_cup_clearance_m"])),
        left_min_arm_z_m=float(np.min(log["left_min_arm_z"])),
        right_min_arm_z_m=float(np.min(log["right_min_arm_z"])),
        # EE-SPEED-LIMIT verification metric: peak right-EE Cartesian speed
        # (from consecutive logged EE positions). Should sit at/below V_EE_MAX_MS
        # when the fix is on; ~0.79 (park) when off.
        right_ee_speed_max_ms=float(np.max(np.linalg.norm(np.diff(log["right_ee_pos"], axis=0), axis=1) / sc.DT))
                             if log["right_ee_pos"].shape[0] > 1 else 0.0,
    )


def _make_spill_monitor(config: str, monitor: str = "classifier"):
    """Build the REAL semantic-safety shield once (config="geometric_spill"
    only) -- checkpoints loaded ONCE here, never per-step/per-episode;
    run_episode() calls .assess() at a vision cadence and the caller calls
    .reset() before each fresh episode. Returns None for
    config="geometric_only" regardless of `monitor`.

    monitor="classifier" (default, UNCHANGED): RealWMSpillMonitor's
    instantaneous P(spill) classifier veto -- byte-identical to before this
    task's --monitor flag existed.
    monitor="value" (NEW, this task): LatentValueShield -- the trained
    latent HJ value V_theta(z), veto'ing at V(z) < VALUE_V_THRESH (small
    positive, anticipatory margin) instead of P(spill) >= risk_threshold.
    Everything downstream (vision cadence, freeze-clock veto in
    run_episode) is unchanged -- only this veto SOURCE differs.
    """
    if config == "geometric_only":
        return None
    assert monitor in MONITOR_CHOICES, f"unknown monitor {monitor!r}"
    t_load = time.perf_counter()
    if monitor == "classifier":
        print(f"[setup] loading REAL WM spill monitor (config={config}, monitor=classifier): "
              f"wm={WM_CHECKPOINT}\n         head={HEAD_CHECKPOINT}")
        shield = RealWMSpillMonitor(wm_checkpoint=WM_CHECKPOINT, head_checkpoint=HEAD_CHECKPOINT,
                                     risk_threshold=SPILL_RISK_THRESHOLD)
    else:
        print(f"[setup] loading LATENT HJ VALUE shield (config={config}, monitor=value): "
              f"wm={WM_CHECKPOINT}\n         head={HEAD_CHECKPOINT}\n"
              f"         value={VALUE_CHECKPOINT}  (v_thresh={VALUE_V_THRESH})")
        shield = LatentValueShield(wm_checkpoint=WM_CHECKPOINT, head_checkpoint=HEAD_CHECKPOINT,
                                    value_checkpoint=VALUE_CHECKPOINT, v_thresh=VALUE_V_THRESH)
    print(f"  loaded in {time.perf_counter() - t_load:.1f}s (device={shield.device})")
    return shield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-video", action="store_true", help="skip rendering/encoding (metrics only)")
    parser.add_argument("--baseline-only", action="store_true", help="only run the no-QP baseline (fast smoke test)")
    parser.add_argument("--config", choices=CONFIG_CHOICES, default="geometric_spill",
                         help="geometric_only: original v4-C* HOCBF-QP only (measured failure, unchanged). "
                              "geometric_spill (default): + the REAL WM spill monitor freezing the right-arm "
                              "deposit clock (--scenario crossing) or the LEFT-arm tilt clock "
                              "(--scenario spill) while spill risk is high.")
    parser.add_argument("--scenario", choices=SCENARIO_CHOICES, default="crossing",
                         help="crossing (default, unchanged): RIGHT arm pick/carry/deposit crosses near the "
                              "held cup -- arm-arm collision is the geometric layer's job, WM freezes the "
                              "deposit clock while spill risk is high. spill: RIGHT arm just holds home; "
                              "LEFT arm (holding the cup) does a FAST wrist tilt/transfer that spills the cup "
                              "-- a failure the geometric HOCBF-QP is structurally blind to (no arm-arm "
                              "proximity involved); WM freezes the LEFT arm's OWN tilt clock instead.")
    parser.add_argument("--controller", choices=CONTROLLER_CHOICES, default="v4c",
                         help="v4c (default, unchanged): decentralized cooperative HOCBF-QP (v4-C*). "
                              "v7c: both arms driven by the cooperative MPC-HOCBF (mpc7.py's JointMPCv7, "
                              "reused unmodified) -- forces the WM spill monitor OFF regardless of --config "
                              "(geometric run only for now; see module docstring CHANGE 2). "
                              "flmpc: both arms driven by the feedback-linearized cooperative MPC "
                              "(experiments/ab_smooth_vs_bangbang/mpc_fl.py's FLMPC -- plans per-arm "
                              "accelerations and realizes tau=M*a+c so the horizon prediction is exact, "
                              "no per-step MuJoCo rollout) -- forces the WM spill monitor OFF, same as v7c.")
    parser.add_argument("--vision-cadence-steps", type=int, default=VISION_CADENCE_STEPS,
                         help="physics steps between WM/monitor consultations (veto held between updates)")
    parser.add_argument("--monitor", choices=MONITOR_CHOICES, default="classifier",
                         help="classifier (default, UNCHANGED): RealWMSpillMonitor's instantaneous P(spill) "
                              "veto. value (NEW): LatentValueShield -- the trained latent HJ value V_theta(z) "
                              "(world_model/train_latent_safety_value.py), veto'ing anticipatorily at "
                              "V(z) < VALUE_V_THRESH. Only meaningful when --config geometric_spill.")
    parser.add_argument("--max-steps", type=int, default=None,
                         help="SMOKE ONLY: truncate the episode to this many steps (full episode when omitted)")
    parser.add_argument("--cup-in-barrier", choices=("on", "off"), default="on",
                         help="on (default, the fix): the held cup's keypoints are added to the LEFT arm's "
                              "HOCBF-QP barrier set, so the RIGHT arm's return/park leg (and any other "
                              "motion) is filtered to stay clear of the cup. off: VERIFICATION ONLY -- "
                              "reproduces the pre-fix hole (cup absent from the barrier) to prove the fix "
                              "does something; never use for the real demo.")
    parser.add_argument("--table-floor", choices=("on", "off"), default="on",
                         help="on (default, the fix): each arm's own real collision-proxy keypoints get "
                              "extra HOCBF-QP rows keeping them above the table_top surface (v4c "
                              "decentralized box-QP branch only -- see sim_common.table_plane_barrier_rows), "
                              "so a link/gripper can no longer dip below the tabletop while descending to "
                              "grasp a cube. off: VERIFICATION ONLY -- reproduces the pre-fix hole (no table "
                              "awareness in either QP) to prove the fix does something; never use for the "
                              "real demo.")
    parser.add_argument("--ee-speed-limit", choices=("on", "off"), default="on",
                         help="on (default, the fix): the RIGHT (intruder) EE is held to V_EE_MAX_MS "
                              f"(~{V_EE_MAX_MS} m/s) two ways -- the scripted right-arm reference is time-scaled "
                              "so no phase exceeds the cap (part 1, build_episode) AND a hard next-step Cartesian "
                              "velocity constraint is added to that arm's QP under BOTH controllers (part 2, "
                              "ee_speed_limit.ee_velocity_cap_rows / mpc7 v_ee_max). off: VERIFICATION ONLY -- "
                              "reproduces the un-capped fast return-to-home (~0.79 m/s park) to prove the fix "
                              "does something.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cubes", type=str, default=None,
                         help="comma-separated cube body names to force the pick set, e.g. "
                              "'cube_3,cube_4,cube_5' (default: auto-select the N_CUBES farthest-from-cup "
                              "IK-reachable). Still IK-filtered; order kept.")
    parser.add_argument("--cup-jitter-m", type=float, default=0.0,
                         help="SMOKE ONLY: per-episode randomized cup placement jitter [m] "
                              "(scene_setup.init_cup_grasped_state's cup_jitter_m; default 0.0, unchanged).")
    parser.add_argument("--n-cubes", type=int, default=None,
                         help="SMOKE ONLY: override the number of crossing cubes (default: module constant "
                              "N_CUBES=6, unchanged).")
    parser.add_argument("--adaptive-yield", choices=("on", "off"), default="off",
                         help="--controller flmpc only: on (this task's ADAPTIVE COOPERATIVE YIELDING -- see "
                              "mpc_fl.FLMPC's __init__ docstring): widens the anticipatory clearance margin "
                              "as the worst pair closes fast (avoids the QP-infeasibility cliff instead of "
                              "softening its fallout) and adaptively modulates the cup(ego) arm's hold "
                              "weight/leash toward more-yielding bounds under urgency, reverting as cup tilt "
                              "rises. off (default): byte-identical to the original flmpc behavior.")
    parser.add_argument("--grasp-mode", choices=GRASP_MODE_CHOICES, default="faked",
                         help="faked (default, byte-identical to before): fake grasp close + reattach_cup_rigidly "
                              "+ ball carry. real (SIM ONLY): genuine two-pad friction pinch under grip-force "
                              "control, no cup teleport/weld/velocity overwrite, FL-MPC drives the lift, cup "
                              "barrier keypoints from the actual cup pose (requires --controller flmpc|"
                              "flmpc_decentB, --scenario crossing). Output files get a '_realgrasp' suffix.")
    parser.add_argument("--bugfixes", choices=("on", "off"), default="on",
                         help="on (default): CHANGE-2 harness bug fixes (re-seat a toppled faked cup, gripper "
                              "clock on t_r, FL-MPC joint-friction feedforward; the weld gate is separate: --weld-gate-m). off: "
                              "pre-fix behavior, byte-identical. New metrics are logged either way.")
    parser.add_argument("--weld-gate-m", type=float, default=None,
                         help="(4) OFF by default: attach a cube weld only once the right grasp site is within this "
                              "distance [m] of the cube's grasp target (else a missed pick). See the --bugfixes block "
                              "comment: with the current right-arm grasp reference, 0.02 misses every pick.")
    parser.add_argument("--ctrlfixes", choices=("on", "off"), default="on",
                         help="on (default): controller fixes F1-F6 (see the --ctrlfixes block comment: "
                              "gravity double-comp, FL-MPC EE speed cap + anti-windup, top-down table-aware "
                              "waypoints, keypoint radii in the table rows, delay compensation, per-joint "
                              "tau_max, true-tilt cone). off: pre-fix behavior, byte-identical.")
    parser.add_argument("--ctrlfix-parts", type=str, default="all",
                         help=f"comma subset of {','.join(CTRLFIX_ALL_PARTS)} active under --ctrlfixes on "
                              "(default all) -- for per-fix ablations.")
    parser.add_argument("--ctrlfix-delay-steps", type=int, default=None,
                         help="F5: known actuation delay [control steps] FL-MPC compensates (default: "
                              "CTRLFIX_DELAY_STEPS=0 -> no-op).")
    parser.add_argument("--controller-impl", choices=CONTROLLER_IMPL_CHOICES, default="py",
                        help="--controller flmpc implementation: py (default, mpc_fl.FLMPC, unchanged) | rt "
                             "(experiments/realtime/rt_flmpc.py: one compiled, iteration-capped Numba step on a "
                             "robot-only dynamics source) | rt_pipe (rt + pipelined anytime solver thread). "
                             "Non-default impls get an '_<impl>' output-file suffix.")
    parser.add_argument("--rt-pipe-inject", type=str, default=None,
                        help="rt_pipe only: artificial solver delay injection 'P:MS[,MS...]' (e.g. 0.05:10,20 = "
                             "5%% of solves delayed by 10 or 20 ms) to exercise the deadline-miss path.")
    args = parser.parse_args()
    global BUGFIXES, WELD_ATTACH_GATE_M, CTRLFIXES, CTRLFIX_PARTS, CTRLFIX_DELAY_STEPS
    global CONTROLLER_IMPL, RT_PIPE_OPTS
    CONTROLLER_IMPL = args.controller_impl
    if args.rt_pipe_inject:
        _p, _ms = args.rt_pipe_inject.split(":")
        RT_PIPE_OPTS = dict(inject_p=float(_p), inject_ms=tuple(float(v) for v in _ms.split(",")))
    WELD_ATTACH_GATE_M = args.weld_gate_m
    BUGFIXES = (args.bugfixes == "on")
    CTRLFIXES = (args.ctrlfixes == "on")
    CTRLFIX_PARTS = parse_ctrlfix_parts(args.ctrlfix_parts)
    if args.ctrlfix_delay_steps is not None:
        CTRLFIX_DELAY_STEPS = int(args.ctrlfix_delay_steps)
    if args.cubes:
        global _CUBE_OVERRIDE
        _CUBE_OVERRIDE = [c.strip() for c in args.cubes.split(",") if c.strip()]
        print(f"[main] cube-set override: {_CUBE_OVERRIDE}")

    t0 = time.perf_counter()
    print(f"[setup] building episode (seed={args.seed}, config={args.config}, "
          f"controller={args.controller}, scenario={args.scenario}, monitor={args.monitor}, "
          f"grasp_mode={args.grasp_mode})...")
    info = build_episode(seed=args.seed, scenario=args.scenario, ee_speed_limit=(args.ee_speed_limit == "on"),
                         cup_jitter_m=args.cup_jitter_m, n_cubes=args.n_cubes,
                         grasp_mode=args.grasp_mode)
    print(f"  left lift IK err={info['lift_err']:.4f} m; right-arm IK errs={info['ik_errs']}")
    print(f"  duration_s={info['duration_s_total']:.2f}  n_steps={info['n_steps_total']}  "
          f"RIGHT_START_T={RIGHT_START_T:.2f}"
          + (f"  LEFT_TILT_START_T={LEFT_TILT_START_T:.2f}" if args.scenario == "spill" else ""))
    if args.max_steps is not None:
        print(f"  [smoke] max_steps={args.max_steps} (episode truncated)")

    # v7c (CHANGE 2): keep the controller axis orthogonal to the geometric_only/
    # geometric_spill config axis by simply not building the WM monitor -- a
    # geometric (no-WM) run, per the task spec ("if not easy, just make v7c a
    # geometric run for now").
    if args.controller in ("v7c", "flmpc"):
        if args.config != "geometric_only":
            print(f"  [note] --controller {args.controller} forces the WM spill monitor OFF "
                  f"(--config {args.config} is ignored for spill-monitoring purposes)")
        spill_monitor = None
    else:
        spill_monitor = _make_spill_monitor(args.config, args.monitor)

    if args.baseline_only:
        print("[baseline] running no-QP baseline only...")
        if spill_monitor is not None:
            spill_monitor.reset()
        log_base, _ = run_episode(info, controller=args.controller, qp_enabled=False, capture=False, ee_speed_limit=(args.ee_speed_limit == "on"),
                                   spill_monitor=spill_monitor,
                                   vision_cadence_steps=args.vision_cadence_steps,
                                   max_steps=args.max_steps,
                                   cup_in_barrier=(args.cup_in_barrier == "on"),
                                   table_floor=(args.table_floor == "on"))
        m_base = summarize(log_base)
        print(json.dumps(m_base, indent=2))
        return

    cam_main = mujoco.MjvCamera()
    cam_main.lookat = np.array([0.40, -0.18, 0.70])
    cam_main.distance = 1.55
    cam_main.azimuth = 145
    cam_main.elevation = -16

    print(f"[main] running {args.controller}-filtered episode ({args.config}) with capture...")
    info_main = build_episode(seed=args.seed, scenario=args.scenario, ee_speed_limit=(args.ee_speed_limit == "on"),
                              cup_jitter_m=args.cup_jitter_m, n_cubes=args.n_cubes,
                         grasp_mode=args.grasp_mode)  # fresh model/data instance for the real run
    if spill_monitor is not None:
        spill_monitor.reset()
    log_main, frames = run_episode(info_main, controller=args.controller, qp_enabled=True, capture=not args.no_video, ee_speed_limit=(args.ee_speed_limit == "on"),
                                    cam_main=cam_main, spill_monitor=spill_monitor,
                                    vision_cadence_steps=args.vision_cadence_steps,
                                    max_steps=args.max_steps,
                                    cup_in_barrier=(args.cup_in_barrier == "on"),
                                    table_floor=(args.table_floor == "on"),
                                    adaptive_yield=(args.adaptive_yield == "on"))
    m_main = summarize(log_main)
    print("[main] metrics:")
    print(json.dumps(m_main, indent=2))

    if args.controller == "flmpc":
        # CUP-ARM YIELD-REFINEMENT smoke measurement (this task): the cup
        # EE's Cartesian displacement + tilt should be BOUNDED (leash/tilt
        # constraints), not glued (~0) nor unbounded. Two displacement
        # numbers: the literal max||left_ee_pos - left_ee_pos[0]|| over the
        # WHOLE episode (dominated by the ~11cm initial v4c-path lift ramp,
        # which happens BEFORE flmpc/its leash even engage -- see the
        # t_r>=0 gating in the flmpc branch above) and, more meaningfully,
        # the max displacement from the SETTLED hold pose measured only once
        # flmpc has actually taken over (t >= RIGHT_START_T) -- what the new
        # leash constraint (CUP_LEASH_M) actually bounds.
        ee_pos = log_main["left_ee_pos"]
        ee_disp_from_t0 = np.linalg.norm(ee_pos - ee_pos[0], axis=1)
        idx_settle = int(np.searchsorted(log_main["t"], RIGHT_START_T))
        idx_settle = min(idx_settle, len(ee_pos) - 1)
        ee_disp_from_hold = np.linalg.norm(ee_pos[idx_settle:] - ee_pos[idx_settle], axis=1)
        tilt = log_main["left_cup_tilt_rad"]
        print("[cup-arm yield refinement] smoke measurements:")
        print(json.dumps(dict(
            cup_ee_disp_max_from_t0_m=float(np.max(ee_disp_from_t0)),
            cup_ee_disp_max_from_hold_pose_m=float(np.max(ee_disp_from_hold)),
            cup_tilt_max_rad_whole_episode=float(np.max(tilt)),
            cup_tilt_max_rad_after_hold=float(np.max(tilt[idx_settle:])),
            leash_max_m=CUP_LEASH_M, tilt_max_rad_cfg=CUP_TILT_MAX_RAD,
        ), indent=2))

    print("[baseline] running no-QP baseline (same references, safety filter OFF) for the deflection proof...")
    info_base = build_episode(seed=args.seed, scenario=args.scenario, ee_speed_limit=(args.ee_speed_limit == "on"),
                              cup_jitter_m=args.cup_jitter_m, n_cubes=args.n_cubes,
                         grasp_mode=args.grasp_mode)  # fresh model/data instance, identical references
    if spill_monitor is not None:
        spill_monitor.reset()
    log_base, _ = run_episode(info_base, controller=args.controller, qp_enabled=False, capture=False, ee_speed_limit=(args.ee_speed_limit == "on"),
                               spill_monitor=spill_monitor,
                               vision_cadence_steps=args.vision_cadence_steps,
                               max_steps=args.max_steps,
                               cup_in_barrier=(args.cup_in_barrier == "on"),
                               table_floor=(args.table_floor == "on"))
    m_base = summarize(log_base)
    print("[baseline] metrics:")
    print(json.dumps(m_base, indent=2))

    # deflection proof: compare LEFT arm joint trajectories between the QP-filtered
    # run and the no-QP baseline (identical initial conditions + references).
    n = min(len(log_main["t"]), len(log_base["t"]))
    joint_dev = np.linalg.norm(log_main["left_q"][:n] - log_base["left_q"][:n], axis=1)
    ee_dev = np.linalg.norm(log_main["left_ee_pos"][:n] - log_base["left_ee_pos"][:n], axis=1)
    k_worst = int(np.argmax(ee_dev))
    deflection_proof = dict(
        max_left_joint_deviation_rad=float(np.max(joint_dev)),
        max_left_ee_deviation_m=float(np.max(ee_dev)),
        t_of_max_ee_deviation_s=float(log_main["t"][k_worst]),
        h_min_at_max_deviation=float(log_main["h_min"][k_worst]),
    )
    print("[deflection proof] QP-filtered vs no-QP baseline, LEFT (cup) arm:")
    print(json.dumps(deflection_proof, indent=2))

    meta = dict(config=args.config, controller=args.controller, scenario=args.scenario,
                monitor=args.monitor, cup_in_barrier=args.cup_in_barrier,
                grasp_mode=args.grasp_mode, grasp_info=info_main.get("grasp_info"),
                table_floor=args.table_floor,
                ee_speed_limit=args.ee_speed_limit, v_ee_max_ms=V_EE_MAX_MS,
                bugfixes=args.bugfixes, ctrlfixes=args.ctrlfixes,
                ctrlfix_parts=sorted(CTRLFIX_PARTS) if CTRLFIXES else [],
                ctrlfix_delay_steps=CTRLFIX_DELAY_STEPS, wp_modes=info_main.get("wp_modes"),
                main_metrics=m_main, baseline_metrics=m_base,
                deflection_proof=deflection_proof,
                duration_s=info_main["duration_s_total"], n_steps=info_main["n_steps_total"],
                cubes=[c["name"] for c in info_main["cube_plan"]],
                vision_cadence_steps=args.vision_cadence_steps,
                wall_time_setup_to_here_s=time.perf_counter() - t0)

    # config-suffixed output filenames so both configs' artifacts can coexist
    # for the parent's side-by-side comparison (e.g. "..._geometric_only.mp4"
    # vs "..._geometric_spill.mp4"); controller suffix ONLY appended for the
    # non-default v7c, scenario suffix ONLY appended for the non-default
    # "spill" (--controller v4c --scenario crossing keeps the ORIGINAL
    # filenames exactly, so existing crossing/v4c artifacts are unaffected).
    ctrl_suffix = "" if args.controller == "v4c" else f"_{args.controller}"
    scenario_suffix = "" if args.scenario == "crossing" else f"_{args.scenario}"
    monitor_suffix = "" if args.monitor == "classifier" else f"_{args.monitor}"
    monitor_suffix += "" if args.grasp_mode == "faked" else "_realgrasp"   # never overwrite faked artifacts
    monitor_suffix += "" if args.controller_impl == "py" else f"_{args.controller_impl}"   # never overwrite py artifacts
    video_path = ART_DIR / f"cup_collect_safety_demo_{args.config}{ctrl_suffix}{scenario_suffix}{monitor_suffix}.mp4"
    meta_path = ART_DIR / f"cup_collect_safety_demo_meta_{args.config}{ctrl_suffix}{scenario_suffix}{monitor_suffix}.json"

    if not args.no_video:
        print(f"[video] compositing HUD + encoding {len(frames)} frames...")
        vid_meta = build_hud_and_encode(log_main, frames, video_path, config=args.config,
                                         controller=args.controller, scenario=args.scenario,
                                         monitor=args.monitor)
        meta["video"] = vid_meta
        print(f"  wrote {video_path}")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] total wall time {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
