"""Latent-space HJ reachability (LatentSafe / Nakamura et al. RSS 2025 style) with
FL-MPC as the smoothing layer underneath. SIMULATION / SAVED DATA ONLY.

Modules
  slosh_pend      analytic liquid proxy (first slosh mode = spherical pendulum), numpy+torch
  grid_hj         exact planar grid HJ (discounted safety Bellman, semi-Lagrangian), numba
  reach_value     neural reachability learner (Fisac discounted safety Bellman, max over
                  controls, K-step greedy rollout bootstrapping) over a generic latent dynamics
  validate_stage0 learned-vs-grid validation + lead-time gate on held-out spill_v2 episodes
  kin_wm          Stage-1 learned latent WM (recurrent, cup kinematics + commanded cup accel)
  wm_adapter      LatentWM interface (encode / imagine / spill_margin) + adapters
                  (kin_wm, wm_dino.DinoLatentWM, world_model.v2.dinowm.api if present)
  flmpc_value     FL-MPC + linearized latent-value constraint rows (slackable, torque box hard)
  scenario        Stage-2 aggressive-carry controller comparison (modes a-e)
"""
