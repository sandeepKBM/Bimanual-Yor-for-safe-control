"""Shared harness for the rt_flmpc gates (G1-G4). SIMULATION ONLY -- nothing here
opens any hardware / network connection.

configure(profile)  sets the demo module globals exactly like the CLIs do:
  "flagship" -- scripts/make_cup_collect_safety_demo.py CLI defaults with
                --grasp-mode faked --bugfixes on --ctrlfixes on --ctrlfix-parts all
                (N_CUBES=6, cup_jitter 0.0);
  "ce"       -- experiments/ab_smooth_vs_bangbang/comprehensive_eval.run_one's
                faked/all-fixes job (N_CUBES=3, cup_jitter 0.04, V_EE_MAX 0.28,
                sc.D_MIN=0.03, ALPHA=30);
  "gate"     -- the library defaults gate_equivalence.py uses (bugfixes/ctrlfixes off).
build(profile, seed) -> info ; run(info, profile, **kw) -> (log, summary).
"""
from __future__ import annotations

import os

import _demo_env as env

demo, sc, mpc_fl = env.demo, env.sc, env.mpc_fl

_DEFAULTS = {}


def _snap():
    if not _DEFAULTS:
        for k in ("BUGFIXES", "CTRLFIXES", "CTRLFIX_PARTS", "N_CUBES", "V_EE_MAX_MS",
                  "CTRLFIX_DELAY_STEPS", "CTRLFIX_TAU_MAX_VEC", "CTRLFIX_CONE_FACETS"):
            _DEFAULTS[k] = getattr(demo, k)
        _DEFAULTS["sc"] = (sc.D_MIN, sc.ALPHA1, sc.ALPHA2, sc.COMBINED_RADIUS)


def configure(profile: str):
    _snap()
    for k, v in _DEFAULTS.items():
        if k != "sc":
            setattr(demo, k, v)
    sc.D_MIN, sc.ALPHA1, sc.ALPHA2, sc.COMBINED_RADIUS = _DEFAULTS["sc"]
    if profile == "gate":
        return
    demo.BUGFIXES = True
    demo.CTRLFIXES = True
    demo.CTRLFIX_PARTS = demo.parse_ctrlfix_parts("all")
    if profile == "ce":
        demo.N_CUBES = 3
        demo.V_EE_MAX_MS = 0.28
        sc.D_MIN = 0.03
        sc.ALPHA1 = sc.ALPHA2 = 30.0
        sc.COMBINED_RADIUS = sc.R_EGO + sc.R_INT + sc.D_MIN
    elif profile != "flagship":
        raise ValueError(profile)


def build(profile: str, seed: int):
    configure(profile)
    if profile == "gate":
        return demo.build_episode(seed=seed)
    if profile == "ce":
        return demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True, cup_jitter_m=0.04,
                                  n_cubes=3, grasp_mode="faked", bugfixes=True, ctrlfixes=True)
    return demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True, cup_jitter_m=0.0,
                              n_cubes=None, grasp_mode="faked", bugfixes=True, ctrlfixes=True)


def run(info, profile: str, max_steps=None, **kw):
    if profile == "gate":
        log, _ = demo.run_episode(info, controller="flmpc", capture=False, max_steps=max_steps, **kw)
    else:
        log, _ = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                                  ee_speed_limit=True, table_floor=True, cup_in_barrier=True,
                                  max_steps=max_steps, **kw)
    return log, demo.summarize(log)


RUN_DIR = os.environ.get("RT_RUN_DIR", "/data/local/ss5772/bimanual_runs/rt_flmpc")
