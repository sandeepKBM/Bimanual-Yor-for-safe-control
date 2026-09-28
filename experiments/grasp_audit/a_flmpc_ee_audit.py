"""Part A audit: FL-MPC end-effector tracking + right-EE speed spike root cause.

SIMULATION ONLY. Imports scripts/make_cup_collect_safety_demo.py READ-ONLY and
instruments it non-invasively by replacing the demo module's `mujoco` reference
with a proxy whose mj_step() records extra per-step state before stepping.
Config = the comprehensive_eval flagship config (N_CUBES=3, cup_jitter_m=0.04,
V_EE_MAX_MS=0.28, ee_speed_limit/table_floor on, faked grasp, flmpc).

usage: python a_flmpc_ee_audit.py SEED OUT.npz
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO, REPO / "scripts", REPO / "experiments" / "ab_smooth_vs_bangbang", REPO / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mujoco  # noqa: E402

for _try in range(20):
    try:
        import make_cup_collect_safety_demo as demo  # noqa: E402
        break
    except Exception as e:  # other agent may be mid-edit
        print("import failed, retrying:", repr(e), flush=True)
        time.sleep(15)
import sim_common as sc  # noqa: E402
import mpc_fl  # noqa: E402

REC: dict = {}
STATE = {"fl_feasible": True, "fl_relaxed": False, "fl_calls": 0}

_orig_relaxed = mpc_fl.FLMPC._relaxed_fallback
_orig_step = mpc_fl.FLMPC.step


def _relaxed(self, *a, **k):
    STATE["fl_relaxed"] = True
    return _orig_relaxed(self, *a, **k)


def _step(self, *a, **k):
    STATE["fl_relaxed"] = False
    STATE["fl_calls"] += 1
    out = _orig_step(self, *a, **k)
    STATE["fl_feasible"] = bool(out[2])
    STATE["fl_called_this_step"] = True
    return out


mpc_fl.FLMPC._relaxed_fallback = _relaxed
mpc_fl.FLMPC.step = _step


class _MjProxy:
    def __init__(self, real, ctx):
        self._real = real
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._real, name)

    def mj_step(self, model, data):
        c = self._ctx
        v6 = np.zeros(6)
        row = {}
        for tag, bid in (("r", c["rb"]), ("l", c["lb"]), ("cup", c["cb"])):
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, bid, v6, 0)
            row[f"{tag}_pos"] = data.xpos[bid].copy()
            row[f"{tag}_quat"] = data.xquat[bid].copy()
            row[f"{tag}_w"] = v6[:3].copy()
            row[f"{tag}_v"] = v6[3:].copy()
        row["qpos"] = data.qpos.copy()
        row["qvel_r"] = data.qvel[c["rdof"]].copy()
        row["qvel_l"] = data.qvel[c["ldof"]].copy()
        row["tau_r"] = data.qfrc_applied[c["rdof"]].copy()
        row["tau_l"] = data.qfrc_applied[c["ldof"]].copy()
        row["eq_active"] = data.eq_active.copy().astype(float)
        row["fl_called"] = float(STATE.pop("fl_called_this_step", False))
        row["fl_feasible"] = float(STATE["fl_feasible"])
        row["fl_relaxed"] = float(STATE["fl_relaxed"])
        row["ncon"] = float(data.ncon)
        # max constraint force on the right gripper subtree (weld/contact transients)
        row["qfrc_constraint_r"] = data.qfrc_constraint[c["rdof"]].copy()
        for kk, vv in row.items():
            REC.setdefault(kk, []).append(vv)
        self._real.mj_step(model, data)


def main():
    seed = int(sys.argv[1])
    out = sys.argv[2]
    ncub = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    demo.N_CUBES = ncub
    demo.V_EE_MAX_MS = 0.28
    sc.D_MIN = 0.03
    sc.ALPHA1 = sc.ALPHA2 = 30.0
    sc.COMBINED_RADIUS = sc.R_EGO + sc.R_INT + sc.D_MIN
    jit = float(sys.argv[3]) if len(sys.argv) > 3 else 0.04
    info = demo.build_episode(seed=seed, scenario="crossing", ee_speed_limit=True,
                              cup_jitter_m=jit, n_cubes=ncub)
    ctx = dict(rb=info["right_ee_body"], lb=info["left_ee_body"], cb=info["cup_body"],
               rdof=np.asarray(info["right_dofadr"]), ldof=np.asarray(info["left_dofadr"]))
    real_mj = demo.mujoco
    demo.mujoco = _MjProxy(real_mj, ctx)
    t0 = time.time()
    try:
        log, _ = demo.run_episode(info, controller="flmpc", qp_enabled=True, capture=False,
                                  ee_speed_limit=True, table_floor=True)
    finally:
        demo.mujoco = real_mj
    summ = demo.summarize(log)
    # references via FK on a scratch MjData
    model = info["model"]
    scratch = mujoco.MjData(model)
    qpos = np.asarray(REC["qpos"])
    n = qpos.shape[0]
    ref_r = np.zeros((n, 3)); ref_rq = np.zeros((n, 4))
    ref_l = np.zeros((n, 3)); ref_lq = np.zeros((n, 4))
    track_l, track_r = info["track_l"], info["track_r"]
    for k in range(n):
        t = log["t"][k]
        t_r = log["right_track_t"][k]
        qr = track_r.sample(t_r) if t_r >= 0 else info["q_home"]
        ql = track_l.sample(t) if t < demo.RIGHT_START_T else info["q_lift"]
        scratch.qpos[:] = qpos[k]
        scratch.qpos[info["right_qposadr"]] = qr
        scratch.qpos[info["left_qposadr"]] = ql
        mujoco.mj_kinematics(model, scratch)
        ref_r[k] = scratch.xpos[ctx["rb"]]; ref_rq[k] = scratch.xquat[ctx["rb"]]
        ref_l[k] = scratch.xpos[ctx["lb"]]; ref_lq[k] = scratch.xquat[ctx["lb"]]
    arrs = {k: np.asarray(v) for k, v in REC.items()}
    arrs.update({f"log_{k}": np.asarray(v) for k, v in log.items() if isinstance(v, np.ndarray)})
    arrs.update(ref_r=ref_r, ref_rq=ref_rq, ref_l=ref_l, ref_lq=ref_lq,
                right_start_t=demo.RIGHT_START_T, dt=sc.DT, wall_s=time.time() - t0,
                eq_names=np.array([model.equality(i).name for i in range(model.neq)]))
    np.savez_compressed(out, **arrs)
    print("SUMMARY", seed, {k: summ[k] for k in ("collision", "right_ee_speed_max_ms", "qp_infeasible_frac_left",
                                                  "h_min", "placed_final")}, "wall", round(time.time() - t0, 1),
          flush=True)


if __name__ == "__main__":
    main()
