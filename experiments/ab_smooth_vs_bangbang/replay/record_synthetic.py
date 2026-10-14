"""record_synthetic.py -- branch-coverage records on RANDOM robot states of the cup-collect model (no episode).

Complements the flagship-episode records: it exercises the option combinations a short demo does not hit
(true / eff-gravity cone, tilt box, leash on/off, table rows, speed cap, ego_hold, F6 tau_max_vec, F5 delay,
adaptive_yield, large H_MARGIN -> relaxed / zero fallback) for three controller classes:
  FLMPC (the reference Python path), ValueFLMPC (the monkeypatched-solver path, synthetic value rows),
  RTFLMPC (the realtime drop-in `step`, numba -- pure computation, no hardware/ROS/socket).
Modelled on experiments/realtime/rt_unit_branches.py (same state generator).

  python record_synthetic.py --out <dir> --n-flmpc 40 --n-value 12 --n-rt 12 --seed 7

SIMULATION ONLY.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

import replay_core as rc


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--models-dir", default=None, help="shared model store (default <out>/models)")
    ap.add_argument("--n-flmpc", type=int, default=40, help="random FLMPC option combos (x --trials steps each)")
    ap.add_argument("--n-value", type=int, default=12)
    ap.add_argument("--n-rt", type=int, default=12)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    import mujoco
    import cup_collect_task_utils as cct
    import sim_common as sc
    import mpc_fl
    import flmpc_value as fv

    rt = None
    if args.n_rt > 0:
        import rt_flmpc as rt  # noqa: F401  (numba kernels; pure computation)

    m, d = cct.load_yor_cup_collect_model(keep_zone_marker=False)
    _, lq, ld = sc.joint_addrs(m, sc.LEFT_ARM_JOINTS)
    _, rq, rd = sc.joint_addrs(m, sc.RIGHT_ARM_JOINTS)
    lee = m.body("left_arm_gripper_base").id
    ree = m.body("right_arm_gripper_base").id
    cup = m.body("cup").id
    lk = sc.barrier_keypoints(m, sc.EGO_PREFIX)
    rk = sc.barrier_keypoints(m, sc.INT_PREFIX)
    rng = np.random.default_rng(args.seed)
    rec = rc.Recorder(args.out, sc, source=f"synthetic:s{args.seed}", always=True,
                      max_records=10 ** 6, max_special=10 ** 6, models_dir=args.models_dir)
    t0 = time.perf_counter()
    idx = [0]

    def one_controller(cls, cname, kw, ncases_tag, value=False, rt_mode=False):
        ctrl = cls(m, **kw) if not value else cls(m, **kw)
        rec.note_ctor(ctrl, cname, (), kw)
        NP = ctrl.NP
        for trial in range(args.trials):
            mujoco.mj_resetData(m, d)
            scale = float(rng.choice([0.4, 0.4, 0.8]))
            d.qpos[lq] = rng.normal(0, scale, 6)
            d.qpos[rq] = rng.normal(0, scale, 6)
            d.qvel[ld] = rng.normal(0, 0.6, 6)
            d.qvel[rd] = rng.normal(0, 0.6, 6)
            mujoco.mj_forward(m, d)
            qer = np.tile(d.qpos[lq] + rng.normal(0, 0.05, 6), (NP + 1, 1))
            qir = np.tile(d.qpos[rq] + rng.normal(0, 0.1, 6), (NP + 1, 1))
            qde = rng.normal(0, 0.05, (NP + 1, 6)) if rng.random() < 0.3 else np.zeros((NP + 1, 6))
            qdi = np.zeros((NP + 1, 6))
            pairs = sc.all_pair_barriers(m, d, ld, rd)
            if ctrl.adaptive_yield:      # exercise the hdot fallbacks (pair dicts from other callers lack fields)
                mode = str(rng.choice(["full", "no_hdot", "no_hdot_no_v"]))
                drop = {"full": (), "no_hdot": ("hdot",), "no_hdot_no_v": ("hdot", "v_ego", "v_int")}[mode]
                pairs = [{k_: v_ for k_, v_ in p_.items() if k_ not in drop} for p_ in pairs]
            Mf = sc.full_mass_matrix(m, d)
            Mb, bb = sc.payload_reflected_terms(m, d, lee, cup, ld)
            payload = rng.random() < 0.7
            Ml = Mf[np.ix_(ld, ld)] + (Mb if payload else 0)
            Mr = Mf[np.ix_(rd, rd)]
            bl = d.qfrc_bias[ld] - d.qfrc_passive[ld] + (bb if payload else 0)
            br = d.qfrc_bias[rd] - d.qfrc_passive[rd]
            tr = None
            if rng.random() < 0.7:
                a_te, b_te = sc.table_plane_barrier_rows(m, d, lk, ld, np.linalg.inv(Ml), bl, 0.72, margin=0.006, kp_radius=True)
                a_ti, b_ti = sc.table_plane_barrier_rows(m, d, rk, rd, np.linalg.inv(Mr), br, 0.72, margin=0.006, kp_radius=True)
                tr = (a_te, b_te, a_ti, b_ti)
            # controller warm-start state (what an episode would carry): previous commands + in-flight queue
            ctrl.a_ego_prev = rng.normal(0, 1, 6)
            ctrl.a_int_prev = rng.normal(0, 1, 6)
            if ctrl.delay_steps > 0:
                ctrl._a_queue = [(rng.normal(0, 1, 6), rng.normal(0, 1, 6)) for _ in range(ctrl.delay_steps)]
            cone = rng.choice([None, 0, 8])
            kwa = dict(M_ego_extra=(Mb if payload else None), bias_ego_extra=(bb if payload else None),
                       table_rows=tr, ego_hold=bool(rng.random() < 0.6))
            if rng.random() < 0.85:
                kwa.update(left_ee_body=lee, tilt_max_rad=float(rng.choice([0.05, 0.1, 0.2])),
                           cup_cone_facets=(None if cone is None else int(cone)))
            if rng.random() < 0.6:
                kwa.update(leash_ee_body=lee, leash_max_m=float(rng.choice([0.04, 0.08, 0.14])))
            if (not rt_mode) and rng.random() < 0.3:     # per-call speed-cap kwargs (py path)
                kwa.update(int_ee_body=ree, v_ee_max=float(rng.choice([0.15, 0.28])))
            if value:
                n_blk = ctrl.n_blk
                mv = int(rng.integers(2, 7))
                Av = rng.normal(0, 0.3, (mv, n_blk))
                bv = rng.normal(0.5, 1.0, mv)
                xref = rng.normal(0, 5.0, n_blk)
                kwa["value_rows"] = (Av, bv, xref) if rng.random() < 0.8 else (Av, bv)
                # next-call warm start mirrors a live run: leave _rows to be overwritten by step()
            if rng.random() < 0.15:
                for k_ in ("M_ego_extra", "bias_ego_extra"):
                    kwa[k_] = None
            ca = (lq, ld, rq, rd, qer, qde, qir, qdi, pairs)
            try:
                rec.record_call(ctrl, cls.step, m, d, ca, kwa, idx=idx[0])
            except Exception as e:      # an unsupported option combo for this class: log and go on
                import traceback; traceback.print_exc(); print(f"[rec-syn] {cname} case skipped: {type(e).__name__}: {e}", flush=True)
            idx[0] += 1

    def rand_ctor_kw(allow_adaptive=True):
        kw = dict(NC=4, L=8, W_POS_EGO=float(rng.choice([800.0, 3000.0])), H_MARGIN=float(rng.choice([0.02, 0.02, 0.15, 0.4])),
                  delay_steps=int(rng.choice([0, 0, 1, 3])), cone_mode=str(rng.choice(["effgrav", "true"])))
        if rng.random() < 0.5:
            kw.update(ee_speed_cap=0.28, ee_speed_body_int=ree)
        if rng.random() < 0.4:
            kw.update(tau_max_vec=np.array([40.0, 40.0, 30.0, 10.0, 10.0, 5.0]))
        if rng.random() < 0.2:
            kw.update(left_ee_body=lee, tilt_max_rad=0.1, leash_ee_body=lee, leash_max_m=0.08)
        if rng.random() < 0.15:
            kw.update(K_PAIRS=int(rng.choice([2, 6])), coll_stride=int(rng.choice([1, 4])))
        if allow_adaptive and rng.random() < 0.2:
            kw.update(adaptive_yield=True, H_MARGIN_URGENT=0.06, URGENCY_HDOT_REF=0.15,
                      W_POS_EGO_MIN=200.0, LEASH_MAX_URGENT_M=0.14, CUP_TILT_SAFE_FRAC=0.6)
        return kw

    for i in range(args.n_flmpc):
        one_controller(mpc_fl.FLMPC, "FLMPC", rand_ctor_kw(), i)
    print(f"[rec-syn] FLMPC done: {rec.n_saved} records ({time.perf_counter() - t0:.0f}s)", flush=True)
    for i in range(args.n_value):
        kw = rand_ctor_kw()
        kw = dict(kw, rho_v=float(rng.choice([1e4, 1e3])), trust=(40.0 if rng.random() < 0.8 else None),
                  a_cap=(60.0 if rng.random() < 0.6 else None))
        one_controller(fv.ValueFLMPC, "ValueFLMPC", kw, i, value=True)
    print(f"[rec-syn] ValueFLMPC done: {rec.n_saved} records ({time.perf_counter() - t0:.0f}s)", flush=True)
    if args.n_rt > 0:
        for i in range(args.n_rt):
            kw = rand_ctor_kw(allow_adaptive=False)
            kw.pop("K_PAIRS", None); kw.pop("coll_stride", None)
            one_controller(rt.RTFLMPC, "RTFLMPC", kw, i, rt_mode=True)
        print(f"[rec-syn] RTFLMPC done: {rec.n_saved} records ({time.perf_counter() - t0:.0f}s)", flush=True)
    meta = dict(source="synthetic", seed=args.seed, records=rec.n_saved, fallback_hist=rec.stats["fallback"])
    print("[rec-syn]", json.dumps(meta), flush=True)
    with open(f"{args.out}/meta_synthetic_s{args.seed}.json", "w") as f:
        json.dump(meta, f, indent=1)


if __name__ == "__main__":
    main()
