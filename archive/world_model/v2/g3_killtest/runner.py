"""runner.py -- policies + parallel sweep driver for the G3 kill test.

    python -m world_model.v2.g3_killtest.runner --classes T,S,P --sigmas 0,1,2 --seeds 100000-100199 \
        --arm main --workers 24 --tag main
Outputs JSONL (one line per episode) under /common/users/ss5772/bimanual_data/g3_killtest/results/.
SIMULATION ONLY: no hardware, no network.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.g3_killtest import core as C  # noqa: E402
from world_model.v2.g3_killtest import config as K  # noqa: E402
from world_model.v2.g3_killtest import episode as E  # noqa: E402

PRIMARY = ["never", "always_robust", "always_probe", "g3", "g3_tau", "tvar", "eig", "oracle"]
SENS = ["g3_tau@0.25", "g3_tau@0.75", "tvar@0.25", "tvar@0.75", "eig@0.25", "eig@0.75", "tvar_ug", "eig_ug"]
SECONDARY = ["dg", "prmppi", "shield", "safeopt", "g3_cost"]
ALL_POLICIES = PRIMARY + SENS + SECONDARY
DECIDERS = {"always_probe", "g3", "g3_tau", "tvar", "eig", "tvar_ug", "eig_ug", "dg", "safeopt", "g3_cost"}
CALIB_POLICIES = ("g3", "never")
TABLES = {}


def parse(pol):
    if "@" in pol:
        a, b = pol.split("@")
        return a, float(b)
    return pol, E.TAU_RULE


def score(base, act, U):
    if base == "g3_tau":
        return act["r_g3"]
    if base == "tvar":
        return act["r_tvar"]
    if base == "eig":
        return act["r_eig"]
    if base == "safeopt":
        return act["so"][E.SO_L_POLICY] / U if U > 0 else 0.0
    raise ValueError(base)


def decide(pol, info, n_here, n_ep, ep, st, k):
    base, tau = parse(pol)
    ungated = base.endswith("_ug")
    base = base.replace("_ug", "")
    U = info["U"]
    acts = info["actions"]
    probes = [x for x in acts if x != "a"]
    out = dict(act="continue", b=None)
    if base == "always_probe":
        if U >= E.U_GATE and n_here < 1 and n_ep < E.MAX_PROBES_EP and probes:
            b = max(probes, key=lambda x: acts[x]["dU"])
            return dict(act="probe", b=b[1])
        return out
    if base == "dg":            # Dual-gatekeeper-style: theta-width shrinkage + cost budget, U-blind
        if info["width0"] <= E.DG_WIDTH_GOAL or n_ep >= E.MAX_PROBES_EP or n_here >= E.MAX_PROBES_PER_DP:
            return out
        c = [x for x in probes if acts[x]["dur"] <= E.DG_PROBE_MAX_S + 1e-9
             and st.probe_time + acts[x]["dur"] <= E.DG_BUDGET_S + 1e-9]
        if not c:
            return dict(act="robust", b=None)
        b = max(c, key=lambda x: acts[x]["r_width"])
        if acts[b]["r_width"] >= E.DG_TAU:
            return dict(act="probe", b=b[1], r=acts[b]["r_width"])
        return dict(act="robust", b=None)
    if not ungated and U < E.U_GATE:
        return out
    if base == "g3":            # value-level VOI: minimize one-step-lookahead expected remaining time
        if n_here >= E.MAX_PROBES_PER_DP or n_ep >= E.MAX_PROBES_EP or not probes:
            return dict(act="continue" if acts["a"]["r_g3"] >= tau else "robust", b=None)
        v = E.voi(ep, st, k, info)
        x = min(v, key=lambda z: v[z])
        if x == "a":
            return dict(act="continue" if acts["a"]["r_g3"] >= tau else "robust", b=None, Ta=round(v["a"], 3))
        return dict(act="probe", b=x[1], Ta=round(v["a"], 3), Tb=round(v[x], 3))
    if base == "g3_cost":
        ra = acts["a"]["dU_safe"]
        if n_here >= E.MAX_PROBES_PER_DP or n_ep >= E.MAX_PROBES_EP or not probes or U <= 0:
            return dict(act="continue" if acts["a"]["r_g3"] >= tau else "robust", b=None)
        stake = E.stake_time(ep, st, k, info)
        gain = {x: (acts[x]["dU_safe"] - max(ra, 0.0)) / U * stake - acts[x]["dur"] for x in probes}
        b = max(probes, key=lambda x: gain[x])
        if gain[b] > 0:
            return dict(act="probe", b=b[1], stake=stake, gain=gain[b])
        return dict(act="continue" if acts["a"]["r_g3"] >= tau else "robust", b=None, stake=stake)
    ra = score(base, acts["a"], U)
    if ra >= tau:
        return dict(act="continue", b=None, ra=ra)
    if n_here >= E.MAX_PROBES_PER_DP or n_ep >= E.MAX_PROBES_EP or not probes:
        return dict(act="robust", b=None, ra=ra)
    b = max(probes, key=lambda x: score(base, acts[x], U))
    rb = score(base, acts[b], U)
    if rb >= tau:
        return dict(act="probe", b=b[1], ra=ra, rb=rb)
    return dict(act="robust", b=None, ra=ra, rb=rb)


# ------------------------------------------------------------------------------ calibration pairs
def collect_calib(ep, st, k, hist, info, calib, src):
    if info["U"] <= 0:
        return
    acts = info["actions"]
    probes = [x for x in acts if x != "a"]
    chosen = {"a"}
    for key in ("dU", "r_tvar", "r_width", "r_eig"):
        if probes:
            chosen.add(max(probes, key=lambda x: acts[x][key]))
    if probes:
        chosen.add(max(probes, key=lambda x: acts[x]["so"][E.SO_L_POLICY]))
        rng = E.key_rng(ep.seed, hist, salt=23)
        for j in rng.choice(len(probes), min(3, len(probes)), replace=False):
            chosen.add(probes[j])
    for x in sorted(chosen, key=repr):
        ck = (hist, x)
        if ck in ep.real_cache:          # already paired (dedupe across runs)
            continue
        real = E.realize(ep, st, k, hist, x, info)
        a = acts[x]
        calib.append(dict(src=src, k=k, nh=len(hist), U=info["U"], var0=info["var0"], width0=info["width0"],
                          action="a" if x == "a" else "p", b=None if x == "a" else x[1],
                          pred=dict(dU=a["dU"], dU_sd=a["dU_sd"], dU_safe=a["dU_safe"], r_g3=a["r_g3"],
                                    r_tvar=a["r_tvar"], r_width=a["r_width"], eig=a["eig"], r_eig=a["r_eig"],
                                    so=a["so"], dur=a["dur"], exc=a["excitation"], s=a["s"]),
                          real=real))


# ------------------------------------------------------------------------------ PRMPPI-style
def prmppi_segment(ep, st, k, hist, sample=True):
    """PRMPPI-style (Vahs et al.): posterior-SAMPLED nominal plan + per-step reactive switch to
    the robust backup when the credible-set safety check fails; no forecasting, no probes.
    sample=False ('shield'): the plan is the unscaled nominal segment."""
    seg = ep.segs[k]
    B = st.B
    if sample:
        rng = E.key_rng(ep.seed, hist, salt=5)
        j = int(rng.choice(len(B.IDX), p=B.w()))
        s_t, ch = E.certify(ep, st, lambda s_: E.seg_chunk(seg, s_), B=B.sub(j))
    else:
        s_t, ch = 1.0, E.seg_chunk(seg, 1.0)
    if s_t is None:
        s_t = float(E.SCALES[-1])
        ch = E.seg_chunk(seg, s_t)
    n_prof = len(ch) - E.DWELL
    flag = dict(sw=False, i=0)

    def ctrl(stt, i):
        if i >= len(ch):
            return None
        a = ch[i]
        Bs = stt.B
        vals = C.next_values(ep.cid, Bs.S, Bs.IDX, Bs.THg, Bs.AXg, Bs.Vg, float(a))
        if C.wquantile(vals, Bs.w(), E.QL) > 0.0:
            return float(a)
        flag["sw"] = True
        flag["i"] = i
        return None
    key = hist + (("pr", k, s_t),)
    E.execute_stepwise(ep, st, len(ch), ctrl, key)
    info = dict(s_sample=s_t, switched=flag["sw"])
    if not flag["sw"] or st.failed:
        return key, info
    key = key + (("prbk", k),)

    def bctrl(stt, i):
        Bs = stt.B
        v = C.values_at(Bs.S, Bs.IDX, Bs.AXg, Bs.Vg)
        if i > 0 and C.wquantile(v, Bs.w(), E.QL) >= 0.5:
            return None
        a, _ = C.robust_backup_action(ep.cid, Bs.S, Bs.IDX, Bs.THg, Bs.AXg, Bs.Vg, Bs.w(), C.U_FB, E.QL)
        return float(a)
    n = E.execute_stepwise(ep, st, E.BACKUP_STEPS, bctrl, key)
    st.backup_time += n * C.DT
    frac0 = min(flag["i"], n_prof) / max(n_prof, 1)
    if frac0 < 0.999 and not st.failed:
        key, s, nm, n_bk, unc = E.exec_segment(ep, st, k, key, frac0=frac0, tag="prrem")
        info.update(s_rem=s, frac0=frac0)
    return key, info


# ------------------------------------------------------------------------------ one policy run
def run_policy(ep, pol, calib=None):
    t0 = time.time()
    B = ep.oracle_belief() if pol == "oracle" else ep.belief(frozen=(pol == "always_robust"))
    st = E.State(ep, B)
    hist = ()
    rec = dict(policy=pol, decisions=[], segs=[])
    n_ep = 0
    base = parse(pol)[0]
    U0 = None
    for k in range(E.K_SEG):
        if st.failed:
            break
        n_here = 0
        while base in DECIDERS:
            info = E.compute_info(ep, st, k, hist)
            if calib is not None:
                collect_calib(ep, st, k, hist, info, calib, pol)
            d = decide(pol, info, n_here, n_ep, ep, st, k)
            rec["decisions"].append(dict(k=k, U=round(info["U"], 4), **{a: (round(v, 4) if isinstance(v, float) else v)
                                                                           for a, v in d.items()}))
            if d["act"] != "probe":
                break
            b = d["b"]
            hist = hist + (("p", b),)
            nm = E.execute(ep, st, E.probe_chunk(b), hist)
            st.probe_time += len(E.probe_chunk(b)) * C.DT
            n_ep += 1
            n_here += 1
            rec["decisions"][-1].update(nm_moved=nm["moved"], nm_snr2=round(nm["snr2"], 2))
            if st.failed:
                break
        if st.failed:
            break
        segrec = dict(k=k)
        if pol not in ("oracle", "always_robust"):
            rs = E.route_state(ep, st, k)
            len_k = rs["lens"][0]
            useg = float(rs["und"][:len_k].mean())
            segrec.update(U=round(rs["U"], 4), undec=bool(rs["seg_und"][0]), useg=round(useg, 4),
                          undec_tol=bool(useg >= E.K1_TOL))
            if U0 is None:
                U0 = rs["U"]
            if calib is not None and base not in DECIDERS:
                info = E.compute_info(ep, st, k, hist)
                collect_calib(ep, st, k, hist, info, calib, pol)
        if pol in ("prmppi", "shield"):
            hist, pinfo = prmppi_segment(ep, st, k, hist, sample=(pol == "prmppi"))
            segrec.update(pinfo)
        else:
            hist, s, nm, n_bk, unc = E.exec_segment(ep, st, k, hist)
            segrec.update(s=s, n_bk=n_bk, uncert=unc, **{f"nm_{a}": (round(v, 3) if isinstance(v, float) else v)
                                                           for a, v in nm.items()})
        rec["segs"].append(segrec)
    rec.update(violated=bool(st.failed), fail_t=st.fail_t, time=round(st.t * C.DT, 3),
               ratio=round(st.t * C.DT / ep.nom_time, 4), n_probes=n_ep, probe_time=round(st.probe_time, 3),
               backup_time=round(st.backup_time, 3), lmin=round(float(st.lmin), 4), U0=U0, n_rs=st.n_resample,
               secs=round(time.time() - t0, 2))
    return rec


def run_job(args):
    cls, sig_idx, seed, arm, policies = args
    t0 = time.time()
    ep = E.Episode(cls, sig_idx, seed, TABLES[cls], arm)
    rs0 = E.route_state(ep, E.State(ep, ep.belief()), 0)
    U0 = rs0["U"]
    seg_und0 = rs0["seg_und"]
    calib = []
    recs = []
    for pol in policies:
        recs.append(run_policy(ep, pol, calib if pol in CALIB_POLICIES else None))
    return dict(cls=cls, sigma=ep.sigma_label, sigma_idx=sig_idx, seed=seed, arm=arm, theta=ep.theta.tolist(),
                route_type=ep.route["type"], peaks=[s["A"] for s in ep.segs], gain=ep.gain, U0=U0,
                seg_und0=seg_und0,
                nom_time=round(ep.nom_time, 3), recs=recs, calib=calib, secs=round(time.time() - t0, 1))


def load_tables(cls):
    z = np.load(K.value_path(cls))
    TH = z["TH"]
    assert np.allclose(TH, K.particle_grid(cls))
    return (TH, np.ascontiguousarray(z["V"]), z["AX"])


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, (str, int, float)) else k: _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def parse_seeds(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", default="T,S,P")
    ap.add_argument("--sigmas", default="0,1,2")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--arm", default="main")
    ap.add_argument("--policies", default=",".join(ALL_POLICIES))
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    classes = a.classes.split(",")
    for c in classes:
        TABLES[c] = load_tables(c)
    pols = a.policies.split(",")
    jobs = [(c, int(si), sd, a.arm, pols) for c in classes for si in a.sigmas.split(",") for sd in parse_seeds(a.seeds)]
    # interleave classes so P (slowest) does not all land at the end
    jobs.sort(key=lambda j: (j[2], j[1], j[0]))
    out = K.OUT / "results" / f"{a.tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
                done.add((r["cls"], r["sigma_idx"], r["seed"], r["arm"]))
            except Exception:
                pass
    jobs = [j for j in jobs if (j[0], j[1], j[2], j[3]) not in done]
    print(f"[run] {len(jobs)} jobs ({len(done)} already done) -> {out}", flush=True)
    t0 = time.time()
    ctx = get_context("fork")
    with ctx.Pool(a.workers, maxtasksperchild=20) as pool, open(out, "a") as f:
        for i, r in enumerate(pool.imap_unordered(run_job, jobs)):
            f.write(json.dumps(_jsonable(r)) + "\n")
            f.flush()
            if i % 20 == 0 or i == len(jobs) - 1:
                el = time.time() - t0
                print(f"[run] {i + 1}/{len(jobs)} {el:.0f}s (last {r['cls']} {r['sigma']} seed {r['seed']} "
                      f"{r['secs']}s)", flush=True)
    (K.OUT / "results" / f"_DONE_{a.tag}").write_text(time.strftime("%Y-%m-%dT%H:%M:%S"))


if __name__ == "__main__":
    main()
