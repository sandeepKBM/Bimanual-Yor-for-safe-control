"""world_model/v2/dinowm/gains_gate.py -- can ANY world model beat physics on spill_v2?

SIMULATION DATA ONLY (reads shards; no robot, no network).

Per contents group, on the deterministic per-seed split of shards.py:
  (a)  b3-nominal     world-frame LTI filter, params from experiments/slosh
                      scores_phys.npz (fit on the legacy nominal-bead rig data)
  (a') b3-pooled      b3 refit on THIS data's train split for the group, WITHOUT
                      contents info (what physics gets when contents are hidden)
  (b)  b3-oracle-type b3 refit per TRUE contents type (beads | particles)
  (b+) b3-oracle-fill b3 refit per true type x fill-height tercile + per-bin Platt
                      (true contents known up to fill level)
  (p*) pend-generator ONLY for the pend labels: the labelling pendulum itself with
                      the true fill h (logged pend_phi_deg minus rim angle phi*(h)).
                      This is the physics ceiling for pend labels.
  (c)  GRU            model/proprio_gru.HazardGRU on proprio + FK cup kin + action
                      (trained here on the train split, --gru-steps)
b3 runs at 200 Hz on the LOGGED FK cup accel/axis (causal) and is sampled at the
WM ticks (frame t <-> dense index 13 t).

Metrics (test split; threshold for events chosen on val at <= 1 FA/min calm):
  epA_pre   episode AUROC of max score over PRE-ONSET steps (prediction)
  epA_full  episode AUROC of max score over the whole episode (what contents_study
            reports; includes post-onset -> partly detection)
  A@0.5     step-level pre-onset AUROC, spill within 0.5 s (eval/metrics.py)
  det, lead_med, det>=0.5s   event metrics from eval/harness.py
  + bootstrap 95% CI on epA_pre over val+test episodes (params never fit there).

  python -m world_model.v2.dinowm.gains_gate --out $TMP/gate.json [--gru-steps 1500 --device cuda]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from world_model.v2.dinowm import shards as S  # noqa: E402
from world_model.v2.eval import harness as Hn  # noqa: E402
from world_model.v2.eval import metrics as M  # noqa: E402

GRID_W = (5, 10, 15, 20, 35, 50, 70)
GRID_Z = (0.2, 0.5, 1.0, 2.5)
GRID_K = (0.0, 0.03, 0.1, 0.3, 1.0)
G = np.array([0.0, 0.0, -9.81])
H_FIT = 0.5


def nominal_b3():
    P = json.loads(str(np.load(REPO / "experiments/slosh/data/scores_phys.npz", allow_pickle=True)["params"]))
    p = P["b3_world_lti"]; return (float(p["w"]), float(p["zeta"]), float(p["kappa"]))


# --------------------------------------------------------------------------- b3 (batched, 200 Hz)
def b3_phi_batch(eps, w, zeta, dt=S.DT200):
    """-> list of phi (T200,) arrays [rad] (no kappa)."""
    L = [len(e["dense"]["acc"]) for e in eps]; Tm = max(L); E = len(eps)
    F = np.empty((E, Tm, 3)); N = np.empty((E, Tm, 3))
    for i, e in enumerate(eps):
        f = e["dense"]["acc"] - G; n = e["dense"]["axis"]
        F[i, :L[i]] = f; F[i, L[i]:] = f[-1]; N[i, :L[i]] = n; N[i, L[i]:] = n[-1]
    Ev = F / np.linalg.norm(F, axis=-1, keepdims=True)
    N /= np.linalg.norm(N, axis=-1, keepdims=True)
    x = Ev[:, 0].copy(); xd = np.zeros((E, 3)); X = np.empty((E, Tm, 3)); c = 2 * zeta * w
    for k in range(Tm):
        xd += dt * (w * w * (Ev[:, k] - x) - c * xd)
        x += dt * xd
        X[:, k] = x
    cos = np.sum(X * N, -1) / (np.linalg.norm(X, axis=-1) + 1e-9)
    phi = np.arccos(np.clip(cos, -1, 1))
    return [phi[i, :L[i]] for i in range(E)]


def with_kappa(phis, eps, kappa, dt=S.DT200):
    out = []
    for p, e in zip(phis, eps):
        r = p + kappa * np.clip(np.r_[0.0, np.diff(p) / dt], 0, None)
        out.append(r[e["dense"]["idx"]])
    return out


# --------------------------------------------------------------------------- metrics
def ep_scores(scores, eps, pre=True):
    s, y = [], []
    for p, e in zip(scores, eps):
        on = e["onset"]; p = np.asarray(p, float)
        p = p if p.ndim == 1 else p[:, -1]
        seg = p[:on] if (pre and on > 0) else p
        s.append(seg.max() if len(seg) else p[0]); y.append(on >= 0)
    return np.array(s), np.array(y)


def step_auroc(scores, eps, H=H_FIT):
    s, y, _ = M.stack_valid([np.asarray(p, float) if np.ndim(p) == 1 else np.asarray(p)[:, 1] for p in scores],
                            eps, H, 0, 1)
    return M.auroc(s, y)


def boot_ci(s, y, n=1000, seed=0):
    rng = np.random.default_rng(seed); v = []
    for _ in range(n):
        i = rng.integers(0, len(s), len(s))
        a = M.auroc(s[i], y[i])
        if np.isfinite(a):
            v.append(a)
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if v else (np.nan, np.nan)


class _Pre:
    """harness adapter for precomputed per-episode scores."""
    probabilistic = False

    def __init__(self, name, table, horizons_s=None):
        self.name, self.t, self.horizons_s = name, table, horizons_s

    def predict(self, ep):
        return self.t[ep["id"]]


def summarize(name, fn, split, horizons_s=None):
    """fn(eps)->scores. Returns metric dict on test (+CI on val+test)."""
    te, va = split["test"], split["val"]
    st, sv = fn(te), fn(va)
    table = {e["id"]: s for e, s in zip(te + va, st + sv)}
    rep = Hn.evaluate(_Pre(name, table, horizons_s), va, {"test": te})
    ev = rep["sets"]["test"]["event"]
    s, y = ep_scores(st, te, True); sf, _ = ep_scores(st, te, False)
    s2, y2 = ep_scores(st + sv, te + va, True)
    ph = rep["sets"]["test"]["per_horizon"]
    return dict(name=name, n_test=len(te), n_pos_test=int(y.sum()), epA_pre=M.auroc(s, y), epA_full=M.auroc(sf, y),
                epA_pre_valtest=M.auroc(s2, y2), epA_pre_valtest_ci=boot_ci(s2, y2),
                A025=ph["0.25s"]["auroc_pre_onset"], A05=ph["0.5s"]["auroc_pre_onset"], A1=ph["1s"]["auroc_pre_onset"],
                det=ev["detection_rate"], lead_med=ev["lead_med_s"], det05=ev["det@lead>=0.5s"],
                fa_test=ev["fa_per_min"])


# --------------------------------------------------------------------------- fitting
class B3Bank:
    """phi for every (w,zeta) on a set of episodes, cached."""

    def __init__(self, eps):
        self.eps = eps; self.cache = {}; self.idx = {e["id"]: i for i, e in enumerate(eps)}

    def phi(self, w, z):
        if (w, z) not in self.cache:
            self.cache[(w, z)] = b3_phi_batch(self.eps, w, z)
        return self.cache[(w, z)]

    def scores(self, sub_ids, w, z, k):
        ph = self.phi(w, z)
        sel = [self.idx[i] for i in sub_ids]
        return with_kappa([ph[i] for i in sel], [self.eps[i] for i in sel], k)


def fit_b3(bank, train_eps):
    ids = [e["id"] for e in train_eps]; best = None
    for w in GRID_W:
        for z in GRID_Z:
            for k in GRID_K:
                sc = bank.scores(ids, w, z, k)
                s, y = ep_scores(sc, train_eps, True)
                obj = M.auroc(s, y) + step_auroc(sc, train_eps)
                if np.isfinite(obj) and (best is None or obj > best[0]):
                    best = (obj, (w, z, k))
    return best[1]


def platt(sc, eps, iters=100):
    """Episode-level 1-D logistic fit of P(episode spills | pre-onset max score);
    returns the LOGIT map a*s+b (monotone, a>0, never saturates) so per-bin scores
    become comparable across bins."""
    s, y = ep_scores(sc, eps, True)
    mu, sd = s.mean(), s.std() + 1e-9; x = (s - mu) / sd; y = y.astype(float)
    A, B = 1.0, 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(A * x + B, -30, 30)))
        g = np.array([np.sum((p - y) * x), np.sum(p - y)]) + 1e-2 * np.array([A, 0.0])
        wgt = p * (1 - p) + 1e-6
        Hm = np.array([[np.sum(wgt * x * x) + 1e-2, np.sum(wgt * x)], [np.sum(wgt * x), np.sum(wgt) + 1e-6]])
        st = np.linalg.solve(Hm, g); A, B = A - st[0], B - st[1]
        if np.abs(st).max() < 1e-9:
            break
    A = max(A, 1e-3)
    return lambda v: A * (np.asarray(v) - mu) / sd + B


def fill_bins(train_eps, nb=3):
    h = np.array([e["meta"]["fill_h"] for e in train_eps])
    return np.quantile(h, np.linspace(0, 1, nb + 1)[1:-1])


def oracle_fill_fn(bank, split, key=lambda e: e["meta"]["kind"], nb=3):
    """per (type, fill tercile) b3 refit + per-bin Platt -> comparable probabilities."""
    models = {}
    tr = split["train"]
    types = sorted({key(e) for e in tr})
    for t in types:
        tre = [e for e in tr if key(e) == t]
        edges = fill_bins(tre, nb)
        for b in range(nb):
            be = [e for e in tre if np.searchsorted(edges, e["meta"]["fill_h"]) == b]
            if sum(e["onset"] >= 0 for e in be) < 3 or len(be) < 10:
                be = tre                                   # too few positives: fall back to type level
            p = fit_b3(bank, be)
            models[(t, b)] = (p, platt(bank.scores([e["id"] for e in be], *p), be), edges)

    def fn(eps):
        out = []
        for e in eps:
            t = key(e); edges = models[(t, 0)][2]
            b = int(np.searchsorted(edges, e["meta"]["fill_h"]))
            p, cal, _ = models[(t, b)]
            out.append(cal(bank.scores([e["id"]], *p)[0]))
        return out
    return fn, {f"{k[0]}/fill{k[1]}": v[0] for k, v in models.items()}


# --------------------------------------------------------------------------- GRU baseline
def train_gru(split, steps=1500, device="cpu", seed=0, log=print):
    import torch
    from world_model.v2.model import common as C
    from world_model.v2.model.proprio_gru import HazardGRU, _prep_eps
    tr = split["train"]
    norm = C.Normalizer().fit([C.step_features(e, True, True) for e in tr])
    X, Y, V = _prep_eps(tr, norm, True, True, C.HORIZONS_S)
    pw = torch.tensor(C.auto_pos_weight(Y, V), device=device)
    C.seed_all(seed); rng = np.random.default_rng(seed)
    net = HazardGRU(X[0].shape[1], 128, 3).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tens = lambda A: torch.tensor(C.pad_batch(A), device=device)
    t0 = time.time()
    for it in range(1, steps + 1):
        b = rng.integers(0, len(X), 32)
        x, y, v = tens([X[i] for i in b]), tens([Y[i] for i in b]), tens([V[i].astype(np.float32) for i in b])
        lam, _ = net(x)
        loss = (C.MonotoneHazard.bce(lam, y, pw) * v[..., None]).sum() / v.sum().clamp_min(1) / 3
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 10.0); opt.step()
        if it % 500 == 0 or it == 1:
            log(f"    [gru] step {it} loss {float(loss.detach()):.4f} ({time.time()-t0:.0f}s)")
    net.eval()

    @torch.no_grad()
    def fn(eps):
        return [(-torch.expm1(-net(torch.tensor(norm(C.step_features(e, True, True)), device=device)[None])[0][0])
                 ).cpu().numpy() for e in eps]
    return fn


# --------------------------------------------------------------------------- plan-aware physics
PLAN_H = (0.25, 0.5, 1.0)


def plan_aware(fn):
    """1-D causal score -> (T,3): column j = max of the score over [t, t+H_j] using the
    FUTURE cup kinematics. In rig mode the realized cup motion tracks the scripted plan
    (IK playback), so this is 'physics rolled forward on the known plan' -- the fair
    comparator for WM imagination under planned actions. NOT causal in observations."""
    def f(es):
        out = []
        for e, sc in zip(es, fn(es)):
            sc = np.asarray(sc, float); T = len(sc); cols = []
            for h in PLAN_H:
                n = M.horizon_steps(h, e["hz"])
                pad = np.r_[sc, np.full(n, sc[-1])]
                cols.append(np.lib.stride_tricks.sliding_window_view(pad, n + 1)[:T].max(1))
            out.append(np.stack(cols, 1))
        return out
    return f


def _pend_one(args):
    pos, quat, h, idx = args
    sys.path.insert(0, str(REPO / "world_model/v2/data"))
    from contents_study import pendulum_run
    import contents as Cn
    phi, _, _, _, _, phs = pendulum_run(pos.astype(float), quat.astype(float), h, zeta=0.02, sub=4)
    return phi[idx] - phs


def pend_phys(eps, h_fn, workers=16):
    """equivalent-mechanical pendulum (the pend-label generator) with fill h = h_fn(ep)."""
    import multiprocessing as mp
    jobs = [(e["dense"]["pos"], e["dense"]["quat"], float(np.clip(h_fn(e), 0.02, 0.07)), e["dense"]["idx"]) for e in eps]
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_pend_one, jobs, chunksize=8)
    return {e["id"]: r for e, r in zip(eps, res)}


# --------------------------------------------------------------------------- main
def run(a):
    log = lambda *x: print(*x, flush=True)
    t0 = time.time()
    eps = S.load_all(workers=a.workers)
    log(f"[gate] {len(eps)} episodes from {len(S.shard_paths())} complete shards ({time.time()-t0:.0f}s)")
    nom = nominal_b3(); log(f"[gate] b3 nominal params (w,zeta,kappa) = {nom}")
    bank = B3Bank(eps)
    t1 = time.time(); [bank.phi(w, z) for w in GRID_W for z in GRID_Z]
    if (nom[0], nom[1]) not in bank.cache:
        bank.phi(nom[0], nom[1])
    log(f"[gate] b3 grid ({len(bank.cache)} (w,zeta)) in {time.time()-t1:.0f}s")
    results = {}
    gru_models = {}; pend_cache = {}
    for label in ("spill", "pend"):
        E = [S.set_label(e, label) for e in eps]
        spl_all = S.by_split(E)
        groups = [("beads", lambda e: e["meta"]["kind"] == "beads"), ("particles", lambda e: e["meta"]["kind"] == "particles")]
        if label == "pend":
            groups = [("all", lambda e: True)] + groups
        if a.gru_steps > 0:
            log(f"[gate] training GRU on label={label} (all contents, train split, {a.gru_steps} steps)")
            gru_models[label] = train_gru(spl_all, a.gru_steps, a.device, log=log)
        for gname, pred in groups:
            key = f"{label}/{gname}"
            spl = {k: [e for e in v if pred(e)] for k, v in spl_all.items()}
            if sum(e["onset"] >= 0 for e in spl["test"]) < 2:
                log(f"[gate] {key}: too few test positives, skip"); continue
            rows = []
            ids = lambda es: [e["id"] for e in es]
            from world_model.v2.eval.predictors import ClockBaseline
            clk = ClockBaseline(spl["train"])
            rows.append(summarize("(0) clock (time-since-start only)", lambda es: [clk.predict(e) for e in es], spl, PLAN_H))
            rows.append(summarize("(a) b3-nominal", lambda es: bank.scores(ids(es), *nom), spl))
            rows.append(summarize("(a+plan) b3-nominal on the plan", plan_aware(lambda es: bank.scores(ids(es), *nom)), spl, PLAN_H))
            p_pool = fit_b3(bank, spl["train"])
            rows.append(summarize(f"(a') b3-pooled-refit {p_pool}", lambda es, p=p_pool: bank.scores(ids(es), *p), spl))
            if gname == "all":
                # type-level oracle: per true kind refit + Platt so the two kinds are comparable
                fn_t, pt = oracle_fill_fn(bank, spl, nb=1)
                rows.append(summarize(f"(b) b3-oracle-type {pt}", fn_t, spl))
            fn_f, pf = oracle_fill_fn(bank, spl, nb=3)
            rows.append(summarize(f"(b+) b3-oracle-type x fill {pf}", fn_f, spl))
            rows.append(summarize("(b+plan) b3-oracle-type x fill on the plan", plan_aware(fn_f), spl, PLAN_H))
            if label == "pend":
                def gen_fn(es):
                    out = []
                    for e in es:
                        sys.path.insert(0, str(REPO / "world_model/v2/data")); import contents as Cn
                        h = float(np.clip(e["meta"]["fill_h"], 0.02, 0.07))
                        phs = np.degrees(np.arctan((Cn.CUP_HEIGHT - h) / Cn.CUP_INNER_R))
                        out.append(e["pend_phi"] - phs)
                    return out
                rows.append(summarize("(p*) pend-generator, true h", gen_fn, spl))
                rows.append(summarize("(p*+plan) pend-generator on plan, true h", plan_aware(gen_fn), spl, PLAN_H))
                if "h0" not in pend_cache:
                    h0 = float(np.median([e["meta"]["fill_h"] for e in spl_all["train"]]))
                    log(f"[gate] pendulum with contents-blind fill h0={h0:.4f} (train median) on {len(E)} eps")
                    pend_cache["h0"] = (h0, pend_phys(E, lambda e: h0, a.workers))
                h0, tab = pend_cache["h0"]
                rows.append(summarize(f"(p0) pendulum, blind h0={h0:.3f}", lambda es: [tab[e["id"]] for e in es], spl))
                rows.append(summarize(f"(p0+plan) pendulum on plan, blind h0", plan_aware(lambda es: [tab[e["id"]] for e in es]), spl, PLAN_H))
            if label in gru_models:
                rows.append(summarize("(c) proprio-GRU (all contents)", gru_models[label], spl, (0.25, 0.5, 1.0)))
            results[key] = rows
            log(f"\n== {key}: test eps={rows[0]['n_test']} (pos {rows[0]['n_pos_test']})")
            log(f"  {'predictor':60s} {'epA_pre':>7s} {'epA_full':>8s} {'epA_vt [95%CI]':>22s} {'A@.25':>6s} {'A@.5':>6s} {'A@1':>6s}"
                f" {'det':>5s} {'lead':>5s} {'det>=.5':>7s} {'FA/m':>5s}")
            for r in rows:
                ci = r["epA_pre_valtest_ci"]
                log(f"  {r['name'][:60]:60s} {r['epA_pre']:7.3f} {r['epA_full']:8.3f} {r['epA_pre_valtest']:6.3f} [{ci[0]:.3f},{ci[1]:.3f}]"
                    f" {r['A025']:6.3f} {r['A05']:6.3f} {r['A1']:6.3f} {r['det']:5.2f} {r['lead_med']:5.2f} {r['det05']:7.2f} {r['fa_test']:5.2f}")
    if a.out:
        Path(a.out).write_text(json.dumps(results, indent=1, default=float))
    log(f"[gate] done in {time.time()-t0:.0f}s")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--gru-steps", type=int, default=1500)
    ap.add_argument("--device", default="cpu")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
