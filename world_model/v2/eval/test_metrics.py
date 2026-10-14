"""Unit tests for the prediction-framed harness on SYNTHETIC sequences with known
lead times.   python -m pytest -q world_model/v2/eval/test_metrics.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.eval import metrics as M  # noqa: E402
from world_model.v2.eval import harness as Hn  # noqa: E402
from world_model.v2.eval.predictors import OraclePredictor, PostOnsetDetector, ConstantPredictor  # noqa: E402

HZ = 15.0


def make_eps(n_pos=30, n_neg=30, T=150, t_start=5, seed=0):
    rng = np.random.default_rng(seed); eps = []
    for i in range(n_pos + n_neg):
        c = np.zeros(T, int)
        if i < n_pos:
            on = int(rng.integers(60, 140)); c[on:] = 1 + rng.integers(0, 7)
        eps.append(dict(id=f"e{i}", hz=HZ, spill_count=c, t_start=t_start, meta=dict(light=float(i % 5))))
    return eps


class RampPredictor:
    """Score jumps to 1 exactly `lead` steps before onset (noise elsewhere)."""
    probabilistic = False
    horizons_s = None

    def __init__(self, lead, noise=0.05, seed=1):
        self.lead = lead; self.rng = np.random.default_rng(seed); self.name = f"ramp{lead}"

    def predict(self, ep):
        T = M.episode_len(ep); on = M.episode_onset(ep)
        s = self.rng.uniform(0, 0.3, T)
        if on >= 0:
            s[max(0, on - self.lead):] = 1.0
        return s


# ------------------------------------------------------------------ labels
def test_labels_exclude_post_onset():
    y, v = M.labels(onset=50, T=100, t_start=5, H_steps=10)
    assert not v[:5].any() and v[5:50].all() and not v[50:].any()
    assert y.sum() == 10 and y[40:50].all() and not y[:40].any()
    y, v = M.labels(onset=-1, T=100, t_start=0, H_steps=10)
    assert v.all() and not y.any()


def test_onset_baseline_rise():
    c = np.r_[np.full(10, 3), np.full(10, 3), np.full(5, 4)]      # settled baseline 3 at t_start
    assert M.onset_from_counts(c, t_start=10) == 20
    assert M.onset_from_counts(np.zeros(10), 0) == -1


# ------------------------------------------------------------------ ranking
def test_auroc_auprc_against_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(20):
        s = rng.integers(0, 5, 40).astype(float); y = rng.random(40) < 0.3
        if y.all() or not y.any():
            continue
        pos, neg = s[y], s[~y]
        brute = np.mean([(p > q) + 0.5 * (p == q) for p, q in itertools.product(pos, neg)])
        assert abs(M.auroc(s, y) - brute) < 1e-9
        # AP brute force over distinct thresholds
        ths = np.unique(s)[::-1]; ap = 0.0; prev_r = 0.0
        for t in ths:
            pr = y[s >= t].mean(); r = y[s >= t].sum() / y.sum(); ap += pr * (r - prev_r); prev_r = r
        assert abs(M.auprc(s, y) - ap) < 1e-9


def test_oracle_is_perfect_at_every_horizon():
    eps = make_eps()
    rep = Hn.evaluate(OraclePredictor(), eps[::2], {"test": eps[1::2]})
    for h, d in rep["sets"]["test"]["per_horizon"].items():
        assert d["auroc_pre_onset"] == pytest.approx(1.0) and d["auprc_pre_onset"] == pytest.approx(1.0)


# ------------------------------------------------------------------ events / leads
@pytest.mark.parametrize("lead", [1, 4, 8, 15])
def test_known_lead_time_recovered(lead):
    eps = make_eps(seed=lead)
    p = RampPredictor(lead)
    preds = [p.predict(e) for e in eps]
    st = M.event_stats(preds, eps, thr=0.5, credit_window_s=2.0)
    s = st.summary(tick_s=1 / HZ)
    assert s["detection_rate"] == 1.0
    assert np.allclose(st.leads_s, lead / HZ)
    assert s["n_fa"] == 0 and s["fa_per_min"] == 0.0


def test_detector_scores_as_chance_not_prediction():
    eps = make_eps()
    rep = Hn.evaluate(PostOnsetDetector(), eps[::2], {"test": eps[1::2]})
    t = rep["sets"]["test"]
    for d in t["per_horizon"].values():
        assert abs(d["auroc_pre_onset"] - 0.5) < 0.1          # prediction: chance
        assert d["detect_auroc_INCLUDES_POST_ONSET"] > 0.95    # old framing: "perfect"
    assert t["event"]["detection_rate"] == 0.0


def test_false_alarm_rate_exact():
    T, t0 = 150, 0
    eps = [dict(hz=HZ, spill_count=np.zeros(T, int), t_start=t0) for _ in range(6)]
    preds = []
    for k in range(6):                       # k bumps of 2 steps each in episode k
        s = np.zeros(T)
        for b in range(k):
            s[10 + 20 * b: 12 + 20 * b] = 1.0
        preds.append(s)
    st = M.event_stats(preds, eps, 0.5)
    calm_min = 6 * T / HZ / 60.0
    assert st.n_fa == sum(range(6))
    assert st.fa_per_min == pytest.approx(15 / calm_min)
    assert st.calm_alarm_steps == 2 * 15
    assert st.neg_eps_with_fa == 5


def test_alarm_inside_credit_window_is_not_a_false_alarm():
    T = 150; on = 100
    ep = dict(hz=HZ, spill_count=np.r_[np.zeros(on), np.ones(T - on)], t_start=0)
    s = np.zeros(T); s[95:] = 1.0                           # 5 steps before onset
    st = M.event_stats([s], [ep], 0.5, credit_window_s=2.0)
    assert st.n_fa == 0 and st.n_detected == 1 and st.leads_s[0] == pytest.approx(5 / HZ)
    s2 = np.zeros(T); s2[20:] = 1.0                         # latched 80 steps early (> 2 s window)
    st2 = M.event_stats([s2], [ep], 0.5, credit_window_s=2.0)
    assert st2.n_fa == 1                                     # the early latch is a false alarm
    assert st2.leads_s[0] == pytest.approx(30 / HZ)          # credited only up to the window


def test_threshold_selection_meets_budget_on_val():
    eps = make_eps(n_pos=40, n_neg=40, seed=3)
    p = RampPredictor(6, seed=4)
    preds = [p.predict(e) for e in eps]
    thr = M.select_threshold(preds, eps, fa_per_min=1.0)
    st = M.event_stats(preds, eps, thr)
    assert st.fa_per_min <= 1.0
    assert st.summary()["detection_rate"] == 1.0            # ramp is separable
    thr0 = M.select_threshold(preds, eps, fa_per_min=0.0)
    assert M.event_stats(preds, eps, thr0).n_fa == 0


def test_constant_predictor_never_detects():
    eps = make_eps()
    rep = Hn.evaluate(ConstantPredictor(), eps[::2], {"test": eps[1::2]})
    assert rep["sets"]["test"]["event"]["detection_rate"] == 0.0
    for d in rep["sets"]["test"]["per_horizon"].values():
        assert d["auroc_pre_onset"] == pytest.approx(0.5)


# ------------------------------------------------------------------ calibration
def test_calibration_ece():
    rng = np.random.default_rng(0)
    p = rng.random(200000)
    good = M.calibration(p, rng.random(p.size) < p)
    bad = M.calibration(p, rng.random(p.size) < p ** 4)
    assert good["ece"] < 0.01 and bad["ece"] > 0.2
    assert good["brier"] < bad["brier"]


# ------------------------------------------------------------------ OOD + multi-horizon plumbing
def test_ood_split_and_threshold_frozen():
    eps = make_eps(seed=9)
    ind, ood = Hn.split_by_meta_quantile(eps, "light", q=0.6)
    assert len(ind) and len(ood) and all(e["meta"]["light"] > 2 for e in ood)
    rep = Hn.evaluate(RampPredictor(5), ind[::2], {"test": ind[1::2], "ood": ood})
    assert set(rep["sets"]) == {"val", "test", "ood"}
    assert rep["threshold_chosen_on"] == "val"
    assert rep["sets"]["ood"]["event"]["thr"] == rep["threshold"]
    txt = Hn.format_report(rep)
    assert "pre-onset AUROC" in txt and "lead med" in txt


def test_multi_horizon_column_alignment():
    class TwoCol:
        probabilistic = True; horizons_s = (0.5, 0.25); name = "twocol"   # deliberately permuted

        def predict(self, ep):
            o = OraclePredictor((0.5, 0.25)).predict(ep); return o
    eps = make_eps(seed=2)
    rep = Hn.evaluate(TwoCol(), eps[::2], {"test": eps[1::2]}, horizons_s=(0.25, 0.5))
    for d in rep["sets"]["test"]["per_horizon"].values():
        assert d["auroc_pre_onset"] == pytest.approx(1.0)
        assert d["calibration"]["ece"] == pytest.approx(0.0, abs=1e-9)


def test_chance_detection_flags_random_alarms():
    """Spill-independent random alarms: detection ~= chance, lift ~= 1."""
    eps = make_eps(n_pos=200, n_neg=200, T=300, seed=11)
    rng = np.random.default_rng(5)
    preds = [(rng.random(M.episode_len(e)) < 0.02).astype(float) for e in eps]
    s = M.event_stats(preds, eps, 0.5).summary()
    assert abs(s["detection_rate"] - s["chance_detection_rate"]) < 0.12
    assert 0.6 < s["alarm_lift_window_vs_calm"] < 1.6
    p = RampPredictor(6)
    s2 = M.event_stats([p.predict(e) for e in eps], eps, 0.5).summary()
    assert s2["detection_rate"] == 1.0 and s2["chance_detection_rate"] == 0.0


# ------------------------------------------------------------------ conformal hooks
def test_calm_conformal_coverage_and_aci_tracks_shift():
    from world_model.v2.eval.conformal import CalmConformalThreshold, AdaptiveThreshold, calm_scores
    rng = np.random.default_rng(0)
    cal = make_eps(n_pos=50, n_neg=50, seed=21); new = make_eps(n_pos=50, n_neg=50, seed=22)
    sc = lambda eps: [rng.normal(0, 1, M.episode_len(e)) for e in eps]
    thr = CalmConformalThreshold(alpha=0.02).fit(sc(cal), cal)
    fresh = calm_scores(sc(new), new)
    assert abs((fresh >= thr).mean() - 0.02) < 0.01
    # distribution shift: calm scores +1.0 -> fixed threshold over-alarms, ACI recovers
    aci = AdaptiveThreshold(calm_scores(sc(cal), cal), alpha=0.02, gamma=0.01, confirm_s=1.0)
    stream = rng.normal(1.0, 1, 30000)
    alarms = np.array([s >= aci.update(s) for s in stream])
    assert (stream[-5000:] >= thr).mean() > 0.08            # static threshold broken by the shift
    assert abs(alarms[-5000:].mean() - 0.02) < 0.015         # ACI back near target
