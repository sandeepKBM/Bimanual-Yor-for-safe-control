"""synthetic VLM prior generator + split-conformal quantile (no simulation)."""
import math

import numpy as np

from world_model.v2.tray_rep.sets.vlmprior import conformal as CF
from world_model.v2.tray_rep.sets.vlmprior import prior as PR


def test_deterministic_and_seed_dependent():
    a = PR.draw_prior(-6.0, 100123, 0.2, 0)
    assert a == PR.draw_prior(-6.0, 100123, 0.2, 0)
    assert a != PR.draw_prior(-6.0, 100123, 0.2, 1)
    assert a != PR.draw_prior(-6.0, 100124, 0.2, 0)


def test_rho_share_and_sigmas():
    seeds = np.arange(100000, 104000)
    for rho in (0.0, 0.1, 0.2, 0.4, 1.0):
        m, sg, cf = PR.draw_many(np.full(len(seeds), -6.0), seeds, rho, 0)
        assert abs(cf.mean() - rho) < 0.02, (rho, cf.mean())
        assert np.all(sg[cf] == PR.SIGMA_CONF) and np.all(sg[~cf] == PR.SIGMA_HONEST)


def test_confident_wrong_error_range_and_sign_balance():
    seeds = np.arange(5000)
    m, sg, cf = PR.draw_many(np.full(len(seeds), -6.0), seeds, 1.0, 0)
    err = m - (-6.0)
    assert cf.all() and np.all(np.abs(err) >= 1.0 - 1e-12) and np.all(np.abs(err) <= 2.0 + 1e-12)
    assert abs((err > 0).mean() - 0.5) < 0.03
    assert abs(np.abs(err).mean() - 1.5) < 0.03


def test_honest_error_is_scaled_student_t3():
    seeds = np.arange(20000)
    m, sg, cf = PR.draw_many(np.zeros(len(seeds)), seeds, 0.0, 0)
    assert not cf.any()
    e = m / 0.30
    # |T3| quantiles: P(|T|<1.638)=0.8, P(|T|<2.353)=0.9 (3 dof); median 0.765
    assert abs(np.quantile(np.abs(e), 0.8) - 1.638) < 0.08
    assert abs(np.quantile(np.abs(e), 0.9) - 2.353) < 0.15
    assert abs(np.median(np.abs(e)) - 0.765) < 0.04


def test_common_random_numbers_across_rho():
    seeds = np.arange(2000)
    l = np.full(len(seeds), -6.0)
    m1, _, c1 = PR.draw_many(l, seeds, 0.1, 0)
    m2, _, c2 = PR.draw_many(l, seeds, 0.2, 0)
    assert np.all(c2[c1])                                  # wrong at 0.1 => wrong at 0.2
    same = ~c2
    assert np.array_equal(m1[same], m2[same])             # honest draws do not change with rho


def test_conformal_rank_formula():
    s = np.arange(1, 10, dtype=float)                      # n = 9, alpha 0.1: rank ceil(10 * 0.9) = 9 -> the max
    assert CF.conformal_q(s, 0.1) == 9.0
    assert CF.conformal_q(np.arange(1, 6.0), 0.1) == float("inf")        # n = 5: rank 6 > n
    n = 99
    s = np.random.default_rng(0).random(n)
    assert CF.conformal_q(s, 0.1) == np.sort(s)[math.ceil((n + 1) * 0.9) - 1]


def test_conformal_marginal_coverage_on_synthetic_data():
    rng = np.random.default_rng(3)
    n_cal, n_rep, hits = 600, 400, []
    for _ in range(n_rep):
        sc = np.abs(rng.standard_t(3, n_cal)) * 0.75
        q = CF.conformal_q(sc, 0.1)
        new = np.abs(rng.standard_t(3, 50)) * 0.75
        hits.append((new <= q).mean())
    assert 0.89 <= np.mean(hits) <= 0.925                  # >= 0.9 in expectation, tight for n = 600


def test_conformal_degenerates_at_high_rho_on_val_seeds():
    """the finding: with a few percent confidently-wrong draws the 90% score quantile jumps into the wrong draws, so the prior interval
    becomes wider than the population range (2.485) for honest-sigma draws."""
    seeds = np.arange(100000, 100600)
    l = np.log(np.exp(np.random.default_rng(0).uniform(np.log(5e-4), np.log(6e-3), len(seeds))))
    qs = {rho: CF.conformal_q(CF.val_scores(l, seeds, rho)[0], 0.1) for rho in (0.0, 0.1, 0.2, 0.4)}
    assert qs[0.0] < 2.5 and qs[0.1] > 4.0 and qs[0.4] > qs[0.2] > qs[0.1] > qs[0.0]
    assert 2 * qs[0.2] * 0.40 > np.log(6e-3 / 5e-4)
