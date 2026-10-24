"""fit_calib freshness: stale / frozen blob readings must not enter the per-side affine / bias fit (default); --calib-legacy reproduces the v3 fit.
SIMULATION ONLY, synthetic records."""
import numpy as np

from world_model.v2.house.ocs import exp_e234 as E


def _row(raw, uv, fresh, vis, side="right"):
    return dict(raw={side: list(raw)}, uv={side: list(uv)}, truth_tray=[0.0, 0.0, 0.0], visible=vis, blob_vis=vis, blob_sides=[side],
                blob_fresh={side: fresh}, e_tilt_rel=0.01)


def _records(seed=0, n_fresh=200, n_stale=600):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n_fresh):          # fresh: constant 5 mm bias + 3 mm noise, pixel varies a little
        uv = (80 + rng.normal() * 10, 45 + rng.normal() * 5)
        rows.append(_row([0.005 + 0.003 * rng.normal(), 0.003 * rng.normal(), 0.0], uv, True, True))
    for _ in range(n_stale):          # stale: the reading frozen 12 cm off (cup left the view long ago), pixel frozen too
        rows.append(_row([0.12, -0.09, 0.0], (150.0, 20.0), False, False))
    rng.shuffle(rows)
    return [dict(e2=dict(base=[], tray=[], cup=rows, person=[]))]


def _old_fit(cup):
    """verbatim v3 logic for the bias / affine (all rows with a reading)."""
    out = {}
    for s_ in ("right",):
        rows = [(np.array([1.0, r["uv"][s_][0], r["uv"][s_][1]]), (np.array(r["raw"][s_]) - np.array(r["truth_tray"]))[:2]) for r in cup if s_ in r["raw"] and s_ in r.get("uv", {})]
        X = np.array([a for a, _ in rows]); Y = np.array([b for _, b in rows])
        W = np.linalg.solve(X.T @ X + 1e-3 * len(X) * np.diag([0.0, 1.0, 1.0]), X.T @ Y)
        d = [np.array(r["raw"][s_]) - np.array(r["truth_tray"]) for r in cup if s_ in r["raw"]]
        out = dict(aff=W.T.tolist(), bias=np.median(d, 0).tolist())
    return out


def test_fixed_fit_ignores_stale_rows():
    res = _records()
    fresh_only = [dict(e2=dict(base=[], tray=[], cup=[r for r in res[0]["e2"]["cup"] if r["blob_fresh"]["right"]], person=[]))]
    c = E.fit_calib(res)                       # default = fixed
    c_ref = E.fit_calib(fresh_only)            # the same data with the stale rows physically removed
    for k in ("cup_affine", "cup_bias"):
        assert np.allclose(np.ravel(list(c.scalars[k]["right"])), np.ravel(list(c_ref.scalars[k]["right"])), atol=1e-12)
    assert abs(c.scalars["cup_bias"]["right"][0] - 0.005) < 2e-3          # the fresh bias, not the 12 cm frozen reading
    assert c.scalars["cup_pos"] < 0.02                                    # visible radius is the fresh scatter (~mm), not inflated by the frozen rows
    # the stale rows are still the occluded (held-estimate) population: their error is the frozen 15 cm offset, reported separately
    assert c.scalars["cup_pos_occ"] > 0.05


def test_legacy_reproduces_v3_fit():
    res = _records()
    c = E.fit_calib(res, legacy=True)
    old = _old_fit(res[0]["e2"]["cup"])
    assert np.allclose(np.asarray(c.scalars["cup_affine"]["right"]), np.asarray(old["aff"]), atol=1e-12)
    assert np.allclose(np.asarray(c.scalars["cup_bias"]["right"])[:2], np.asarray(old["bias"])[:2], atol=1e-12)
    # and it is visibly biased by the stale rows (the bug)
    assert abs(c.scalars["cup_bias"]["right"][0] - 0.005) > 0.05
    cf = E.fit_calib(res)                       # the legacy affine "explains" the frozen reading at its frozen pixel; the fixed one does not
    at_stale = lambda cc: np.asarray(cc.scalars["cup_affine"]["right"]) @ np.array([1.0, 150.0, 20.0])
    assert np.linalg.norm(at_stale(c) - np.array([0.12, -0.09])) < 0.02 < np.linalg.norm(at_stale(cf) - np.array([0.12, -0.09]))


def test_records_without_freshness_fall_back_to_legacy():
    res = _records()
    for r in res[0]["e2"]["cup"]:
        r.pop("blob_fresh"); r.pop("blob_sides")
    a, b = E.fit_calib(res), E.fit_calib(res, legacy=True)
    assert a.scalars["cup_bias"] == b.scalars["cup_bias"] and a.scalars["cup_affine"] == b.scalars["cup_affine"]
