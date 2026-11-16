"""vdirect/rb_selftest.py -- equivalence check for rb.make_branches_at (PREREG_ADDENDUM_RB). CPU, SAVED DATA ONLY.
Draws make_branches' own roots with make_branches' own rng consumption, then requires bitwise-equal tensors."""
import numpy as np


def main(n=64, seed=7):
    import torch
    from world_model.v2.vdirect import branches as BRS
    from world_model.v2.vdirect import rb as RB
    from world_model.v2.vdirect.trainer import train_data
    from world_model.v2.wm_audit.train_kinwm import make_branches
    tr = train_data("cpu")[0]
    W, Lb = BRS.W_WIN, BRS.LB
    ref = make_branches(tr, n, Lb, W, "cpu", seed=seed)
    rng = np.random.default_rng(seed)
    ok = np.flatnonzero(tr["lens"] > W - Lb + 5)
    e = rng.choice(ok, n)
    t0 = np.array([rng.integers(W - Lb, tr["lens"][i] - 1) for i in e])
    got = RB.make_branches_at(tr, e, t0, Lb, W, "cpu", rng)
    same = {k: bool(torch.equal(ref[k], got[k])) for k in ref}
    print("[rb_selftest]", same, flush=True)
    assert all(same.values()), same
    print("[rb_selftest] PASS", flush=True)


if __name__ == "__main__":
    main()
