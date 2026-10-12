"""wm/prep.py -- CPU preparation steps run BEFORE the GPU arms. SIMULATION / SAVED DATA ONLY.

  norm      the fixed reference normalisation of a dataset (n_ref law-R branches, seed 12345), cached in <data>/norm/
  branches  the branch files <data>/branches/<law>_b<budget>_s<seed>.npz for a grid of laws x budgets x seeds
  refs      hold-still and oracle-policy closed-loop references on the certified states, cached beside the oracle file
  link_oracle  copy the oracle labels of a dataset with IDENTICAL episodes (e.g. data_piper_full -> a pc_est dataset),
            after checking that the TEST episodes match (seed, length, mu_roll). --src defaults to the dataset's `same_episodes_as`.

  python -m world_model.v2.tray_rep.wm.prep norm     --data D --workers 16
  python -m world_model.v2.tray_rep.wm.prep branches --data D --laws R,Rc --budgets 300,1000,3000 --seeds 0,1,2 --workers 16
  python -m world_model.v2.tray_rep.wm.prep refs     --data D --oracle D/oracle/test_v0.npz
  python -m world_model.v2.tray_rep.wm.prep link_oracle --data D_pc [--src D_full]
Every step skips what already exists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.wm import common as C
from world_model.v2.tray_rep.wm import run_arm as RA


def link_oracle(data, src=None, allow_subset=False, log=print):
    """Copy <src>/oracle/test_v0.{npz,json} to <data>/oracle/. The oracle labels (privileged truth: savable / hold_ok /
    uncertified at test states (ep, t)) depend only on the episodes, so they carry over to a dataset that logs the same
    episodes with a different observation (pc_est). Checked before copying, on the TEST split: same episode count (or, with
    allow_subset, a prefix: a small smoke dataset whose test split is the first episodes), same per-episode seed, length and
    mu_roll, and every oracle tick inside its episode and on the snapshot grid. Returns the oracle path."""
    data = Path(data)
    meta = json.loads((data / "meta.json").read_text())
    if src is None:
        same = meta.get("same_episodes_as")
        assert same, "meta.json has no same_episodes_as: pass --src"
        from world_model.v2.tray_rep import common as K
        src = K.ROOT / same                      # e.g. data_piper_full under /common/users/ss5772/bimanual_data/tray_rep
    src = Path(src)
    assert (src / "oracle" / "test_v0.npz").exists(), f"no oracle at {src}/oracle/test_v0.npz"
    f_s, f_d = np.load(src / "test.npz"), np.load(data / "test.npz")
    ns, nd = len(f_s["lens"]), len(f_d["lens"])
    n = min(ns, nd)
    assert ns == nd or (allow_subset and nd < ns), f"test split has {nd} episodes here, {ns} in {src} (use --allow-subset for a prefix)"
    for k in ("seed", "lens", "onset"):
        assert np.array_equal(f_s[k][:n], f_d[k][:n]), f"test-split '{k}' differs between {src} and {data}: episodes are NOT identical"
    # mu_roll is recomputed per generator and can differ by one float rounding step (owner, 2026-10-05: max 8.7e-19 abs,
    # 2e-16 rel between data_piper_full and the pc dataset, with identical seeds, lengths and failure ticks)
    assert np.allclose(f_s["hid"][:n], f_d["hid"][:n], rtol=1e-12, atol=0), \
        f"test-split 'hid' differs beyond float rounding between {src} and {data}: episodes are NOT identical"
    o = dict(np.load(src / "oracle" / "test_v0.npz"))
    keep = o["ep"] < nd
    if not keep.all():
        o = {k: v[keep] for k, v in o.items()}
    lens = f_d["lens"]
    assert (o["t"] < lens[o["ep"]]).all() and (o["t"] >= meta["snap_min_tick"]).all() and \
        ((o["t"] - meta["snap_min_tick"]) % meta["snap_stride"] == 0).all(), "oracle ticks outside the episodes / snapshot grid"
    assert np.allclose(o["hid"], f_d["hid"][o["ep"]], rtol=1e-12, atol=0), "oracle mu_roll != dataset mu_roll"
    out = data / "oracle"
    out.mkdir(exist_ok=True)
    if keep.all():
        shutil.copy(src / "oracle" / "test_v0.npz", out / "test_v0.npz")
    else:
        np.savez(out / "test_v0.npz", **o)
    j = json.loads((src / "oracle" / "test_v0.json").read_text()) if (src / "oracle" / "test_v0.json").exists() else {}
    j.update(linked_from=str(src), n_states_linked=int(len(o["ep"])), n_test_eps=int(nd),
             counts={str(k): int((o["label"] == k).sum()) for k in np.unique(o["label"])})
    (out / "test_v0.json").write_text(json.dumps(j, indent=1))
    log(f"[link_oracle] {src.name} -> {data.name}: {len(o['ep'])} states, test eps {nd}, label counts {j['counts']}")
    return out / "test_v0.npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["norm", "branches", "refs", "link_oracle"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n-ref", type=int, default=C.N_REF)
    ap.add_argument("--laws", default="R,Rc")
    ap.add_argument("--budgets", default="300,1000,3000")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--oracle", default="")
    ap.add_argument("--cl-T", type=int, default=RA.CL_T)
    ap.add_argument("--cl-max", type=int, default=None)
    ap.add_argument("--src", default="", help="link_oracle: dataset to copy the oracle from (default: meta same_episodes_as)")
    ap.add_argument("--allow-subset", action="store_true", help="link_oracle: this dataset's test split is a prefix (smoke only)")
    a = ap.parse_args()
    t0 = time.time()
    if a.cmd == "link_oracle":
        link_oracle(a.data, a.src or None, a.allow_subset, RA.log)
    elif a.cmd == "norm":
        n = C.get_norm(a.data, a.n_ref, C.REF_SEED, a.workers, RA.log)
        RA.log(f"[prep norm] d_obs={n['d_obs']} ref branch fail {n['ref_branch_fail_frac']:.3f}")
    elif a.cmd == "branches":
        for law in a.laws.split(","):
            for b in (int(x) for x in a.budgets.split(",")):
                for s in (int(x) for x in a.seeds.split(",")):
                    RA.get_branches(a.data, law, b, s, a.workers, "cpu")
    else:
        o = Path(a.oracle) if a.oracle else Path(a.data) / "oracle" / "test_v0.npz"
        r = RA.cl_references(a.data, o, a.cl_T, a.cl_max)
        RA.log("[prep refs] " + str({k: (v["fail_rate"], v["n"]) for k, v in r.items()}))
    RA.log(f"[prep {a.cmd}] done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
