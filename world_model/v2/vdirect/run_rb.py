"""vdirect/run_rb.py -- one cell of PREREG_ADDENDUM_RB (arms R / Rc / Rb). SIMULATION / SAVED DATA ONLY
(no hardware, no network). Outputs under $VD_ROOT/runs/<arm>_b<budget>_s<seed>.

  --arm R   R's original recipe (train_kinwm.make_branches roots + actions), unchanged
  --arm Rc  R's action law from roots drawn uniformly from the eligible candidate set (root-law control)
  --arm Rb  R's action law from roots in the lower half of the nominal value's accepted set (see rb.py)

Every arm: fresh Trainer(seed), all branches added at step 0, S_TOTAL steps, run_arm.finalize (unchanged), _DONE.
Rb's nominal value = $VD_SRC_ROOT/runs/nom_s<seed>_m0 (made by run_arm.py --arm NOM, unchanged). Rb computes its
roots BEFORE the Trainer is built and loads the nominal WM without Trainer.load_ckpt, so the global torch RNG at the
start of training is the same as R's (Trainer.__init__ calls torch.manual_seed(seed)).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["R", "Rc", "Rb"])
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import torch
    from world_model.v2.latent_hj import kin_wm
    from world_model.v2.latent_hj import reach_value as rv
    from world_model.v2.vdirect import branches as BRS
    from world_model.v2.vdirect import rb as RB
    from world_model.v2.vdirect import run_arm as RA
    from world_model.v2.vdirect.trainer import Trainer, dump_json, train_data
    cfg = dict(RA.CFG) | (RA.SMOKE if a.smoke else {})
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    t_start = time.time()
    RA.log(f"[run_rb] arm={a.arm} b={a.budget} s={a.seed} smoke={a.smoke} dev={dev} "
           f"cuda_visible={os.environ.get('CUDA_VISIBLE_DEVICES')} root={RA.ROOT} src={RA.SRC}")
    norm = RA.get_norm(dev)
    tr = train_data(dev)[0]
    sm = "smoke_" if a.smoke else ""
    out = Path(a.out) if a.out else RA.ROOT / "runs" / f"{sm}{a.arm}_b{a.budget}_s{a.seed}"
    out.mkdir(parents=True, exist_ok=True)
    roots, meta = None, {}
    t0 = time.time()
    if a.arm in ("Rc", "Rb"):
        V = None
        if a.arm == "Rb":
            nd = RA.SRC / "runs" / f"{sm}nom_s{a.seed}_m0"
            ck = torch.load(nd / "ckpt.pt", map_location=dev, weights_only=False)
            wm0 = kin_wm.build_model().to(dev); wm0.load_state_dict(ck["wm"]); wm0.eval()
            net, _ = rv.load(str(nd / "probe" / "value.pt"), dev)
            for p in list(wm0.parameters()) + list(net.parameters()):
                p.requires_grad_(False)
            cand = BRS.candidates(tr, cfg["N_CAND"], RA.cand_seed(a.seed, 0))
            V = RB.nominal_V(tr, cand, wm0, net, dev)
            del wm0, net, ck
        roots, _, meta = RB.pick_roots(tr, a.budget, RA.cand_seed(a.seed, 0), cfg["N_CAND"], V=V)
        np.save(out / "roots.npy", roots)
    T = Trainer(a.seed, norm, total_steps=cfg["S_TOTAL"], dev=dev)
    if a.arm == "R":
        BR, m2 = BRS.random_branches(tr, a.budget, dev, seed=a.seed + 1)
    else:
        BR, m2 = RB.band_branches(tr, roots, dev, seed=a.seed + 1)
    meta.update(m2, kind=a.arm, round=0, gen_sec=time.time() - t0, wm_step=0)
    RA.log(f"[branches {a.arm}] {meta}")
    T.add_branches(BR)
    T.train(cfg["S_TOTAL"], tag=f" {a.arm} b{a.budget} s{a.seed}")
    wm = T.frozen_wm()
    T.save_wm(out / "wm.pt")
    dump_json(T.curve, out / "wm_curve.json")
    dump_json([meta], out / "rounds.json")
    res = RA.finalize(wm, dev, out, a.seed, cfg, rounds_meta=[meta], t_start=t_start)
    RA.log("[result] " + json.dumps({k: v["cl_spill_savable"] for k, v in res["values"].items()}))
    (out / "_DONE").write_text(time.strftime("%FT%TZ", time.gmtime()))


if __name__ == "__main__":
    main()
