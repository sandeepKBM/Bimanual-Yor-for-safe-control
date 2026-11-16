"""robust/test3.py -- TEST 3: the latency sweep. SIMULATION / SAVED DATA ONLY (MuJoCo CPU + torch CPU; no hardware, no network).

Inject latency k in {0, 1, 2, 3, 5} ticks and run the F3 filter (plain, at the bridge's chosen tau) and, if tuned, the robust-margin
filter (rob, at the thr matched on VAL for the base condition) on the TEST carries. The thresholds are FIXED across the sweep: this
asks what a filter tuned for one latency does when the real latency is different.
  piper     models trained on exact obs : condition  lat<k>     (latency only, no noise, 50 Hz, no dropout)
  piper_trk models trained on the tracker : condition trklat<k>  (3 mm, 30 Hz, 2% dropout, latency k; the models saw k = 3)
The drop rate against k is then compared with the DERIVED budget (budget.py): lead minus the fallback's recovery time.

  python -m world_model.v2.tray_rep.robust.test3 run --setup piper_trk --arm Rc3k --seed 0 --n-test 600 --workers 2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from world_model.v2.tray_rep import common as K          # noqa: F401
from world_model.v2.tray_rep.robust import rfilters as RF
from world_model.v2.tray_rep.robust import run2
from world_model.v2.tray_rep.robust import tune
from world_model.v2.tray_rep.robust import util as RU

LATS = (0, 1, 2, 3, 5)


def cond_for(setup, k):
    return f"trklat{k}" if setup == "piper_trk" else f"lat{k}"


def specs_for(setup, arm, seed, base_cond="trk", tag="main", out_root=RU.OUT):
    """plain at the bridge tau; rob at the matched thr of the base condition if its chosen file exists (else thr = tau, scale 0.4)."""
    tau = RU.chosen_tau(setup, arm)
    out = dict(plain=RF.spec_str("plain", arm, tau))
    cp = tune.chosen_path(out_root, tag, setup, arm, seed, base_cond)
    if cp.exists() and "rob" in json.loads(cp.read_text()):
        out["rob"] = json.loads(cp.read_text())["rob"]["spec"]
    else:
        out["rob"] = RF.spec_str("rob", arm, tau, scale=0.4)
    return out


def run(setup, arm, seed, lats=LATS, kinds=("plain", "rob"), n_test=None, workers=1, tag="main", out_root=RU.OUT, split="test",
        base_cond="trk", base_tag=None):
    sp = specs_for(setup, arm, seed, base_cond, base_tag or tag, out_root)
    res = {}
    for k in lats:
        for kind in kinds:
            summ, _ = run2.run(sp[kind], setup, seed, cond_for(setup, k), split, n_test, workers, f"{tag}_lat", out_root)
            res[f"{kind}@lat{k}"] = dict(drop=summ["drop"], interv=summ["interv_rate"])
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--setup", required=True, choices=RU.SETUPS)
    ap.add_argument("--arm", default="Rc3k")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lats", default=",".join(str(x) for x in LATS))
    ap.add_argument("--kinds", default="plain,rob")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="main")
    ap.add_argument("--base-tag", default="")
    ap.add_argument("--out", default=str(RU.OUT))
    a = ap.parse_args()
    res = run(a.setup, a.arm, a.seed, tuple(int(x) for x in a.lats.split(",")), tuple(a.kinds.split(",")), a.n_test, a.workers,
              a.tag, a.out, a.split, base_tag=a.base_tag or None)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
