"""world_model/v2/eval/run_eval.py -- evaluate any predictor with the v2 harness.

  python -m world_model.v2.eval.run_eval --prep $PREP/ds \
      --model gru:$RUN/gru/model.pt --model slosh:b2 --model dino:$RUN/dino/model.pt \
      [--ood name=$PREP/other_ds[:split]] [--fa-per-min 1.0] [--out report.json]

Model specs: clock (time-only script-phase baseline, fit on train) | gru:<ckpt> | dino:<ckpt> | rssm:<ckpt> | slosh:<a0|a1|b1|b2|b3> |
             oracle | detector | constant
Threshold is chosen on <prep>'s VAL split and frozen for test + every --ood set.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from world_model.v2.eval import harness as Hn  # noqa: E402
from world_model.v2.eval import predictors as P  # noqa: E402
from world_model.v2.model import data as D  # noqa: E402


def build(spec: str, device: str, prep=None):
    kind, _, arg = spec.partition(":")
    if kind == "clock":
        return P.ClockBaseline(D.load_prepared(prep, "train", with_images=False)), False
    if kind == "gru":
        from world_model.v2.model.proprio_gru import GRUPredictor
        return GRUPredictor(arg, device), False
    if kind == "dino":
        from world_model.v2.model.wm_dino import DinoWMPredictor
        return DinoWMPredictor(arg, device), True
    if kind == "rssm":
        from world_model.v2.model.wm_rssm import RSSMPredictor
        return RSSMPredictor(arg, device), True
    if kind == "slosh":
        return P.SloshPhysicsPredictor(arg or "b3"), False
    if kind == "oracle":
        return P.OraclePredictor(), False
    if kind == "detector":
        return P.PostOnsetDetector(), False
    if kind == "constant":
        return P.ConstantPredictor(), False
    raise ValueError(spec)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--ood", action="append", default=[], help="name=<prep dir>[:split]")
    ap.add_argument("--test-split", default="test")
    ap.add_argument("--fa-per-min", type=float, default=1.0)
    ap.add_argument("--credit-window-s", type=float, default=2.0)
    ap.add_argument("--alarm-horizon-s", type=float, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    cache = {}

    def load(pdir, split, images):
        k = (pdir, split, images)
        if k not in cache:
            cache[k] = D.load_prepared(pdir, split, with_images=images, limit=a.limit)
        return cache[k]

    reports = []
    for spec in a.model:
        pred, needs_img = build(spec, a.device, a.prep)
        sets = {a.test_split: load(a.prep, a.test_split, needs_img)}
        for o in a.ood:
            name, _, rest = o.partition("=")
            pdir, _, split = rest.partition(":")
            sets[name] = load(pdir, split or None, needs_img)
        rep = Hn.evaluate(pred, load(a.prep, "val", needs_img), sets, fa_per_min=a.fa_per_min,
                          credit_window_s=a.credit_window_s, alarm_horizon_s=a.alarm_horizon_s)
        print(Hn.format_report(rep), flush=True)
        reports.append(rep)
    if a.out:
        Path(a.out).write_text(json.dumps(reports, indent=1, default=float))


if __name__ == "__main__":
    main()
