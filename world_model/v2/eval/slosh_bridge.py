"""world_model/v2/eval/slosh_bridge.py -- score the experiments/slosh predictors
(analytic a0..b3, residual c*, GRU d_gru) with the v2 harness on IDENTICAL data.

experiments/slosh is imported READ-ONLY (data_prep.load_rig, slosh_models).

  1. rig -> canonical prepared dir (same trajectories, same 70/30 split as the slosh
     study: slosh TRAIN -> our train (85%) + val (15%, threshold selection); slosh
     TEST -> our test, untouched):
       python -m world_model.v2.eval.slosh_bridge prep --out $PREP/rig_v1
  2. evaluate the study's own per-step scores (bit-identical numbers) next to any
     v2 model trained on the same prepared dir:
       python -m world_model.v2.eval.slosh_bridge eval --prep $PREP/rig_v1 \
           --keys b3_world_lti d_gru a0_static_tilt --H 0.25 [--gru $RUN/gru_rig/model.pt]
  3. `crosscheck`: recompute the slosh study's own step AUROC/AUPRC with OUR
     metrics on its test split and assert agreement (harness validation).

Live (non-precomputed) versions for NEW data: predictors.SloshPhysicsPredictor
('a0','a1','b1','b2','b3'; params read from scores_phys.npz). d_gru was trained on
200 Hz central-difference kinematics, so it is only valid where 200 Hz cup
kinematics exist (ep['kin_hi']); use the precomputed route on rig data.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
SLOSH = REPO / "experiments" / "slosh"
sys.path.insert(0, str(REPO))
from world_model.v2.eval import metrics as M  # noqa: E402
from world_model.v2.eval import harness as Hn  # noqa: E402

SCORE_FILES = ("scores_phys.npz", "scores_nets_gru.npz", "scores_nets_c.npz", "scores_nets_cdyn.npz")


def _slosh():
    if str(SLOSH) not in sys.path:
        sys.path.insert(0, str(SLOSH))
    import data_prep, slosh_models  # noqa: E401
    return data_prep, slosh_models


def axis_to_quat(n):
    """minimal rotation z -> n as wxyz (cup is axisymmetric; twist irrelevant)."""
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)
    q = np.stack([1.0 + n[..., 2], -n[..., 1], n[..., 0], np.zeros_like(n[..., 0])], -1)
    bad = q[..., 0] < 1e-6
    q[bad] = np.array([0.0, 1.0, 0.0, 0.0])
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def rig_to_prepared(out, pkl=SLOSH / "data" / "rig_v1.pkl", val_frac=0.15, seed=0):
    from world_model.v2.model.data import save_canonical
    data_prep, sm = _slosh()
    D = data_prep.load_rig(str(pkl))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    E, T = len(D["onset"]), D["T"]; hz = 1.0 / sm.DT
    quat = axis_to_quat(D["n"])
    recs = []
    for i in range(E):
        sc = np.zeros(T, np.int32)
        if D["onset"][i] >= 0:
            sc[D["onset"][i]:] = 1
        ep = dict(id=f"rig/{i:05d}", hz=hz, T=T, t_start=int(D["k0"][i]), onset=int(D["onset"][i]),
                  spill_count=sc, collision=np.zeros(T, bool), proprio=np.zeros((T, 0), np.float32),
                  action=np.zeros((T, 0), np.float32), images={}, cup_pos=D["pos"][i].astype(np.float32),
                  cup_quat=quat[i].astype(np.float32), rig_index=np.array(i),
                  meta=dict(family=str(D["family"][i]), slosh_split="test" if D["test"][i] else "train",
                            perturbation_family=str(D["family"][i])))
        fn = f"ep_{i:05d}.npz"; save_canonical(ep, out / fn)
        recs.append(dict(file=fn, id=ep["id"], T=T, hz=hz, onset=ep["onset"], t_start=ep["t_start"], meta=ep["meta"]))
    rng = np.random.default_rng(seed)
    tr = [r["id"] for r in recs if r["meta"]["slosh_split"] == "train"]
    rng.shuffle(tr); nv = int(round(val_frac * len(tr)))
    splits = dict(train=sorted(tr[nv:]), val=sorted(tr[:nv]),
                  test=[r["id"] for r in recs if r["meta"]["slosh_split"] == "test"], ood=[])
    (out / "index.json").write_text(json.dumps(dict(source=f"rig:{pkl}", hz=hz, episodes=recs, splits=splits), indent=1))
    print(f"[rig->prep] {E} eps ({int((D['onset'] >= 0).sum())} spill) -> {out}; " +
          ", ".join(f"{k}={len(v)}" for k, v in splits.items()), flush=True)


class PrecomputedSlosh:
    """The slosh study's own per-step scores (E,800), indexed by ep['rig_index']."""
    probabilistic = False
    horizons_s = None

    def __init__(self, key):
        for f in SCORE_FILES:
            d = np.load(SLOSH / "data" / f, allow_pickle=True)
            if key in d.files:
                self.S = d[key]; break
        else:
            raise KeyError(key)
        self.name = f"slosh[{key}] (precomputed)"

    def predict(self, ep):
        return self.S[int(ep["rig_index"])]


def crosscheck(H_steps=50):
    """Our metrics vs slosh_models.auroc/auprc on the slosh TEST split (all steps)."""
    data_prep, sm = _slosh()
    D = data_prep.load_rig(str(SLOSH / "data" / "rig_v1.pkl"))
    y, valid = data_prep.labels(D["onset"], D["k0"], D["T"], H_steps)
    te = D["test"]; rows = []
    for key in ("b3_world_lti", "d_gru", "a0_static_tilt"):
        S = PrecomputedSlosh(key).S
        s, yy = S[te][valid[te]], y[te][valid[te]]
        ours = (M.auroc(s, yy), M.auprc(s, yy)); theirs = (sm.auroc(s, yy), sm.auprc(s, yy))
        rows.append((key, ours, theirs))
        print(f"[crosscheck H={H_steps}] {key:16s} ours AUROC={ours[0]:.4f} AUPRC={ours[1]:.4f} | "
              f"slosh_models AUROC={theirs[0]:.4f} AUPRC={theirs[1]:.4f}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep"); p.add_argument("--out", required=True)
    e = sub.add_parser("eval"); e.add_argument("--prep", required=True)
    e.add_argument("--keys", nargs="+", default=["b3_world_lti", "d_gru", "a0_static_tilt"])
    e.add_argument("--live", nargs="*", default=[], help="causal live SloshPhysicsPredictor models, e.g. b3 a0")
    e.add_argument("--gru", default=None, help="v2 proprio_gru ckpt trained on the same prep dir")
    e.add_argument("--H", type=float, nargs="+", default=[0.1, 0.25, 0.5])
    e.add_argument("--fa-per-min", type=float, default=1.0); e.add_argument("--out", default=None)
    c = sub.add_parser("crosscheck"); c.add_argument("--H-steps", type=int, default=50)
    a = ap.parse_args(argv)
    if a.cmd == "prep":
        rig_to_prepared(a.out)
    elif a.cmd == "crosscheck":
        crosscheck(a.H_steps)
    else:
        from world_model.v2.model.data import load_prepared
        va, te = load_prepared(a.prep, "val"), load_prepared(a.prep, "test")
        from world_model.v2.eval.predictors import SloshPhysicsPredictor
        preds = [PrecomputedSlosh(k) for k in a.keys] + [SloshPhysicsPredictor(m) for m in a.live]
        if a.gru:
            from world_model.v2.model.proprio_gru import GRUPredictor
            preds.append(GRUPredictor(a.gru))
        reps = []
        for p_ in preds:
            rep = Hn.evaluate(p_, va, {"test": te}, horizons_s=a.H, fa_per_min=a.fa_per_min)
            print(Hn.format_report(rep), flush=True); reps.append(rep)
        if a.out:
            Path(a.out).write_text(json.dumps(reps, indent=1, default=float))


if __name__ == "__main__":
    main()
