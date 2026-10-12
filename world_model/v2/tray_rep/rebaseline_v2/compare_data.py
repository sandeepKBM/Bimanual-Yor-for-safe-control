"""Compare stored outcomes (onset) old vs v2 data, by split and mu bin. python -m ...compare_data"""
import json, numpy as np
from pathlib import Path
R = Path("/common/users/ss5772/bimanual_data/tray_rep")
edges = np.exp(np.linspace(np.log(5e-4), np.log(6e-3), 7))
out = {}
for setup in ["piper", "piper_trk"]:
    for sp in ["val", "test"]:
        o = np.load(R / f"data_{setup}_full/{sp}.npz"); n = np.load(R / f"data_{setup}_full_v2/{sp}.npz")
        assert (o["seed"] == n["seed"]).all() and np.allclose(o["hid"], n["hid"]) and (o["lens"] == n["lens"]).all()
        od, nd_ = o["onset"] >= 0, n["onset"] >= 0
        r = dict(n=int(len(od)), drops_old=int(od.sum()), drops_new=int(nd_.sum()),
                 both=int((od & nd_).sum()), old_only=int((od & ~nd_).sum()), new_only=int((~od & nd_).sum()),
                 onset_shift_both_mean_ticks=float((n["onset"] - o["onset"])[od & nd_].mean()) if (od & nd_).any() else None,
                 onset_shift_both_median_ticks=float(np.median((n["onset"] - o["onset"])[od & nd_])) if (od & nd_).any() else None,
                 ball_xy_maxdev_mean_old=float(np.mean([np.abs(o["obs"][s:s+l, 9:11]).max() for s, l in zip(o["start"], o["lens"])])),
                 ball_xy_maxdev_mean_new=float(np.mean([np.abs(n["obs"][s:s+l, 9:11]).max() for s, l in zip(n["start"], n["lens"])])))
        b = np.digitize(o["hid"], edges[1:-1])
        r["by_mu_bin"] = [dict(bin=f"{edges[i]*1e3:.2f}-{edges[i+1]*1e3:.2f}mm", n=int((b == i).sum()), old=int(od[b == i].sum()),
                                new=int(nd_[b == i].sum()), old_only=int((od & ~nd_)[b == i].sum()), new_only=int((~od & nd_)[b == i].sum()),
                                mean_onset_old=float(o["onset"][b == i][od[b == i]].mean()) if od[b == i].any() else None,
                                mean_onset_new=float(n["onset"][b == i][nd_[b == i]].mean()) if nd_[b == i].any() else None) for i in range(6)]
        out[f"{setup}/{sp}"] = r
Path(R / "rebaseline_v2/data_compare.json").write_text(json.dumps(out, indent=1))
for k, r in out.items():
    print(k, {a: r[a] for a in r if a != "by_mu_bin"})
    for x in r["by_mu_bin"]: print("   ", x)
