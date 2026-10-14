"""wrongpoint/analyze.py -- table per cell: drops (point / conf / oracle / F0), interventions, progress, gap point-conf (paired bootstrap over carries)."""
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path("/common/users/ss5772/bimanual_data/tray_rep/wrongpoint/runs")


def read(p):
    out = {}
    if not p.exists():
        return out
    for l in p.read_text().splitlines():
        try:
            r = json.loads(l); out[r["ep"]] = r
        except Exception:
            pass
    return out


def boot(d, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(d)
    idx = rng.integers(0, n, (B, n))
    m = d[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cell_rows(name, root=ROOT):
    d = root / name
    cfg = json.loads((d / "cfg.json").read_text())
    arms = {a.stem: read(a) for a in d.glob("*.jsonl")}
    rows = []
    for rho in ("0.2", "0.4"):
        P, C, O = arms.get(f"point@{rho}", {}), arms.get(f"conf@{rho}", {}), arms.get("oracle", {})
        eps = sorted(set(P) & set(C) & set(O))
        if not eps:
            continue
        g = lambda A, k: np.array([A[e][k] for e in eps], float)
        dp, dc, do = g(P, "dropped"), g(C, "dropped"), g(O, "dropped")
        gap = dp - dc
        lo, hi = boot(gap)
        F0 = arms.get("F0", {})
        f0 = np.mean([F0[e]["dropped"] for e in eps if e in F0]) if F0 else float("nan")
        rows.append(dict(cell=name, rho=float(rho), n=len(eps), dp=int(dp.sum()), dc=int(dc.sum()), do=int(do.sum()), f0=float(f0),
                         ip=g(P, "interv_rate").mean(), ic=g(C, "interv_rate").mean(), io=g(O, "interv_rate").mean(),
                         pp=g(P, "progress").mean(), pc=g(C, "progress").mean(), po=g(O, "progress").mean(),
                         gap=float(gap.mean() * 100), lo=lo * 100, hi=hi * 100))
    return rows, cfg


def fmt(rows):
    s = f"{'cell':26s} rho  n   drops p/c/o  F0    interv p/c/o        progress p/c/o      gap(pp) [95%CI]\n"
    for r in rows:
        s += (f"{r['cell']:26s} {r['rho']:.1f} {r['n']:3d}  {r['dp']:2d}/{r['dc']:2d}/{r['do']:2d}   {r['f0']*100:4.0f}%  "
              f"{r['ip']:.3f}/{r['ic']:.3f}/{r['io']:.3f}  {r['pp']:.3f}/{r['pc']:.3f}/{r['po']:.3f}  {r['gap']:+5.1f} [{r['lo']:+.1f},{r['hi']:+.1f}]\n")
    return s


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--glob", default="*")
    a = ap.parse_args()
    allr = []
    for d in sorted(Path(a.root).glob(a.glob)):
        if (d / "cfg.json").exists():
            r, _ = cell_rows(d.name, Path(a.root))
            allr += r
    print(fmt(allr))
