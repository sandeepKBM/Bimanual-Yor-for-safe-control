"""checks/dynamic.py -- sampled rows: a backup family stays out of U and ends in R over worlds drawn from the belief cell.  SIMULATION ONLY.
Guarantee: sampled (n worlds stated) with a one-sided Clopper-Pearson upper bound on the failure rate (i.i.d. worlds, conf).  margin = None (a sampled
family has no signed distance); ok says whether a family member passed every world."""
from __future__ import annotations

from world_model.v2.stack.checks.row import Row


def cp_upper(k, n, conf=0.95):
    """one-sided Clopper-Pearson upper bound on a failure rate after k failures in n i.i.d. trials."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(conf, k + 1, n - k))


def _key(fam, d):
    """family dicts are keyed by float in memory and by str after a JSON round trip."""
    return d if d in fam else str(float(d))


def counts_row(name: str, n: int, n_bad: int, conf: float = 0.95, **worst) -> Row:
    eps = cp_upper(n_bad, n, conf) if n else 1.0
    return Row(name, bool(n and n_bad == 0), margin=None, unit="", guarantee="sampled" if (n and n_bad == 0) else "refuted", worst=worst or None,
               n=n, eps_upper=round(eps, 4), assumptions=dict(conf=conf, bound="Clopper-Pearson, i.i.d. worlds from the cell"),
               why="" if (n and n_bad == 0) else (f"{n_bad} of {n} worlds failed" if n else "no world qualified: nothing was sampled"))


def family_row(records, seg, family, conf=0.95) -> Row:
    """records: per-world dicts with 'segments' (each with 'seg', 'nominal' {viol}, 'family' {d: {n, n_bad, first_bad}}); the longest-first `family` of
    backup parameters.  ok = some member passes every qualifying world and the nominal run had no U entry."""
    worlds = [(r, next((s for s in r["segments"] if s["seg"] == seg), None)) for r in records if not r.get("refused")]
    worlds = [(r, s) for r, s in worlds if s is not None]
    n = len(worlds)
    if not n:
        return counts_row(f"dynamic[{seg}]", 0, 0, conf)
    nominal_bad = [(r, s) for r, s in worlds if s["nominal"]["viol"]]
    for d in family:
        if not nominal_bad and not any(s["family"][_key(s["family"], d)]["n_bad"] for _, s in worlds):
            return counts_row(f"dynamic[{seg}]", n, 0, conf, backup_retreat=float(d))
    best = min(family, key=lambda d: sum(bool(s["family"][_key(s["family"], d)]["n_bad"]) for _, s in worlds))
    tags = {r.get("tag") for r, s in nominal_bad} | {r.get("tag") for r, s in worlds if s["family"][_key(s["family"], best)]["first_bad"]}
    return counts_row(f"dynamic[{seg}]", n, max(len(tags), 1), conf, best_retreat=float(best))
