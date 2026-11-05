"""stack/eval/schema.py -- the one result record every eval run writes (one per world), builder + validator.  SIMULATION ONLY."""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

KEYS = ("universe", "world_id", "seed", "theta", "task", "planner", "proposer", "producers", "verdicts", "outcome", "metrics",
        "assumptions", "guarantee", "code_rev", "wall_s")
OUTCOMES = ("ok", "fail", "refused", "violation", "error")
GUARANTEES = ("proof", "sampled", "refuted", "none")
PROPOSER_KEYS = ("spec", "model", "reasoning")
_REPO = Path(__file__).resolve().parents[4]


@lru_cache(maxsize=1)
def code_rev() -> dict:
    """git rev-parse HEAD + dirty flag of the repo this code runs from."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(_REPO), *a], capture_output=True, text=True, timeout=30, check=True).stdout.strip()
        return dict(rev=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain", "--untracked-files=no")))
    except Exception as e:                                   # no git: say so, never fake a rev
        return dict(rev="unknown", dirty=None, error=repr(e))


def proposer_record(spec=None, model=None, reasoning=None) -> dict:
    return dict(spec=spec, model=model, reasoning=reasoning)


def build_record(*, universe, world_id, seed, theta, task, planner, proposer=None, producers=(), verdicts=(), outcome, metrics=None,
                 assumptions=None, guarantee="none", wall_s=0.0) -> dict:
    rec = dict(universe=universe, world_id=world_id, seed=int(seed), theta=theta, task=task, planner=planner,
               proposer=proposer if proposer is not None else proposer_record(), producers=list(producers), verdicts=list(verdicts),
               outcome=outcome, metrics=metrics or {}, assumptions=assumptions or {}, guarantee=guarantee, code_rev=code_rev(),
               wall_s=round(float(wall_s), 2))
    validate(rec)
    return rec


def validate(rec: dict) -> dict:
    """raises ValueError listing every problem; returns rec."""
    bad = []
    if not isinstance(rec, dict):
        raise ValueError("record is not a dict")
    bad += [f"missing key {k}" for k in KEYS if k not in rec]
    if not bad:
        if rec["outcome"] not in OUTCOMES:
            bad.append(f"outcome {rec['outcome']!r} not in {OUTCOMES}")
        if rec["guarantee"] not in GUARANTEES:
            bad.append(f"guarantee {rec['guarantee']!r} not in {GUARANTEES}")
        if not isinstance(rec["proposer"], dict) or any(k not in rec["proposer"] for k in PROPOSER_KEYS):
            bad.append(f"proposer needs keys {PROPOSER_KEYS}")
        for k, t in (("theta", dict), ("metrics", dict), ("assumptions", dict), ("code_rev", dict), ("verdicts", list), ("producers", list)):
            if not isinstance(rec[k], t):
                bad.append(f"{k} must be {t.__name__}")
        if not isinstance(rec["seed"], int):
            bad.append("seed must be int")
        if not isinstance(rec["wall_s"], (int, float)):
            bad.append("wall_s must be a number")
        if isinstance(rec["code_rev"], dict) and not {"rev", "dirty"} <= set(rec["code_rev"]):
            bad.append("code_rev needs rev + dirty")
        for i, v in enumerate(rec["verdicts"] if isinstance(rec["verdicts"], list) else []):
            if not isinstance(v, dict) or not {"checker", "admit", "guarantee"} <= set(v):
                bad.append(f"verdicts[{i}] needs checker/admit/guarantee")
    if bad:
        raise ValueError("invalid result record: " + "; ".join(bad))
    return rec
