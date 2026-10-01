"""bead_rep/oracle.py -- search-certified "savable" check (DESIGN Q5) for (split, episode, t) states of the bead dataset.
SIMULATION / SAVED DATA ONLY (MuJoCo CPU; no GPU, no network, no hardware).

A state is CERTIFIED SAVABLE iff some action sequence in a fixed library keeps the beads in the cup (no rim spill, no
tunnel artifact) for the full horizon T (default 100 ticks = 2 s) from that state.  This is an existence proof (the
library is NOT exhaustive): not-certified states are reported as AMBIGUOUS, never as doomed, and the certified set is an
UNDER-approximation of the savable set (biased toward easily saved states).  State this wherever the numbers are used.

Library, in order, early exit at the first certificate (every branch truncated at the first rim spill / tunnel):
  1. hold      u = 0 for T ticks (coast).  Hold safe => the state is TRIVIALLY safe (reported as trivial, kind=1; the DESIGN
               pre-filter says to drop these from the candidate set).
  2. brake     retreat/decel: a = -v_h/|v_h| min(A_MAX, |v_h|/dt) (stop in <= 1 tick, no overshoot), w = axis back to vertical
               at <= W_MAX, then hold (u ~ 0)                                                           (kind=2)
  3. const_i   each of the 84 other held constants of the 85-element fallback set, for T ticks          (kind=3)
  4. rand2_j   n_rand (300) random 2-segment sequences: u1 for L1, u2 for L2 (L ~ U{3..15} ticks, u uniform in the 85-set
               == the branch law of vdirect/branches), then the brake controller to the horizon          (kind=4)
               [DESIGN left the tail unspecified; a coasting tail would run the cup to > 2.4 m/s and mostly tunnel]
best_margin = max over evaluated valid sequences of (min over ticks of l); a spilled sequence counts as <= -1e-3, so
best_margin > 0 <=> certified.  Tunnelled sequences (artifact) are invalid and excluded.

States file: CSV with header `split,ep,t` (split in train/val/test, ep = split-level episode number == row in
load_split, t = WM tick) or an int .npy (N,3) with split ids 0/1/2.  `candidates` builds one from a generated split.

  python -m world_model.v2.bead_rep.oracle candidates --split val --out cand.csv --n-near 1500 --n-safe 500
  python -m world_model.v2.bead_rep.oracle run --states cand.csv --tag val_v0 --shard 3/50
  python -m world_model.v2.bead_rep.oracle merge --tag val_v0
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import argparse
import csv
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from world_model.v2.bead_rep import bead_env as B  # noqa: E402
from world_model.v2.bead_rep import util as U  # noqa: E402

SPLIT_ID = {"train": 0, "val": 1, "test": 2}
KINDS = {0: "none", 1: "hold(trivial)", 2: "brake", 3: "const", 4: "rand2+brake"}


# ============================================================================================ state access
class StateStore:
    """(split, ep, t) -> restores the MuJoCo state into an env. Uses the nearest saved snapshot at or before t and replays
    the logged cup poses (spill_v2 dense) for the remaining ticks; from episode start if no snapshot precedes t."""

    def __init__(self, root=U.DATA):
        self.root = Path(root)
        self.index, self.snaps, self.dense = {}, {}, {}
        self.src = {g: p for _, g, p in U.source_shards()}

    def _index(self, split):
        if split not in self.index:
            self.index[split] = U.split_index(self.root, split)
        return self.index[split]

    def _snap(self, split, gidx):
        key = (split, gidx)
        if key not in self.snaps:
            f = self.root / split / f"snap_{gidx:04d}.npz"
            self.snaps[key] = dict(np.load(f)) if f.exists() else None
        return self.snaps[key]

    def _dense(self, gidx):
        if gidx not in self.dense:
            from world_model.v2.latent_hj import spill_data as sd
            self.dense = {gidx: sd.load_dense(self.src[gidx])}     # keep one shard resident
        return self.dense[gidx]

    def restore(self, env: B.BeadEnv, split, ep, t):
        """-> info dict. After this call env is in the state 'WM tick t' (pose 4t known, not yet pinned)."""
        rec, e = U.locate(self._index(split), ep)
        gidx = rec["gidx"]
        S = self._snap(split, gidx)
        k_tgt = t * B.STRIDE
        assert t < rec["lens"][e], f"t={t} beyond episode length {rec['lens'][e]}"
        best = None
        if S is not None:
            ok = np.flatnonzero((S["snap_ep"] == e) & (S["snap_tick"] <= t))
            if len(ok):
                best = int(ok[np.argmax(S["snap_tick"][ok])])
        if best is not None and S["snap_tick"][best] == t:
            env.restore(B.Snapshot.from_arrays(S["snap_state"][best], S["snap_prev"][best], S["snap_has_prev"][best],
                                               S["snap_kin"][best], S["snap_status"][best], S["snap_exit_step"][best], S["snap_k"][best]))
            return dict(gidx=gidx, e=e, from_tick=t, replayed_ticks=0)
        dn = self._dense(gidx)[e]
        pos, quat = dn["pos"], dn["quat"]
        vb = np.zeros_like(pos); vb[1:] = (pos[1:] - pos[:-1]) / B.DT
        if best is None:
            env.reset(pos[0], quat[0]); k0 = 0; tk = 0
        else:
            env.restore(B.Snapshot.from_arrays(S["snap_state"][best], S["snap_prev"][best], S["snap_has_prev"][best],
                                               S["snap_kin"][best], S["snap_status"][best], S["snap_exit_step"][best], S["snap_k"][best]))
            k0 = int(S["snap_k"][best]); tk = int(S["snap_tick"][best])
        for k in range(k0, k_tgt):
            env.set_kin(pos[k], quat[k], vb[k])
            env.phys_step()
        env.set_kin(pos[k_tgt], quat[k_tgt], vb[k_tgt])
        return dict(gidx=gidx, e=e, from_tick=tk, replayed_ticks=t - tk)


# ============================================================================================ search
def _seg_actions(rng, Uset, n_ticks):
    seq = np.zeros((n_ticks, 6))
    t = 0
    for _ in range(2):
        L = int(rng.integers(3, 16)); seq[t:t + L] = Uset[int(rng.integers(len(Uset)))]; t += L
    return seq, t


def run_sequence(env, snap, T, seq=None, n_open=0, brake_tail=False):
    """restore -> run up to T ticks: actions seq[:n_open] open loop, then brake_action (brake_tail) or u=0.
    seq None & brake_tail -> pure brake.  Returns (outcome, min_l, tick_of_event): outcome in safe | spill | tunnel."""
    env.restore(snap)
    lmin = 1.0
    for t in range(T):
        if seq is not None and t < n_open:
            u = seq[t]
        elif brake_tail:
            u = B.brake_action(env.v, env.axis())
        else:
            u = np.zeros(6)
        r = env.tick(u)
        lmin = min(lmin, r["l"])
        if r["tunnel"]:
            return "tunnel", lmin, t
        if r["n_spilled"] > 0:
            return "spill", min(lmin, -1e-3), t
    return "safe", lmin, T


def certify(env, snap, T=B.T_BRANCH, n_rand=300, rng=None, Uset=None):
    """search-certified savable check from `snap` (env already holds that state). -> record dict."""
    Uset = B.fallback_U() if Uset is None else Uset
    rng = np.random.default_rng(0) if rng is None else rng
    rec = dict(certified=0, kind=0, idx=-1, trivial=0, best_margin=float("nan"), hold_margin=float("nan"), hold_spill_tick=-1,
               n_eval=0, n_spill=0, n_tunnel=0, already_spilled=int((snap.status == B.ST_RIM).any()), l0=env.margin(),
               out_of_domain=0)
    if rec["already_spilled"]:
        return rec
    if float(np.arccos(np.clip(env.axis()[2], -1.0, 1.0))) > B.TILT_LIM:
        # The fallback abstraction clamps the axis to <= TILT_LIM (1.5 rad) EVERY step; from a state already beyond it the
        # first step would snap the kinematic cup by up to ~20 deg in 5 ms and tunnel the beads (smoke: all 386 branches).
        # Such states are outside the action set's domain: not searched, reported as out_of_domain (never as doomed).
        rec["out_of_domain"] = 1
        return rec
    best = -9.0

    def attempt(**kw):
        nonlocal best
        out, lm, te = run_sequence(env, snap, T, **kw)
        rec["n_eval"] += 1
        if out == "tunnel":
            rec["n_tunnel"] += 1
        else:
            best = max(best, lm)
            rec["best_margin"] = best
            if out == "spill":
                rec["n_spill"] += 1
        return out, lm, te

    # 1. hold
    out, lm, te = attempt(seq=np.zeros((T, 6)), n_open=T)
    rec["hold_margin"] = lm if out != "tunnel" else float("nan")
    rec["hold_spill_tick"] = te if out == "spill" else -1
    if out == "safe":
        rec.update(certified=1, kind=1, idx=0, trivial=1)
        return rec
    # 2. brake
    out, _, _ = attempt(seq=None, n_open=0, brake_tail=True)
    if out == "safe":
        rec.update(certified=1, kind=2, idx=0)
        return rec
    # 3. the other 84 held constants
    for i in range(len(Uset)):
        if not np.any(Uset[i]):
            continue
        out, _, _ = attempt(seq=np.tile(Uset[i], (T, 1)), n_open=T)
        if out == "safe":
            rec.update(certified=1, kind=3, idx=i)
            return rec
    # 4. random 2-segment + brake tail
    for j in range(n_rand):
        seq, n_open = _seg_actions(rng, Uset, T)
        out, _, _ = attempt(seq=seq, n_open=n_open, brake_tail=True)
        if out == "safe":
            rec.update(certified=1, kind=4, idx=j)
            return rec
    return rec


# ============================================================================================ states file
def read_states(path):
    path = Path(path)
    if path.suffix == ".npy":
        a = np.load(path)
        inv = {v: k for k, v in SPLIT_ID.items()}
        return [(inv[int(s)], int(e), int(t)) for s, e, t in a]
    out = []
    with open(path) as f:
        for r in csv.DictReader(f):
            out.append((r["split"], int(r["ep"]), int(r["t"])))
    return out


def write_states(path, states):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["split", "ep", "t"]); w.writerows(states)


def cmd_candidates(a):
    """Candidate states from a generated split, on the snapshot grid (cheap saved-label proxy; NOT the savable set):
    near-boundary = bead margin l > 0 now and the LOGGED continuation spills within 2 s (onset - t in [1, 100] ticks);
    clearly safe = l > 0 and no logged spill for the rest of the episode.  DESIGN Q5 'hold spills within T' is then decided
    by the oracle's own hold branch (kind=1 = trivially safe)."""
    idx = U.split_index(a.data, a.split)
    rng = np.random.default_rng(a.seed)
    near, safe, e0 = [], [], 0
    for rec in idx:
        with np.load(rec["file"]) as z:
            l, lens, onset, obs = z["l"], z["lens"], z["onset"], z["obs"]; tun = z["tunnel"]
        st = np.concatenate([[0], np.cumsum(lens)[:-1]])
        sn = Path(a.data) / a.split / f"snap_{rec['gidx']:04d}.npz"
        if not sn.exists():
            e0 += rec["n_eps"]; continue
        with np.load(sn) as s:
            se, stk = s["snap_ep"], s["snap_tick"]
        for e, tk in zip(se, stk):
            if tun[e] or l[st[e] + tk] <= 0 or np.arccos(np.clip(obs[st[e] + tk, 2], -1, 1)) > B.TILT_LIM:
                continue                      # tilt > TILT_LIM: outside the fallback action set's domain (see certify)
            if onset[e] >= 0 and 1 <= onset[e] - tk <= 100:
                near.append((a.split, e0 + int(e), int(tk)))
            elif onset[e] < 0 and tk + 100 < lens[e]:
                safe.append((a.split, e0 + int(e), int(tk)))
        e0 += rec["n_eps"]
    pick = lambda L, n: [L[i] for i in rng.choice(len(L), min(n, len(L)), replace=False)] if L else []
    states = pick(near, a.n_near) + pick(safe, a.n_safe)
    write_states(a.out, states)
    print(f"[candidates] {a.split}: near pool {len(near)}, safe pool {len(safe)} -> wrote {len(states)} to {a.out}", flush=True)


# ============================================================================================ run / merge
def cmd_merge(a):
    od = Path(a.root) / "oracle" / a.tag
    recs = {}
    for f in sorted(od.glob("part_*.jsonl")):
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line); recs[r["row"]] = r
            except Exception:
                pass
    rows = [recs[k] for k in sorted(recs)]
    keys = ("certified", "kind", "idx", "trivial", "best_margin", "hold_margin", "hold_spill_tick", "n_eval", "n_spill", "n_tunnel",
            "already_spilled", "l0", "out_of_domain", "wall_s", "ep", "t")
    out = {k: np.array([r[k] for r in rows]) for k in keys}
    out["split"] = np.array([SPLIT_ID[r["split"]] for r in rows], np.int8)
    out["row"] = np.array([r["row"] for r in rows])
    U.atomic_savez(od.parent / f"{a.tag}.npz", **out)
    cert, triv = out["certified"] > 0, out["trivial"] > 0
    summ = dict(tag=a.tag, n=len(rows), certified=int(cert.sum()), trivially_safe=int(triv.sum()),
                certified_nontrivial=int((cert & ~triv).sum()), ambiguous=int((~cert & (out["out_of_domain"] == 0)).sum()),
                out_of_domain=int((out["out_of_domain"] > 0).sum()),
                by_kind={KINDS[k]: int((out["kind"] == k).sum()) for k in KINDS}, mean_wall_s=float(out["wall_s"].mean()) if rows else 0.0,
                note="certified = existence proof from the fixed library (under-approximation of savable); ambiguous != doomed")
    U.jdump(summ, od.parent / f"{a.tag}_summary.json")
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates"); c.add_argument("--split", default="val"); c.add_argument("--out", required=True)
    c.add_argument("--data", default=str(U.DATA)); c.add_argument("--n-near", type=int, default=1500)
    c.add_argument("--n-safe", type=int, default=500); c.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("run"); r.add_argument("--states", required=True); r.add_argument("--tag", required=True)
    r.add_argument("--shard", default="0/1")
    r.add_argument("--data", default=str(U.DATA), help="generated dataset root (shard/snap files)")
    r.add_argument("--root", default=str(U.ROOT), help="bead_replication root; results go to <root>/oracle/<tag>/")
    r.add_argument("--horizon", type=int, default=B.T_BRANCH)
    r.add_argument("--n-rand", type=int, default=300); r.add_argument("--seed", type=int, default=0)
    r.add_argument("--max-states", type=int, default=0)
    m = sub.add_parser("merge"); m.add_argument("--tag", required=True); m.add_argument("--root", default=str(U.ROOT))
    a = ap.parse_args()
    {"candidates": cmd_candidates, "run": cmd_run, "merge": cmd_merge}[a.cmd](a)


def cmd_run(a):
    states = read_states(a.states)
    i, n = U.parse_shard(a.shard)
    mine = list(range(i, len(states), n))
    if a.max_states:
        mine = mine[:a.max_states]
    od = Path(a.root) / "oracle" / a.tag
    U.safe_mkdir(od)
    part, sent = od / f"part_{i:04d}.jsonl", od / f"part_{i:04d}._DONE"
    if sent.exists():
        print(f"[oracle] part {i} already done", flush=True); return
    have = set()
    if part.exists():
        for line in part.read_text().splitlines():
            try:
                have.add(json.loads(line)["row"])
            except Exception:
                pass
    store = StateStore(a.data)
    env = B.BeadEnv()
    Uset = B.fallback_U()
    t0 = time.perf_counter()
    with open(part, "a") as fo:
        for row in mine:
            if row in have:
                continue
            split, ep, t = states[row]
            ts = time.perf_counter()
            info = store.restore(env, split, ep, t)
            snap = env.snapshot()
            rng = np.random.default_rng([a.seed, SPLIT_ID[split], ep, t])
            rec = certify(env, snap, a.horizon, a.n_rand, rng, Uset)
            rec.update(row=row, split=split, ep=ep, t=t, wall_s=time.perf_counter() - ts, replayed_ticks=info["replayed_ticks"],
                       kind_name=KINDS[rec["kind"]])
            fo.write(json.dumps(rec) + "\n"); fo.flush()
            print(f"[oracle] row {row} {split} ep {ep} t {t}: cert={rec['certified']} kind={rec['kind_name']} best={rec['best_margin']:.3f} "
                  f"hold_spill_tick={rec['hold_spill_tick']} n_eval={rec['n_eval']} {rec['wall_s']:.1f}s", flush=True)
    U.write_sentinel(sent)
    print(f"[oracle] part {i}/{n} done: {len(mine)} states in {time.perf_counter() - t0:.0f}s "
          f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
