"""gen_v2.py -- regenerate the VAL + TEST shards of data_piper_full / data_piper_trk_full on the FIXED pin_cup physics.
Same generator (gen_nominal.process_shard), same seeds (seed0 + g*25 + j), same settings (meta.json copied from the old dataset).
Only shards with g%5 in {3 (val), 4 (test)}. No train. SIMULATION ONLY (CPU).
  python -m world_model.v2.tray_rep.rebaseline_v2.gen_v2 --setup piper_trk --workers 8 [--merge] [--splits val,test]
"""
import argparse, json, os, shutil, time
from multiprocessing import Pool
from pathlib import Path
assert not os.environ.get("BIMANUAL_LEGACY_PIN"), "BIMANUAL_LEGACY_PIN must not be set"
from world_model.v2.tray_rep import common as K
from world_model.v2.tray_rep import gen_nominal as GN

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", required=True, choices=["piper", "piper_trk"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--max-shards", type=int, default=None, help="smoke: first N shards per split")
    a = ap.parse_args()
    old = K.ROOT / f"data_{a.setup}_full"; new = K.ROOT / f"data_{a.setup}_full_v2"
    meta = json.loads((old / "meta.json").read_text())
    new.mkdir(parents=True, exist_ok=True)
    if not (new / "meta.json").exists():
        shutil.copy(old / "meta.json", new / "meta.json")
    splits = a.splits.split(",")
    n_shards = -(-meta["n_eps"] // meta["eps_per_shard"])
    if a.merge:
        for sp in splits:
            exp = sum(1 for g in range(n_shards) if K.shard_split(g) == sp)
            print("[merge]", GN.merge_split(new, sp, False, exp), flush=True)
        return
    trk = dict(obs_noise_m=meta["obs_noise_m"], obs_latency_ticks=meta["obs_latency_ticks"],
               obs_rate_hz=meta["obs_rate_hz"], obs_dropout_p=meta["obs_dropout_p"])
    gs = [g for g in range(n_shards) if K.shard_split(g) in splits]
    tasks = [(g, str(new), meta["obs_mode"], meta["a_scale"], meta["lip"], meta["intensity"], meta["eps_per_shard"],
              meta["seed0"], meta["snap_stride"], meta["snap_min_tick"], meta["w_scale"], meta["alpha"],
              meta["carry_mode"], trk) for g in gs]
    if a.max_shards:
        tasks = tasks[:a.max_shards]
    t0 = time.perf_counter()
    with Pool(a.workers, maxtasksperchild=8) as P:
        for r in P.imap_unordered(GN.process_shard, tasks):
            if not r["skipped"]:
                print(f"[gen_v2] shard {r['g']:04d} {r['split']}: {r['n_eps']} eps fail {r['fail_eps']}/{r['n_eps']} {r['wall_s']:.1f}s", flush=True)
    print(f"[gen_v2] done {len(tasks)} shards wall {time.perf_counter()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    main()
