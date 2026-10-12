"""Hidden-parameter sensitivity (SIM ONLY): same replayed action sequences, vary item friction / bounce / radius; report the
fraction of episodes whose spill label flips vs the baseline (friction 0.4, dampratio 1.0, r 7 mm) and the onset shift."""
import json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
from world_model.v2.testbeds.a_loose_items.tray_env import TrayEnv, TrayCfg
from world_model.v2.testbeds.a_loose_items import carry
from world_model.v2.wm_audit import common as WC
E_N = 30
D = WC.load_split("test"); rng = np.random.default_rng(11)
eps = [carry.replay(D, int(rng.integers(len(D["lens"]))), T=300) for _ in range(E_N)]
def labels(**kw):
    env = TrayEnv(TrayCfg(n_items=10, **kw)); out = []
    for i, acts in enumerate(eps):
        env.reset(500 + i); on = -1
        for t, u in enumerate(acts):
            r = env.tick(u)
            if r["n_spilled"] > 0: on = t + 1; break
        out.append(on)
    return np.array(out)
base = labels(); res = dict(base_fail=float((base >= 0).mean()))
for name, kw in dict(friction_0p15=dict(friction=0.15), friction_0p8=dict(friction=0.8), bounce_0p5=dict(dampratio=0.5),
                     radius_5p5mm=dict(item_r=0.0055), radius_8p5mm=dict(item_r=0.0085)).items():
    L = labels(**kw)
    both = (base >= 0) & (L >= 0)
    res[name] = dict(fail=float((L >= 0).mean()), label_flip=float(((base >= 0) != (L >= 0)).mean()),
                     median_abs_onset_shift_ticks=float(np.median(np.abs(base[both] - L[both]))) if both.any() else None)
    print(name, res[name], flush=True)
json.dump(res, open(sys.argv[1], "w"))
print(res["base_fail"])
