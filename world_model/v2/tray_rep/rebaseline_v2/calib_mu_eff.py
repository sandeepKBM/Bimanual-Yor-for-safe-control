"""Refit the mu_roll -> mu_eff map on the FIXED pin_cup physics into calib_v2/ (never touches boed/calib.json).
Same procedure as boed/check_model.py calib (n_start=3, VAL-like ball starts, noise-free, open-loop pre+hold window)."""
import json, os, sys
from pathlib import Path
assert not os.environ.get("BIMANUAL_LEGACY_PIN")
from world_model.v2.tray_rep.boed import model as MD
from world_model.v2.tray_rep.boed import check_model as CM
OUT = Path("/common/users/ss5772/bimanual_data/tray_rep/calib_v2/mu_eff_calib.json")
MD.CALIB_PATH = OUT
d = CM.calib(n_start=int(sys.argv[1]) if len(sys.argv) > 1 else 3)
old = json.loads((Path(__file__).resolve().parents[1] / "boed" / "calib.json").read_text())
import numpy as np
print("mu_roll[mm]  old_eff  new_eff  new/old   ratio_old  ratio_new")
for m, o, n in zip(d["mu"], old["mu_eff"], d["mu_eff"]):
    print(f"{m*1e3:8.3f} {o*1e3:8.3f} {n*1e3:8.3f} {n/o:8.3f} {o/m:9.3f} {n/m:9.3f}")
