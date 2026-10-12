"""Nominal carry generators (action sequences, 50 Hz, u = [a(3), w(3)])."""
import numpy as np
from world_model.v2.testbeds.a_loose_items.tray_env import fallback_U

def synthetic(rng, T=200, seg=(8, 40)):
    """Random held-action carry from the 85-element set (documented synthetic distribution, NOT spill_v2)."""
    U = fallback_U(); out = []
    while len(out) < T:
        u = U[rng.integers(len(U))]
        out += [u] * int(rng.integers(seg[0], seg[1] + 1))
    return np.array(out[:T])

def replay(D, e, T=300):
    """first T ticks of the logged spill_v2 action sequence of episode e of a load_split dict (act = [a, w_eff])."""
    s, L = D["start"][e], D["lens"][e]
    return D["act"][s:s + min(L, T)].astype(np.float64)
