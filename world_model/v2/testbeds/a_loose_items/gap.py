"""Reduced model vs MuJoCo (SIM ONLY). Replays spill_v2 test action sequences through TrayEnv, logs the true item state,
then (1) predicts the episode label open-loop from the settled start, (2) plan-aware prediction from mid-episode states
(label = nominal continuation spills within 100 ticks), with privileged state vs a camera-like degraded state.
  python -m world_model.v2.testbeds.a_loose_items.gap --episodes 40 --out <json>"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[4]; sys.path.insert(0, str(REPO))
import mujoco
from world_model.v2.testbeds.a_loose_items.tray_env import TrayEnv, TrayCfg, quat_axis
from world_model.v2.testbeds.a_loose_items.reduced import Reduced, rotmat
from world_model.v2.testbeds.a_loose_items import carry
from world_model.v2.wm_audit import common as WC

def tray_frame_state(env, w_world):
    R = rotmat(env.quat)
    pw = env.d.qpos[env.idx].reshape(env.n, 3)
    vw = np.stack([env.d.qvel[env.m.jnt_dofadr[env.m.body_jntadr[env.m.body(f"item_{i}").id]]:][:3] for i in range(env.n)])
    rv = pw - env.pos
    vrel_w = vw - env.v - np.cross(w_world, rv)
    return rv @ R, vrel_w @ R

def auc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    if y.all() or (~y).all(): return None
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--episodes", type=int, default=40)
    ap.add_argument("--beta", type=float, default=45.0); ap.add_argument("--h", type=float, default=0.025)
    ap.add_argument("--n", type=int, default=10); ap.add_argument("--out", required=True); a = ap.parse_args()
    cfg = TrayCfg(n_items=a.n, beta_deg=a.beta, h=a.h); env = TrayEnv(cfg); red = Reduced(cfg)
    D = WC.load_split("test"); rng = np.random.default_rng(7)
    rows_ep, rows_mid = [], []
    t0 = time.perf_counter()
    for ep in range(a.episodes):
        acts = carry.replay(D, int(rng.integers(len(D["lens"]))), T=300)
        env.reset(1000 + ep)
        st = []
        p0, v0 = tray_frame_state(env, np.zeros(3))
        pos0, quat0, v0t = env.pos.copy(), env.quat.copy(), env.v.copy()
        onset, w = -1, np.zeros(3)
        for t, u in enumerate(acts):
            if t % 10 == 0 and t >= 10:
                pp, vv = tray_frame_state(env, w)
                st.append((t, pp, vv, env.pos.copy(), env.quat.copy(), env.v.copy()))
            r = env.tick(u); w = u[3:]
            if r["n_spilled"] > 0 and onset < 0:
                onset = t + 1; break
        sp, tk, mn = red.rollout(p0, v0, pos0, quat0, v0t, acts)
        rows_ep.append(dict(true=onset >= 0, onset=onset, red=bool(sp), red_onset=tk, red_min=mn))
        L = len(acts)
        for (t, pp, vv, ps, qs, vs) in st:
            if onset < 0 and t + 100 > L: continue          # label horizon must fit in the logged episode
            y = (onset >= 0) and (onset - t <= 100)
            fut = acts[t:t + 100]
            out = dict(t=t, y=bool(y))
            for name, (p_in, v_in) in dict(priv=(pp, vv), cam=(pp + rng.normal(0, 0.003, pp.shape), np.zeros_like(vv))).items():
                s2, _, m2 = red.rollout(p_in, v_in, ps, qs, vs, fut)
                out[name + "_spill"] = bool(s2); out[name + "_min"] = m2
            rows_mid.append(out)
    ep_t = np.array([r["true"] for r in rows_ep]); ep_r = np.array([r["red"] for r in rows_ep])
    res = dict(cfg=dict(n=a.n, beta=a.beta, h=a.h), episodes=a.episodes, wall_s=time.perf_counter() - t0,
               open_loop=dict(true_fail=float(ep_t.mean()), red_fail=float(ep_r.mean()), agree=float((ep_t == ep_r).mean()),
                              tpr=float(ep_r[ep_t].mean()) if ep_t.any() else None, fpr=float(ep_r[~ep_t].mean()) if (~ep_t).any() else None,
                              auc_min_margin=auc([-r["red_min"] for r in rows_ep], ep_t)))
    y = np.array([r["y"] for r in rows_mid])
    for name in ("priv", "cam"):
        pr = np.array([r[name + "_spill"] for r in rows_mid]); sc = [-r[name + "_min"] for r in rows_mid]
        res["plan_aware_" + name] = dict(n=len(y), pos_frac=float(y.mean()), acc=float((pr == y).mean()),
                                         tpr=float(pr[y].mean()) if y.any() else None, fpr=float(pr[~y].mean()) if (~y).any() else None,
                                         auc=auc(sc, y))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(summary=res, ep=rows_ep, mid=rows_mid), open(a.out, "w"))
    print(json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
