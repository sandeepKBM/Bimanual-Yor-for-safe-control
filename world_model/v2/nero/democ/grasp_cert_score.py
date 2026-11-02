"""Offline scorer: re-evaluate grasp_cert mode='compliant' on saved release scenes for a list of gap_tol values.
usage: python score.py KN KAPPA GAPS(comma) OUT.json IN1.json [IN2.json ...]
Truth label recomputed from raw sim metrics (disp < 5 mm, tilt < 3 deg, end speed < 1 mm/s) and checked against the stored label."""
import json, sys
from multiprocessing import Pool
from world_model.v2.nero.democ import grasp_cert as GC, grasp_cert_compliant_xcheck as XC


def stays(r):
    s = r["sim"]
    v = s["max_disp_mm"] < 5.0 and s["max_tilt_deg"] < 3.0 and s["end_speed_mm_s"] < 1.0
    assert v == s["stays"]
    return v


def one(a):
    r, kn, kap, gaps = a
    st = GC.Stiffness(kn=kn, kappa=kap)
    if r["scene"] is None:                       # scene not modelable -> certificate rejects
        return [XC.verdict(False, stays(r)) for _ in gaps]
    sc = XC.scene_from_json(r["scene"])
    ms, cx = r.get("mass", XC.M_TRUE), r.get("com_x", 0.0)
    P = GC.Params(m_lo=ms, m_hi=ms, com_lo=(cx, r["com_y"], 0), com_hi=(cx, r["com_y"], 0))
    rel = "right" if r["keep"] == "left" else "left"
    s = stays(r)
    out = []
    for g in gaps:
        ok, _, _ = GC.release_ok(sc, rel, P, mode="compliant", stiffness=st, gap_tol=g)
        out.append(XC.verdict(bool(ok), s))
    return out


if __name__ == "__main__":
    kn, kap = float(sys.argv[1]), float(sys.argv[2])
    gaps = [float(x) for x in sys.argv[3].split(",")]
    out_path, ins = sys.argv[4], sys.argv[5:]
    res = dict(kn=kn, kappa=kap, gaps=gaps, sets={})
    with Pool(4) as pool:
        for f in ins:
            rows = json.load(open(f))["rows"]
            v = pool.map(one, [(r, kn, kap, gaps) for r in rows], chunksize=4)
            tab = {}
            for i, g in enumerate(gaps):
                col = [x[i] for x in v]
                tab[str(g)] = {k: col.count(k) for k in ("TP", "TN", "FALSE_SAFE", "false_unsafe")}
            res["sets"][f] = dict(n=len(rows), n_stays=sum(stays(r) for r in rows), table=tab)
            print(f, len(rows), flush=True)
    json.dump(res, open(out_path, "w"), indent=1)
    print(json.dumps({f: {g: list(t.values()) for g, t in s["table"].items()} for f, s in res["sets"].items()}, indent=0))
