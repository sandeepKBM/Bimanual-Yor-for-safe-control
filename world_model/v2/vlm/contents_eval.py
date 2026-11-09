"""vlm/contents_eval.py -- how well does a VLM estimate HIDDEN CONTENTS as an interval?  SIMULATION ONLY, offline except `ask`.

The VLM interval becomes the robot's prior Theta_0, so COVERAGE of the truth comes first, then width.
Subcommands: render | ask | score | mock-server.   (see report / --help)
Labels (AGENTS.md): truth = MODEL (sim masses); prior boxes = ASSUMED (tray fill 0..0.25 kg per mug, box contents 0.5..5 kg).
Caveats: the sim mug is a SOLID cylinder with a hidden mass-only fill geom, and the box is closed -> the fill is NOT visible in any render
(see `visible` field in records).  A model can only guess from priors; this measures calibration, not perception.
"""
from __future__ import annotations
import argparse, base64, json, math, os, pathlib, re, sys, time, urllib.request

import numpy as np

PROMPT_VERSION = "v1-2026-11-03"
PROMPT = """A two-armed robot will lift the object in the image and must NOT underestimate the load. You must estimate the hidden contents.
Known facts: {facts}
Give a 90% interval for each quantity: a range [lo, hi] that you believe contains the true value with probability 0.9. Too narrow an interval that misses the truth is worse than a wide one.
Reply with ONE JSON object and nothing after it:
{{"total_mass_kg": [lo, hi], "total_fill_ml": [lo, hi] or null, "contents_kg": [lo, hi], "p_cover": <probability the true total mass is inside your interval>, "evidence": "<=20 words"}}
total_mass_kg = everything lifted (object + mugs + contents). contents_kg = mass of what is carried on/in the object excluding the empty tray/box ({cdef}). total_fill_ml = total liquid over all mugs (trays only, else null)."""

PROMPT_V2_VERSION = "v2-2026-11-03"
# v2 (owner, 11-03 after v1 Nano: the model got the fill range right but did the mass arithmetic wrong, always low): ask ONLY for what the
# image or the label can tell (counts, item kind, whether the fill is visible, fill level if visible).  The code composes the mass
# interval from those answers and the known facts (compose_v2, MODEL); anything not visible is the full physical range.
PROMPT_V2 = {
    "tray": """A two-armed robot will lift the tray in the image. Known facts: {facts}
Answer ONLY what the image shows. Reply with ONE JSON object and nothing after it:
{{"n_mugs": <count of mugs you see>, "fill_visible": <true if you can see the liquid level inside the mugs, else false>,
"fill_frac": [lo, hi] fraction of capacity per mug (0..1) if fill_visible else null, "evidence": "<=20 words"}}""",
    "box": """A two-armed robot will lift the closed box in the image. Known facts: {facts}
Answer ONLY what the image or the label can tell. Reply with ONE JSON object and nothing after it:
{{"contents_visible": <true if you can see what is inside, else false>, "item_kind": "<what is most likely inside, e.g. cans>",
"n_items": [lo, hi] plausible count of such items that fit, "item_mass_kg": [lo, hi] mass of one such item,
"evidence": "<=20 words"}}""",
}
TRAY_EMPTY, MUG_EMPTY, MUG_CAP_KG, BOX_EMPTY = 0.30, 0.30, 0.25, 0.25    # MODEL / ASSUMED as in nero/contents/objects.py


def compose_v2(task, o):
    """v2 answer -> the same interval fields the scorer uses (MODEL arithmetic; invisible -> full physical range)."""
    if not o:
        return None
    try:
        if task.startswith("tray"):
            n = int(o.get("n_mugs") or int(task[4:]))
            fr = o.get("fill_frac") if o.get("fill_visible") else None
            flo, fhi = (0.0, 1.0) if not fr else (max(0.0, float(min(fr))), min(1.0, float(max(fr))))
            fill = [n * flo * MUG_CAP_KG, n * fhi * MUG_CAP_KG]
            cont = [n * MUG_EMPTY + fill[0], n * MUG_EMPTY + fill[1]]
            return dict(o, total_mass_kg=[TRAY_EMPTY + cont[0], TRAY_EMPTY + cont[1]], contents_kg=cont,
                        total_fill_ml=[1000 * fill[0], 1000 * fill[1]], composed="v2")
        nl, nh = sorted(float(x) for x in o["n_items"])
        ml, mh = sorted(float(x) for x in o["item_mass_kg"])
        cont = [nl * ml, nh * mh]
        return dict(o, total_mass_kg=[BOX_EMPTY + cont[0], BOX_EMPTY + cont[1]], contents_kg=cont, total_fill_ml=None, composed="v2")
    except (TypeError, ValueError, KeyError):
        return None


VIEWS = {   # MuJoCo free camera (lookat, distance, azimuth, elevation); the obj is at (0.38,-0.30,0.82)
    "torso": dict(lookat=[0.38, -0.30, 0.86], distance=0.85, azimuth=0.0, elevation=-38.0),
    "high": dict(lookat=[0.38, -0.30, 0.84], distance=0.80, azimuth=0.0, elevation=-80.0),
}
TASKS = ("tray2", "tray4", "tray6", "box")
# ASSUMED full-prior boxes (carry_cert cells): per-mug fill kg, box contents kg
from world_model.v2.stack import tasks as _TK
PRIOR_FILL, PRIOR_BOX = _TK.prior_box("tray2"), _TK.prior_box("box")      # one copy: stack/tasks.py (11-09)


# ------------------------------------------------------------------ facts / truth
def facts_and_truth(task, w):
    from world_model.v2.nero.contents.objects import TRAY, CUP, BOX
    if task.startswith("tray"):
        n = w.n_cups
        fills = [float(f) for f in w.fills_kg]
        facts = (f"a coffee tray (empty {TRAY['mass']:.2f} kg) carrying {n} identical ceramic mugs (empty {CUP['empty']:.2f} kg each, capacity "
                 f"{int(CUP['fill_max_kg']*1000)} ml, diameter {CUP['D']*100:.0f} cm, height {CUP['H']*100:.1f} cm). Each mug holds between 0 and its capacity of liquid (1 ml = 1 g).")
        cdef = "mugs plus their liquid"
        truth = dict(total_mass_kg=TRAY["mass"] + n * CUP["empty"] + sum(fills), contents_kg=n * CUP["empty"] + sum(fills),
                     total_fill_ml=1000.0 * sum(fills), per_mug_fill_kg=fills)
        prior = dict(total_mass_kg=[TRAY["mass"] + n * CUP["empty"] + n * PRIOR_FILL[0], TRAY["mass"] + n * CUP["empty"] + n * PRIOR_FILL[1]],
                     contents_kg=[n * CUP["empty"], n * (CUP["empty"] + PRIOR_FILL[1])], total_fill_ml=[0.0, 1000.0 * n * PRIOR_FILL[1]])
    else:
        bx = w.box_outer
        facts = (f"a closed cardboard box, outer size {bx[0]:.2f} x {bx[1]:.2f} x {bx[2]:.2f} m, empty mass {BOX['mass']:.2f} kg, "
                 "holding hidden items (e.g. cans or jars) as one load.")
        cdef = "the hidden items"
        truth = dict(total_mass_kg=BOX["mass"] + w.contents_kg, contents_kg=float(w.contents_kg), total_fill_ml=None, per_mug_fill_kg=None)
        prior = dict(total_mass_kg=[BOX["mass"] + PRIOR_BOX[0], BOX["mass"] + PRIOR_BOX[1]], contents_kg=list(PRIOR_BOX), total_fill_ml=None)
    return facts, cdef, truth, prior


def _cam(spec):
    import mujoco
    c = mujoco.MjvCamera()
    c.lookat[:] = spec["lookat"]; c.distance, c.azimuth, c.elevation = spec["distance"], spec["azimuth"], spec["elevation"]
    return c


def cmd_render(a):
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    import mujoco, imageio.v2 as iio
    from world_model.v2.nero.contents import carry_cert as CC, scenes as SC
    from world_model.v2.nero.pan import scene as PS
    out = pathlib.Path(a.out); (out / "img").mkdir(parents=True, exist_ok=True)
    recs = []
    for task in a.tasks:
        cell = PRIOR_BOX if task == "box" else PRIOR_FILL
        for i in range(a.n_per_task):
            seed = int(np.random.SeedSequence([a.seed, TASKS.index(task), i]).generate_state(1)[0])
            w = CC.sample_world(task, cell, np.random.default_rng(seed))
            facts, cdef, truth, prior = facts_and_truth(task, w)
            m = PS.apply_sim_defaults(SC.build_spec(w).compile())
            d = mujoco.MjData(m)
            SC.settle(m, d, 0.5)
            r = mujoco.Renderer(m, 480, 640)
            for view, spec in VIEWS.items():
                r.update_scene(d, camera=_cam(spec))
                p = out / "img" / f"{task}_{i:03d}_{view}.png"
                iio.imwrite(p, r.render())
                recs.append(dict(id=f"{task}_{i:03d}_{view}", task=task, view=view, seed=seed, image=str(p), truth=truth, prior=prior,
                                 facts=facts, contents_def=cdef, visible=dict(fill=False, contents=False,
                                 note="mug = solid cylinder, fill = hidden mass-only geom; box closed")))
            r.close()
    with open(out / "records.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(recs)} records -> {out/'records.jsonl'}")


# ------------------------------------------------------------------ parsing
def parse_last_json(text, finish=None):
    """last JSON object in text; None if the reply was cut at the token cap (finish == 'length') or holds no object."""
    if finish == "length" or not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    text = text.rsplit("</think>", 1)[-1]
    dec, best, i = json.JSONDecoder(), None, 0
    while True:
        j = text.find("{", i)
        if j < 0:
            break
        try:
            o, k = dec.raw_decode(text, j)
            if isinstance(o, dict):
                best = o
            i = k
        except ValueError:
            i = j + 1
    return best


def _iv(x):
    try:
        lo, hi = float(x[0]), float(x[1])
    except (TypeError, ValueError, IndexError, KeyError):
        return None
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return None
    return [min(lo, hi), max(lo, hi)]


def clean(o):
    if not isinstance(o, dict):
        return None
    tm = _iv(o.get("total_mass_kg"))
    if tm is None:
        return None
    pc = o.get("p_cover")
    try:
        pc = float(pc)
    except (TypeError, ValueError):
        pc = None
    return dict(total_mass_kg=tm, total_fill_ml=_iv(o.get("total_fill_ml")), contents_kg=_iv(o.get("contents_kg")), p_cover=pc,
                evidence=str(o.get("evidence", ""))[:200])


# ------------------------------------------------------------------ ask
def cmd_ask(a):
    recs = [json.loads(l) for l in open(a.records)]
    if a.limit:
        recs = recs[:a.limit]
    if a.mock:
        chat = _mock_chat
    else:
        from world_model.v2.vlm.openai_backend import OpenAIChat
        cl = OpenAIChat(a.model, base_url=a.endpoint, timeout_s=a.timeout)
        def chat(r, text):
            res = cl.run_one(dict(id=r["id"], images=[r["image"]], text=text, thinking=None if a.thinking is None else (a.thinking == "on")), a.max_tokens)
            return res["raw_texts"][0], res["finish"][0], res["seconds"]
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if getattr(a, "resume", False) and pathlib.Path(a.out).exists():
        done = {json.loads(l)["id"] for l in open(a.out) if l.strip() and json.loads(l).get("parsed")}
        keep = [l for l in open(a.out) if l.strip() and json.loads(l)["id"] in done]
        pathlib.Path(a.out).write_text("".join(keep))
    else:
        pathlib.Path(a.out).write_text("")
    todo = [r for r in recs if r["id"] not in done]

    def one(r):
        if a.prompt == "v2":
            text = PROMPT_V2["tray" if r["task"].startswith("tray") else "box"].format(facts=r["facts"])
        else:
            text = PROMPT.format(facts=r["facts"], cdef=r["contents_def"])
        t0 = time.time()
        try:
            raw, fin, sec = chat(r, text)
            err = None
        except Exception as e:
            raw, fin, sec, err = "", "error", time.time() - t0, repr(e)
        o = parse_last_json(raw, fin) if a.prompt == "v2" else clean(parse_last_json(raw, fin))
        if a.prompt == "v2":
            o = clean(compose_v2(r["task"], o)) if o else None
        return dict(id=r["id"], task=r["task"], view=r["view"], model=a.model, prompt_version=(PROMPT_V2_VERSION if a.prompt == "v2" else PROMPT_VERSION), thinking=a.thinking,
                    raw=raw, finish=fin, parsed=o, v2_fields=(parse_last_json(raw, fin) if a.prompt == "v2" else None), latency_s=round(sec, 3), error=err, truth=r["truth"], prior=r["prior"])

    from concurrent.futures import ThreadPoolExecutor
    with open(a.out, "a") as f, ThreadPoolExecutor(max(1, getattr(a, "workers", 1))) as ex:
        for row in ex.map(one, todo):                   # ex.map keeps record order
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row["id"], "parsed" if row["parsed"] else "NO ANSWER", row["finish"], flush=True)


def _mock_chat(r, text):
    """canned reply: centre = truth*0.9 (biased low), +-30%.  Deterministic from the id; no model, no network."""
    t = r["truth"]["total_mass_kg"]
    c = 0.9 * t
    out = dict(total_mass_kg=[round(c * 0.7, 3), round(c * 1.3, 3)], total_fill_ml=None if t is None or r["task"] == "box" else [0, 400],
               contents_kg=[round(r["truth"]["contents_kg"] * 0.6, 3), round(r["truth"]["contents_kg"] * 1.4, 3)], p_cover=0.9, evidence="mock")
    return "thinking... {\"x\": 1}\n" + json.dumps(out), "stop", 0.0


def cmd_mock_server(a):
    from http.server import BaseHTTPRequestHandler, HTTPServer
    class H(BaseHTTPRequestHandler):
        def log_message(self, *x): pass
        def _send(self, o):
            b = json.dumps(o).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers(); self.wfile.write(b)
        def do_GET(self): self._send({"data": [{"id": "mock"}]})
        def do_POST(self):
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            reply = dict(total_mass_kg=[0.9, 2.5], total_fill_ml=[0, 300], contents_kg=[0.5, 2.0], p_cover=0.9, evidence="mock server")
            self._send({"choices": [{"message": {"content": json.dumps(reply)}, "finish_reason": "stop"}], "usage": {}})
    print(f"mock endpoint on http://127.0.0.1:{a.port}/v1", flush=True)
    HTTPServer(("127.0.0.1", a.port), H).serve_forever()


# ------------------------------------------------------------------ scoring
def winkler(lo, hi, y, alpha=0.1):
    s = hi - lo
    if y < lo:
        s += 2 / alpha * (lo - y)
    elif y > hi:
        s += 2 / alpha * (y - hi)
    return s


def conformal_k(ivs, ys, cov=0.9):
    """smallest k >= 0 with centre +- k*halfwidth covering >= cov of ys (zero-width intervals: k = inf unless centre == y)."""
    need = []
    for (lo, hi), y in zip(ivs, ys):
        c, h = (lo + hi) / 2, (hi - lo) / 2
        need.append(0.0 if abs(y - c) == 0 else (math.inf if h == 0 else abs(y - c) / h))
    need = np.sort(need)
    n = len(need)
    j = int(math.ceil(cov * n)) - 1
    return float(need[min(max(j, 0), n - 1)])


def _stats(rows, key, alpha=0.1, seed=0):
    """rows: list of (iv or None, y).  Missing/None intervals count as NOT covering (no answer)."""
    n = len(rows)
    ok = [(iv, y) for iv, y in rows if iv is not None]
    out = dict(n=n, parse=len(ok) / n if n else float("nan"))
    if not ok:
        return out
    cov = np.array([iv[0] <= y <= iv[1] for iv, y in ok])
    rel = np.array([(iv[1] - iv[0]) / y if y > 0 else np.nan for iv, y in ok])
    out.update(cover=float(cov.sum() / n), cover_parsed=float(cov.mean()), w_med=float(np.nanmedian(rel)), w_max=float(np.nanmax(rel)),
               winkler=float(np.mean([winkler(iv[0], iv[1], y, alpha) for iv, y in ok])))
    if len(ok) >= 4:
        idx = np.random.default_rng(seed).permutation(len(ok)); h = len(ok) // 2
        A, B = [ok[i] for i in idx[:h]], [ok[i] for i in idx[h:]]
        k = conformal_k([x[0] for x in A], [x[1] for x in A], 1 - alpha)
        if math.isfinite(k):
            sc = [([(iv[0] + iv[1]) / 2 - k * (iv[1] - iv[0]) / 2, (iv[0] + iv[1]) / 2 + k * (iv[1] - iv[0]) / 2], y) for iv, y in B]
            out.update(conf_k=k, conf_cover=float(np.mean([a[0] <= y <= a[1] for a, y in sc])),
                       conf_w_med=float(np.nanmedian([(a[1] - a[0]) / y for a, y in sc if y > 0])))
        else:
            out.update(conf_k=k)
    return out


def _fmt(x, p=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.{p}f}" if isinstance(x, float) else str(x))


def cmd_score(a):
    groups = {}
    for fn in a.answers:
        for l in open(fn):
            r = json.loads(l)
            groups.setdefault((r["model"], r["task"]), []).append(r)
    lines = ["Coverage counts a missing answer as a miss (cover) ; cover_parsed = among parsed.  conf_* = split conformal (seeded half -> other half), 90%.",
             "Sample sizes are small: examples, not rates.  Truth/priors: see contents_eval.py header (fill invisible in the sim render).", "",
             "| model | task | key | n | parse | cover | cover_parsed | w_med | w_max | winkler | conf_k | conf_cover | conf_w_med |", "|" + "---|" * 13]
    def row(name, task, key, s):
        lines.append(f"| {name} | {task} | {key} | {s['n']} | {_fmt(s['parse'])} | {_fmt(s.get('cover'))} | {_fmt(s.get('cover_parsed'))} | {_fmt(s.get('w_med'))} | "
                     f"{_fmt(s.get('w_max'))} | {_fmt(s.get('winkler'),3)} | {_fmt(s.get('conf_k'))} | {_fmt(s.get('conf_cover'))} | {_fmt(s.get('conf_w_med'))} |")
    for (model, task), rs in sorted(groups.items()):
        for key in ("total_mass_kg", "contents_kg", "total_fill_ml"):
            rows = [((r["parsed"] or {}).get(key), r["truth"][key]) for r in rs if r["truth"].get(key) is not None]
            if rows:
                row(model, task, key, _stats(rows, key))
    for task in sorted({t for _, t in groups}):                    # model-free references, per task
        rs = next(v for (m, t), v in groups.items() if t == task)
        key = "total_mass_kg"
        row("prior (full box)", task, key, _stats([(r["prior"][key], r["truth"][key]) for r in rs], key))
        row("point-centre", task, key, _stats([([sum(r["prior"][key]) / 2] * 2, r["truth"][key]) for r in rs], key))
    txt = "\n".join(lines)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(txt + "\n")
    print(txt)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("render"); r.add_argument("--n-per-task", type=int, default=2); r.add_argument("--out", required=True)
    r.add_argument("--seed", type=int, default=0); r.add_argument("--tasks", nargs="+", default=list(TASKS))
    q = sp.add_parser("ask"); q.add_argument("--endpoint", default=None); q.add_argument("--model", default="mock"); q.add_argument("--records", required=True)
    q.add_argument("--out", required=True); q.add_argument("--max-tokens", type=int, default=4096); q.add_argument("--thinking", choices=["on", "off"], default=None)
    q.add_argument("--prompt", choices=["v1", "v2"], default="v1")
    q.add_argument("--workers", type=int, default=1); q.add_argument("--resume", action="store_true")
    q.add_argument("--limit", type=int, default=0); q.add_argument("--timeout", type=float, default=600.0); q.add_argument("--mock", action="store_true")
    s = sp.add_parser("score"); s.add_argument("--answers", nargs="+", required=True); s.add_argument("--out", required=True)
    ms = sp.add_parser("mock-server"); ms.add_argument("--port", type=int, default=8799)
    a = ap.parse_args(argv)
    {"render": cmd_render, "ask": cmd_ask, "score": cmd_score, "mock-server": cmd_mock_server}[a.cmd](a)


if __name__ == "__main__":
    main()
