"""pick/planner.py -- single vs bimanual decision: physical RULE (assign.py), optional VLM prior, hard VETO.  SIMULATION ONLY.

  planner = "rule"      : assign.assign() only.
  planner = "vlm"       : the VLM's choice is used as is (research baseline; NEVER use on a robot -- it can relax a physical limit).
  planner = "vlm+veto"  : the VLM chooses, the rule is authoritative about FEASIBILITY: a VLM plan that is infeasible (width > usable jaw opening for 1 arm, mass above
                          the per-arm payload for 1 arm, hands cannot fit / span for 2 arms, the chosen arm cannot reach) is replaced by the rule's plan.  The VLM can only
                          choose inside the feasible set; it can never relax a limit (mirrors lang's mu_policy="conservative").
The VLM is asked in a WORKER thread (never the 50 Hz sim thread), cached per (object id, extent/centroid bucket); a down / slow (timeout) / unparseable VLM falls back to the
rule and says so (`fallback`).  Every request is logged (raw text, parsed choice, rule choice, final choice, latency) for the disagreement analysis.

Backends (lang/backend.py): "mock" (default; MockVLM with the pick rule below -- deterministic, NO network / daemon), "live" (the owner-launched file-queue daemon
world_model.v2.vlm.service; only when explicitly requested), "auto" (live if its status says up, else mock).  Nothing here contacts a daemon unless backend in ("live","auto").

VLM output (strict JSON):  {"arms": 1|2, "arm": "left"|"right"|"both", "grasp": "pinch"|"side"|"squeeze", "mass_kg": [lo, hi] (a bare float = [m, m]),
                            "contents": "liquid"|"loose"|"solid"|"empty"|"unknown", "fragile": bool, "handle": bool, "why": "<short>"}
The veto uses mass HI only (a VLM mass can only raise the estimate); contents / fragile / handle are recorded, not yet used by any check.

veto_mode (OPT-IN; default "rule" = the behaviour above, byte-identical; env BIMANUAL_PICK_VETO_MODE can set it when the argument is None):
  "rule"  : a vetoed VLM plan is replaced by the rule's plan (which uses its own, narrower density-prior mass).
  "probe" : closes the VETO GAP.  Belief = mass in [min(prior, VLM lo), max(prior, VLM hi)] (the VLM can only WIDEN).  If the VLM plan is vetoed AND the rule's plan
            is not admissible on that whole belief, the rule's plan is NOT returned (nothing is admissible: evidence must narrow the set first); the decision is the
            Checker-4 output instead (checkers/probe4.py): a PROBE plan (mode "probe", `probe` = chosen probe + value) or an infeasible plan.  If the rule's plan is
            admissible on the widened belief (the veto was about geometry, not mass) the rule's plan is returned as before.  Extra decision keys: veto_mode, probe.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

from . import assign as A

SYSTEM = ("You plan grasps for a mobile manipulator with TWO 6-DoF arms (left, right), each with a parallel-jaw gripper. You see the robot's head-camera image with ONE object "
          "boxed. Decide whether ONE arm or BOTH arms should pick it up. Facts: the usable jaw opening is {open_cm:.0f} cm; one arm can lift roughly {pay:.2f} kg safely; two hands "
          "squeeze an object from opposite sides (needs a width between {minsep:.0f} and {maxspan:.0f} cm); the left arm is on the robot's left, the right arm on the right. "
          "The label comes from a detector and may be wrong: trust the image. Also report what only the image can tell: a plausible mass RANGE, what is "
          "inside, whether it is fragile, whether it has a handle. Answer with ONE JSON object and nothing else.")
USER = ("Boxed object: label \"{label}\". Measured from depth (robot frame): footprint {a_cm:.1f} x {b_cm:.1f} cm, height {h_cm:.1f} cm; centre x={x:.2f} m forward, y={y:+.2f} m "
        "({side} of the robot centre line).\nJSON schema: {{\"arms\": 1 or 2, \"arm\": \"left\"|\"right\"|\"both\", \"grasp\": \"pinch\"|\"side\"|\"squeeze\", "
        "\"mass_kg\": [low, high], \"contents\": \"liquid\"|\"loose\"|\"solid\"|\"empty\"|\"unknown\", \"fragile\": true|false, "
        "\"handle\": true|false, \"why\": \"<=15 words\"}}")
ASK_TOKENS = 220            # answer only; reasoning models get runner.EXTRA_TOKENS on top
REPAIR = ("Your last answer was not valid JSON for the schema. Reply with ONLY the JSON object, e.g. {\"arms\": 1, \"arm\": \"left\", \"grasp\": \"pinch\", "
          "\"mass_kg\": [0.2, 0.4], \"contents\": \"solid\", \"fragile\": false, \"handle\": false, \"why\": \"small\"}.")


def mock_pick_rules(policy="default"):
    """a deterministic MOCK VLM prior for the pick prompt: decides from the label words and the footprint it is told (no hidden state), deliberately NOT identical to the rule
    (it ignores jaw opening / payload numbers: 'large / wide / crate / box|basket|tray' -> two arms; small things -> one arm on the nearer side; it also over-trusts
    'bottle' etc.), so the veto path is exercised."""
    def pick(text, system, images, n):
        label = (re.search(r'label "([^"]*)"', text) or [None, ""])[1].lower()
        mm = re.search(r"footprint ([0-9.]+) x ([0-9.]+) cm, height ([0-9.]+) cm", text)
        a, b, h = (float(v) for v in mm.groups()) if mm else (10, 10, 10)
        yy = float((re.search(r"y=([+-][0-9.]+)", text) or [None, "0"])[1])
        two = any(k in label for k in ("crate", "basket", "tray", "big", "large", "heavy")) or (b > 12 and "object" in label) or a > 25
        if policy == "naive_single":        # adversarial mock: always one arm, ignores size / mass (the veto must catch the wide crate / basket)
            two = False
        elif policy == "naive_two":          # adversarial mock: always two arms (the veto must catch the small objects: hands cannot be that close)
            two = True
        side = "left" if yy > -0.21 else "right"
        if policy == "nano_basket" and any(k in label for k in ("crate", "basket", "tray")):     # Cosmos-Nano's observed basket answer: 2 arms, widened mass hi above even the 2-arm limit
            return json.dumps(dict(arms=2, arm="both", grasp="squeeze", mass_kg=[1.5, 2.5], why="large basket, both hands"))
        if two:
            return json.dumps(dict(arms=2, arm="both", grasp="squeeze", mass_kg=round(a * b * h * 0.0001, 2), why="wide object, use both hands"))
        return json.dumps(dict(arms=1, arm=side, grasp="pinch", mass_kg=round(a * b * h * 0.0003, 2), why="small object, one hand"))
    return [(r"Boxed object", pick)]


def _json_objects(text):
    """top-level balanced {...} substrings, last first."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0:
                out.append(text[start:i + 1])
    return out[::-1]


def parse_vlm_plan(text):
    """defensive JSON parse -> dict(arms, arm, grasp, mass_kg (= range hi), mass_range, contents, fragile, handle, why) or None."""
    if not text:
        return None
    j = None
    for cand in _json_objects(text):           # LAST complete object first: a reasoning model's answer comes after its reasoning
        for t in (cand, re.sub(r",\s*}", "}", cand.replace("'", '"'))):
            try:
                j = json.loads(t)
                break
            except Exception:      # noqa: BLE001
                pass
        if isinstance(j, dict) and ("arms" in j or "arm" in j):
            break
        j = None
    if j is None:
        return None
    try:
        arms = int(j.get("arms"))
    except Exception:      # noqa: BLE001
        arm0 = str(j.get("arm", "")).lower()
        arms = 2 if arm0 == "both" else 1 if arm0 in ("left", "right") else None
    if arms not in (1, 2):
        return None
    arm = str(j.get("arm", "both" if arms == 2 else "")).lower()
    if arms == 2:
        arm = "both"
    if arms == 1 and arm not in ("left", "right"):
        return None
    grasp = str(j.get("grasp", "squeeze" if arms == 2 else "pinch")).lower()
    mr = j.get("mass_kg")
    try:
        v = sorted(float(x) for x in mr) if isinstance(mr, (list, tuple)) else [float(mr)] if mr is not None else []
        mr = [v[0], v[-1]] if v else None
    except Exception:      # noqa: BLE001
        mr = None
    contents = str(j.get("contents", "unknown")).lower()
    flag = lambda k: j[k] if isinstance(j.get(k), bool) else None
    return dict(arms=arms, arm=arm, grasp=grasp, mass_kg=mr[1] if mr else None, mass_range=mr,
                contents=contents if contents in ("liquid", "loose", "solid", "empty") else "unknown", fragile=flag("fragile"), handle=flag("handle"),
                why=str(j.get("why", ""))[:160])


def veto(vlm, rule, est, reach=None):
    """VLM plan (dict from parse_vlm_plan) vs the rule's Plan.  -> (final Plan, vetoed: bool, reasons [str]).
    The VLM's plan survives only if every physical check passes; otherwise the rule's plan is returned."""
    a, b, h = est["extent"]
    mass = A.est_mass((a, b, h), vlm.get("mass_kg"))
    reasons = []
    if vlm["arms"] == 1:
        ok, why = A.single_feasible(b, mass)
        if not ok:
            reasons.append("1-arm infeasible: " + why)
        if vlm["arm"] not in ("left", "right"):
            reasons.append("1-arm plan without a valid arm")
        elif reach is not None and not reach.get(vlm["arm"], True):
            reasons.append(f"{vlm['arm']} arm cannot reach")
    else:
        w, ang = A.bimanual_axis((a, b), est.get("yaw", 0.0))
        if w is None:
            reasons.append(f"2-arm infeasible: no footprint axis in [{A.MIN_HAND_SEP_M*100:.0f}, {A.MAX_SPAN_M*100:.0f}] cm")
        if mass > 2 * A.SINGLE_MAX_KG:
            reasons.append(f"2-arm infeasible: mass est {mass:.2f} kg > {2*A.SINGLE_MAX_KG:.2f} kg")
    if reasons:
        return rule, True, reasons
    if vlm["arms"] == 1:
        return A.Plan("single", vlm["arm"], "pinch", f"VLM: {vlm['why']}", width=b, mass_est=mass), False, []
    w, ang = A.bimanual_axis((a, b), est.get("yaw", 0.0))
    return A.Plan("bimanual", "both", "squeeze", f"VLM: {vlm['why']}", width=w, mass_est=mass, axis=float(ang)), False, []


@dataclass
class ProbePlan(A.Plan):
    """veto_mode="probe": mode "probe" = do NOT grasp yet; run `probe` (name) to narrow the belief, then re-decide.  JSON-serialisable like Plan."""
    probe: str = ""
    probe_value: dict = None
    belief: dict = None


def _candidates(est, rule):
    """geometric options on the depth estimate (mass-free): the rule's plan first, then single (nearer arm), then bimanual."""
    from world_model.v2.house.checkers import probe4 as PB
    a, b, h = est["extent"]
    out = []
    if rule.mode != "infeasible":
        out.append(PB.Candidate(rule.mode, rule.arm, rule.width, rule.axis, "rule"))
    out.append(PB.Candidate("single", A.arm_for(est["centroid"][1]), b, 0.0, "geometry"))
    w, ang = A.bimanual_axis((a, b), est.get("yaw", 0.0))
    out.append(PB.Candidate("bimanual", "both", w if w is not None else 0.0, float(ang) if ang is not None else 0.0, "geometry"))
    seen, res = set(), []
    for c in out:
        if c.name not in seen:
            seen.add(c.name)
            res.append(c)
    return res


def probe_gate(vlm, rule, est, criterion="expected"):
    """the veto_mode="probe" branch for a VETOED VLM plan.  -> (Plan | ProbePlan, info dict | None).
    rule plan admissible on the VLM-widened belief -> (rule, None) [unchanged behaviour]; else Checker 4's decision."""
    from world_model.v2.house.checkers import probe4 as PB
    prior = A.est_mass(est["extent"])
    mr = vlm.get("mass_range") or ([vlm["mass_kg"], vlm["mass_kg"]] if vlm.get("mass_kg") is not None else [prior, prior])
    bel = PB.Belief(min(prior, mr[0]), max(prior, mr[1]))
    cands = _candidates(est, rule)
    rc = cands[0] if rule.mode != "infeasible" else None
    if rc is not None and PB.admissible(rc, bel):
        return rule, None
    d = PB.decide(bel, cands, criterion=criterion)
    info = dict(kind=d[0], belief=bel.to_json(), prior_mass=prior, candidates=[c.name for c in cands])
    if d[0] == "execute":          # some other candidate is admissible on the whole belief
        c = d[1]
        info["plan"] = c.name
        if c.mode == "single":
            return A.Plan("single", c.arm, "pinch", f"probe-gate: {c.name} admissible on the widened belief", width=c.width, mass_est=bel.m_hi), info
        return A.Plan("bimanual", "both", "squeeze", "probe-gate: bimanual admissible on the widened belief", width=c.width, mass_est=bel.m_hi, axis=c.axis), info
    if d[0] == "probe":
        pr, val = d[1], d[2]
        info.update(probe=pr.to_json(), value=val)
        why = (f"VLM plan vetoed and the rule plan is not admissible on the VLM-widened mass [{bel.m_lo:.2f}, {bel.m_hi:.2f}] kg: PROBE {pr.name} "
               f"(P(progress) {val['frac']:.2f}, net {val['net']:.2f}, {pr.cost_s:.0f} s) before any grasp")
        return ProbePlan("probe", "both" if pr.arms == 2 else "none", "none", why, width=rule.width, mass_est=bel.m_hi, probe=pr.name, probe_value=val, belief=bel.to_json()), info
    info["reason"] = d[1]
    return A.Plan("infeasible", "none", "none", f"VLM plan vetoed, rule plan inadmissible on the widened belief and no probe can help: {d[1]}", width=rule.width, mass_est=bel.m_hi), info


class GraspPlanner:
    def __init__(self, planner="rule", backend="mock", timeout_s=10.0, log=None, mock_rules=None, cache_dir=None, veto_mode=None):
        assert planner in ("rule", "vlm", "vlm+veto"), planner
        veto_mode = veto_mode or os.environ.get("BIMANUAL_PICK_VETO_MODE", "rule")
        assert veto_mode in ("rule", "probe"), veto_mode
        self.veto_mode = veto_mode
        self.planner, self.kind, self.timeout_s = planner, backend, timeout_s
        self.log = log or (lambda *a: None)
        self._be = None
        self._mock_rules = mock_rules
        self._cache_dir = cache_dir
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pickvlm")
        self._lock = threading.Lock()
        self._cache = {}                 # key -> record
        self._futs = {}                  # key -> Future
        self.records = []                # the JSON log (one record per VLM request, in order)

    # ------------------------------------------------------------------ backend (lazy; the live daemon is only touched for kind in live/auto)
    def backend(self):
        if self._be is None:
            from world_model.v2.house.lang import backend as BE
            if self.kind == "mock":
                self._be = BE.MockVLM(rules=self._mock_rules or mock_pick_rules(), use_default=False)
            else:
                self._be = BE.make_backend(self.kind, cache_dir=self._cache_dir, log=self.log)
        return self._be

    def backend_name(self):
        return getattr(self._be, "name", self.kind) if self._be is not None else self.kind

    @staticmethod
    def key(obj_id, est):
        a, b, h = est["extent"]
        c = est["centroid"]
        return (int(obj_id), round(a * 50), round(b * 50), round(h * 50), round(float(c[0]) * 20), round(float(c[1]) * 20))

    # ------------------------------------------------------------------ the VLM request (worker thread)
    def _ask(self, obj_id, label, est, image_path):
        t0 = time.time()
        a, b, h = est["extent"]
        c = est["centroid"]
        text = USER.format(label=label, a_cm=a * 100, b_cm=b * 100, h_cm=h * 100, x=float(c[0]), y=float(c[1]), side="left" if c[1] > A.CENTRE_Y else "right")
        system = SYSTEM.format(open_cm=A.SINGLE_MAX_WIDTH_M * 100, pay=A.SINGLE_MAX_KG, minsep=A.MIN_HAND_SEP_M * 100, maxspan=A.MAX_SPAN_M * 100)
        rec = dict(id=obj_id, label=label, t_unix=time.time(), backend=self.backend_name(), raw=[], parsed=None, error=None, repaired=False)
        try:
            be = self.backend()
            rec["backend"] = self.backend_name()
            o = be.ask(text, images=[image_path] if image_path else [], system=system, max_new_tokens=ASK_TOKENS, temperature=0.0)
            rec["raw"].append(o["texts"][0] if o.get("texts") else "")
            p = parse_vlm_plan(rec["raw"][-1])
            if p is None:
                rec["repaired"] = True
                o = be.ask(REPAIR, images=[image_path] if image_path else [], system=system, max_new_tokens=ASK_TOKENS, temperature=0.0)
                rec["raw"].append(o["texts"][0] if o.get("texts") else "")
                p = parse_vlm_plan(rec["raw"][-1])
            rec["parsed"] = p
        except Exception as e:      # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        rec["latency_s"] = round(time.time() - t0, 3)
        return rec

    def request(self, obj_id, label, est, image_path=None):
        """non-blocking: start the VLM query for this object unless cached / pending.  -> key."""
        k = self.key(obj_id, est)
        if self.planner == "rule":
            return k
        with self._lock:
            if k in self._cache or k in self._futs:
                return k
            self._futs[k] = self._ex.submit(self._ask, obj_id, label, est, image_path)
        return k

    def _collect(self, k):
        with self._lock:
            f = self._futs.get(k)
        if f is not None and f.done():
            rec = f.result()
            with self._lock:
                self._cache[k] = rec
                self._futs.pop(k, None)
                self.records.append(rec)
        return self._cache.get(k)

    def status(self, k):
        if self.planner == "rule":
            return "rule"
        rec = self._collect(k)
        if rec is not None:
            return "done"
        return "pending" if k in self._futs else "none"

    # ------------------------------------------------------------------ decision (pure, cheap: sim-thread safe)
    def decide(self, obj_id, est, reach=None):
        """-> dict(plan: Plan, rule: Plan, vlm: dict|None, source: 'rule'|'vlm'|'vlm+veto', vetoed, reasons, fallback, latency_s, status).  Never blocks."""
        rule = A.assign(est["extent"], est["centroid"], est.get("yaw", 0.0))
        out = dict(plan=rule, rule=rule, vlm=None, source="rule", vetoed=False, reasons=[], fallback=None, latency_s=None, status="rule")
        if self.planner == "rule":
            return out
        k = self.key(obj_id, est)
        rec = self._collect(k)
        if rec is None:
            out.update(status="pending", fallback="VLM answer pending: rule plan in use" if k in self._futs else "VLM not requested: rule plan in use")
            return out
        out.update(latency_s=rec["latency_s"], status="done")
        if rec["error"] or rec["parsed"] is None:
            out["fallback"] = rec["error"] or "unparseable VLM output: rule plan in use"
            return out
        if rec["latency_s"] > self.timeout_s:
            out["fallback"] = f"VLM latency {rec['latency_s']:.1f} s > timeout {self.timeout_s:.0f} s: rule plan in use"
            return out
        out["vlm"] = rec["parsed"]
        if self.planner == "vlm":
            vp = rec["parsed"]
            out["plan"] = A.Plan("single" if vp["arms"] == 1 else "bimanual", vp["arm"] if vp["arms"] == 1 else "both", "pinch" if vp["arms"] == 1 else "squeeze",
                                 f"VLM (no veto): {vp['why']}", width=est["extent"][1], mass_est=A.est_mass(est["extent"], vp.get("mass_kg")))
            out["source"] = "vlm"
            return out
        plan, vetoed, reasons = veto(rec["parsed"], rule, est, reach)
        out.update(plan=plan, source="vlm+veto", vetoed=vetoed, reasons=reasons)
        if self.veto_mode == "probe":
            out.update(veto_mode="probe", probe=None)
            if vetoed:
                gp, info = probe_gate(rec["parsed"], rule, est)
                if info is not None:
                    out.update(plan=gp, probe=info, source="vlm+veto+probe")
                    out["reasons"] = reasons + [gp.why]
        if rule.mode == "infeasible":      # the rule says nothing can be done: the veto already returned the rule's infeasible plan (vetoed True) or the VLM plan was feasible
            pass
        return out

    def agree(self, d):
        """VLM mode/arm vs rule mode/arm (for the agreement table)."""
        v = d.get("vlm")
        if v is None:
            return None
        r = d["rule"]
        vm = "single" if v["arms"] == 1 else "bimanual"
        return vm == r.mode and (r.mode != "single" or v["arm"] == r.arm)

    def close(self):
        self._ex.shutdown(wait=False, cancel_futures=True)
