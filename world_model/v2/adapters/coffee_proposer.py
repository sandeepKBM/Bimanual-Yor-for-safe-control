"""adapters/coffee_proposer.py -- (moved from stack/proposer/coffee.py 11-09: it reads the coffee scene, which stack engine may not import) the proposer for the coffee-tray task "tray0" (nero/contents/coffee.py).  SIMULATION ONLY.   (USER 11-09)

The plan is written by the proposer model (propose -> schema validation -> compile), ONE model spec for both methods (default codex:gpt-5.6-sol:high).
Extension of the skill vocabulary, nothing else of schema.py / prompt.py / compile.py is edited (this module patches them per process, `install()`):
  place_capsule()   NEW step skill, args {}: the LEFT arm picks the coffee capsule from its pad and puts it into the tray (rigid weld, ASSUMED), then both claws are
                    on the tray handles again.  It must be the FIRST step.  It has no recovery (it is not a tray motion); the capsule phase itself is the skill's.
  the remaining steps are the unchanged lift / move / place / release of the tray (both claws welded), compiled by compile.compile_plan on the steps after step 0.
Hidden parameters: variant "unknown": load_kg (mass of a hidden load inside the tray; sanity range 0..10 kg, NOT the prior); variant "known": none.
The prompt holds no prior box and no hidden truth.  The weigh probe is NOT chosen by the model: it is the fixed skill the stack runs by its own rule.
"""
from __future__ import annotations

from world_model.v2.stack.proposer import compile as PC
from world_model.v2.stack.proposer import prompt as PR
from world_model.v2.stack.proposer import schema as S

TASK = "tray0"
LOAD_SANITY = (0.0, 10.0)           # ASSUMED sanity range of the hidden load (like the box's 0..20 kg), not the prior
_DONE = False


def install():
    global _DONE
    if _DONE:
        return
    _DONE = True
    from world_model.v2.nero.contents import coffee as CF
    CF.install()
    S.TASKS = tuple(S.TASKS) + (TASK,)
    S.STEP_SKILLS["place_capsule"] = {}
    p0, po0, v0 = S.params, S.probe_offset, S.validate

    def params(task):
        if task != TASK:
            return p0(task)
        if CF.variant() == "known":
            return {}
        return {"load_kg": dict(range=LOAD_SANITY, unit="kg", probed=True, tag="ASSUMED sanity range",
                                doc="mass of an unknown load lying inside the tray (the tray and the capsule are known)")}
    S.params = params
    S.probe_offset = lambda task: ("load_kg", 0.0) if task == TASK else po0(task)

    def validate(d, task, supports=("counter", "shelf")):
        tp = v0(d, task, supports)
        if task == TASK:
            sk = [s.call.skill for s in tp.steps]
            errs = []
            if sk[0] != "place_capsule":
                errs.append("steps: the first step must be place_capsule()")
            if sk.count("place_capsule") != 1:
                errs.append(f"steps: exactly one place_capsule() allowed, found {sk.count('place_capsule')}")
            if errs:
                raise S.PlanRejected("schema", errs)
        return tp
    S.validate = validate
    ts0, sf0 = PR.task_spec, PR.scene_facts

    def task_spec(task, world=None):
        if task != TASK:
            return ts0(task, world)
        from world_model.v2.nero.contents import objects as OB, scenes as SC
        w = world if world is not None else CF.CoffeeWorld()
        top = w.counter_top
        sup = {"counter": dict(top=round(float(top), 4), x=[round(float(w.counter_centre[0] - w.counter_half[0]), 4), round(float(w.counter_centre[0] + w.counter_half[0]), 4)],
                               y=[round(float(w.counter_centre[1] - w.counter_half[1]), 4), round(float(w.counter_centre[1] + w.counter_half[1]), 4)])}
        sx, sy = round(float(w.obj_xy[0]), 4), round(float(w.obj_xy[1]), 4)
        goal = [round(sx + CF.FORWARD[0], 4), round(sy + CF.FORWARD[1], 4)]
        hidden = CF.variant() == "unknown"
        text = (f"Set up the coffee tray. First the left arm picks the coffee capsule (object 'capsule') and puts it into the tray (place_capsule). Then both arms hold "
                f"the tray by its two handles; lift it, carry it 0.10 m toward the robot (x decreases by 0.10, y unchanged) and set it down on the counter at "
                f"xy = ({goal[0]}, {goal[1]}) (tray reference point = centre of its underside), keeping the tray level so the capsule stays inside, then let go."
                + ("  There may be an unknown load lying inside the tray." if hidden else ""))
        obj = dict(name="coffee tray with a coffee capsule", footprint_xy_m=[OB.TRAY["lx"], OB.TRAY["ly"]], rim_height_m=OB.TRAY["rim_h"],
                   empty_mass_kg=round(CF.EMPTY_KG, 3), start_xy=[sx, sy], start_support="counter",
                   grasp="handles (tabs) on the +y and -y sides, centre %.4f m above the support" % SC.GRASP_Z,
                   capsule=dict(mass_kg=CF.CAP["mass"], radius_m=CF.CAP["r"], height_m=CF.CAP["h"], start_xy=[round(float(c), 4) for c in w.capsule_xy], on="a fixed pad",
                                goes_to="inside the tray, 0.06 m toward +y of the tray reference point"))
        return dict(task=task, instruction=text, start_xy=obj["start_xy"], goal_xy=goal, goal_support="counter", goal_top=sup["counter"]["top"], goal_tol_m=PR.GOAL_TOL_M,
                    supports=sup, object=obj)
    PR.task_spec = task_spec

    def scene_facts(task, world=None):
        f = sf0(task, world)
        if task == TASK:
            f["skill_notes"] = dict(place_capsule="NEW skill, no arguments, must be the FIRST step: the left arm takes the capsule from its pad and puts it into the tray, "
                                                   "then both claws are on the tray handles again (rigid grasp). It needs no recovery entry. After it, use lift / move / place / release "
                                                   "for the TRAY as before.")
        return f
    PR.scene_facts = scene_facts


def compile_coffee(plan, world=None, build=True):
    """plan: a status-ok plan dict (propose()) -> CompiledPlan of the tray steps (step 0 = place_capsule is stripped; its index is not in the segment names)."""
    install()
    if plan.get("status") != "ok":
        raise S.PlanRejected(plan.get("stage", "schema"), plan.get("reasons", ["not an ok plan"]))
    spec0 = PR.task_spec(TASK, world)
    tp = S.validate(dict(belief=plan["belief"], steps=plan["steps"], recoveries=plan["recoveries"], fallback=plan["fallback"]), TASK, tuple(spec0["supports"]))
    pid = plan.get("plan_id") or S.plan_id(tp.to_dict())
    tp.steps = tp.steps[1:]
    cp = PC.compile_plan(tp, TASK, world=world, build=build)
    cp.plan_id = pid
    return cp


def propose_coffee(backend_spec=None, cache_dir=None, offline=False, world=None):
    """-> (plan dict, record).  One proposer call per variant (cached)."""
    install()
    from world_model.v2.stack.proposer import propose as PP
    return PP.propose(TASK, backend_spec=backend_spec or PP.DEFAULT_SPEC, cache_dir=cache_dir, offline=offline, world=world)


def propose_coffee_until_accepted(backend_spec=None, cache_dir=None, offline=False, world=None, build=True):
    """propose -> compile_coffee; every rejection (schema / compile / planner) is fed back and the proposer is re-asked, no cap (USER 11-09).
    -> (plan, CompiledPlan or None, record, info) as propose.propose_until_accepted (None only after a proposer ERROR)."""
    install()
    from world_model.v2.stack.proposer import propose as PP
    return PP.propose_until_accepted(TASK, lambda pl: compile_coffee(pl, world=world, build=build), backend_spec=backend_spec or PP.DEFAULT_SPEC,
                                     cache_dir=cache_dir, offline=offline, world=world)
