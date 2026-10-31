"""tamp/spec_default.py -- the HAND-WRITTEN constraint spec equal to today's plan-time rows (tamp/plan_cert.py), in the IR dict format a VLM would emit.  SIMULATION ONLY.

    spec = default_spec(world, people)                      # walls="grid" (step 1): EDT wall rows stay in checks.py;  people / cup are IR atoms
    spec = default_spec(world, people, walls="zono", zonos=Zs)   # step 2: Clear(footprint_wall, obs_j, d_wall) atoms from perceived zonotopes
    spec.as_dict()  ==  as_dict(spec)                       # {"spec": <tree>, "params": {...}, "rows": {...}, "n_people": n, "n_obstacles": m}  (json-serialisable)
    spec.node                                                # the ir.Node = ir.from_dict(spec.d["spec"], entities, params)  (the SAME builder a VLM spec goes through)

THE SPEC (names refer to the entity catalogue of spec_cert.make_entities; numbers are params):
    and(
      G[0,T] and_i( clear_egress(footprint_people, person_i, d_person) ),   # prediction disc of every person: Clear with the separating-motion exit clause
      G[0,T] and_j( clear(footprint_wall, obs_j, d_wall) ),                 # step 2 only (walls="zono")
      G[0,T] upright(tray, theta_max),                                      # the carried tray stays level
      G[0,T] inside(cup, shrink(tray_top, r_cup)) )                         # the cup stays on the tray
G[0,T] = ir.G(., 0, None): over every trajectory sample.  d_person, d_wall = the belief's buffers; theta_max = ShieldCfg.theta_tray_ir; r_cup = ShieldCfg.r_cup.
NOT in the spec (IR cannot express them; spec_cert keeps them from checks.py): cup acceleration bound, hardware slew / speed caps, and (walls="grid") the EDT wall rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from world_model.v2.tray_rep.sets import ir

from . import spec_cert as SC
from .plan_cert import world_people


def _G(of):
    return {"G": {"lo": 0, "hi": None, "of": of}}


def build_dict(n_people: int, n_obstacles: int = 0, walls: str = "grid") -> dict:
    """the plain-dict spec tree (see the module docstring)."""
    parts = []
    if n_people:
        pe = [{"clear_egress": {"a": "footprint_people", "b": f"person_{i}", "d": "d_person", "far": 1e9, "tag": f"person_{i}"}} for i in range(n_people)]
        parts.append(_G(pe[0] if n_people == 1 else {"and": pe}))
    if walls == "zono" and n_obstacles:
        ob = [{"clear": {"a": "footprint_wall", "b": f"obs_{j}", "d": "d_wall", "exact": True, "far": 0.4, "tag": f"obs_{j}"}} for j in range(n_obstacles)]
        parts.append(_G(ob[0] if n_obstacles == 1 else {"and": ob}))
    parts.append(_G({"upright": {"a": "tray", "theta_max": "theta_max", "tag": "tray_upright"}}))
    parts.append(_G({"inside": {"a": "cup", "b": {"shrink": {"of": "tray_top", "by": "r_cup"}}, "margin": 0.0, "tag": "cup_on_tray"}}))
    return {"and": parts}


@dataclass
class Spec:
    """a bound spec: the dict, its params, the entity catalogue and the ir tree.  walls: "grid" (EDT rows from checks.py) | "zono" (IR atoms).
    gen(n_people) rebuilds the default spec for another number of people (the person atoms are one per person); a Spec made by from_dict has no gen."""
    d: dict
    params: dict
    ents: dict
    node: object
    walls: str = "grid"
    n_people: int = 0
    n_obstacles: int = 0
    gen: object = None
    rows: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return as_dict(self)

    def for_people(self, n: int):
        if n == self.n_people:
            return self
        if self.gen is None:
            raise ValueError(f"spec is written for {self.n_people} people, the belief has {n}")
        return self.gen(n)


def as_dict(spec: Spec) -> dict:
    """json-serialisable form: {"spec": tree, "params": {...}, "rows": {...}, "n_people": n, "n_obstacles": m}."""
    return {"spec": spec.d, "params": dict(spec.params), "rows": dict(spec.rows), "n_people": spec.n_people, "n_obstacles": spec.n_obstacles}


def from_dict(dd: dict, ents: dict, walls: str | None = None) -> Spec:
    """a Spec from the as_dict form (e.g. json.loads of what a VLM wrote) and an entity catalogue (spec_cert.make_entities)."""
    node = ir.from_dict(dd["spec"], ents, dd["params"])
    w = walls or ("grid" if dd.get("rows", {}).get("walls") == "grid_edt" else "zono")
    return Spec(dd["spec"], dict(dd["params"]), ents, node, w, int(dd.get("n_people", 0)), int(dd.get("n_obstacles", 0)), None, dict(dd.get("rows", {})))


def default_spec(world, people=None, walls: str = "grid", zonos=(), d_wall: float | None = None, d_person: float | None = None) -> Spec:
    """the hand-written spec equal to today's plan-time rows.  people: [(c, v_hat), ...] (only their number matters here; None: world.bel_people).  zonos: perceived obstacle
    zonotopes (tray_rep Zonotope, or (c, G) pairs), world frame -- used when walls == "zono"."""
    assert walls in ("grid", "zono")
    zonos = list(zonos) if walls == "zono" else []
    b = world.bel0
    sc = world.shield_cfg
    params = dict(d_person=float(b.d_person if d_person is None else d_person), d_wall=float(b.d_wall if d_wall is None else d_wall),
                  theta_max=float(getattr(sc, "theta_tray_ir", 0.10)), r_cup=float(sc.r_cup))
    ext = world.items["tray"].extent
    tray_half = (0.5 * float(ext[0]), 0.5 * float(ext[1]))

    def make(n):
        ents = SC.make_entities(n, zonos, b.fp, tray_half)
        dd = build_dict(n, len(zonos), walls)
        rows = {"walls": "grid_edt" if walls == "grid" else "ir", "cup": "checks.check1 (a_cup, hw)", "people": "ir clear_egress"}
        return Spec(dd, params, ents, ir.from_dict(dd, ents, params), walls, n, len(zonos), make, rows)

    return make(len(world_people(world) if people is None else people))


def dumps(spec: Spec) -> str:
    return json.dumps(as_dict(spec), indent=1)
