"""Scene geometry as a structure Codex fills with PRIORS and the probes fill with READINGS (pure python, no physics).

Rulings (USER 11-19): Codex establishes the scene geometry itself (no scene numbers in its first prompt); it gives PRIORS,
then asks for PROBES; the checker uses ONLY what readings (MEASURED) or the robot model (MODEL) established. Intervals
first (one Interval per field), no factor graph.

Fail-closed rule: `require` / `box_hull` never return a prior. A field with no reading, or whose readings contradict each
other, raises GeometryMissing. A reading outside the prior is kept (the set follows the readings) and logged as a
prior-contradiction, shown to Codex later, never discarded.

Tags: priors are ASSUMED (Codex's guesses, never used to narrow `set`); every Reading carries its own source / guarantee /
noise_model / n. This module holds no hidden-truth fields.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field as dc_field

from world_model.v2.stack.theta import Interval

FIELDS = ("cx", "cy", "cz", "sx", "sy", "sz", "yaw")     # m, m, m (box centre height), m, m, m (full sizes), rad about +z
KINDS = ("object", "support", "obstacle")
ROLES = ("carried", "goal_support")
SOURCES = ("MEASURED", "MODEL")
GUARANTEES = ("sampled", "proof")
STATUSES = ("ok", "missing", "contradiction")


class GeometryMissing(Exception):
    """a field the checker needs has no reading, or its readings contradict."""


@dataclass(frozen=True)
class Reading:
    probe: str
    value: float
    err: float                  # half-width, >= 0
    guarantee: str              # "sampled" | "proof"
    noise_model: str
    n: "int | None" = None      # sample count behind err
    source: str = "MEASURED"    # "MEASURED" | "MODEL"
    note: str = ""

    def __post_init__(self):
        if not (math.isfinite(self.value) and math.isfinite(self.err) and self.err >= 0):
            raise ValueError(f"reading {self.probe!r}: need finite value and err >= 0 (got {self.value}, {self.err})")
        if self.guarantee not in GUARANTEES:
            raise ValueError(f"unknown guarantee {self.guarantee!r}; have {GUARANTEES}")
        if self.source not in SOURCES:
            raise ValueError(f"unknown reading source {self.source!r}; have {SOURCES}")

    @property
    def interval(self) -> Interval:
        return Interval(self.value - self.err, self.value + self.err)

    def to_json(self) -> dict:
        return {"probe": self.probe, "value": self.value, "err": self.err, "guarantee": self.guarantee,
                "noise_model": self.noise_model, "n": self.n, "source": self.source, "note": self.note}

    @staticmethod
    def from_json(d) -> "Reading":
        return Reading(probe=d["probe"], value=float(d["value"]), err=float(d["err"]), guarantee=d["guarantee"],
                       noise_model=d.get("noise_model", ""), n=d.get("n"), source=d.get("source", "MEASURED"),
                       note=d.get("note", ""))


@dataclass
class Field:
    prior: "Interval | None" = None       # ASSUMED (Codex); never narrows `set`
    set: "Interval | None" = None         # intersection of all reading intervals
    source: "str | None" = None           # "MODEL" only if every reading is MODEL, else "MEASURED"
    readings: list = dc_field(default_factory=list)
    why: str = ""
    status: str = "missing"

    def to_json(self) -> dict:
        return {"prior": self.prior.to_json() if self.prior else None, "set": self.set.to_json() if self.set else None,
                "source": self.source, "readings": [r.to_json() for r in self.readings], "why": self.why,
                "status": self.status}

    @staticmethod
    def from_json(d) -> "Field":
        return Field(prior=Interval.from_json(d["prior"]) if d.get("prior") else None,
                     set=Interval.from_json(d["set"]) if d.get("set") else None, source=d.get("source"),
                     readings=[Reading.from_json(r) for r in d.get("readings", [])], why=d.get("why", ""),
                     status=d.get("status", "missing"))


@dataclass
class Entity:
    name: str
    kind: str
    roles: set = dc_field(default_factory=set)
    fields: dict = dc_field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"entity {self.name!r}: kind {self.kind!r} not in {KINDS}")
        bad = set(self.roles) - set(ROLES)
        if bad:
            raise ValueError(f"entity {self.name!r}: unknown roles {sorted(bad)}; have {ROLES}")
        self.roles = set(self.roles)
        for f in FIELDS:
            self.fields.setdefault(f, Field())


class GeometrySet:
    def __init__(self):
        self.entities: dict = {}
        self.contradictions: list = []         # reading-vs-reading (fail closed)
        self.prior_contradictions: list = []   # reading outside prior (kept, shown to Codex)

    # ------------------------------------------------------------------ build
    def add_entity(self, name: str, kind: str, roles=()) -> Entity:
        if name in self.entities:
            raise ValueError(f"entity {name!r} already exists")
        e = Entity(name, kind, set(roles))
        self.entities[name] = e
        return e

    def _field(self, entity: str, fname: str) -> Field:
        if entity not in self.entities:
            raise KeyError(f"unknown entity {entity!r}")
        if fname not in FIELDS:
            raise ValueError(f"unknown field {fname!r}; have {FIELDS}")
        return self.entities[entity].fields[fname]

    def set_prior(self, entity: str, fname: str, lo: float, hi: float, why: str = "") -> None:
        """ASSUMED prior (Codex's guess). Stored only; never narrows `set`."""
        f = self._field(entity, fname)
        f.prior = Interval(float(lo), float(hi))
        f.why = why
        for r in f.readings:                      # prior set AFTER readings: same disjointness rule as add_reading
            self._note_prior_clash(entity, fname, f, r)

    def _note_prior_clash(self, entity: str, fname: str, f: Field, r: Reading) -> None:
        """a reading contradicts the prior when its interval is DISJOINT from the prior interval. Recorded, never
        discarded; the set is unchanged (priors never narrow)."""
        if f.prior is None or f.prior.intersect(r.interval) is not None:
            return
        rec = {"entity": entity, "field": fname, "prior": f.prior.to_json(), "reading": r.to_json()}
        if rec not in self.prior_contradictions:
            self.prior_contradictions.append(rec)

    def add_reading(self, entity: str, fname: str, reading: Reading) -> None:
        f = self._field(entity, fname)
        iv = reading.interval
        f.readings.append(reading)
        self._note_prior_clash(entity, fname, f, reading)
        if f.status == "contradiction":
            return                                    # stays contradicted (fail closed); reading is recorded
        if f.set is None:
            f.set, f.status = iv, "ok"
        else:
            new = f.set.intersect(iv)
            if new is None:
                prev = [r.to_json() for r in f.readings[:-1]]
                self.contradictions.append({"entity": entity, "field": fname, "set_before": f.set.to_json(),
                                            "earlier_readings": prev, "reading": reading.to_json()})
                f.set, f.status = None, "contradiction"
            else:
                f.set = new
        f.source = "MODEL" if all(r.source == "MODEL" for r in f.readings) else "MEASURED"

    # ------------------------------------------------------------------ use
    def require(self, entity: str, fname: str) -> Interval:
        f = self._field(entity, fname)
        if f.status == "contradiction":
            raise GeometryMissing(f"{entity}.{fname}: readings contradict (fail closed)")
        if f.set is None:
            raise GeometryMissing(f"{entity}.{fname}: no reading (a prior is never used)")
        return f.set

    def box_hull(self, entity: str) -> dict:
        return {f: self.require(entity, f) for f in FIELDS}

    def entities_with_role(self, role: str) -> list:
        return [e for e in self.entities.values() if role in e.roles]

    def assumptions(self) -> list:
        out = []
        for e in self.entities.values():
            for fn, f in e.fields.items():
                if f.prior is not None:
                    out.append({"entity": e.name, "field": fn, "tag": "ASSUMED", "kind": "prior",
                                "prior": f.prior.to_json(), "why": f.why})
                for r in f.readings:
                    out.append({"entity": e.name, "field": fn, "tag": r.source, "kind": "reading", "probe": r.probe,
                                "guarantee": r.guarantee, "noise_model": r.noise_model, "n": r.n})
        return out

    # ------------------------------------------------------------------ json
    def to_json(self) -> dict:
        return {"entities": [{"name": e.name, "kind": e.kind, "roles": sorted(e.roles),
                              "fields": {k: f.to_json() for k, f in e.fields.items()}}
                             for e in self.entities.values()],
                "contradictions": self.contradictions, "prior_contradictions": self.prior_contradictions}

    @staticmethod
    def from_json(d) -> "GeometrySet":
        g = GeometrySet()
        for ed in d["entities"]:
            e = g.add_entity(ed["name"], ed["kind"], ed.get("roles", ()))
            for k, fd in ed.get("fields", {}).items():
                if k not in FIELDS:
                    raise ValueError(f"unknown field {k!r}")
                e.fields[k] = Field.from_json(fd)
        g.contradictions = list(d.get("contradictions", []))
        g.prior_contradictions = list(d.get("prior_contradictions", []))
        return g

    def codex_view(self) -> dict:
        """JSON-able view for the prompt: priors, sets, status, readings (no hidden truth exists here)."""
        ents = []
        for e in self.entities.values():
            fl = {}
            for k, f in e.fields.items():
                fl[k] = {"prior": f.prior.to_json() if f.prior else None, "set": f.set.to_json() if f.set else None,
                         "status": f.status, "source": f.source,
                         "readings": [{"probe": r.probe, "value": r.value, "err": r.err, "guarantee": r.guarantee,
                                       "n": r.n} for r in f.readings]}
            ents.append({"name": e.name, "kind": e.kind, "roles": sorted(e.roles), "fields": fl})
        return {"entities": ents, "contradictions": self.contradictions,
                "prior_contradictions": self.prior_contradictions}


def schema_for_prompt() -> str:
    return (
        "Describe the scene as a list of entities. Every entity is an UPRIGHT ORIENTED BOX with 7 fields: "
        "cx, cy (box centre, metres, world frame), cz (box centre HEIGHT, metres), sx, sy, sz (FULL sizes along the box "
        "axes, metres), yaw (radians about +z).\n"
        f"kind is one of {list(KINDS)}. roles (optional list) from {list(ROLES)}: 'carried' = the object to carry, "
        "'goal_support' = the surface it must end on. Choose entity names yourself (e.g. 'crate', 'bookcase_lower_floor').\n"
        "Give each field a PRIOR interval [lo, hi] and a one-line reason. Priors are your guesses from the photos; give a "
        "reason for each; they are never used by the checker (only probe readings and the robot model are). A field you "
        "cannot guess may be omitted.\n"
        'Reply JSON: {"entities":[{"name":str,"kind":str,"roles":[str],"priors":{"cx":[lo,hi],...},'
        '"why":{"cx":str,...}}]}'
    )


def parse_priors(obj) -> GeometrySet:
    """Build a GeometrySet from Codex's JSON reply (dict or JSON string). ValueError on any malformed input."""
    if isinstance(obj, str):
        obj = json.loads(obj)
    if not isinstance(obj, dict) or not isinstance(obj.get("entities"), list):
        raise ValueError("reply must be an object with an 'entities' list")
    g = GeometrySet()
    for i, ed in enumerate(obj["entities"]):
        if not isinstance(ed, dict):
            raise ValueError(f"entities[{i}] is not an object")
        name = ed.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"entities[{i}]: missing name")
        kind = ed.get("kind")
        if kind is None:
            raise ValueError(f"entity {name!r}: missing kind")
        g.add_entity(name, kind, ed.get("roles", []))        # kind/roles validated by Entity
        priors, why = ed.get("priors", {}) or {}, ed.get("why", {}) or {}
        for k in list(priors) + list(why):
            if k not in FIELDS:
                raise ValueError(f"entity {name!r}: unknown field {k!r}; have {FIELDS}")
        for k, iv in priors.items():
            try:
                lo, hi = float(iv[0]), float(iv[1])
                if len(iv) != 2:
                    raise TypeError
            except (TypeError, ValueError, IndexError):
                raise ValueError(f"entity {name!r} field {k!r}: prior must be [lo, hi] numbers (got {iv!r})")
            if math.isnan(lo) or math.isnan(hi):
                raise ValueError(f"entity {name!r} field {k!r}: NaN in prior")
            if lo > hi:
                raise ValueError(f"entity {name!r} field {k!r}: lo > hi ({lo} > {hi})")
            g.set_prior(name, k, lo, hi, str(why.get(k, "")))
    return g
