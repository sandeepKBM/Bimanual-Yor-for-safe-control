"""adapters/geom_probe_run.py -- run a geometry probe request against the sim and write the READINGS into a GeometrySet.  SIMULATION ONLY.   (11-19)

Pipeline: Codex's request {probe, entity} -> sim target name (explicit mapping; Phase 8 will derive it from Codex's 2D boxes) -> geom_probe.lab_clouds (four lab cameras,
depth noise `noise`) -> geom_probe.measure_box -> one stack.geometry.Reading per OBSERVED field whose (target, field) has an error-table entry.

Fail closed: a field with no table entry gets NO reading (never a default error); a field the cameras did not observe gets none; a failed fit gives none.
Segmentation = the privileged sim segmentation (ASSUMED stand-in until Phase 8).  err_table = {target: {field: half_width}} (MEASURED once the noise study of
nero/results/geom_noise_2026-11-19/ exists; today the caller supplies it, e.g. a test table).
"""
from __future__ import annotations

import json
import math

from world_model.v2.stack import geometry as G

PROBES = ("measure_box", "measure_support")


def err_table_from_study(path):
    """read the owner's study JSON {target: {field: {"max": half_width, "n": int}}} -> (err_table {target: {field: half_width}}, n_table {target: {field: n}})."""
    with open(path) as fh:
        raw = json.load(fh)
    errs, ns = {}, {}
    for tgt, fl in raw.items():
        errs[tgt], ns[tgt] = {}, {}
        for f, rec in fl.items():
            if f not in G.FIELDS:
                raise ValueError(f"study {path}: {tgt}: unknown field {f!r}")
            errs[tgt][f] = float(rec["max"])
            ns[tgt][f] = rec.get("n")
    return errs, ns


def _observed_fields(res):
    """-> {field: (value, observed, faces_used)} from a geom_probe.measure_box result: the 'fields' form, or the flat form (fit_ok gates all seven)."""
    if isinstance(res.get("fields"), dict):
        return {f: (float(v["value"]), bool(v.get("observed", True)), v.get("faces_used")) for f, v in res["fields"].items() if f in G.FIELDS}
    if not res.get("fit_ok"):
        return {}
    return {f: (float(res[f]), True, None) for f in G.FIELDS if f in res}


def run_probe(request, m, d, world, gs, err_table, noise="clean", rng=None, mapping=None, n_table=None, clouds=None):
    """run one probe request; write the readings into `gs` and return them.
    request: dict(probe="measure_box"|"measure_support", entity=<Codex's name>, target=<sim entity name> optional; else mapping[entity]); optional support_top_z (m).
    err_table {target: {field: half_width}}; n_table {target: {field: n}} (optional, goes into Reading.n).  clouds: reuse a lab_clouds result (else rendered here)."""
    from world_model.v2.nero.contents import geom_probe as GP
    probe, entity = request.get("probe"), request.get("entity")
    if probe not in PROBES:
        raise ValueError(f"unknown probe {probe!r}; have {PROBES}")
    if entity not in gs.entities:
        raise ValueError(f"probe entity {entity!r} is not in the geometry set (have {sorted(gs.entities)})")
    target = request.get("target") or (mapping or {}).get(entity)
    if not target:
        raise ValueError(f"no sim target for entity {entity!r} (pass request['target'] or mapping)")
    ids = GP.entity_geoms(m, target)
    if clouds is None:
        clouds = GP.lab_clouds(m, d, world, noise=noise, rng=rng)
    res = GP.measure_box(clouds, ids, support_top_z=request.get("support_top_z"))
    tab, ntab = (err_table or {}).get(target, {}), (n_table or {}).get(target, {})
    nm = f"lab_clouds noise={noise if isinstance(noise, str) else 'custom'}; {GP.SEGMENTATION}"
    out = []
    for f, (val, observed, faces) in _observed_fields(res).items():
        if not observed or not math.isfinite(val) or f not in tab:     # no table entry = no reading (fail closed)
            continue
        note = f"{probe} on sim target {target!r}; n_points={res.get('n_points')}, n_cams={res.get('n_cams')}" + (f"; faces_used={faces}" if faces is not None else "")
        r = G.Reading(probe=probe, value=val, err=float(tab[f]), guarantee="sampled", noise_model=nm, n=ntab.get(f), source="MEASURED", note=note)
        gs.add_reading(entity, f, r)
        out.append(r)
    return out
