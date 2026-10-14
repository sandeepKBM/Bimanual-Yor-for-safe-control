"""people_track/source.py -- the people SOURCE switch of the runtime shield.  SIMULATION ONLY.

    make_shield(sim, source="truth")    -> teleop.shield.Shield on sim.people_state (the original, bit-identical)
    make_shield(sim, source="tracker")  -> LookShield-capable PerceivedShield: head-camera detections -> KF tracks -> (conformal or k-sigma) inflated
                                           keep-out; optional look-before-move glance (`look=True`).  `sim.people_state` stays the GROUND TRUTH and is
                                           used for evaluation metrics only; the shield sees `PerceptionStack.proxy(sim)`.
Entry points: TeleopSim.make_shield(...) (teleop/sim.py), teleop.run --people tracker, flagship.simulate --people tracker, people_track.ps_run --cond.
"""
from __future__ import annotations

from dataclasses import replace

from world_model.v2.house.teleop import shield as SH

PEOPLE_SOURCES = ("truth", "tracker")


def load_conformal(c):
    if c is None or c is False:
        return None
    from .conformal import ConformalMargin
    if isinstance(c, ConformalMargin):
        return c
    return ConformalMargin.load(c)


def make_shield(sim, source="truth", cfg=None, perception="depth", look=False, look_cfg=None, conformal=None, ucfg=None, seed=0):
    """cfg: ShieldCfg (e.g. replace(ShieldCfg(), **PEOPLE_V2)).  perception: run_pt condition string ('depth', 'seg', 'param:0.1:3:0.0', with the +flags of
    run_pt.parse_cond).  look: enable the glance (LookCfg overrides via look_cfg).  conformal: ConformalMargin | path to its json | None (k-sigma)."""
    assert source in PEOPLE_SOURCES, source
    if source == "truth":
        return SH.Shield(cfg, occ=sim.occ, a_hw=sim.a_hw)
    from . import perception as PC, shield_pt as SPT
    from .run_pt import parse_cond
    from .look import Looker, LookCfg
    pcfg, u0 = parse_cond(perception)
    assert pcfg is not None, "perception='truth' is the truth source; use source='truth'"
    u = ucfg or u0
    cm = load_conformal(conformal)
    if cm is not None:
        u = replace(u, conformal=cm, soft_floor=0.05 if u.soft_floor is None else u.soft_floor)
    perc = PC.PerceptionStack(sim, replace(pcfg, seed=seed))
    sh = SPT.PerceivedShield(perc, cfg, occ=(sim.occ_ctrl if getattr(sim, "loc", None) is not None else sim.occ), a_hw=sim.a_hw, ucfg=u)
    if look:
        sh.look = Looker(perc, look_cfg or LookCfg())
    return sh
