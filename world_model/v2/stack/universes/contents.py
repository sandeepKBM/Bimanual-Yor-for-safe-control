"""stack/universes/contents.py -- ContentsUniverse: the hidden-contents carry tasks (tray2/tray4/tray6/box) as a stack Universe.  SIMULATION ONLY.

Worlds are EXACTLY the certificate's: carry_cert.sample_world + carry_cert.cell_seed (set "cert"), tilt_sweep.corners (set "corners").
build() still returns the legacy CarryEpisode (sim.episode) while the planner/executor migration is in progress.
"""
from __future__ import annotations

import numpy as np

from ..types import Bound, Sim, TaskSpec, ThetaBox, WorldSpec

TASKS = ("tray2", "tray4", "tray6", "box")
SETS = ("cert", "corners")


def _imports():
    from ...nero.contents import carry as CR
    from ...nero.contents import carry_cert as CC
    from ...nero.contents import objects as OB
    from ...nero.contents import scenes as SC
    from ...nero.contents import tilt_sweep as TS
    return CR, CC, OB, SC, TS


class ContentsUniverse:
    name = "contents"

    def __init__(self, tasks=TASKS):
        self.tasks_ = tuple(tasks)
        bad = [t for t in self.tasks_ if t not in TASKS]
        if bad:
            raise ValueError(f"unknown contents tasks {bad}; have {TASKS}")

    # ------------------------------------------------------------------ prior box
    def theta_box(self, task=None) -> ThetaBox:
        """tags copied from the source code's own comments (objects.py / scenes.py / carry_cert.py)."""
        CR, CC, OB, SC, TS = _imports()
        task = task or self.tasks_[0]
        f = {}
        if task.startswith("tray"):
            w = SC.TrayWorld(n_cups=int(task[4:]))
            hx = w.tray_lx / 2 - w.rim_t - w.cup_D / 2 - 0.002
            hy = w.tray_ly / 2 - w.rim_t - w.cup_D / 2 - 0.002
            f["n_cups"] = Bound(w.n_cups, w.n_cups, "ASSUMED", "objects.py N_CUPS: the number of cups is visible (count known)")
            f["fill_kg"] = Bound(0.0, OB.CUP["fill_max_kg"], "ASSUMED", "objects.CUP fill_max_kg per mug (cells: carry_cert.TRAY_FILL_CELLS)", "kg")
            f["cup_x"] = Bound(-hx, hx, "MODEL", "scenes.sample_tray_world: tray inner rim - D/2 - 2 mm (objects.TRAY dims MODEL)", "m")
            f["cup_y"] = Bound(-hy, hy, "MODEL", "scenes.sample_tray_world: tray inner rim - D/2 - 2 mm (objects.TRAY dims MODEL)", "m")
            f["cup_mass"] = Bound(w.cup_mass, w.cup_mass, "ASSUMED", "objects.CUP empty: ceramic mug 300 g", "kg")
            f["cup_mu"] = Bound(w.cup_mu, w.cup_mu, "ASSUMED", "scenes.TrayWorld cup_mu: ceramic on ceramic glaze")
            f["tray_mu"] = Bound(w.tray_mu, w.tray_mu, "ASSUMED", "scenes.TrayWorld tray_mu")
        else:
            w = SC.BoxWorld()
            hx = w.box_outer[0] / 2 - w.wall_t - 0.04
            hy = w.box_outer[1] / 2 - w.wall_t - 0.06
            f["contents_kg"] = Bound(OB.BOX_CONTENTS_KG[0], OB.BOX_CONTENTS_KG[1], "ASSUMED", "objects.BOX_CONTENTS_KG (user-approved 11-02)", "kg")
            f["contents_x"] = Bound(-hx, hx, "MODEL", "carry_cert.sample_world: box interior - wall - 0.04 (crate dims ASSUMED, scenes.BoxWorld)", "m")
            f["contents_y"] = Bound(-hy, hy, "MODEL", "carry_cert.sample_world: box interior - wall - 0.06 (crate dims ASSUMED, scenes.BoxWorld)", "m")
            f["contents_mu"] = Bound(w.contents_mu, w.contents_mu, "ASSUMED", "scenes.BoxWorld contents_mu")
            f["box_mu"] = Bound(w.box_mu, w.box_mu, "ASSUMED", "scenes.BoxWorld box_mu: cardboard")
        f["counter_mu"] = Bound(w.counter_mu, w.counter_mu, "ASSUMED", "scenes.Common counter_mu (= table_mu of the pan scene)")
        return ThetaBox(f)

    def cells(self, task):
        CR, CC, OB, SC, TS = _imports()
        return [c for (c,) in CC.cells(task)]

    # ------------------------------------------------------------------ worlds
    def worlds(self, set_name: str, n: int | None = None, seed: int = 0) -> list[WorldSpec]:
        """set "cert": for every task and cell, n worlds with carry_cert.sample_world(task, cell, default_rng(carry_cert.cell_seed(task, cell, i)))
        -- identical to the certificate runs (`seed` is NOT used: the cert seeds are fixed by (task, cell, i)).
        set "corners": tilt_sweep.corners(task, cell) for every cell (deterministic; n and seed unused)."""
        CR, CC, OB, SC, TS = _imports()
        if set_name not in SETS:
            raise ValueError(f"unknown world set {set_name!r}; have {SETS}")
        out = []
        for task in self.tasks_:
            for cell in self.cells(task):
                cell = tuple(cell)
                cid = f"{task}/cell({cell[0]},{cell[1]})"
                if set_name == "cert":
                    for i in range(int(n if n is not None else 20)):
                        s = CC.cell_seed(task, cell, i)
                        w = CC.sample_world(task, cell, np.random.default_rng(s))
                        out.append(WorldSpec(self.name, f"{cid}/i{i}", dict(CR._theta(w), task=task, cell=list(cell), i=i), s, w))
                else:
                    for name, w in TS.corners(task, cell):
                        out.append(WorldSpec(self.name, f"{cid}/{name}", dict(CR._theta(w), task=task, cell=list(cell), corner=name), 0, w))
        return out

    # ------------------------------------------------------------------ sim
    def build(self, world: WorldSpec, **kw) -> Sim:
        """legacy path: sim.episode = carry.CarryEpisode(world.native, **kw) (kw: a_plan, m_hat, ctl, fric_comp, weld_solref, ...)."""
        CR, *_ = _imports()
        ep = CR.CarryEpisode(world.native, **kw)
        return Sim(ep.m, ep.d, world, handles=dict(obj=ep.obj, addr=ep.addr, dofs=ep.dofs, cups=list(ep.cups)), episode=ep)

    def tasks(self, world: WorldSpec) -> list[TaskSpec]:
        CR, CC, OB, SC, TS = _imports()
        task = world.theta["task"]
        disp = np.sum([np.asarray(dl, float) for _, dl in CR.PLANS["tray" if task.startswith("tray") else "box"]], axis=0)
        return [TaskSpec(
            "carry", "obj",
            goal=dict(region="counter" if task.startswith("tray") else "shelf", displacement_m=disp.tolist(),
                      note="PLANS (ASSUMED, user-confirmed 11-02): common TCP delta lift+carry+place; success = carry.in_R at the end"),
            limits=dict(obj_tilt_deg=Bound(0, CR.R_TILT_OBJ, "ASSUMED", "carry.R_TILT_OBJ (rest region R)", "deg"),
                        cup_tilt_deg=Bound(0, CR.R_TILT_CUP, "ASSUMED", "carry.R_TILT_CUP (rest region R)", "deg"),
                        obj_speed=Bound(0, CR.R_SPEED, "ASSUMED", "carry.R_SPEED (rest region R)", "m/s"),
                        u_tilt_obj_deg=Bound(0, CR.U_TILT_OBJ, "ASSUMED", "carry.U_TILT_OBJ (unsafe set U)", "deg"),
                        u_tilt_cup_deg=Bound(0, CR.U_TILT_CUP, "ASSUMED", "carry.U_TILT_CUP (unsafe set U)", "deg")))]

    def producers(self, mode: str = "truth"):
        from .. import registry
        if mode != "truth":
            raise NotImplementedError(f"contents producers: only 'truth' exists (no perception in the carry sim); got {mode!r}")
        return [registry.get("producers", "truth")()]

    def success(self, sim: Sim, task: TaskSpec) -> dict:
        ok, why = sim.episode.in_R()          # carry.CarryEpisode.in_R (rest region R); not re-derived
        return dict(ok=bool(ok), R_fail=list(why))

    def assumptions(self) -> dict:
        CR, CC, OB, SC, TS = _imports()
        from ...nero import config as C
        return dict(
            theta={t: self.theta_box(t).assumptions() for t in self.tasks_},
            cells=f"tray fill cells {CC.TRAY_FILL_CELLS} ASSUMED; box cells {CC.BOX_CELLS[0]}..{CC.BOX_CELLS[-1]} kg in 0.5 kg steps ASSUMED",
            grasp="ASSUMED rigid weld (user 11-01)", plans="carry.PLANS ASSUMED (user-confirmed 11-02)",
            a_nom=f"nero.contents.envelope.A_NOM ASSUMED", rest_region=dict(tilt_obj=CR.R_TILT_OBJ, tilt_cup=CR.R_TILT_CUP, speed=CR.R_SPEED, tag="ASSUMED"),
            physics=C.carry_physics(), worlds="theta ~ uniform on the cell (carry_cert.sample_world); seeds carry_cert.cell_seed")
