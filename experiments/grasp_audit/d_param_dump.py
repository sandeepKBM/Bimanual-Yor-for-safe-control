"""Part D: dump every grasp/EE-relevant parameter of the compiled cup-collect model
(real-grasp variant: restore_finger_object_collision=True). SIMULATION ONLY."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.cup_collect_task_utils import load_yor_cup_collect_model  # noqa: E402

model, data = load_yor_cup_collect_model(restore_finger_object_collision=True)
o = model.opt
print("opt: timestep", o.timestep, "integrator", int(o.integrator), "cone", int(o.cone), "(0=pyr,1=ell)",
      "impratio", o.impratio, "noslip_iter", o.noslip_iterations, "solver", int(o.solver),
      "iterations", o.iterations, "tolerance", o.tolerance, "gravity", o.gravity)
print("enableflags", o.enableflags, "disableflags", o.disableflags)


def geom_row(name):
    g = model.geom(name)
    return (f"{name:28s} type={int(g.type[0])} size={np.round(g.size,4)} fric={np.round(g.friction,4)} "
            f"condim={int(g.condim[0])} solref={np.round(g.solref,4)} solimp={np.round(g.solimp,4)} "
            f"margin={g.margin[0]:.4f} gap={g.gap[0]:.4f} priority={int(g.priority[0])} solmix={g.solmix[0]:.2f} "
            f"contype={int(g.contype[0])} conaff={int(g.conaffinity[0])} body={model.body(int(g.bodyid[0])).name}")


for side in ("left", "right"):
    for gname in (f"{side}_arm_link7_pad", f"{side}_arm_link8_pad"):
        print(geom_row(gname))
cup = model.body("cup").id
print("cup body mass", model.body_mass[cup], "inertia", model.body_inertia[cup], "subtree_mass",
      model.body_subtreemass[cup])
for gi in range(model.ngeom):
    if model.geom_bodyid[gi] == cup:
        nm = model.geom(gi).name or f"geom{gi}"
        print(" CUP", geom_row(nm) if model.geom(gi).name else f"geom{gi} fric={model.geom_friction[gi]} "
              f"condim={model.geom_condim[gi]} size={model.geom_size[gi]} type={model.geom_type[gi]} "
              f"solref={model.geom_solref[gi]} solimp={model.geom_solimp[gi]} priority={model.geom_priority[gi]}")
balls = [b for b in range(model.nbody) if model.body(b).name.startswith("ball_")]
print("n balls", len(balls), "ball mass", model.body_mass[balls[0]] if balls else None,
      "total", sum(model.body_mass[b] for b in balls))
cubes = [b for b in range(model.nbody) if model.body(b).name.startswith("cube_")]
for b in cubes[:1]:
    print("cube", model.body(b).name, "mass", model.body_mass[b])
    for gi in range(model.ngeom):
        if model.geom_bodyid[gi] == b:
            print("  cube geom fric", model.geom_friction[gi], "size", model.geom_size[gi], "condim",
                  model.geom_condim[gi], "solref", model.geom_solref[gi], "solimp", model.geom_solimp[gi])
for side in ("left", "right"):
    for j in (7, 8):
        a = model.actuator(f"{side}_arm_joint{j}_pos")
        jn = model.joint(f"{side}_arm_joint{j}")
        print(f"act {side} j{j}: gain={a.gainprm[:3]} bias={a.biasprm[:3]} forcerange={a.forcerange} "
              f"ctrlrange={a.ctrlrange} | joint range={jn.range} damping={model.dof_damping[jn.dofadr[0]]} "
              f"armature={model.dof_armature[jn.dofadr[0]]} frictionloss={model.dof_frictionloss[jn.dofadr[0]]}")
    for j in range(1, 7):
        jn = model.joint(f"{side}_arm_joint{j}")
        d = jn.dofadr[0]
        a = model.actuator(f"{side}_arm_joint{j}_pos")
        print(f" {side} j{j}: range={np.round(jn.range,3)} damp={model.dof_damping[d]} arm={model.dof_armature[d]} "
              f"fricloss={model.dof_frictionloss[d]} act gain={a.gainprm[0]} bias={a.biasprm[:3]} "
              f"forcerange={a.forcerange} actgravcomp={int(model.jnt_actgravcomp[jn.id])}")
for b in range(model.nbody):
    if model.body_gravcomp[b] != 0:
        print("gravcomp body", model.body(b).name, model.body_gravcomp[b])
for side in ("left", "right"):
    ee = model.body(f"{side}_arm_gripper_base").id
    print(side, "EE subtree mass", model.body_subtreemass[ee])
    for nm in ("link7", "link8"):
        bid = model.body(f"{side}_arm_{nm}").id
        print(f"  {side}_{nm} mass", model.body_mass[bid])
