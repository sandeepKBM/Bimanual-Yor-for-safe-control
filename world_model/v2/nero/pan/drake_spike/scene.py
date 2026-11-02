"""Drake build of the covered-pan scene: pan/lid/puck/stove/table rebuilt from MuJoCo geom data as compliant hydroelastic
primitives; optional robot loaded from the (OBJ-rewritten) MJCF via Drake's MJCF parser. Offline, CPU only.

Inputs prepared in the scratch dir by prep.py/strip.py (STL->OBJ copies of the MJCF + robot_only.xml with pan/lid/hidden/
stove/table-top stripped).  ASSUMED material values: see common.py and PAN_* below.
"""
import numpy as np
import mujoco
from pydrake.all import (RigidTransform, RotationMatrix, Quaternion, SpatialInertia, UnitInertia, RotationalInertia, Box, Cylinder,
                         Parser, RoleAssign, PdControllerGains, Simulator, Context)
from world_model.v2.nero.pan.drake_spike.common import make_plant, hydro_props

SCR = "/common/home/ss5772/.tmp/claude-1905239669/-common-users-ss5772-bimanual/0fdf97c2-ab12-45ef-8d3a-31b6454afd0a/scratchpad/drake"
PAN_RES = 0.02            # ASSUMED mesh resolution hint for pan-scene compliant primitives (m)
PAN_MODULUS = 5e6         # ASSUMED hydroelastic modulus (Pa), same as unit tests
STATIC_MODULUS = 5e6      # ASSUMED stove/table modulus (Pa)


def _geoms(m, body_id):
    out = []
    for gi in range(m.ngeom):
        if m.geom_bodyid[gi] == body_id and m.geom_contype[gi] + m.geom_conaffinity[gi] > 0:
            out.append(gi)
    return out


def _shape(m, gi):
    t = m.geom_type[gi]; s = m.geom_size[gi]
    if t == mujoco.mjtGeom.mjGEOM_BOX: return Box(2 * s[0], 2 * s[1], 2 * s[2])
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER: return Cylinder(s[0], 2 * s[1])
    raise ValueError(t)


def _pose(m, gi):
    q = m.geom_quat[gi]
    return RigidTransform(RotationMatrix(Quaternion(q[0], q[1], q[2], q[3])), m.geom_pos[gi])


def build(dt=0.005, approx="sap", with_robot=False, hydro_fingers=True, walls_hydro=True, gravity=True):
    mm = mujoco.MjModel.from_xml_path(SCR + "/scene_obj.xml")
    b, plant, sg = make_plant(dt, approx, gravity=gravity)
    info = {}
    if with_robot:
        parser = Parser(plant)
        parser.AddModels(SCR + "/robot_only.xml")
    mi = plant.AddModelInstance('pan_scene')
    # static world geoms: stove pad and table top (compliant boxes, welded)
    for nm in ("stove_pad", "pan_table_top"):
        gi = mujoco.mj_name2id(mm, mujoco.mjtObj.mjOBJ_GEOM, nm)
        sz = mm.geom_size[gi]
        body = plant.AddRigidBody(nm, mi, SpatialInertia(1.0, np.zeros(3), UnitInertia.SolidBox(*(2 * sz))))
        plant.WeldFrames(plant.world_frame(), body.body_frame(), RigidTransform(mm.geom_pos[gi]))
        plant.RegisterCollisionGeometry(body, RigidTransform(), Box(*(2 * sz)), nm + "_col",
                                        hydro_props(float(mm.geom_friction[gi][0]), modulus=STATIC_MODULUS, res=PAN_RES))
    bodies = {}
    for nm in ("pan", "lid", "hidden"):
        bid = mm.body(nm).id
        I = np.zeros((3, 3)); 
        R = np.zeros(9); mujoco.mju_quat2Mat(R, mm.body_iquat[bid]); R = R.reshape(3, 3)
        Ic = R @ np.diag(mm.body_inertia[bid]) @ R.T
        M = SpatialInertia.MakeFromCentralInertia(float(mm.body_mass[bid]), mm.body_ipos[bid].copy(),
                                                  RotationalInertia(Ic[0, 0], Ic[1, 1], Ic[2, 2], Ic[0, 1], Ic[0, 2], Ic[1, 2]))
        body = plant.AddRigidBody(nm, mi, M)
        bodies[nm] = body
        for gi in _geoms(mm, bid):
            gname = mujoco.mj_id2name(mm, mujoco.mjtObj.mjOBJ_GEOM, gi)
            is_wall = gname.startswith("pan_wall")
            hy = walls_hydro or not is_wall
            fr = float(mm.geom_friction[gi][0])
            if hy:
                plant.RegisterCollisionGeometry(body, _pose(mm, gi), _shape(mm, gi), gname,
                                                hydro_props(fr, modulus=PAN_MODULUS, res=PAN_RES))
            else:
                from pydrake.all import CoulombFriction
                plant.RegisterCollisionGeometry(body, _pose(mm, gi), _shape(mm, gi), gname, CoulombFriction(fr, fr))
    if with_robot and hydro_fingers:
        # replace gripper/finger MESH collision props by rigid hydroelastic (rigid-vs-compliant pairs are supported)
        from pydrake.all import AddRigidHydroelasticProperties, ProximityProperties, AddContactMaterial, CoulombFriction
        n = 0
        for i in range(plant.num_bodies()):
            bd = plant.get_body(__import__("pydrake.all", fromlist=["BodyIndex"]).BodyIndex(i))
            if "finger" in bd.name() or "gripper_base" in bd.name():
                for gid in plant.GetCollisionGeometriesForBody(bd):
                    p = ProximityProperties(); AddRigidHydroelasticProperties(p)
                    AddContactMaterial(dissipation=1.0, point_stiffness=None, friction=CoulombFriction(1.0, 1.0), properties=p)
                    sg.AssignRole(plant.get_source_id(), gid, p, RoleAssign.kReplace); n += 1
        info["n_rigid_hydro_gripper_geoms"] = n
    return b, plant, sg, bodies, mm, info


def _unit_about_com(Ic, mass):
    # UnitInertia from a full rotational inertia about the COM divided by mass
    G = Ic / mass
    return UnitInertia(G[0, 0], G[1, 1], G[2, 2], G[0, 1], G[0, 2], G[1, 2])
