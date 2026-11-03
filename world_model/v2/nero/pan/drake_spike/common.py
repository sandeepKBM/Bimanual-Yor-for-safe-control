"""Shared Drake helpers for the contact feasibility spike (offline, CPU only).

All numeric material values below are ASSUMED (no measured data for the real pads/pan):
  HYDRO_MODULUS  = 5e6 Pa   # ASSUMED hydroelastic modulus (Pa) for every compliant body
  DISSIPATION    = 1.0 s/m  # ASSUMED Hunt-Crossley dissipation
  RES_HINT       = 0.01 m   # ASSUMED mesh resolution hint for compliant primitives
"""
import numpy as np
from pydrake.all import (
    AddContactMaterial, Box, Cylinder, CoulombFriction, DiagramBuilder, DiscreteContactApproximation,
    AddMultibodyPlantSceneGraph, ProximityProperties, RigidTransform, SpatialInertia,
    UnitInertia, RotationMatrix, Simulator, ContactModel, MultibodyPlantConfig, ApplyMultibodyPlantConfig,
    AddRigidHydroelasticProperties, AddCompliantHydroelasticProperties, Parser, PrismaticJoint, JointActuator,
)

HYDRO_MODULUS = 5e6   # ASSUMED
DISSIPATION = 1.0     # ASSUMED
RES_HINT = 0.01       # ASSUMED


def hydro_props(mu, modulus=HYDRO_MODULUS, dissipation=DISSIPATION, res=RES_HINT, rigid=False):
    p = ProximityProperties()
    if rigid:
        AddRigidHydroelasticProperties(res, p)
    else:
        AddCompliantHydroelasticProperties(res, modulus, p)
    AddContactMaterial(dissipation=dissipation, point_stiffness=None, friction=CoulombFriction(mu, mu), properties=p)
    return p


def make_plant(dt, approx="sap", gravity=True, model="hydroelastic_with_fallback"):
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(b, dt)
    plant.set_contact_model({"hydroelastic_with_fallback": ContactModel.kHydroelasticWithFallback,
                             "hydroelastic": ContactModel.kHydroelastic,
                             "point": ContactModel.kPoint}[model])
    plant.set_discrete_contact_approximation(
        {"sap": DiscreteContactApproximation.kSap, 
         "lagged": DiscreteContactApproximation.kLagged, "similar": DiscreteContactApproximation.kSimilar}[approx])
    if not gravity:
        plant.mutable_gravity_field().set_gravity_vector([0, 0, 0])
    return b, plant, sg


def add_box_body(plant, name, size, mass, mu, pose=None, welded=False, hydro=True, modulus=HYDRO_MODULUS, res=RES_HINT):
    """Box body (full side lengths `size`), mass `mass`, compliant hydroelastic collision geom."""
    M = SpatialInertia(mass, np.zeros(3), UnitInertia.SolidBox(*size))
    body = plant.AddRigidBody(name, M) if not welded else plant.AddRigidBody(name, M)
    if welded:
        plant.WeldFrames(plant.world_frame(), body.body_frame(), pose or RigidTransform())
    props = hydro_props(mu, modulus=modulus, res=res) if hydro else None
    geom = Box(*size)
    if props is not None:
        plant.RegisterCollisionGeometry(body, RigidTransform(), geom, name + "_col", props)
    else:
        plant.RegisterCollisionGeometry(body, RigidTransform(), geom, name + "_col", CoulombFriction(mu, mu))
    return body
