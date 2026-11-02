"""Robot-side API probe: M(q), bias, Jacobian timing; mimic coupler; PD-held arms in the full scene (speed)."""
import time, json, numpy as np
from pydrake.all import (Simulator, JacobianWrtVariable, JointIndex, PdControllerGains, RigidTransform, JointActuatorIndex)
from world_model.v2.nero.pan.drake_spike.scene import build
from world_model.v2.nero.pan.drake_spike.passive import pose_init

out = {}
b, plant, sg, bodies, mm, info = build(0.005, with_robot=True)
# mimic: MJCF <equality joint polycoef 0 -1 0 0 0> => q_j2 = -q_j1 ; Drake coupler constraint (discrete SAP)
for side in ("left", "right"):
    j1 = plant.GetJointByName(f"{side}_arm_gripper_joint1"); j2 = plant.GetJointByName(f"{side}_arm_gripper_joint2")
    plant.AddCouplerConstraint(j2, j1, -1.0, 0.0)
# actuators: PD position servos from MJCF gains (kp 200/100, kd 20/10; tau limit 100)
gains = {1: (200, 20), 2: (200, 20), 3: (200, 20), 4: (100, 10), 5: (100, 10), 6: (100, 10), 7: (100, 10)}
n_act = 0
for side in ("left", "right"):
    for k in range(1, 8):
        j = plant.GetJointByName(f"{side}_arm_joint{k}")
        a = plant.AddJointActuator(f"{side}_a{k}", j, 100.0)   # effort limit 100 Nm (MODEL: YOR forcerange)
        a.set_controller_gains(PdControllerGains(p=gains[k][0], d=gains[k][1])); n_act += 1
plant.Finalize()
diagram = b.Build(); sim = Simulator(diagram)
ctx = sim.get_mutable_context(); pctx = plant.GetMyMutableContextFromRoot(ctx)
pose_init(plant, pctx, bodies, mm)
q0 = np.zeros(plant.num_positions()); 
plant.SetPositions(pctx, plant.GetModelInstanceByName if False else plant.GetPositions(pctx))
# desired = current arm positions (hold); desired state port: [q_act, v_act]
na = plant.num_actuated_dofs()
plant.get_desired_state_input_port(plant.GetModelInstanceByName("robot_only") if False else plant.GetBodyByName("left_arm_link0").model_instance()).FixValue(
    pctx, np.zeros(2 * na)) if False else None
out["nq"] = plant.num_positions(); out["nv"] = plant.num_velocities(); out["n_actuated"] = na
# dynamics API timing
N = 200
t0 = time.time()
qq = plant.GetPositions(pctx).copy()
for i in range(N): plant.SetPositions(pctx, qq + 1e-9 * i); M = plant.CalcMassMatrix(pctx)   # perturb q to defeat the cache
out["CalcMassMatrix_ms_incl_SetPositions"] = 1e3 * (time.time() - t0) / N; out["M_shape"] = list(M.shape)
t0 = time.time()
for i in range(N): plant.SetPositions(pctx, qq + 1e-9 * i); bias = plant.CalcBiasTerm(pctx) if hasattr(plant, "CalcBiasTerm") else None
out["CalcBiasTerm_ms"] = 1e3 * (time.time() - t0) / N
fr = plant.GetFrameByName("left_arm_gripper_base"); t0 = time.time()
for i in range(N): plant.SetPositions(pctx, qq + 1e-9 * i); J = plant.CalcJacobianSpatialVelocity(pctx, JacobianWrtVariable.kV, fr, np.zeros(3), plant.world_frame(), plant.world_frame())
out["CalcJacobianSpatialVelocity_ms"] = 1e3 * (time.time() - t0) / N; out["J_shape"] = list(J.shape)
out["has_CalcGravityGeneralizedForces"] = hasattr(plant, "CalcGravityGeneralizedForces")
# held-pose speed: PD actuators hold desired = initial q for 2 s
try:
    arms_q = {}
    from pydrake.all import ModelInstanceIndex
    mi = plant.GetBodyByName("left_arm_link0").model_instance()
    port = plant.get_desired_state_input_port(mi)
    port.FixValue(pctx, np.zeros(2 * plant.num_actuated_dofs(mi)))
    t0 = time.time(); sim.AdvanceTo(2.0); w = time.time() - t0
    out["pd_hold_full_scene_wall_per_sim_s"] = w / 2.0
    qa = plant.GetPositions(pctx)
    out["pd_hold_arm_q_maxabs_after_2s"] = float(np.abs(plant.GetPositions(pctx, mi)).max())
except Exception as e:
    out["pd_hold_error"] = str(e)[:300]
print(json.dumps(out, indent=1))
