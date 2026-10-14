"""experiments/residual_dynamics/perturbed_plant.py

Two MuJoCo models of the SAME scene (scenes/scene_yor_canonical.xml, via
experiments/ab_smooth_vs_bangbang/sim_common.load_model() -- unmodified on
disk, same load path the A/B/C experiment already uses):

  - NOMINAL model (`load_nominal()`): what a feedback-linearization (FL)
    controller "knows" -- used to compute M_nom(q) and
    c_nom = qfrc_bias - qfrc_passive, exactly as
    experiments/ab_smooth_vs_bangbang/mpc_fl.py's FLMPC.step does (lines
    ~511-519: `tau = M_e @ a + c_e` is the FL command intended to give
    q_ddot = a exactly on the model the controller believes it is driving).

  - PERTURBED plant (`make_perturbed_model()`): a stand-in for the real
    robot. Same kinematics (same joints/bodies/geometry), DIFFERENT
    dynamics, so the nominal FL cancellation is imperfect -- that
    imperfection (the "residual") is what this experiment learns.

Arm dof/qpos addressing reuses sim_common.joint_addrs (== the same
mj_name2id-based lookup scripts/make_cup_collect_safety_demo.py's
build_episode uses for left_qposadr/left_dofadr/right_qposadr/right_dofadr)
-- not reinvented here.

Perturbation (fixed, documented, single function `make_perturbed_model`):
  1. + PERTURB_FRICTIONLOSS_ADD_NM additional joint frictionloss on every
     arm dof (both arms' 6 joints each) -- dry/Coulomb friction, resolved by
     MuJoCo's constraint solver, NOT part of qfrc_passive/qfrc_bias, so a
     nominal-model FL controller (which only ever sees the UNPERTURBED
     frictionloss baked into m_nom) cannot cancel it.
  2. + PERTURB_DAMPING_ADD_NM_S_PER_RAD additional joint damping on every
     arm dof -- viscous, DOES show up in qfrc_passive, but at a magnitude
     the nominal model doesn't know about.
  3. * a per-arm-link body_mass scale factor drawn once (fixed seed
     PERTURB_MASS_SEED) i.i.d. ~ U[PERTURB_MASS_SCALE_LO, PERTURB_MASS_SCALE_HI]
     per link (sim_common.LINK_NAMES, both left_arm_/right_arm_ prefixes) --
     changes M(q) AND qfrc_bias (gravity/Coriolis depend on body_mass/inertia
     too), so even the "exact" M_nom @ a + c_nom cancellation is now wrong on
     the perturbed plant.

All perturbation magnitudes are named constants below (nothing is buried
inside a function body).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import mujoco

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SIM_COMMON_DIR = _REPO_ROOT / "experiments" / "ab_smooth_vs_bangbang"
if str(_SIM_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_SIM_COMMON_DIR))

import sim_common as sc  # noqa: E402  (experiments/ab_smooth_vs_bangbang, read-only import)

# ---------------------------------------------------------------------------
# Perturbation magnitudes -- ALL named constants, nothing hidden.
# ---------------------------------------------------------------------------
PERTURB_FRICTIONLOSS_ADD_NM = 1.0          # [Nm] added to every arm joint's frictionloss (range asked: 0.5-2.0)
PERTURB_DAMPING_ADD_NM_S_PER_RAD = 0.5      # [Nm*s/rad] added to every arm joint's damping
PERTURB_MASS_SCALE_LO = 0.85                # per-link body_mass multiplicative factor, drawn ~U[LO, HI]
PERTURB_MASS_SCALE_HI = 1.15
PERTURB_MASS_SEED = 20260910                # fixed seed -> perturbation is deterministic/reproducible

ARM_PREFIXES = (sc.EGO_PREFIX, sc.INT_PREFIX)   # ("left_arm_", "right_arm_")
ARM_JOINTS = {sc.EGO_PREFIX: sc.LEFT_ARM_JOINTS, sc.INT_PREFIX: sc.RIGHT_ARM_JOINTS}


def _disable_arm_position_servos(model: mujoco.MjModel) -> None:
    """Both arms' 6 joint position actuators are zeroed (gain/bias -> 0) so
    they exert zero force regardless of data.ctrl -- REQUIRED here: we set
    data.qpos directly to arbitrary sampled configs without also updating
    data.ctrl, so a live position servo would otherwise fight that with a
    large spurious PD-correction torque on exactly the arm dofs we are
    trying to isolate (this was the root cause of an early ~1e3 rad/s^2
    "residual" on the supposedly-exact nominal-plant sanity check -- pure
    actuator-vs-manually-set-qpos fighting, not a dynamics-model mismatch).
    Reuses sim_common's own disable_left/right_arm_position_servos
    unchanged."""
    sc.disable_left_arm_position_servos(model)
    sc.disable_right_arm_position_servos(model)


def _load_clean_baseline_spec() -> mujoco.MjSpec:
    """The scene's own MJCF ships every arm joint with a non-zero baseline
    frictionloss=0.3 Nm (a <default class="..."> compiler default, not
    something either loader below invented). frictionloss is resolved by
    MuJoCo's CONSTRAINT solver -- it lives in qfrc_constraint, not
    qfrc_bias/qfrc_passive -- so mpc_fl.py's FL law (tau = M@a + (qfrc_bias
    - qfrc_passive)) can NEVER cancel it, even on the model it was computed
    from. That would make the "nominal-plant residual ~= 0" sanity check
    fail (confirmed empirically: max|residual| ~ 17 rad/s^2 with the
    as-shipped 0.3 Nm baseline) for a reason that has NOTHING to do with
    the intentional nominal-vs-perturbed dynamics mismatch this experiment
    is about.

    So: both load_nominal() and make_perturbed_model() start from this SAME
    zeroed-out baseline (arm frictionloss -> 0; damping was already 0 in
    the shipped scene, kept as-is) -- an idealized, genuinely-frictionless
    scene the FL law IS exact on -- and make_perturbed_model() then adds
    the documented perturbation strictly ON TOP of that shared baseline.
    This also makes the nominal/perturbed difference exactly equal to the
    named PERTURB_* constants, matching the task's "add friction/damping"
    framing literally instead of adding onto an arbitrary pre-existing
    value.
    """
    spec = mujoco.MjSpec.from_file(sc.SCENE)
    sc._patch_real_collision_geoms(spec)
    for prefix in ARM_PREFIXES:
        for jname in ARM_JOINTS[prefix]:
            j = spec.joint(jname)
            j.frictionloss = 0.0
    return spec


def load_nominal() -> mujoco.MjModel:
    """The UNPERTURBED model -- what the FL controller believes it is
    driving: the shared clean baseline (see _load_clean_baseline_spec),
    compiled as-is (no perturbation added), plus disabling both arms'
    position servos (see _disable_arm_position_servos) since this
    experiment drives the arms purely via qfrc_applied at manually-set
    states."""
    spec = _load_clean_baseline_spec()
    model = spec.compile()
    assert abs(model.opt.timestep - sc.DT) < 1e-12, (model.opt.timestep, sc.DT)
    _disable_arm_position_servos(model)
    return model


def make_perturbed_model() -> mujoco.MjModel:
    """The 'real robot' stand-in: the SAME clean baseline as load_nominal(),
    with the documented perturbation (see module docstring) added ON TOP.
    Built via MjSpec so the on-disk scene file is never touched, exactly
    like sim_common.load_model()'s own in-memory MjSpec patch pattern."""
    spec = _load_clean_baseline_spec()

    # (1) + (2): frictionloss / damping on every arm dof, both arms (starting from the clean-baseline 0).
    for prefix in ARM_PREFIXES:
        for jname in ARM_JOINTS[prefix]:
            j = spec.joint(jname)
            j.frictionloss = PERTURB_FRICTIONLOSS_ADD_NM
            j.damping = j.damping + PERTURB_DAMPING_ADD_NM_S_PER_RAD  # 3-vector; only dof0 used by a hinge

    # (3) per-link body_mass scale, drawn ONCE with a fixed seed.
    rng = np.random.default_rng(PERTURB_MASS_SEED)
    for prefix in ARM_PREFIXES:
        for link_name in sc.LINK_NAMES:
            body_name = prefix + link_name
            scale = float(rng.uniform(PERTURB_MASS_SCALE_LO, PERTURB_MASS_SCALE_HI))
            b = spec.body(body_name)
            b.mass = float(b.mass) * scale

    model = spec.compile()
    assert abs(model.opt.timestep - sc.DT) < 1e-12, (model.opt.timestep, sc.DT)
    _disable_arm_position_servos(model)
    return model


def get_arm_addrs(model: mujoco.MjModel):
    """(left_qposadr, left_dofadr, right_qposadr, right_dofadr) -- reuses
    sim_common.joint_addrs (the same mj_name2id-based lookup
    scripts/make_cup_collect_safety_demo.py's build_episode uses), not
    reimplemented here."""
    _, left_qposadr, left_dofadr = sc.joint_addrs(model, sc.LEFT_ARM_JOINTS)
    _, right_qposadr, right_dofadr = sc.joint_addrs(model, sc.RIGHT_ARM_JOINTS)
    return left_qposadr, left_dofadr, right_qposadr, right_dofadr


def arm_joint_ranges(model: mujoco.MjModel):
    """(12,2) array of [lo, hi] joint-limit pairs for [left(6), right(6)]
    arm joints, read directly off the compiled model (model.jnt_range) --
    not hardcoded, so it always matches whatever scene is loaded."""
    ranges = []
    for jname in sc.LEFT_ARM_JOINTS + sc.RIGHT_ARM_JOINTS:
        jid = model.joint(jname).id
        ranges.append(model.jnt_range[jid].copy())
    return np.array(ranges)


def has_arm_contact(model: mujoco.MjModel, data) -> bool:
    """True iff ANY current contact (data.contact[:data.ncon]) involves a
    left_arm_*/right_arm_* body -- i.e. self-collision or arm-vs-scene
    (table/box) contact, as opposed to the ever-present, arm-irrelevant
    box-resting-on-table contact (verified empirically: ~100% of random
    full-joint-range configs have >=1 contact from the box alone, but only
    ~4% involve an arm body). Contact forces are NOT part of qfrc_bias/
    qfrc_passive (they live in qfrc_constraint), so an arm-contact sample
    would make even the NOMINAL FL cancellation inexact for reasons that
    have nothing to do with the dynamics-model mismatch this experiment
    targets -- both the sanity check and data collection reject such
    samples and resample."""
    for c in range(data.ncon):
        con = data.contact[c]
        for gid in (con.geom1, con.geom2):
            name = model.body(model.geom_bodyid[gid]).name
            if name.startswith(sc.EGO_PREFIX) or name.startswith(sc.INT_PREFIX):
                return True
    return False


def sample_collision_free_state(model: mujoco.MjModel, data, key_id: int,
                                 left_qposadr, left_dofadr, right_qposadr, right_dofadr,
                                 ranges: np.ndarray, rng: np.random.Generator,
                                 qd_std: float, qd_clip: float, max_tries: int = 200):
    """Sample (q(12,), qd(12,)) uniformly within `ranges` / ~N(0, qd_std^2)
    clipped to +-qd_clip, rejecting configs with an arm-involved contact
    (see has_arm_contact) so the resulting residual reflects only the
    intended dynamics-model mismatch, not contact/constraint forces. Leaves
    `data` freshly mj_forward'd at the accepted state. Returns
    (q, qd, n_tries)."""
    for tries in range(1, max_tries + 1):
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        q = rng.uniform(ranges[:, 0], ranges[:, 1])
        qd = np.clip(rng.normal(0.0, qd_std, size=12), -qd_clip, qd_clip)
        data.qpos[left_qposadr] = q[:6]
        data.qpos[right_qposadr] = q[6:]
        data.qvel[left_dofadr] = qd[:6]
        data.qvel[right_dofadr] = qd[6:]
        mujoco.mj_forward(model, data)
        if not has_arm_contact(model, data):
            return q, qd, tries
    raise RuntimeError(f"could not find a collision-free config in {max_tries} tries")


def fl_torque(M_full: np.ndarray, data, dofadr: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Feedback-linearization torque for one arm: tau = M_e @ a + c_e, with
    c_e = qfrc_bias - qfrc_passive (mpc_fl.py FLMPC.step's exact convention,
    lines ~511-519) so tau makes q_ddot = a EXACTLY on the model M_full/data
    were computed from."""
    M_e = M_full[np.ix_(dofadr, dofadr)]
    c_e = data.qfrc_bias[dofadr] - data.qfrc_passive[dofadr]
    return M_e @ a + c_e


SANITY_NOMINAL_RESIDUAL_THRESHOLD = 0.1
# [rad/s^2] NOT 1e-6: with the shared clean baseline (frictionloss/damping
# zeroed, see _load_clean_baseline_spec), FL IS exact for the ARM-ONLY
# reduced 6x6 submatrix M_e/c_e -- but each arm also carries 2 finger dofs
# and the scene has an independent free-floating box, all coupled to the
# arm through the FULL mass matrix's off-diagonal blocks. mpc_fl.py's own
# FL law (and this experiment, by design, reusing it unchanged) only ever
# builds M_e/c_e from the arm's OWN 6x6 diagonal block -- it does not (and,
# per the task, should not) reflect finger/box inertia into the FL
# command. That leftover coupling alone produces a small but genuine
# nonzero nominal-plant residual (measured empirically over 30 random
# collision-free states: mean 0.015, max 0.035 rad/s^2) that has NOTHING
# to do with the nominal-vs-perturbed dynamics mismatch this experiment
# targets. 0.1 is comfortably above that floor and >=1000x smaller than
# the smallest observed PERTURBED-plant residual (measured min 33
# rad/s^2 over the same 30 states) -- i.e. the threshold cleanly separates
# "FL is working as designed" from "dynamics mismatch is present", which
# is the actual property being sanity-checked.
def sanity_check_nominal_residual(seed: int = 0) -> float:
    """Apply an FL torque (computed FROM the nominal model) TO the nominal
    model itself and confirm the realized qacc matches the commanded
    accel almost exactly (FL is exact for the reduced arm-only submatrix
    it was derived from -- see SANITY_NOMINAL_RESIDUAL_THRESHOLD for why
    the bar isn't machine-epsilon). Returns max|residual| over both arms'
    12 dof. Callers assert this is < SANITY_NOMINAL_RESIDUAL_THRESHOLD."""
    model = load_nominal()
    data = mujoco.MjData(model)
    key = model.key("home").id
    mujoco.mj_resetDataKeyframe(model, data, key)

    left_qposadr, left_dofadr, right_qposadr, right_dofadr = get_arm_addrs(model)
    ranges = arm_joint_ranges(model)
    rng = np.random.default_rng(seed)
    q, qd, _tries = sample_collision_free_state(
        model, data, key, left_qposadr, left_dofadr, right_qposadr, right_dofadr,
        ranges, rng, qd_std=0.5, qd_clip=2.0)

    M_full = sc.full_mass_matrix(model, data)
    a = rng.normal(0.0, 1.0, size=12)
    tau_left = fl_torque(M_full, data, left_dofadr, a[:6])
    tau_right = fl_torque(M_full, data, right_dofadr, a[6:])

    data.qfrc_applied[:] = 0.0
    data.qfrc_applied[left_dofadr] = tau_left
    data.qfrc_applied[right_dofadr] = tau_right
    mujoco.mj_forward(model, data)

    qacc_realized = np.concatenate([data.qacc[left_dofadr], data.qacc[right_dofadr]])
    residual = qacc_realized - a
    return float(np.max(np.abs(residual)))


if __name__ == "__main__":
    max_abs_res = sanity_check_nominal_residual()
    print(f"[sanity] nominal-plant residual max|.| = {max_abs_res:.3e} "
          f"(must be < {SANITY_NOMINAL_RESIDUAL_THRESHOLD})")
    assert max_abs_res < SANITY_NOMINAL_RESIDUAL_THRESHOLD
    print("[sanity] PASS: FL cancellation is exact up to arm/finger-box coupling noise.")
