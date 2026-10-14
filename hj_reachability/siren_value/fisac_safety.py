"""Fisac et al. (2019), "A General Safety Framework for Learning-Based
Control Systems": discounted-safety Bellman operator, specialized to
``hj_reachability.dynamics.BimanualPairAvoidance`` -- the 12-D
``[q_ego(6), q_int(6)]`` kinematic single-integrator arm-arm avoidance game
(``xdot_ego = u`` (ego/controlled arm), ``xdot_int = d`` (intruder, treated
as a worst-case adversarial disturbance), ``||u||<=u_bar_ego``,
``||d||<=u_bar_int``).

Discounted safety Bellman equation (Fisac et al. 2019, Eq. 7-ish; this is
also exactly the form the task spec asked for)::

    V(x) = (1 - gamma) * l(x)
           + gamma * min( l(x),  min_d max_u V(x + dt * [u; d]) )

Sign convention (MUST match ``dynamics.BimanualPairAvoidance.boundary_fn``,
i.e. ``mujoco_fk.MuJoCoFKOracle.boundary_fn``, EXACTLY -- see that module's
docstring): ``l(x) > 0`` is SAFE (clear of the combined-radius+margin
shell), ``l(x) <= 0`` is UNSAFE. ``V`` is trained to share that convention:
``V(x) > 0`` -> (this Bellman recursion's estimate of) safe, ``V(x) <= 0``
-> unsafe. Concretely:

  - The outer ``min(l(x), ...)`` is the standard "avoid" recursion: you are
    only as safe *right now* as the worse of (a) your instantaneous margin
    ``l(x)`` and (b) what a fully-adversarial future promises -- this is
    what makes ``V`` a *viability* value (once unsafe, i.e. ``l(x)<=0``,
    stays <= 0 forever under this recursion) rather than a discounted-sum-
    of-rewards value.
  - The inner ``min_d max_u V(x')`` is the one-step worst-case-for-the-
    intruder, best-case-for-the-ego backup: the disturbance ``d`` picks the
    WORST direction for the ego (minimizes ``V`` -> pushes toward unsafe),
    the control ``u`` picks the BEST direction for the ego (maximizes ``V``
    -> steers toward safe). Because the dynamics are affine and the two
    actions live in disjoint state-derivative blocks (``u`` only drives
    ``q_ego``, ``d`` only drives ``q_int``), this game satisfies Isaacs'
    condition (``min_d max_u = max_u min_d``) and has the CLOSED-FORM
    saddle point below -- no inner optimization loop needed.

Closed-form min-max backup (given ``dV/dx`` split into the ego half
``dvds_ego = dV/dq_ego`` (first 6 dims) and the intruder half
``dvds_int = dV/dq_int`` (last 6 dims), both evaluated via autograd at the
CURRENT state ``x`` using the frozen/target value net):

    u* = +u_bar_ego * dvds_ego / ||dvds_ego||   (ego climbs +grad V: MAXIMIZES V, i.e. steers toward safety)
    d* = -u_bar_int * dvds_int / ||dvds_int||   (intruder descends -grad V: MINIMIZES V, i.e. worst-case closes in)
    x' = x + dt * [u*; d*]

then the backup target is ``min(l(x), V_target(x'))``. Note this is
DELIBERATELY NOT the same sign pattern as
``dynamics.BimanualPairAvoidance.optimal_control``/``optimal_disturbance``
(which negate ``u`` relative to this): those methods implement the classical
toolbox-style Hamiltonian ``H = min_u max_d dV.[u;d]`` for a *different*
value-function role (the level-set-toolbox BRT recursion, driven by the PDE
``dV/dt + H(x,dV/dx) = 0``, min over u then max over d with V's sign meaning
"time to reach the target"), not the Fisac discounted-safety fixed point
implemented here. Re-deriving from first principles for THIS V's convention
(``V>0``=safe, ego maximizes, intruder minimizes) is what gives the ``+``/``-``
signs above; do not mix the two formulas.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

N_JOINTS_PER_ARM = 6  # local copy to avoid importing dynamics.py at module scope; see train_siren_value.py for the wiring.


@dataclass(frozen=True)
class FisacConfig:
    gamma: float = 0.999   # discount; Fisac et al. use gamma -> 1 for the "true" safety value, <1 for a contraction in training.
    dt: float = 0.05       # game timestep [s] for the one-step lookahead backup.
    u_bar_ego: float = 1.5  # rad/s, matches dynamics.DEFAULT_U_BAR_EGO
    u_bar_int: float = 1.5  # rad/s, matches dynamics.DEFAULT_U_BAR_INT
    grad_eps: float = 1e-6  # safe-normalize epsilon for near-zero gradients


def minmax_backup_state(x: torch.Tensor, value_net: torch.nn.Module,
                         cfg: FisacConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """One-step closed-form min_d max_u backup state ``x'`` and the raw
    gradient ``dV/dx`` (both returned so callers can sanity-check
    finiteness), given a (possibly frozen/target) value net.

    ``x``: (B, 12) tensor, [q_ego(6), q_int(6)]. Returns ``(x_next (B,12),
    grad (B,12))``, both detached from the input graph (this is a target
    computation -- no gradient should flow back through it into the caller's
    training graph).
    """
    # Force local autograd tracking regardless of the caller's context (this
    # is called from inside `bellman_target`'s `torch.no_grad()` block, since
    # the OVERALL backup is a stop-gradient target -- but we still need one
    # local, self-contained autograd.grad call to get dV/dx here).
    with torch.enable_grad():
        x_ = x.detach().clone().requires_grad_(True)
        v = value_net(x_).squeeze(-1)  # (B,)
        grad = torch.autograd.grad(v.sum(), x_, create_graph=False)[0]  # (B, 12)

    dvds_ego = grad[:, :N_JOINTS_PER_ARM]
    dvds_int = grad[:, N_JOINTS_PER_ARM:]

    ego_norm = dvds_ego.norm(dim=-1, keepdim=True).clamp_min(cfg.grad_eps)
    int_norm = dvds_int.norm(dim=-1, keepdim=True).clamp_min(cfg.grad_eps)

    # See module docstring: ego maximizes V (+grad), intruder minimizes V (-grad).
    u_star = cfg.u_bar_ego * dvds_ego / ego_norm
    d_star = -cfg.u_bar_int * dvds_int / int_norm

    action = torch.cat([u_star, d_star], dim=-1)  # (B, 12), dsdt = [u; d] (BimanualPairAvoidance.dsdt)
    x_next = (x_.detach() + cfg.dt * action).detach()
    return x_next, grad.detach()


def bellman_target(x: torch.Tensor, l_x: torch.Tensor, target_net: torch.nn.Module,
                    cfg: FisacConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Fisac discounted-safety Bellman target (RHS of the fixed-point
    equation in the module docstring), evaluated with the frozen
    ``target_net``.

    Args:
        x: (B, 12) current states.
        l_x: (B,) boundary values ``l(x)`` at those states (computed by the
            caller via ``dynamics.BimanualPairAvoidance.boundary_fn``, which
            is a per-sample numpy/MuJoCo call -- not batched, and not a
            torch op, hence passed in rather than recomputed here).
        target_net: frozen copy of the value network.
        cfg: FisacConfig (gamma, dt, u_bar_ego, u_bar_int).

    Returns:
        (target (B,) tensor, detached; grad (B,12) tensor, the dV/dx used
        for the backup, returned only for the caller's finite-gradient
        sanity check).
    """
    with torch.no_grad():
        x_next, grad = minmax_backup_state(x, target_net, cfg)
        v_next = target_net(x_next).squeeze(-1)          # V_target(x + dt*[u*;d*])
        backup = torch.minimum(l_x, v_next)                # min(l(x), min_d max_u V(x'))
        target = (1.0 - cfg.gamma) * l_x + cfg.gamma * backup
    return target.detach(), grad


def boundary_fn_batch(dynamics_obj, states: np.ndarray) -> np.ndarray:
    """Batched wrapper around ``dynamics_obj.boundary_fn`` (which is a
    per-sample, non-batched MuJoCo FK call -- see ``mujoco_fk.py``'s
    ``MuJoCoFKOracle.boundary_fn`` docstring: "Not batched ... callers loop
    over samples"). Returns an (N,) float32 array. This loop (not the
    network) is the throughput bottleneck of training (~4-5ms/call on CPU,
    see hj_reachability/siren_value/README.md); left as a plain python loop
    per the oracle's own documented calling convention rather than
    reimplementing batching here.
    """
    return np.array([dynamics_obj.boundary_fn(s) for s in states], dtype=np.float32)
