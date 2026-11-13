"""Vendored subset of `controller_core` (the author's UR5 / cart-pole controller repository, commit f7f5d79).

Only the four modules that experiments/ab_smooth_vs_bangbang/sim_common.py and scripts/make_cup_collect_safety_demo.py import
are copied, byte-for-byte: box_qp.py, constrained_box_qp.py, _box_qp_numba.py, filters.py (numpy / numba only).
The upstream package __init__ (which imports ~25 unrelated modules) is replaced by this empty one.
Set REAL_CARTPOLE_ROOT to a full checkout to use the upstream package instead.
"""
