"""methods.py -- the fit methods compared in the study. Each takes a point set P (n,3) and returns a ZUnion. SIMULATION ONLY."""
from __future__ import annotations

from world_model.v2.perception_zono import fit as F
from world_model.v2.perception_zono.zonoutil import ZUnion

PAD = 0.006          # 2 sigma of the 3 mm noise condition


def _one(Z):
    return ZUnion([Z])


def _inf(U, r):
    return ZUnion([F.inflate(z, r) for z in U.zs])


BASE = {      # every method: (P, views) -> ZUnion
    "pca_box": lambda P, V: _one(F.fit_box(P, "pca")),
    "ubox": lambda P, V: _one(F.fit_box(P, "up")),
    "hz2": lambda P, V: _one(F.fit_hull_zono(P, 2)),
    "hz3": lambda P, V: _one(F.fit_hull_zono(P, 3)),
    "hz4": lambda P, V: _one(F.fit_hull_zono(P, 4)),
    "split_pa": lambda P, V: ZUnion(F.fit_split(P, "pa", gain=0.10)[0]),                                   # hull-volume criterion, PA planes
    "split_q": lambda P, V: ZUnion(F.fit_split(P, "quantile", gain=0.10)[0]),                              # hull-volume criterion, quantile planes
    "split_pa_fs": lambda P, V: ZUnion(F.fit_split(P, "pa", gain=0.08, criterion="free", views=V)[0]),     # ray-carved free-space criterion
    "split_q_fs": lambda P, V: ZUnion(F.fit_split(P, "quantile", gain=0.08, criterion="free", views=V)[0]),
}
PADDED = ["ubox", "hz2", "split_pa_fs"]


