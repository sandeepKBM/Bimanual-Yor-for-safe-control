"""zono_explainer.py -- figure + video explaining point cloud -> zonotope -> faces -> CBF row, using the project's own code.
SIMULATION / SYNTHETIC DATA ONLY.  Outputs: zono_explainer.png, zono_cbf.mp4 (next to this file)."""
import itertools
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from scipy.spatial import ConvexHull

from world_model.v2.perception_zono.fit import fit_hull_zono, inflate
from world_model.v2.house.zono_rows.zono_hrep import zono_hrep, sd_lb

OUT = Path(__file__).resolve().parent
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, BLUE_LT = "#2a78d6", "#eb6834", "#1baf7a", "#86b6ef"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF, "text.color": INK,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold"})
rng = np.random.default_rng(3)

# ---------------------------------------------------------------- 1. a partial-view stereo cloud of a bottle (camera at -x)
def bottle_cloud(n=420):
    pts = []
    while len(pts) < n:
        th = rng.uniform(np.deg2rad(105), np.deg2rad(255))            # only the camera-facing half is seen
        z = rng.uniform(0.0, 0.24)
        r = 0.036 if z < 0.16 else 0.036 - (z - 0.16) / 0.04 * 0.022 if z < 0.20 else 0.014
        p = np.array([0.55 + r * math.cos(th), 0.10 + r * math.sin(th), 0.75 + z])
        p[0] += rng.normal(0, 0.003)                                  # stereo error mostly along the view ray (x)
        pts.append(p)
    return np.array(pts)

P = bottle_cloud()
hull = ConvexHull(P)
V = P[hull.vertices]
Z = fit_hull_zono(P, order=2)                                         # the project's LP fit (17 candidate dirs -> <= 6 gens)
DELTA = 0.010                                                         # measured stereo p95 error at this range [m]
Zi = inflate(Z, DELTA)
H = zono_hrep(Zi.c, Zi.G)


def zono_edges(c, G):
    """vertex set of <c,G> -> convex hull -> edges between non-coplanar faces (clean wireframe) + triangles."""
    S = np.array(list(itertools.product([-1.0, 1.0], repeat=G.shape[1])))
    X = c[None] + S @ G.T
    h = ConvexHull(X)
    tri = X[h.simplices]
    nrm = h.equations[:, :3]
    edge_tris = {}
    for t, s in enumerate(h.simplices):
        for a, b in ((s[0], s[1]), (s[1], s[2]), (s[0], s[2])):
            edge_tris.setdefault(tuple(sorted((a, b))), []).append(t)
    E = [X[list(e)] for e, ts in edge_tris.items() if len(ts) == 2 and nrm[ts[0]] @ nrm[ts[1]] < 0.999]
    return tri, np.array(E)


def style3d(ax, title):
    ax.set_title(title, loc="left", pad=2)
    ax.set_xlim(0.47, 0.63); ax.set_ylim(0.02, 0.18); ax.set_zlim(0.72, 1.02)
    ax.set_box_aspect((1, 1, 1.9))
    ax.view_init(elev=18, azim=-128)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor(SURF); a.pane.set_edgecolor(GRID); a._axinfo["grid"]["color"] = GRID
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])


def draw_zono(ax, z, color, ls="-", alpha=0.10, lw=1.2):
    tri, E = zono_edges(z.c, z.G)
    ax.add_collection3d(Poly3DCollection(tri, facecolor=color, alpha=alpha, edgecolor="none"))
    ax.add_collection3d(Line3DCollection(E, colors=color, linewidths=lw, linestyles=ls))


# ---------------------------------------------------------------- the 6-panel figure
fig = plt.figure(figsize=(13.5, 7.6))
gs = fig.add_gridspec(2, 3, hspace=0.18, wspace=0.05)

ax = fig.add_subplot(gs[0, 0], projection="3d"); style3d(ax, "1  Stereo point cloud (one cluster)")
ax.scatter(*P.T, s=3, c=INK2, alpha=0.55, depthshade=False)
ax.quiver(0.44, 0.10, 0.88, 0.05, 0, 0, color=INK2, arrow_length_ratio=0.35, lw=1.2)
ax.text2D(0.02, 0.02, f"{len(P)} points; only the camera-facing half\nis seen (camera at -x)", transform=ax.transAxes, color=INK2, bbox=dict(facecolor=SURF, edgecolor='none', alpha=0.92, pad=1.5))

ax = fig.add_subplot(gs[0, 1], projection="3d"); style3d(ax, "2  Keep only the convex-hull vertices")
ax.scatter(*P.T, s=2, c="#c3c2b7", alpha=0.5, depthshade=False)
ax.scatter(*V.T, s=16, c=BLUE, depthshade=False)
ax.text2D(0.02, 0.02, f"{len(V)} hull vertices: interior points\ncannot change the fit", transform=ax.transAxes, color=INK2, bbox=dict(facecolor=SURF, edgecolor='none', alpha=0.92, pad=1.5))

ax = fig.add_subplot(gs[0, 2], projection="3d"); style3d(ax, "3  LP fit: smallest zonotope containing them")
ax.scatter(*V.T, s=10, c=BLUE, depthshade=False)
draw_zono(ax, Z, BLUE)
ax.text2D(0.02, 0.02, f"Z = <c, G>,  G is 3 x {Z.G.shape[1]}\nc + sum lam_i |u_i.n| >= max_v v.n  for all n", transform=ax.transAxes, color=INK2, bbox=dict(facecolor=SURF, edgecolor='none', alpha=0.92, pad=1.5))

ax = fig.add_subplot(gs[1, 0], projection="3d"); style3d(ax, "4  Inflate by stereo error  Z (+) Box(delta)")
draw_zono(ax, Z, BLUE, alpha=0.06, lw=0.8)
draw_zono(ax, Zi, BLUE_LT, ls="--", alpha=0.08, lw=1.2)
ax.text2D(0.02, 0.02, f"delta = {DELTA*1000:.0f} mm (p95 depth error)\n{Zi.G.shape[1]} generators -> {len(H.N)} face planes", transform=ax.transAxes, color=INK2, bbox=dict(facecolor=SURF, edgecolor='none', alpha=0.92, pad=1.5))

# panel 5: active face for a gripper sphere
p_s, r_s, margin = np.array([0.455, 0.020, 0.86]), 0.025, 0.01
sd, act = sd_lb(H, p_s[None], return_active=True)
sd, act = float(np.ravel(sd)[0]), int(np.ravel(act)[0])
n_a, d_a = H.N[act], H.d[act]
h_val = n_a @ p_s - d_a - r_s - margin
ax = fig.add_subplot(gs[1, 1], projection="3d"); style3d(ax, "5  Active face for a gripper sphere")
ax.set_xlim(0.40, 0.63); ax.set_ylim(-0.03, 0.20)
draw_zono(ax, Zi, BLUE, alpha=0.08)
foot = p_s - (n_a @ p_s - d_a) * n_a
ax.plot(*np.c_[p_s, foot], color=GREEN, lw=2)
ax.quiver(*foot, *(0.05 * n_a), color=GREEN, lw=2, arrow_length_ratio=0.3)
u, v = np.mgrid[0:2 * np.pi:16j, 0:np.pi:10j]
ax.plot_wireframe(p_s[0] + r_s * np.cos(u) * np.sin(v), p_s[1] + r_s * np.sin(u) * np.sin(v), p_s[2] + r_s * np.cos(v), color=ORANGE, lw=0.6)
ax.text2D(0.02, 0.02, f"active face f* = argmax_f (n_f.p - d_f)\nh = n.p - d - r - margin = {h_val*100:.1f} cm", transform=ax.transAxes, color=INK2, bbox=dict(facecolor=SURF, edgecolor='none', alpha=0.92, pad=1.5))

# panel 6: top view, CBF keeps the sphere outside (closed-form single-row projection, as in the base/arm rows)
GAMMA, DT, SPEED = 4.0, 0.01, 0.25
goal = np.array([0.66, 0.12])
WAY = np.array([0.53, 0.185])
def rollout(filtered, plan=False):
    p = np.array([0.43, 0.085]); traj, hs, acts = [p.copy()], [], []
    tgt = WAY if plan else goal
    for _ in range(700):
        if plan and np.linalg.norm(WAY - p) < 0.02:
            tgt = goal
        u_nom = tgt - p; u_nom = SPEED * u_nom / max(np.linalg.norm(u_nom), 1e-9)
        q = np.array([p[0], p[1], 0.86])
        _, a = sd_lb(H, q[None], return_active=True); a = int(np.ravel(a)[0])
        n3, d = H.N[a], H.d[a]; n = n3[:2]
        h = n3 @ q - d - r_s - margin
        u = u_nom.copy()
        if filtered and n @ u_nom < -GAMMA * h:                       # row: n.u >= -gamma h   (project onto the half-plane)
            u = u_nom + (-GAMMA * h - n @ u_nom) * n / (n @ n)
        p = p + DT * u; traj.append(p.copy()); hs.append(h); acts.append(a)
        if np.linalg.norm(goal - p) < 0.004:
            break
    return np.array(traj), np.array(hs), acts

T_raw, h_raw, _ = rollout(False)
T_stop, h_stop, _ = rollout(True)
T_cbf, h_cbf, acts_cbf = rollout(True, plan=True)
ax = fig.add_subplot(gs[1, 2]); ax.set_title("6  CBF row  n.u >= -gamma h  (top view)", loc="left", pad=2)
S = np.array(list(itertools.product([-1.0, 1.0], repeat=Zi.G.shape[1])))
XY = (Zi.c[None] + S @ Zi.G.T)[:, :2]; hh = ConvexHull(XY)
ax.fill(*XY[hh.vertices].T, color=BLUE, alpha=0.18, lw=0); ax.plot(*np.r_[XY[hh.vertices], XY[hh.vertices[:1]]].T, color=BLUE, lw=1.2)
ax.plot(*T_raw.T, color=INK2, ls="--", lw=1.4, label="operator command (nominal)")
ax.plot(*T_stop.T, color=ORANGE, ls=":", lw=2, label="CBF row only (stops at the boundary)")
ax.plot(*T_cbf.T, color=ORANGE, lw=2, label="planner proposes, CBF row certifies")
ax.add_patch(plt.Circle(T_cbf[0], r_s, color=ORANGE, alpha=0.25, lw=0))
ax.plot(*goal, marker="*", ms=10, color=INK)
ax.set_aspect("equal"); ax.set_xlim(0.41, 0.68); ax.set_ylim(-0.01, 0.23)
ax.legend(loc="upper left", frameon=False, fontsize=8); ax.grid(color=GRID, lw=0.6)
ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
ax.text(0.98, 0.02, f"min h: nominal {h_raw.min()*100:.1f} cm (collides)\nCBF only {h_stop.min()*100:.1f} cm, planner+CBF {h_cbf.min()*100:.1f} cm", transform=ax.transAxes,
        ha="right", va="bottom", color=INK2, fontsize=8)
fig.suptitle("From a stereo point cloud to one linear safety constraint (project code: fit_hull_zono, zono_hrep, sd_lb)",
             x=0.01, ha="left", fontsize=11, fontweight="bold")
fig.savefig(OUT / "zono_explainer.png", dpi=150, bbox_inches="tight")
print("png", OUT / "zono_explainer.png", "gens", Z.G.shape[1], Zi.G.shape[1], "faces", len(H.N), "hull", len(V),
      "h_raw_min", round(h_raw.min(), 4), "h_cbf_min", round(h_cbf.min(), 4))

# ---------------------------------------------------------------- video: top view of the CBF at work, with h(t)
fig2, (axA, axB) = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1]))
axA.fill(*XY[hh.vertices].T, color=BLUE, alpha=0.18, lw=0); axA.plot(*np.r_[XY[hh.vertices], XY[hh.vertices[:1]]].T, color=BLUE, lw=1.2)
axA.plot(*T_raw.T, color=INK2, ls="--", lw=1.0, label="nominal (straight to goal)"); axA.plot(*goal, marker="*", ms=10, color=INK)
axA.set_aspect("equal"); axA.set_xlim(0.41, 0.68); axA.set_ylim(-0.01, 0.23)
axA.plot(*T_stop.T, color=ORANGE, ls=':', lw=1.2, label='CBF only (stalls)'); axA.grid(color=GRID, lw=0.6)
axA.set_title("Gripper sphere vs. inflated bottle zonotope (top view)", loc="left")
trail, = axA.plot([], [], color=ORANGE, lw=2, label="planner + CBF")
disc = plt.Circle((0, 0), r_s, color=ORANGE, alpha=0.35, lw=0); axA.add_patch(disc)
face_ln, = axA.plot([], [], color=GREEN, lw=3, label="active face")
q_nom = axA.quiver([0], [0], [0], [0], color=INK2, scale=1.2, width=0.006)
q_cbf = axA.quiver([0], [0], [0], [0], color=ORANGE, scale=1.2, width=0.008)
txt = axA.text(0.02, 0.98, "", transform=axA.transAxes, va="top", fontsize=8, color=INK, family="monospace")
axA.legend(loc="lower right", frameon=False, fontsize=8)
t_ax = np.arange(len(h_cbf)) * DT
axB.axhline(0, color=INK2, lw=0.8); axB.plot(np.arange(len(h_raw)) * DT, np.array(h_raw) * 100, color=INK2, ls="--", lw=1, label="nominal")
axB.plot(np.arange(len(h_stop)) * DT, np.array(h_stop) * 100, color=ORANGE, ls=":", lw=1.2, label="CBF only")
hl, = axB.plot([], [], color=ORANGE, lw=2, label="planner + CBF"); axB.set_xlim(0, max(len(h_raw), len(h_cbf)) * DT)
axB.set_ylim(min(h_raw.min(), 0) * 100 - 1, max(h_cbf.max(), h_raw.max()) * 100 + 1); axB.grid(color=GRID, lw=0.6)
axB.set_title("Barrier h(t) = n.p - d - r - margin  [cm]", loc="left"); axB.set_xlabel("time [s]"); axB.legend(frameon=False, fontsize=8)

def frame(i):
    p = T_cbf[min(i, len(T_cbf) - 1)]; disc.center = tuple(p); trail.set_data(*T_cbf[:i + 1].T)
    k = min(i, len(acts_cbf) - 1); a = acts_cbf[k]; n3, d = H.N[a], H.d[a]
    t = np.array([-n3[1], n3[0]]); foot = p - (n3[:2] @ p + n3[2] * 0.86 - d) * n3[:2]
    face_ln.set_data(*np.c_[foot - 0.03 * t, foot + 0.03 * t])
    u_nom = goal - p; u_nom = SPEED * u_nom / max(np.linalg.norm(u_nom), 1e-9)
    u = (T_cbf[min(i + 1, len(T_cbf) - 1)] - p) / DT
    q_nom.set_offsets([p]); q_nom.set_UVC([u_nom[0]], [u_nom[1]]); q_cbf.set_offsets([p]); q_cbf.set_UVC([u[0]], [u[1]])
    h = h_cbf[k]; lhs = n3[:2] @ u_nom
    txt.set_text(f"face f*={a:2d}  n=({n3[0]:+.2f},{n3[1]:+.2f})\nh = {h*100:5.1f} cm\n"
                 f"row: n.u >= -gamma h = {-GAMMA*h:+.3f}\nnominal n.u = {lhs:+.3f}  "
                 + ("ACTIVE -> projected" if lhs < -GAMMA * h else "inactive"))
    hl.set_data(t_ax[:k + 1], h_cbf[:k + 1] * 100)
    return trail, disc, face_ln, q_nom, q_cbf, txt, hl

anim = animation.FuncAnimation(fig2, frame, frames=range(0, len(T_cbf), 1), blit=False)
anim.save(OUT / "zono_cbf.mp4", writer=animation.FFMpegWriter(fps=15, bitrate=2400), dpi=130)
print("mp4", OUT / "zono_cbf.mp4", "frames", len(T_cbf) // 2)
