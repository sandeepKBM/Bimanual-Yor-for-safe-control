import numpy as np, mujoco, pytest
from world_model.v2.nero.contents import scenes as S

REF_L, REF_R = np.array([0.3815, -0.1141, 0.8554]), np.array([0.3778, -0.4861, 0.8554])   # pre-11-03 scene position
REF_L, REF_R = REF_L + [0.0, S.SCENE_DY, 0.0], REF_R + [0.0, S.SCENE_DY, 0.0]                  # 11-03: scene shifted by SCENE_DY (ASSUMED) in y


def _worlds():
    out = []
    for n, seed in ((2, 0), (4, 1), (6, 2)):
        out.append((f"tray{n}", S.sample_tray_world(n, np.random.default_rng(seed))))
    out.append(("tray_full", S.sample_tray_world(4, np.random.default_rng(3))))
    out[-1][1].fills_kg = [0.25] * 4
    out.append(("box0.5", S.BoxWorld(contents_kg=0.5, contents_xy=(0.055, 0.085))))
    out.append(("box5", S.BoxWorld(contents_kg=5.0, contents_xy=(-0.055, -0.085))))
    return out


_cache = {}
def run(name):
    if name not in _cache:
        w = dict(_worlds())[name]
        m = S.compose(w); d = mujoco.MjData(m)
        S.settle(m, d, 1.0)
        _cache[name] = (w, m, d)
    return _cache[name]

NAMES = [n for n, _ in _worlds()]


@pytest.mark.parametrize("name", NAMES)
def test_settles(name):
    w, m, d = run(name)
    ob = m.body("obj").id
    v = np.linalg.norm(d.cvel[ob][3:])
    assert v < 1e-3
    R = d.xmat[ob].reshape(3, 3)
    assert np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))) < 1.0
    assert abs(d.xpos[ob][2] - w.counter_top) < 2e-3
    if isinstance(w, S.TrayWorld):
        for i in range(w.n_cups):
            Rc = d.xmat[m.body(f"cup{i}").id].reshape(3, 3)
            assert Rc[2, 2] > 0.999
            rel = d.xpos[m.body(f"cup{i}").id] - d.xpos[ob]
            assert abs(rel[2] - w.base_t) < 2e-3 and abs(rel[0]) < w.tray_lx / 2 and abs(rel[1]) < w.tray_ly / 2
    else:
        rel = d.xpos[m.body("contents").id] - d.xpos[ob]
        X, Y, Z = w.box_outer; ch = w.contents_half
        assert abs(rel[0]) + ch[0] < X / 2 - w.wall_t + 1e-3 and abs(rel[1]) + ch[1] < Y / 2 - w.wall_t + 1e-3   # corner placement touches the wall
        assert w.wall_t - 3e-3 < rel[2] - ch[2] and rel[2] + ch[2] < Z - w.wall_t


@pytest.mark.parametrize("name", ["tray2", "box5"])
def test_grasp_sites(name):
    w, m, d = run(name)
    assert np.linalg.norm(d.site_xpos[m.site("obj_grasp_left").id] - REF_L) < 5e-3
    assert np.linalg.norm(d.site_xpos[m.site("obj_grasp_right").id] - REF_R) < 5e-3


def test_fill_freeboard():
    h = 0.25 / (1000 * np.pi * S.CUP_RIN ** 2)
    assert 0.095 - (S.CUP_BOT + h) >= 0.01


@pytest.mark.parametrize("name", NAMES)
def test_truth_masses(name):
    w, m, d = run(name)
    t = S.object_truth(m, d)
    if isinstance(w, S.TrayWorld):
        _, fills = w.layout()
        assert abs(t["total_mass"] - (0.30 + 0.30 * w.n_cups + sum(fills))) < 1e-6
        assert abs(t["empty_mass"] - 0.30) < 1e-6
    else:
        assert abs(t["total_mass"] - (0.25 + w.contents_kg)) < 1e-6
        assert abs(t["contents_mass"] - w.contents_kg) < 1e-6


def test_truth_com_hand():
    w = S.TrayWorld(n_cups=2, cup_xy=[(0.0, -0.07), (0.02, 0.08)], fills_kg=[0.0, 0.2])
    m = S.compose(w); d = mujoco.MjData(m); mujoco.mj_forward(m, d)    # un-settled: exact layout
    t = S.object_truth(m, d)
    h = 0.2 / (1000 * np.pi * 0.035 ** 2)
    z_cup = 0.008 + 1e-4 + 0.0475                      # empty mug COM (initial gap 1e-4 above the base, 1e-4 obj drop is common)
    z_c1 = (0.30 * 0.0475 + 0.2 * (0.006 + h / 2)) / 0.5 + 0.008 + 1e-4
    mc = 0.30 + 0.5
    com = (0.30 * np.array([0.0, -0.07, z_cup]) + 0.5 * np.array([0.02, 0.08, z_c1])) / mc
    # the obj frame origin sits drop_eps below... cups sit 1e-4 above the base inside the obj frame (as built)
    assert np.allclose(t["contents_com"], com, atol=1e-6)
    assert abs(t["contents_mass"] - 0.8) < 1e-9


def test_no_arm_contact_with_supports():
    for name in ("tray2", "box5"):
        w, m, d = run(name)
        sup = {m.geom("counter").id} | ({m.geom("shelf").id} if name.startswith("box") else set())
        for k in range(d.ncon):
            g1, g2 = d.contact[k].geom1, d.contact[k].geom2
            if g1 in sup or g2 in sup:
                o = g2 if g1 in sup else g1
                bn = m.body(m.geom_bodyid[o]).name
                # walk up the tree to see whether it is an arm body
                b = m.geom_bodyid[o]
                while b > 0:
                    if m.body(b).name.startswith(("left_arm", "right_arm")):
                        pytest.fail(f"arm geom {m.geom(o).name} touches support in {name}")
                    b = m.body_parentid[b]


def test_support_tops():
    s = S.support_tops(S.BoxWorld())
    assert abs(s["counter"]["top"] - 0.82) < 1e-9 and abs(s["shelf"]["top"] - 0.97) < 1e-9


def test_sample_fits_six():
    for seed in range(5):
        S.sample_tray_world(6, np.random.default_rng(seed))
