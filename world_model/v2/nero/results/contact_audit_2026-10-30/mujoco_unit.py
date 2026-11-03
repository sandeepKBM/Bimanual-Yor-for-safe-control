"""MuJoCo side of the contact-fidelity unit tests (spec shared with the Drake spike, 2026-10-30).  SIMULATION ONLY.
(a) TWIST: free 0.10 x 0.10 x 0.045 m block (0.5 kg, no gravity) squeezed by two 0.02 x 0.02 x 0.01 m pads at 10 N each,
    torque about z ramped at 0.05 Nm/s; slip torque = first |w_z| > 0.05 rad/s.  Analytic (uniform square patches) 0.153 Nm.
(b) CREEP: 1 kg 0.10 x 0.10 x 0.02 m box on a plate, mu 1, after 0.5 s a constant 4.905 N (0.5 mu m g) for 3 s;
    mean slide speed over the last 2 s (ideal Coulomb 0).
Contact settings copied from the pan scene: elliptic cone, impratio 10, Newton solver, finger solref 0.005 / pan 0.02,
solimp (0.9, 0.95, 0.001, 0.5, 2)."""
import json
import sys

import mujoco
import numpy as np

SOLIMP = "0.9 0.95 0.001 0.5 2"


def twist_xml(condim=3, tors=0.0, dt=0.005, noslip=0):
    return f"""<mujoco><option timestep="{dt}" gravity="0 0 0" cone="elliptic" impratio="10" noslip_iterations="{noslip}"/>
<worldbody>
 <body name="blk"><freejoint/><geom type="box" size="0.05 0.05 0.0225" mass="0.5" friction="1 {tors} 0.0001" condim="{condim}"
   solref="0.02 1" solimp="{SOLIMP}"/></body>
 <body name="padA" pos="0 0 0.0275"><joint name="jA" type="slide" axis="0 0 1"/>
   <geom type="box" size="0.01 0.01 0.005" mass="0.05" friction="1 {tors} 0.0001" condim="{condim}" priority="1" solref="0.005 1" solimp="{SOLIMP}"/></body>
 <body name="padB" pos="0 0 -0.0275"><joint name="jB" type="slide" axis="0 0 1"/>
   <geom type="box" size="0.01 0.01 0.005" mass="0.05" friction="1 {tors} 0.0001" condim="{condim}" priority="1" solref="0.005 1" solimp="{SOLIMP}"/></body>
</worldbody>
<actuator><motor joint="jA" gear="1"/><motor joint="jB" gear="1"/></actuator></mujoco>"""


def run_twist(**kw):
    m = mujoco.MjModel.from_xml_string(twist_xml(**kw))
    d = mujoco.MjData(m)
    d.ctrl[:] = [-10.0, 10.0]
    b = m.body("blk").id
    for _ in range(int(0.3 / m.opt.timestep)):          # settle the squeeze
        mujoco.mj_step(m, d)
    t0, slip, yaw1, ncon = d.time, None, None, d.ncon
    while d.time - t0 < 4.0:
        tau = 0.05 * (d.time - t0)
        d.xfrc_applied[b, 5] = tau
        mujoco.mj_step(m, d)
        if yaw1 is None and d.time - t0 >= 1.0:
            q = d.xquat[b]
            yaw1 = float(np.degrees(2 * np.arctan2(q[3], q[0])))
        if slip is None and abs(d.qvel[5]) > 0.05:
            slip = tau
            break
    return dict(slip_torque_Nm=slip, yaw_at_1s_deg=yaw1, n_contacts=int(ncon))


def creep_xml(dt=0.005, noslip=0, impratio=10):
    return f"""<mujoco><option timestep="{dt}" cone="elliptic" impratio="{impratio}" noslip_iterations="{noslip}"/>
<worldbody><geom type="box" size="0.3 0.3 0.01" pos="0 0 -0.01" friction="1 0.005 0.0001" solref="0.02 1" solimp="{SOLIMP}"/>
 <body name="box" pos="0 0 0.0101"><freejoint/><geom type="box" size="0.05 0.05 0.01" mass="1" friction="1 0.005 0.0001"
   solref="0.02 1" solimp="{SOLIMP}"/></body></worldbody></mujoco>"""


def run_creep(**kw):
    m = mujoco.MjModel.from_xml_string(creep_xml(**kw))
    d = mujoco.MjData(m)
    b = m.body("box").id
    while d.time < 0.5:
        mujoco.mj_step(m, d)
    x0 = d.xpos[b, 0]
    xs = []
    while d.time < 3.5:
        d.xfrc_applied[b, 0] = 4.905
        mujoco.mj_step(m, d)
        xs.append((d.time, d.xpos[b, 0]))
    xs = np.array(xs)
    last = xs[xs[:, 0] >= 1.5]
    v = (last[-1, 1] - last[0, 1]) / (last[-1, 0] - last[0, 0])
    return dict(mean_speed_mm_s=1e3 * v, slide_mm=1e3 * (xs[-1, 1] - x0))


if __name__ == "__main__":
    out = dict(twist={}, creep={})
    mu_t = 1.0 * 0.3826 * 0.02          # MuJoCo torsional coefficient (a length) matched to the square-patch slip torque per pad
    for name, kw in [("box condim3 dt5", dict(condim=3)), ("box condim3 dt1", dict(condim=3, dt=0.001)),
                     ("box condim4 tors=patch dt5", dict(condim=4, tors=mu_t)), ("box condim6 tors=patch dt5", dict(condim=6, tors=mu_t)),
                     ("box condim3 noslip10 dt5", dict(condim=3, noslip=10))]:
        out["twist"][name] = run_twist(**kw)
    for name, kw in [("base dt5 impratio10", {}), ("dt1", dict(dt=0.001)), ("noslip10", dict(noslip=10)),
                     ("impratio100", dict(impratio=100))]:
        out["creep"][name] = run_creep(**kw)
    print(json.dumps(out, indent=1))
    json.dump(out, open(sys.argv[1], "w"), indent=1)
