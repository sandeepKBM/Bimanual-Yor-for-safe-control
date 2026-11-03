"""MuJoCo version of the 10-30 Drake twist test (pan/drake_spike/unit_tests.twist). SIMULATION ONLY, CPU.
Two 2x2 cm pads squeeze a 10x10x4.5 cm block with F = 10 N each (gravity off, mu = 1); a torque about the squeeze axis ramps at
0.05 Nm/s; slip torque = torque when |w_z| first exceeds 0.05 rad/s.  Analytic (uniform pressure, square pad, both pads):
2 * mu * F * 0.3826 * S_PAD = 0.153 Nm.  Contact settings copied from the scene fingers (priority 1, solref 0.005 1, elliptic,
impratio 10, Newton, implicitfast); swept: condim 3/4, torsional coefficient, timestep."""
import itertools
import json

import mujoco
import numpy as np

MU, S_PAD, F, RATE, TMAX = 1.0, 0.02, 10.0, 0.05, 4.0
ANALYTIC = 2 * MU * F * 0.3826 * S_PAD

XML = """<mujoco><option timestep="{dt}" gravity="0 0 0" cone="elliptic" impratio="10" solver="Newton" integrator="implicitfast"
 iterations="100" tolerance="1e-8"/>
<worldbody>
 <body name="block"><freejoint/><geom type="box" size="0.05 0.05 0.0225" mass="0.5" friction="{mu} 0.005 0.0001" condim="3"/></body>
 <body name="top" pos="0 0 0.0275"><joint name="jt" type="slide" axis="0 0 1"/>
  <geom type="box" size="0.01 0.01 0.005" mass="0.05" friction="{mu} {tor} 0.01" condim="{cd}" priority="1" solref="0.005 1"/></body>
 <body name="bot" pos="0 0 -0.0275"><joint name="jb" type="slide" axis="0 0 1"/>
  <geom type="box" size="0.01 0.01 0.005" mass="0.05" friction="{mu} {tor} 0.01" condim="{cd}" priority="1" solref="0.005 1"/></body>
</worldbody>
<actuator><motor joint="jt" gear="1"/><motor joint="jb" gear="1"/></actuator></mujoco>"""


def twist(dt, cd, tor):
    m = mujoco.MjModel.from_xml_string(XML.format(dt=dt, mu=MU, cd=cd, tor=tor))
    d = mujoco.MjData(m)
    d.ctrl[:] = (-F, F)
    blk = m.body("block").id
    for _ in range(int(0.3 / dt)):                     # squeeze settle, no torque
        mujoco.mj_step(m, d)
    slip, ncon = None, d.ncon
    for k in range(int(TMAX / dt)):
        d.xfrc_applied[blk, 5] = RATE * (k + 1) * dt
        mujoco.mj_step(m, d)
        if abs(d.cvel[blk][2]) > 0.05:
            slip = RATE * (k + 1) * dt
            break
    return dict(dt=dt, condim=cd, torsional=tor, slip_Nm=slip, n_contacts=ncon, ratio=None if slip is None else round(slip / ANALYTIC, 3))


rows = [twist(dt, cd, tor) for dt, (cd, tor) in itertools.product((0.005, 0.001), ((3, 0.01), (4, 0.01), (4, 0.003), (4, 0.001)))]
for r in rows:
    print(r)
print("analytic", round(ANALYTIC, 4))
json.dump(dict(analytic=ANALYTIC, rows=rows), open("twist_mj.json", "w"))
