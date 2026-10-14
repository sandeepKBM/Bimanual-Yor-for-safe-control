"""people_track/sim_fix.py -- TeleopSim whose scripted walkers actually walk.  SIMULATION ONLY.

teleop/sim.py:set_people advances a person's track clock only `if np.linalg.norm(v) > 0`, where v is the velocity AT the current clock.  A
track whose first segment is stationary (h5 p_walker: rows (0, 3.6, 3.8), (2, 3.6, 3.8), (10, 0.9, 3.8)) has v = 0 at tau = 0, so the clock never
starts and the walker stands at (3.6, 3.8) forever.  This subclass (sim.py is not edited) advances the clock of every dynamic person that is not
yielding; the yield rule (stop while the robot footprint is within PERSON_YIELD_M of the torso) is unchanged by default, and `yield_m = 0` gives an
oblivious walker that follows its script regardless of the robot (a harsher test of the shield).
"""
from __future__ import annotations

import numpy as np

from world_model.v2.house.teleop import geom as GM
from world_model.v2.house.teleop.sim import TeleopSim, TICK, PERSON_TORSO_R, PERSON_YIELD_M


class TeleopSimWalking(TeleopSim):
    yield_m = PERSON_YIELD_M          # set to 0.0 (class or instance attribute) for OBLIVIOUS walkers that never stop for the robot

    def set_people(self, advance=False):
        pts = []
        mi_of = {p.name: mi for p, mi in zip(self._dyn(), self.rig.person_mocap)}
        for p in (self.scene.people if self.with_people else []):
            tau = self.ptime.get(p.name, 0.0)
            xy, v = p.at(tau)
            if advance and p.dynamic and tau < p.pts[-1, 0]:
                dd, _, _ = GM.rect_dist_point(xy, self.pose())
                if dd - PERSON_TORSO_R >= self.yield_m:
                    tau += TICK
                    self.ptime[p.name] = tau
                    xy, v = p.at(tau)
                elif np.linalg.norm(v) > 0:
                    v = np.zeros(2)
            if p.name in mi_of:
                self.d.mocap_pos[mi_of[p.name]] = (xy[0], xy[1], 0.0)
            pts.append((xy, v))
        self.people_state = pts
        return pts
