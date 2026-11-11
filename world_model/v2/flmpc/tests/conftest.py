import mujoco
import pytest

from world_model.v2.nero import adapters as A
from world_model.v2.nero import config as C
from world_model.v2.nero import ik as IK


@pytest.fixture(scope="module")
def hold():
    m = A.compose_scene(tau=C.tau_limits())
    d = mujoco.MjData(m)
    info = IK.set_hold_state(m, d)
    A.disable_arm_servos(m)
    return m, d, info
