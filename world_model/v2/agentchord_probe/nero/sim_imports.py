"""agentchord_probe/nero/sim_imports.py -- one place for the (heavy) sim imports, so tests can skip cleanly."""
import mujoco                                                            # noqa: F401
from world_model.v2.nero.contents import carry as CR                     # noqa: F401
from world_model.v2.nero.contents import envelope as E                   # noqa: F401
from world_model.v2.nero.contents import scenes as SC                    # noqa: F401
