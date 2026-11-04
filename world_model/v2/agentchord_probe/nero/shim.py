"""agentchord_probe/nero/shim.py -- make THEIR planner code importable without DexSim / open3d / their deploy path.  Call install() FIRST.

What is real (their files, unedited, loaded from /data/local/ss5772/agentchord/src):
  embodichain.agents.hierarchy.{task,recovery,compile}_agent, agent_base, llm; embodichain.agents.mllm.prompt.*;
  embodichain.agents.agentchord.{agent_graph, graph_spec, monitor_functions, monitor_utils}; embodichain.utils.{llm_json,logger,math,...}.
What is replaced (sys.modules entries, ours):
  embodichain.agents.agentchord.atom_actions  -> nero.atoms (action namespace + drive; the real one pulls DexSim)
  embodichain.utils.utility                   -> load_txt / encode_image, verbatim copies (the real module imports open3d)
  embodichain.data                            -> database_agent_prompt_dir only (the real package pulls assets/dataset)
SAFETY: embodichain.deploy, ROS, CAN and robot SDK imports are blocked (same list as ac_run.py).  Nothing here opens a connection.
"""
from __future__ import annotations

import importlib.abc
import os
import sys
import types

THEIR_SRC = os.environ.get("AC_SRC", "/data/local/ss5772/agentchord/src")      # read-only clone, never edited
BLOCKED = ("embodichain.deploy", "rclpy", "rospy", "piper_sdk", "can", "pyAgxArm", "agx")
_done = False


class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ImportError(f"[agentchord_probe.nero] import of {name!r} blocked (hardware path; offline sim only)")
        return None


def _utility_stub():
    m = types.ModuleType("embodichain.utils.utility")

    def load_txt(path):                                  # verbatim from embodichain/utils/utility.py
        with open(path, "r") as f:
            contents = f.read().strip()
        return contents

    def encode_image(image, format="png"):               # verbatim
        import base64
        import cv2
        image_encode = cv2.imencode(f".{format}", image)[1]
        return base64.b64encode(image_encode).decode("utf-8")

    m.load_txt, m.encode_image = load_txt, encode_image
    return m


def install(scratch_dir=None):
    """idempotent.  scratch_dir: stands in for embodichain.data.database_agent_prompt_dir (we always pass log_dir explicitly)."""
    global _done
    if _done:
        return
    if not any(isinstance(f, _Block) for f in sys.meta_path):
        sys.meta_path.insert(0, _Block())
    if THEIR_SRC not in sys.path:
        sys.path.append(THEIR_SRC)
    # their llm.py reads these at IMPORT time; the gateway URL is set by the caller before the first hierarchy import
    os.environ.setdefault("OPENAI_API_KEY", "offline-gateway")
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost"
    sys.modules["embodichain.utils.utility"] = _utility_stub()
    data = types.ModuleType("embodichain.data")
    data.database_agent_prompt_dir = str(scratch_dir or os.environ.get("AC_SCRATCH", "/tmp/ac_nero_unused"))
    sys.modules["embodichain.data"] = data
    from . import atoms
    sys.modules["embodichain.agents.agentchord.atom_actions"] = atoms.ATOMS_MODULE
    _done = True
