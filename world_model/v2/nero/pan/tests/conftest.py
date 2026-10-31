"""pan tests: render with osmesa (CPU) unless the caller chose a GL backend; must run before mujoco is first imported."""
import os

os.environ.setdefault("MUJOCO_GL", "osmesa")
