"""Bimanual data-collection + training-support infra (world-model prep).

Ported/adapted from VLAism/ur5e_benchmark (episode_schema.py, artifact_keys.py,
perturbations.py, grasp_boundary/converter.py) for the bimanual Piper scene
(scenes/scene_bimanual_pickup.xml) with camera observations, targeting a
future Dreamer/RSSM-style world-model trainer. See collection/README.md for
the full design notes, gripper convention, and command examples.
"""
