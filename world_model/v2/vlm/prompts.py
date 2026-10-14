"""vlm/prompts.py -- prompt builders for (a) IR spec authoring and (b) Phys2Real-style physical-parameter elicitation."""
from __future__ import annotations

import json

GRAMMAR = """You write safety/task constraints for a bimanual robot carrying an object, in a small JSON constraint language.
A spec is ONE JSON object built from these forms (every atom has a signed-distance meaning: > 0 satisfied, < 0 violated):
  {"inside":  {"a": E, "b": E, "margin": number}}   A lies inside region B, shrunk by `margin` metres (B must be a region such as tray_top, cup_interior, target_table)
  {"upright": {"a": NAME, "theta_max": number}}      the z-axis of A is tilted from vertical by at most theta_max radians
  {"clear":   {"a": NAME, "b": NAME, "d": number}}   A and B stay at least d metres apart (for obstacles; both must be solid entities)
  {"support": {"a": NAME, "b": NAME}}                the centre of mass of A stays over the top face of B
  {"and": [spec, ...]}   {"or": [spec, ...]}   {"not": spec}
  {"G": {"lo": 0, "hi": "H", "of": spec}}            spec holds ALWAYS over the time window (H is the planning horizon; write the string "H" for hi)
  {"F": {"lo": 0, "hi": "H", "of": spec}}            spec holds AT SOME TIME in the window
  E := "entity_name"  |  {"shrink": {"of": "entity_name", "by": number}}
Use ONLY the entity names listed for the scene. Numbers are plain JSON numbers in metres / radians (you may also use the names "H", "lo").
Output ONLY the JSON object, nothing else."""

EXAMPLE = {"scene": "a tray with a ball on it", "entities": {"tray": "the tray", "tray_top": "the tray top-face region", "ball": "the ball centre"},
           "instruction": "Carry the tray without dropping the ball.",
           "spec": {"G": {"lo": 0, "hi": "H", "of": {"and": [
               {"inside": {"a": "ball", "b": {"shrink": {"of": "tray_top", "by": 0.02}}, "margin": 0.0}},
               {"upright": {"a": "tray", "theta_max": 0.3}}]}}}}


def spec_system():
    ex = EXAMPLE
    return (GRAMMAR + "\n\nEXAMPLE\nScene: " + ex["scene"] + "\nEntities: " + json.dumps(ex["entities"]) + "\nInstruction: " +
            ex["instruction"] + "\nSpec: " + json.dumps(ex["spec"]))


def spec_user(scene):
    return ("Look at the image(s) of the scene.\nEntities: " + json.dumps(scene["entities"]) + "\nInstruction: " + scene["instruction"] +
            "\nWrite the constraint spec as one JSON object.")


# ---------------------------------------------------------------- physical parameters
PARAM_DEFS = {
    "mu_roll": dict(
        key="ball_rolling_resistance_m", unit="metres",
        define=("rolling resistance coefficient of the ball on the tray, in METRES: rolling torque = mu_roll * normal force "
                "(equivalently the rolling-friction lever arm)"),
        hint="Typical values span about 0.0005 (hard steel ball on a hard flat surface) to 0.006 (soft rubber or foam ball)."),
    "mu": dict(
        key="ball_tray_sliding_friction", unit="dimensionless",
        define="Coulomb sliding friction coefficient between the ball and the tray surface",
        hint="Typical values 0.2 to 0.8."),
    "ball_radius": dict(
        key="ball_radius_m", unit="metres", define="radius of the ball", hint="The tray is 30 cm x 20 cm."),
    "bead_friction": dict(
        key="bead_cup_sliding_friction", unit="dimensionless",
        define="Coulomb sliding friction coefficient between the beads and the cup wall/floor",
        hint="Typical values 0.25 to 0.8."),
    "n_beads": dict(key="number_of_beads", unit="count", define="number of beads in the cup", hint="Between 2 and 12."),
    "bead_radius": dict(key="bead_radius_m", unit="metres", define="radius of one bead", hint="The cup is 6 cm wide."),
}
KIND_PARAMS = {"tray_ball": ["mu_roll", "mu", "ball_radius"], "tray_door": ["mu_roll", "mu", "ball_radius"],
               "cup_beads": ["bead_friction", "n_beads", "bead_radius"]}


def param_system():
    return ("You estimate physical parameters of objects in simulated robot scenes from images. For each requested parameter give your "
            "best ESTIMATE and a self-reported uncertainty SIGMA: one standard deviation, in the SAME units as the estimate "
            "(you should expect the true value to lie within estimate +/- sigma about 68% of the time). Be honest about uncertainty. "
            "Output ONLY a JSON object of the form {\"<key>\": {\"estimate\": number, \"sigma\": number}, ...}.")


def param_user(kind, hint=True):
    lines = []
    for p in KIND_PARAMS[kind]:
        d = PARAM_DEFS[p]
        lines.append(f"- \"{d['key']}\" ({d['unit']}): {d['define']}." + (f" {d['hint']}" if hint else ""))
    fmt = "{" + ", ".join(f'"{PARAM_DEFS[p]["key"]}": {{"estimate": <number>, "sigma": <number>}}' for p in KIND_PARAMS[kind]) + "}"
    return ("Look at the image(s) of the scene and estimate:\n" + "\n".join(lines) +
            "\nEvery parameter needs BOTH an estimate and a sigma (sigma > 0, same units). Answer in exactly this JSON format:\n" + fmt)
