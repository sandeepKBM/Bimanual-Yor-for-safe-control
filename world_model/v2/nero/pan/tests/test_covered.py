"""Smoke tests for the covered pan scene (draft by Codex gpt-5.6-luna 10-28, reviewed). CPU only, no rendering."""
import numpy as np
import mujoco
import pytest

from world_model.v2.nero.pan import covered as CV


@pytest.fixture(scope="module")
def covered_table():
    return CV.table()


def test_placements_settle_with_lid_and_hidden_puck_on_bottom(covered_table):
    for name, result in covered_table.items():
        assert result["lid_on"] is True
        if name != "none":
            assert abs(result["hidden_rest_z_above_bottom_m"] - 0.01) <= 0.003


def test_mass_matches_hidden_loads(covered_table):
    none_mass = covered_table["none"]["mass_total_kg"]

    for name in ("near_handle", "far_centre", "far_left", "far_right"):
        assert abs(covered_table[name]["mass_total_kg"] - (none_mass + 0.2)) <= 1e-3

    assert abs(covered_table["overload"]["mass_total_kg"] - (none_mass + 0.6)) <= 1e-3


def test_roll_probe_sign_and_centre_symmetry(covered_table):
    left = covered_table["far_left"]["probe_B_roll_Nm"]
    right = covered_table["far_right"]["probe_B_roll_Nm"]

    assert left * right < 0
    assert abs(abs(left) - abs(right)) <= 0.05 * max(abs(left), abs(right))

    for name in ("none", "near_handle", "far_centre", "overload"):
        assert abs(covered_table[name]["probe_B_roll_Nm"]) < 0.01


def test_lift_handle_moment_ordering(covered_table):
    moments = [covered_table[name]["lift_handle_moment_Nm"] for name in
               ("none", "near_handle", "far_centre", "overload")]
    assert moments[0] < moments[1] < moments[2] < moments[3]


def test_handle_raise_probe_confound(covered_table):
    near = covered_table["near_handle"]
    overload = covered_table["overload"]

    assert abs(near["probe_A_pitch_about_pivot_Nm"] -
               overload["probe_A_pitch_about_pivot_Nm"]) < 0.1
    assert abs(near["lift_handle_moment_Nm"] -
               overload["lift_handle_moment_Nm"]) > 0.8


def test_default_counter_aabb_does_not_intersect_stove():
    m = CV.compose()

    def geom_aabb(model, name):
        gid = model.geom(name).id
        compiled = model.geom_aabb[gid]
        centre = model.geom_pos[gid] + compiled[:3]
        half = compiled[3:]
        return centre - half, centre + half

    counter_min, counter_max = geom_aabb(m, "counter_near")
    stove_min, stove_max = geom_aabb(m, "stove_pad")

    assert np.any(counter_max < stove_min) or np.any(stove_max < counter_min)
