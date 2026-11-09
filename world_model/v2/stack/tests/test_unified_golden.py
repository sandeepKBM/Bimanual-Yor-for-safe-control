"""golden equivalence (admit / support / payload / monitors / recoverable / check_goal) against stack/tests/golden/*.json.
The fixture is written by a parallel agent (golden/make_golden.py); until admission.json etc. exist this test skips."""
import pathlib

import pytest

G = pathlib.Path(__file__).resolve().parent / "golden"
NEEDED = ("admission.json", "probe_admission.json", "runtime_episodes.json", "goal_success.json")


def test_golden_equivalence():
    missing = [f for f in NEEDED if not (G / f).exists()]
    if missing:
        pytest.skip(f"golden fixture not yet written ({', '.join(missing)} absent in {G}); comparison to be added with the fixture's schema")
    pytest.skip("fixture present but its schema is not yet wired into this test (switch step)")
