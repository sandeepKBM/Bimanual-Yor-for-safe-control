"""F3/PR10 (11-13): probe.l_admit_max does not raise for the table-behind tasks and returns a finite load above the empty mass."""
import math

from world_model.v2.nero.contents import probe as PR


def test_l_admit_max_tray6_behind_and_box_behind():
    for task in ("tray6_behind", "box_behind"):
        v = PR.l_admit_max(task)
        assert math.isfinite(v) and v >= PR.empty_mass(task)
