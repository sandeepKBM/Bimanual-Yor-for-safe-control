"""tests for contents/carry_cert.py: arm-side rows only by default (user 11-02 pm), fixed per-cell seeds."""
from world_model.v2.nero.contents import carry_cert as CC, envelope as E, scenes as SC


def test_cell_seed_depends_only_on_task_cell_index():
    a = CC.cell_seed("box", (2.0, 2.5), 3)
    assert a == CC.cell_seed("box", (2.0, 2.5), 3)
    assert len({CC.cell_seed("box", (2.0, 2.5), i) for i in range(20)}) == 20
    assert a != CC.cell_seed("box", (2.5, 3.0), 3) and a != CC.cell_seed("tray6", (2.0, 2.5), 3)


def test_default_static_rows_have_no_claw_twist():
    cell = (1.5, 2.0)
    st = CC.static_cell("box", cell, SC.BoxWorld())
    assert st["T_cap"] is None
    for seg in ("lift", "carry", "place"):
        assert "torque_payload" in st["rows"][seg] and "torque_payload_twist" not in st["rows"][seg]
    old = CC.static_cell("box", cell, SC.BoxWorld(), T_cap=E.T_CAP)
    assert "torque_payload_twist" in old["rows"]["lift"]
    assert st["u_nom"] <= old["u_nom"] + 1e-9          # dropping a row can only lower the utilisation
    assert st["u_one_nom"] <= old["u_one_nom"] + 1e-9
