"""test helper: the per-plan load tilt input of the tilt-aware collision row (USER 11-17), MEASURED in simulation as nero/contents/probe_select measures it
(tilt_measure.measure_plan -> collision_rows.tilt_from_table).  n is a TEST choice (ASSUMED, smaller than probe_select.TILT_N = 20, to bound the test time):
the max over n worlds of the cell is a measurement, not a bound over the cell.  Memoised per process."""
import os
import tempfile

_MEMO = {}


def measured_tilt(task, cell, a=0.5, n=4, procs=4, wp=None):
    """-> collision_rows.TiltSpec of the default plan of `task` on the belief cell `cell` = (lo, hi) kg at plan acceleration a."""
    key = (task, tuple(cell), float(a), int(n), wp is None)
    if key not in _MEMO:
        from world_model.v2.adapters import collision_rows as RCOL, rule1_contents as RC
        from world_model.v2.nero.contents import tilt_measure as TM
        wp = wp if wp is not None else RC.get_ctx(task, "closed_chain").wp          # None for the fixed-base tasks, the default waypoint plan of the table-behind tasks
        old = os.environ.get("BIMANUAL_PLANNER")
        os.environ["BIMANUAL_PLANNER"] = "closed_chain"          # the spawn workers inherit it: the per-piece rotation record exists only for closed_chain plans
        try:
            tab, info = TM.measure_plan(task, wp, tuple(cell), float(a), n=int(n), procs=int(procs), out_dir=tempfile.mkdtemp(prefix="tilt_input_"), plan_id="test_tilt_input")
        finally:
            if old is None:
                os.environ.pop("BIMANUAL_PLANNER", None)
            else:
                os.environ["BIMANUAL_PLANNER"] = old
        _MEMO[key] = RCOL.tilt_from_table(tab, TM.cell_key(task, tuple(cell)), float(a), source=info["table_path"])
    return _MEMO[key]
