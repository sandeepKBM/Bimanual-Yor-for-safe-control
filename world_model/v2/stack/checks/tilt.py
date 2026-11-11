"""checks/tilt.py -- tilt / recovery / dynamic-tilt rows: NOT BUILT (checker audit 11-06).  Registered as MISSING so coverage shows it."""
from __future__ import annotations

from world_model.v2.stack.checks.row import Row, missing


def tilt_row(*args, **kw) -> Row:
    return missing("tilt", "tilt / recovery rows not built (checker audit 11-06)")
