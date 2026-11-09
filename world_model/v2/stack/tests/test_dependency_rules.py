"""Import-rule test: violations must equal the checked-in exceptions (the list can only shrink)."""
from world_model.v2.stack.tests import depcheck


def _keys():
    now = {depcheck.key(v) for v in depcheck.dedup(depcheck.violations())}
    known = {(x["rule"], x["importer"], x["imported"]) for x in depcheck.load_exceptions()}
    return now, known


def test_no_new_violations():
    now, known = _keys()
    new = sorted(now - known)
    assert not new, "new dependency-rule violations (rule, importer, imported):\n" + "\n".join(map(str, new))


def test_no_stale_exceptions():
    now, known = _keys()
    stale = sorted(known - now)
    assert not stale, ("exceptions that no longer occur; remove them (rerun "
                       "`python -m world_model.v2.stack.tests.depcheck --write-exceptions`):\n"
                       + "\n".join(map(str, stale)))
