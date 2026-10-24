"""tamp/planner.py -- layer 2: PDDL spec -> PDDLStream plan whose every continuous value is certified by our checkers (streams.py).  SIMULATION ONLY.

    from world_model.v2.house.tamp import planner as TP
    res = TP.solve_task(world, "(On tray desk)")          # goal = a PDDL goal formula over the domain's predicates and the world's names
    res["plan"]  -> [(action_name, args), ...]  (tamp/types.py contract), res["stream_log"] -> what each checker admitted / rejected

The goal is the part a VLM will write later; the INIT facts come from the world state (world.py), never from the spec.
PDDLStream: /data/local/ss5772/external/pddlstream (env PDDLSTREAM_ROOT), Fast Downward built in its downward/builds/release.  GPL-3.0 (kept out of this repo).
"""
from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile
import time
from pathlib import Path

from .streams import Streams
from .world import SKILLS, World

PDDLSTREAM_ROOT = os.environ.get("PDDLSTREAM_ROOT", "/data/local/ss5772/external/pddlstream")
PDDL_DIR = Path(__file__).resolve().parent / "pddl"
WORK_ROOT = os.environ.get("TAMP_WORK_ROOT", "/data/local/ss5772/molmospaces/tamp/pddl_work")


def _import_pddlstream():
    if PDDLSTREAM_ROOT not in sys.path:
        sys.path.insert(0, PDDLSTREAM_ROOT)
    from pddlstream.algorithms.meta import solve
    from pddlstream.language.constants import PDDLProblem
    from pddlstream.algorithms import downward
    _bind_fd_translate(downward.TRANSLATE_PATH)
    return solve, PDDLProblem


def _bind_fd_translate(tdir: str) -> None:
    """Make Fast Downward's translate modules use FD's own copies of top-level names that this repo also defines.

    FD's translate/ is APPENDED to sys.path and uses bare names (e.g. `import constraints` in invariants.py).  The repo
    package world_model/v2/constraints is importable as `constraints` whenever world_model/v2 is on sys.path (pytest does
    that), so FD got our package ("module 'constraints' has no attribute 'Assignment'").  Fix: pre-import the FD modules
    that use a clashing name with FD's version bound, then put the repo's module back under that name."""
    import importlib
    import importlib.util

    tdir = os.path.realpath(tdir)
    tpath = Path(tdir)
    names = {p.stem for p in tpath.glob("*.py")} | {p.name for p in tpath.iterdir() if (p / "__init__.py").exists()}

    def _is_fd(mod_or_origin) -> bool:
        f = mod_or_origin if isinstance(mod_or_origin, str) else getattr(mod_or_origin, "__file__", None)
        return bool(f) and os.path.realpath(f).startswith(tdir + os.sep)

    clash = set()
    for n in names:
        if n in sys.modules:
            if not _is_fd(sys.modules[n]):
                clash.add(n)
            continue
        try:
            spec = importlib.util.find_spec(n)
        except (ImportError, ValueError):
            continue
        if spec is not None and spec.origin and not _is_fd(spec.origin):
            clash.add(n)
    if not clash:
        return
    # FD modules that import a clashing name (scan their source; only top-level imports matter here)
    dependents = []
    for p in tpath.glob("*.py"):
        src = p.read_text(errors="ignore")
        if any(f"import {n}" in src for n in clash):
            dependents.append(p.stem)

    saved_path, saved_argv = sys.path[:], sys.argv[:]
    saved = {n: sys.modules.pop(n) for n in clash if n in sys.modules}
    try:
        sys.path.insert(0, tdir)
        sys.argv = sys.argv[:1] + ["domain.pddl", "problem.pddl"]  # FD options.py parses argv at import (as pddlstream does)
        fd = {n: importlib.import_module(n) for n in sorted(clash)}
        for d in sorted(dependents):  # not yet loaded -> imported now with FD's names in sys.modules
            importlib.import_module(d)
    finally:
        sys.path[:] = saved_path
        sys.argv = saved_argv
        for n in clash:  # the repo name goes back to the repo module
            if n in saved:
                sys.modules[n] = saved[n]
            elif n in sys.modules and _is_fd(sys.modules[n]):
                sys.modules.pop(n)
    # Modules loaded earlier keep the wrong object as a global, and other FD modules hold references to those module
    # objects, so re-importing is not enough: re-point the global in place.
    for m in list(sys.modules.values()):
        if m is None or not _is_fd(m):
            continue
        for n, fd_mod in fd.items():
            cur = m.__dict__.get(n)
            if cur is not None and cur is not fd_mod and getattr(cur, "__name__", None) == n and not _is_fd(cur):
                setattr(m, n, fd_mod)


# ---------------------------------------------------------------------------------------------- goal parsing
def parse_goal(text: str):
    """PDDL goal formula (s-expression) -> PDDLStream nested tuple, e.g. "(and (On tray desk) (HandsEmpty))" -> ("and", ("On", "tray", "desk"), ("HandsEmpty",)).
    Keywords and/or/not/exists/forall are lower-cased; predicate and object names are kept as written (they must match the domain / world names)."""
    toks = text.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def rd():
        nonlocal pos
        if pos >= len(toks):
            raise ValueError(f"unexpected end of goal: {text!r}")
        t = toks[pos]
        pos += 1
        if t == "(":
            out = []
            while pos < len(toks) and toks[pos] != ")":
                out.append(rd())
            if pos >= len(toks):
                raise ValueError(f"unbalanced parentheses in goal: {text!r}")
            pos += 1
            if out and isinstance(out[0], str) and out[0].lower() in ("and", "or", "not", "exists", "forall", "imply"):
                out[0] = out[0].lower()
            return tuple(out)
        if t == ")":
            raise ValueError(f"unexpected ')' in goal: {text!r}")
        return t

    g = rd()
    if pos != len(toks):
        raise ValueError(f"trailing tokens in goal: {text!r}")
    if not isinstance(g, tuple):
        raise ValueError(f"goal must be a parenthesised formula: {text!r}")
    return g


def _goal_names(g):
    if isinstance(g, str):
        return set()
    out = set()
    for x in g[1:]:
        out |= {x} if isinstance(x, str) else _goal_names(x)
    return out


# ---------------------------------------------------------------------------------------------- problem
def init_facts(world: World):
    """the INIT state from the world model (layer-2 input that is NOT written by the spec)."""
    init = [("CanMove",), ("Conf", world.q0), ("AtConf", world.q0)]
    holding = []
    for o, it in world.items.items():
        init.append(("Item", o))
        hg = it.extra.get("held_grasp")
        if it.extra.get("grasped") and hg is None:
            raise ValueError(f"{o} is grasped in the sim but world_from_sim got no held=(item, Grasp): cannot plan from an unknown grasp")
        if hg is not None:
            # HELD item.  Carried over from the plan that picked it: Grasp(o, g) -- sound because g was admitted by C3 for the item's whole
            # (mass, CoM, mu, accel) box and has an executor skill; we re-check that the box is still covered by g's certificate and the skill
            # still exists, else refuse.  Nothing else is asserted: Dock/Reach/Supported/routes are NOT carried, the streams re-certify them
            # for the new start conf.  Pose/AtPose are omitted (the item is in the hand), HandsEmpty is omitted (the hand is not empty).
            if hg.mass_range[0] > it.mass_range[0] + 1e-9 or hg.mass_range[1] < it.mass_range[1] - 1e-9:
                raise ValueError(f"held grasp certificate mass box {hg.mass_range} does not cover the item's {it.mass_range}")
            if hg.mu_range[0] > it.mu_range[0] + 1e-9 or hg.mu_range[1] < it.mu_range[1] - 1e-9:     # grip margin is set by the LOWEST mu
                raise ValueError(f"held grasp certificate mu box {hg.mu_range} does not cover the item's {it.mu_range}")
            if SKILLS.get(o) is not None and hg.name not in SKILLS[o]:
                raise ValueError(f"no executor skill for held grasp {hg.name}")
            init += [("Grasp", o, hg), ("Holding", o, hg)]
            holding.append(o)
            continue
        if it.pose is not None:
            init += [("Pose", o, it.pose), ("AtPose", o, it.pose)]
            if it.pose.surface is not None:
                init.append(("Supported", o, it.pose, it.pose.surface))
    if len(holding) > 1:
        raise ValueError("planning with more than one held item is not supported (domain has one gripper pair)")
    if not holding:
        init.append(("HandsEmpty",))
    for s in world.surfaces:
        init.append(("Surface", s))
    return init


def build_problem(world: World, goal, streams: Streams | None = None):
    _, PDDLProblem = _import_pddlstream()
    streams = streams or Streams(world)
    g = parse_goal(goal) if isinstance(goal, str) else goal
    known = set(world.items) | set(world.surfaces)
    unknown = sorted(n for n in _goal_names(g) if n not in known and not n.startswith("?"))
    if unknown:
        raise ValueError(f"goal names not in the world: {unknown} (items {sorted(world.items)}, surfaces {sorted(world.surfaces)})")
    domain = (PDDL_DIR / "fetch_domain.pddl").read_text()
    stream = (PDDL_DIR / "fetch_stream.pddl").read_text()
    return PDDLProblem(domain, {}, stream, streams.stream_map(), init_facts(world), g), streams


@contextlib.contextmanager
def _workdir():
    """PDDLStream / Fast Downward write temp/ (and statistics/) under the CWD: run every solve in its own directory so parallel runs never clobber."""
    os.makedirs(WORK_ROOT, exist_ok=True)
    d = tempfile.mkdtemp(prefix=f"solve_{os.getpid()}_", dir=WORK_ROOT)
    old = os.getcwd()
    os.chdir(d)
    try:
        yield d
    finally:
        os.chdir(old)


def _plain(x):
    """PDDLStream returns python values (our dataclasses / strings) inside its objects; unwrap if needed."""
    return getattr(x, "value", x)


def solve_task(world: World, goal, algorithm: str = "adaptive", max_time: float = 60.0, verbose: bool = False, streams: Streams | None = None,
               unit_costs: bool = True, **kw) -> dict:
    """-> dict(solved, plan [(name, args)], cost, time_s, goal, stream_log, checker_stats, workdir, pddlstream_out (tail if not verbose))."""
    solve, _ = _import_pddlstream()
    t0 = time.time()
    problem, streams = build_problem(world, goal, streams)
    buf = io.StringIO()
    with _workdir() as wd:
        ctx = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(buf)
        with ctx:
            sol = solve(problem, algorithm=algorithm, max_time=max_time, unit_costs=unit_costs, verbose=verbose, **kw)
    plan, cost, _certs = sol
    out = None if plan is None else [(a.name, tuple(_plain(x) for x in a.args)) for a in plan]
    return dict(solved=out is not None, plan=out, cost=float(cost), time_s=round(time.time() - t0, 3), goal=goal,
                stream_log=list(streams.log), checker_stats={k: v for k, v in world.stats.items() if isinstance(v, list)}, workdir=wd,
                pddlstream_out=None if verbose else buf.getvalue()[-3000:])
