"""Static import-rule checker for world_model/v2 (ast only; nothing is imported or executed).

Usage (run from the worktree root):
  python -m world_model.v2.stack.tests.depcheck --summary
  python -m world_model.v2.stack.tests.depcheck --write-exceptions
"""
from __future__ import annotations

import ast
import json
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path

V2 = Path(__file__).resolve().parents[2]            # .../world_model/v2
ROOT = V2.parents[1]                                # worktree root
PKG = "world_model.v2"
EXCEPTIONS_FILE = Path(__file__).with_name("dependency_exceptions.json")
SKIP_DIRS = {"tests", "results", "archive", "__pycache__", "scratch", "tmp", "golden", "plan_cache", "web"}
DEFAULT_REASON = "pre-unification 11-09; remove in switch step"

# ---------------------------------------------------------------- rules (data)
STACK_PURE_PARTS = {"types", "tasks", "goal", "state", "outcome", "registry", "checks"}
STACK_ENGINE_PARTS = {"planners", "kin", "proposer", "eval", "executor", "producers", "skills"}
# the only nero modules stack_engine may import (robot / scene model); exact module or its submodules
ROBOT_MODEL = {
    "world_model.v2.nero.config",
    "world_model.v2.nero.democ.scene",
    "world_model.v2.nero.contents.scenes",
    "world_model.v2.nero.contents.envelope",
    "world_model.v2.nero.nyu_gripper",
}
# LLM backend modules: stack may reach at most ONE (rule single_llm_backend; today two -> exceptions)
LLM_BACKENDS = {"world_model.v2.house.lang.backend", "world_model.v2.nero.pan.llm"}
FORBIDDEN_FOR_ENGINE = {"nero_base", "nero_contents", "nero_pan", "agentchord_nero", "agentchord_coffee",
                        "house", "vlm", "flmpc", "teleop", "adapters"}
FORBIDDEN_FOR_NERO_CONTENTS = {"agentchord_nero", "agentchord_coffee", "flmpc", "teleop"}
FORBIDDEN_FOR_ADAPTERS = {"flmpc", "teleop"}
TRANSITIVE_TARGETS = {"flmpc", "teleop"}
# agentchord_nero -> nero_contents modules that are allowed: {imported module: reason}
ALLOW_AGENTCHORD_NERO_TO_CONTENTS: dict[str, str] = {}
PROBE_MODULE_FILE = "nero/contents/probe.py"        # the only file that may assign PROBE_METHOD

RULES = {
    "stack_pure_no_outside": "stack_pure (types,tasks,goal,state,outcome,registry,checks) imports only stack_pure",
    "stack_engine_no_episode": "stack_engine may import stack_pure + ROBOT_MODEL only; no episode/controller/checker/probe, agentchord, house, vlm, flmpc, teleop, adapters",
    "single_llm_backend": "stack may reach at most one LLM backend module",
    "adapters_no_flmpc_teleop": "adapters (adapters/**, stack/universes) must not import flmpc/teleop (direct)",
    "adapters_transitive": "adapters must not reach flmpc/teleop transitively",
    "nero_contents_no_forbidden": "nero/contents must not import agentchord_*, flmpc (except the basic tracker entry flmpc.basic), teleop (direct)",
    "nero_contents_transitive": "nero/contents must not reach flmpc/teleop through other modules (flmpc.basic, the basic tracker skill, is the only allowed flmpc entry)",
    "agentchord_nero_no_contents": "agentchord_probe/nero must not import nero/contents internals (ALLOW table)",
    "probe_method_single_writer": "only nero/contents/probe.py may assign PROBE_METHOD",
}


def in_robot_model(mod: str) -> bool:
    return any(mod == m or mod.startswith(m + ".") for m in ROBOT_MODEL)


# ---------------------------------------------------------------- buckets
def bucket_of_parts(p: tuple[str, ...]) -> str:
    """p = path parts below world_model/v2 with no .py suffix (package __init__ dropped)."""
    if not p:
        return "other"
    top = p[0]
    if top == "adapters":
        return "adapters"
    if top == "stack":
        if len(p) > 1 and p[1] == "universes":
            return "adapters"
        if len(p) == 1 or p[1] in STACK_PURE_PARTS:
            return "stack_pure"
        return "stack_engine"           # planners, kin, proposer, eval, executor, producers, skills, anything new
    if top == "nero":
        if len(p) > 1 and p[1] == "contents":
            return "nero_contents"
        if len(p) > 1 and p[1] == "pan":
            return "nero_pan"
        return "nero_base"
    if top == "agentchord_probe":
        if len(p) > 1 and p[1] == "nero":
            return "agentchord_nero"
        if len(p) > 1 and p[1] == "coffee_tray":
            return "agentchord_coffee"
        return "other"
    if top == "flmpc":
        # USER 11-09: the BASIC tracker (flmpc/basic.py, flmpc_basic skill) is the ONLY flmpc entry research code may reach; the full FL-MPC
        # (mpc_fl, qp, _sc, ...) stays for teleop / cup demo / latent_hj / house only.  The walk below does not look inside the entry point.
        return "flmpc_basic" if (len(p) > 1 and p[1] == "basic") else "flmpc"
    if top == "house":
        return "teleop" if (len(p) > 1 and p[1] == "teleop") else "house"
    if top == "vlm":
        return "vlm"
    if any(x.startswith("teleop") for x in p):
        return "teleop"
    return "other"


def bucket_of_module(mod: str) -> str:
    if mod == PKG:
        return "other"
    if mod.startswith(PKG + "."):
        return bucket_of_parts(tuple(mod[len(PKG) + 1:].split(".")))
    if mod.split(".")[-1] == "mpc_fl":          # legacy alias of flmpc (experiments/ab_smooth_vs_bangbang)
        return "flmpc"
    return "external"


# ---------------------------------------------------------------- walking
def iter_files():
    for f in sorted(V2.rglob("*.py")):
        rel = f.relative_to(V2)
        if any(part in SKIP_DIRS for part in rel.parts[:-1]):
            continue
        yield f


def module_of(f: Path) -> str:
    parts = list(f.relative_to(ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def known_modules() -> set[str]:
    return {module_of(f) for f in iter_files()}


class Edge:
    __slots__ = ("file", "line", "src", "dst", "dst_bucket", "src_bucket")

    def __init__(self, file, line, src, dst):
        self.file, self.line, self.src, self.dst = file, line, src, dst
        self.src_bucket, self.dst_bucket = bucket_of_module(src), bucket_of_module(dst)


def collect():
    """Return (edges, probe_method_writes). Every import incl. function-level and relative."""
    mods = known_modules()
    edges: list[Edge] = []
    writes: list[tuple[str, int]] = []
    for f in iter_files():
        rel = str(f.relative_to(V2))
        src = module_of(f)
        is_pkg = f.name == "__init__.py"
        pkg_parts = src.split(".") if is_pkg else src.split(".")[:-1]
        try:
            tree = ast.parse(f.read_text(encoding="utf-8"), filename=str(f))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for a in n.names:
                    edges.append(Edge(rel, n.lineno, src, a.name))
            elif isinstance(n, ast.ImportFrom):
                if n.level:
                    base = pkg_parts[: len(pkg_parts) - (n.level - 1)]
                    mod = ".".join(base + ([n.module] if n.module else []))
                else:
                    mod = n.module or ""
                for a in n.names:
                    sub = f"{mod}.{a.name}"
                    edges.append(Edge(rel, n.lineno, src, sub if sub in mods else mod))
            elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                targets = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in targets:
                    for x in ast.walk(t):
                        if isinstance(x, ast.Attribute) and x.attr == "PROBE_METHOD":
                            writes.append((rel, n.lineno))
    edges = [e for e in edges if e.dst_bucket != "external"]
    return edges, writes


# ---------------------------------------------------------------- violations
def violations() -> list[dict]:
    edges, writes = collect()
    out: list[dict] = []

    def add(rule, e, detail=""):
        out.append({"rule": rule, "file": e.file, "line": e.line, "imported": e.dst, "detail": detail})

    for e in edges:
        sb, db = e.src_bucket, e.dst_bucket
        is_llm = e.dst in LLM_BACKENDS
        if sb in ("stack_pure", "stack_engine") and is_llm:
            add("single_llm_backend", e, f"{sb} -> {e.dst}")
        elif sb == "stack_pure" and db != "stack_pure":
            add("stack_pure_no_outside", e, f"{sb} -> {db}")
        elif sb == "stack_engine":
            if db in FORBIDDEN_FOR_ENGINE and not in_robot_model(e.dst):
                add("stack_engine_no_episode", e, f"{sb} -> {db}")
        if sb == "adapters" and db in FORBIDDEN_FOR_ADAPTERS:
            add("adapters_no_flmpc_teleop", e, f"{sb} -> {db}")
        if sb == "nero_contents" and db in FORBIDDEN_FOR_NERO_CONTENTS:
            add("nero_contents_no_forbidden", e, f"{sb} -> {db}")
        if sb == "agentchord_nero" and db == "nero_contents" and e.dst not in ALLOW_AGENTCHORD_NERO_TO_CONTENTS:
            add("agentchord_nero_no_contents", e, f"{sb} -> {db}")

    # transitive: <bucket> -> (module outside it) -> ... -> flmpc/teleop
    graph: dict[str, list[Edge]] = defaultdict(list)
    for e in edges:
        graph[e.src].append(e)
    seen = set()
    for src_bucket, rule, direct_forbidden in (
            ("nero_contents", "nero_contents_transitive", FORBIDDEN_FOR_NERO_CONTENTS),
            ("adapters", "adapters_transitive", FORBIDDEN_FOR_ADAPTERS)):
        for e in edges:
            if e.src_bucket != src_bucket or e.dst_bucket in {src_bucket} | direct_forbidden:
                continue
            prev = {e.dst: None}
            q = deque([e.dst])
            hit = None
            while q and hit is None:
                m = q.popleft()
                if bucket_of_module(m) == "flmpc_basic":      # the allowed entry point: do not look inside it
                    continue
                for e2 in graph.get(m, ()):
                    if e2.dst_bucket in TRANSITIVE_TARGETS:
                        hit = (m, e2)
                        break
                    if e2.dst not in prev:
                        prev[e2.dst] = m
                        q.append(e2.dst)
            if hit:
                chain, m = [], hit[0]
                while m is not None:
                    chain.append(m)
                    m = prev[m]
                chain = chain[::-1] + [hit[1].dst]
                k = (rule, e.file, e.dst)
                if k not in seen:
                    seen.add(k)
                    add(rule, e, " -> ".join(c.replace(PKG + ".", "") for c in chain))

    for rel, line in writes:
        if rel != PROBE_MODULE_FILE:
            out.append({"rule": "probe_method_single_writer", "file": rel, "line": line,
                        "imported": "PROBE_METHOD", "detail": "assignment to *.PROBE_METHOD"})
    return out


def key(v) -> tuple[str, str, str]:
    return (v["rule"], v["file"], v["imported"])


def dedup(vs: list[dict]) -> list[dict]:
    seen, res = set(), []
    for v in vs:
        if key(v) not in seen:
            seen.add(key(v))
            res.append(v)
    return res


def load_exceptions() -> list[dict]:
    if not EXCEPTIONS_FILE.exists():
        return []
    return json.loads(EXCEPTIONS_FILE.read_text())


def write_exceptions() -> int:
    old = {(x["rule"], x["importer"], x["imported"]): x["reason"] for x in load_exceptions()}
    ents = [{"rule": v["rule"], "importer": v["file"], "imported": v["imported"],
             "reason": old.get(key(v), "two LLM backends; unify later" if v["rule"] == "single_llm_backend" else DEFAULT_REASON)} for v in dedup(violations())]
    ents.sort(key=lambda x: (x["rule"], x["importer"], x["imported"]))
    EXCEPTIONS_FILE.write_text(json.dumps(ents, indent=1) + "\n")
    return len(ents)


def summary() -> str:
    vs = dedup(violations())
    c = Counter(v["rule"] for v in vs)
    lines = [f"{'rule':34s} count", "-" * 42]
    lines += [f"{r:34s} {c.get(r, 0)}" for r in RULES]
    lines.append(f"{'TOTAL':34s} {len(vs)}")
    return "\n".join(lines)


if __name__ == "__main__":
    if "--write-exceptions" in sys.argv:
        print(f"wrote {write_exceptions()} exceptions -> {EXCEPTIONS_FILE}")
    elif "--list" in sys.argv:
        for v in dedup(violations()):
            print(f"{v['rule']}  {v['file']}:{v['line']} -> {v['imported']}  [{v['detail']}]")
    else:
        print(summary())
