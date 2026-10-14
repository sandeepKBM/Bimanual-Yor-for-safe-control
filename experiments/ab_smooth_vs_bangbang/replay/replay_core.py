"""replay_core.py -- shared record / replay / compare machinery for the FL-MPC bitwise replay harness.

SIMULATION / OFFLINE ONLY. Nothing here imports YOR/robot/* or YOR/extra/*, opens a socket, or touches
hardware. New file; edits no existing file (everything is done by run-time monkeypatching in OUR scripts).

A *record* is one call of `<controller>.step(model, data, ...)` captured with everything it reads:
  - ctor        : class name + constructor args/kwargs (so the replayer rebuilds the controller),
  - state_diff  : controller attributes (pre-call) that differ from a freshly constructed twin
                  (warm starts a_*_prev, _a_queue, _cup_axis_local, counters, ValueFLMPC._rows, ...),
  - data        : every small ndarray field of MjData (qpos, qvel, qM, qfrc_bias, qfrc_passive, xpos, xmat,
                  cdof, subtree_com, ...; stale derived fields included, because step() reads them as-is),
  - sc          : the sim_common module globals step() reads (TAU_MAX, DT),
  - call        : the positional args (after model, data) and kwargs of step(),
  - solver      : digests of every solve_constrained_box_qp call (inputs, outputs) in order,
  - ret         : the full return value,
  - post_changed: controller attributes that the call changed (side effects).
Records live as <dir>/rec_<NNNNN>.pkl.gz (+ <dir>/models/<sha>.mjb for the MuJoCo model).
"""
from __future__ import annotations

import copy
import gzip
import hashlib
import os
import pickle
import sys
import weakref
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
for _p in (REPO, REPO / "scripts", REPO / "experiments" / "ab_smooth_vs_bangbang",
           REPO / "experiments" / "realtime", REPO / "eval", REPO / "world_model" / "v2" / "latent_hj"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import mujoco  # noqa: E402

DEFAULT_DATA_DIR = "/common/users/ss5772/bimanual_data/flmpc_replay"
SKIP_STATE_KEYS = {"model"}
# MjData ndarray fields that are NOT read by FLMPC.step (collision/constraint buffers; big) -- skipped.
_DATA_SKIP_PREFIX = ("efc_", "iefc_", "bvh", "contact", "arena")
_DATA_MAX_BYTES = 200_000


# --------------------------------------------------------------------------- digests / equality
def digest(a) -> str:
    if a is None:
        return "None"
    a = np.ascontiguousarray(a)
    h = hashlib.sha1()
    h.update(str((a.dtype.str, a.shape)).encode())
    h.update(a.tobytes())
    return h.hexdigest()[:20]


def _is_simple(v) -> bool:
    if v is None or isinstance(v, (bool, int, float, str, bytes, np.generic)):
        return True
    if isinstance(v, np.ndarray):
        return v.dtype != object
    if isinstance(v, (list, tuple)):
        return all(_is_simple(x) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, (str, int, float, tuple)) and _is_simple(x) for k, x in v.items())
    return False


def snap_state(obj) -> dict:
    """Deep copy of the controller's plain-data attributes (everything except the model and non-plain
    members such as numba workspaces / lean-dynamics objects, which are rebuilt by the constructor)."""
    out = {}
    for k, v in vars(obj).items():
        if k in SKIP_STATE_KEYS or not _is_simple(v):
            continue
        out[k] = copy.deepcopy(v)
    return out


class Cmp:
    """Bitwise (NaN-aware array_equal) comparison with a tolerance fallback report."""

    def __init__(self, atol: float = 1e-9):
        self.atol = atol
        self.n = 0
        self.n_exact = 0           # array_equal (NaN-aware)
        self.n_bytes = 0           # even byte-identical (distinguishes -0.0 / NaN payloads)
        self.n_tol = 0             # equal within atol (superset of exact)
        self.max_abs = 0.0
        self.fails = []

    def eq(self, a, b, path="") -> bool:
        if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
            if len(a) != len(b) or type(a) is not type(b):
                return self._fail(path, "len/type %s %d vs %s %d" % (type(a).__name__, len(a), type(b).__name__, len(b)))
            return all([self.eq(x, y, f"{path}[{i}]") for i, (x, y) in enumerate(zip(a, b))])
        if isinstance(a, dict) and isinstance(b, dict):
            if set(a) != set(b):
                return self._fail(path, f"keys differ {sorted(set(a) ^ set(b), key=str)}")
            return all([self.eq(a[k], b[k], f"{path}.{k}") for k in a])
        if a is None or b is None or isinstance(a, (str, bytes)) or isinstance(b, (str, bytes)):
            self.n += 1
            if a is None or b is None:
                ok = a is None and b is None
            else:
                ok = isinstance(a, (str, bytes)) and isinstance(b, (str, bytes)) and a == b
            if ok:
                self.n_exact += 1; self.n_bytes += 1; self.n_tol += 1
                return True
            return self._fail(path, f"{type(a).__name__} vs {type(b).__name__}")
        aa, bb = np.asarray(a), np.asarray(b)
        self.n += 1
        if aa.shape != bb.shape or aa.dtype != bb.dtype:
            return self._fail(path, f"shape/dtype {aa.shape}{aa.dtype} vs {bb.shape}{bb.dtype}")
        exact = bool(np.array_equal(aa, bb, equal_nan=True)) if aa.dtype.kind in "fc" else bool(np.array_equal(aa, bb))
        by = aa.tobytes() == bb.tobytes()
        d = 0.0
        if aa.dtype.kind in "fc" and aa.size:
            with np.errstate(invalid="ignore"):
                dd = np.abs(aa - bb)
            dd = dd[~(np.isnan(aa) & np.isnan(bb))]
            d = float(np.max(dd)) if dd.size and not np.isnan(dd).any() else (float("inf") if dd.size else 0.0)
        self.max_abs = max(self.max_abs, d if np.isfinite(d) else 1e300)
        if exact:
            self.n_exact += 1
        if by:
            self.n_bytes += 1
        if d <= self.atol:
            self.n_tol += 1
        if not exact:
            self.fails.append((path, d))
            return False
        return True

    def _fail(self, path, msg):
        self.fails.append((path, msg))
        return False


# --------------------------------------------------------------------------- MuJoCo data / model
def snap_data(data, ref=None) -> dict:
    """ref: a fresh MjData of the same model; fields whose size is dynamic (island/constraint sized) are skipped."""
    out = {}
    for k in dir(data):
        if k.startswith("_") or k.startswith(_DATA_SKIP_PREFIX):
            continue
        try:
            v = getattr(data, k)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.size and v.nbytes <= _DATA_MAX_BYTES and v.flags.writeable:
            if ref is not None and getattr(ref, k).shape != v.shape:
                continue
            out[k] = v.copy()
    out["__time"] = float(data.time)
    return out


def restore_data(model, snap: dict):
    d = mujoco.MjData(model)
    for k, v in snap.items():
        if k == "__time":
            d.time = v
        else:
            getattr(d, k)[...] = v
    return d


def model_bytes(model) -> bytes:
    buf = np.empty(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return buf.tobytes()


class ModelStore:
    """MuJoCo model blobs (compiled binary, ~64 MB raw each) stored once per distinct sha, gzip-compressed."""

    _GLOBAL = {}          # (abs path) -> MjModel, shared by all stores in the process

    def __init__(self, models_dir: Path, search=()):
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.search = [self.dir] + [Path(x) for x in search]
        self._cache = ModelStore._GLOBAL

    def put(self, model) -> str:
        b = model_bytes(model)
        sha = hashlib.sha1(b).hexdigest()[:16]
        if not any((d / f"{sha}.mjb.gz").exists() or (d / f"{sha}.mjb").exists() for d in self.search):
            with gzip.open(self.dir / f"{sha}.mjb.gz", "wb", compresslevel=3) as f:
                f.write(b)
        return sha

    def get(self, sha):
        found = None
        for d in self.search:
            for ext in (".mjb", ".mjb.gz"):
                if (d / f"{sha}{ext}").exists():
                    found = d / f"{sha}{ext}"
                    break
            if found:
                break
        if found is None:
            raise FileNotFoundError(f"model {sha} not found in {[str(d) for d in self.search]}")
        key = str(found)
        if key not in self._cache:
            import tempfile
            if found.suffix == ".mjb":
                self._cache[key] = mujoco.MjModel.from_binary_path(key)
            else:
                scratch = os.environ.get("REPLAY_SCRATCH", "/common/users/ss5772/bimanual_data/tmp_replay")
                os.makedirs(scratch, exist_ok=True)
                with tempfile.NamedTemporaryFile(dir=scratch, suffix=".mjb", delete=False) as t:
                    with gzip.open(found, "rb") as f:
                        t.write(f.read())
                try:
                    self._cache[key] = mujoco.MjModel.from_binary_path(t.name)
                finally:
                    os.unlink(t.name)
        return self._cache[key]


# --------------------------------------------------------------------------- solver tracing
class SolverTracer:
    """Wraps sim_common.solve_constrained_box_qp (the module global every caller / ValueFLMPC patches).
    Logs a digest of inputs and outputs of every call, in order."""

    def __init__(self, sc_mod):
        self.sc = sc_mod
        self.inner = None
        self.calls = []

    def __enter__(self):
        self.inner = self.sc.solve_constrained_box_qp
        tracer = self

        def traced(H, f, lo, hi, A=None, b=None, **kw):
            out = tracer.inner(H, f, lo, hi, A, b, **kw)
            x, lam, feas = out[0], out[1], out[2]
            tracer.calls.append(dict(
                n=int(np.asarray(H).shape[0]), m=0 if A is None else int(np.asarray(A).shape[0]),
                kw=sorted(kw), inp=[digest(H), digest(f), digest(lo), digest(hi), digest(A), digest(b)],
                out=[digest(x), digest(lam) if lam is not None else "None", bool(feas)]))
            return out
        self.sc.solve_constrained_box_qp = traced
        return self

    def __exit__(self, *exc):
        self.sc.solve_constrained_box_qp = self.inner
        return False


# --------------------------------------------------------------------------- coverage tags
def make_tags(ctor_cls, ctor_kwargs, call_kwargs, solver_calls, pre, post, ret) -> dict:
    ck = dict(ctor_kwargs)
    kw = call_kwargs
    cf = kw.get("cup_cone_facets")
    cf = cf if cf is not None else int(ck.get("cup_cone_facets", 0) or 0)
    leb = kw.get("left_ee_body", ck.get("left_ee_body"))
    tmax = kw.get("tilt_max_rad", ck.get("tilt_max_rad"))
    t = dict(
        cls=ctor_cls,
        table_rows=kw.get("table_rows") is not None,
        tilt_box=bool(leb is not None and tmax is not None and cf == 0),
        cone=bool(leb is not None and tmax is not None and cf > 0),
        cone_mode=ck.get("cone_mode", "effgrav") if (leb is not None and tmax is not None and cf > 0) else None,
        leash=bool((kw.get("leash_ee_body", ck.get("leash_ee_body")) is not None)
                   and (kw.get("leash_max_m", ck.get("leash_max_m")) is not None)),
        speed_cap=bool((ck.get("ee_speed_cap") is not None and ck.get("ee_speed_body_int") is not None)
                       or (kw.get("int_ee_body") is not None and kw.get("v_ee_max") is not None)),
        ego_hold=bool(kw.get("ego_hold", True)),
        payload_M=kw.get("M_ego_extra") is not None,
        payload_bias=kw.get("bias_ego_extra") is not None,
        delay=int(ck.get("delay_steps", 0) or 0) > 0,
        tau_max_vec=ck.get("tau_max_vec") is not None,
        adaptive_yield=bool(ck.get("adaptive_yield", False)),
        value_rows=kw.get("value_rows") is not None and kw.get("value_rows")[0] is not None,
        n_solver_calls=len(solver_calls),
        hard_rows=(solver_calls[0]["m"] if solver_calls else None),
        hard_feasible=(solver_calls[0]["out"][2] if solver_calls else None),
        fallback=int(post.get("last_fallback", pre.get("last_fallback", 0))),
        n_relaxed_inc=int(post.get("n_relaxed", 0)) - int(pre.get("n_relaxed", 0)) if "n_relaxed" in post else 0,
        taumax_relaxed_inc=("n_taumax_relaxed" in post),
        ret_feasible=bool(ret[2]),
    )
    return t


# --------------------------------------------------------------------------- recorder
class Recorder:
    """Records step() calls. Use `record_call(ctrl, step_fn, model, data, args, kwargs, source, idx)` around a
    call; or `install_class_patch(cls)` to wrap cls.step at run time (demo episodes)."""

    def __init__(self, out_dir, sc_mod, source="", stride=25, max_records=300, max_special=120,
                 always=False, models_dir=None):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.models = ModelStore(models_dir or (self.dir / "models"))
        self.sc = sc_mod
        self.source = source
        self.stride = stride
        self.max_records = max_records
        self.max_special = max_special
        self.always = always
        self.ctor = weakref.WeakKeyDictionary()
        self.twins = {}
        self._refs = {}
        self._sha_cache = {}
        self.n_calls = 0
        self.n_saved = 0
        self.n_special = 0
        self.n_reg = 0
        self.seen_sigs = set()
        self.stats = dict(fallback={0: 0, 1: 0, 2: 0})
        self._next = len(list(self.dir.glob("rec_*.pkl.gz")))

    # -- ctor capture
    def note_ctor(self, obj, cls_name, args, kwargs):
        if obj not in self.ctor:
            self.ctor[obj] = (cls_name, copy.deepcopy(args), copy.deepcopy(kwargs))

    def _twin_state(self, obj, cls, model):
        cname, args, kwargs = self.ctor[obj]
        key = (cname, digest(np.frombuffer(pickle.dumps((args, kwargs), protocol=4), np.uint8)))
        if key not in self.twins:
            twin = cls(model, *args, **kwargs)
            self.twins[key] = snap_state(twin)
        return self.twins[key]

    # -- the call wrapper
    def record_call(self, ctrl, step_fn, model, data, args, kwargs, idx=None):
        """Run step_fn(ctrl, model, data, *args, **kwargs) and (maybe) save a record. A failure of the RECORDING
        logic never changes or kills the call: it is counted and printed."""
        box = {}
        try:
            return self._record_call(ctrl, step_fn, model, data, args, kwargs, idx, box)
        except Exception:
            if "ret" in box:
                import traceback
                self.n_errors = getattr(self, "n_errors", 0) + 1
                if self.n_errors <= 3:
                    traceback.print_exc()
                    print(f"[record] recording error #{self.n_errors} (step result unaffected)", flush=True)
                return box["ret"]
            if not box.get("called"):       # failed while preparing the snapshot: just run the step
                import traceback
                traceback.print_exc()
                print("[record] pre-call snapshot error (step run unrecorded)", flush=True)
                return step_fn(ctrl, model, data, *args, **kwargs)
            raise

    def _record_call(self, ctrl, step_fn, model, data, args, kwargs, idx, box):
        self.n_calls += 1
        if self.n_reg >= self.max_records and self.n_special >= self.max_special:
            return step_fn(ctrl, model, data, *args, **kwargs)
        pre = snap_state(ctrl)
        if id(model) not in self._refs:
            self._refs[id(model)] = mujoco.MjData(model)
        dsnap = snap_data(data, self._refs[id(model)])
        cargs = copy.deepcopy(args)
        ckw = copy.deepcopy(kwargs)
        box["called"] = True
        with SolverTracer(self.sc) as tr:
            ret = step_fn(ctrl, model, data, *args, **kwargs)
        box["ret"] = ret
        post = snap_state(ctrl)
        fb = int(getattr(ctrl, "last_fallback", 0))
        self.stats["fallback"][fb] = self.stats["fallback"].get(fb, 0) + 1
        sol = tr.calls
        cname, ctor_args, ctor_kwargs = self.ctor[ctrl]
        tags = make_tags(cname, ctor_kwargs, ckw, sol, pre, post, ret)
        sig = (tags["table_rows"], tags["tilt_box"], tags["cone"], tags["leash"], tags["speed_cap"],
               tags["ego_hold"], tags["hard_rows"], tags["fallback"], tags["hard_feasible"],
               tags["n_solver_calls"], tags["taumax_relaxed_inc"])
        special = (fb != 0) or (tags["hard_feasible"] is False)
        new_sig = sig not in self.seen_sigs
        take = self.always or (idx is not None and idx % self.stride == 0) or new_sig or special
        if not take:
            return ret
        if special and self.n_special >= self.max_special:
            return ret
        if (not special) and self.n_reg >= self.max_records:
            return ret
        self.seen_sigs.add(sig)
        cls = type(ctrl)
        try:
            twin = self._twin_state(ctrl, cls, model)
        except Exception as e:  # twin failed -> store the full pre-state
            print(f"[record] twin construction failed ({e}); storing full pre-state", flush=True)
            twin = {}
        c = Cmp(0.0)
        state_diff = {k: v for k, v in pre.items() if k not in twin or not c.eq(twin[k], v)}
        post_changed = {k: v for k, v in post.items() if k not in pre or not Cmp(0.0).eq(pre[k], v)}
        msha = self._model_sha(model)
        rec = dict(
            version=1, source=self.source, idx=idx, cls=cname,
            ctor=dict(cls=cname, args=ctor_args, kwargs=ctor_kwargs),
            model_sha=msha,
            sc=dict(TAU_MAX=float(self.sc.TAU_MAX), DT=float(self.sc.DT)),
            solver_impl=f"{self.sc.solve_constrained_box_qp.__module__}.{self.sc.solve_constrained_box_qp.__name__}",
            state_diff=state_diff, data=dsnap, call=dict(args=cargs, kwargs=ckw),
            solver=sol, ret=ret, post_changed=post_changed, post_keys=sorted(post), tags=tags)
        self._save(rec)
        self.n_saved += 1
        if special:
            self.n_special += 1
        else:
            self.n_reg += 1
        return ret

    def _model_sha(self, model):
        """Hashing the 64 MB compiled model on every record is slow: cache by id(model) + a fingerprint of the
        kinematic structure (the only model data step() uses: mj_jac / mj_fullM index tables). Masses, geoms,
        damping etc. only matter through MjData fields that are stored per record. If any OTHER model field
        ever mattered, the replay compare itself would FAIL."""
        fp = digest(np.concatenate([np.ravel(getattr(model, k)).astype(np.float64) for k in
                                    ("body_pos", "body_quat", "jnt_pos", "jnt_axis", "dof_parentid", "dof_Madr",
                                     "body_dofadr", "body_dofnum")]))
        hit = self._sha_cache.get(id(model))
        if hit is None or hit[0] != fp:
            hit = (fp, self.models.put(model))
            self._sha_cache[id(model)] = hit
        return hit[1]

    def _save(self, rec):
        p = self.dir / f"rec_{self._next:05d}.pkl.gz"
        self._next += 1
        with gzip.open(p, "wb", compresslevel=3) as f:
            pickle.dump(rec, f, protocol=4)

    # -- class-level patch for demo episodes
    def install_class_patch(self, mpc_fl_mod, counter):
        orig_step = mpc_fl_mod.FLMPC.step
        orig_init = mpc_fl_mod.FLMPC.__init__
        rec = self

        def init(self_, model, *a, **k):
            rec.note_ctor(self_, type(self_).__name__, a, k)
            return orig_init(self_, model, *a, **k)

        def step(self_, model, data, *a, **k):
            counter[0] += 1
            return rec.record_call(self_, orig_step, model, data, a, k, idx=counter[0] - 1)

        mpc_fl_mod.FLMPC.step = step
        mpc_fl_mod.FLMPC.__init__ = init
        return orig_step, orig_init


# --------------------------------------------------------------------------- replay
CLASS_MAP = {
    "FLMPC": ("mpc_fl", "FLMPC"),
    "ValueFLMPC": ("flmpc_value", "ValueFLMPC"),
    "RTFLMPC": ("rt_flmpc", "RTFLMPC"),
}


def resolve_class(name):
    import importlib
    mod, cn = CLASS_MAP[name]
    return getattr(importlib.import_module(mod), cn)


def load_record(path):
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def replay_one(rec, models: ModelStore, sc_mod, step_override=None):
    """Rebuild controller + data from the record, run step(), return (outputs dict) for comparison."""
    model = models.get(rec["model_sha"])
    cls = resolve_class(rec["cls"])
    ctrl = cls(model, *copy.deepcopy(rec["ctor"]["args"]), **copy.deepcopy(rec["ctor"]["kwargs"]))
    for k, v in copy.deepcopy(rec["state_diff"]).items():
        setattr(ctrl, k, v)
    pre = snap_state(ctrl)
    data = restore_data(model, rec["data"])
    saved = (sc_mod.TAU_MAX, sc_mod.DT)
    sc_mod.TAU_MAX, sc_mod.DT = rec["sc"]["TAU_MAX"], rec["sc"]["DT"]
    try:
        with SolverTracer(sc_mod) as tr:
            step_fn = step_override or cls.step
            ret = step_fn(ctrl, model, data, *copy.deepcopy(rec["call"]["args"]), **copy.deepcopy(rec["call"]["kwargs"]))
    finally:
        sc_mod.TAU_MAX, sc_mod.DT = saved
    post = snap_state(ctrl)
    return dict(ret=ret, solver=tr.calls, pre=pre, post=post)


def check_one(rec, out, atol=1e-9):
    """Compare replay output to the record. Returns (exact_ok, tol_ok, info)."""
    c = Cmp(atol)
    ok = True
    ok &= c.eq(list(rec["ret"][:2]), list(out["ret"][:2]), "ret.tau")
    ok &= c.eq(bool(rec["ret"][2]), bool(out["ret"][2]), "ret.feasible")
    # solver call sequence (protects the hidden patch point S1 / S2)
    rs, os_ = rec["solver"], out["solver"]
    if len(rs) != len(os_):
        c._fail("solver.ncalls", f"{len(rs)} vs {len(os_)}")
        ok = False
    else:
        for i, (a, b) in enumerate(zip(rs, os_)):
            for fld in ("n", "m", "kw", "inp", "out"):
                if a[fld] != b[fld]:
                    c._fail(f"solver[{i}].{fld}", f"{a[fld]} vs {b[fld]}")
                    ok = False
    # side effects on the controller
    post_rec = rec["post_changed"]
    for k, v in post_rec.items():
        if k not in out["post"]:
            c._fail(f"post.{k}", "missing")
            ok = False
        else:
            ok &= c.eq(v, out["post"][k], f"post.{k}")
    for k in rec["post_keys"]:
        if k in post_rec:
            continue
        if k not in out["post"] or k not in out["pre"]:
            c._fail(f"post.{k}", "missing")
            ok = False
        else:
            ok &= c.eq(out["pre"][k], out["post"][k], f"post.{k}[unchanged]")
    extra = set(out["post"]) - set(rec["post_keys"])
    if extra:
        c._fail("post.keys", f"new attributes {sorted(extra)}")
        ok = False
    tol_ok = (not c.fails) or all(isinstance(f[1], float) and f[1] <= atol for f in c.fails)
    return bool(ok), tol_ok, c
