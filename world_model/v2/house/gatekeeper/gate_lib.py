"""gatekeeper/gate_lib.py -- Gatekeeper + certified VIABILITY LIBRARY + LOOK recovery.  SIMULATION ONLY.

LibGatekeeper(Gatekeeper) overrides only _commit / gk_step / filter_base / counters; every default of the base class is untouched (a LibGatekeeper whose
recovery never fires -- stall_s = inf -- reproduces the base gatekeeper's golden counters bit for bit; tests/test_gate_lib.py).

  * every COMMIT (hold-u_op or deviation candidate) is offered to the library: lib.add(tr, belief it was certified under, ...)  (re-checks check1+check2,
    grows the precondition region, dedupes -- checkers/library.py).
  * RECOVERY trigger: after a gatekeeper step in which no candidate passed (keep / regime 2) for >= stall_s (ASSUMPTION 1.0 s) of consecutive such steps, OR on
    entering regime 2, lib.recover(current belief, v0) is called (at most every rec_period_s while it persists, never while a recovery trajectory is running):
      direct -> commit the entry's trajectory re-based at now (T2c: current belief contained in the entry's X => certified).  debug=True re-runs check1/check2
                and counts any mismatch (n_rec_mismatch; the entry is then NOT committed);
      move   -> commit the first certified connector leg, then the next ones (each re-certified under the then-current belief; abort if one is not), then the entry;
      look   -> LOOK (tracker source only, FROM REST only): force a Looker glance toward the stale / uncertain person so the next frame re-observes it, then the next
                gatekeeper steps re-evaluate (a fresh, smaller state set may now hit the library or pass a candidate).  Truth source: no-op (counted);
      none   -> keep the base behaviour (keep-backup / regime 2).
  A candidate that passes during a recovery trajectory pre-empts it (the operator's own command being certified is always preferred).
The glance is a body twist (0, 0, +-wz) that goes through the SAME certification as any u_op (gk_step), so it is committed only if the rotation is certified; when it
is not, the glance is cancelled (n_look_blocked).
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..checkers.library import Library, RouteProgress
from ..teleop.sim import Cmd
from .checks import check1, check2
from .gate import Gatekeeper
from .traj import Traj

REST_EPS = 1e-3                    # |v|_inf [m/s, rad/s] below which the base counts as at rest (LOOK is only issued from rest)
GROW = dict(xy=0.15, p=0.3, mu=0.10)


def _ms(xs):
    a = np.array(xs) if len(xs) else np.zeros(1)
    return float(a.mean()), float(a.max())


class LibGatekeeper(Gatekeeper):
    def __init__(self, cfg=None, gk=None, perc=None, *, lib: Library | None = None, stall_s: float = 1.0, rec_period_s: float = 0.5, grow: dict | None = GROW,
                 debug: bool = False, strict: bool = False, looker=None, look_cfg=None, add_stride: int = 1, max_entries: int = 4000, v_nom: float = 0.3):
        """stall_s: ASSUMPTION, seconds of consecutive no-candidate gatekeeper steps before recovery is tried (regime 2 triggers immediately).  debug: re-verify every direct
        recovery with check1/check2 (mismatch counted and the entry refused); strict: raise AssertionError on a mismatch.  looker: injected people_track.look.Looker-like
        (tests); default: built lazily on the tracker's PerceptionStack with all automatic triggers OFF (glances are only issued by recovery).  add_stride: offer every
        add_stride-th commit to the library (1 = all)."""
        super().__init__(cfg, gk, perc)
        self.lib = lib
        self._lib_kw = dict(grow=grow, max_entries=max_entries, v_nom=v_nom)
        self.stall_s, self.rec_period_s = float(stall_s), float(rec_period_s)
        self.debug, self.strict = bool(debug), bool(strict)
        self.add_stride = max(1, int(add_stride))
        self.looker = looker
        self._look_cfg = look_cfg
        self._looker_tried = looker is not None
        self._stall_t0 = None            # time of the first gatekeeper step of the current no-candidate streak
        self._last_rec_t = -1e9
        self._rec_until = -1e9           # a recovery trajectory (or connector) is running until then
        self._rq = None                  # pending MOVE: dict(legs=[Traj], entry=Entry, idx, end_t)
        self._prev_mode = "init"
        self._look_t = None              # time of the last glance issued (for n_look_resolved)
        self._n_offer = 0
        self.rec_ms, self.add_ms = [], []
        self.none_reasons = {}            # why recover() returned none (library.Recovery.reason head)
        for k in ("n_rec_calls", "n_rec_direct", "n_rec_move", "n_rec_look", "n_rec_none", "n_rec_commit", "n_rec_mismatch", "n_rec_move_abort",
                  "n_rec_move_abandoned", "n_rec_move_done", "n_look_glances", "n_look_noop", "n_look_deferred", "n_look_nofix", "n_look_infov", "n_look_cooldown",
                  "n_look_blocked", "n_look_resolved", "n_regime2_entries", "n_lib_offers", "glance_ticks"):
            setattr(self, k, 0)

    # ------------------------------------------------------------------------------------------ library / looker
    def _ensure_lib(self, sim):
        if self.lib is None:
            route = getattr(getattr(sim, "scene", None), "out_route", None)
            prog = RouteProgress(route).remaining if route is not None and len(route) > 1 else None
            self.lib = Library(self.gk, progress_fn=prog, **self._lib_kw)
        return self.lib

    def _ensure_looker(self):
        if not self._looker_tried:
            self._looker_tried = True
            if self.tracker and self.perc is not None and hasattr(self.perc, "cam"):
                from ..people_track.look import Looker, LookCfg
                cfg = self._look_cfg or LookCfg(lost_track=False, corridor=False)            # glances ONLY on recovery request
                self.looker = Looker(self.perc, cfg)
        return self.looker

    # ------------------------------------------------------------------------------------------ hooks
    def _commit(self, tr, T_S, dev, bel=None, t=None):
        super()._commit(tr, T_S, dev, bel, t)
        self._n_offer += 1
        if self.lib is not None and (self._n_offer - 1) % self.add_stride == 0:
            b = bel if bel is not None else self.last_bel
            t0 = time.perf_counter()
            self.lib.add(tr, b, self.gk, v0=tr.V[0], t=self.n_calls * 0.02 if t is None else t, step=self.n_steps, T_S=T_S, dev=dev)
            self.add_ms.append((time.perf_counter() - t0) * 1e3)
            self.n_lib_offers += 1

    def filter_base(self, sim, u_op):
        lk = self.looker
        if lk is not None and lk.active:                    # a glance is running: the operator's translation is replaced by the in-place yaw (still certified in gk_step)
            c2 = lk.apply(sim, Cmd(twist=np.asarray(u_op, float).copy()))
            if lk.active:
                u_op = c2.twist
                self.glance_ticks += 1
        return super().filter_base(sim, u_op)

    def gk_step(self, sim, u_op, t):
        self._ensure_lib(sim)
        n_c0 = self.n_commit
        super().gk_step(sim, u_op, t)
        if self.mode == "regime2" and self._prev_mode != "regime2":
            self.n_regime2_entries += 1
            new_r2 = True
        else:
            new_r2 = False
        self._prev_mode = self.mode
        if self.n_commit > n_c0:                            # a candidate was certified: any recovery in progress is pre-empted
            if self._rq is not None:
                self.n_rec_move_abandoned += 1
                self._rq = None
            if self._look_t is not None and t - self._look_t <= 3.0:
                self.n_look_resolved += 1
            self._look_t = None
            self._stall_t0 = None
            return
        lk = self.looker
        if lk is not None and lk.active and self.mode != "commit":      # the glance twist could not be certified (or the commit was lost): cancel it
            lk.active = False
            lk.cool_until = max(lk.cool_until, sim.t + lk.cfg.cooldown_s)
            self.n_look_blocked += 1
        if self._stall_t0 is None:
            self._stall_t0 = t
        bel = self.last_bel
        v0 = np.asarray(sim.base.v, float)
        if self._rq is not None:
            if self.mode == "regime2":
                self.n_rec_move_abort += 1
                self._rq = None
            elif t >= self._rq["end_t"] - 1e-9:
                self._advance_move(bel, v0, t)
            return
        if t < self._rec_until - 1e-9 and self.mode != "regime2":       # an entry / leg is executing and still certified
            return
        if lk is not None and lk.active:
            return
        if not (new_r2 or (self.mode == "regime2" or t - self._stall_t0 >= self.stall_s - 1e-9) and t - self._last_rec_t >= self.rec_period_s - 1e-9):
            return
        self._recover(sim, bel, v0, t)

    # ------------------------------------------------------------------------------------------ recovery
    def _recover(self, sim, bel, v0, t):
        """one library recovery attempt on belief `bel` (the belief the checkers use now) at body twist v0."""
        self.n_rec_calls += 1
        self._last_rec_t = t
        rc = self.lib.recover(bel, v0, gk=self.gk)
        self.rec_ms.append(rc.ms)
        if rc.mode == "direct":
            self.n_rec_direct += 1
            self._start_traj(rc.entry.traj(t0=t), bel, t, "rec_direct", check=self.debug)
        elif rc.mode == "move":
            self.n_rec_move += 1
            self._start_move(rc, bel, t)
        elif rc.mode == "look":
            self.n_rec_look += 1
            self._do_look(sim, rc, bel, v0, t)
        else:
            self.n_rec_none += 1
            k = rc.reason.split("(")[-1].split(":")[0].strip(") ") or "empty"
            self.none_reasons[k] = self.none_reasons.get(k, 0) + 1
        return rc

    def _start_traj(self, tr: Traj, bel, t, mode, check=True, strict=None):
        """commit a library trajectory (re-based at t).  check: re-run check1+check2 under `bel`; a failure is counted (n_rec_mismatch) and the trajectory refused."""
        if check:
            ok1 = check1(tr, bel, self.gk)[0]
            ok2 = ok1 and check2(tr, bel, self.gk)[0]
            if not (ok1 and ok2):
                self.n_rec_mismatch += 1
                if self.strict if strict is None else strict:
                    raise AssertionError(f"{mode}: library trajectory not certified under the current belief (check1={ok1}, check2={ok2})")
                return False
        self.traj, self.mode = tr, mode
        self._cb = (bel, tr, t)
        self.n_rec_commit += 1
        self._rec_until = t + tr.N * tr.h
        self._stall_t0 = None
        return True

    @staticmethod
    def _rebase(leg_traj: Traj, t):
        return Traj(leg_traj.V.copy(), leg_traj.h, t, leg_traj.t_sw, leg_traj.T_S)

    def _start_move(self, rc, bel, t):
        legs = [self._rebase(l.traj, t) for l in rc.legs]
        if not legs:
            self._start_traj(rc.entry.traj(t0=t), bel, t, "rec_direct", check=True)
            return
        self._rq = dict(legs=legs, entry=rc.entry, idx=0, end_t=t + legs[0].N * legs[0].h)
        if not self._start_traj(legs[0], bel, t, "rec_move", check=True, strict=False):       # leg 0 starts at the current pose: it must be certified under the current belief
            self._rq = None
            self.n_rec_move_abort += 1
            return
        self._rec_until = self._rq["end_t"]

    def _advance_move(self, bel, v0, t):
        """previous leg finished: start the next one (re-certified under the current belief) or, after the last leg, the entry (found by containment)."""
        q = self._rq
        q["idx"] += 1
        if q["idx"] < len(q["legs"]):
            tr = self._rebase(q["legs"][q["idx"]], t)
            if not self._start_traj(tr, bel, t, "rec_move", check=True, strict=False):
                self._rq = None
                self.n_rec_move_abort += 1
                return
            q["end_t"] = t + tr.N * tr.h
            self._rec_until = q["end_t"]
            return
        self._rq = None
        ms = self.lib.lookup(bel, v0, gk=self.gk)
        ent = next((m.entry for m in ms if m.entry.id == q["entry"].id), ms[0].entry if ms else None)
        if ent is not None and self._start_traj(ent.traj(t0=t), bel, t, "rec_direct", check=True, strict=False):
            self.n_rec_move_done += 1
        else:
            self.n_rec_move_abort += 1

    # ------------------------------------------------------------------------------------------ LOOK
    def _do_look(self, sim, rc, bel, v0, t):
        """LOOK = re-observe so the state set shrinks.  Tracker source only (truth people have no observation age), only from rest, only toward a person whose
        track is the soft deficit and whose bearing is outside the camera FOV; the glance itself is certified like any u_op."""
        if not self.tracker:
            self.n_look_noop += 1                                           # truth source: nothing to re-observe (the belief is already exact)
            return False
        lk = self._ensure_looker()
        if lk is None:
            self.n_look_noop += 1
            return False
        if float(np.abs(v0).max()) > REST_EPS:                              # only from rest (the glance rotates the base; a moving base finishes its backup first)
            self.n_look_deferred += 1
            return False
        if lk.active or sim.t < lk.cool_until:
            self.n_look_cooldown += 1
            return False
        tgt = self._look_target(rc.look, lk)
        if tgt is None:
            return False
        delta = tgt
        lk.active, lk.target, lk.t0, lk.last_trigger = True, delta[0], sim.t, "recovery"
        lk.last_side = 1 if delta[1] > 0 else -1
        lk.n_glances += 1
        self.n_look_glances += 1
        self._look_t = t
        return True

    def _look_target(self, look, lk):
        """-> (world bearing, signed delta from the camera axis) of the person with the largest soft deficit that is outside the FOV, else None (counted)."""
        keys = [(k, cur / max(req, 1e-6)) for k, (cur, req) in (look or {}).get("reduce", {}).items() if k.startswith("person")]
        if not keys or self._proxy is None:
            self.n_look_nofix += 1                                          # the deficit is pose / velocity-state only: a person glance cannot reduce it
            return None
        tracks = self._track_list()
        o, yaw_cam = lk._cam()
        half = lk._half()
        best = None
        for k, r in sorted(keys, key=lambda x: -x[1]):
            j = int(k[len("person"):].split(":")[0])
            if j >= len(tracks):
                continue
            ang = math.atan2(tracks[j]["xy"][1] - o[1], tracks[j]["xy"][0] - o[0])
            d = (ang - yaw_cam + math.pi) % (2 * math.pi) - math.pi
            if abs(d) <= half:
                self.n_look_infov += 1                                      # already in view: the stale track is occluded / undetected, a glance does not help
                continue
            if abs(d) <= math.radians(lk.cfg.max_glance_deg):
                best = (ang, d)
                break
        if best is None:
            self.n_look_nofix += 1
        return best

    # ------------------------------------------------------------------------------------------ counters
    def counters(self):
        c = super().counters()
        lib = self.lib
        st = lib.stats() if lib is not None else {}
        rm, rx = _ms(self.rec_ms)
        am, ax = _ms(self.add_ms)
        c.update(n_rec_calls=self.n_rec_calls, n_rec_direct=self.n_rec_direct, n_rec_move=self.n_rec_move, n_rec_look=self.n_rec_look, n_rec_none=self.n_rec_none,
                 n_rec_commit=self.n_rec_commit, n_rec_mismatch=self.n_rec_mismatch, n_rec_move_abort=self.n_rec_move_abort, n_rec_move_abandoned=self.n_rec_move_abandoned,
                 n_rec_move_done=self.n_rec_move_done, n_look_glances=self.n_look_glances, n_look_noop=self.n_look_noop, n_look_deferred=self.n_look_deferred,
                 n_look_nofix=self.n_look_nofix, n_look_infov=self.n_look_infov, n_look_cooldown=self.n_look_cooldown, n_look_blocked=self.n_look_blocked,
                 n_look_resolved=self.n_look_resolved, n_regime2_entries=self.n_regime2_entries, glance_ticks=self.glance_ticks, n_lib_offers=self.n_lib_offers,
                 none_reasons=dict(self.none_reasons), lib_size=len(lib) if lib is not None else 0, lib_added=st.get("added", 0), lib_rejected=st.get("rejected", 0), lib_dominated=st.get("dominated", 0),
                 lib_merged=st.get("merged", 0), lookup_ms_mean=st.get("lookup_ms_mean", 0.0), lookup_ms_max=st.get("lookup_ms_max", 0.0),
                 recover_ms_mean=rm, recover_ms_max=rx, lib_add_ms_mean=am, lib_add_ms_max=ax)
        return c
