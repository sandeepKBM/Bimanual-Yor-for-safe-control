"""lang/session.py -- one language session: grounding + spec authoring + mu prior -> LangResult (what the shield / rows / HUD consume).  SIMULATION ONLY.

    sess = LangSession(backend, mu_queries=4, mu_k=1.0, out_dir=...)
    lr = sess.run("bring the coffee to my desk without spilling and keep it away from the monitor", image_rgb, proposals, mu_images=[png...], tag="desk")
    lr.mu_overrides()      -> {'mu_lo':..., 'mu_hi':...}                 for ShieldCfg
    lr.monitor_spec(cfg)   -> compiled IR node (shield IRMonitor)
    lr.clearance_by_zono   -> {perception zonotope id: d}                for the zonotope rows / base filter (hooks.clearance_result)
Every decision is in lr.log and lr.to_json().
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from world_model.v2.tray_rep.sets import ir as IR

from . import authoring as AU, grounding as GR, prior as PR


@dataclass
class LangResult:
    instruction: str
    grounding: GR.Grounding
    catalog: AU.Catalog
    spec: AU.SpecResult
    mu: PR.MuInterval
    clearance_by_zono: dict
    latency: dict
    log: list
    backend: str = ""
    tag: str = ""
    t_unix: float = field(default_factory=time.time)

    # -------- consumers
    # 'conservative' (default, owner decision 2026-10-12): an unverified VLM prior may only WIDEN the engineering prior, never raise the
    # lower bound the shield's spill budget rests on (live smoke: VLM mu [0.30, 0.68] EXCLUDED the sim truth 0.25 -> a 'vlm' shield would
    # allow ~20% more acceleration than the cup can take).  Narrowing needs identification evidence (probe / BOED), not a VLM opinion.
    # 'vlm' = use the raw VLM interval (research ablation only).
    mu_policy: str = "conservative"

    def mu_overrides(self):
        lo, hi = float(self.mu.lo), float(self.mu.hi)
        if self.mu_policy == "conservative":
            lo, hi = min(lo, PR.DEFAULT_MU[0]), max(hi, PR.DEFAULT_MU[1])
        elif self.mu_policy != "vlm":
            raise ValueError(f"mu_policy {self.mu_policy!r}")
        return dict(mu_lo=lo, mu_hi=hi)

    def monitor_spec(self, cfg):
        """the IR node the shield IRMonitor evaluates each tick (same parameters as the hand spec: r_shrink = r_cup, theta_max = theta_tray_ir, H = lo = 0)."""
        p = dict(AU.PARAMS, r_shrink=float(cfg.r_cup), theta_max=float(cfg.theta_tray_ir))
        return IR.from_dict(self.spec.spec_expanded, self.catalog.entities(), p)

    # -------- reporting
    def hud(self):
        g = self.grounding
        return dict(
            instruction=self.instruction, backend=self.backend, tag=self.tag,
            grounded={nm: dict(n=len(ids), conf=round(float(np.mean([g.names[i][1] for i in ids])), 2)) for nm, ids in g.groups.items()},
            n_marks=len(g.props), n_unknown=sum(1 for v in g.names.values() if v[0] == "unknown"),
            spec_status=self.spec.status, spec_pretty=self.spec.pretty(), spec_reasons=self.spec.reasons[-3:], unresolved=self.spec.unresolved,
            checks=dict(compile=self.spec.check.get("compile"), eval=self.spec.check.get("eval"), sem=self.spec.check.get("sem")),
            clearances={k: round(v, 3) for k, v in self.spec.clearances.items()},
            mu=dict(lo=round(self.mu.lo, 3), hi=round(self.mu.hi, 3), center=None if self.mu.center is None else round(self.mu.center, 3),
                    sigma=None if self.mu.sigma is None else round(self.mu.sigma, 3), source=self.mu.source, n=self.mu.n_parsed),
            latency={k: (round(v, 2) if isinstance(v, float) else v) for k, v in self.latency.items()})

    def to_json(self):
        g = self.grounding
        d = self.hud()
        d.update(
            grounding=dict(names={str(i): list(v) for i, v in g.names.items()}, raw_names={str(i): v for i, v in g.raw_names.items()},
                           groups={k: list(v) for k, v in g.groups.items()}, marks_png=g.marks_png, parse_ok=g.parse_ok, log=g.log,
                           boxes={str(p.id): [round(x, 1) for x in p.box] for p in g.props}, text=g.texts[0] if g.texts else ""),
            spec=dict(spec=self.spec.spec, expanded=self.spec.spec_expanded, status=self.spec.status, reasons=self.spec.reasons, unresolved=self.spec.unresolved,
                      check=self.spec.check, attempts=self.spec.attempts, mode=self.spec.mode),
            mu_full=dict(self.mu.__dict__), clearance_by_zono={str(k): v for k, v in self.clearance_by_zono.items()}, log=self.log, t_unix=self.t_unix)
        return d


class LangSession:
    def __init__(self, backend, mu_queries=4, mu_k=1.0, spec_mode="replace", max_repair=1, out_dir=None, log=print, reuse_mu=True):
        self.be, self.mu_queries, self.mu_k, self.spec_mode, self.max_repair = backend, mu_queries, mu_k, spec_mode, max_repair
        self.out_dir = Path(out_dir) if out_dir else None
        self.log_fn = log
        self.reuse_mu = reuse_mu
        self.mu: PR.MuInterval | None = None
        self.results = []

    def _log(self, lg, msg):
        lg.append(msg)
        self.log_fn("[lang] " + msg)

    def run(self, instruction, image, proposals, mu_images=None, tag="lang", with_mu=True):
        lg = []
        t0 = time.time()
        out = self.out_dir
        if out:
            out.mkdir(parents=True, exist_ok=True)
        g = GR.ground(self.be, image, proposals, out, tag)
        self._log(lg, f"grounding: {len(proposals)} marks -> " + json.dumps({k: len(v) for k, v in g.groups.items()}) + f" ({g.seconds:.1f}s model, parse_ok={g.parse_ok})")
        cat = AU.build_catalog(g)
        sr = AU.author(self.be, instruction, cat, images=[g.marks_png] if g.marks_png else [], mode=self.spec_mode, max_repair=self.max_repair, log=lambda s: self._log(lg, s))
        self._log(lg, f"spec: status={sr.status} clearances={sr.clearances} unresolved={sr.unresolved} ({sr.seconds:.1f}s model)")
        for r in sr.reasons:
            lg.append("  " + r)
        mu_s = 0.0
        if with_mu:
            if self.mu is None or not self.reuse_mu:
                imgs = list(mu_images or [])
                if not imgs and out:
                    import cv2
                    raw = out / f"{tag}_raw.png"
                    cv2.imwrite(str(raw), np.ascontiguousarray(image[:, :, ::-1]))
                    imgs = [str(raw)]
                self.mu = PR.estimate_mu(self.be, imgs, self.mu_queries, self.mu_k)
                mu_s = self.mu.seconds
                self._log(lg, f"mu prior: [{self.mu.lo:.3f}, {self.mu.hi:.3f}] source={self.mu.source} " + "; ".join(self.mu.reasons))
            else:
                self._log(lg, f"mu prior reused: [{self.mu.lo:.3f}, {self.mu.hi:.3f}]")
        mu = self.mu or PR.MuInterval(*PR.DEFAULT_MU)
        cz = {pid: d for nm, d in sr.clearances.items() for pid in cat.zono_ids(nm)}
        lat = dict(ground_s=g.seconds, author_s=sr.seconds, mu_s=mu_s, model_s=g.seconds + sr.seconds + mu_s, wall_s=time.time() - t0,
                   ground_wall_s=g.wall_s, author_wall_s=sr.wall_s, mu_wall_s=mu.wall_s if mu_s else 0.0,
                   n_vlm_calls=1 + len(sr.attempts) + (mu.n_images if mu_s else 0))
        lr = LangResult(instruction, g, cat, sr, mu, cz, lat, lg, getattr(self.be, "name", ""), tag)
        self.results.append(lr)
        if out:
            (out / f"{tag}_lang.json").write_text(json.dumps(lr.to_json(), indent=1, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))
        return lr
