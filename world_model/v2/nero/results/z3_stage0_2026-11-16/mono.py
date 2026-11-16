"""Stage 0 (Z3 plan): is torque utilisation non-decreasing in the acceleration a?  box, closed_chain default path, every cell, every segment."""
import json, numpy as np
from world_model.v2.stack.planners import closed_chain as CCP
from world_model.v2.adapters import payload_envelope as PE
from world_model.v2.stack.checks import payload as PAY
from world_model.v2.nero.contents import carry_cert as CC, probe as P
offs, per_seg, cap = CCP.static_path("box")
be = PE.EnvelopeBackend()
A = np.round(np.arange(0.0, 1.0001, 0.025), 4)
out, viol = {}, []
for cell, *_ in CC.cells("box"):
    spec = P.interval_spec("box", cell[0], cell[1])
    env = be.make_envelope(spec, cap, offs)
    for seg, keys in per_seg.items():
        u = [float(env.worst(spec.contents, keys, float(a), PAY.MODE)[0]) for a in A]
        d = np.diff(u)
        out[f"{cell}|{seg}"] = dict(u=u, min_step=float(d.min()))
        if d.min() < -1e-9:
            viol.append((cell, seg, float(d.min()), float(A[int(np.argmin(d))])))
    print(cell, {s: round(out[f'{cell}|{s}']['min_step'], 6) for s in per_seg}, flush=True)
json.dump(dict(a=A.tolist(), mode=PAY.MODE, rows=out, violations=viol), open(__file__.replace("mono.py", "mono.json"), "w"))
print("violations:", viol)
