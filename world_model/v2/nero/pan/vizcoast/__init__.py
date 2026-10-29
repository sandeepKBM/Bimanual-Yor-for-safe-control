"""pan/vizcoast -- VIZ-COAST-style constraint layer for the covered-pan lift.  SIMULATION ONLY.

Lineage: VIZ-COAST (Yan .. Kavraki, Kingston; arXiv 2510.25548): a VLM writes constraints that prune a discrete task
planner's search; continuous parts are checked by streams; failed stream instances are cached.  What is ours here:

  facts.py     the proposer writes FACTS + CONSTRAINTS (never a plan) in a fixed JSON grammar; any lang Backend
               (mock now; the open-weight Qwen daemon via house/lang/backend.LiveVLM later).  Facts may WIDEN the belief
               set, never narrow it (the VLM is not trusted with safety-critical bounds).
  domain.py    symbolic pan skills (optimistic, AgentChord-level semantics: any held part + lift = pan lifted) and the
               belief box Theta (pose, yaw, friction, hidden mass, payload model).
  planner.py   bounded best-first search over skill sequences; VLM constraints prune (VIZ-COAST); the certificate
               admits; CEGIS: a rejected plan returns a COUNTEREXAMPLE parameter theta*, and theta* blocks every later
               candidate with the same physical signature whose belief box contains theta* (not just the one instance).
  certify.py   Stage A, cheap: model checks (MJCF masses/joints) + recorded sim counterexamples (evidence.py).
               Closed world: a skeleton with no rule and no executor is rejected.
  simcheck.py  Stage B: falsification search, the candidate plan run in sim at the vertices of its belief box;
               failures become new evidence and the loop replans.  Vertex search, not a proof.

Every verdict carries its evidence class: model | empirical (recorded runs) | assumed (e.g. the post-observation pose
bound -- closed-loop perception is NOT built).
"""
