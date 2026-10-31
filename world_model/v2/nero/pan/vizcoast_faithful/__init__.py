"""pan/vizcoast_faithful -- a faithful re-implementation of VIZ-COAST (Yan, Mengdibayev, Floros, Guo, Kavraki, Kingston;
arXiv 2510.25548) on the covered-pan task, as a BASELINE.  SIMULATION ONLY.  No code was released; built from the paper.

As in the paper:
  * PDDL domain + problem (pan.pddl), unchanged by the VLM;
  * task planner = SATPlan-style encoding in Z3 with an increasing horizon (satplan.py);
  * the VLM Reasoning Module (vrm.py) gets the PDDL, a scene image, a text geometric representation, a grasp specification and
    an example compliant output; it first states refinement issues in words, then writes a PYTHON PROGRAM against the Z3 API
    that adds clauses to the SMT encoding (arbitrary first-order constraints; Fig. 3 style "block action assignments");
  * motion grounding = COAST streams (streams.py): sampled grasps, IK, contact-free pre-grasps on the KNOWN (nominal) scene;
  * on a refinement failure: cache (plan, failed stream) and add a TEMPORARY state-action block, replan; after K failures drop
    the blocks and ask the VLM to revise its constraints from the K feedback entries; no plan -> ask it to relax; a constraint
    error -> send the error back; run fails after 100 refinement failures or 600 s (paper limits).

Deviations (stated, not hidden): sequential one-action-per-step encoding (the paper cites Madagascar-style encodings, which
allow parallel steps); the VLM is a scripted mock for now (paper: Gemini 3 Flash, medium reasoning); streams are this repo's
MuJoCo IK + contact checks (paper: PyBullet).  The planner knows nothing about uncertainty or load physics -- that is the
point of comparison with pan/vizcoast (ours).
"""
