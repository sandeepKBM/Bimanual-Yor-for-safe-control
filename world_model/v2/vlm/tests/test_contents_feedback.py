"""contents_feedback replay: falsification, re-ask and arm scoring with a fake model (no server, no sim)."""
from world_model.v2.vlm import contents_feedback as FB

REC = dict(id="box_000_torso", task="box", view="torso", facts="a closed box", contents_def="the box contents",
           truth=dict(total_mass_kg=3.0, contents_kg=2.75))
RUNGS = [dict(m_ff=0.75, L=0.74, free=False, Theta=[0.7, 5.25]),
         dict(m_ff=1.75, L=1.74, free=False, Theta=[1.7, 5.25]),
         dict(m_ff=2.75, L=2.74, free=False, Theta=[2.7, 5.25]),
         dict(m_ff=3.25, L=3.0, free=True, Theta=[2.9, 3.1])]
TRACE = dict(rungs=RUNGS, why="free", M_true=3.0)


def _fake(answers):
    calls = []

    def ask(rec, text):
        calls.append(text)
        return answers[min(len(calls) - 1, len(answers) - 1)], "stop", 0.1, None
    return ask, calls


def test_wrong_belief_is_falsified_and_reasked_with_evidence():
    ask, calls = _fake(['{"total_mass_kg": [2.5, 3.5], "contents_kg": [2.25, 3.25]}'])
    row = FB.replay(REC, dict(parsed=dict(total_mass_kg=[0.3, 0.4])), TRACE, ceil=2.5, empty=0.25, ask=ask, max_reask=2, e_read=0.16)
    assert row["n_asks"] == 1 and row["rung_falsified"] == 0              # V0 hi 0.4 < P_0 lo 0.7
    assert "at least 0.70 kg" in calls[0] and "[0.30, 0.40]" in calls[0]
    assert row["arms"]["vlm0"]["covers"] is False and row["arms"]["reask"]["covers"] is True
    assert row["arms"]["vlm0"]["false_safe"] is True                      # VLM alone would carry a 2.75 kg load over a 2.5 ceiling
    assert row["arms"]["reask_and_probe_end"]["B"] == [2.9, 3.1] and row["arms"]["probe"]["carry"] is False


def test_consistent_belief_is_never_reasked_and_reask_budget_holds():
    ask, calls = _fake(['{"total_mass_kg": [0.5, 0.6]}'])                  # keeps answering wrong
    row = FB.replay(REC, dict(parsed=dict(total_mass_kg=[0.3, 0.4])), TRACE, ceil=2.5, empty=0.25, ask=ask, max_reask=2, e_read=0.16)
    assert row["n_asks"] == 2 and len(calls) == 2
    assert any(e["kind"] == "falsified_unresolved" for e in row["events"])
    ask2, calls2 = _fake(["{}"])
    row2 = FB.replay(REC, dict(parsed=dict(total_mass_kg=[0.5, 5.0])), TRACE, ceil=2.5, empty=0.25, ask=ask2, max_reask=2, e_read=0.16)
    assert row2["n_asks"] == 0 and not calls2
