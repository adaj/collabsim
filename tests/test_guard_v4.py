"""Leak-guard precision: evidence-required LLM leak flags (1), own_material / visible_echo rules
applied to LLM flags (2), questions are not leaks (3), fallback safety (4) and the guard-precision
harness (5)."""
import pytest

from collabsim.models import ContentPlan, RNG, load_task
from collabsim.sim import Simulation
from collabsim.sim.content import asserts_in, is_interrogative, leak_triggers, presupposes, true_leaks

TASK = load_task("sci-enzyme-01")


def _student_without(*fids, seed=3):
    sim = Simulation("sci-enzyme-01", seed, n_students=3)
    s = sim.state.students[0]
    s.known_facts = [f for f in s.known_facts if f not in fids]
    s.jigsaw_facts = [f for f in s.jigsaw_facts if f not in fids]
    s.shared_facts = []
    for k in s.kcs.values():
        k.p = 0.2
    return sim, s


Q_PLAN = ContentPlan(act="question", kc="kc_temp")


# ================================================================== Fix 3: questions are not leaks
@pytest.mark.parametrize("text", ["is there a best temperature for it?", "zoe is there a best temp?",
                                  "what is the best temp for it", "does it have an optimum?"])
def test_3_plain_question_is_not_a_leak(text):
    sim, s = _student_without("f_optimum", "f_denature")
    assert true_leaks(TASK, s, Q_PLAN, text, sim.state) == []
    trig = leak_triggers(TASK, s, Q_PLAN, text, sim.state)
    assert trig and all(t["type"] == "question" for t in trig)


@pytest.mark.parametrize("text,fid", [("why does it denature above 60?", "f_denature"),
                                      ("why does it denature when hot?", "f_denature"),
                                      ("how come it works best at body temp?", "f_optimum"),
                                      ("does it denature above 60?", "f_denature"),        # spells it out
                                      ("it denatures above 60, right?", "f_denature"),      # tag question
                                      ("i think its body temp?", "f_optimum"),              # uptalk
                                      ("is there a best temp? its 37 like body temp", "f_optimum")])
def test_3_presupposing_or_asserting_question_is_a_leak(text, fid):
    sim, s = _student_without("f_optimum", "f_denature")
    assert fid in true_leaks(TASK, s, Q_PLAN, text, sim.state)


def test_3_heuristic_units():
    d = TASK.fact("f_denature")
    assert is_interrogative("whats the answer tho") and is_interrogative("im lost, what do we do now?")
    assert not is_interrogative("so it denatures, right?") and not is_interrogative("enzymes arent used up")
    assert presupposes("why does it denature?", d) and not presupposes("does it denature?", d)
    assert asserts_in("above 60 it denatures", "denatur", d)


# ================================================================== Fixes 1, 2: evidence-required LLM flags
import json  # noqa: E402

from collabsim.llm import LLMClient, LLMRealizer, MockLLM  # noqa: E402
from collabsim.llm.mock import default_handlers  # noqa: E402
from collabsim.models import Message  # noqa: E402
from collabsim.sim.content import validate_llm_flag  # noqa: E402


def _v(s, draft, item, quote, st=None, plan=None):
    return validate_llm_flag(TASK, s, plan, draft, item, quote, st)


def test_1_flag_needs_a_quote_in_the_draft():
    sim, s = _student_without("f_denature")
    d = "above 60 the enzyme denatures and stops working"
    assert _v(s, d, "f_denature", None) == (False, "no_quote")
    assert _v(s, d, "f_denature", "it gets destroyed at 80") == (False, "quote_not_in_draft")
    assert _v(s, d, "f_nope", d) == (False, "unknown_item")
    assert _v(s, d, "f_denature", "Above 60, the enzyme DENATURES") == (True, "true_leak")


def test_1_question_and_no_content_flags_rejected():
    sim, s = _student_without("f_denature", "f_optimum")
    assert _v(s, "whats the answer tho", "f_denature", "whats the answer tho")[1] == "interrogative"
    assert _v(s, "is there a best temperature for it?", "f_optimum", "best temperature")[1] == "interrogative"
    assert _v(s, "why does it denature above 60?", "f_denature", "denature above 60") == (True, "true_leak")
    assert _v(s, "one enzyme can work on thousands", "f_denature", "one enzyme can work on thousands")[1] \
        == "no_item_content"


def test_2_kc_items_map_to_own_facts_and_visible_window():
    sim, s = _student_without("f_denature")
    s.known_facts = ["f_reuse", "f_rate"]
    d = "enzymes arent used up, one of them can work on like thousands of starch molecules"
    assert _v(s, d, "kc_enzyme", d)[1] == "own_material"
    peer = sim.state.students[1]
    peer.known_facts = ["f_surface"]
    peer.kcs["kc_enzyme"].p = 0.2
    d2 = "yeah like ethan said, enzymes arent used up"
    assert _v(peer, d2, "kc_enzyme", d2) == (True, "true_leak")  # nobody said it yet
    sim.state.messages.append(Message(id=0, t=5.0, speaker=s.id, kind="student", text=d))
    assert _v(peer, d2, "kc_enzyme", d2, st=sim.state)[1] == "visible_echo"


def test_2_can_state_allows_a_known_correct_claim():
    sim, s = _student_without("f_denature", "f_optimum")
    d = "the enzyme gets denatured at 80 so it stops working"
    assert _v(s, d, "f_denature", d) == (True, "true_leak")
    s.kcs["kc_temp"].p = 0.7
    assert _v(s, d, "f_denature", d)[1] == "can_state"


def test_1_realizer_logs_rejected_flags_and_does_not_fall_back():
    sim, s = _student_without("f_optimum", "f_denature")
    h = default_handlers()
    h["realize"] = lambda req: "im kinda lost, what are we supposed to do now?"
    h["check"] = lambda req: json.dumps({"leak": True, "act": "question",
                                         "leaked_items": [{"id": "kc_temp", "quote": "what are we supposed to do now"},
                                                          {"id": "f_denature"}, 0]})
    llm = LLMRealizer(LLMClient(MockLLM(h)))
    plan = ContentPlan(act="question", kc="kc_temp", target_len=9)
    llm.realize(plan, s, TASK, RNG(0), sim.names, st=sim.state)
    rec = llm.log[-1]
    rej = [t for t in rec["triggers"] if t["type"] == "llm_flag_rejected"]
    assert len(rej) == 3 and {t["reason"] for t in rej} >= {"interrogative", "no_quote"}
    assert not rec["pre_guard_leak"] and not rec.get("fallback") and rec["attempts"] == 1
    assert llm.summary()["leak_triggers_by_type"]["llm_flag_rejected"] == 3


# ================================================================== Fix 4: fallback safety
from collabsim.llm.realizer import MINIMAL_ACT_LINES  # noqa: E402
from collabsim.sim import promptleak  # noqa: E402


def test_4_fallback_never_equals_a_rejected_draft_and_passes_final_checks():
    """A question draft was rejected and the template fallback produced the SAME sentence: a rejected
    draft is never reused; the fallback passes the final checks."""
    sim, s = _student_without("f_optimum", "f_denature")
    ask = TASK.fact("f_optimum").ask
    h = default_handlers()
    h["realize"] = lambda req: ask
    # a (bogus but well-evidenced) flag on every draft forces regeneration + fallback
    h["check"] = lambda req: json.dumps({"leak": True, "act": "question",
                                         "leaked_items": [{"id": "f_optimum", "quote": "best temp"}]})
    llm = LLMRealizer(LLMClient(MockLLM(h)))
    plan = ContentPlan(act="question", kc="kc_temp", asked_fact="f_optimum", target_len=7)
    import collabsim.sim.content as content
    orig = content.validate_llm_flag
    try:  # simulate a checker whose flags survive validation
        import collabsim.llm.realizer as R
        R.validate_llm_flag = lambda *a, **k: (True, "true_leak")
        llm.realize(plan, s, TASK, RNG(0), sim.names, st=sim.state)
    finally:
        R.validate_llm_flag = orig
    rec = llm.log[-1]
    assert rec["fallback"] and content.norm(rec["fallback_text"]) != content.norm(ask)
    assert llm.final_ok(TASK, s, plan, rec["fallback_text"], sim.names, sim.state, [ask])
    assert not rec["post_guard_leak"]


def test_4_minimal_line_when_no_safe_variant():
    sim, s = _student_without("f_optimum", "f_denature")
    llm = LLMRealizer(LLMClient(MockLLM()))
    plan = ContentPlan(act="question", kc="kc_temp", asked_fact="f_optimum", target_len=7)
    llm.fallback.candidates = lambda *a, **k: []
    llm.fallback.safe_line = lambda *a, **k: "why does it denature above 60?"   # unsafe
    out = llm._safe_fallback(TASK, s, plan, RNG(0), sim.names, sim.state, rejected=["anyone know?"])
    assert out in MINIMAL_ACT_LINES["question"] and out != "anyone know?"
    for lines in MINIMAL_ACT_LINES.values():
        for c in lines:
            assert not TASK.facts_mentioned(c) and not promptleak.hits(c)


def test_4_mock_sessions_have_zero_post_guard_leaks():
    from collabsim.sim.factory import build_session
    for seed in range(4):
        sim = build_session(("sci-enzyme-01", "math-ratio-01")[seed % 2], seed, realizer="mock")
        sim.run()
        assert sim.realizer.summary()["post_guard_leak_rate"] == 0
        assert not [x for x in sim.state.slippage_log if x.get("type") == "final_true_leak"]


# ================================================================== Fix 5: guard-precision harness
from pathlib import Path  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "d1_v4_guard_precision.jsonl"


def test_5_harness_rejects_v4_false_positives_and_keeps_true_leaks():
    from tests.support.guard_precision import evaluate
    res = evaluate(FIXTURE)
    bad = [r for r in res["rows"] if not r["ok"]]
    assert not bad, bad
    v4_llm = [r for r in res["rows"] if r["case"].startswith("v4_") and r["kind"] == "llm_flag"]
    assert len(v4_llm) == 8 and all(r["decision"] == "reject" for r in v4_llm)
    assert res["true_leaks_kept"] == res["true_leaks"] == 3


@pytest.mark.parametrize("case,quote", [("v4_msg26", "the answer"), ("v4_msg22", "thousands of starch molecules"),
                                        ("v4_msg22", "enzymes arent used up"), ("v4_msg8", "more surface"),
                                        ("v4_msg36_first", "enzymes arent used up")])
def test_5_decisions_hold_for_partial_quotes(case, quote):
    from tests.support.guard_precision import replay
    rec = next(json.loads(x) for x in FIXTURE.read_text().splitlines() if json.loads(x)["case"] == case)
    assert replay({**rec, "quote": quote})["decision"] == "reject"


def test_5_constructed_leak_kept_for_partial_quote():
    from tests.support.guard_precision import replay
    rec = next(json.loads(x) for x in FIXTURE.read_text().splitlines()
               if json.loads(x)["case"] == "constructed_true_leak")
    for q in ("above 60 the enzyme denatures", "stops working", "denatures"):
        assert replay({**rec, "quote": q})["decision"] == "keep", q
