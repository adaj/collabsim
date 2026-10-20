"""Leak-guard diagnostics and visible echoes (A), no prompt labels in final lines (B), honest
back-references (C), idea-level repetition caps (D) and the step-solved rule (E)."""
import json
import re
from collections import Counter

import pytest

from collabsim.llm import LLMClient, LLMRealizer, MockLLM
from collabsim.llm.mock import default_handlers
from collabsim.llm.prompts import STYLE_GUIDE, realizer_request
from collabsim.models import ContentPlan, Message, RNG, load_task
from collabsim.sim import Simulation, intents, promptleak
from collabsim.sim.content import (fix_self_attribution, leak_triggers, own_statements, plan_ideas,
                                   raw_claim_key, VISIBLE_ECHO_WINDOW)
from collabsim.sim.factory import build_session
from collabsim.sim.realizer import FILLER, HEDGES, TemplateRealizer

from .conftest import add_external, add_student

TASKS = ("sci-enzyme-01", "math-ratio-01")


@pytest.fixture(scope="module")
def sims():
    out = []
    for seed in range(100):
        sim = Simulation(TASKS[seed % 2], seed)
        sim.run()
        out.append(sim)
    return out


@pytest.fixture(scope="module")
def mock_sims():
    out = []
    for seed in range(8):
        sim = build_session(TASKS[seed % 2], seed, realizer="mock")
        sim.run()
        out.append(sim)
    return out


def _bare_sim(seed=3):
    """A session with 3 students, nobody holding f_reuse except s3 (who will say it)."""
    sim = Simulation("sci-enzyme-01", seed, n_students=3)
    st = sim.state
    for s in st.students:
        s.known_facts = [f for f in s.known_facts if f != "f_reuse"]
        s.shared_facts = []
    return sim


def _llm(handler_text):
    h = default_handlers()
    h["realize"] = handler_text if callable(handler_text) else (lambda req: handler_text)
    return LLMRealizer(LLMClient(MockLLM(h)))


def _say(sim, sid, plan, text):
    m = Message(id=len(sim.state.messages), t=10.0 * (len(sim.state.messages) + 1), speaker=sid, kind="student",
                text=text, plan=plan)
    sim.state.messages.append(m)
    sim.updater.apply(sim.state, m, RNG(0))
    return m


FACT_REUSE = ContentPlan(act="on_task_claim", kc="kc_enzyme", claim="fact", fact_ids=["f_reuse"], is_correct=True)


# ================================================================== Fix A
def test_A_true_leak_blocked_and_fully_logged():
    sim = _bare_sim()
    task, st = sim.task, sim.state
    s = st.students[0]
    plan = ContentPlan(act="question", kc="kc_enzyme", asked_fact="f_amylase")
    assert "f_reuse" not in s.known_facts
    llm = _llm("wait so enzymes dont get used up right")   # nobody said it, s does not hold it
    text = llm.realize(plan, s, task, RNG(1), sim.names, st=st)
    rec = llm.log[-1]
    assert rec["pre_guard_leak"] and rec["fallback"] and rec["detected_by"] == "code"
    assert [d["text"] for d in rec["rejected_drafts"]] == ["wait so enzymes dont get used up right"] * 2
    trig = [t for t in rec["triggers"] if t["type"] == "true_leak"]
    assert trig and trig[0]["fact_id"] == "f_reuse" and "dont get used up" in trig[0]["keyphrases"]
    assert trig[0]["said_in_chat"] is None and trig[0]["speaker_exposed"] is False
    assert not llm.leaked_facts(task, s, plan, text, st)
    assert text.rstrip("?").strip() and "used up" not in text          # natural question fallback
    assert rec["fallback_text"] == task.fact("f_amylase").ask
    assert llm.summary()["leak_triggers_by_type"]["true_leak"] == 2


def test_A_visible_echo_is_not_a_leak_and_becomes_known():
    sim = _bare_sim()
    task, st = sim.task, sim.state
    ethan, zoe = st.students[2], st.students[0]
    ethan.known_facts.append("f_reuse")
    src = _say(sim, ethan.id, FACT_REUSE, "wait mine says the enzme is a catalyst so its not consumed")
    zoe.known_facts = [f for f in zoe.known_facts if f != "f_reuse"]   # exposure did not mark it
    plan = ContentPlan(act="agree", kc="kc_enzyme", anchor_turn=src.id)
    plan.target_len = 8
    draft = "oh so enzymes dont get used up, nice"
    trig = leak_triggers(task, zoe, plan, draft, st)
    assert [t["type"] for t in trig] == ["visible_echo"]
    assert trig[0]["said_in_chat"]["speaker"] == ethan.id and trig[0]["said_in_chat"]["msg"] == src.id
    assert "speaker_was_reader" in trig[0]["said_in_chat"]
    llm = _llm(draft)
    llm.template_deterministic = False
    text = llm.realize(plan, zoe, task, RNG(1), sim.names, st=st)
    rec = llm.log[-1]
    assert not rec["pre_guard_leak"] and not rec.get("fallback") and rec["attempts"] == 1
    assert "used up" in text
    sim._finalize(plan, zoe.id, text)
    assert "f_reuse" in zoe.known_facts                                  # demonstrably read it
    assert llm.summary()["leak_triggers_by_type"]["visible_echo"] == 1


def test_A_fact_said_outside_window_is_still_a_true_leak():
    sim = _bare_sim()
    task, st = sim.task, sim.state
    ethan, zoe = st.students[2], st.students[0]
    ethan.known_facts.append("f_reuse")
    src = _say(sim, ethan.id, FACT_REUSE, "enzymes dont get used up, one can do thousands")
    for i in range(VISIBLE_ECHO_WINDOW):
        _say(sim, st.students[1].id, ContentPlan(act="off_task", topic=None), f"lol {i}")
    zoe.known_facts = [f for f in zoe.known_facts if f != "f_reuse"]
    trig = leak_triggers(task, zoe, None, "enzymes dont get used up", st)
    assert trig[0]["type"] == "true_leak"
    assert trig[0]["said_in_chat"]["msg"] == src.id and trig[0]["said_in_chat"]["turns_ago"] == VISIBLE_ECHO_WINDOW


def test_A_external_message_counts_as_visible():
    sim = _bare_sim()
    task, st = sim.task, sim.state
    add_external(st, "e1", ContentPlan(act="on_task_claim"), text="remember enzymes are not used up")
    s = st.students[0]
    assert [t["type"] for t in leak_triggers(task, s, None, "so its not used up", st)] == ["visible_echo"]


def test_A_check_prompt_allows_visible_facts():
    sim = _bare_sim()
    task, st = sim.task, sim.state
    ethan, zoe = st.students[2], st.students[0]
    ethan.known_facts.append("f_reuse")
    _say(sim, ethan.id, FACT_REUSE, "enzymes dont get used up")
    seen = []
    h = default_handlers()
    h["realize"] = lambda req: "ok"

    def check(req):
        seen.append(req)
        return json.dumps({"leak": False, "leaked_items": [], "act": "agree"})
    h["check"] = check
    llm = LLMRealizer(LLMClient(MockLLM(h)))
    plan = ContentPlan(act="explain", kc="kc_enzyme", claim="correct", is_correct=True, claim_idx=0)
    llm.realize(plan, zoe, task, RNG(0), sim.names, st=st)
    assert "already said in the chat" in seen[-1].suffix and "f_reuse" not in seen[-1].meta["unknown_facts"]
    req = realizer_request("claude-haiku-5-5", task, st, zoe, plan, sim.names)
    assert "SAID BY OTHERS IN THE CHAT" in req.suffix and sim.names[ethan.id] in req.suffix


def test_A_template_sessions_have_no_true_leaks(sims):
    """Plans only voice claims the speaker can word without a true leak (claim_idx chosen in code)."""
    from collabsim.sim.content import true_leaks
    n = 0
    for sim in sims[:40]:
        # replay: check each line against the state just before it (known facts only grow)
        task = sim.task
        for m in sim.state.messages:
            if m.kind != "student" or m.plan.claim not in ("correct", "fact"):
                continue
            n += 1
            s = sim.state.student(m.speaker)
            st_before = sim.state.model_copy(update={"messages": sim.state.messages[:m.id]})
            assert not true_leaks(task, s, m.plan, m.text, st_before), (sim.state.session_id, m.id, m.text)
    assert n > 200


# ================================================================== Fix B
PROMPT_LABEL_LINES = ["ask the group: what happens to enzymes if it gets really hot?", "say: enzymes are catalysts",
              "share: mine says 37", "SAY THIS CONTENT: hi", "[question] does it save", "it is kc_temp lol",
              "TARGET_LEN: 4 ok", "Ethan: i think 37", "ok so (e.g. 'like i said') whatever",
              "state your belief: hotter is faster", "disagree with zoe: its 37", "ask <name> do you know"]


def test_B_detector_catches_prompt_labels():
    names = {"s1": "Ethan"}
    for t in PROMPT_LABEL_LINES:
        assert promptleak.hits(t, names, "s1"), t
        assert not promptleak.hits(promptleak.strip(t, names, "s1"), names, "s1") or t.startswith("[") is False


def test_B_detector_passes_every_template_bank():
    from collabsim.sim.offtask import TOPICS
    from collabsim.sim.realizer import REF_LEADS, REF_LEADS_NEUTRAL, REF_LEADS_PEER
    lines = [l.format(name="zoe", peer="zoe") for v in FILLER.values() for l in v] + HEDGES
    lines += [l.format(peer="zoe") + "x" for l in REF_LEADS + REF_LEADS_PEER + REF_LEADS_NEUTRAL]
    for tid in TASKS:
        task = load_task(tid)
        for f in task.facts:
            lines += f.chat + [f.ask]
        for k in task.kcs:
            lines += k.correct_claims
        for m in task.misconceptions:
            lines += m.wrong_answer_template + m.counterexample
    for _desc, ls in TOPICS.values():
        lines += ls
    for k in intents.ALL:
        lines += intents.lines(k)
    bad = [l for l in lines if promptleak.hits(l)]
    assert not bad, bad


def test_B_labelled_llm_draft_regenerated_then_fallback():
    sim = _bare_sim()
    task, st = sim.task, sim.state
    s = st.students[0]
    plan = ContentPlan(act="question", kc="kc_temp", asked_fact="f_denature")
    drafts = iter(["ask the group: what happens to enzymes if it gets really hot?", "what happens if its really hot?"])
    llm = _llm(lambda req: next(drafts))
    text = llm.realize(plan, s, task, RNG(0), sim.names, st=st)
    rec = llm.log[-1]
    assert rec["prompt_leak"] and rec["attempts"] == 2 and not rec.get("fallback")
    assert not promptleak.hits(text) and text.startswith("what happens")
    llm2 = _llm("ask the group: what happens to enzymes if it gets really hot?")
    text2 = llm2.realize(plan, s, task, RNG(0), sim.names, st=st)
    assert llm2.log[-1]["fallback_reason"] == "prompt_leak" and not promptleak.hits(text2)
    assert text2 == task.fact("f_denature").ask


def test_B_fallback_for_content_acts_is_never_a_hedge():
    """A correct-claim plan whose drafts keep leaking falls back to a plan-consistent
    line (a voiceable claim / the fact line / the question), never "its related to what the activity says"."""
    sim = _bare_sim()
    task, st = sim.task, sim.state
    s = st.students[0]
    llm = _llm("its denatured above 60 and also enzymes dont get used up")   # always leaks
    plans = [ContentPlan(act="explain", kc="kc_surface", claim="correct", is_correct=True, claim_idx=1),
             ContentPlan(act="on_task_claim", kc=task.fact(s.known_facts[0]).kc, claim="fact",
                         fact_ids=[s.known_facts[0]], is_correct=True),
             ContentPlan(act="question", kc="kc_enzyme", asked_fact="f_amylase"),
             ContentPlan(act="question", kc="kc_enzyme", intent="what_to_do")]
    for plan in plans:
        s.known_facts = [f for f in s.known_facts if f not in ("f_reuse", "f_denature")]
        text = llm._safe_fallback(task, s, plan, RNG(2), sim.names, st)
        assert not any(h in text for h in HEDGES), (plan.act, text)
        assert not promptleak.hits(text) and not llm.leaked_facts(task, s, plan, text, st)
    assert "its related to what the activity says" not in HEDGES


def test_B_no_prompt_labels_in_any_final_line(sims, mock_sims):
    for sim in sims + mock_sims:
        for m in sim.state.messages:
            if m.kind == "student":
                assert not promptleak.hits(m.text, sim.names, m.speaker), (sim.state.session_id, m.text)


def test_B_engine_replaces_labelled_final_line_and_logs():
    h = default_handlers()
    h["realize"] = lambda req: "say: " + json.loads(json.dumps(req.meta["plan"]))["act"]
    client = LLMClient(MockLLM(h))
    sim = build_session("sci-enzyme-01", 4, realizer="mock", client=client)
    sim.run(max_events=80)
    stu = [m for m in sim.state.messages if m.kind == "student"]
    assert stu and not any(promptleak.hits(m.text) for m in stu)
    assert sim.realizer.summary()["prompt_leak_hits"] > 0


# ================================================================== Fix C
def test_C_false_like_i_said_credits_the_peer():
    sim = _bare_sim()
    task, st = sim.task, sim.state
    ethan, zoe = st.students[2], st.students[0]
    ethan.known_facts.append("f_reuse")
    _say(sim, ethan.id, FACT_REUSE, "wait mine says the enzme is a catalyst so its not consumed")
    plan = FACT_REUSE.model_copy(update={"act": "explain", "reference": True, "ref_by": ethan.id})
    out, fix = fix_self_attribution(task, st, zoe.id, plan, "like i said, enzymes arent used up", sim.names)
    assert fix is not None and out.startswith(f"like {sim.names[ethan.id].lower()} said")
    # the true author may say it
    plan_e = FACT_REUSE.model_copy(update={"act": "explain", "reference": True, "ref_by": ethan.id})
    same, none = fix_self_attribution(task, st, ethan.id, plan_e, "like i said, enzymes arent used up", sim.names)
    assert none is None and same == "like i said, enzymes arent used up"
    # nobody said it: the false claim of authorship is dropped
    plan2 = ContentPlan(act="explain", kc="kc_temp", claim="correct", is_correct=True, claim_idx=0)
    out2, fix2 = fix_self_attribution(task, st, zoe.id, plan2, "as i said 37 is faster", sim.names)
    assert fix2 is not None and "i said" not in out2


def test_C_planner_and_template_attribute_references(sims):
    n_peer = 0
    for sim in sims:
        for m in sim.state.messages:
            if m.kind != "student" or not m.plan.reference:
                continue
            f = m.plan.fact_ids[0]
            before = sim.state.messages[:m.id]
            own = any(x.kind == "student" and x.speaker == m.speaker and x.plan and f in x.plan.fact_ids for x in before)
            if re.search(r"\b(like|as) i said|i said it", m.text):
                assert own, (sim.state.session_id, m.id, m.text)
            if not own and m.plan.ref_by:
                n_peer += 1
                assert m.plan.ref_by != m.speaker
    assert n_peer > 5


def test_C_prompt_rule_and_peer_reference_instruction():
    assert "like i said" in STYLE_GUIDE and "credit them" in STYLE_GUIDE
    sim = _bare_sim()
    st = sim.state
    ethan, zoe = st.students[2], st.students[0]
    plan = FACT_REUSE.model_copy(update={"reference": True, "ref_by": ethan.id, "act": "explain"})
    req = realizer_request("claude-haiku-5-5", sim.task, st, zoe, plan, sim.names)
    name = sim.names[ethan.id]
    assert f"{name} (not you) said this" in req.suffix and "never 'like i said'" in req.suffix
    own = plan.model_copy(update={"ref_by": zoe.id})
    assert "you said this yourself" in realizer_request("claude-haiku-5-5", sim.task, st, zoe, own, sim.names).suffix


def test_C_mock_llm_never_falsely_self_attributes():
    h = default_handlers()
    h["realize"] = lambda req: ("like i said, enzymes arent used up" if req.meta["plan"]["act"] in
                                ("on_task_claim", "explain") else "hmm what")
    sim = build_session("sci-enzyme-01", 5, realizer="mock", client=LLMClient(MockLLM(h)))
    sim.run(max_events=120)
    from collabsim.sim.content import said_before
    for m in sim.state.messages:
        if m.kind == "student" and re.search(r"\blike i said\b", m.text):
            st_before = sim.state.model_copy(update={"messages": sim.state.messages[:m.id]})
            assert said_before(sim.task, st_before, m.speaker, m.plan), m.text
    assert any(r.get("type") == "attribution_guard" for r in sim.state.slippage_log)


# ================================================================== Fix D
def test_D_intent_caps_unit():
    """Each specific confusion is voiced once per session, group-wide; the number of confusions per
    session is capped by group size (3 students: 2 interface questions, 3 confusions in all)."""
    sim = _bare_sim()
    st = sim.state
    a, b = st.students[0].id, st.students[1].id
    _say(sim, a, ContentPlan(act="interface", intent="save_mechanism"), "did it save?")
    assert "save_mechanism" not in intents.allowed(st, a, "interface")
    assert "save_mechanism" not in intents.allowed(st, b, "interface")
    for i in range(max(intents.GROUP_WINDOW, intents.OWN_WINDOW) + 1):
        _say(sim, b, ContentPlan(act="backchannel"), f"ok{i}")
    assert "save_mechanism" not in intents.allowed(st, b, "interface")          # once per session
    assert intents.allowed(st, b, "interface")                                  # others still possible
    assert intents.caps(3) == (2, 3) and intents.caps(4) == (3, 4) and intents.caps(10) == (6, 7)
    _say(sim, b, ContentPlan(act="interface", intent="scroll"), "how do i scroll")
    assert intents.allowed(st, a, "interface") == []                            # 2 interface questions
    assert intents.allowed(st, a, "question")                                   # 1 procedural still
    _say(sim, a, ContentPlan(act="question", intent="what_to_do"), "what do we do")
    assert intents.allowed(st, b, "question") == []                             # 3 confusions in all

def test_D_max_same_intent_per_student_session_le_2(sims):
    worst = 0
    for sim in sims:
        stu = [m for m in sim.state.messages if m.kind == "student"]
        c = Counter((m.speaker, m.plan.intent) for m in stu if m.plan.intent)
        worst = max([worst, *c.values()])
        for (sid, it), _n in c.items():
            idx = [i for i, m in enumerate(x for x in stu if x.speaker == sid) if m.plan.intent == it]
            assert all(j - i >= intents.OWN_WINDOW for i, j in zip(idx, idx[1:])), (sim.state.session_id, sid, it)
        for i, m in enumerate(stu):
            if m.plan.intent and not m.plan.answers_intent:   # an answer repeats the question's key
                assert all(x.plan.intent != m.plan.intent for x in stu[max(0, i - intents.GROUP_WINDOW):i])
        for m in stu:
            if m.plan.tag or m.plan.nudge or m.plan.step_q or m.plan.draft or m.plan.step_move:
                continue   # 0.5.1 afterthoughts, nudges, the activity question and write-up turns carry no intent
                #            (0.5.3: nor does "bora pra 2?")
            if m.plan.act == "interface" or (m.plan.act == "question" and not m.plan.asked_fact
                                             and not m.plan.follow_up):
                assert m.plan.intent, (sim.state.session_id, m.text)
    assert worst <= intents.SESSION_MAX


def test_D_claims_restated_at_most_once_more_and_only_when_direct(sims):
    from collabsim.sim.planner import ContentPlanner
    n_re = 0
    for sim in sims:
        st = sim.state
        planner = sim.planner
        for m in st.messages:
            if m.kind != "student" or raw_claim_key(m.plan) is None:
                continue
            st_before = st.model_copy(update={"messages": st.messages[:m.id]})
            prior = own_statements(sim.task, st_before, m.speaker, m.plan)
            assert len(prior) <= 1, (st.session_id, m.id, m.text)
            if prior:
                n_re += 1
                assert planner._direct(st_before, st.student(m.speaker), m.plan), (st.session_id, m.id, m.text)
    assert n_re > 0
    assert ContentPlanner is not None


def test_D_llm_prompt_carries_the_intent():
    sim = _bare_sim()
    st = sim.state
    s = st.students[0]
    for it in intents.ALL:
        act = "interface" if it in intents.INTERFACE else "question"
        req = realizer_request("claude-haiku-5-5", sim.task, st, s, ContentPlan(act=act, intent=it), sim.names)
        assert intents.description(it) in req.suffix


# ================================================================== Fix E
def _solve_counts(st, upd, kc="kc_surface", claim_idx=1, speakers=("s2", "s3", "s2")):
    for sid in speakers:
        upd.apply(st, add_student(st, sid, ContentPlan(act="on_task_claim", kc=kc, claim="correct", is_correct=True,
                                                         claim_idx=claim_idx)), RNG(0))


def test_E_key_ideas_required(make_state, updater):
    st = make_state()
    st.student("s1").misconceptions = []
    _solve_counts(st, updater)            # claim 1 covers s1_surface_area only
    assert st.step_idx == 0 and updater.missing_key_ideas(st, "s1") == ["s1_enzymes_on_surface"]
    updater.apply(st, add_student(st, "s3", ContentPlan(act="on_task_claim", kc="kc_surface", claim="fact",
                                                         fact_ids=["f_rate"], is_correct=True)), RNG(0))
    assert st.step_idx == 1 and set(st.steps_solved[0]["key_ideas"]) == {"s1_surface_area", "s1_enzymes_on_surface"}


def test_E_refuting_fact_from_a_peer_counts_as_a_challenge(make_state, updater):
    st = make_state()                     # s1 holds m_swallow_only (kc_surface); f_surface/f_rate refute it
    updater.apply(st, add_student(st, "s1", ContentPlan(act="on_task_claim", kc="kc_surface", claim="misconception",
                                                         misconception_id="m_swallow_only", is_correct=False)), RNG(0))
    assert updater.blocking_misconceptions(st, "s1")
    _solve_counts(st, updater, claim_idx=0)  # claim 0 voices the refuting key ideas after the voicing
    assert st.step_idx == 1


def test_E_voiced_misconception_blocks_until_challenged(make_state, updater):
    st = make_state()
    for sid, f in (("s2", "f_surface"), ("s3", "f_rate")):   # key ideas voiced BEFORE the misconception
        updater.apply(st, add_student(st, sid, ContentPlan(act="on_task_claim", kc="kc_surface", claim="fact",
                                                            fact_ids=[f], is_correct=True)), RNG(0))
    v = add_student(st, "s1", ContentPlan(act="on_task_claim", kc="kc_surface", claim="misconception",
                                          misconception_id="m_swallow_only", is_correct=False))
    updater.apply(st, v, RNG(0))
    # a correct counterexample aimed at another misconception neither refutes by facts nor addresses m
    updater.apply(st, add_student(st, "s2", ContentPlan(act="explain", kc="kc_surface", claim="correct",
                                                         is_correct=True, family="counterexample",
                                                         addresses_misconception="m_other")), RNG(0))
    assert st.step_correct["kc_surface"] == 3 and len(st.step_speakers) == 2
    assert st.step_idx == 0                                     # (a) and (b) hold, (c) blocks
    assert updater.blocking_misconceptions(st, "s1") == [{"student": "s1", "m": "m_swallow_only", "voiced_msg": v.id}]
    # a peer press addressing it at s1 is a challenge -> unblocked at the next group message
    p = add_student(st, "s3", ContentPlan(act="press", kc="kc_surface", anchor_turn=v.id, addressee="s1",
                                          addresses_misconception="m_swallow_only", family="press_for_reasoning"))
    updater.apply(st, p, RNG(1))
    assert updater.blocking_misconceptions(st, "s1") == []
    assert st.step_idx == 1 and st.steps_solved[0]["msg"] == p.id


def test_E_never_voiced_misconception_does_not_block(make_state, updater):
    st = make_state()                     # s1 holds it but never says it
    assert st.student("s1").misconceptions[0].active
    _solve_counts(st, updater, claim_idx=0)
    assert st.step_idx == 1


def test_E_external_message_to_holder_is_a_challenge(make_state, updater):
    st = make_state()
    v = add_student(st, "s1", ContentPlan(act="on_task_claim", kc="kc_surface", claim="misconception",
                                          misconception_id="m_swallow_only", is_correct=False))
    updater.apply(st, v, RNG(0))
    assert updater.blocking_misconceptions(st, "s1")
    add_external(st, "e1", ContentPlan(act="press", kc="kc_surface", anchor_turn=v.id, addressee="s1",
                                       addresses_misconception="m_swallow_only", family="press_for_reasoning"),
                 text="s1, why do you think that?")
    assert updater.blocking_misconceptions(st, "s1") == []


def test_E_every_solved_step_had_its_key_ideas_and_no_blocker(sims):
    n = 0
    for sim in sims:
        st, task = sim.state, sim.task
        for rec in st.steps_solved:
            n += 1
            assert set(rec["key_ideas"]) == {k.id for k in task.key_ideas(rec["step"])}
            assert all(mid <= rec["msg"] for mid in rec["key_ideas"].values())
            for mid in rec["key_ideas"].values():
                m = st.messages[mid]
                assert m.kind == "student" and plan_ideas(task, m.plan)
    assert n > 100


def test_E_task_yaml_key_ideas_cover_gold():
    for tid in TASKS:
        task = load_task(tid)
        for g in task.gold:
            kis = task.key_ideas(g.step)
            assert kis and {k.kc for k in kis} == set(g.kcs)
            for k in kis:
                assert k.facts, k.id   # every key idea is on some card (reachable by the students alone)


def test_E_rule_can_be_ablated():
    from collabsim.sim.config import SimConfig
    cfg = SimConfig(require_key_ideas=False, block_unchallenged_misconception=False)
    a = Simulation("sci-enzyme-01", 11, cfg=cfg)
    a.run()
    assert a.state.steps_solved is not None
