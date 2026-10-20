"""0.5.0: one deterministic test group per defect of the latest real-Haiku review (4 students,
pt-BR, sci-enzyme-01): (a) filler overload and self-answered questions, (b) too many interface
confusions, (c) no progress / no closing before the time cap, (d) "no meu card fala que..." framing,
(e) truncated messages, (f) a misconception repeated after being corrected. Offline only: template
and mock backends, several seeds, groups of 2-10."""
from __future__ import annotations

import json
import re
from collections import Counter

import pytest

from collabsim import Simulator
from collabsim.llm import LLMClient, LLMRealizer, MockLLM
from collabsim.llm.mock import default_handlers
from collabsim.models import ContentPlan, RNG, load_task
from collabsim.models.message import CONTENT_FREE_ACTS
from collabsim.sim import Simulation, intents, realism
from collabsim.sim.config import SimConfig
from collabsim.sim.factory import build_session
from collabsim.sim.planner import is_content_free

PT_TASKS = [load_task(t, language="pt-BR") for t in ("sci-enzyme-01", "math-ratio-01")]
SIZES = (2, 3, 4, 6, 10)


def _sessions(backend="template", seeds=range(1, 9), sizes=SIZES, **kw):
    out = []
    for n in sizes:
        for seed in seeds:
            task = PT_TASKS[seed % 2]
            if backend == "template":
                sim = Simulation(task, seed, n_students=n, **kw)
            else:
                sim = build_session(task, seed, realizer="mock", n_students=n, **kw)
            sim.run()
            out.append(sim)
    return out


@pytest.fixture(scope="module")
def runs():
    return _sessions()


@pytest.fixture(scope="module")
def mock_runs():
    return _sessions("mock", seeds=range(1, 4), sizes=(2, 4, 6))


def _stu(sim):
    return [m for m in sim.state.messages if m.kind == "student"]


def _cf(sim, m) -> bool:
    if m.plan.opener:
        return False      # (0.5.3: "mano" / "pera" leading one's own content line is not chatter)
    return is_content_free(m.plan) or realism.content_free_text(m.text, sim.task)


IDK_TEXT = re.compile(r"\b(sla|sei la|nao sei|n sei|nao faco ideia|sei nada|nem ideia|nao lembro)\b")


# ================================================================== (a) fillers
def test_a_idk_and_filler_shares_capped(runs, mock_runs):
    cfg = SimConfig()
    for sim in runs + mock_runs:
        stu = _stu(sim)
        n = len(stu)
        idk = [m for m in stu if m.plan.act in ("idk", "refusal") and not m.plan.replies_to_external]
        assert len(idk) <= max(1, cfg.idk_share_cap * n) + 1, (sim.state.session_id, len(idk), n)
        per = Counter(m.speaker for m in idk)
        assert max(per.values() or [0]) <= cfg.idk_per_student_max
        idk_text = sum(bool(IDK_TEXT.search(realism.fold(m.text))) for m in stu)
        assert idk_text <= max(2, 0.08 * n), (sim.state.session_id, idk_text, n)


def test_a_never_a_filler_after_ones_own_message(runs, mock_runs):
    """No "como eu envio a resposta?" -> "nao faço ideia" by the same student (nor any filler right
    after one's own message)."""
    for sim in runs + mock_runs:
        msgs = sim.state.messages
        for a, b in zip(msgs, msgs[1:]):
            if a.speaker == b.speaker and b.kind == "student" and not b.plan.replies_to_external:
                assert not (b.plan.act in CONTENT_FREE_ACTS or b.plan.claim == "hedge"), (a.text, b.text)


def test_a_no_long_runs_of_content_free_messages(runs, mock_runs):
    """Never more than 2 content-free messages in a row, group-wide (the pilot maximum)."""
    for sim in runs + mock_runs:
        run = 0
        for m in _stu(sim):
            run = run + 1 if _cf(sim, m) else 0
            assert run <= SimConfig().max_content_free_run, (sim.state.session_id, m.id, m.text)


def test_a_filler_forms_vary(runs):
    for sim in runs:
        forms = Counter(realism.fold(m.text).strip() for m in _stu(sim)
                        if m.plan.act in CONTENT_FREE_ACTS and len(m.text.split()) <= 3)
        assert max(forms.values() or [0]) <= 3, (sim.state.session_id, forms.most_common(3))


def test_a_unit_filler_gate():
    sim = Simulation(PT_TASKS[0], 3, n_students=3)
    st, pl = sim.state, sim.planner
    a, b = st.students[0], st.students[1]
    sim.inject_student(a.id, ContentPlan(act="question", intent="send_how"), "como eu envio a resposta?")
    assert pl.filler_blocked(st, a, ContentPlan(act="idk")) == "own_continuation"
    assert pl.filler_blocked(st, b, ContentPlan(act="idk")) is None
    sim.inject_student(b.id, ContentPlan(act="backchannel"), "ok")
    sim.inject_student(st.students[2].id, ContentPlan(act="agree"), "vdd")
    assert pl.filler_blocked(st, a, ContentPlan(act="backchannel")) == "content_free_run"
    assert pl.filler_blocked(st, a, ContentPlan(act="on_task_claim", kc="kc_surface", claim="fact",
                                                fact_ids=["f_surface"], is_correct=True)) is None


# ================================================================== (b) interface confusions
def test_b_interface_questions_capped_by_group_size_and_answered(runs):
    asked = answered = 0
    for sim in runs:
        stu = _stu(sim)
        cap_i, cap_c = intents.caps(len(sim.state.students), sim.cfg)
        q = [m for m in stu if m.plan.intent and not m.plan.answers_intent]
        iface = [m for m in q if m.plan.act == "interface"]
        assert len(iface) <= cap_i and len(q) <= cap_c, (sim.state.session_id, [m.text for m in q])
        asked += len(iface)
        answered += sum(any(x.plan.answers_intent and x.plan.anchor_turn == m.id for x in stu) for m in iface)
    assert intents.caps(4) == (3, 4)
    print(f"\ninterface questions answered: {answered}/{asked}")
    assert asked and answered / asked >= 0.5


# ================================================================== (c) progress and closing
def test_c_every_session_ends_with_a_closing_synthesis(runs, mock_runs):
    ends = Counter()
    for sim in runs + mock_runs:
        st = sim.state
        ends[st.ended_reason] += 1
        assert st.closing == "done", (st.session_id, st.ended_reason, st.closing)
        prop = st.messages[st.closing_msgs[0]]
        assert prop.plan.closing == "propose" and st.messages[st.closing_msgs[1]].speaker != prop.speaker
        if st.ended_reason == "complete":
            assert not st.closing_forced
        else:
            assert st.closing_forced and prop.plan.partial
            assert prop.t <= sim.end_t                       # the wrap-up fits before the time cap
            # it sums up only what the group established
            assert set(prop.plan.ideas) <= set(st.ideas_voiced)
    print(f"\nended: {dict(ends)}")
    assert ends["complete"] >= 0.6 * sum(ends.values())


def test_c_closing_wraps_up_near_the_time_cap():
    """A group that cannot finish wraps up in the last minutes instead of fading out."""
    cfg = SimConfig(session_minutes=12.0)
    for seed in range(1, 6):
        sim = Simulation(PT_TASKS[seed % 2], seed, cfg=cfg, n_students=4)
        sim.run()
        st = sim.state
        assert st.closing == "done" and st.t <= sim.end_t
        if not st.task_complete:
            prop = st.messages[st.closing_msgs[0]]
            assert prop.t >= sim.end_t - cfg.closing_headroom_min * 60.0 - 1e-6 or prop.t >= sim.end_t - 12 * 9.0


def test_c_progress_pressure_when_stalled():
    from collabsim.sim.scheduler import stalled
    sim = Simulation(PT_TASKS[0], 7, n_students=3)
    st, pl = sim.state, sim.planner
    for i in range(SimConfig().stall_turns):
        sim.inject_student(st.students[i % 3].id, ContentPlan(act="off_task", topic="snack"), f"lanche {i}")
    assert stalled(st, sim.cfg)
    holder = next(s for s in st.students if any(f in s.known_facts for f in ("f_surface", "f_rate")))
    p = pl._progress_move(st, holder, "kc_surface", RNG(0))
    assert p is not None and p.claim == "fact" and p.fact_ids[0] in ("f_surface", "f_rate")
    lacking = next((s for s in st.students if not set(s.known_facts) & {"f_surface", "f_rate"}), None)
    if lacking is not None:
        q = pl._progress_move(st, lacking, "kc_surface", RNG(0))
        assert q is not None and q.act == "question" and q.asked_fact in ("f_surface", "f_rate")


def test_c_no_dead_tail(runs):
    """No long tail of content-free chatter before the end: the last messages before the closing are not
    a run of fillers."""
    for sim in runs:
        st = sim.state
        before = [m for m in _stu(sim) if m.id < st.closing_msgs[0]][-6:]
        assert sum(_cf(sim, m) for m in before) <= 3, [m.text for m in before]


# ================================================================== (d) framing leads
def _leads(sim):
    return Counter(k for k in ((realism.lead_form(m.text, sim.task) or ("",))[0] for m in _stu(sim)) if k)


def test_d_framing_leads_capped_and_card_rare(runs, mock_runs):
    cfg = SimConfig()
    for sim in runs + mock_runs:
        leads = _leads(sim)
        assert all(v <= cfg.lead_form_max for v in leads.values()), (sim.state.session_id, leads)
        assert sum(realism.has_card(m.text) for m in _stu(sim)) <= cfg.card_lead_max


def test_d_model_that_always_says_card_is_capped():
    h = default_handlers()
    base = h["realize"]

    def card(req):
        p = req.meta["plan"]
        if p["claim"] == "fact" and not p["reference"]:
            return "no meu card fala que " + base(req)
        return base(req)
    h["realize"] = card
    for seed in (1, 2, 3):
        sim = build_session(PT_TASKS[0], seed, realizer="mock", client=LLMClient(MockLLM(h)), n_students=4)
        sim.run()
        stu = _stu(sim)
        assert sum(realism.has_card(m.text) for m in stu) <= 1
        assert _leads(sim).get("card", 0) <= 1
        assert any(r["type"] == "framing_guard" for r in sim.state.slippage_log)


def test_d_unit_cap_framing():
    sim = Simulation(PT_TASKS[0], 1, n_students=3)
    st, cfg = sim.state, sim.cfg
    for i, t in enumerate(["no meu card fala que pedaço menor tem mais area", "aparentemente x", "aparentemente y"]):
        sim.inject_student(st.students[i].id, ContentPlan(act="on_task_claim"), t)
    out, info = realism.cap_framing("no meu card ta que a saliva tem amilase", st, sim.task, cfg, RNG(0))
    assert "card" not in out and info and info["lead"] == "card"
    out2, _ = realism.cap_framing("aparentemente a saliva tem amilase", st, sim.task, cfg, RNG(0))
    assert not out2.startswith("aparentemente") and "a saliva tem amilase" in out2
    out3, info3 = realism.cap_framing("li que a saliva tem amilase", st, sim.task, cfg, RNG(0))
    assert out3 == "li que a saliva tem amilase" and info3 is None


# ================================================================== (e) truncation
def test_e_truncated_drafts_regenerated_or_trimmed():
    from collabsim.llm.config import MAX_TOKENS
    from collabsim.llm.realizer import trim_truncated
    assert MAX_TOKENS["realize"] >= 100
    long = "entao a amilase continua quebrando o amido " * 12 + "e a enzima continua quebr"
    calls = []

    def realize(req):
        calls.append(req.suffix)
        return long if len(calls) % 3 else "a amilase continua quebrando o amido sem se gastar"
    h = default_handlers()
    h["realize"] = realize
    llm = LLMRealizer(LLMClient(MockLLM(h)))
    sim = Simulation(PT_TASKS[0], 2, n_students=3)
    s = sim.state.students[0]
    plan = ContentPlan(act="explain", kc="kc_enzyme", claim="correct", claim_idx=1, is_correct=True, target_len=12)
    out = llm.realize(plan, s, sim.task, RNG(0), sim.names, st=sim.state)
    rec = llm.log[-1]
    assert rec["truncated"] and rec["truncated"][0]["regenerated"]
    assert "CURTO" in calls[1] and "CURTO" not in calls[0]          # the regeneration asked for brevity
    assert not out.endswith("quebr") and len(out) < len(long)
    t = trim_truncated("a enzima nao se gasta. ela continua quebr")
    assert t == "a enzima nao se gasta." and trim_truncated("a enzima continua quebr") == "a enzima continua"


def test_e_mock_cuts_long_output_like_the_api():
    from collabsim.llm.client import LLMRequest
    r = MockLLM({"x": lambda req: "palavra " * 200}).complete(
        LLMRequest(model="claude-haiku-5-5", purpose="x", suffix="hi", max_tokens=20))
    assert r.stop_reason == "max_tokens" and len(r.text) <= 80


def test_e_sessions_with_a_rambling_model_have_no_cut_words():
    h = default_handlers()
    base = h["realize"]
    h["realize"] = lambda req: (base(req) + " ") * 12 + "e continua quebr"
    sim = build_session(PT_TASKS[0], 4, realizer="mock", client=LLMClient(MockLLM(h)), n_students=4)
    sim.run(max_events=60)
    llm_turns = [r for r in sim.realizer.log if not r.get("templated")]
    assert llm_turns and sum(bool(r.get("truncated")) for r in llm_turns) >= len(llm_turns) // 2
    assert sim.realizer.summary()["truncated_drafts"] > 0
    for m in _stu(sim):
        assert not m.text.endswith("quebr"), m.text


# ================================================================== (f) misconception persistence
def test_f_a_misconception_is_voiced_once_plus_at_most_one_pushback(runs, mock_runs):
    pushbacks = 0
    for sim in runs + mock_runs:
        st = sim.state
        by = {}
        for m in _stu(sim):
            if m.plan.claim == "misconception":
                by.setdefault((m.speaker, m.plan.misconception_id), []).append(m)
        for (sid, mid), ms in by.items():
            assert len(ms) <= 2, (st.session_id, sid, mid, [m.text for m in ms])
            if len(ms) == 2:
                first, second = ms
                assert second.plan.pushback, (st.session_id, second.text)
                pushbacks += 1
                assert realism.fold(first.text) != realism.fold(second.text)       # never verbatim
                challenged = any(x.plan is not None and x.plan.addresses_misconception == mid
                                 and x.speaker != sid for x in st.messages[first.id + 1:second.id])
                assert challenged, (st.session_id, sid, mid)
    print(f"\npush-backs: {pushbacks}")


def test_f_gate_unit():
    sim = Simulation(PT_TASKS[0], 5, n_students=3)
    st, pl = sim.state, sim.planner
    s = st.students[0]
    from collabsim.models.student import MisconceptionState
    s.misconceptions = [MisconceptionState(id="m_enzyme_used_up", kc="kc_enzyme", strength=0.8)]
    voice = ContentPlan(act="on_task_claim", kc="kc_enzyme", claim="misconception",
                        misconception_id="m_enzyme_used_up", is_correct=False)
    m0 = sim.inject_student(s.id, voice.model_copy(), "a enzima vai se gastando")
    # not challenged: never voiced again
    assert pl._misconception_gate(st, s, "kc_enzyme", voice.model_copy(), RNG(1)).claim != "misconception"
    sim.inject_student(st.students[1].id, ContentPlan(act="disagree", kc="kc_enzyme", claim="fact",
                                                      fact_ids=["f_reuse"], is_correct=True, anchor_turn=m0.id,
                                                      addressee=s.id, family="evidence_from_material",
                                                      addresses_misconception="m_enzyme_used_up"), "nada a ver")
    outs = [pl._misconception_gate(st, s, "kc_enzyme", voice.model_copy(), RNG(i)) for i in range(40)]
    assert any(o.claim == "misconception" and o.pushback for o in outs)        # pushes back once...
    assert any(o.claim != "misconception" for o in outs)                       # ...or goes quiet
    sim.inject_student(s.id, voice.model_copy(update={"pushback": True}), "mas eu ainda acho que gasta")
    sim.inject_student(st.students[1].id, ContentPlan(act="press", kc="kc_enzyme", anchor_turn=len(st.messages) - 1,
                                                      addressee=s.id, family="press_for_reasoning",
                                                      addresses_misconception="m_enzyme_used_up"), "pq?")
    assert all(pl._misconception_gate(st, s, "kc_enzyme", voice.model_copy(), RNG(i)).claim != "misconception"
               for i in range(40))                                             # never a third time


def test_f_wording_never_voices_a_belief_the_plan_does_not():
    """A model that keeps restating the student's belief in other turns ("enzima se gasta tipo
    combustível") is overridden by a template wording of the plan."""
    t = PT_TASKS[0]
    belief = t.misconception("m_enzyme_used_up").wrong_answer_template[0]
    h = default_handlers()
    base = h["realize"]

    def stubborn(req):
        sp = req.meta["speaker"]
        holds = any(m["id"] == "m_enzyme_used_up" for m in sp["misconceptions"])
        return (base(req) + ", " + belief) if holds and req.meta["plan"]["claim"] != "misconception" else base(req)
    h["realize"] = stubborn
    for seed in range(1, 7):
        sim = build_session(t, seed, realizer="mock", client=LLMClient(MockLLM(h)), n_students=4)
        sim.run()
        for m in _stu(sim):
            if m.plan.claim != "misconception" and st_holds(sim, m.speaker, "m_enzyme_used_up"):
                assert "m_enzyme_used_up" not in t.misconceptions_expressed(m.text), m.text
        by = Counter(m.speaker for m in _stu(sim) if "m_enzyme_used_up" in t.misconceptions_expressed(m.text))
        assert max(by.values() or [0]) <= 2


def st_holds(sim, sid, mid) -> bool:
    s = sim.state.student(sid)
    return s.misconception(mid) is not None
