"""0.5.2: one deterministic test group per defect of the 0.5.1 real-Haiku batch review (12 sessions, n
2/4/6, pt-BR, sci-enzyme-01): (1) a realizer instruction typed as a chat line, (2) incoherent challenges
and disagreements, (3) a tag question after almost every claim, (4) unanswered name pings, (5) a
student restating their own fact, (6) "pq vc acha isso?" left unanswered; plus the coherence audit
helper. Offline only (template and mock backends; adversarial mock handlers stand in for a model that
misbehaves), several seeds, groups of 2-10."""
from __future__ import annotations

import math
import re

import pytest

from collabsim.llm import LLMClient, LLMRealizer, MockLLM
from collabsim.llm.client import approx_tokens
from collabsim.llm.config import MIN_CACHEABLE_TOKENS
from collabsim.llm.mock import default_handlers
from collabsim.llm.prompts import coherence_prefix, say_this
from collabsim.llm.realizer import instruction_payloads, needs_coherence
from collabsim.models import ContentPlan, load_task
from collabsim.models.task import fold
from collabsim.sim import Simulation, coherence, intents, promptleak, realism
from collabsim.sim.factory import build_session
from collabsim.sim.realizer import TemplateRealizer
from tests.support import coherence_audit as audit

LANGS = ("pt-BR", "en")
TASK_IDS = ("sci-enzyme-01", "math-ratio-01")
ALL_TASKS = [load_task(t, language=lg) for lg in LANGS for t in TASK_IDS]
PT_TASKS = [load_task(t, language="pt-BR") for t in TASK_IDS]
ENZ = load_task("sci-enzyme-01", language="pt-BR")
SIZES = (2, 3, 4, 6, 10)


def _sessions(backend="template", seeds=range(1, 6), sizes=SIZES, tasks=PT_TASKS, client_factory=None):
    out = []
    for n in sizes:
        for seed in seeds:
            task = tasks[seed % len(tasks)]
            if backend == "template":
                sim = Simulation(task, seed, n_students=n)
            else:
                client = client_factory() if client_factory else None
                sim = build_session(task, seed, realizer="mock", client=client, n_students=n)
            sim.run()
            out.append(sim)
    return out


@pytest.fixture(scope="module")
def runs():
    return _sessions() + _sessions(tasks=[load_task(t) for t in TASK_IDS], seeds=range(1, 3), sizes=(2, 4, 6))


@pytest.fixture(scope="module")
def mock_runs():
    return _sessions("mock", seeds=range(1, 3), sizes=(2, 4, 6))


def _stu(sim):
    return [m for m in sim.state.messages if m.kind == "student"]


def _audit(sim):
    return audit.audit_session(sim.task, sim.result()["transcript"])


# ------------------------------------------------------------------ plan coverage for the realizer paths
def _plans(task):
    """One plan per say_this / template branch (anchors point at message 0)."""
    k, m, f = task.kcs[0].id, task.misconceptions[0], task.facts[0].id
    rf = task.refuting_facts(m.id)[0]
    ki = task.all_key_ideas()[0].id
    P = [ContentPlan(act="question", follow_up=True, addressee="s2", asked_fact=f, kc=k, anchor_turn=0),
         ContentPlan(act="question", follow_up=True, asked_fact=f, kc=k, anchor_turn=0),
         ContentPlan(act="off_task", topic="snack", target_len=4),
         ContentPlan(act="on_task_claim", claim="misconception", misconception_id=m.id, kc=m.kc, target_len=8),
         ContentPlan(act="explain", claim="misconception", misconception_id=m.id, kc=m.kc, pushback=True, target_len=8),
         ContentPlan(act="explain", claim="misconception", misconception_id=m.id, kc=m.kc, rationale=True,
                     anchor_turn=0, addressee="s2", answers_ping=True, target_len=8),
         ContentPlan(act="press", family="press_for_reasoning", addressee="s1", addresses_misconception=m.id, kc=m.kc,
                     anchor_turn=0),
         ContentPlan(act="disagree", family="counterexample", claim="correct", addressee="s1",
                     addresses_misconception=m.id, kc=m.kc, anchor_turn=0, target_len=8),
         ContentPlan(act="disagree", family="peer_explanation", claim="correct", claim_idx=task.refuting_claims(m.id)[0],
                     addressee="s1", addresses_misconception=m.id, kc=m.kc, anchor_turn=0, target_len=8),
         ContentPlan(act="disagree", family="evidence_from_material", claim="fact", fact_ids=[rf], addressee="s1",
                     addresses_misconception=m.id, kc=m.kc, anchor_turn=0, target_len=8),
         ContentPlan(act="on_task_claim", claim="fact", fact_ids=[f], kc=k, target_len=8),
         ContentPlan(act="explain", claim="fact", fact_ids=[f], kc=k, reference=True, ref_by="s2", target_len=6),
         ContentPlan(act="on_task_claim", claim="correct", claim_idx=0, kc=k, target_len=8),
         ContentPlan(act="on_task_claim", claim="hedge", kc=k, target_len=5),
         ContentPlan(act="question", asked_fact=f, kc=k, target_len=6),
         ContentPlan(act="idk", kc=k, unsure=True, anchor_turn=0, addressee="s2", answers_ping=True),
         ContentPlan(act="agree", concedes=m.id, addressee="s2", anchor_turn=0, target_len=5),
         ContentPlan(act="question", tag=True, target_len=2),
         ContentPlan(act="explain", closing="propose", target_len=16),
         ContentPlan(act="explain", closing="propose", partial=True, ideas=[ki], target_len=16),
         ContentPlan(act="agree", closing="agree", addressee="s2", target_len=4),
         ContentPlan(act="explain", closing="adjust", fact_ids=[f], addressee="s2", target_len=6),
         ContentPlan(act="question", nudge=True, kc=k, target_len=4),
         ContentPlan(act="question", step_q=True, kc=k, target_len=8),
         ContentPlan(act="interface", intent="save_mechanism", answers_intent=True, addressee="s2", anchor_turn=0),
         ContentPlan(act="off_task", topic="snack", anchor_turn=0, addressee="s2", target_len=4)]
    for d in ("propose", "confidence", "typed", "agree"):
        P.append(ContentPlan(act="explain", draft=d, ideas=[ki], kc=k, anchor_turn=0, target_len=8))
    P += [ContentPlan(act="interface", intent=i) for i in intents.INTERFACE]
    P += [ContentPlan(act="question", intent=i, kc=k) for i in intents.PROCEDURAL]
    P += [ContentPlan(act=a, addressee="s2", kc=k, anchor_turn=0) for a in
          ("greeting", "backchannel", "agree", "idk", "refusal")]
    return P


def _started(task, n=3, seed=1, turns=6):
    sim = Simulation(task, seed, n_students=n)
    sim.run_student_turns(turns)
    return sim


# ================================================================== (1) prompt leak
REVIEW_LEAK = ("você acredita nisso e fala como opinião sua: mastigar só serve pra deixar a comida mais fácil de "
               "engolir, não afeta a digestão")


def test_1_review_leak_line_is_caught():
    for t in (REVIEW_LEAK, fold(REVIEW_LEAK), "Lara: " + REVIEW_LEAK,
              "you believe this and say it as your own opinion: chewing only makes food easier to swallow",
              "diga que a enzima se gasta", "escreva so a pergunta: oq tem na saliva?",
              "questionaram sua ideia; defenda UMA vez"):
        assert promptleak.hits(t, {"s1": "Lara"}, "s1"), t


def test_1_every_instruction_is_flagged_by_the_guard():
    """Whatever plan, the realizer instruction itself (SAY THIS CONTENT) is instruction-style text: a
    model that types it back is caught."""
    names = {"s1": "Lara", "s2": "Caio", "s3": "Luana"}
    for task in ALL_TASKS:
        for p in _plans(task):
            for v in range(3):
                ins = say_this(task, p, names, variant=v, speaker="s1")
                assert promptleak.hits(ins) or promptleak.hits(ins.lower()), (task.language, p.act, ins)


def test_1_bank_and_task_lines_are_never_flagged():
    from collabsim import lang
    from collabsim.lang import pt_br
    from collabsim.sim import realizer as R
    skip = re.compile(r"NOTE|^REALISM|^PROMPT_LEAK|_RE$|^(STYLE_GUIDE|H|REPHRASE|STRICT|FACT_LEAD_HINTS|TASK_LABELS|"
                      r"LEAD_FORMS|LEAD_PREFIX|PERIOD_WORDS|ROLE_WORDS_RE|CARD_REPLACE|STOPWORDS|QUESTION_START|"
                      r"TYPO_ABBREV|INTERFACE|PROCEDURAL|INTERFACE_INTENTS|PROCEDURAL_INTENTS|TOPICS|PERIOD_NAMES)$")

    def strs(x):
        if isinstance(x, str):
            yield x
        elif isinstance(x, dict):
            for v in x.values():
                yield from strs(v)
        elif isinstance(x, (list, tuple)):
            for v in x:
                yield from strs(v)
    lines = []
    for mod in (pt_br, R):
        for k in dir(mod):
            if k.isupper() and not skip.search(k):
                lines += list(strs(getattr(mod, k)))
    for mod in (pt_br, R):
        intents_ = getattr(mod, "INTERFACE", None) or R.INTERFACE_INTENTS
        lines += [l for _k, (_d, ls) in intents_.items() for l in ls]
    lines += [l for _k, (_d, ls) in pt_br.TOPICS.items() for l in ls]
    for task in ALL_TASKS:
        lines += [x for f in task.facts for x in f.chat + [f.ask]] + [c for k in task.kcs for c in k.correct_claims]
        lines += [x for m in task.misconceptions for x in m.wrong_answer_template + m.counterexample + m.because]
    lines += [f["pt"] for f in pt_br.fewshots()]
    assert len(lines) > 300
    bad = [l for l in lines if promptleak.hits(l.replace("{peer}", "ana").replace("{name}", "ana").replace("{pct}", "50"))]
    assert not bad, bad


def test_1_no_instruction_text_through_any_realizer_path():
    """Every plan of every task in both languages, worded by the template realizer, the mock LLM pipeline
    and the LLM realizer's fallbacks: no instruction phrasing and no run of the instruction's own words."""
    for task in ALL_TASKS:
        sim = _started(task)
        st, names = sim.state, sim.names
        llm = LLMRealizer(LLMClient(MockLLM()))
        tr = TemplateRealizer()
        s = st.students[0]
        for p in _plans(task):
            ins = say_this(task, p, names, variant=0, speaker=s.id)
            pay = instruction_payloads(task) + list(names.values())
            outs = []
            for seed in range(4):
                rng = sim.rngs["realize"]
                outs += [tr.realize(p.model_copy(), s, task, rng, names, st=st),
                         tr.variant(p.model_copy(), s, task, rng, names, st=st),
                         llm.realize(p.model_copy(), s, task, rng, names, st=st),
                         llm.variant(p.model_copy(), s, task, rng, names, st=st),
                         llm._safe_fallback(task, s, p.model_copy(), rng, names, st),
                         llm.minimal_line(task, s, p.model_copy(), rng, names, st),
                         llm._coherent_fallback(task, s, p.model_copy(), rng, names, st, [])]
            outs += tr.candidates(p, s, task, names, st)
            for o in outs:
                assert not promptleak.hits(o, names, s.id), (task.language, p.act, o)
                assert not audit.INSTRUCTION_RE.search(fold(o)), (task.language, p.act, o)
                assert not promptleak.echo_hits(o, ins, pay, n=5), (task.language, p.act, o, ins)


def _echo_handlers():
    """A misbehaving model: it types the SAY THIS CONTENT instruction back as the chat line."""
    h = default_handlers()

    def echo(req):
        for ln in req.suffix.split("\n"):
            for head in ("DIGA ESTE CONTEÚDO:", "SAY THIS CONTENT:"):
                if ln.startswith(head):
                    return ln[len(head):].strip()
        return h["realize"](req)
    h["realize"] = echo
    return h


def test_1_a_model_that_echoes_its_instruction_never_reaches_the_chat():
    sims = _sessions("mock", seeds=(1, 2), sizes=(2, 4, 10), tasks=PT_TASKS + [load_task("sci-enzyme-01")],
                     client_factory=lambda: LLMClient(MockLLM(_echo_handlers())))
    for sim in sims:
        for m in _stu(sim):
            assert not promptleak.hits(m.text, sim.names, m.speaker), m.text
            assert not audit.INSTRUCTION_RE.search(fold(m.text)), m.text
        assert sim.realizer.summary()["prompt_leak_hits"] > 0


# ================================================================== (2) coherence
def test_2_task_files_carry_refutation_and_belief_markers():
    for task in ALL_TASKS:
        for m in task.misconceptions:
            assert m.keyphrases and m.because and task.refuting_claims(m.id) and task.refuting_facts(m.id)
            for line in m.wrong_answer_template + m.because:
                assert not coherence.refutation_hits(task, m.id, line), (m.id, line)
            for i in task.refuting_claims(m.id):
                c = task.kc(m.kc).correct_claims[i]
                assert coherence.refutation_hits(task, m.id, c) and not coherence.belief_hits(task, m.id, c), c
            for f in task.refuting_facts(m.id):
                for c in task.fact(f).chat:
                    assert not coherence.belief_hits(task, m.id, c), c
            for line in m.wrong_answer_template:
                assert coherence.belief_hits(task, m.id, line), (m.id, line)


def test_2_review_cases_are_flagged_in_code():
    t = ENZ
    belief = ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_enzyme_used_up", kc="kc_enzyme")
    c = coherence.contradiction(t, belief, "acho que a enzima não se gasta, ela funciona como um combustível que vai "
                                           "sendo consumido")
    assert c and c["reason"] == "belief_states_refutation"
    assert coherence.contradiction(t, belief, "acho que a enzima vai se gastando tipo combustivel") is None
    # Luana: "discordo um pouco do caio, pra mim a amilase quebra o amido em açúcar" -- not a refutation
    unrelated = ContentPlan(act="disagree", claim="correct", claim_idx=0, kc="kc_enzyme", family="peer_explanation",
                            addresses_misconception="m_enzyme_used_up", anchor_turn=0, addressee="s1")
    assert not coherence.refutes(t, unrelated)
    assert coherence.refutes(t, unrelated.model_copy(update={"claim_idx": 1}))
    assert not coherence.refutes(t, unrelated.model_copy(update={"claim": "fact", "fact_ids": ["f_amylase"]}))
    assert coherence.refutes(t, unrelated.model_copy(update={"claim": "fact", "fact_ids": ["f_reuse"]}))
    refuting = unrelated.model_copy(update={"claim_idx": 1})
    assert coherence.contradiction(t, refuting, "a enzima nao se gasta, a mesma continua") is None
    assert coherence.contradiction(t, refuting, "pois é ela se gasta tipo combustivel")["reason"] == \
        "refutation_asserts_belief"


def test_2_disagreement_needs_a_claim_the_speaker_opposes():
    sim = _started(ENZ, n=3, seed=2, turns=0)
    st = sim.state
    from tests.conftest import add_student
    a = add_student(st, "s1", ContentPlan(act="on_task_claim", claim="correct", claim_idx=0, kc="kc_enzyme"))
    a.text = "a amilase da saliva quebra o amido em açucar"
    plain = ContentPlan(act="explain", claim="correct", claim_idx=0, kc="kc_enzyme", anchor_turn=a.id, addressee="s1")
    caio = "lara, discordo um pouco. pra mim é a amilase da saliva que quebra o amido em açúcar"
    assert coherence.coherence_issue(ENZ, st, "s2", plain, caio)["reason"] == "disagree_without_opposition"
    assert coherence.strip_disagree("discordo um pouco do caio, pra mim a amilase quebra", ENZ).startswith("pra mim")
    b = add_student(st, "s1", ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_enzyme_used_up",
                                          kc="kc_enzyme"))
    b.text = "acho que a enzima vai se gastando"
    for s in st.students:
        s.misconceptions = [x for x in s.misconceptions if x.id != "m_enzyme_used_up" or s.id == "s1"]
    ch = ContentPlan(act="disagree", claim="correct", claim_idx=1, kc="kc_enzyme", family="peer_explanation",
                     addresses_misconception="m_enzyme_used_up", anchor_turn=b.id, addressee="s1")
    assert coherence.opposes(st, "s2", ch)
    assert coherence.coherence_issue(ENZ, st, "s2", ch, "acho que nao, a enzima nao se gasta") is None
    for t in ("nao sei", "nao to achando a caixa", "pera nao entendi", "no meu diz que", "verdade, nao tinha pensado"):
        assert not coherence.disagree_marker(t, ENZ), t
    for t in ("discordo", "acho que nao, lara", "nada a ver", "nao, a enzima nao se gasta"):
        assert coherence.disagree_marker(t, ENZ), t


def test_2_challenges_refute_and_disagreements_oppose(runs, mock_runs):
    for sim in runs + mock_runs:
        a = _audit(sim)
        assert a["challenge_unrelated"] == 0, a["examples"]
        assert a["disagree_no_opposition"] == 0, a["examples"]
        assert a["disagree_restating"] == 0 and a["self_contradiction"] == 0, a["examples"]
        for m in _stu(sim):
            if m.plan.act == "disagree":
                assert m.plan.addresses_misconception or m.plan.claim == "misconception", m


def _incoherent_handlers(always: bool):
    """A model whose challenge / belief lines do not hang together: a challenge brings an unrelated fact,
    a belief line states its refutation. With always=False the regenerated draft (it carries the
    coherence note) is the clean template line."""
    h = default_handlers()
    base = h["realize"]

    def bad(req):
        p = req.meta["plan"]
        out = base(req)
        if not always and ("COERENTE:" in req.suffix or "COHERENT:" in req.suffix):
            return out
        if p["claim"] == "misconception":
            return out + ", mas na verdade a enzima nao se gasta" if req.meta["task_id"].startswith("sci") else out
        if p.get("addresses_misconception") and p["claim"] in ("fact", "correct") and p.get("family") != "counterexample":
            return "discordo, sla, pra mim tem a ver com outra coisa"
        return out
    h["realize"] = bad
    return h


@pytest.mark.parametrize("always", [False, True])
def test_2_haiku_coherence_check_regenerates_or_falls_back(always):
    sims = _sessions("mock", seeds=(1, 2, 3), sizes=(3, 6), tasks=[ENZ],
                     client_factory=lambda: LLMClient(MockLLM(_incoherent_handlers(always))))
    verdicts = []
    for sim in sims:
        a = _audit(sim)
        assert a["self_contradiction"] == 0 and a["challenge_unrelated"] == 0, a["examples"]
        for m in _stu(sim):
            if m.plan.addresses_misconception and m.plan.claim in ("fact", "correct") and m.plan.family != "counterexample" \
                    and not m.plan.revoice and not m.plan.reference:     # (a credit line: "acho que thiago ta certo")
                assert coherence.refutation_hits(ENZ, m.plan.addresses_misconception, m.text), m.text
        recs = [r for r in sim.realizer.log if "coherence" in r]
        verdicts += [r["coherence"]["verdict"] for r in recs]
        calls = sim.client.meter.by.get("coherence@claude-haiku-5-5", {}).get("calls", 0)
        assert calls == sum(r["coherence"]["checks"] for r in sim.realizer.log + getattr(sim.realizer, "skipped_log", [])
                            if "coherence" in r) or calls >= sum(r["coherence"]["checks"] for r in recs)
    assert ("template" if always else "regenerated") in verdicts
    if always:
        assert "regenerated" not in verdicts


def test_2_coherence_check_runs_only_on_challenge_and_belief_lines(mock_runs):
    for sim in mock_runs:
        for r in sim.realizer.log:
            if "coherence" in r:
                assert r["act"] in ("disagree", "explain", "on_task_claim"), r
        n = sum(r.get("coherence", {}).get("checks", 0) for r in sim.realizer.log)
        assert n == sim.realizer.summary()["coherence_checks"]
        calls = sim.client.meter.by.get("coherence@claude-haiku-5-5", {}).get("calls", 0)
        assert calls >= n and calls > 0


def test_2_coherence_prefix_is_cacheable():
    for task in ALL_TASKS:
        assert approx_tokens(coherence_prefix(task)) >= MIN_CACHEABLE_TOKENS


# ================================================================== (3) tag questions
def test_3_tag_questions_are_capped(runs, mock_runs):
    from collabsim.sim.config import SimConfig
    cap = SimConfig().tag_share_cap
    for sim in runs + mock_runs:
        stu = _stu(sim)
        tags = [m for m in stu if m.plan.tag]
        claims = sum(1 for m in stu if sim.planner.content_claim(m.plan))
        assert len(tags) <= max(1, math.floor(cap * claims)), (len(tags), claims)
        forms = [fold(m.text).strip(" ?!.") for m in tags]
        assert len(forms) == len(set(forms)), forms
        a = _audit(sim)
        assert a["tag_consecutive_same_speaker"] == 0


# ================================================================== (4) name pings
def test_4_pings_are_capped_and_never_at_silent_students(runs, mock_runs):
    for sim in runs + mock_runs:
        st = sim.state
        pings = sim.planner.pings(st)
        assert len(pings) <= sim.planner.ping_cap(st)
        targets = [m.plan.addressee for m in pings]
        assert len(targets) == len(set(targets)), targets
        for m in pings:
            spoke = sum(1 for x in st.messages[:m.id] if x.speaker == m.plan.addressee)
            assert spoke >= sim.cfg.ping_min_messages, (m.text, spoke)


def test_4_pinged_students_answer(runs):
    tot = sum(_audit(s)["pings"] for s in runs)
    unanswered = sum(_audit(s)["pings_unanswered"] for s in runs)
    # (0.5.3: fewer pings -- no asking what one asserted -- so the rate is noisier: <= .25, pilot reply rate ~.6)
    assert tot >= 5 and unanswered / tot <= 0.25, (unanswered, tot)


def test_4_a_pinged_holder_answers_with_the_pilot_rate():
    """A classmate asked by name about a fact they hold answers (planner._respond_to_peer) at about the
    pilot reply rate; one who lacks it says so ("no meu nao tem isso")."""
    from collabsim.models.session import OpenAddress
    ans = no = 0
    for seed in range(60):
        sim = _started(ENZ, n=3, seed=seed, turns=4)
        st = sim.state
        s1, s2 = st.students[0], st.students[1]
        f = ENZ.facts[0].id
        holds = seed % 2 == 0
        s2.known_facts = [x for x in s2.known_facts if x != f] + ([f] if holds else [])
        q = sim.inject_student(s1.id, ContentPlan(act="question", kc=ENZ.facts[0].kc, asked_fact=f, follow_up=True,
                                                  addressee=s2.id, anchor_turn=0), "caio vc sabe?")
        st.open_address[s2.id] = OpenAddress(msg_id=q.id, t=st.t, source=s1.id)
        p = sim.planner.plan(st, s2.id, sim.rngs["sched"])
        if p is not None and p.opener and st.pending_contribution:     # "pera" + the answer
            p = ContentPlan.model_validate(st.pending_contribution["plan"])
        if p is not None and p.answers_ping:
            ans += 1
            if holds:
                assert p.claim == "fact" and f in p.fact_ids
            else:
                assert p.act in ("idk", "explain") and (p.unsure or p.claim == "correct")
                no += p.unsure
    assert 0.3 <= ans / 60 <= 0.8, ans    # p_reply_when_addressed 8/13; the rest lapse (gates, quiet)
    line = TemplateRealizer().realize(ContentPlan(act="idk", unsure=True, answers_ping=True, anchor_turn=0),
                                      st.students[1], ENZ, sim.rngs["realize"], sim.names, st=st)
    assert line and not promptleak.hits(line)


# ================================================================== (5) self-repetition
def test_5_own_content_repeat_is_detected_numbers_included():
    sim = _started(ENZ, n=3, seed=3, turns=0)
    st = sim.state
    from tests.conftest import add_student
    m = add_student(st, "s1", ContentPlan(act="on_task_claim", claim="correct", claim_idx=1, kc="kc_temp"))
    m.text = "a 80 a enzima desnatura e para de funcionar"
    again = "a 80 a enzima desnatura entao basicamente para"
    assert realism.own_content_repeat(again, st, "s1", ENZ, sim.names)["reason"] == "own_content_repeat"
    assert realism.own_content_repeat(again, st, "s2", ENZ, sim.names) is None          # another student
    asked = ContentPlan(act="explain", claim="correct", claim_idx=1, kc="kc_temp", restate=True)
    assert realism.own_content_repeat(again, st, "s1", ENZ, sim.names, asked) is None  # asked, other words
    assert realism.own_content_repeat(m.text, st, "s1", ENZ, sim.names, asked) is not None  # same words


def _repeating_handlers():
    """A model that tacks the speaker's first card fact onto every content line it writes."""
    h = default_handlers()
    base = h["realize"]
    from collabsim.models.student import StudentState

    def rep(req):
        out = base(req)
        p = req.meta["plan"]
        s = StudentState.model_validate(req.meta["speaker"])
        if p["claim"] in ("fact", "correct") and s.shared_facts and "NÃO SE REPITA" not in req.suffix:
            t = load_task("sci-enzyme-01", language="pt-BR")
            f = sorted(s.shared_facts)[0]
            if any(x.id == f for x in t.facts) and f not in p.get("fact_ids", []):
                return out + ", e " + t.fact(f).chat[0]
        return out
    h["realize"] = rep
    return h


def test_5_students_do_not_restate_their_own_facts(runs, mock_runs):
    for sim in runs + mock_runs:
        a = _audit(sim)
        assert a["fact_repeats"] == 0 and a["near_dup_same_speaker"] == 0, a["examples"]
    sims = _sessions("mock", seeds=(1, 2), sizes=(2, 4, 6), tasks=[ENZ],
                     client_factory=lambda: LLMClient(MockLLM(_repeating_handlers())))
    for sim in sims:
        a = _audit(sim)
        assert a["near_dup_same_speaker"] == 0, a["examples"]


def test_5_own_reference_only_when_asked_or_disputed():
    sim = _started(ENZ, n=3, seed=4, turns=0)
    st = sim.state
    s = st.students[0]
    f = s.known_facts[0]
    from tests.conftest import add_student
    add_student(st, s.id, ContentPlan(act="on_task_claim", claim="fact", fact_ids=[f], kc=ENZ.fact(f).kc))
    s.shared_facts = [f]
    add_student(st, st.students[1].id, ContentPlan(act="backchannel"))
    ref = ContentPlan(act="explain", claim="fact", fact_ids=[f], kc=ENZ.fact(f).kc, reference=True)
    out = sim.planner._enforce(st, s, ENZ.fact(f).kc, ref, sim.rngs["sched"], final=True)
    assert not (out.claim == "fact" and f in out.fact_ids), out


# ================================================================== (6) "pq vc acha isso?"
def test_6_every_belief_has_a_rationale_and_the_template_uses_it():
    sim = _started(ENZ, n=3, seed=5, turns=2)
    for m in ENZ.misconceptions:
        p = ContentPlan(act="explain", claim="misconception", misconception_id=m.id, kc=m.kc, rationale=True,
                        anchor_turn=0, answers_ping=True)
        line = TemplateRealizer().content(p, sim.state.students[0], ENZ, sim.rngs["realize"], sim.names, sim.state)
        assert any(b in line for b in m.because), line


def test_6_a_pressed_holder_gives_a_reason():
    from collabsim.models.session import OpenAddress
    from collabsim.models.student import MisconceptionState
    kinds = []
    for seed in range(40):
        sim = _started(ENZ, n=3, seed=seed, turns=3)
        st = sim.state
        h, c = st.students[0], st.students[1]
        h.misconceptions = [MisconceptionState(id="m_enzyme_used_up", kc="kc_enzyme", strength=0.8)]
        v = sim.inject_student(h.id, ContentPlan(act="on_task_claim", claim="misconception",
                                                 misconception_id="m_enzyme_used_up", kc="kc_enzyme"),
                               "a enzima vai se gastando")
        h.misconceptions[0].active = True
        pr = sim.inject_student(c.id, ContentPlan(act="press", family="press_for_reasoning", kc="kc_enzyme",
                                                  addresses_misconception="m_enzyme_used_up", anchor_turn=v.id,
                                                  addressee=h.id), "pq vc acha isso?")
        st.open_address[h.id] = OpenAddress(msg_id=pr.id, t=st.t, source=c.id)
        p = sim.planner.plan(st, h.id, sim.rngs["sched"])
        kinds.append("rationale" if p is not None and p.rationale else
                     "concede" if p is not None and p.concedes else "unsure" if p is not None and p.unsure else "none")
    assert kinds.count("rationale") + kinds.count("concede") >= 30, kinds


def test_6_presses_get_answered(runs, mock_runs):
    tot = sum(_audit(s)["presses_at_holder"] for s in runs + mock_runs)
    un = sum(_audit(s)["press_unanswered"] for s in runs + mock_runs)
    assert tot >= 20 and un / tot <= 0.12, (un, tot)
    reasons = [(s, m) for s in runs for m in _stu(s) if m.plan.rationale]
    assert reasons
    for s, m in reasons:
        assert m.plan.anchor_turn is not None and s.state.messages[m.plan.anchor_turn].plan.act == "press"


# ================================================================== audit helper
def test_audit_flags_the_review_lines():
    def msg(i, sp, text, **plan):
        return {"id": i, "t": i * 10.0, "speaker": sp, "kind": "student", "text": text, "addressee": plan.get("addressee"),
                "plan": {"act": "on_task_claim", "claim": "none", **plan}}
    tr = [msg(0, "s1", REVIEW_LEAK, claim="misconception", misconception_id="m_swallow_only", kc="kc_surface"),
          msg(1, "s2", "acho que a enzima não se gasta, ela funciona como um combustível que vai sendo consumido",
              claim="misconception", misconception_id="m_enzyme_used_up", kc="kc_enzyme"),
          msg(2, "s3", "discordo um pouco do caio, pra mim a amilase quebra o amido em açúcar, por isso fica doce",
              act="disagree", claim="correct", claim_idx=0, kc="kc_enzyme", addresses_misconception="m_enzyme_used_up",
              anchor_turn=1, family="peer_explanation"),
          msg(3, "s1", "discordo, a amilase quebra o amido", act="explain", claim="correct", claim_idx=0, kc="kc_enzyme"),
          msg(4, "s2", "vcs concordam?", act="question", tag=True),
          msg(5, "s2", "a enzima nao se gasta", claim="correct", claim_idx=1, kc="kc_enzyme"),
          msg(6, "s2", "ne isso?", act="question", tag=True),
          msg(7, "s3", "júlia vc sabe?", act="question", follow_up=True, addressee="s9", asked_fact="f_reuse"),
          msg(8, "s2", "a enzima nao se gasta ne", claim="correct", claim_idx=1, kc="kc_enzyme")]
    a = audit.audit_session(ENZ, tr)
    assert a["instruction_leak"] == 1
    assert a["self_contradiction"] >= 1
    assert a["challenge_unrelated"] == 1
    assert a["disagree_no_opposition"] == 1
    assert a["tags"] == 2 and a["tag_consecutive_same_speaker"] == 1
    assert a["pings"] == 1 and a["pings_unanswered"] == 1 and a["pings_to_quiet"] == 1
    assert a["fact_repeats"] == 1
    s = audit.summarize([a])
    assert s["instruction_leak/lines"] > 0


def test_audit_rates_over_template_sessions_are_low(runs):
    s = audit.summarize([_audit(x) for x in runs])
    assert s["instruction_leak/lines"] == 0 and s["challenge_unrelated/challenges"] == 0
    assert s["disagree_no_opposition/disagree_lines"] == 0 and s["self_contradiction/lines"] == 0
    assert s["tags/content_claims"] <= 0.15 and s["tag_consecutive_same_speaker_max"] == 0
    assert s["pings_to_quiet/pings"] == 0 and s["ping_target_max_max"] <= 1
