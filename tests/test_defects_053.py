"""0.5.3: one deterministic test group per defect of the real-Haiku batch on 0.5.2 (seeds 5, 6 x n 2/4/6, pt-BR,
sci-enzyme-01) and of an independent qualitative review of mock transcripts:

R1 thoughts split mid-clause ("... em maltose que" | "é um açucar"), and too often;
R2 lead-word-only bursts ("pera" / "galera" / "gente olha isso", then the fact) and doubled leads;
R3 a holder giving in with no refuting move ("mano kkk" -> "ah entendi, faz sentido"), concessions that do not say
   why, and holders okaying the final answer as if nothing happened;
R4 asking what one just asserted, and re-asking an equivalent question;
R5 plural address in dyads ("galera", "vcs"), and the same stock closing okays in every session;
Q1 credit for content the credited person never said (or circular credit), one burst mixing a claim with its own
   belief, beliefs voiced for another activity;
Q2 a final answer that is a fixed join of the task's claims instead of what the group established;
Q3 step transitions, off-task threads and seed diversity of the off-task / interface pools; off-task lines out of
   the blue addressed to a classmate;
plus the coherence-audit detectors. Offline only (template and mock backends; an adversarial mock stands in for a
model that writes the real-Haiku phrasing above), several seeds, groups of 2-10.
"""
from __future__ import annotations

import hashlib
import math

import pytest

from collabsim.llm import LLMClient, MockLLM
from collabsim.llm.mock import default_handlers
from collabsim.llm.realizer import is_templated
from collabsim.models import ContentPlan, load_task
from collabsim.models.student import MisconceptionState
from collabsim.sim import Simulation, realism
from collabsim.sim.factory import build_session
from collabsim.sim.realizer import TemplateRealizer, reason_clause
from tests.conftest import add_student
from tests.support import coherence_audit as audit

TASK_IDS = ("sci-enzyme-01", "math-ratio-01")
PT_TASKS = [load_task(t, language="pt-BR") for t in TASK_IDS]
EN_TASKS = [load_task(t) for t in TASK_IDS]
ENZ = load_task("sci-enzyme-01", language="pt-BR")
SIZES = (2, 3, 4, 6, 10)


def haiku_like_handlers():
    """A model that writes the real-Haiku phrasing of the 0.5.2 batch: card facts with a mid-clause "que é", every
    fact led by "olha," (doubling an opener), "galera" / "vcs" in any group, the same stock okay to every closing
    proposal, off-task lines addressed to a classmate ("feliz aniversário, pedro"), back-references credited to the
    wrong classmate, and a generic question re-asked in other words. It follows a correction note when one comes."""
    h = default_handlers()
    base = h["realize"]
    facts = {"f_amylase": "pelo que eu li, a amilase quebra o amido em maltose que é um açucar",
             "f_optimum": "aparentemente a melhor temperatura pra amilase é uns 37, tipo a do corpo",
             "f_denature": "acima de 60 a amilase desnatura e para de funcionar, li que é isso"}

    def rz(req):
        m = req.meta
        p = m["plan"]
        names = m["names"]
        sp = m["speaker"]["id"]
        others = [pid for pid in names if pid != sp]
        hv = int(hashlib.sha256(req.key().encode()).hexdigest()[:8], 16)
        if any(x in req.suffix for x in ("REESCREVA", "NÃO PERGUNTE", "DIGA DE OUTRO JEITO", "COERENTE", "LIMPO",
                                         "ESTRITO", "NÃO SE REPITA", "CURTO")):
            return base(req)
        if p.get("closing") == "agree" and not p.get("brief"):
            return "fechou, pode mandar assim"
        if p["act"] == "off_task" and p.get("anchor_turn") is None and others:
            return f"feliz aniversário, {names[others[hv % len(others)]].lower()}"
        if p["claim"] == "fact" and p.get("fact_ids") and not p.get("closing") and not p.get("draft"):
            f = p["fact_ids"][0]
            if p.get("reference") and p.get("ref_by") and p["ref_by"] != sp:
                wrong = [o for o in others if o != p["ref_by"]]
                if wrong:
                    return f"é como {names[wrong[hv % len(wrong)]].lower()} falou, " + base(req).split(", ", 1)[-1]
            line = facts.get(f) or base(req)
            return ("olha, " if hv % 2 else "galera, olha, ") + line
        if p["act"] == "question" and not p.get("asked_fact") and not p.get("follow_up") and not p.get("intent") \
                and not p.get("step_q") and not p.get("nudge"):
            return "enzimas agem onde exatamente?"
        if p["act"] == "question" and p.get("asked_fact") == "f_rate":
            return "onde as enzimas agem de verdade na comida?"
        out = base(req)
        if p["act"] in ("question", "press") and hv % 3 == 0:
            return out.rstrip("?") + ", vcs acham?"
        return out
    h["realize"] = rz
    return h


def _sim(backend, task, seed, n, handlers=None):
    if backend == "template":
        return Simulation(task, seed, n_students=n)
    client = LLMClient(MockLLM(handlers)) if handlers is not None else None
    return build_session(task, seed, realizer="mock", client=client, n_students=n)


def _sessions(backend="template", seeds=range(1, 6), sizes=SIZES, tasks=PT_TASKS, handlers=None):
    out = []
    for n in sizes:
        for seed in seeds:
            sim = _sim(backend, tasks[seed % len(tasks)], seed, n, handlers() if handlers else None)
            sim.run()
            out.append(sim)
    return out


@pytest.fixture(scope="module")
def runs():
    return _sessions() + _sessions(tasks=EN_TASKS, seeds=range(1, 3), sizes=(2, 4, 6))


@pytest.fixture(scope="module")
def mock_runs():
    return _sessions("mock", seeds=range(1, 3), sizes=(2, 4, 6))


@pytest.fixture(scope="module")
def adv_runs():
    """The review's seeds and sizes (5, 6 x 2, 4, 6) and two more, through the Haiku-like adversarial model."""
    return _sessions("mock", seeds=(5, 6, 7, 8), sizes=(2, 4, 6), tasks=[ENZ], handlers=haiku_like_handlers)


def _stu(sim):
    return [m for m in sim.state.messages if m.kind == "student"]


def _audit(sim):
    return audit.audit_session(sim.task, sim.result()["transcript"], len(sim.state.roster))


def _started(task, n=3, seed=1, turns=0):
    sim = Simulation(task, seed, n_students=n)
    sim.step()
    if turns:
        sim.run_student_turns(turns)
    return sim


# ================================================================== R1 split thoughts
REVIEW_SPLITS = ["pelo que eu li, a amilase quebra o amido em maltose que é um açucar",
                 "aparentemente a melhor temperatura pra amilase é uns 37, tipo a do corpo",
                 "acima de 60 a amilase desnatura e para de funcionar, li que é isso"]


def test_r1_review_lines_split_only_at_a_clause_boundary():
    for text in REVIEW_SPLITS:
        head, tail = realism.split_point(text, ENZ)
        if tail is not None:
            assert not realism.bad_split(head, tail, ENZ), (head, tail)
            assert tail.split()[0] not in ("e", "é", "que"), tail
    # never at "é" (folds to "e") nor before a bare "e": a new clause after a comma is fine
    assert realism.split_point(REVIEW_SPLITS[0], ENZ)[1] is None
    assert realism.split_point(REVIEW_SPLITS[1], ENZ) == ("aparentemente a melhor temperatura pra amilase é uns 37",
                                                           "tipo a do corpo")
    assert realism.split_point(REVIEW_SPLITS[2], ENZ) == ("acima de 60 a amilase desnatura e para de funcionar",
                                                           "li que é isso")
    assert realism.split_point("a amilase da saliva quebra o amido em açucar, por isso fica doce", ENZ) == (
        "a amilase da saliva quebra o amido em açucar", "por isso fica doce")
    en = EN_TASKS[0]
    h, t = realism.split_point("the amylase in saliva breaks starch into sugar so it gets sweet", en)
    assert t == "so it gets sweet" and not realism.bad_split(h, t, en)
    assert realism.split_point("the enzyme works best at 37 and stops at 80 degrees", en)[1] is None


def test_r1_splits_are_clean_and_rare(runs, mock_runs, adv_runs):
    for sim in runs + mock_runs + adv_runs:
        a = _audit(sim)
        assert a["bad_split"] == 0, [e for e in a["examples"] if e[0] == "bad_split"]
        assert a["splits_max_per_student"] <= sim.cfg.split_per_student_max
    s = audit.summarize([_audit(x) for x in runs])
    assert s["splits/lines"] <= 0.07


# ================================================================== R2 lead-word bursts
def test_r2_repeated_lead_is_dropped():
    assert realism.strip_repeated_lead("olha, pelo que eu li, a amilase quebra o amido", "olha", ENZ) == \
        "pelo que eu li, a amilase quebra o amido"
    assert realism.strip_repeated_lead("pera a enzima nao se gasta", "pera achei aqui", ENZ) == "a enzima nao se gasta"
    assert realism.strip_repeated_lead("a enzima nao se gasta", "olha", ENZ) == "a enzima nao se gasta"


def test_r2_lead_only_messages_are_rare_varied_and_never_doubled(runs, mock_runs, adv_runs):
    rows = [_audit(x) for x in runs + mock_runs + adv_runs]
    for sim, a in zip(runs + mock_runs + adv_runs, rows):
        assert a["lead_dup"] == 0, [e for e in a["examples"] if e[0] == "lead_dup"]
        assert a["lead_only"] <= max(1, math.ceil(a["lines"] / sim.cfg.opener_every)), (a["lead_only"], a["lines"])
        ops = [m.text for m in _stu(sim) if m.plan.opener]
        assert len(ops) == len(set(ops)), ops            # each wording once
        per = {}
        for m in _stu(sim):
            if m.plan.opener:
                per[m.speaker] = per.get(m.speaker, 0) + 1
        assert max(per.values(), default=0) <= sim.cfg.opener_per_student_max
    s = audit.summarize(rows)
    assert s["lead_only/lines"] <= 1 / 15


# ================================================================== R3 stance changes
def _holder_state(seed=3):
    sim = _started(ENZ, n=3, seed=seed)
    st = sim.state
    h, c, o = st.students
    for x in st.students:
        x.misconceptions = []
    h.misconceptions = [MisconceptionState(id="m_enzyme_used_up", kc="kc_enzyme", strength=0.8, active=True)]
    st.step_idx = 1
    return sim, st, h, c, o


def test_r3_no_concession_without_a_refuting_move():
    """The review case: belief -> press -> "não tenho certeza" -> "mano kkk" -> never "ah entendi, faz sentido"."""
    for seed in range(12):
        sim, st, h, c, o = _holder_state(seed)
        v = add_student(st, h.id, ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_enzyme_used_up",
                                              kc="kc_enzyme"))
        v.text = "acho que as enzimas se gastam na reação"
        pr = add_student(st, c.id, ContentPlan(act="press", family="press_for_reasoning", kc="kc_enzyme",
                                               addresses_misconception="m_enzyme_used_up", anchor_turn=v.id, addressee=h.id))
        add_student(st, h.id, ContentPlan(act="explain", claim="misconception", misconception_id="m_enzyme_used_up",
                                          kc="kc_enzyme", rationale=True, anchor_turn=pr.id))
        add_student(st, c.id, ContentPlan(act="press", family="press_for_reasoning", kc="kc_enzyme",
                                          addresses_misconception="m_enzyme_used_up", anchor_turn=pr.id + 1, addressee=h.id))
        add_student(st, o.id, ContentPlan(act="backchannel"))
        h.misconceptions[0].active = False              # even if the hidden state dropped it
        for urgent in (False, True):
            assert sim.planner._concede(st, h, "kc_enzyme", sim.rngs["sched"], urgent=urgent) is None
        p = sim.planner._answer_press(st, h, st.messages[pr.id + 3], sim.rngs["sched"])
        assert p is None or not p.concedes, p
        # okaying an unrelated line is not a concession either
        plan = sim.planner._stance_gate(st, h, "kc_enzyme", ContentPlan(act="agree", anchor_turn=len(st.messages) - 1),
                                        sim.rngs["sched"])
        assert not plan.concedes


def test_r3_concession_follows_a_refuting_move_and_quotes_the_reason():
    for seed in range(8):
        sim, st, h, c, o = _holder_state(seed)
        v = add_student(st, h.id, ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_enzyme_used_up",
                                              kc="kc_enzyme"))
        r = add_student(st, c.id, ContentPlan(act="disagree", claim="correct", kc="kc_enzyme", is_correct=True,
                                              family="counterexample", addresses_misconception="m_enzyme_used_up",
                                              anchor_turn=v.id, addressee=h.id))
        r.text = "pera nao, mas se ela acabasse o biscoito ia parar de ficar mais doce e nao para"
        add_student(st, o.id, ContentPlan(act="backchannel"))
        h.misconceptions[0].active = False
        p = sim.planner._concede(st, h, "kc_enzyme", sim.rngs["sched"], urgent=True)
        assert p is not None and p.concedes == "m_enzyme_used_up" and p.reason_turn == r.id
        line = TemplateRealizer().content(p, h, ENZ, sim.rngs["realize"], sim.names, st)
        assert "se ela acabasse o biscoito ia parar de ficar mais doce" in line, line
        assert realism.near_dup(line, st, sim.names, p) is None          # quoting the reason is allowed


def test_r3_reason_clause_strips_leads():
    assert reason_clause(ENZ, "pera nao, mas se ela acabasse o biscoito ia parar de ficar mais doce e nao para") == \
        "se ela acabasse o biscoito ia parar de ficar mais doce"
    assert reason_clause(ENZ, "caio no meu diz que as enzimas nao se gastam, uma faz milhares", {"s1": "Caio"}) == \
        "as enzimas nao se gastam, uma faz milhares"


def test_r3_holder_does_not_switch_sides_silently():
    sim, st, h, c, o = _holder_state(5)
    h.misconceptions[0].active = False
    add_student(st, h.id, ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_enzyme_used_up",
                                      kc="kc_enzyme"))
    add_student(st, o.id, ContentPlan(act="backchannel"))
    claim = ContentPlan(act="on_task_claim", claim="correct", kc="kc_enzyme", claim_idx=1, is_correct=True)
    out = sim.planner._stance_gate(st, h, "kc_enzyme", claim, sim.rngs["sched"])
    assert not (out.claim == "correct" and out.claim_idx == 1)


def test_r3_flips_are_explained_and_holders_hedge_at_the_end(runs, mock_runs, adv_runs):
    rows = [_audit(x) for x in runs + mock_runs + adv_runs]
    for a in rows:
        assert a["flip_unexplained"] == 0, [e for e in a["examples"] if e[0] == "flip_unexplained"]
        assert a["closing_accept_unresolved"] == 0, [e for e in a["examples"] if e[0] == "closing_accept_unresolved"]
        assert a["flip_silent"] == 0, [e for e in a["examples"] if e[0] == "flip_silent"]
    s = audit.summarize(rows)
    assert s["concessions"] if "concessions" in s else True
    tot = sum(a["concessions"] for a in rows)
    assert tot >= 10 and sum(a["concede_no_reason"] for a in rows) / tot <= 0.1


# ================================================================== R4 questions
def test_r4_no_question_about_what_one_just_asserted():
    sim, st, h, c, o = _holder_state(2)
    add_student(st, h.id, ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_enzyme_used_up",
                                      kc="kc_enzyme"))
    h.known_facts = [f for f in h.known_facts if f != "f_reuse"]
    assert not sim.planner.askable(st, h, "f_reuse")       # "enzima acaba quando ela trabalha?" after the belief
    c.known_facts = [f for f in c.known_facts if f not in ("f_surface", "f_rate")]
    add_student(st, c.id, ContentPlan(act="on_task_claim", claim="correct", kc="kc_surface", claim_idx=0, is_correct=True))
    assert not sim.planner.askable(st, c, "f_surface") and not sim.planner.askable(st, c, "f_rate")
    o.known_facts = [f for f in o.known_facts if f != "f_optimum"]
    add_student(st, c.id, ContentPlan(act="on_task_claim", claim="fact", kc="kc_temp", fact_ids=["f_optimum"], is_correct=True))
    assert not sim.planner.askable(st, o, "f_optimum")       # already answered in the chat


def test_r4_same_student_never_reasks_an_equivalent_question():
    sim = _started(ENZ, n=3, seed=4)
    st = sim.state
    s = st.students[0]
    q = add_student(st, s.id, ContentPlan(act="question", kc="kc_surface", asked_fact="f_rate"))
    q.text = "onde as enzimas agem de verdade na comida?"
    add_student(st, st.students[1].id, ContentPlan(act="backchannel"), t=st.t + 240)
    st.t = q.t + 240
    again = ContentPlan(act="question", kc="kc_surface")
    assert realism.question_repeat("enzimas agem onde exatamente?", st, s.id, ENZ, again) is not None
    assert realism.question_repeat("oq tem na saliva afinal?", st, s.id, ENZ, again) is None


def test_r4_sessions_have_no_assert_then_ask_or_reasks(runs, mock_runs, adv_runs):
    for sim in runs + mock_runs + adv_runs:
        a = _audit(sim)
        assert a["assert_then_ask"] == 0 and a["reasked"] == 0 and a["asked_answered"] == 0, \
            [e for e in a["examples"] if e[0] in ("assert_then_ask", "reasked", "asked_answered")]


# ================================================================== R5 dyads and closings
def test_r5_singularize_rewrites_plural_address():
    for text in ("oi gente", "gente olha isso", "vcs concordam?", "oq vcs acham?", "alguem sabe?", "gente?",
                 "e ai gente, bora continuar?", "gente cade vcs", "galera a amilase quebra o amido"):
        out = realism.singularize(text, ENZ, "Ana")
        assert not realism.plural_address(out, ENZ), (text, out)
    assert not realism.plural_address("a gente coloca assim", ENZ)
    for text in ("hey guys", "anyone know?", "so does anyone know", "you guys agree?", "right guys?"):
        assert not realism.plural_address(realism.singularize(text, EN_TASKS[0], "Ana"), EN_TASKS[0]), text


def test_r5_no_plural_address_in_dyads(runs, mock_runs, adv_runs):
    dyads = [s for s in runs + mock_runs + adv_runs if len(s.state.roster) == 2]
    assert len(dyads) >= 10
    for sim in dyads:
        a = _audit(sim)
        assert a["plural_dyad"] == 0, [e for e in a["examples"] if e[0] == "plural_dyad"]


def test_r5_closing_okays_are_varied(runs, mock_runs, adv_runs):
    for sim in runs + mock_runs + adv_runs:
        a = _audit(sim)
        assert a["closing_repeat"] == 0, [e for e in a["examples"] if e[0] == "closing_repeat"]
    big = _sessions(seeds=range(0, 40), sizes=(3, 4))
    s = audit.summarize([_audit(x) for x in big])
    assert s["closing_top_session_share"] <= 0.15, s["closing_top_session_share"]
    assert s["closing_distinct_per_line"] >= 0.6


def test_r5_closing_okays_are_worded_by_the_model_and_add_nothing_proposed(mock_runs):
    assert not is_templated(ContentPlan(act="agree", closing="agree", anchor_turn=0))
    assert is_templated(ContentPlan(act="agree", closing="agree", brief=True, anchor_turn=0))
    for sim in mock_runs:
        recs = {r["msg"]: r for r in sim.realizer.log}
        for i in sim.state.closing_msgs[1:]:
            m = sim.state.messages[i]
            if m.plan.closing == "agree" and not (m.plan.brief or m.plan.holds_out or m.plan.concedes):
                assert not recs[i].get("templated"), recs[i]
    for sim in mock_runs + _sessions(seeds=range(0, 30), sizes=(3, 4, 6)):
        st = sim.state
        if not st.closing_msgs:
            continue
        prop = st.messages[st.closing_msgs[0]]
        covered = set(sim.task.facts_mentioned(prop.text))
        for i in st.closing_msgs[1:]:
            m = st.messages[i]
            if m.plan.closing == "adjust":
                assert not set(m.plan.fact_ids) & covered, (prop.text, m.text)


# ================================================================== Q1 dialogue state
def test_q1_credit_only_for_what_the_peer_said():
    from collabsim.sim.content import credit_source, fix_peer_attribution
    sim = _started(ENZ, n=3, seed=6)
    st = sim.state
    a, b, c = st.students
    names = sim.names
    m0 = add_student(st, a.id, ContentPlan(act="on_task_claim", claim="fact", fact_ids=["f_reuse"], kc="kc_enzyme",
                                           is_correct=True))
    m0.text = "as enzimas nao se gastam, uma faz milhares"
    m1 = add_student(st, b.id, ContentPlan(act="explain", claim="fact", fact_ids=["f_reuse"], kc="kc_enzyme",
                                           is_correct=True, reference=True, ref_by=a.id))
    m1.text = f"é como {names[a.id].lower()} falou, as enzimas nao se gastam"
    # circular: a's own fact, quoted back by b, never credited to b
    src = credit_source(ENZ, st, "f_reuse", a.id)
    assert src is None or src.speaker != b.id
    plan = ContentPlan(act="explain", claim="fact", fact_ids=["f_reuse"], kc="kc_enzyme", reference=True)
    out, fix = fix_peer_attribution(ENZ, st, c.id, plan, f"como {names[b.id].lower()} disse, as enzimas nao se gastam", names)
    assert fix is None                         # b did say it (quoting a)
    q = add_student(st, b.id, ContentPlan(act="question", kc="kc_temp", asked_fact="f_optimum"))
    q.text = "tem uma temperatura que é melhor pra ela?"
    plan = ContentPlan(act="explain", claim="fact", fact_ids=["f_optimum"], kc="kc_temp", reference=True)
    out, fix = fix_peer_attribution(ENZ, st, c.id, plan, f"como {names[b.id].lower()} disse, a melhor é 37", names)
    assert fix is not None and names[b.id].lower() not in out.lower(), out    # b only asked


def test_q1_one_burst_never_mixes_a_claim_with_its_belief():
    sim, st, h, c, o = _holder_state(7)
    h.misconceptions[0].active = True
    add_student(st, h.id, ContentPlan(act="on_task_claim", claim="fact", fact_ids=["f_amylase"], kc="kc_enzyme",
                                      is_correct=True))
    belief = ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_enzyme_used_up", kc="kc_enzyme")
    assert sim.planner._burst_gate(st, h, "kc_enzyme", belief, sim.rngs["sched"]).claim != "misconception"
    sim, st, h, c, o = _holder_state(8)
    add_student(st, h.id, belief.model_copy())
    q = ContentPlan(act="question", kc="kc_enzyme", asked_fact="f_amylase")
    assert sim.planner._burst_gate(st, h, "kc_enzyme", q, sim.rngs["sched"]).asked_fact != "f_amylase"


def test_q1_beliefs_are_voiced_for_the_activity_under_discussion():
    sim = _started(ENZ, n=3, seed=9)
    st = sim.state
    s = st.students[0]
    s.misconceptions = [MisconceptionState(id="m_hotter_faster", kc="kc_temp", strength=0.9, active=True)]
    st.step_idx = 0                         # activity 1: kc_surface; next: kc_enzyme
    plan = ContentPlan(act="on_task_claim", claim="misconception", misconception_id="m_hotter_faster", kc="kc_temp")
    assert sim.planner._burst_gate(st, s, "kc_surface", plan, sim.rngs["sched"]).claim != "misconception"
    st.step_idx = 1
    assert sim.planner._burst_gate(st, s, "kc_enzyme", plan.model_copy(), sim.rngs["sched"]).claim == "misconception"


def test_q1_no_false_attribution_in_sessions(runs, mock_runs, adv_runs):
    rows = [_audit(x) for x in runs + mock_runs + adv_runs]
    tot = sum(a["credits"] for a in rows)
    bad = sum(a["false_attribution"] for a in rows)
    assert tot >= 20 and bad / tot <= 0.03, [e for a in rows for e in a["examples"] if e[0] == "false_attribution"]
    for a in [_audit(x) for x in adv_runs]:
        assert a["false_attribution"] == 0, [e for e in a["examples"] if e[0] == "false_attribution"]


# ================================================================== Q2 final answer
def test_q2_final_answer_is_built_from_what_was_established(runs, mock_runs):
    texts, proposers = set(), []
    for sim in runs + mock_runs:
        st = sim.state
        if not st.closing_msgs:
            continue
        prop = st.messages[st.closing_msgs[0]]
        assert prop.plan.closing == "propose"
        assert set(prop.plan.ideas) <= set(sim.planner.established_ideas(st)), (prop.plan.ideas, st.ideas_voiced)
        task = sim.task
        for m in st.messages[:prop.id]:
            if m.plan is not None and m.plan.concedes:     # a belief given up: its correct version is in the answer
                mid = m.plan.concedes
                kc = task.misconception(mid).kc
                ok = {k.id for k in task.all_key_ideas() if k.kc == kc and (
                    set(k.facts) & set(task.refuting_facts(mid)) or set(k.claims) & set(task.refuting_claims(mid)))}
                assert not ok or ok & set(prop.plan.ideas), (mid, prop.plan.ideas)
        texts.add(prop.text)
        stu = _stu(sim)
        proposers.append(sum(1 for m in stu if m.speaker == prop.speaker and m.plan.claim in ("fact", "correct")))
    assert len(texts) >= 0.9 * len(proposers)                      # no fixed join of the task's claims
    assert sum(p >= 1 for p in proposers) / len(proposers) >= 0.85    # someone who took part writes it up


# ================================================================== Q3 transitions, off-task
def test_q3_step_transitions_and_offtask_threads(runs):
    rows = [_audit(x) for x in runs]
    multi = [s for s in runs if len(s.state.steps_solved) >= 2]
    moved = sum(1 for s in multi if any(m.plan.step_move for m in _stu(s)))
    assert moved >= 0.2 * len(multi), (moved, len(multi))
    s = audit.summarize(rows)
    assert s["offtask_replied/offtask_threads"] >= 0.28, s["offtask_replied/offtask_threads"]
    for sim in runs:
        for m in _stu(sim):
            if m.plan.step_move:
                prev = [x for x in sim.state.messages[:m.id] if x.kind == "student"]
                assert sim.state.steps_solved and any(d["msg"] < m.id for d in sim.state.steps_solved)


def test_q3_offtask_lines_vary_across_seeds_and_are_never_addressed_out_of_the_blue(adv_runs):
    big = _sessions(seeds=range(0, 60), sizes=(3, 4, 5))
    s = audit.summarize([_audit(x) for x in big])
    assert s["offtask_common_line_share"] <= 0.08, s["offtask_common_line_share"]
    assert s["offtask_distinct_per_line"] >= 0.55
    for sim in big + adv_runs:
        a = _audit(sim)
        assert a["offtask_named"] == 0, [e for e in a["examples"] if e[0] == "offtask_named"]


# ================================================================== audit helper
def test_audit_flags_the_review_lines():
    def msg(i, sp, text, t=None, **plan):
        return {"id": i, "t": (t if t is not None else i * 10.0), "speaker": sp, "kind": "student", "text": text,
                "display_name": {"s1": "Enzo", "s2": "Caio", "s3": "Malu"}[sp], "addressee": plan.get("addressee"),
                "plan": {"act": "on_task_claim", "claim": "none", **plan}}
    tr = [msg(0, "s1", "pelo que eu li, a amilase quebra o amido em maltose que", claim="fact", fact_ids=["f_amylase"],
              kc="kc_enzyme"),
          msg(1, "s1", "é um açucar", act="explain", split_tail="é um açucar"),
          msg(2, "s2", "olha", act="backchannel", opener=True),
          msg(3, "s2", "olha, pelo que eu li, a enzima é um catalisador", claim="fact", fact_ids=["f_reuse"], kc="kc_enzyme"),
          msg(4, "s1", "acho que as enzimas se gastam na reação, tipo um combustível que vai acabando",
              claim="misconception", misconception_id="m_enzyme_used_up", kc="kc_enzyme"),
          msg(5, "s1", "enzima acaba quando ela trabalha?", act="question", asked_fact="f_reuse", kc="kc_enzyme"),
          msg(6, "s3", "mano kkk", act="backchannel"),
          msg(7, "s1", "ah entendi, faz sentido", act="agree", concedes="m_enzyme_used_up"),
          msg(8, "s3", "como caio disse, a amilase funciona melhor perto de 37", claim="fact", fact_ids=["f_optimum"],
              kc="kc_temp", reference=True),
          msg(9, "s2", "onde as enzimas agem de verdade na comida?", act="question", asked_fact="f_rate", kc="kc_surface"),
          msg(10, "s2", "enzimas agem onde exatamente?", act="question", kc="kc_surface"),
          msg(11, "s3", "bora fechar: a amilase quebra o amido", act="explain", closing="propose"),
          msg(12, "s1", "fechou, pode mandar assim", act="agree", closing="agree"),
          msg(13, "s2", "fechou, pode mandar assim", act="agree", closing="agree")]
    a = audit.audit_session(ENZ, tr, n_students=3)
    assert a["bad_split"] == 1
    assert a["lead_only"] == 1 and a["lead_dup"] == 1
    assert a["assert_then_ask"] == 1
    assert a["flip_unexplained"] == 1
    assert a["reasked"] == 1
    assert a["false_attribution"] == 1
    assert a["closing_repeat"] == 1
    d = audit.audit_session(ENZ, [dict(m, text="gente vcs acham?") if m["id"] == 6 else m for m in tr], n_students=2)
    assert d["plural_dyad"] >= 1
    s = audit.summarize([a, d])
    assert s["closing_top_session_share"] == 1.0
