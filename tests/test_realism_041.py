"""Realism controls: one test group per defect of an earlier real-Haiku review (off-task density,
incoherent bursts, cross-speaker near-duplicates, mentions of people who are not in the chat,
typos, laughter, world consistency, weak ending, bare fragments). Offline only (template / mock)."""
from __future__ import annotations

import re
import typing

import pytest

from collabsim import Simulator
from collabsim.models import load_task
from collabsim.models.message import Act
from collabsim.sim import realism
from collabsim.sim.config import SimConfig

SEEDS = range(1, 13)


def _run(backend="template", seeds=SEEDS, **kw):
    sim = Simulator(backend=backend)
    out = []
    for seed in seeds:
        s = sim.new_session(seed=seed, pacing="manual", **{"n_students": 4, **kw})
        s.run_to_end()
        out.append(s._live.sim)
        s.close()
    return out


@pytest.fixture(scope="module")
def template_runs():
    return _run("template")


@pytest.fixture(scope="module")
def mock_runs():
    return _run("mock", seeds=range(1, 5))


@pytest.fixture(scope="module")
def size_runs():
    return [r for n in (2, 3, 6, 10) for r in _run("template", seeds=range(1, 4), n_students=n)]


def _stu(sim):
    return [m for m in sim.state.messages if m.kind == "student"]


def _all(*groups):
    return [r for g in groups for r in g]


# ------------------------------------------------------------------ 1. off-task rate / threads / topics
def _off_share(runs):
    return [sum(m.plan.act == "off_task" for m in _stu(r)) / len(_stu(r)) for r in runs]


def test_offtask_rate_near_default_target(template_runs):
    shares = _off_share(template_runs)
    mean = sum(shares) / len(shares)
    assert 0.11 <= mean <= 0.19, (mean, shares)
    assert max(shares) <= 0.26


def test_offtask_rate_is_configurable_and_validated():
    low = _off_share(_run("template", seeds=range(1, 7), offtask_rate=0.05))
    assert sum(low) / len(low) <= 0.08
    assert all(s == 0 for s in _off_share(_run("template", seeds=range(1, 4), offtask_rate=0.0)))
    high = _off_share(_run("template", seeds=range(1, 7), offtask_rate=0.3))
    assert sum(high) / len(high) > sum(low) / len(low) + 0.1
    with pytest.raises(ValueError):
        Simulator(backend="template").new_session(offtask_rate=0.6)
    with pytest.raises(ValueError):
        Simulator(backend="template").new_session(offtask_rate=-0.1)


def test_offtask_spacing_no_same_speaker_runs_and_threads(template_runs, size_runs):
    for r in _all(template_runs, size_runs):
        stu = _stu(r)
        for a, b in zip(stu, stu[1:]):
            assert not (a.speaker == b.speaker and a.plan.act == "off_task" and b.plan.act == "off_task"), (a.text, b.text)
        idx = [i for i, m in enumerate(stu) if m.plan.act == "off_task"]
        for i, j in zip(idx, idx[1:]):
            reply = stu[j].plan.anchor_turn is not None
            assert reply or j - i > r.cfg.offtask_min_spacing, (stu[i].text, stu[j].text)
            if reply:
                assert stu[j].plan.anchor_turn == stu[i].id and stu[j].plan.topic == stu[i].plan.topic
                assert stu[j].speaker != stu[i].speaker


def test_offtask_new_topics_never_repeat_in_session(template_runs, size_runs):
    from collabsim.sim.offtask import TOPIC_FAMILY, TOPICS_NOT_ONLINE
    for r in _all(template_runs, size_runs):
        new = [m.plan.topic for m in _stu(r) if m.plan.act == "off_task" and m.plan.anchor_turn is None]
        fams = [TOPIC_FAMILY.get(t, t) for t in new]
        assert len(fams) == len(set(fams)), new
        assert not set(new) & set(TOPICS_NOT_ONLINE)


def test_some_offtask_threads_have_replies(template_runs):
    assert sum(m.plan.act == "off_task" and m.plan.anchor_turn is not None
               for r in template_runs for m in _stu(r)) >= 3


# ------------------------------------------------------------------ 2. same-speaker bursts are continuations
def test_bursts_are_coherent_and_capped(template_runs, mock_runs, size_runs):
    for r in _all(template_runs, mock_runs, size_runs):
        stu = _stu(r)
        run = 1
        for a, b in zip(stu, stu[1:]):
            run = run + 1 if a.speaker == b.speaker else 1
            assert run <= r.cfg.max_run_hard_cap
            if a.speaker == b.speaker and b.plan.closing is None:
                assert b.plan.act != "off_task", (a.text, b.text)
                if a.plan.act != "greeting":   # "oi gente" -> "where do i type it?" is fine
                    assert b.plan.act != "interface" and not (b.plan.act == "question" and b.plan.intent), (a.text, b.text)
                if not a.plan.opener:          # "pera" -> "acho que ..." is one contribution in two messages
                    assert a.plan.act not in ("off_task", "idk", "refusal", "backchannel"), (a.text, b.text)
                assert b.plan.act not in ("idk", "refusal", "backchannel", "agree"), (a.text, b.text)


# ------------------------------------------------------------------ 3. session-wide near-duplicates
def test_no_cross_speaker_near_duplicates(template_runs, mock_runs, size_runs):
    for r in _all(template_runs, mock_runs, size_runs):
        stu = [m for m in _stu(r) if len(realism._tokens(m.text)) >= realism.NEAR_DUP_MIN_TOKENS]
        names = {realism.fold(n) for n in r.names.values()}
        core = [" ".join(t for t in realism._tokens(m.text) if t not in names) for m in stu]
        for i in range(len(stu)):
            for j in range(i):
                if stu[i].plan.concedes and stu[i].plan.reason_turn is not None and \
                        stu[i].plan.reason_turn <= stu[j].id <= stu[i].plan.reason_turn + 1:
                    continue      # 0.5.3: a concession quotes the reason it gives in to
                jac, cont = realism.similarity(core[i], core[j])
                # typos are added after the check, so allow a small margin
                assert jac < realism.NEAR_DUP_JACCARD + 0.05 and cont < realism.NEAR_DUP_CONTAIN + 0.05, (
                    stu[j].text, stu[i].text)


def test_near_dup_detector_is_typo_and_laughter_robust():
    a = "kkk ah entao a amilase da saliva quebra o amido em acucar, por isso o biscoito fica doce"
    b = "lara, então a amilase da saliva quebra o amido em açúcar, por isso o biscoito fica doce, né? faz sentido kkkk"
    jac, cont = realism.similarity(b, a)
    assert jac >= realism.NEAR_DUP_JACCARD or cont >= realism.NEAR_DUP_CONTAIN
    jac, cont = realism.similarity("é como caio falou, pedaço menor tem mais area",
                                   "acho que tem a ver com isso: pedaço menor = mt mais area de superficie")
    assert jac < realism.NEAR_DUP_JACCARD and cont < realism.NEAR_DUP_CONTAIN


def test_interface_confusions_once_per_session_and_answerable(template_runs):
    answered = 0
    for r in template_runs:
        keys = [m.plan.intent for m in _stu(r) if m.plan.intent and not m.plan.answers_intent]
        assert len(keys) == len(set(keys)), keys
        for m in _stu(r):
            if m.plan.answers_intent:
                q = r.state.messages[m.plan.anchor_turn]
                assert q.plan.intent == m.plan.intent and q.speaker != m.speaker
                answered += 1
    assert answered >= 3


def test_content_questions_asked_once_per_session(template_runs):
    for r in template_runs:
        asked = [m.plan.asked_fact for m in _stu(r) if m.plan.act == "question" and m.plan.asked_fact
                 and not m.plan.follow_up]
        assert len(asked) == len(set(asked)), asked


# ------------------------------------------------------------------ 4. people who are not in the chat
def test_nobody_absent_is_ever_mentioned(template_runs, mock_runs, size_runs):
    """No role words (tutor, bot, AI...), no "@", no pings: students talk only to who is there."""
    for r in _all(template_runs, mock_runs, size_runs):
        for m in _stu(r):
            assert realism.participant_mention_violation(m.text, r.state, r.task) is None, m.text


def test_no_ping_few_shots_or_banks():
    from collabsim.lang import pt_br
    from collabsim.llm.prompts import fewshots, realizer_prefix
    from collabsim.sim.realizer import FILLER
    assert not any("ping" in k for k in FILLER) and not any("ping" in k for k in pt_br.FILLER)
    assert {f["act"] for f in fewshots() + pt_br.fewshots()} <= set(typing.get_args(Act))
    r = _run("mock", seeds=[1])[0]
    pre = realizer_prefix(r.task, r.state)
    assert not re.search(r"@\w", pre) and "Fale só com quem já escreveu neste chat" in pre


def test_a_silent_participant_is_never_mentioned():
    """A declared participant who has not written yet is never named by the students; once they
    have written, students may address them."""
    sim = Simulator(backend="template")
    seen_named = False
    for seed in range(1, 7):
        s = sim.new_session(seed=seed, pacing="manual", n_students=4, participants=["Maria", "Joana"])
        for _ in range(12):
            s.step()
        st = s._live.sim.state
        assert not any("maria" in realism.fold(m.text) or "joana" in realism.fold(m.text)
                       for m in st.messages if m.kind == "student")
        s.say("gente, acho que mastigar é so pra comida descer mais facil", sender="Maria")
        s.run_to_end()
        later = [m for m in st.messages if m.kind == "student"]
        assert not any("joana" in realism.fold(m.text) for m in later)          # Joana never wrote
        seen_named |= any(m.addressee == "e1" for m in later)
        s.close()
    assert seen_named


# ------------------------------------------------------------------ 5. typos
def test_typos_sparse_one_per_message(template_runs):
    typos = sum(1 for r in template_runs for x in r.state.slippage_log
                if x.get("type") == "realism_polish" and "typo" in x)
    n = sum(len(_stu(r)) for r in template_runs)
    assert 0.02 <= typos / n <= 0.25, typos / n


def test_typo_unit_at_most_one_word_changed():
    from collabsim.models.rng import RNG
    r = _run("template", seeds=[2])[0]
    s = r.state.students[0]
    s.traits.typo_rate = 0.12   # p(message) = .3
    task = r.task
    changed = 0
    text = "acho que você também sabe porque a gente não terminou essa parte hoje"
    for i in range(300):
        info = {}
        out = realism._typo(text, None, s, task, RNG(i), info)
        diff = sum(a != b for a, b in zip(text.split(" "), out.split(" ")))
        assert diff <= 1 and len(out.split(" ")) == len(text.split(" "))
        changed += diff
    assert 40 <= changed <= 140


def test_prompt_no_longer_asks_for_typos():
    from collabsim.llm.prompts import realizer_prefix
    r = _run("mock", seeds=[1])[0]
    assert "erros de digitação de vez em quando" not in realizer_prefix(r.task, r.state)
    assert "erros de digitação de vez em quando" not in realizer_prefix(r.task)   # typos are added in code


# ------------------------------------------------------------------ 6. laughter
def test_laughter_capped_and_rare_on_content(template_runs, mock_runs):
    for r in _all(template_runs, mock_runs):
        stu = _stu(r)
        assert sum(realism.has_laugh(m.text) for m in stu) / len(stu) <= r.cfg.laughter_cap + 0.08
    content = [m for r in _all(template_runs, mock_runs) for m in _stu(r)
               if m.plan.claim in ("fact", "correct", "misconception")]
    assert sum(realism.has_laugh(m.text) for m in content) / len(content) <= 0.1


def test_laughter_openers_not_repeated(template_runs):
    for r in template_runs:
        stu = _stu(r)
        for i, m in enumerate(stu):
            if realism.starts_with_laugh(m.text) and realism.strip_laugh(m.text):
                prev = [p for p in stu[max(0, i - 5):i] if realism.starts_with_laugh(p.text) and realism.strip_laugh(p.text)]
                assert not prev, (prev[0].text, m.text)


# ------------------------------------------------------------------ 7. session context
def test_world_consistency(template_runs, mock_runs, size_runs):
    for r in _all(template_runs, mock_runs, size_runs):
        assert r.state.context["time_of_day"] in realism.PERIODS and r.state.context["setting"] == "online"
        for m in _stu(r):
            assert realism.world_violation(m.text, r.state, r.task) is None, (r.state.context, m.text)


def test_time_of_day_param_and_prompt():
    from collabsim.llm.prompts import realizer_prefix
    r = _run("mock", seeds=[4], time_of_day="noite")[0]
    assert r.state.context["time_of_day"] == "noite"
    assert "Período do dia: noite" in realizer_prefix(r.task, r.state)
    for m in _stu(r):
        if m.plan.act == "greeting":
            assert "bom dia" not in m.text and "boa tarde" not in m.text
    r = _run("template", seeds=[4], time_of_day="morning")[0]
    assert r.state.context["time_of_day"] == "manha"
    with pytest.raises(ValueError):
        Simulator(backend="template").new_session(time_of_day="madrugada")


def test_world_violation_unit():
    r = _run("template", seeds=[1], time_of_day="noite")[0]
    st, task = r.state, r.task
    assert realism.world_violation("kkkk alguém aí já almoçou?", st, task)["reason"] == "wrong_time_of_day"
    assert realism.world_violation("o prof ta falando sozinho de novo", st, task)["reason"] == "physical_classroom"
    assert realism.world_violation("o tutor cadê? preciso de ajuda", st, task)["reason"] == "participant_mention"
    assert realism.world_violation("@ana oq vc acha", st, task)["reason"] == "participant_mention"
    st.participants["e1"] = {"name": "Maria", "avatar": None, "first_spoke_t": None}
    assert realism.world_violation("maria cadê vc?", st, task)["reason"] == "participant_mention"
    st.participants["e1"]["first_spoke_t"] = 10.0
    assert realism.world_violation("maria cadê vc?", st, task) is None
    assert realism.world_violation("boa noite gente", st, task) is None


# ------------------------------------------------------------------ 8. closing synthesis
def test_closing_synthesis_when_complete(template_runs, mock_runs, size_runs):
    done = 0
    for r in _all(template_runs, mock_runs, size_runs):
        st = r.state
        if st.ended_reason != "complete":
            continue
        done += 1
        # 0.5.1: the proposer and, with 3+ students, at least 2 others answer (tests/test_defects_051.py)
        assert st.closing == "done" and len(st.closing_msgs) >= 2, (st.closing, st.closing_msgs)
        prop, *confs = (st.messages[i] for i in st.closing_msgs)
        assert prop.plan.closing == "propose" and prop.t >= st.solved_at
        for conf in confs:
            assert conf.plan.closing in ("agree", "adjust") and prop.speaker != conf.speaker
            assert conf.plan.anchor_turn == prop.id
        assert len({c.speaker for c in confs}) == len(confs)
        assert all(m.id <= confs[-1].id for m in _stu(r))   # the chat ends right after the answers
    assert done >= 10


# ------------------------------------------------------------------ 9. framed content statements
def test_fact_statements_framed_not_fragments(template_runs):
    from collabsim.lang.pt_br import FACT_LEADS_FRAMED as FRAMED, NATURAL_LEADS, NEUTRAL_LEAD
    FACT_LEADS_FRAMED = list(FRAMED) + list(NATURAL_LEADS) + [NEUTRAL_LEAD]   # capped leads are swapped
    for r in template_runs:
        for m in _stu(r):
            p = m.plan
            if p.act in ("on_task_claim", "explain") and p.claim == "fact" and not p.reference and not p.family \
                    and not p.revoice:
                import difflib
                low = re.sub(r"\bq\b", "que", realism.fold(m.text))   # "li q ..." (chat abbreviation)
                assert any(difflib.SequenceMatcher(None, low[:len(l.strip())], realism.fold(l).strip()).ratio() >= 0.75
                           for l in FACT_LEADS_FRAMED), m.text   # (a typo may hit the lead)
            if p.claim in ("fact", "correct", "misconception") and not p.reference:
                assert p.target_len >= 7


def test_llm_request_carries_format_and_continuation_lines():
    from collabsim.llm.prompts import realizer_request
    from collabsim.models.message import ContentPlan
    r = _run("mock", seeds=[1])[0]
    st, s = r.state, r.state.students[0]
    f = s.known_facts[0]
    req = realizer_request("claude-haiku-5-5", r.task, st, s, ContentPlan(act="on_task_claim", kc=r.task.fact(f).kc,
                           claim="fact", fact_ids=[f], continuation=True, target_len=8), r.names)
    assert "FORMATO:" in req.suffix and "CONTINUAÇÃO:" in req.suffix
    prop = realizer_request("claude-haiku-5-5", r.task, st, s, ContentPlan(act="explain", closing="propose"), r.names)
    assert "resposta final do grupo" in prop.suffix


# ------------------------------------------------------------------ determinism
def test_realism_sessions_deterministic():
    a = [m.text for m in _run("template", seeds=[5])[0].state.messages]
    b = [m.text for m in _run("template", seeds=[5])[0].state.messages]
    assert a == b


def test_one_mode_only_realism_is_always_on():
    assert "realism" not in SimConfig.model_fields
    sim = Simulator(backend="template").new_session(seed=1, pacing="manual")
    assert sim._live.sim.state.context["time_of_day"] in realism.PERIODS

