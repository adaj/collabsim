"""Configurable language (pt-BR default for a Simulator session, English available)."""
import re

import pytest

from collabsim import lang
from collabsim.lang import pt_br as PT
from collabsim.llm.client import approx_tokens
from collabsim.llm.prompts import judge_prefix, realizer_prefix, realizer_request
from collabsim.models import RNG, ContentPlan, Message, load_task
from collabsim.models.session import SessionState
from collabsim.sim import promptleak
from collabsim.sim.config import SimConfig
from collabsim.sim.content import fix_self_attribution, is_interrogative, leak_triggers, presupposes
from collabsim.sim.factory import build_session
from collabsim.sim.group import generate_group
from collabsim.sim.realizer import FILLER, PT_MARKERS_RE

# English template / chat leftovers that must never appear in a pt-BR line
EN_LEFTOVER = re.compile(r"\b(lol|lmao|haha\w*|idk|yeah|yep|guys|anyone|mine says|my card|my sheet|i think|like i said|"
                         r"what do|the|and|you|apparently|wait|so what|does|is it|maybe|dont|im|whats)\b", re.I)


def test_normalize_and_packs():
    assert lang.normalize("pt") == lang.normalize("pt_br") == lang.normalize("PT-BR") == "pt-BR"
    assert lang.normalize("en-US") == "en"
    with pytest.raises(lang.UnsupportedLanguage):
        lang.normalize("fr")
    assert set(PT.FILLER) == set(FILLER)
    from collabsim.sim import intents, offtask
    assert set(PT.INTERFACE) == set(intents.INTERFACE) and set(PT.PROCEDURAL) == set(intents.PROCEDURAL)
    assert set(PT.TOPICS) == set(offtask.TOPICS) and all(len(v[1]) >= 5 for v in PT.TOPICS.values())
    assert len(PT.NAMES) >= 10 and len(set(PT.NAMES)) == len(PT.NAMES)


def test_pt_banks_pass_prompt_leak_detector_and_have_no_english():
    banks = [l for v in PT.FILLER.values() for l in v] + [l for _, ls in PT.TOPICS.values() for l in ls]
    banks += PT.HEDGES + PT.SAFE_QUESTIONS + [l for v in PT.MINIMAL_ACT_LINES.values() for l in v]
    banks += [p + "x" for p in PT.FACT_LEADS + PT.REF_LEADS + PT.REF_LEADS_NEUTRAL + PT.HEDGE_PREFIX
              + PT.CLAIM_PREFIX + PT.DISAGREE_PREFIX]
    for line in banks:
        line = line.replace("{name}", "ana").replace("{peer}", "ana")
        assert not promptleak.hits(line, {}, None), line
        assert not EN_LEFTOVER.search(line), line


@pytest.mark.parametrize("task_id", ["sci-enzyme-01", "math-ratio-01"])
def test_pt_task_mirrors_en(task_id):
    en, pt = load_task(task_id), load_task(task_id, language="pt-BR")
    assert pt.language == "pt-BR" and en.language == "en"
    assert [f.id for f in en.facts] == [f.id for f in pt.facts] and en.jigsaw == pt.jigsaw
    assert en.refutes == pt.refutes and [m.id for m in en.misconceptions] == [m.id for m in pt.misconceptions]
    for f1, f2 in zip(en.facts, pt.facts):
        for c in f2.chat:
            assert f2.id in pt.facts_mentioned(c)
    for k1, k2 in zip(en.kcs, pt.kcs):   # same leak structure: claims carry the same fact keyphrases
        for c1, c2 in zip(k1.correct_claims, k2.correct_claims):
            assert sorted(en.facts_mentioned(c1)) == sorted(pt.facts_mentioned(c2))
    assert not any(any(k in pt.public_text() for k in f.keyphrases) for f in pt.facts)


@pytest.mark.parametrize("realizer", ["template", "mock"])
@pytest.mark.parametrize("task_id", ["sci-enzyme-01", "math-ratio-01"])
def test_pt_sessions_have_no_english_leftovers(realizer, task_id):
    t = load_task(task_id, language="pt-BR")
    seen_kkk = False
    for seed, n in [(1, 3), (2, 4), (3, 6)]:
        sim = build_session(t, seed, realizer=realizer, n_students=n, cfg=SimConfig(max_student_turns=80))
        sim.run()
        names = set(sim.names.values())
        for m in sim.state.messages:
            words = [w for w in m.text.split() if w.strip("?,.!") .capitalize() not in names]
            txt = " ".join(words)
            assert not EN_LEFTOVER.search(txt), (seed, m.text)
            seen_kkk |= "kkk" in m.text
        assert all(s.name in PT.NAMES for s in sim.state.students)
    assert seen_kkk   # pt-BR laughter is right in pt-BR


@pytest.mark.parametrize("realizer", ["template", "mock"])
def test_en_sessions_have_no_pt_slang(realizer):
    for seed in (1, 2):
        sim = build_session("sci-enzyme-01", seed, realizer=realizer, n_students=4,
                            cfg=SimConfig(max_student_turns=80))
        sim.run()
        for m in sim.state.messages:
            assert not PT_MARKERS_RE.search(m.text), m.text


def test_pt_question_detection_for_leak_guard():
    t = load_task("sci-enzyme-01", language="pt-BR")
    den = t.fact("f_denature")
    assert is_interrogative("a enzima desnatura acima de 60?", "pt-BR")       # pt yes/no keeps word order
    assert is_interrogative("tem uma temperatura melhor pra ela?", "pt-BR")
    assert is_interrogative("o que acontece se esquentar", "pt-BR")
    assert not is_interrogative("acho que é a temperatura do corpo?", "pt-BR")  # hedged assertion
    assert not is_interrogative("ela desnatura acima de 60, né?", "pt-BR")      # tag question asserts
    assert not is_interrogative("ela desnatura", "pt-BR")
    assert presupposes("pq ela desnatura acima de 60?", den, "pt-BR")
    g = generate_group(t, RNG(1), SimConfig(), n_students=3)
    s = next(x for x in g if "f_denature" not in x.known_facts)
    st = SessionState(session_id="x", task_id=t.id, seed=0, students=g)
    plan = ContentPlan(act="on_task_claim", kc="kc_temp", claim="hedge")
    kinds = lambda text: {r["fact_id"]: r["type"] for r in leak_triggers(t, s, plan, text, st)}  # noqa: E731
    assert kinds("acima de 60 ela desnatura e para")["f_denature"] == "true_leak"
    assert kinds("será que ela desnatura?")["f_denature"] == "question"
    assert kinds("pq ela desnatura acima de 60?")["f_denature"] == "true_leak"


def test_pt_self_attribution_fix():
    t = load_task("sci-enzyme-01", language="pt-BR")
    g = generate_group(t, RNG(3), SimConfig(), n_students=3)
    st = SessionState(session_id="x", task_id=t.id, seed=0, students=g)
    a, b = g[0], g[1]
    f = a.jigsaw_facts[0]
    p0 = ContentPlan(act="on_task_claim", kc=t.fact(f).kc, claim="fact", fact_ids=[f], is_correct=True)
    st.messages.append(Message(id=0, t=1.0, speaker=a.id, kind="student", text=t.fact(f).chat[0], plan=p0))
    a.shared_facts.append(f)
    plan = ContentPlan(act="explain", kc=t.fact(f).kc, claim="fact", fact_ids=[f], is_correct=True, reference=True)
    names = {s.id: s.name for s in g}
    out, fix = fix_self_attribution(t, st, b.id, plan, "como eu disse " + t.fact(f).chat[0], names)
    assert fix is not None and out.startswith(f"como {a.name.lower()} falou")
    same, nofix = fix_self_attribution(t, st, a.id, plan, "como eu disse " + t.fact(f).chat[0], names)
    assert nofix is None


def test_pt_prompts():
    t = load_task("sci-enzyme-01", language="pt-BR")
    pre = realizer_prefix(t)
    assert "português do Brasil" in pre and "kkkk" in pre and "oq a gente tem q fazer msm?" in pre   # invented pt-BR few-shot
    assert "STYLE" not in pre and "STYLE EXAMPLES" not in pre and "EXEMPLOS DE ESTILO" in pre
    assert approx_tokens(pre) >= 512 and approx_tokens(judge_prefix(t)) >= 512
    assert "português do Brasil" in judge_prefix(t)
    assert "pt-BR" not in judge_prefix(load_task("sci-enzyme-01"))                       # English prefix in English
    sim = build_session(t, 2, realizer="mock", n_students=3)
    sim.run(max_events=10)
    s = sim.state.students[0]
    req = realizer_request("claude-haiku-5-5", t, sim.state, s, ContentPlan(act="greeting"), sim.names)
    assert "DIGA ESTE CONTEÚDO" in req.suffix and "VOCÊ NÃO SABE" in req.suffix
    assert "YOU KNOW" not in req.suffix and "SAY THIS" not in req.suffix
    assert not promptleak.hits("DIGA ESTE CONTEÚDO: oi") == []
