"""The package carries no facilitator-agent concept: no such words in its source or in any prompt it
builds, no ping act, and students never call or mention anyone who is not in the chat."""
from __future__ import annotations

import re
import typing
from pathlib import Path

import pytest

from collabsim import Simulator
from collabsim.llm.prompts import check_request, judge_prefix, realizer_prefix, realizer_request
from collabsim.models import ContentPlan, load_task
from collabsim.models.message import Act
from collabsim.runtime.interpret import interpret_prefix, interpret_request
from collabsim.runtime.taskspec import draft_request
from collabsim.sim import realism
from collabsim.types import EventKind

PKG = Path(__file__).resolve().parents[1] / "collabsim"
# a few terms are written with a character class so the pattern itself does not contain them verbatim
FORBIDDEN = re.compile(r"\bcl[a]ir\b|agent|talk[ _-]?moves?|\bapt\b|w[o]z|wiz[a]rd|benchmark|ladder|prereg|"
                       r"theor(?:y|ies)|intrusion|scaffold", re.I)
PING = re.compile(r"\b(?:you there|are you there|ta ai|tá aí|cade vc|cadê vc|help us|ajuda (?:a gente|nois))\b", re.I)


def test_package_source_is_clean():
    hits = []
    for p in sorted(PKG.rglob("*")):
        if p.suffix not in (".py", ".yaml", ".md") or "__pycache__" in p.parts:
            continue
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if FORBIDDEN.search(line):
                hits.append(f"{p.relative_to(PKG)}:{i}: {line.strip()}")
    assert not hits, "\n".join(hits)


def test_literals_have_no_ping_or_agent_values():
    acts = typing.get_args(Act)
    kinds = typing.get_args(EventKind)
    assert not any("ping" in a or "agent" in a for a in acts)
    assert set(kinds) == {"student", "external", "system"}


def _prompts(language: str, participants: list[str]):
    task = load_task("sci-enzyme-01", language=language)
    sim = Simulator(backend="mock")
    s = sim.new_session(task="sci-enzyme-01", language=language, n_students=4, seed=2, pacing="manual",
                        participants=participants)
    for _ in range(6):
        s.step()
    live = s._live
    if participants:
        s.say(f"{s.roster[0]['name']}, o que você acha da mastigação?", sender=participants[0])
        s.step()
    st, eng = live.sim.state, live.sim
    stu = st.students[0]
    out = [realizer_prefix(task, st), judge_prefix(task), interpret_prefix(task),
           draft_request("claude-haiku-5-5", "uma atividade sobre plantas", language).prefix]
    for plan in (ContentPlan(act="on_task_claim", kc="kc_surface", claim="fact", fact_ids=["f_surface"],
                             is_correct=True, target_len=8),
                 ContentPlan(act="question", kc="kc_enzyme", asked_fact="f_amylase", target_len=6),
                 ContentPlan(act="backchannel", target_len=1), ContentPlan(act="idk", target_len=2),
                 ContentPlan(act="explain", closing="propose", partial=True, ideas=[], target_len=16)):
        out.append(realizer_request("claude-haiku-5-5", task, st, stu, plan, eng.names).suffix)
    out.append(check_request("claude-haiku-5-5", task, stu, "oi", [], [], [], "-").suffix)
    out.append(interpret_request("claude-haiku-5-5", task, st, "oi gente", "e1" if participants else "s1").suffix)
    s.close()
    return out, participants


@pytest.mark.parametrize("language", ["pt-BR", "en"])
@pytest.mark.parametrize("participants", [[], ["Maria", "Bot"]])
def test_built_prompts_are_clean(language, participants):
    prompts, parts = _prompts(language, participants)
    for p in prompts:
        bad = [m.group(0) for m in FORBIDDEN.finditer(p)]
        assert not bad, (bad, p[:200])
    if parts:  # the speaker sees only participants who have written; "Bot" never wrote
        suffix = prompts[4]
        line = next(l for l in suffix.splitlines() if l.startswith(("IN THE CHAT", "NO CHAT")))
        assert "Maria" in line and "Bot" not in line


@pytest.mark.parametrize("backend", ["template", "mock"])
def test_sessions_without_participants_never_ping_or_mention_anyone(backend):
    sim = Simulator(backend=backend)
    seeds = range(1, 31) if backend == "template" else range(1, 11)
    for seed in seeds:
        lang_ = ("pt-BR", "en")[seed % 2]
        s = sim.new_session(task=("sci-enzyme-01", "math-ratio-01")[seed % 2], language=lang_, seed=seed,
                            n_students=(2, 3, 4, 6)[seed % 4], pacing="manual")
        s.run_to_end()
        st, task = s._live.sim.state, s._live.sim.task
        for m in st.messages:
            assert "@" not in m.text, m.text
            assert realism.participant_mention_violation(m.text, st, task) is None, m.text
            assert not PING.search(m.text), m.text
        s.close()
