"""0.5.1: one deterministic test group per defect of the latest real-Haiku review (seed 1, 4 students,
pt-BR, sci-enzyme-01, on 0.5.0): (1) an 8-minute dead silence, (2) too easy / too fast and no conflict,
(3) information dumps and "como eu disse" about one's own point, (4) bookish first content line,
(5) only one student answered the closing proposal; plus the batch command. Offline only (template and
mock backends), several seeds, groups of 2-10."""
from __future__ import annotations

import json
import statistics
from collections import Counter

import pytest

from collabsim import Simulator
from collabsim.llm import LLMClient, MockLLM
from collabsim.llm.mock import default_handlers
from collabsim.llm.prompts import realizer_request
from collabsim.models import load_task
from collabsim.sim import Simulation, conflict
from collabsim.sim.config import CONFLICT_PRESETS, PACE_PRESETS, SimConfig
from collabsim.sim.content import SELF_REF_RE
from collabsim.sim.factory import build_session
from collabsim.sim.planner import ContentPlanner, is_content_free
from collabsim.sim.realizer import fact_gist
from tests.support.pilot_stats import load_targets

PT_TASKS = [load_task(t, language="pt-BR") for t in ("sci-enzyme-01", "math-ratio-01")]
SIZES = (2, 3, 4, 6, 10)


def _sessions(backend="template", seeds=range(1, 7), sizes=SIZES, cfg=None):
    out = []
    for n in sizes:
        for seed in seeds:
            task = PT_TASKS[seed % 2]
            if backend == "template":
                sim = Simulation(task, seed, cfg=cfg, n_students=n)
            else:
                sim = build_session(task, seed, realizer="mock", n_students=n, cfg=cfg)
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


def _held(sim, sid: str) -> set[str]:
    s = next(x for x in sim.initial_students if x["id"] == sid)
    return {m["id"] for m in s["misconceptions"]}


# ================================================================== (1) dead silence
def test_1_silence_cap_is_calibrated_to_the_pilot_tail():
    """The cap sits between the pilot's 95th and 99th gap percentiles (aggregates in pilot_targets.json)."""
    tg = load_targets()["pooled"]
    cap = SimConfig().max_silence_s
    assert tg["gap_p95_s"] <= cap <= tg["gap_p99_s"], (tg["gap_p95_s"], cap, tg["gap_p99_s"])


def test_1_max_gap_never_exceeds_the_cap(runs, mock_runs):
    cap = SimConfig().max_silence_s
    worst = 0.0
    for sim in runs + mock_runs:
        msgs = sim.state.messages
        gaps = [b.t - a.t for a, b in zip(msgs, msgs[1:])]
        worst = max([worst, *gaps])
        assert max(gaps) <= cap + 1e-6, (sim.state.session_id, max(gaps))
    print(f"\nmax gap {worst:.0f}s (cap {cap:.0f}s)")


def test_1_owner_case_has_no_dead_silence():
    """Seed 1, n=4, pt-BR, sci-enzyme-01 (the 0.5.0 run had 8.3 minutes of silence at 4.3-12.6 min)."""
    for backend in ("template", "mock"):
        sim = build_session(PT_TASKS[0], 1, realizer=backend, n_students=4)
        sim.run()
        msgs = sim.state.messages
        assert max(b.t - a.t for a, b in zip(msgs, msgs[1:])) <= SimConfig().max_silence_s + 1e-6


def test_1_forced_turns_say_something(runs):
    """A turn forced by the cap is never skipped, and it is content, the activity question or a nudge
    (never a filler)."""
    n_forced = 0
    for sim in runs:
        for r in sim.state.slippage_log:
            if r.get("type") == "silence_cap":
                n_forced += 1
                m = sim.state.messages[r["msg"]] if r["msg"] < len(sim.state.messages) else None
                assert m is not None and m.speaker == r["speaker"]
                assert not is_content_free(m.plan) or m.plan.opener, (sim.state.session_id, m.text)
    assert n_forced > 0


def test_1_cap_can_be_turned_off():
    sim = Simulation(PT_TASKS[0], 1, cfg=SimConfig(max_silence_s=None), n_students=4)
    sim.run()
    assert not any(r.get("type") == "silence_cap" for r in sim.state.slippage_log)


# ================================================================== (2) conflict and pace
def test_2_presets_fill_only_unset_fields():
    c = SimConfig()
    assert (c.pace, c.conflict) == ("pilot", "normal")
    assert c.target_minutes == PACE_PRESETS["pilot"]["target_minutes"]
    assert c.slated_misconceptions == CONFLICT_PRESETS["normal"]["slated_misconceptions"]
    b = SimConfig(pace="brisk", conflict="high", session_minutes=30.0)
    assert b.target_minutes == PACE_PRESETS["brisk"]["target_minutes"] and b.session_minutes == 30.0
    assert b.conflict_resolution_required and b.max_pushbacks == 2
    with pytest.raises(ValueError):
        SimConfig(pace="slow")
    with pytest.raises(ValueError):
        SimConfig(conflict="extreme")


def test_2_default_conflict_voices_and_discusses_misconceptions(runs, mock_runs):
    """Default ("normal"): at least one misconception voiced by a student who holds it and challenged by
    someone else, typically two."""
    voiced = []
    for sim in runs + mock_runs:
        st = sim.state
        v, d = conflict.voiced_and_discussed(st)
        voiced.append(v)
        assert v >= 1 and d >= 1, (st.session_id, v, d, st.slates)
        for m in _stu(sim):
            if m.plan.claim == "misconception":
                assert m.plan.misconception_id in _held(sim, m.speaker), (st.session_id, m.speaker, m.text)
    assert statistics.mean(voiced) >= 1.6, statistics.mean(voiced)
    print(f"\nvoiced per session: mean {statistics.mean(voiced):.2f}, min {min(voiced)}")


def test_2_slates_belong_to_holders_on_distinct_steps(runs):
    for sim in runs:
        sl = sim.state.slates
        assert 1 <= len(sl) <= 2
        assert len({x["step"] for x in sl}) == len(sl)
        for x in sl:
            assert x["m"] in _held(sim, x["student"])


def test_2_low_and_high_conflict():
    lows, highs = [], []
    for seed in range(1, 7):
        for level, acc in (("low", lows), ("high", highs)):
            sim = Simulation(PT_TASKS[seed % 2], seed, cfg=SimConfig(conflict=level), n_students=4)
            sim.run()
            if level == "low":
                assert sim.state.slates == []
            else:
                held = sum(len(s["misconceptions"]) for s in sim.initial_students)
                assert len(sim.state.slates) == held and all(len(s["misconceptions"]) >= 1 for s in sim.initial_students)
                # persistent beliefs: a holder gives in only after both push-backs and two explicit challenges
                for f in sim.state.flip_log:
                    if f.get("cond") == "concede":
                        before = [m for m in sim.state.messages if m.id < f["msg"]]
                        own = [m for m in before if m.speaker == f["student"] and m.plan.claim == "misconception"
                               and m.plan.misconception_id == f["m"]]
                        chal = [m for m in before if m.speaker != f["student"]
                                and m.plan.addresses_misconception == f["m"]]
                        assert len(own) >= 3 and len(chal) >= 2, (sim.state.session_id, f)
            acc.append(conflict.voiced_and_discussed(sim.state)[0])
    assert statistics.mean(highs) > statistics.mean(lows), (lows, highs)


def test_2_high_conflict_may_stall():
    """conflict="high": fewer activities get done in the same time than at the default level."""
    done = {"normal": 0, "high": 0}
    for seed in range(1, 9):
        for level in done:
            sim = Simulation(PT_TASKS[seed % 2], seed, cfg=SimConfig(conflict=level), n_students=4)
            sim.run()
            done[level] += len(sim.state.steps_solved)
    assert done["high"] < done["normal"], done


def test_2_steps_stay_open_for_the_pace_dwell(runs):
    dwell = SimConfig().step_dwell_s(3)
    assert dwell >= 6 * 60
    for sim in runs:
        prev = 0.0
        for s in sim.state.steps_solved:
            assert s["t"] - prev >= dwell - 1e-6, (sim.state.session_id, s)
            prev = s["t"]


def test_2_default_pace_reproduces_pilot_length_and_brisk_is_short():
    """Default pace: the calibration configuration (n 3/4) lasts ~25-40 min with 12-19 messages per
    student (median); brisk: ~12-22 min."""
    def run(pace):
        mins, per = [], []
        for seed in range(12):
            task = PT_TASKS[seed % 2]
            n = (3, 4)[(seed // 2) % 2]
            sim = Simulation(task, seed, cfg=SimConfig(pace=pace), n_students=n)
            sim.run()
            mins.append(sim.state.t / 60.0)
            c = Counter(m.speaker for m in _stu(sim))
            per += [c.get(s, 0) for s in sim.state.roster]
        return statistics.median(mins), statistics.median(per)
    m, p = run("pilot")
    assert 25.0 <= m <= 40.0 and 12 <= p <= 19, (m, p)
    mb, pb = run("brisk")
    assert 12.0 <= mb <= 22.0 and mb < m, mb
    print(f"\npilot pace: {m:.1f} min, {p} msgs/student; brisk: {mb:.1f} min, {pb} msgs/student")


def test_2_session_config_carries_pace_and_conflict():
    sim = Simulator(backend="template")
    s = sim.new_session(n_students=3, seed=2, pacing="manual", pace="brisk", conflict="low")
    cfg = s._live.sim.cfg
    assert (cfg.pace, cfg.conflict, cfg.session_minutes, cfg.target_minutes) == ("brisk", "low", 25.0, 14.0)
    s2 = sim.new_session(n_students=3, seed=2, pacing="manual", max_minutes=30, target_minutes=20)
    assert (s2._live.sim.cfg.session_minutes, s2._live.sim.cfg.target_minutes) == (30.0, 20.0)
    o = sim.options()
    assert o["pace"] == ["pilot", "brisk"] and o["conflict"] == ["low", "normal", "high"]
    with pytest.raises(ValueError):
        sim.new_session(pace="slow")
    sim.close()


# ================================================================== (3) information dumps
def test_3_one_new_statement_per_student_per_window(runs, mock_runs):
    cfg = SimConfig()
    for sim in runs + mock_runs:
        pl = sim.planner
        last: dict[str, object] = {}
        stu = _stu(sim)
        for i, m in enumerate(stu):
            if not ContentPlanner.announces(m.plan):
                continue
            prev = last.get(m.speaker)
            last[m.speaker] = m
            if prev is None:
                continue
            asked = m.plan.anchor_turn is not None and sim.state.messages[m.plan.anchor_turn].speaker != m.speaker
            if asked or m.plan.addresses_misconception or m.plan.replies_to_external:
                continue
            between = [x for x in stu if prev.id < x.id < m.id]
            assert m.t - prev.t >= cfg.fact_window_s or len(between) >= cfg.fact_window_turns, \
                (sim.state.session_id, prev.text, m.text)


def test_3_no_consecutive_distinct_statements_by_one_student(runs, mock_runs):
    for sim in runs + mock_runs:
        stu = _stu(sim)
        for a, b in zip(stu, stu[1:]):
            if a.speaker == b.speaker and ContentPlanner.announces(a.plan) and ContentPlanner.announces(b.plan):
                direct = b.plan.anchor_turn is not None and sim.state.messages[b.plan.anchor_turn].speaker != b.speaker
                assert direct or b.plan.addresses_misconception, (sim.state.session_id, a.text, b.text)


def test_3_content_runs_capped(runs, mock_runs):
    """A same-speaker run carries at most 2 content turns (openers and afterthoughts aside)."""
    for sim in runs + mock_runs:
        run_who, n = None, 0
        for m in _stu(sim):
            content = not is_content_free(m.plan) and not m.plan.opener and not m.plan.tag and m.plan.act != "greeting" \
                and not m.plan.split_tail and not m.plan.after_hedge  # (0.5.2: the second half of the same line;
            #                                                           0.5.3: a short hedge after it, "acho")
            if m.speaker != run_who:
                run_who, n = m.speaker, 0
            n += content
            assert n <= 2 or m.plan.closing or m.plan.nudge, (sim.state.session_id, m.id, m.text)


def test_3_like_i_said_only_after_a_dispute(runs, mock_runs):
    for sim in runs + mock_runs:
        for m in _stu(sim):
            if SELF_REF_RE.search(m.text) or __import__("collabsim.lang.pt_br", fromlist=["x"]).SELF_REF_RE.search(m.text):
                assert m.plan.restate, (sim.state.session_id, m.text)


def test_3_closing_proposal_is_not_a_fourth_message_in_a_row(runs):
    for sim in runs:
        st = sim.state
        prop = st.messages[st.closing_msgs[0]]
        run = 0
        for m in reversed(_stu(sim)[:[x.id for x in _stu(sim)].index(prop.id)]):
            if m.speaker != prop.speaker:
                break
            run += 1
        assert run < 3, (st.session_id, run)


# ================================================================== (4) bookish lines
def test_4_content_lines_in_student_register(runs, mock_runs):
    cfg = SimConfig()
    lens = []
    for sim in runs + mock_runs:
        for m in _stu(sim):
            p = m.plan
            if p.claim in ("fact", "correct", "misconception") and not p.closing and not p.draft:
                lens.append(len(m.text.split()))
                assert len(m.text.split()) <= cfg.content_max_words, (sim.state.session_id, m.text)
    assert statistics.median(lens) <= 14


def test_4_realizer_gets_a_short_gist_not_the_material_sentence():
    task = PT_TASKS[0]
    for f in task.facts:
        g = fact_gist(task, f.id)
        assert len(g.split()) < len(f.text.split()) and "card" not in g, (f.id, g)
    sim = Simulation(task, 1, n_students=4)
    from collabsim.models import ContentPlan
    s = sim.state.students[0]
    fid = s.known_facts[0]
    req = realizer_request("claude-haiku-5-5", task, sim.state, s,
                           ContentPlan(act="on_task_claim", kc=task.fact(fid).kc, claim="fact", fact_ids=[fid],
                                       is_correct=True, target_len=8), sim.names)
    say = next(l for l in req.suffix.splitlines() if l.startswith("DIGA ESTE"))
    assert fact_gist(task, fid) in say and task.fact(fid).text not in say
    assert "nunca copie" in say


def test_4_a_long_bookish_draft_is_redone_short():
    """A model that answers content turns with a textbook-length sentence: regenerated once, else the
    template wording; no final content line is longer than the cap."""
    h = default_handlers()
    base = h["realize"]

    def long(req):
        p = req.meta["plan"]
        out = base(req)
        if p["claim"] in ("fact", "correct") and "CURTO" not in req.suffix and "SHORT" not in req.suffix:
            return out + " " + " ".join(["porque segundo o material isso acontece de forma bastante significativa"] * 3)
        return out
    h["realize"] = long
    sim = build_session(PT_TASKS[0], 2, realizer="mock", client=LLMClient(MockLLM(h)), n_students=4)
    sim.run()
    cap = SimConfig().content_max_words
    for m in _stu(sim):
        if m.plan.claim in ("fact", "correct", "misconception") and not m.plan.closing and not m.plan.draft:
            assert len(m.text.split()) <= cap, m.text
    recs = [r for r in sim.realizer.log if r.get("too_long")]
    assert recs and all(x["resolution"] in ("regenerated", "template") for r in recs for x in r["too_long"])


# ================================================================== (5) closing engagement
def test_5_closing_gets_answers_from_the_group(runs, mock_runs):
    for sim in runs + mock_runs:
        st = sim.state
        assert st.closing == "done", st.session_id
        prop = st.messages[st.closing_msgs[0]]
        responders = {st.messages[i].speaker for i in st.closing_msgs[1:]}
        assert prop.speaker not in responders
        need = 1 if len(st.students) < 3 else 2
        assert len(responders) >= need, (st.session_id, len(st.students), responders)
        # the chat ends after the responses (nothing but responses after the proposal)
        assert all(m.id in st.closing_msgs for m in _stu(sim) if m.id > prop.id), st.session_id


def test_5_owner_case_three_answer_the_proposal():
    sim = Simulation(PT_TASKS[0], 1, n_students=4)
    sim.run()
    st = sim.state
    assert len({st.messages[i].speaker for i in st.closing_msgs[1:]}) >= 2


# ================================================================== batch command
def test_batch_command_template(tmp_path, monkeypatch, capsys):
    from collabsim.__main__ import main
    monkeypatch.chdir(tmp_path)
    rc = main(["batch", "--backend", "template", "--seeds", "1,2", "--n", "2,3", "--out", str(tmp_path / "t"), "--yes"])
    assert rc == 0
    out = list((tmp_path / "t").iterdir())
    assert len(out) == 1
    files = sorted(p.name for p in out[0].iterdir())
    assert files == sorted(["2-1.md", "2-1.json", "2-2.md", "2-2.json", "3-1.md", "3-1.json", "3-2.md", "3-2.json",
                            "summary.md"])
    summ = (out[0] / "summary.md").read_text()
    for col in ("messages", "minutes", "msgs_per_student", "max_gap_s", "voiced", "discussed", "idk_share",
                "offtask_share", "ended", "calls", "usd", "**totals**"):
        assert col in summ
    rec = json.loads((out[0] / "3-2.json").read_text())
    assert rec["seed"] == 2 and len(rec["roster"]) == 3


def test_batch_command_asks_and_caps(tmp_path, monkeypatch, capsys):
    from collabsim.__main__ import main
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def eof(_prompt=""):
        raise EOFError
    monkeypatch.setattr("builtins.input", eof)
    rc = main(["batch", "--seeds", "1,2,3,4", "--n", "2,4,6", "--out", str(tmp_path / "t")])
    out = capsys.readouterr().out
    assert rc == 3 and "estimated cost: ~US$" in out and "12 sessions" in out
    assert not (tmp_path / "t").exists()            # nothing ran, no client was created
    rc = main(["batch", "--backend", "template", "--seeds", "1,2,3,4,5", "--n", "2,4,6", "--yes"])
    assert rc == 2                                   # 15 > --max-sessions 12
