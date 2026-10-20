"""Live-session runtime (collabsim.runtime.live via the public Simulator): real-time pacing (fake
clock), external participants, errors, caps. Most tests drive the internal LiveSession's
pump() directly (no background thread) for determinism; test_api.py covers the threaded API."""
import json
import re

import pytest

from collabsim.llm.client import LLMClient, MockLLM
from collabsim.models import ContentPlan
from collabsim import CapacityError, Simulator
from collabsim.runtime.interpret import Interpreter, build_plan, heuristic_interpretation


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(backend="mock", clock=None, backend_factory=None, **cfg):
    """(Simulator, LiveSession, FakeClock). Shorthand kwargs: pacing={"mode", "speed"} (mode "sync" ->
    "manual"), human={"join", "name"} (one external participant, id e1), task_id."""
    clock = clock or FakeClock()
    sim = Simulator(backend=backend, clock=clock, wall_ms=lambda: int(clock.t * 1000),
                    backend_factory=backend_factory)
    base = dict(task="sci-enzyme-01", n_students=3, seed=4)
    pacing = cfg.pop("pacing", None) or {}
    if pacing:
        base["pacing"] = {"sync": "manual"}.get(pacing.get("mode", "realtime"), pacing.get("mode", "realtime"))
        if "speed" in pacing:
            base["speed"] = pacing["speed"]
    human = cfg.pop("human", None)
    if human and human.get("join"):
        base["participants"] = [human.get("name") or "Visitante"]
    if "task_id" in cfg:
        base["task"] = cfg.pop("task_id")
    base.update(cfg)
    sess = sim.new_session(**base)
    sess._live.start()     # start the realtime clock without the background thread
    return sim, sess._live, clock


def student_events(s):
    return [e for e in s.events if e["kind"] == "student"]


def test_realtime_release_on_time_with_speed_and_typing():
    for speed in (1.0, 2.0, 5.0):
        _, s, clk = make(pacing={"mode": "realtime", "speed": speed})
        s.pump()
        assert s.pending is not None and not s.events          # pre-generated, held until its time
        t_rel = s.pending["t"]
        clk.t = t_rel / speed - 0.01
        s.pump()
        assert not student_events(s)
        assert s.typing() == [s.pending["speaker"]]            # about to speak
        clk.t = t_rel / speed + 0.001
        s.events_after(0)                                      # a poll releases what is due, without the runner
        ev = student_events(s)
        assert len(ev) == 1 and ev[0]["t_sim"] == round(t_rel, 2) and ev[0]["ts_ms"] == int(clk.t * 1000)
        # the next turn is pre-generated right after the release
        s.pump()
        assert s.pending is not None and s.pending["t"] >= t_rel


def test_pause_resume_freeze_sim_clock():
    _, s, clk = make()
    for _ in range(6):
        s.pump()
        clk.t += 40
        s.pump()
    n = len(s.events)
    s.control("pause")
    frozen = s.sim_now()
    clk.t += 10_000
    for _ in range(5):
        s.pump()
    assert len(s.events) == n and s.sim_now() == frozen and s.events_after(0)["status"] == "paused"
    s.control("resume")
    assert abs(s.sim_now() - frozen) < 1e-9
    s.control("speed", 4.0)
    clk.t += 60
    for _ in range(10):
        s.pump()
    assert len(s.events) > n and s.events[-1]["t_sim"] <= frozen + 4 * 60 + 1e-6


def test_sync_mode_runs_to_completion():
    _, s, _ = make(pacing={"mode": "sync"}, max_minutes=10)
    s.run_sync()
    assert s.public_status == "done" and s.ended_reason in ("complete", "time_cap", "turn_cap")
    seqs = [e["seq"] for e in s.events]
    assert seqs == list(range(1, len(seqs) + 1))
    ts = [e["t_sim"] for e in s.events if e["kind"] != "system"]
    assert ts == sorted(ts) and s.events[-1]["kind"] == "system" and s.events[-1]["code"] == "ended"


def test_external_message_discards_stale_pregenerated_turn():
    _, s, clk = make(human={"join": True, "name": "Ana"})
    for _ in range(4):
        s.pump()
        clk.t += 30
        s.pump()
    for _ in range(50):
        if s.pending is not None:
            break
        s.pump()
    assert s.pending is not None
    pm = s.pending["msgs"][0]
    stale = (pm.speaker, pm.text, round(pm.t, 3))     # (another student may have typed the same greeting)
    n_msgs = len(s.sim.state.messages)
    r = s.submit("e1", "oi gente")
    assert r["accepted"] and s.events[-1]["seq"] == r["seq"] and s.events[-1]["kind"] == "external"
    s.pump()
    assert s.discarded_turns == 1
    # the stale turn was rolled back: the participant's message is the next engine message
    assert s.sim.state.messages[n_msgs - 1].speaker == "e1" and s.sim.state.messages[n_msgs - 1].text == "oi gente"
    assert s.pending is None or s.pending["msgs"][0].t >= r["t_sim"]
    for _ in range(10):
        clk.t += 30
        s.pump()
    ts = [e["t_sim"] for e in s.events if e["kind"] != "system"]
    assert ts == sorted(ts)
    assert stale not in [(e["speaker"], e["text"], round(e["t_sim"], 3)) for e in s.events if e["seq"] < r["seq"]]


def test_external_misconception_is_challenged_and_students_address_the_participant():
    """A participant voicing a misconception is challenged by students (who may address them)."""
    reacted = False
    for seed in range(6):
        _, s, clk = make(seed=seed, n_students=4, human={"join": True, "name": "Ana"})
        s.pump()
        s.submit("e1", "acho que mastigar é so pra comida descer mais facil")
        s.pump()
        rec = s.interpretations[-1]
        assert rec["source"] == "llm" and rec["plan"]["claim"] == "misconception"
        assert rec["plan"]["misconception_id"] == "m_swallow_only"
        hmsg = next(m for m in s.sim.state.messages if m.speaker == "e1")
        assert hmsg.labels["expresses"] == "m_swallow_only"
        for _ in range(25):
            clk.t += 30
            s.pump()
        if any(e["kind"] == "student" and e["addressee"] == "e1" for e in s.events):
            reacted = True
            break
    assert reacted


def test_external_fact_teaches_readers_and_challenge_is_anchored():
    _, s, clk = make(n_students=3, human={"join": True, "name": "Ana"})
    sim = s.sim
    st = sim.state
    holder = next(x for x in st.students if not any(m.kc == "kc_temp" for m in x.misconceptions))
    plan = ContentPlan(act="on_task_claim", kc="kc_temp", claim="misconception", misconception_id="m_hotter_faster",
                       is_correct=False)
    sim.inject_student(holder.id, plan, "80 é mais rapido, mais quente sempre acelera")
    text = "nada a ver, acima de 60 ela desnatura e para"
    raw = heuristic_interpretation(sim.task, st, text, "e1")
    assert raw["facts"] == ["f_denature"] and raw["is_correct"] is True
    raw["challenges_misconception"] = "m_hotter_faster"
    p, notes = build_plan(sim.task, st, raw, text, "e1")
    assert p.claim == "fact" and p.fact_ids == ["f_denature"] and p.addresses_misconception == "m_hotter_faster"
    assert p.anchor_turn == len(st.messages) - 1 and p.addressee == holder.id and p.family == "evidence_from_material"
    msg = sim.inject_external("e1", text, p, t=st.t + 5)
    readers = msg.labels["readers"]
    assert all("f_denature" in st.student(r).known_facts for r in readers)
    assert msg.labels["flips"] or readers == [] or all(f["cond"] == "targeted" for f in msg.labels["flips"]
                                                      if f["student"] == holder.id)


def test_interpretation_validation_drops_unknown_ids():
    _, s, _ = make(human={"join": True})
    st, task = s.sim.state, s.sim.task
    raw = {"act": "explain", "kc": "kc_nope", "facts": ["f_amylase", "f_bogus"], "is_correct": True,
           "voices_misconception": "m_bogus", "challenges_misconception": "m_hotter_faster", "addressee": "Nobody"}
    p, notes = build_plan(task, st, raw, "a saliva tem amilase", "e1")
    assert p.fact_ids == ["f_amylase"] and p.kc == "kc_enzyme" and p.addressee is None
    assert any("f_bogus" in n for n in notes) and any("not voiced recently" in n for n in notes)
    assert p.addresses_misconception is None   # never voiced: no anchored challenge


@pytest.mark.parametrize("bad", ["not json at all", "{}", "[]"])
def test_interpretation_fallback_is_opaque_social(bad):
    client = LLMClient(MockLLM(handlers={"interpret": lambda req: bad}))
    interp = Interpreter(client)
    _, s, _ = make(human={"join": True})
    p = interp(s.sim.task, s.sim.state, "alguem sabe a resposta?", "e1")
    assert interp.log[-1]["source"] == "fallback"
    assert p.act == "question" and p.kc is None and p.claim == "none" and not p.fact_ids


def test_interpretation_exception_fallback():
    def boom(req):
        raise RuntimeError("model down")
    interp = Interpreter(LLMClient(MockLLM(handlers={"interpret": boom})))
    _, s, _ = make(human={"join": True})
    p = interp(s.sim.task, s.sim.state, "kkkk", "e1")
    assert p.act == "backchannel" and interp.log[-1]["source"] == "fallback"


def test_participant_addressing_a_student_by_name():
    """Writing a student's name addresses them: the event names them, the engine opens an address and
    boosts their hazard; the reply (if any) is marked as a reply to the participant."""
    _, s, clk = make(n_students=3, human={"join": True, "name": "Bot"})
    for _ in range(4):
        s.pump()
        clk.t += 30
        s.pump()
    name = s.sim.state.students[1].name
    r = s.submit("e1", f"{name}, você poderia explicar melhor isso?")
    assert r["accepted"] and s.events[-1]["kind"] == "external" and s.events[-1]["addressee"] == "s2"
    assert s.events[-1]["display_name"] == "Bot"
    s.pump()
    em = [m for m in s.sim.state.messages if m.kind == "external"]
    assert len(em) == 1 and em[0].addressee == "s2" and em[0].plan.addressee == "s2"
    assert s.sim.state.addressed_at["s2"] == em[0].t


def test_unknown_sender_is_rejected_by_the_live_session():
    _, s, _ = make()
    with pytest.raises(ValueError):
        s.submit("e9", "oi")


def test_billing_error_sets_error_status_and_keeps_events():
    _, s, clk = make(backend_factory=lambda: MockLLM(fail_after=12), human={"join": True, "name": "Ana"})
    for _ in range(40):
        s.pump()
        clk.t += 30
    out = s.events_after(0)
    assert out["status"] == "error" and "credit" in out["error"].lower() and s.error_kind == "billing"
    assert "saldo" in out["error"]                 # readable pt-BR message
    assert out["events"][-1]["reason"] == "billing"
    assert out["events"][-1]["kind"] == "system" and out["events"][-1]["code"] == "error"
    assert any(e["kind"] == "student" for e in out["events"])
    assert s.submit("e1", "oi")["accepted"] is False


def test_stop_and_turn_cap():
    _, s, clk = make()
    s.pump()
    s.control("stop")
    assert s.public_status == "done" and s.ended_reason == "stopped" and not s.pump()
    _, s2, _ = make(pacing={"mode": "sync"}, max_student_turns=12, end_on_complete=False)
    s2.run_sync()
    # near the cap the group wraps up (closing synthesis) instead of being cut off mid-chat
    ev = student_events(s2)
    assert s2.ended_reason == "turn_cap" and 10 <= len(ev) <= 12
    assert s2.sim.state.closing_forced and s2.sim.state.closing == "done"


def test_session_cap():
    clock = FakeClock()
    sim = Simulator(backend="template", max_sessions=2, clock=clock)
    a = sim.new_session(task="sci-enzyme-01")
    sim.new_session(task="sci-enzyme-01")
    with pytest.raises(CapacityError):
        sim.new_session(task="sci-enzyme-01")
    a.stop()                   # finished sessions do not count
    sim.new_session(task="sci-enzyme-01")


def test_idle_ttl_reaps_sessions():
    clock = FakeClock()
    sim = Simulator(backend="template", idle_ttl_s=1800, clock=clock)
    a = sim.new_session(task="sci-enzyme-01")
    b = sim.new_session(task="sci-enzyme-01")
    clock.t = 1000
    b.poll(0)                  # polling keeps a session alive
    clock.t = 1801
    assert sim.reap() == [a.id] and sim.get_session(a.id) is None and a.status == "done"
    assert a.ended_reason == "idle" and a.events()[-1].reason == "idle"
    clock.t = 2900
    assert sim.reap() == [b.id]


@pytest.mark.parametrize("n", [2, 6, 10])
def test_group_sizes_through_the_service(n):
    _, s, _ = make(backend="template", n_students=n, pacing={"mode": "sync"}, end_on_complete=False)
    s.run_sync()
    roster = s.roster()
    assert [r["avatar"] for r in roster] == [i % 6 for i in range(n)]
    assert [r["ring"] for r in roster] == [i // 6 for i in range(n)]
    spk = [e["speaker"] for e in student_events(s)]
    assert set(spk) == {r["id"] for r in roster}
    run = longest = 1
    for a, b in zip(spk, spk[1:]):
        run = run + 1 if a == b else 1
        longest = max(longest, run)
    assert longest <= 7 and len(spk) <= 200


def test_participant_name_never_collides_with_a_student():
    for seed in range(30):
        _, s, _ = make(backend="template", seed=seed, n_students=10, participants=["Ana", "Júlia"])
        names = [r["name"] for r in s.roster()]
        assert len(names) == len(set(n.lower() for n in names)) == 12 and names[-2:] == ["Ana", "Júlia"]


EN_LEFTOVER = re.compile(r"\b(lol|lmao|haha\w*|idk|yeah|guys|anyone|mine says|my card|i think|like i said|the|and|"
                         r"you|wait|maybe|dont)\b", re.I)


def test_pt_service_session_language():
    _, s, clk = make(n_students=4, pacing={"mode": "sync"}, human={"join": True, "name": "Ana"}, max_minutes=15)
    s.pump()
    s.submit("e1", "oi gente, oq vcs acham?")
    s.run_sync()
    names = {r["name"].lower() for r in s.roster()}
    for e in student_events(s):
        words = " ".join(w for w in e["text"].split() if w.strip("?,.!").lower() not in names)
        assert not EN_LEFTOVER.search(words), e["text"]
    assert s.sim.task.language == "pt-BR"


def test_en_service_session_language():
    from collabsim.sim.realizer import PT_MARKERS_RE
    _, s, _ = make(n_students=4, pacing={"mode": "sync"}, language="en", max_minutes=15)
    s.run_sync()
    assert s.sim.task.language == "en"
    for e in student_events(s):
        assert not PT_MARKERS_RE.search(e["text"]), e["text"]


def test_trait_presets_apply_after_sampling():
    _, a, _ = make(backend="template", seed=9, n_students=4)
    _, b, _ = make(backend="template", seed=9, n_students=4, traits={"preset": "chatty", "overrides": {"typo_rate": 0},
                                                                     "per_student": {"0": {"off_task": 0.5}}})
    for x, y in zip(a.sim.state.students, b.sim.state.students):
        assert x.name == y.name and x.jigsaw_facts == y.jigsaw_facts and x.misconceptions == y.misconceptions
        assert y.traits.talkativeness == pytest.approx(x.traits.talkativeness * 1.6) and y.traits.typo_rate == 0
    assert b.sim.state.students[0].traits.off_task == 0.5
    with pytest.raises(ValueError):
        make(backend="template", traits={"preset": "nope"})
    _, c, _ = make(backend="template", seed=9, n_students=4, traits="chatty")
    assert c.sim.state.students[0].traits.talkativeness == pytest.approx(a.sim.state.students[0].traits.talkativeness * 1.6)
