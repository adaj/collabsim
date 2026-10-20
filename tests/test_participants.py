"""External participants (people or bots) in a session: roster, say(), integration through the same
state rules as a peer, replies when addressed by name, and no special-casing of anyone's identity."""
from __future__ import annotations

import pytest

from collabsim import ParticipantSpec, Simulator
from collabsim.models import ContentPlan
from collabsim.sim import Simulation
from collabsim.sim import realism


def manual(backend="template", **kw):
    sim = Simulator(backend=backend)
    base = dict(task="sci-enzyme-01", language="pt-BR", n_students=4, seed=3, pacing="manual")
    base.update(kw)
    return sim.new_session(**base)


# ------------------------------------------------------------------ roster and say()
def test_roster_ids_avatars_and_renamed_students():
    for seed in range(20):
        s = manual(seed=seed, participants=["Ana", ParticipantSpec(name="Bot", avatar="bot.png"), {"name": "Júlia"}])
        ext = s.participants
        assert [p["id"] for p in ext] == ["e1", "e2", "e3"] and [p["avatar"] for p in ext] == [None, "bot.png", None]
        names = [realism.fold(r["name"]) for r in s.roster]
        assert len(names) == len(set(names))                  # a student who would clash was renamed
        s.close()


def test_say_by_name_id_or_none_and_auto_join():
    s = manual(participants=["Ana"])
    assert s.say("oi gente").speaker == "e1"                   # the only participant
    assert s.say("tudo bem?", sender="ANA").speaker == "e1"    # folded name
    assert s.say("bora", sender="e1").speaker == "e1"          # id
    ev = s.say("oi, cheguei", sender="Bruno")                  # undeclared: joins the chat
    assert ev.speaker == "e2" and s.participants[-1]["name"] == "Bruno" and s.participants[-1]["spoken"]
    with pytest.raises(ValueError):
        s.say("e agora?")                                      # two participants: sender required
    with pytest.raises(ValueError):
        s.say("oi", sender=s.roster[0]["name"])                # a simulated student's name
    with pytest.raises(ValueError):
        s.say("   ", sender="Ana")
    s.close()
    z = manual()
    with pytest.raises(ValueError):
        z.say("oi")                                            # nobody declared: sender required
    assert z.say("oi", sender="Carla").speaker == "e1"
    z.close()


def test_add_participant_mid_session_and_limits():
    s = manual()
    for _ in range(5):
        s.step()
    p = s.add_participant("Bea", avatar="b.png")
    assert p == {**p, "id": "e1", "name": "Bea", "avatar": "b.png", "kind": "external", "spoken": False}
    with pytest.raises(ValueError):
        s.add_participant("bea")
    for i in range(7):
        s.add_participant(f"P{i}")
    with pytest.raises(ValueError):
        s.add_participant("P8")
    s.close()


# ------------------------------------------------------------------ the same rules as a peer
def _engine(seed=4, n=3):
    sim = Simulation("sci-enzyme-01", seed, n_students=n)
    sim.add_participant("e1", "Ana")
    sim.run(max_events=6)
    return sim


def test_external_fact_is_learned_by_readers():
    sim = _engine()
    st = sim.state
    for s in st.students:
        s.known_facts = [f for f in s.known_facts if f != "f_reuse"]
        s.affect.engagement = 1.0
    msg = sim.inject_external("e1", "x", ContentPlan(act="on_task_claim", kc="kc_enzyme", claim="fact",
                                                     fact_ids=["f_reuse"], is_correct=True))
    assert msg.kind == "external" and msg.labels["readers"]
    assert all("f_reuse" in st.student(r).known_facts for r in msg.labels["readers"])


def test_external_misconception_is_challenged_and_addressed_to_them():
    hits = 0
    for seed in range(20):
        sim = _engine(seed=seed, n=4)
        st = sim.state
        sim.inject_external("e1", "acho que mastigar é so pra comida descer mais facil",
                            ContentPlan(act="on_task_claim", kc="kc_surface", claim="misconception",
                                        misconception_id="m_swallow_only", is_correct=False))
        later = sim.run_student_turns(4)
        hits += any(m.plan.addresses_misconception == "m_swallow_only" and m.addressee == "e1" for m in later)
    assert hits >= 5


def test_external_content_question_is_answered():
    """A participant's content question is a pending question like a peer's: a student who holds the
    fact (or can state the KC) answers it, anchored to it, usually within a few turns."""
    answered = by_holder = 0
    for seed in range(30):
        sim = _engine(seed=seed, n=4)
        q = sim.inject_external("e1", "oq tem na saliva afinal?", ContentPlan(act="question", kc="kc_enzyme",
                                                                         asked_fact="f_amylase"))
        later = sim.run_student_turns(6)
        ans = [m for m in later if m.plan.anchor_turn == q.id and m.plan.claim in ("fact", "correct")]
        answered += bool(ans)
        by_holder += any("f_amylase" in m.plan.fact_ids for m in ans)
    assert answered >= 12 and by_holder >= 8, (answered, by_holder)


def test_one_external_addressing_another_boosts_no_student():
    sim = _engine()
    sim.add_participant("e2", "Bot")
    st = sim.state
    sim.inject_external("e2", "oi", ContentPlan(act="greeting"))
    before = dict(st.addressed_at)
    sim.inject_external("e1", "Bot, o que você acha?", ContentPlan(act="question", addressee="e2"))
    assert st.addressed_at == before and not any(k in st.open_address for k in st.roster)


def test_students_address_only_participants_who_have_written():
    for seed in range(30):
        s = manual(seed=seed, participants=["Maria", "Joana"])
        for _ in range(8):
            s.step()
        s.say("gente, a saliva tem amilase", sender="Maria")
        s.run_to_end()
        st = s._live.sim.state
        spoke_t = {p: d["first_spoke_t"] for p, d in st.participants.items()}
        for m in st.messages:
            if m.kind == "student" and m.addressee in spoke_t:
                assert spoke_t[m.addressee] is not None and spoke_t[m.addressee] <= m.t, (seed, m.text)
        s.close()


# ------------------------------------------------------------------ live integration
def test_rollback_of_a_pending_turn_and_manual_integration():
    s = manual(participants=["Ana"], pacing="realtime")
    live = s._live
    live.start()
    for _ in range(20):
        live.pump()
        if live.pending is not None:
            break
    assert live.pending is not None
    n = len(live.sim.state.messages)
    live.submit("e1", "oi gente")
    live.pump()
    assert live.discarded_turns == 1 and live.sim.state.messages[n - 1].speaker == "e1"
    s.close()
    m = manual(participants=["Ana"])
    m.step()
    m.say("a amilase quebra o amido em açucar", sender="Ana")
    assert m._live.interpretations and m._live.sim.state.messages[-1].kind == "external"   # before say() returned
    m.close()


def test_addressed_student_replies_at_the_calibrated_rate():
    """Writing a student's name addresses them: they reply within 2 student turns at about the pilot
    rate (9/13 within 2 messages; p_reply_when_addressed = 8/13 plus their own turns)."""
    n = replied = 0
    for seed in range(60):
        s = manual(seed=seed, participants=["Bot"])
        for _ in range(6):
            s.step()
        target = s.roster[seed % 4]
        ev = s.say(f"{target['name']}, o que você acha disso?", sender="Bot")
        if ev is None:
            continue
        nxt = []
        while len(nxt) < 2 and s.status != "done":
            nxt += [e for e in s.step() if e.kind == "student"]
        n += 1
        replied += any(e.speaker == target["id"] for e in nxt[:2])
        s.close()
    rate = replied / n
    print(f"\naddressee replied within 2 student turns: {rate:.2f} (n={n}; pilot 9/13 = 0.69)")
    assert 0.5 <= rate <= 0.85


def test_no_identity_special_casing():
    """The same seed and text, sent once by a participant named "Nova" and once by "Maria", give the
    same plan, the same hidden state and the same following turns (apart from the name)."""
    def run(name):
        s = manual(seed=11, participants=[name])
        for _ in range(6):
            s.step()
        s.say("acho que mastigar é so pra comida descer mais facil")
        s.say(f"{s.roster[1]['name']}, o que você acha?")
        s.run_to_end()
        st = s._live.sim.state
        out = ([m.plan.model_dump() for m in st.messages], st.timeline,
               [realism.fold(m.text).replace(realism.fold(name), "<p>") for m in st.messages],
               [r["plan"] for r in s._live.interpretations])
        s.close()
        return out
    a, b = run("Nova"), run("Maria")
    assert a[0] == b[0] and a[1] == b[1] and a[3] == b[3]
    assert a[2] == b[2]
