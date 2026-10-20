"""Schema validation, serialization round-trips, snapshot/restore exact replay."""
import json

from collabsim.models import RNG, StudentState, Task, list_tasks, load_task
from collabsim.models.session import SessionState
from collabsim.sim import Simulation
from collabsim.sim.group import generate_group


def test_tasks_load_and_validate():
    assert {"sci-enzyme-01", "math-ratio-01"} <= set(list_tasks())
    for tid in ("sci-enzyme-01", "math-ratio-01"):
        t = load_task(tid)
        assert len(t.kcs) == 3 and len(t.steps) == 3
        assert all(m.flip_by and m.trigger for m in t.misconceptions)
        assert not hasattr(t, "quiz") and not hasattr(t, "transfer")
        assert "gold" not in t.public_view() and "misconceptions" not in t.public_view()


def test_legacy_task_keys_and_families_load():
    """Older task dicts (quiz / transfer keys, unknown flip families) still load: the extra keys are
    ignored and the unknown families dropped, with warnings from validate_spec."""
    from collabsim.runtime.taskspec import validate_spec
    d = load_task("sci-enzyme-01").model_dump(mode="json")
    d["quiz"] = [{"id": "q1"}]
    d["transfer"] = {"stem": "x"}
    d["misconceptions"][0]["flip_by"] = ["old_family_a", "old_family_b", "counterexample"]
    d["misconceptions"][1]["flip_by"] = ["old_family_b"]
    t = Task.model_validate(d)
    assert t.misconceptions[0].flip_by == ["counterexample"]
    assert set(t.misconceptions[1].flip_by) == {"counterexample", "peer_explanation", "evidence_from_material",
                                                "press_for_reasoning"}
    rep = validate_spec(d)
    assert rep["valid"] and any("quiz" in w for w in rep["warnings"]) and any("flip_by" in w for w in rep["warnings"])


def test_task_roundtrip():
    t = load_task("math-ratio-01")
    assert Task.model_validate_json(t.model_dump_json()) == t


def test_student_and_session_roundtrip(task):
    students = generate_group(task, RNG(3))
    for s in students:
        assert StudentState.model_validate_json(s.model_dump_json()) == s
    sim = Simulation(task, 3)
    sim.run(max_events=80)
    st2 = SessionState.model_validate_json(sim.state.model_dump_json())
    assert st2.model_dump() == sim.state.model_dump()


def test_group_generation_seeded(task):
    a = [s.model_dump() for s in generate_group(task, RNG(11))]
    b = [s.model_dump() for s in generate_group(task, RNG(11))]
    assert a == b
    for seed in range(40):
        g = generate_group(task, RNG(seed), n_students=3 + seed % 2)
        assert 3 <= len(g) <= 4
        assert sum(len(s.misconceptions) for s in g) >= 1
        assert all(len(s.misconceptions) <= 2 for s in g)
        if len(g) == 4:
            assert g[3].role == "A" and g[3].jigsaw_facts == g[0].jigsaw_facts


def test_rng_state_roundtrip():
    r = RNG(5)
    r.random()
    s = json.loads(json.dumps(r.get_state()))
    x = [r.random() for _ in range(5)]
    r.set_state(s)
    assert [r.random() for _ in range(5)] == x


def test_snapshot_restore_exact_replay():
    from collabsim.models import ContentPlan
    for seed in (42, 43):
        sim = Simulation("sci-enzyme-01", seed)
        sim.add_participant("e1", "Ana")
        sim.run(max_events=30)
        sim.inject_external("e1", "x", ContentPlan(act="on_task_claim", kc="kc_surface", claim="fact",
                                                   fact_ids=["f_surface"], is_correct=True))
        sim.run(max_events=10)
        snap = json.loads(json.dumps(sim.snapshot()))  # must survive JSON
        sim.run()
        full = sim.result()
        sim.restore(snap)
        sim.run()
        again = sim.result()
        assert full["transcript"] == again["transcript"]
        assert full["hidden_state_timeline"] == again["hidden_state_timeline"]
        assert full["participants"] == again["participants"]


def test_same_seed_same_session():
    r1 = Simulation("math-ratio-01", 9)
    r2 = Simulation("math-ratio-01", 9)
    r1.run(); r2.run()
    assert r1.result()["transcript"] == r2.result()["transcript"]
