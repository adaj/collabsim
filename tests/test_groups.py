"""Groups of 2-10 students (role cycling, misconception sampling, structure)."""
import pytest

from collabsim.models import RNG, load_task
from collabsim.sim.config import SimConfig
from collabsim.sim.engine import Simulation
from collabsim.sim.group import assign_roles, generate_group

TASKS = ("sci-enzyme-01", "math-ratio-01")


def service_cfg(**kw):
    base = dict(max_student_turns=200, lurk_cap_minutes=10, unspoken_boost_per_min=1.0, end_on_complete=False)
    base.update(kw)
    return SimConfig(**base)


def test_roles_cycle():
    t = load_task("sci-enzyme-01")
    assert [r for r, _ in assign_roles(t, 4)] == ["A", "B", "C", "A"]      # original 4th = A rule
    assert [r for r, _ in assign_roles(t, 6)] == ["A", "B", "C"] * 2
    r10 = [r for r, _ in assign_roles(t, 10)]
    assert r10.count("A") == 4 and r10.count("B") == 3 and r10.count("C") == 3
    two = assign_roles(t, 2)   # n < roles: the unassigned card is dealt out, every fact stays in the group
    assert {f for _, fs in two for f in fs} == {f.id for f in t.facts}


@pytest.mark.parametrize("task_id", TASKS)
@pytest.mark.parametrize("n", [2, 3, 5, 6, 8, 10])
def test_misconception_sampling_rules_any_n(task_id, n):
    t = load_task(task_id)
    for seed in range(150):
        g = generate_group(t, RNG(seed), SimConfig(), n_students=n)
        assert len({s.name for s in g}) == n
        for s in g:
            own = t.refuted_by(s.jigsaw_facts)
            peers = t.refuted_by([f for o in g if o.id != s.id for f in o.jigsaw_facts])
            for m in s.misconceptions:
                assert m.id not in own, (seed, s.id, m.id)          # never refuted by own card
                if n > 2:
                    assert m.id in peers, (seed, s.id, m.id)       # refutable by another member's card
        if n > 2:
            assert any(s.misconceptions for s in g)


def test_group_size_bounds():
    t = load_task("sci-enzyme-01")
    for bad in (1, 11):
        with pytest.raises(ValueError):
            generate_group(t, RNG(0), SimConfig(), n_students=bad)


@pytest.mark.parametrize("n", [2, 6, 10])
@pytest.mark.parametrize("task_id", TASKS)
def test_structural_sanity(task_id, n):
    """The pilot calibration covers 2-4 students; >4 is extrapolation, so only structure is checked."""
    for seed in range(6):
        sim = Simulation(task_id, seed, cfg=service_cfg(), n_students=n)
        sim.run()
        st = sim.state
        spk = [m.speaker for m in st.messages if m.kind == "student"]
        assert spk and st.done
        assert set(spk) == {s.id for s in st.students}, (seed, "everyone speaks")
        run, longest = 1, 1
        for a, b in zip(spk, spk[1:]):
            run = run + 1 if a == b else 1
            longest = max(longest, run)
        assert longest <= 7, (seed, longest)
        assert st.student_turns <= 200
