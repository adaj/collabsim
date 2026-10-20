from __future__ import annotations

import pytest

from collabsim.models import ContentPlan, Message, RNG, load_task
from collabsim.models.session import SessionState
from collabsim.sim.config import SimConfig
from collabsim.sim.group import generate_group
from collabsim.sim.learning import LearningParams
from collabsim.sim.updater import StateUpdater


@pytest.fixture
def task():
    return load_task("sci-enzyme-01")


@pytest.fixture
def params():
    return LearningParams()


@pytest.fixture
def cfg():
    # rule-level unit tests: no minimum dwell per step (the pace's dwell has its own tests,
    # tests/test_defects_051.py)
    return SimConfig(min_step_dwell_frac=0.0)


@pytest.fixture
def make_state(task, cfg):
    """Deterministic 3-student state; s1 holds m_swallow_only (kc_surface)."""
    def _make(engagement: float = 1.0) -> SessionState:
        students = generate_group(task, RNG(7), cfg, n_students=3)
        for s in students:
            s.misconceptions = []
            s.affect.engagement = engagement
            s.traits.deference_peer = 1.0
            for k in s.kcs.values():
                k.p = 0.3
        from collabsim.models.student import MisconceptionState
        students[0].misconceptions.append(MisconceptionState(id="m_swallow_only", kc="kc_surface", strength=0.7))
        return SessionState(session_id="t", task_id=task.id, seed=0, students=students, n_steps=len(task.steps))
    return _make


def add_student(st: SessionState, sid: str, plan: ContentPlan, t: float | None = None) -> Message:
    m = Message(id=len(st.messages), t=t if t is not None else len(st.messages) * 10.0, speaker=sid,
                kind="student", text="x", plan=plan, addressee=plan.addressee)
    st.messages.append(m)
    return m


def add_external(st: SessionState, pid: str, plan: ContentPlan, text: str = "x", t: float | None = None,
                 name: str | None = None) -> Message:
    """An external participant's message (registered on first use)."""
    d = st.participants.setdefault(pid, {"name": name or pid.upper(), "avatar": None, "first_spoke_t": None})
    tt = t if t is not None else len(st.messages) * 10.0
    if d["first_spoke_t"] is None:
        d["first_spoke_t"] = tt
    m = Message(id=len(st.messages), t=tt, speaker=pid, kind="external", text=text, plan=plan,
                addressee=plan.addressee)
    st.messages.append(m)
    return m


@pytest.fixture
def updater(task, params, cfg):
    return StateUpdater(task, params, cfg)
