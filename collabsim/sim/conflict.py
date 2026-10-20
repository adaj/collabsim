"""Conflict control (0.5.1): which held misconceptions surface in a session.

The group generator (sim/group.py) gives students 0-2 misconceptions from the task's priors. A
belief, however, only shows in the chat if its holder happens to make a claim on that step, so a
fast group could finish without any disagreement. `conflict` (SimConfig / SessionConfig) sets how
many held beliefs are SLATED to surface:

  low     no slates (0.5.0 behaviour)
  normal  up to 2 slates on different steps (at least 1 when the group holds any misconception)
  high    every held belief is slated; beliefs are stronger and must be given up before the step
          is solved (the group may stall)

A slated belief is voiced by the student who HOLDS it (never assigned to anyone else), once the
step is underway; the step cannot be solved before it was voiced and challenged (updater rule 6).
Until it is voiced, generic talk on the topic does not dissolve it (rule 4 skips it).

Top-ups and slates use their own RNG stream, so the 0.5.0 group draws are unchanged.
"""
from __future__ import annotations

from ..models.rng import RNG
from ..models.session import SessionState
from ..models.student import MisconceptionState, StudentState
from ..models.task import Task
from .config import SimConfig


def step_of(task: Task, kc: str, trigger: str | None = None) -> str | None:
    """The step a misconception belongs to: its trigger step, else the first step whose gold KCs hold it."""
    ids = [s.id for s in task.steps]
    if trigger in ids:
        return trigger
    for g in task.gold:
        if kc in g.kcs:
            return g.step
    return None


def top_up(task: Task, students: list[StudentState], cfg: SimConfig, rng: RNG) -> None:
    """normal: the group holds misconceptions on at least min(2, available) different steps;
    high: every student holds at least one, and every belief is strong."""
    from .group import _assign, _eligible
    if cfg.conflict == "low" or not task.misconceptions:
        return
    relaxed = len(students) == 2
    if cfg.conflict == "high":
        for s in students:
            if not s.misconceptions:
                _assign(task, s, 1, rng, students, relaxed=relaxed and not _eligible(task, s, students))
        for s in students:
            for m in s.misconceptions:
                m.strength = max(m.strength, rng.uniform(0.75, 0.95))
        return
    # normal: the beliefs that are slated to surface should belong to students who actually talk -- the
    # more talkative half of the group (those present from the start first) covers `want` different KCs
    want = min(cfg.slated_misconceptions, len({m.kc for m in task.misconceptions}))
    talkers = sorted(students, key=lambda s: -s.traits.talkativeness)
    top = talkers[:max(1, (len(students) + 1) // 2)]
    for _ in range(4):
        covered = {m.kc for s in top for m in s.misconceptions}
        if len(covered) >= want:
            return
        done = False
        for s in sorted(top, key=lambda s: (len(s.misconceptions), talkers.index(s))):
            if len(s.misconceptions) >= 2:
                continue
            pool = [m for m in _eligible(task, s, students) if m.kc not in covered]
            if not pool and relaxed:
                pool = [m for m in _eligible(task, s, students, relaxed=True) if m.kc not in covered]
            if not pool:
                continue
            m = rng.choice(pool)
            s.misconceptions.append(MisconceptionState(id=m.id, kc=m.kc, strength=rng.uniform(0.5, 0.9)))
            s.kcs[m.kc].p = min(s.kcs[m.kc].p, 0.45)
            done = True
            break
        if not done:
            return


def make_slates(task: Task, students: list[StudentState], cfg: SimConfig, rng: RNG) -> list[dict]:
    """[{"student", "m", "step"}] in step order."""
    k = cfg.slated_misconceptions
    if k <= 0:
        return []
    order = {s.id: i for i, s in enumerate(task.steps)}
    cands = []
    for s in students:
        for m in s.misconceptions:
            spec = task.misconception(m.id)
            step = step_of(task, m.kc, spec.trigger)
            if step is not None:
                # holders who talk more are slated first (on the first activity, only those who are there
                # from the start)
                first = step == task.steps[0].id
                cands.append({"student": s.id, "m": m.id, "step": step,
                              "_key": (first and s.traits.lurk_until_min > 0,
                                       -s.traits.talkativeness * rng.uniform(0.8, 1.25))})
    if k >= len(cands):
        out = cands
    else:
        out, steps, who = [], set(), set()
        for c in sorted(cands, key=lambda c: (c["student"] in who, c["_key"])):
            if len(out) >= k:
                break
            if c["step"] in steps:
                continue
            out.append(c)
            steps.add(c["step"])
            who.add(c["student"])
    for c in out:
        c.pop("_key", None)
    for c in cands:
        c.pop("_key", None)
    return sorted(out, key=lambda c: order.get(c["step"], 0))


def pending(st: SessionState, step_id: str) -> list[dict]:
    """Slates of this step that have not surfaced yet while their holder still holds the belief."""
    out = []
    for sl in st.slates:
        if sl["step"] != step_id or sl.get("voiced") is not None:
            continue
        s = st.student(sl["student"])
        m = s.misconception(sl["m"])
        if m is not None and m.active:
            out.append(sl)
    return out


def undiscussed(st: SessionState, step_id: str) -> list[dict]:
    """Slates of this step that were voiced but not yet challenged explicitly by someone else (a turn
    addressing the belief), while the holder still holds it."""
    out = []
    for sl in st.slates:
        if sl["step"] != step_id or sl.get("voiced") is None:
            continue
        m = st.student(sl["student"]).misconception(sl["m"])
        if m is None or not m.active:
            continue
        if not any(x.speaker != sl["student"] and x.plan is not None and x.plan.addresses_misconception == sl["m"]
                   for x in st.messages[sl["voiced"] + 1:]):
            out.append(sl)
    return out


def protected(st: SessionState) -> set[tuple[str, str]]:
    """(student, misconception) pairs not to be dissolved by generic talk before they surface."""
    return {(sl["student"], sl["m"]) for sl in st.slates if sl.get("voiced") is None}


def mark_voiced(st: SessionState, sid: str, mid: str, msg_id: int) -> None:
    for sl in st.slates:
        if sl["student"] == sid and sl["m"] == mid and sl.get("voiced") is None:
            sl["voiced"] = msg_id


def voiced_and_discussed(st: SessionState) -> tuple[int, int]:
    """(distinct (student, misconception) voiced by its holder, of which challenged afterwards by someone else)."""
    voiced: dict[tuple[str, str], int] = {}
    for m in st.messages:
        p = m.plan
        if p is not None and p.claim == "misconception" and p.misconception_id and st.is_student(m.speaker):
            voiced.setdefault((m.speaker, p.misconception_id), m.id)
    disc = 0
    for (sid, mid), first in voiced.items():
        if any(x.speaker != sid and x.plan is not None and x.plan.addresses_misconception == mid
               for x in st.messages[first + 1:]):
            disc += 1
    return len(voiced), disc
