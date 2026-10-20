"""Seeded group generation (names, cards, KCs, misconceptions, traits)."""
from __future__ import annotations

import math

from ..models.rng import RNG
from ..models.student import Affect, KCState, MisconceptionState, StudentState, Traits
from ..models.task import Task
from .config import SimConfig

NAMES = ["Maya", "Leo", "Sofia", "Noah", "Ava", "Liam", "Zoe", "Ethan", "Nina", "Owen",
         "Ines", "Caio", "Lara", "Theo", "Bia", "Rafa"]
MIN_STUDENTS, MAX_STUDENTS = 2, 10
DEFAULT_STUDENTS = 4


def assign_roles(task: Task, n: int) -> list[tuple[str, list[str]]]:
    """(role, card facts) per student, for any group size.

    Roles cycle in the task's jigsaw order: with 3 roles, n=3 -> A B C and n=4 -> A B C A; n=6 -> A B C A B C;
    n=10 -> A x4, B x3, C x3. With fewer students than roles (n=2 on a 3-role task) the cards of the
    unassigned roles are dealt round-robin to the students, so every card fact is still in the group
    (the student keeps the first role's name; their card is the union)."""
    roles = task.roles
    out = [(roles[i % len(roles)], list(task.role_facts(roles[i % len(roles)]))) for i in range(n)]
    for j, extra in enumerate(roles[n:]):
        r, facts = out[j % n]
        out[j % n] = (r, facts + [f for f in task.role_facts(extra) if f not in facts])
    return out


def sample_traits(rng: RNG) -> Traits:
    """Trait priors (pilot-calibrated). Two retired traits are still drawn and discarded, so the same
    seed produces the same group as in 0.4.x (keeps calibration comparisons exact)."""
    talkativeness = rng.lognormal(0.0, 0.55)
    latency_scale = rng.lognormal(math.log(12.0), 1.1) / 12.0
    burstiness = rng.beta(3.3, 6.7)               # mean .33 -> realised P(same speaker) ~.37
    off_task = rng.beta(2, 9)
    lurk_until_min = 0.0 if rng.bernoulli(0.8) else rng.uniform(5, 30)
    rng.beta(3, 6)                                # retired draw (kept for seed stability)
    deference_peer = rng.beta(4, 4)
    rng.beta(1.2, 8.0)                            # retired draw (kept for seed stability)
    return Traits(
        talkativeness=talkativeness, latency_scale=latency_scale, burstiness=burstiness, off_task=off_task,
        lurk_until_min=lurk_until_min, deference_peer=deference_peer,
        interface_confusion=rng.beta(2, 12),
        verbosity=rng.lognormal(0.0, 0.3),
        typo_rate=rng.beta(2, 30),
        lowercase=rng.bernoulli(0.75),
    )


def generate_group(task: Task, rng: RNG, cfg: SimConfig | None = None,
                   n_students: int | None = None) -> list[StudentState]:
    from .. import lang
    cfg = cfg or SimConfig()
    n = n_students or DEFAULT_STUDENTS
    if not MIN_STUDENTS <= n <= MAX_STUDENTS:
        raise ValueError(f"n_students must be in [{MIN_STUDENTS}, {MAX_STUDENTS}], got {n}")
    names = rng.sample(lang.attr(task, "NAMES", NAMES), n)
    kc_ids = [k.id for k in task.kcs]
    roles = assign_roles(task, n)
    students: list[StudentState] = []
    for i in range(n):
        role, facts = roles[i]
        strong = rng.choice(kc_ids)  # one KC Beta(5,3), the others Beta(2,5)
        kcs = {k: KCState(p=rng.beta(5, 3) if k == strong else rng.beta(2, 5)) for k in kc_ids}
        students.append(StudentState(
            id=f"s{i + 1}", name=names[i], role=role, kcs=kcs,
            jigsaw_facts=facts, known_facts=list(facts), traits=sample_traits(rng),
            affect=Affect(engagement=min(1.0, max(0.35, rng.beta(8, 3))),
                          confidence={k: v.p for k, v in kcs.items()}),
        ))
    # misconceptions: 0-2 per student, >=1 per group; prefer weak KCs; never one
    # refuted by the student's own card, and only ones refutable by another member's card.
    for s in students:
        k = rng.weighted([0, 1, 2], [0.4, 0.45, 0.15])
        _assign(task, s, k, rng, students)
    if not any(s.misconceptions for s in students):
        able = [s for s in students if _eligible(task, s, students)]
        if able or len(students) > 2:
            _assign(task, rng.choice(able or students), 1, rng, students)
        else:
            # n=2 relaxation (documented): when no misconception is refutable by the partner's card,
            # one that is not refuted by the holder's own card may be assigned (a peer can still
            # refute it with a KC claim, or an external participant can); else none
            relaxed = [s for s in students if _eligible(task, s, students, relaxed=True)]
            if relaxed:
                _assign(task, rng.choice(relaxed), 1, rng, students, relaxed=True)
    # personal off-task topics (wording only; drawn last so earlier draws are unchanged)
    from .offtask import TOPIC_IDS, TOPICS_PER_STUDENT
    for s in students:
        s.off_task_topics = rng.sample(TOPIC_IDS, TOPICS_PER_STUDENT)
    if cfg.lurk_cap_minutes is not None:  # None = the pilot prior unchanged
        for s in students:
            s.traits.lurk_until_min = min(s.traits.lurk_until_min, cfg.lurk_cap_minutes)
    return students


def _eligible(task: Task, s: StudentState, group: list[StudentState], relaxed: bool = False) -> list:
    """Misconceptions `s` may hold: not refuted by their own card, refutable by a peer's card
    (with no `refutes` mapping in the task, every misconception is eligible). `relaxed` (n=2 only)
    drops the peer-card requirement."""
    if not task.refutes:
        return [m for m in task.misconceptions if s.active_misconception(m.kc) is None]
    own = task.refuted_by(s.jigsaw_facts)
    peers = task.refuted_by([f for o in group if o.id != s.id for f in o.jigsaw_facts])
    return [m for m in task.misconceptions if s.active_misconception(m.kc) is None
            and m.id not in own and (relaxed or m.id in peers)]


def _assign(task: Task, s: StudentState, k: int, rng: RNG, group: list[StudentState] | None = None,
            relaxed: bool = False) -> None:
    pool = _eligible(task, s, group or [s], relaxed)
    weights = [1.0 - s.kcs[m.kc].p for m in pool]
    for _ in range(min(k, len(pool))):
        m = rng.weighted(pool, weights)
        idx = pool.index(m)
        pool.pop(idx)
        weights.pop(idx)
        s.misconceptions.append(MisconceptionState(id=m.id, kc=m.kc, strength=rng.uniform(0.5, 0.9)))
        # holding a misconception caps mastery of that KC below can_state
        s.kcs[m.kc].p = min(s.kcs[m.kc].p, 0.45)
