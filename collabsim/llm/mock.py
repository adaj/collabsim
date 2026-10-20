"""Deterministic MockLLM handlers (offline pipeline checks; NOT findings).

A faithful stand-in for Haiku: the realizer follows SAY THIS CONTENT via the template realizer,
with a small, seeded slip rate (it adds one unknown fact) so the leakage guard has something to
catch; the check flags exactly the code-detectable leaks; interpretation and task drafting use
the code heuristics.

Every decision is a function of a hash of the request, so results are reproducible and
independent of call order.
"""
from __future__ import annotations

import hashlib
import json
import random
import re

from ..models.message import ContentPlan
from ..models.rng import RNG
from ..models.student import StudentState
from ..models.task import Task, load_task
from ..sim.content import sentences

SLIP_RATE = 0.06          # realizer adds an unknown fact (pre-guard leakage)
SLIP_RATE_STRICT = 0.01   # after "STRICT" regeneration prompt


def _rng(req, salt: str = "") -> random.Random:
    h = hashlib.sha256((req.key() + salt).encode()).digest()
    return random.Random(int.from_bytes(h[:8], "big"))


_TASKS: dict[str, Task] = {}


def _task(tid: str) -> Task:
    from .prompts import resolve_task
    t = resolve_task(tid)
    if t is not None:
        return t
    if tid not in _TASKS:
        _TASKS[tid] = load_task(tid)
    return _TASKS[tid]


def _templ():
    from ..sim.realizer import TemplateRealizer
    return TemplateRealizer()


class _Seeded(RNG):
    def __init__(self, r: random.Random):
        self._r = r


def realize(req) -> str:
    m = req.meta
    task = _task(m["task_id"])
    plan = ContentPlan.model_validate(m["plan"])
    s = StudentState.model_validate(m["speaker"])
    r = _rng(req)
    from types import SimpleNamespace   # the template sees the session context (not the transcript)
    st = SimpleNamespace(context=m.get("context") or {}, participants=m.get("participants") or {},
                         messages=[], closing_forced=False)
    text = _templ().content(plan, s, task, _Seeded(r), m["names"], st)
    rate = SLIP_RATE_STRICT if m.get("strict") else SLIP_RATE
    unknown = m["unknown_facts"]
    if unknown and plan.act in ("on_task_claim", "explain", "agree", "question") and r.random() < rate:
        f = task.fact(r.choice(unknown))
        from .. import lang
        text = f"{text}, {lang.attr(task, 'MOCK_ALSO', 'also')} {f.chat[0]}"
    return text


def guard(req) -> str:
    """Faithful classifier: flags a leak iff the message contains a keyphrase of a fact the
    speaker does not know (the code-level check; a real Haiku guard approximates this)."""
    m = req.meta
    task = _task(m["task_id"])
    found = [f for f in task.facts_mentioned(m["text"]) if f in m["unknown_facts"]]
    items = []
    for f in found:  # evidence: the sentence holding the keyphrase, copied from the message
        from ..models.task import fold
        kp = next(fold(k) for k in task.fact(f).keyphrases if fold(k) in fold(m["text"]))
        sent = next((x for x in sentences(m["text"]) if kp in fold(x)), m["text"])
        items.append({"id": f, "quote": sent})
    return json.dumps({"leak": bool(found), "leaked_items": items})


def classify_turn(req) -> str:
    m = req.meta
    task = _task(m["task_id"])
    text = m["text"]
    ex = task.misconceptions_expressed(text)
    kcs = task.content_kcs(text)
    facts = task.facts_mentioned(text)
    if ex:
        act, kc, ok = "on_task_claim", task.misconception(ex[0]).kc, False
    elif kcs:
        act, kc, ok = "on_task_claim", kcs[0], True
    elif text.strip().endswith("?"):
        act, kc, ok = "question", None, None
    else:
        act, kc, ok = "backchannel", None, None
    return json.dumps({"act": act, "kc": kc, "expresses_misconception": ex[0] if ex else None,
                       "is_correct": ok, "facts": facts})


def check(req) -> str:
    """Merged guard + slippage audit (one call): faithful leak detection + turn labels."""
    g = json.loads(guard(req))
    c = json.loads(classify_turn(req))
    return json.dumps({**g, **c})


def coherence(req) -> str:
    """Faithful stand-in for the Haiku coherence check (0.5.2): the code-side checks of sim/coherence.py
    (a belief line stating its refutation, a refuting line asserting the belief, a challenge whose
    words do not refute the belief, a disagreement repeating its target)."""
    from ..sim import coherence as C
    m = req.meta
    task = _task(m["task_id"])
    plan = ContentPlan.model_validate(m["plan"])
    text, target = m["text"], m.get("target") or ""
    c = C.contradiction(task, plan, text)
    if c is not None:
        return json.dumps({"coherent": False, "reason": c["reason"].replace("_", " ")})
    mid = plan.addresses_misconception
    if m.get("role") == "challenge" and mid and plan.claim in ("fact", "correct") and plan.family != "counterexample":
        mine = set(C.refutation_hits(task, mid, text))
        if not mine:
            return json.dumps({"coherent": False, "reason": "the line does not speak against the belief"})
        if C.disagree_marker(text, task) and mine & set(C.refutation_hits(task, mid, target)):
            return json.dumps({"coherent": False, "reason": "it disagrees but repeats what the target said"})
    return json.dumps({"coherent": True, "reason": ""})


def default_handlers() -> dict:
    from ..runtime import interpret, taskspec
    return {"realize": realize, "guard": guard, "classify_turn": classify_turn, "check": check, "coherence": coherence,
            "ping": lambda req: "ok", "interpret": interpret.mock_interpret, "task_draft": taskspec.mock_draft}
