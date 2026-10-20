"""Code-side intent keys for interface and procedural questions.

The planner picks the intent (never the LLM), under per-student and group caps, so that a student
does not keep asking whether the slider saves or "what are we supposed to do". Each intent has a
description for the LLM realizer and natural template lines (also the guard's fallback).
"""
from __future__ import annotations

# per student: the same intent at most once per OWN_WINDOW of their own turns and SESSION_MAX times
# per session; the group: nobody re-asks an intent asked in the last GROUP_WINDOW student turns
OWN_WINDOW = 10
SESSION_MAX = 2
GROUP_WINDOW = 6

INTERFACE = {
    "save_mechanism": ("ask whether your answer or slider gets saved",
                       ["did it save?", "does it save by itself?", "wait did my answer save",
                        "if i change it after submitting does it still save?"]),
    "slider_meaning": ("ask what the confidence slider / bar is for",
                       ["what does the slider do", "what is this bar for?", "the slider is for how sure we are?"]),
    "where_to_type": ("ask where you are supposed to type the answer",
                      ["where do we write the answer?", "i cant see the text box", "where do i type the hypothesis?"]),
    "send_how": ("ask how to send or submit the answer",
                 ["how do i send it", "do i press the send button?", "how do we submit"]),
    "next_button": ("ask how to move on to the next activity",
                    ["is there a next button?", "how do we get to the next page?"]),
    "scroll": ("say the page is not scrolling or ask how to scroll",
               ["wait how do i scroll", "my page wont scroll lol"]),
    "find_activity": ("say you cannot find the activity on the screen",
                      ["where is the activity", "where is the activity tab", "which tab is it on?"]),
    "answer_each": ("ask whether everyone answers or one answer for the group",
                    ["do we each answer or one for all?", "do we each write one?"]),
}
PROCEDURAL = {
    "what_to_do": ("ask what the group is supposed to do now",
                   ["what do we have to do?", "so what are we supposed to do now?", "what do we have to do now"]),
    "what_to_write": ("ask what you are supposed to write as the answer",
                      ["what are we supposed to write?", "what are we supposed to write here?"]),
    "which_activity": ("ask which activity / step the group is on",
                       ["which activity are we on?", "are we on the next step already?"]),
    "whats_the_answer": ("ask the group what the answer is",
                         ["so whats the answer?", "so what goes there?"]),
    "lost": ("say you are lost / confused about what is going on",
             ["wait what?", "im lost lol", "wait im confused"]),
}
ALL = {**INTERFACE, **PROCEDURAL}


def bank(act: str, lang_of=None) -> dict:
    """Intent bank for an act. `lang_of`: a Task or language code (None = English); keys are the same
    in every language pack, only the descriptions and lines are translated."""
    from .. import lang
    if act == "interface":
        return lang.attr(lang_of, "INTERFACE", INTERFACE)
    return lang.attr(lang_of, "PROCEDURAL", PROCEDURAL)


def _all(lang_of=None) -> dict:
    from .. import lang
    return {**lang.attr(lang_of, "INTERFACE", INTERFACE), **lang.attr(lang_of, "PROCEDURAL", PROCEDURAL)}


def description(intent: str, lang_of=None) -> str:
    return _all(lang_of)[intent][0]


def lines(intent: str, lang_of=None) -> list[str]:
    return _all(lang_of)[intent][1]


def caps(n_students: int, cfg=None) -> tuple[int, int]:
    """(interface questions, all confusions = interface + procedural) allowed per session, by group size."""
    base_i = getattr(cfg, "interface_cap_base", 1)
    base_c = getattr(cfg, "confusion_cap_base", 2)
    top = getattr(cfg, "confusion_cap_max", 8)
    return min(top, base_i + n_students // 2), min(top, base_c + n_students // 2)


def confusions(st) -> tuple[int, int]:
    """(interface questions, all confusions) voiced so far by the simulated students (answers excluded)."""
    stu = [m for m in st.messages if m.kind == "student" and m.plan is not None and m.plan.intent
           and not m.plan.answers_intent]
    return sum(1 for m in stu if m.plan.act == "interface"), len(stu)


def allowed(st, sid: str, act: str, cfg=None) -> list[str]:
    """Intent keys `sid` may use now under the caps: per student, per group window, once per session
    group-wide, and a session cap on interface / all confusions that grows with the group size."""
    cap_i, cap_c = caps(len(st.students), cfg)
    n_i, n_c = confusions(st)
    if n_c >= cap_c or (act == "interface" and n_i >= cap_i):
        return []
    stu = [m for m in st.messages if m.kind == "student"]
    own = [m for m in stu if m.speaker == sid and m.plan is not None and m.plan.intent]
    own_recent = {m.plan.intent for m in [m for m in stu if m.speaker == sid][-OWN_WINDOW:]
                  if m.plan is not None and m.plan.intent}
    group_recent = {m.plan.intent for m in stu[-GROUP_WINDOW:] if m.plan is not None and m.plan.intent}
    used: dict[str, int] = {}
    for m in own:
        used[m.plan.intent] = used.get(m.plan.intent, 0) + 1
    # each specific confusion is voiced at most once per session, group-wide
    session = {m.plan.intent for m in stu if m.plan is not None and m.plan.intent and not m.plan.answers_intent}
    return [k for k in bank(act) if used.get(k, 0) < SESSION_MAX and k not in own_recent
            and k not in group_recent and k not in session]
