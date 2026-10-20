"""Prompt builders: the realizer, the per-turn check (leak guard + audit) and, in runtime/, the
interpretation of external participants' messages.

Every builder returns an `LLMRequest` whose `prefix` is stable across a session (cacheable)
and whose `suffix` carries the per-call specifics. `meta` carries structured context that
ONLY the MockLLM uses (it is never sent to a real model).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml

from ..models.message import ContentPlan, Message
from ..models.session import SessionState
from ..models.student import StudentState
from ..models.task import Task
from .client import LLMRequest
from .config import MAX_TOKENS

FEWSHOT_PATH = Path(__file__).resolve().parent / "fewshots.yaml"

# Process-wide registry of the tasks prompts were built for, so the offline MockLLM can resolve a
# request's task (library task in any language, or a custom TaskSpec posted to the service). `meta`
# is never sent to a model and is not part of the cache key.
_TASK_REGISTRY: dict = {}


_REF_CACHE: dict[int, tuple] = {}


def task_ref(task: Task) -> str:
    """English library tasks (unchanged content) keep their plain id; any other task (pt-BR, custom
    TaskSpec, edited copy) is keyed by id, language and a content hash."""
    hit = _REF_CACHE.get(id(task))
    if hit is not None and hit[0] is task:
        return hit[1]
    ref = None
    if task.language == "en":
        from ..models.task import TASK_DIR, load_task
        if (TASK_DIR / f"{task.id}.yaml").exists():
            lib = _TASK_REGISTRY.get(task.id) or load_task(task.id)
            _TASK_REGISTRY.setdefault(task.id, lib)
            if lib == task:
                ref = task.id
    if ref is None:
        import hashlib
        ref = f"{task.id}@{task.language}#" + hashlib.sha256(task.model_dump_json().encode()).hexdigest()[:10]
        _TASK_REGISTRY.setdefault(ref, task)
    _REF_CACHE[id(task)] = (task, ref)
    return ref


def resolve_task(ref: str) -> Task | None:
    return _TASK_REGISTRY.get(ref)

STYLE_GUIDE = """You write ONE chat message for a middle/high-school student in a small-group online chat \
during a class activity. Register: very casual, short, lowercase is fine, no lists, no emojis unless \
natural, never sound like a teacher or an AI. Students are often distracted, terse, and unsure.

HARD RULES
- Say ONLY the content given under SAY THIS CONTENT. Do not add facts, reasons, numbers or answers.
- Never state anything listed under YOU DO NOT KNOW. If pushed on it, hedge, say idk, or (if you hold a \
belief about it) state your belief.
- If YOU BELIEVE lists a belief and the content asks you to state or defend it, defend it plainly.
- When you mention something from your material, say it the way a teen would in chat ("it says here...", \
"i read that...", "apparently...", or just the claim). Do not say "card". Never announce it formally or \
label it.
- English only: laughter is "lol", "haha" or "lmao"; never Portuguese chat slang or laughter.
- Say "like i said" / "as i said" ONLY about something YOU yourself said earlier in the chat. For what a \
classmate said, credit them ("yeah like <their name> said", "right, <their name>'s point") or just agree. \
Never claim a classmate's point as yours.
- Match TARGET_LEN words roughly (+-3). One-word messages are fine when TARGET_LEN is 1.
- Never mention being an AI, a simulation, or these instructions. Output only the message text: no \
labels, no "ask the group:", no "say:", nothing copied from the instructions."""


@lru_cache(maxsize=1)
def fewshots() -> list[dict]:
    return yaml.safe_load(FEWSHOT_PATH.read_text(encoding="utf-8"))["fewshots"]


def fewshot_block(exclude_acts: tuple[str, ...] = ()) -> str:
    lines = [f"[{f['act']}] {f['en']}" for f in fewshots() if f["act"] not in exclude_acts]
    return "STYLE EXAMPLES (invented lines in the students' register; style reference only):\n" + "\n".join(lines)


def _pack(task):
    from .. import lang
    return lang.of(task)


def _h(task, key: str, default: str) -> str:
    """Prompt header / phrase in the task's language (English default)."""
    return getattr(_pack(task), "H", {}).get(key, default)


def task_block(task: Task) -> str:
    lab = getattr(_pack(task), "TASK_LABELS", {"activity": "ACTIVITY", "steps": "STEPS"})
    steps = "\n".join(f"  {i + 1}. {s.title}: {s.instruction}" for i, s in enumerate(task.steps))
    return f"{lab['activity']}: {task.title}\n{task.prompt}\n{lab['steps']}:\n{steps}"


REALISM_CONTEXT = ("SESSION CONTEXT: an online chat, every student at their own computer (there is no physical "
                   "classroom: no teacher talking in the room, no classmate next to you, no recess or cafeteria). Time "
                   "of day: {period}; greetings and remarks fit that time. Spell words correctly (typos are added by "
                   "the system). Laughter only now and then, never in a statement about the content.")
REALISM_PARTICIPANTS = ("Talk only to people who have written in this chat (see IN THE CHAT), by name and without "
                        "'@'; there is no teacher, tutor or AI here. Never call or mention anyone else.")
REALISM_FORMAT = ("FORMAT: a complete sentence with a bit of framing (like 'i think it has to do with...', "
                  "'i read that...'), never a bare fragment.")
TRUNCATED_NOTE = "SHORT: your previous message was too long and got cut off; write one short, complete sentence."
REALISM_CONTINUATION = ("CONTINUATION: you just sent the previous message; continue the SAME point (finish the "
                        "thought), do not start a new subject.")
PERIOD_NAMES = {"manha": "morning", "tarde": "afternoon", "noite": "evening"}


def realizer_prefix(task: Task, st: SessionState | None = None) -> str:
    """Stable per session (cacheable): style guide, task, few-shots, the session context (time of day,
    online setting) and the static rule on whom students talk to."""
    L = _pack(task)
    ctx = st.context if st is not None and getattr(st, "context", None) else {}
    if L.code != "en":  # pt-BR: style guide and few-shots in pt-BR
        style = L.STYLE_GUIDE.replace(getattr(L, "REALISM_STYLE_DROP", "\0"), "")
        parts = [style, task_block(task), L.fewshot_block()]
    else:
        parts = [STYLE_GUIDE, task_block(task), fewshot_block()]
    period = getattr(L, "PERIOD_NAMES", PERIOD_NAMES).get(ctx.get("time_of_day"), ctx.get("time_of_day") or "")
    parts.append(getattr(L, "REALISM_CONTEXT", REALISM_CONTEXT).format(period=period))
    parts.append(getattr(L, "REALISM_PARTICIPANTS", REALISM_PARTICIPANTS))
    return "\n\n".join(parts)


def history(st: SessionState, names: dict[str, str], n: int = 8, viewer: str | None = None,
            task: Task | None = None) -> str:
    """The last n chat turns, each labelled by the writer's display name."""
    out = [f"{names.get(m.speaker, m.speaker)}: {m.text}" for m in st.messages[-n:]]
    return "\n".join(out) or _h(task, "no_msgs", "(no messages yet)")


def in_chat(task: Task, st: SessionState, s: StudentState, names: dict[str, str]) -> str:
    """Who is in the chat, as the speaker sees it: classmates, and external participants who have written."""
    mates = ", ".join(names.get(x.id, x.name) for x in st.students if x.id != s.id) or "-"
    line = f"{_h(task, 'in_chat', 'IN THE CHAT')}: {_h(task, 'classmates', 'classmates')} {mates}"
    ext = [f"{names.get(p, p)} ({_h(task, 'joined', 'joined')})" for p in st.spoken_externals()]
    return line + ("; " + ", ".join(ext) if ext else "")


# ------------------------------------------------------------------ knowledge listing
def knowledge(task: Task, s: StudentState, can_state: float = 0.6) -> tuple[list[str], list[str], list[str]]:
    know = [task.fact(f).text for f in s.known_facts]
    und = _h(task, "understand", "(you understand this)")
    know += [f"{task.kc(k).name} {und}" for k, v in s.kcs.items()
             if v.p >= can_state and s.active_misconception(k) is None]
    believe = [task.misconception(m.id).wrong_belief for m in s.misconceptions if m.active]
    dont = [f.text for f in task.facts if f.id not in s.known_facts]
    dont += [task.kc(k).name for k, v in s.kcs.items() if v.p < can_state]
    return know, believe, dont


GIST_NOTE = (" (that is the gist: put it in your own words, short, the way a student types; never copy the "
             "material's wording)")


def fact_gist(task: Task, fid: str) -> str:
    from ..sim.realizer import fact_gist as _g
    return _g(task, fid)


# How a fact from the student's material is introduced; varied per turn so the model does not
# fall into one formula. The model is told to word it casually.
FACT_LEAD_HINTS = ("you could start with 'mine says'", "you could start with 'my sheet has'",
                   "you could start with 'apparently'", "just state it as a plain claim",
                   "you could start with 'it says'", "just state it, no intro")


def fact_hint(variant: int) -> str:
    return FACT_LEAD_HINTS[variant % len(FACT_LEAD_HINTS)]


def say_this(task: Task, plan: ContentPlan, names: dict[str, str], variant: int = 0, speaker: str | None = None) -> str:
    """What the turn must convey, as an instruction to the model. Worded so that nothing in it reads
    like a line the model could copy verbatim (e.g. "ask the group: ...")."""
    L = _pack(task)
    if plan.closing or plan.answers_intent or plan.nudge or plan.step_q or plan.draft or plan.tag or plan.concedes or (
            plan.act == "off_task" and plan.anchor_turn is not None):
        r = (getattr(L, "say_this_realism", None) or say_this_realism)(task, plan, names)
        if r:
            return r
    if L.code != "en":
        return L.say_this(task, plan, names, variant=variant, speaker=speaker)
    from ..sim import intents
    a = plan.act
    to = names.get(plan.addressee, "") if plan.addressee else ""
    if a == "question" and plan.follow_up:
        if to:
            return (f"the question was already asked and nobody answered; ask {to} directly if they know "
                    "(very short), and do NOT repeat the question")
        return ("the question was already asked and nobody answered; a very short follow-up nudge to the group, "
                "and do NOT repeat the question")
    if a == "off_task":
        from ..sim.offtask import description
        return f"an off-task remark about {description(plan.topic)}, to the whole chat (not addressing a classmate by name)"
    if plan.claim == "misconception" and plan.misconception_id and plan.rationale:
        mm = task.misconception(plan.misconception_id)
        why = mm.because[variant % len(mm.because)] if mm.because else "a reason of your own"
        return (f"{to or 'a classmate'} asked why you think that; answer with your reason, short and casual, in other "
                f"words than before (only the reason, never the correct idea). The reason: {why}. Your belief: "
                + mm.wrong_belief)
    if plan.claim == "misconception" and plan.misconception_id:
        if plan.pushback:
            return ("your idea was questioned; defend it ONCE, with a reason of your own and in other words (do not "
                    "repeat what you already said): " + task.misconception(plan.misconception_id).wrong_belief)
        return f"you believe this and say it as your own opinion: {task.misconception(plan.misconception_id).wrong_belief}"
    ref = _reference_instruction(task, plan, names, speaker) if plan.claim == "fact" and plan.reference else None
    claim = (task.kc(plan.kc).correct_claims[plan.claim_idx if plan.claim_idx is not None else 0]
             if plan.claim == "correct" and plan.kc else None)
    if plan.unsure:
        return (f"{to or 'a classmate'} asked you; say briefly that you are not sure / your material does not have "
                "that (no content)")
    if a == "disagree" or plan.family:
        lead = f"you disagree with {to}. " if to else "you push back on what was said. "
        if plan.addresses_misconception:
            lead = (f"{to or 'someone'} thinks this (wrong): {task.misconception(plan.addresses_misconception).wrong_belief} "
                    f"You {'disagree with' if a == 'disagree' else 'answer'} {to or 'them'}, showing why it is wrong. ")
        if plan.family == "counterexample" and plan.addresses_misconception:
            cx = task.misconception(plan.addresses_misconception).counterexample
            return lead + "Give this counterexample in your own words: " + (cx[0] if cx else "a counterexample")
        if plan.family == "press_for_reasoning":
            return f"ask {to or 'them'} why they think that (no content of your own)"
        if ref:
            return lead + ref
        if plan.claim == "fact" and plan.fact_ids:
            return (lead + f"Use what your material says, casually ({fact_hint(variant)}). The point: "
                    + fact_gist(task, plan.fact_ids[0]) + GIST_NOTE)
        if claim:
            return lead + "The idea you use, in your own words (it must speak against that belief): " + claim
    if ref:
        return ref
    if plan.claim == "fact" and plan.fact_ids:
        return (f"mention this from your material in your own casual words ({fact_hint(variant)}). The point: "
                + fact_gist(task, plan.fact_ids[0]) + GIST_NOTE)
    if claim:
        verb = "restate in your own words the idea" if plan.revoice else "get this idea across in your own words"
        return f"{verb}: {claim}"
    if plan.claim == "hedge":
        return "you are unsure; hedge vaguely without stating any answer or reason"
    if a == "interface":
        return ("one short line about the activity screen: "
                + (intents.description(plan.intent) if plan.intent else "how the interface works"))
    if a == "question" and plan.asked_fact:
        return ("ask the others this question, in your own words, writing only the question itself. "
                "The question: " + task.fact(plan.asked_fact).ask)
    if a == "question":
        return intents.description(plan.intent) if plan.intent else "ask what you are supposed to do"
    return {
        "greeting": "greet the group", "backchannel": "a short reaction (ok/lol/yeah)",
        "agree": f"agree with {to or 'the last message'} without adding content",
        "idk": "say you don't know", "refusal": "decline / say no or idk",
        "press": f"ask {to or 'them'} why they think that",
    }.get(a, "a short reaction")


def say_this_realism(task: Task, plan: ContentPlan, names: dict[str, str]) -> str | None:
    """Wording instructions for the realism-only plans (closing synthesis, interface answers, off-task replies)."""
    from ..sim import intents
    from ..sim.offtask import description
    to = names.get(plan.addressee, "") if plan.addressee else ""
    if plan.closing == "propose" and plan.partial:
        from ..sim.realizer import idea_lines
        items = "; ".join(idea_lines(task, plan.ideas))
        if not items:
            return "time is almost up; suggest sending the group's answer as it is, briefly, with no new content"
        return ("time is almost up: suggest sending the group's answer with what you already concluded, in 1-2 "
                "sentences, and ask the others to confirm. Only this (nothing new): " + items)
    if plan.closing == "propose":
        from ..sim.realizer import idea_lines
        items = "; ".join(idea_lines(task, plan.ideas) if plan.ideas else [k.correct_claims[0] for k in task.kcs])
        return ("propose the group's final answer, summing up in 1-2 sentences what you concluded in the chat (your "
                "own words), and ask the others to confirm. What the group concluded: " + items)
    if plan.closing == "agree" and plan.brief:
        return f"okay the final answer {to or 'your classmate'} proposed in one or two words"
    if plan.closing == "agree":
        return (f"okay the final answer {to or 'your classmate'} proposed, in your own words, short; no new content, "
                "and not the way the others already okayed it")
    if plan.draft:
        return say_this_draft(task, plan, names, to)
    if plan.concedes:
        return (f"you were convinced by {to or 'your classmate'}: say briefly that they are right and you see it "
                "now (no new content, do not repeat your old idea)")
    if plan.tag:
        return ("a very short follow-up to your own previous message (e.g. checking whether the others agree), "
                "no new content")
    if plan.nudge:
        return "the chat has gone quiet; a short nudge to the group to keep going (no content, no question about the answer)"
    if plan.step_q:
        from ..sim.realizer import step_question
        return ("the chat has gone quiet; ask the group the activity question you are on, short and in your own "
                "words: " + step_question(task, plan.kc))
    if plan.closing == "adjust" and plan.fact_ids:
        return (f"agree with {to or 'your classmate'}'s answer but add this detail, briefly: "
                + task.fact(plan.fact_ids[0]).text)
    if plan.act == "interface" and plan.answers_intent and plan.intent:
        return (f"answer {to or 'a classmate'}'s question about the screen ({intents.description(plan.intent)}): "
                "a short, plausible tip")
    if plan.act == "off_task" and plan.anchor_turn is not None:
        return f"reply to {to or 'a classmate'}'s remark about {description(plan.topic)} (short, keep that chat going)"
    return None


def say_this_draft(task: Task, plan: ContentPlan, names: dict[str, str], to: str) -> str:
    from ..sim.realizer import idea_lines
    if plan.draft == "propose":
        items = "; ".join(idea_lines(task, plan.ideas)[:2])
        return ("suggest what the group writes as the answer to this activity, as a hypothesis starting with "
                "'I think', in one short sentence, and ask if it is ok. Only this content: " + items)
    if plan.draft == "confidence":
        return ("a short remark about the confidence slider for this answer (how sure you are, e.g. a "
                "percentage), or ask the others what they put")
    if plan.draft == "typed":
        return "say very briefly that you typed / submitted the answer"
    return f"say the answer {to or 'your classmate'} suggested is fine to write (short, no new content)"


def _reference_instruction(task: Task, plan: ContentPlan, names: dict[str, str], speaker: str | None) -> str:
    """Fix C: a back-reference names its true author."""
    fact = task.fact(plan.fact_ids[0]).text
    if plan.ref_by and plan.ref_by == speaker and plan.restate:
        return ("you said this yourself earlier and it was questioned; restate it BRIEFLY, a few words (e.g. "
                "starting 'like i said'); it was already said, so do not announce it as new: " + fact)
    if plan.ref_by and plan.ref_by == speaker:
        return ("you said this yourself earlier in the chat; say it again BRIEFLY and plainly, a few words, "
                "without 'like i said'; do not announce it as new: " + fact)
    who = names.get(plan.ref_by) if plan.ref_by else None
    if who:
        return (f"{who} (not you) said this earlier in the chat; agree / build on it BRIEFLY, crediting {who} "
                f"(e.g. starting 'yeah like {who.lower()} said' or 'right, {who.lower()}s point'); never 'like i said'; "
                "it was already said, so do not announce it as new: " + fact)
    return ("this was already said in the chat (not by you); agree with it BRIEFLY in a few words, never "
            "'like i said'; do not announce it as new: " + fact)


def mood(s: StudentState, task: Task | None = None) -> str:
    L = _pack(task)
    if L.code != "en":
        return L.mood(s)
    a = s.affect
    parts = ["engaged" if a.engagement > 0.6 else ("bored" if a.engagement < 0.35 else "so-so")]
    if a.frustration > 0.3:
        parts.append("frustrated")
    return ", ".join(parts)


OWN_RECENT_N = 5
OFF_TASK_RECENT_N = 4


def own_recent(st: SessionState, s: StudentState, n: int = OWN_RECENT_N) -> list[str]:
    return [m.text for m in st.messages if m.speaker == s.id][-n:]


def group_off_task_recent(st: SessionState, n: int = OFF_TASK_RECENT_N, window: int = 30) -> list[str]:
    return [m.text for m in st.messages[-window:]
            if m.plan is not None and m.plan.act == "off_task" and len(m.text.split()) > 1][-n:]


def realizer_request(model: str, task: Task, st: SessionState, s: StudentState, plan: ContentPlan,
                     names: dict[str, str], replicate_idx: int = 0, strict: str = "",
                     can_state: float = 0.6, rephrase: str = "") -> LLMRequest:
    know, believe, dont = knowledge(task, s, can_state)
    h = lambda k, d: _h(task, k, d)  # noqa: E731  (headers in the task's language)
    content = say_this(task, plan, names, variant=replicate_idx + len(s.name), speaker=s.id)
    from ..sim.content import visible_fact_sources
    vis = visible_fact_sources(task, st, s.id)
    dont = [d for d in dont if not any(d == task.fact(f).text for f in vis)]
    said_by = h("said_by", "said by {who}")
    said = [f"{task.fact(f).text} ({said_by.format(who=names.get(m.speaker, m.speaker))})"
            for f, m in vis.items() if f not in s.known_facts]
    suffix = "\n".join([
        f"{h('you_are', 'YOU ARE')}: {s.name}", f"{h('act', 'ACT')}: {plan.act}",
        f"{h('target_len', 'TARGET_LEN')}: {plan.target_len}", f"{h('say', 'SAY THIS CONTENT')}: {content}",
        f"{h('know', 'YOU KNOW')}: " + ("; ".join(know) or h("nothing", "nothing specific")),
        f"{h('believe', 'YOU BELIEVE (defend if challenged)')}: " + ("; ".join(believe) or "-"),
        f"{h('dont', 'YOU DO NOT KNOW')}: " + ("; ".join(dont) or "-"),
        f"{h('said', 'SAID BY OTHERS IN THE CHAT (you may refer to it, crediting them)')}: " + ("; ".join(said) or "-"),
        f"{h('mood', 'MOOD')}: {mood(s, task)}",
        in_chat(task, st, s, names),
        f"{h('last', 'LAST MESSAGES')}:\n" + history(st, names, viewer=s.id, task=task),
        f"{h('own', 'YOUR OWN LAST MESSAGES (do NOT repeat or closely rephrase any of them)')}: "
        + (" | ".join(own_recent(st, s)) or "-"),
        f"{h('off', 'RECENT OFF-TOPIC LINES IN THE GROUP (do NOT reuse them; a new remark, or a reply to one)')}: "
        + (" | ".join(group_off_task_recent(st)) or "-"),
        *_realism_lines(task, st, plan, s, names),
        strict, rephrase, h("now", "Now write {name}'s message:").format(name=s.name)])
    unknown_facts = [f.id for f in task.facts if f.id not in s.known_facts]
    return LLMRequest(model=model, purpose="realize", prefix=realizer_prefix(task, st), suffix=suffix,
                      temperature=1.0, max_tokens=MAX_TOKENS["realize"], replicate_idx=replicate_idx,
                      meta={"task_id": task_ref(task), "plan": plan.model_dump(), "speaker": s.model_dump(),
                            "unknown_facts": unknown_facts, "names": names,
                            "recent": [_msg_meta(m) for m in st.messages[-8:]], "strict": bool(strict),
                            "rephrase": bool(rephrase), "context": dict(getattr(st, "context", {}) or {}),
                            "participants": dict(getattr(st, "participants", {}) or {})})


DYAD_NOTE = ("TWO OF YOU: only you and {name} are in this chat: talk to {name} in the singular (you, {name}), "
             "never 'guys', 'everyone' or 'anyone'.")


def _realism_lines(task: Task, st: SessionState, plan: ContentPlan, s: StudentState | None = None,
                   names: dict[str, str] | None = None) -> list[str]:
    L = _pack(task)
    out = []
    from ..sim.realism import is_dyad
    if s is not None and is_dyad(st):
        other = next((x for x in st.roster if x != s.id), "")
        out.append(getattr(L, "DYAD_NOTE", DYAD_NOTE).format(name=(names or {}).get(other, other)))
    if plan.claim in ("fact", "correct", "misconception") and not plan.reference and plan.closing is None:
        out.append(getattr(L, "REALISM_FORMAT", REALISM_FORMAT))
    if plan.continuation:
        out.append(getattr(L, "REALISM_CONTINUATION", REALISM_CONTINUATION))
    return out


def _msg_meta(m: Message) -> dict:
    return {"id": m.id, "speaker": m.speaker, "kind": m.kind, "text": m.text, "addressee": m.addressee,
            "plan": m.plan.model_dump() if m.plan else None}


# ------------------------------------------------------------------ check prefix
# One stable, task-specific system prefix (>= 512 tokens, so it is cacheable) for the per-turn CHECK
# (leakage guard + slippage audit). One cache write per task, cache reads for every later check.
ACT_GLOSSARY = """ACT LABELS (what the realized message does):
- on_task_claim: states something about the task content (a fact, an idea, a belief, a guess with content)
- explain: gives a reason or an explanation for something about the task
- question: asks something (about the content, the activity or the screen)
- agree: agrees with someone, with no new content ("true", "makes sense", "same")
- disagree: disagrees with or challenges someone (with or without a reason)
- off_task: talk not about the activity (food, games, being tired, the internet...)
- interface: talk about the activity screen (saving, sliders, buttons, scrolling, where to type)
- backchannel: a short reaction with no content ("ok", "lol", "hmm", laughter)
- refusal: says they do not know, will not answer, or declines ("idk", "no idea", "pass")"""


def judge_prefix(task: Task) -> str:
    kcs = "\n".join(f"- {k.id}: {k.name}. Correct idea: {k.correct_claims[0]}" for k in task.kcs)
    facts = "\n".join(f"- {f.id} ({f.kc}): {f.text}" for f in task.facts)
    mis = "\n".join(f"- {m.id} ({m.kc}): {m.wrong_belief}" for m in task.misconceptions)
    note = "" if _pack(task).code == "en" else (
        f"\nThe chat and the task are in {_pack(task).label} ({_pack(task).code}); quotes must be copied exactly "
        "as written (same language, same spelling).")
    return f"""You are a careful annotator for a research simulator of small-group student chats.
You never write chat messages yourself. You read ONE item and return ONE JSON object, nothing else.{note}

{task_block(task)}

KNOWLEDGE COMPONENTS (KCs):
{kcs}
FACTS (each is printed on one student's material):
{facts}
MISCONCEPTIONS (wrong beliefs some students hold):
{mis}

{ACT_GLOSSARY}

JOB "check" — a simulated student's message. Do two things at once.
(1) Leakage: does the message ASSERT (state or paraphrase as true) any item from the student's DO-NOT-KNOW
list (facts, answers, reasons, numbers)? Stating the student's own (possibly wrong) belief is allowed;
anything listed under ALLOWED THIS TURN is allowed. Questions, confusion and requests for help ("whats the
answer", "what do we write", "is there a best temperature?") assert nothing and are NOT leaks; only a
question that takes the item for granted ("why does it denature above 60?") is. Evidence is required: for
each leaked item give its id and "quote", the exact words copied from MESSAGE that assert it. If you
cannot quote words that assert the item, do not flag it. Words that only appear in the recent chat lines, in the
task text above, or in the student's own material are not leaks.
(2) Labels of the realized text: act (one of the ACT LABELS above), kc (KC id or null),
expresses_misconception (misconception id or null), is_correct (true/false/null), facts (ids of facts it
mentions). Label what the text says, not what you think the student meant; a message can mention a fact
without asserting it (for example inside a question).
Return {{"leak": bool, "leaked_items": [{{"id": "<item id>", "quote": "<exact words from MESSAGE>"}}], "act": str, "kc": str|null,
"expresses_misconception": str|null, "is_correct": bool|null, "facts": [ids]}}"""


def _judge(model: str, task: Task, purpose: str, suffix: str, meta: dict, replicate_idx: int = 0) -> LLMRequest:
    return LLMRequest(model=model, purpose=purpose, prefix=judge_prefix(task), suffix=suffix, temperature=0.0,
                      max_tokens=MAX_TOKENS[purpose], replicate_idx=replicate_idx, json_mode=True, meta=meta)


def check_request(model: str, task: Task, s: StudentState, text: str, allowed: list[str], dont: list,
                  unknown_facts: list[str], context: str) -> LLMRequest:
    """Merged leakage guard + slippage audit: ONE Haiku call per LLM-realized student turn. `dont`:
    DO-NOT-KNOW items as (id, text) pairs (flags are returned by id with a quote); plain strings are
    numbered."""
    numbered = "\n".join(f"- {d[0]}: {d[1]}" if isinstance(d, (tuple, list)) else f"{i}. {d}"
                          for i, d in enumerate(dont))
    suffix = (f"JOB: check\nSTUDENT: {s.name}\nDO-NOT-KNOW:\n{numbered or '-'}\n"
              f"ALLOWED THIS TURN: {'; '.join(allowed) or '-'}\nRECENT CHAT:\n{context}\nMESSAGE: {text}\nJSON:")
    return _judge(model, task, "check", suffix, {"task_id": task_ref(task), "text": text, "unknown_facts": unknown_facts,
                                                 "speaker": s.model_dump()})


# ------------------------------------------------------------------ coherence check (0.5.2)
# One Haiku call (JSON) on LLM-worded challenge / disagreement / belief lines only: does the candidate
# line hang together with the line it answers? Stable task-specific prefix (>= 512 tokens, cached).
COHERENCE_RULES = """JOB "coherence" — you check ONE candidate chat line written for a simulated student, given the
line it responds to (TARGET) and what the line is meant to do (ROLE). Return {"coherent": bool, "reason": str}.
The line is INCOHERENT when any of these holds:
1. ROLE belief: the line is meant to state the student's (wrong) belief, but it also states the correct idea
   that refutes that belief, or it contradicts itself (e.g. "the enzyme is not used up, it gets used up like a
   fuel"). A belief line may sound unsure; it must not refute itself.
2. ROLE challenge: the line is meant to challenge the belief in TARGET, but its content does not speak against
   that belief: it brings an unrelated true fact, or a fact about another part of the task, instead of the
   reason the belief is wrong (see REFUTED BY below).
3. ROLE challenge: the line says it disagrees ("i disagree", "no", "discordo", "acho que nao", "nada a ver")
   but then says the same thing TARGET already said, or something TARGET does not contradict.
4. ROLE challenge: the line restates or supports the wrong belief it is meant to challenge.
5. ROLE pushback: a holder defends their belief against TARGET; it is incoherent only if it gives the belief up
   while still sounding like a defence, or states the refuting idea.
6. Any line that reads like an instruction to the writer ("you believe this...", "say it as your own opinion",
   "diga que...", "fala como opiniao sua") instead of a chat message.
Casual register, typos, missing accents, short lines, hedges ("acho que", "sei la") and laughter are fine and
are NOT reasons for incoherence. Judge only the logic of the line against TARGET and ROLE.
"reason": one short sentence naming the problem (empty when coherent), in English."""


def coherence_prefix(task: Task) -> str:
    lines = []
    for m in task.misconceptions:
        refs = [task.fact(f).text for f in task.refuting_facts(m.id)]
        refs += [task.kc(m.kc).correct_claims[i] for i in task.refuting_claims(m.id)]
        lines.append(f"- {m.id} ({m.kc}): WRONG BELIEF: {m.wrong_belief}\n  REFUTED BY: " + " | ".join(refs)
                     + (f"\n  COUNTEREXAMPLE: {m.counterexample[0]}" if m.counterexample else ""))
    kcs = "\n".join(f"- {k.id}: {k.name}" for k in task.kcs)
    note = "" if _pack(task).code == "en" else (
        f"\nThe chat is in {_pack(task).label} ({_pack(task).code}); judge it in that language.")
    examples = """EXAMPLES (invented; for the format only):
ROLE challenge of m_x | TARGET: "i think hotter always makes it faster" | LINE: "nah at 80 it denatures and stops"
-> {"coherent": true, "reason": ""}
ROLE challenge of m_x | TARGET: "the enzyme gets used up like fuel" | LINE: "i disagree, amylase breaks starch into sugar"
-> {"coherent": false, "reason": "the fact given does not speak against the belief that the enzyme is used up"}
ROLE challenge of m_x | TARGET: "amylase breaks starch into sugar" | LINE: "i disagree, for me amylase breaks starch into sugar"
-> {"coherent": false, "reason": "it disagrees but repeats what the target said"}
ROLE belief m_x | TARGET: (none) | LINE: "i think the enzyme isnt used up, it gets eaten like fuel"
-> {"coherent": false, "reason": "the belief line also states its refutation"}
ROLE pushback m_x | TARGET: "why do you think that?" | LINE: "cause the more you use it the more it runs out"
-> {"coherent": true, "reason": ""}"""
    return f"""You are a careful annotator for a research simulator of small-group student chats.
You never write chat messages yourself. You read ONE item and return ONE JSON object, nothing else.{note}

{task_block(task)}

KNOWLEDGE COMPONENTS:
{kcs}
MISCONCEPTIONS (wrong beliefs some students hold) and what refutes each:
{chr(10).join(lines)}

{COHERENCE_RULES}

{examples}"""


def coherence_role(plan: ContentPlan) -> str:
    if plan.claim == "misconception":
        return "pushback" if (plan.pushback or plan.anchor_turn is not None) else "belief"
    return "challenge"


def coherence_request(model: str, task: Task, plan: ContentPlan, target: str, text: str,
                      replicate_idx: int = 0) -> LLMRequest:
    role = coherence_role(plan)
    mid = plan.misconception_id if plan.claim == "misconception" else plan.addresses_misconception
    suffix = (f"JOB: coherence\nROLE: {role} of {mid or '-'}\nTARGET: {target or '(none)'}\nLINE: {text}\nJSON:")
    return LLMRequest(model=model, purpose="coherence", prefix=coherence_prefix(task), suffix=suffix,
                      temperature=0.0, max_tokens=MAX_TOKENS["coherence"], replicate_idx=replicate_idx,
                      json_mode=True, meta={"task_id": task_ref(task), "plan": plan.model_dump(), "target": target,
                                            "text": text, "role": role})
