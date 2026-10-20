"""Deterministic TEMPLATE realizer (no LLM) + lowercase noiser.

Turns (ContentPlan, speaker state) into a short chat line. Only the wording is decided here;
every epistemic choice is already fixed in the plan. The Haiku realizer (llm/realizer.py)
implements the same `Realizer` protocol.
"""
from __future__ import annotations

import re
from typing import Protocol

from ..models.message import ContentPlan
from .intents import INTERFACE as INTERFACE_INTENTS, PROCEDURAL as PROCEDURAL_INTENTS, lines as intent_lines
from ..models.rng import RNG
from ..models.student import StudentState
from ..models.task import Task

FILLER = {
    "greeting": ["hi", "hey", "hii", "hello", "hey guys", "hi everyone", "yo"],
    "backchannel": ["ok", "lol", "yeah", "hahaha", "true", "hmm", "okk", "yep", "oh", "right", "haha"],
    "agree": ["agree", "yeah thats right", "i agree", "true", "makes sense", "same", "yes",
              "i agree with that", "yeah i think so too"],
    "idk": ["idk", "no idea", "i dont know", "not sure", "idk tbh", "no clue honestly"],
    "off_task": ["im hungry lol", "this is so long", "what time does class end", "lmaooo",
                 "did u guys see the game yesterday", "my snack is calling me", "brb",
                 "my internet is so slow", "hahaha stop", "who else is bored", "nice",
                 "can we go home", "i forgot my charger", "omg", "the music in here lol"],
    "interface": [l for _k, (_d, ls) in INTERFACE_INTENTS.items() for l in ls],
    "question_generic": [l for _k, (_d, ls) in PROCEDURAL_INTENTS.items() for l in ls],
    "follow_up": ["anyone?", "anyone know?", "guys?", "so does anyone know", "hello?? anyone",
                  "no one knows?", "so?"],
    "follow_up_to": ["{name} do you know?", "{name} what does yours say?", "{name} is it on your sheet?",
                     "{name}??", "what about you {name}"],
    "off_task_atom": ["lol", "omg", "bruh", "lmaooo", "brb", "nice", "hahaha", "ugh", "wow", "yikes", "bro"],
    "refusal": ["i dont know", "no", "idk", "pass", "not really", "nah"],
    "press": ["why tho?", "how do you know that?", "why do you think that?", "but why?",
              "what makes you say that?", "are you sure?"],
}

# facts from the material are voiced naturally and varied, never as "share from my card:"
FACT_LEADS = ["", "", "", "mine says ", "it says here ", "apparently ", "i read that ", "the material says ",
              "ok so it says ", "from what i read, "]
_EXISTING_LEAD_RE = re.compile(r"^(my card says|card says|my card:|it says|mine says)\s*", re.I)
CARD_ANNOUNCE_RE = re.compile(
    r"\b(?:i\s*(?:'ll|will)\s+|let me\s+)?share\s+(?:from|frm|form)\s+(?:my|the|your|our)\s+"
    r"(?:card|sheet|material)\s*[:,-]?\s*", re.I)


def _L(task, name: str, default):
    from .. import lang
    return lang.attr(task, name, default)


def fact_line(task: Task, fid: str, rng: RNG) -> str:
    """A chat phrasing of a card fact with a varied (often empty) natural lead-in."""
    body = rng.choice(task.fact(fid).chat)
    lead = rng.choice(_L(task, "FACT_LEADS", FACT_LEADS))
    if lead:
        body = _L(task, "EXISTING_LEAD_RE", _EXISTING_LEAD_RE).sub("", body)
    return lead + body


# a fact already said is only pointed back to, briefly. "like i said" ONLY for the speaker's own
# earlier statement; someone else's statement is credited to them ("yeah like zoe said") or just
# agreed with -- authorship is never claimed falsely.
REF_LEADS = ["like i said ", "i said it already, ", "yeah like i said ", "again, ", "like i said before "]
REF_LEADS_PEER = ["yeah like {peer} said ", "right, {peer}'s point, ", "{peer} said it, ", "like {peer} said, "]
REF_LEADS_NEUTRAL = ["yeah, ", "right, ", "true, ", "yeah thats the thing, "]


def reference_line(task: Task, fid: str, rng: RNG, own: bool = True, peer: str | None = None) -> str:
    body = _L(task, "EXISTING_LEAD_RE", _EXISTING_LEAD_RE).sub("", min(task.fact(fid).chat, key=lambda c: len(c.split())))
    if own:
        return rng.choice(_L(task, "REF_LEADS", REF_LEADS)) + body
    if peer:
        return rng.choice(_L(task, "REF_LEADS_PEER", REF_LEADS_PEER)).format(peer=peer.lower()) + body
    return rng.choice(_L(task, "REF_LEADS_NEUTRAL", REF_LEADS_NEUTRAL)) + body


CONCEDE_LINES = ["oh ok, that makes sense", "hmm ok maybe youre right", "ah ok so its not just that",
                 "true, didnt think of it like that", "ok fair, i get it now"]
# 0.5.3: a concession quotes the reason it gives in to ({r}: the refuting line, its leads stripped)
CONCEDE_REASON_IF = ["oh, {r}, ok that makes sense", "hmm true, {r}... ok i get it", "ah ok, {r}, makes sense"]
CONCEDE_REASON = ["oh, {r}? ok that makes sense", "hmm yeah, {r}, didnt think of that", "ah ok, {r}, i get it now",
                  "true, {r}, youre right"]
REASON_LEAD_RE = re.compile(
    r"^(?:no|nah|wait|hmm|but|so|like|ok|oh|yeah|right|true|i think|i dont think so|i read that|it says here|"
    r"apparently|mine says|the material says|from what i read|i think it has to do with this:?|"
    r"yeah like \w+ said|like \w+ said|\w+ said it|right, \w+'s point|because|cause)[,:]?\s+", re.I)
REASON_IF_RE = re.compile(r"^(?:if|when)\b", re.I)
CLOSING_CONCEDE_TAILS = [", send it", ", ok send it like that", ". fine by me then"]
CLOSING_HEDGE_OK = ["ok fine", "send it", "ok", "fine, put it", "sure", "alright"]
CLOSING_HEDGE_BUT = ["but i still think its not quite that", "but im still not sure tbh", "but i still kinda disagree",
                     "but i still have my doubts", "but im not fully convinced", "but i still think im kinda right lol"]
TAG_LINES = ["right guys?", "you agree?", "does that make sense?", "i think thats it", "not sure tho", "thats what i got",
             "is that right?"]
HEDGE_FOLLOWUPS = ["i think", "at least thats what i got", "if i got it right", "could be wrong tho", "not 100% sure",
                   "i think so at least", "or something like that", "more or less", "if im not wrong",
                   "at least mine says so"]
OPENERS_LONG = ["wait i found it", "ok look at this", "guys look", "ok so like", "hold on guys"]
DRAFT_PROPOSE_LEADS = ["ok so i put: i think ", "can i write this? i think ", "lets write: i think ", "i wrote: i think "]
DRAFT_JOIN = " and "
DRAFT_AGREE = ["yeah put that", "ok write it like that", "sounds good, put it", "yep thats fine"]
DRAFT_CONFIDENCE = ["i put {pct}% on the confidence", "my slider is at {pct}", "how much did you put on the slider?",
                    "is {pct}% ok for confidence?"]
DRAFT_TYPED = ["ok typed it", "done, i put it in", "ok sent", "put it there"]


def _draft_body(line: str) -> str:
    """An idea line without its own lead-in ("i think", "acho que"), for "i think <idea>"."""
    import re as _r
    return _r.sub(r"^(?:i think |i guess |so |acho que |eu acho que |entao )+", "", line.strip(), flags=_r.I)


def draft_line(task: Task, plan, s, rng: RNG, st=None) -> str:
    """Template wording of the write-up turns (0.5.1)."""
    if plan.draft == "propose":
        items = [_draft_body(x) for x in idea_lines(task, plan.ideas)][:2]
        body = _L(task, "DRAFT_JOIN", DRAFT_JOIN).join(items) if items else ""
        return rng.choice(_L(task, "DRAFT_PROPOSE_LEADS", DRAFT_PROPOSE_LEADS)) + body
    if plan.draft == "confidence":
        c = s.affect.confidence.get(plan.kc, 0.6) if plan.kc else 0.6
        pct = int(min(95, max(30, round(c * 100 / 10) * 10 + rng.choice([0, 10, 20]))))
        return rng.choice(_L(task, "DRAFT_CONFIDENCE", DRAFT_CONFIDENCE)).format(pct=pct)
    if plan.draft == "typed":
        return rng.choice(fresh_forms(_L(task, "DRAFT_TYPED", DRAFT_TYPED), st))
    return rng.choice(fresh_forms(_L(task, "DRAFT_AGREE", DRAFT_AGREE), st))


def fact_gist(task: Task, fid: str) -> str:
    """A short gist of a card fact for the LLM realizer (0.5.1): the shortest chat-register phrasing
    without its lead-in, instead of the material's full sentence (which models tend to copy)."""
    f = task.fact(fid)
    rx = _L(task, "EXISTING_LEAD_RE", _EXISTING_LEAD_RE)
    lines = [rx.sub("", c).strip() for c in f.chat]
    best = min((c for c in lines if c), key=lambda c: len(c.split()), default="")
    return best if best and len(best.split()) < len(f.text.split()) else f.text


def rationale_line(task: Task, plan, s, rng: RNG, st=None) -> str:
    """A holder's reason for their belief (0.5.2), from the misconception's `because` lines (a wording the
    speaker has not used yet), with a short casual lead."""
    m = task.misconception(plan.misconception_id)
    said = {fold_text(x.text) for x in (getattr(st, "messages", None) or [])}
    bank = [b for b in m.because if fold_text(b) not in said and not any(fold_text(b) in x for x in said)]
    if not bank:
        bank = list(m.because) or list(_L(task, "RATIONALE_FALLBACK", RATIONALE_FALLBACK))
    return rng.choice(_L(task, "RATIONALE_LEADS", RATIONALE_LEADS)) + rng.choice(bank)


def step_question(task: Task, kc: str | None) -> str:
    """The activity question of the step that works on `kc` (as shown on the screen), lower-cased."""
    sid = next((g.step for g in task.gold if kc in g.kcs), task.steps[0].id)
    q = next(s.instruction for s in task.steps if s.id == sid).strip()
    if "?" in q:
        q = q[:q.index("?") + 1]       # the question itself, not the rest of the instruction
    return q[:1].lower() + q[1:]


# English chat only -- Portuguese laughter / slang never appears in an English line
# ("kkkk" -> "lol"/"haha"/"lmao"). Safety net for model output too.
PT_MARKERS_RE = re.compile(
    r"\b(k{2,}|rs+|vish|eita|mds|nossa|mano|slk|pqp|blz|tbm|n[ée]|nao|não|hein|kkk\w*|ss|uhum|aff)\b", re.I)
EN_LAUGHS = ["lol", "haha", "lmao", "hahaha"]


def english_only(text: str, rng: RNG) -> str:
    if not PT_MARKERS_RE.search(text):
        return text
    def sub(m):
        w = m.group(0).lower()
        if w.startswith("k") or w.startswith("rs"):
            return rng.choice(EN_LAUGHS)
        return {"vish": "oh no", "eita": "whoa", "mds": "omg", "nossa": "wow", "mano": "bro", "slk": "omg",
                "pqp": "omg", "blz": "ok", "tbm": "too", "ne": "right", "né": "right", "nao": "no",
                "não": "no", "hein": "huh", "ss": "yes", "uhum": "mhm", "aff": "ugh"}.get(w, "")
    return re.sub(r"\s{2,}", " ", PT_MARKERS_RE.sub(sub, text)).strip() or "lol"


def strip_card_announcement(text: str, rng: RNG, task=None) -> str:
    """Code safety net for LLM wording: replace a formulaic "share from my card:" announcement
    with a natural lead (or nothing)."""
    rx = _L(task, "CARD_ANNOUNCE_RE", CARD_ANNOUNCE_RE)
    if not rx.search(text):
        return text
    leads = _L(task, "CARD_ANNOUNCE_LEADS", ["", "", "mine says ", "apparently ", "it says "])
    return rx.sub(lambda _m: rng.choice(leads), text).strip()


def normalize_chat(text: str, rng: RNG, task=None) -> str:
    """Language filter on every line: English -> english_only (no pt-BR slang/laughter); pt-BR -> no
    English slang ("lol" -> "kkkk"). Portuguese laughter is right in pt-BR."""
    fn = _L(task, "normalize_chat", None)
    return english_only(text, rng) if fn is None else fn(text, rng)


# English banks (invented lines)
GREETING_BY_PERIOD = {"manha": ["good morning", "morning guys"], "tarde": ["good afternoon"], "noite": ["good evening"]}
OFFTASK_REPLY = ["same", "fr?", "wait really?", "me too", "ugh same", "ok thats true", "no way"]
INTERFACE_ANSWERS = {
    "save_mechanism": ["i think it saves by itself", "mine saved when i clicked outside the box"],
    "slider_meaning": ["i think its how sure we are", "its for how confident we are in the answer"],
    "where_to_type": ["theres a box under the question", "its the text box right below"],
    "send_how": ["theres a send button in the corner", "just click send at the bottom"],
    "next_button": ["i think it shows up after we answer", "scroll to the end, the button is there"],
    "scroll": ["try clicking the page and scrolling again", "mine scrolled after i clicked in the middle"],
    "find_activity": ["its on the first tab at the top", "refresh the page and it shows up"],
    "answer_each": ["i think its one answer for the group", "we each write one but it can be the same"],
}
CLOSING_PROPOSE_LEADS = ["ok lets write the group answer: ", "so we put this: ", "summing up for the group: ",
                         "can i write it like this? "]
CLOSING_JOIN = "; and "
CLOSING_AGREE = ["yep, send it like that", "yeah i agree with that answer", "nice, thats complete", "fine by me"]
# 0.5.3: closing okays are composed (ack + verdict + send), never two alike in a session
CLOSING_ACKS = ["yep", "ok", "nice", "yeah", "cool", "perfect", "sure", "great", "alright", "yes", "true", "good"]
CLOSING_VERDICTS = ["thats it", "looks good", "i agree", "thats complete", "sounds right", "makes sense to me",
                    "nothing missing", "thats what we said", "fine by me", "i think thats right", "good summary",
                    "all good"]
CLOSING_SENDS = ["", "", "send it", "send it like that", "go ahead", "put it in", "submit it", "lets send"]
CLOSING_PROPOSE_TAILS = ["", "", "", " ok?", " good?", " is that right?"]
CLOSING_JOINS = ["; and ", ", and ", ". also ", " + "]
CLOSING_ADJUST_LEADS = ["agree, just also put that ", "yeah, and add that "]
CREDIT_LINES = ["yeah thats what {peer} said", "{peer} already said it, i agree", "right, same as {peer}",
                "i think {peer} is right on that"]
BELIEF_CREDIT_LINES = ["yeah i think the same as {peer}", "same, i agree with {peer} on that",
                       "i was thinking that too, like {peer}"]
# 0.5.1: a short nudge after a long silence, and the activity question asked to the group
NUDGE_LINES = ["hey guys, still here?", "so are we doing this one or what", "anyone still here?",
               "ok so what do we put for this one", "guys lets keep going", "hello?? lets do this"]
STEP_Q_LEADS = ["so the question is ", "ok guys, ", "wait so ", "guys "]
STEP_MOVE_LINES = ["next one?", "ok that ones done, next", "lets do {n} now", "on to {n}?", "ok number {n} now",
                   "done with that one, next"]
CLOSING_AGREE_BRIEF = ["ok", "fine", "sure", "yep"]
FACT_LEADS_FRAMED = ["mine says ", "apparently ", "it says here ", "i read that ", "i think it has to do with this: ",
                     "from what i read, "]
CLOSING_PARTIAL_LEADS = ["ok time is almost up, lets send what we have: ", "we should send it now, so far: ",
                         "lets put what we got: "]
CLOSING_PARTIAL_EMPTY = ["ok time is almost up, we should send what we have", "lets just send what we got so far"]
PUSHBACK_LEADS = ["hmm but i still think ", "hmm for me ", "but wait, i still think ", "no but "]
# 0.5.2: answers to "why do you think that?" / a name ping (invented lines)
RATIONALE_LEADS = ["", "", "well ", "idk ", "like "]
RATIONALE_FALLBACK = ["idk it just makes sense to me", "cause thats what i always heard"]
UNSURE_LINES = ["it was just what i thought", "not sure anymore tbh", "hmm now im not sure"]
PING_NO_LINES = ["mine doesnt have that", "its not on mine sorry", "hmm mine doesnt say", "idk, not on my sheet"]

OPENERS = ["wait", "guys", "so", "ok so", "look", "hold on", "um", "like"]
HEDGE_PREFIX = ["i think ", "maybe ", "not sure but ", "i guess "]
CLAIM_PREFIX = ["", "", "", "i think ", "ok so ", "so ", "i guess "]
DISAGREE_PREFIX = ["no ", "nah ", "i dont think so, ", "wait no, ", "hmm no "]


class Realizer(Protocol):
    def realize(self, plan: ContentPlan, speaker: StudentState, task: Task, rng: RNG,
                names: dict[str, str], st=None, rephrase: str = "") -> str: ...

    def variant(self, plan: ContentPlan, speaker: StudentState, task: Task, rng: RNG,
                names: dict[str, str], st=None) -> str: ...


class TemplateRealizer:
    def realize(self, plan: ContentPlan, speaker: StudentState, task: Task, rng: RNG,
                names: dict[str, str], st=None, rephrase: str = "") -> str:
        return noise(normalize_chat(self.content(plan, speaker, task, rng, names, st), rng, task), speaker, rng, st)

    def variant(self, plan: ContentPlan, speaker: StudentState, task: Task, rng: RNG,
                names: dict[str, str], st=None) -> str:
        """Another template wording of the same plan (repetition guard's last resort): draws over
        nearby target lengths too (pick_len would otherwise keep returning the same filler), drops
        candidates too similar to the recent window, and keeps the one closest to target_len."""
        from .repetition import check
        cands = []
        for _ in range(12):
            p = plan.model_copy(update={"target_len": max(1, plan.target_len + rng.randint(-3, 3))})
            c = self.content(p, speaker, task, rng, names, st)
            if c not in cands:
                cands.append(c)
        if st is not None:
            fresh = [c for c in cands if check(c, st, speaker.id) is None]
            cands = fresh or cands
        best = min(abs(len(c.split()) - plan.target_len) for c in cands)
        return noise(rng.choice([c for c in cands if abs(len(c.split()) - plan.target_len) == best]), speaker, rng, st)

    def candidates(self, plan: ContentPlan, speaker: StudentState, task: Task, names: dict[str, str],
                   st=None, n: int = 40) -> list[str]:
        """All (un-noised) template wordings of this plan, deterministic order (guard fallback)."""
        out: list[str] = []
        for i in range(n):
            c = self.content(plan, speaker, task, RNG(1000 + i), names, st)
            if c not in out:
                out.append(c)
        return out

    def safe_line(self, plan: ContentPlan, s: StudentState, task: Task, rng: RNG, names: dict[str, str],
                  st=None, ok=lambda c: True) -> str:
        """Plan-consistent last resort when no template wording of the plan passes `ok`:
        never a content-free hedge for a content act."""
        if plan.claim == "correct" and plan.kc:
            for i, c in enumerate(task.kc(plan.kc).correct_claims):
                if ok(c):
                    return c
        if plan.claim == "fact" and plan.fact_ids:
            f = task.fact(plan.fact_ids[0])
            for c in [*f.chat, f.text.lower().rstrip(".")]:
                if ok(c):
                    return c
        if plan.claim == "misconception" and plan.misconception_id:
            return rng.choice(task.misconception(plan.misconception_id).wrong_answer_template)
        if plan.act == "question" and plan.asked_fact:
            return task.fact(plan.asked_fact).ask
        if plan.act in ("question", "press", "disagree", "explain", "on_task_claim"):
            return rng.choice(_L(task, "SAFE_QUESTIONS", SAFE_QUESTIONS))
        return pick_len(_L(task, "FILLER", FILLER)["backchannel"], plan.target_len, rng)

    def content(self, plan: ContentPlan, speaker: StudentState, task: Task, rng: RNG,
                names: dict[str, str], st=None) -> str:
        """Un-noised template wording (also the guard's fallback)."""
        F = _L(task, "FILLER", FILLER)
        if plan.split_tail:
            return plan.split_tail       # the rest of the speaker's own line (0.5.2 split thought)
        if plan.opener:
            bank = _L(task, "OPENERS", OPENERS) if plan.target_len <= 1 else _L(task, "OPENERS_LONG", OPENERS_LONG)
            from . import realism
            if realism.is_dyad(st):       # no "galera" / "gente olha isso" when there are only two of them
                bank = [b for b in bank if not realism.plural_address(b, task)] or bank
            return rng.choice(fresh_forms(bank, st, cap=1))
        r = self._realism_content(plan, speaker, task, rng, names, st)
        if r is not None:
            return r
        if plan.act == "question" and plan.follow_up:
            if plan.addressee and plan.addressee in names:
                return rng.choice(F["follow_up_to"]).format(name=names[plan.addressee].lower())
            return rng.choice(F["follow_up"])
        if plan.act == "off_task":
            return off_task_line(plan, rng, st, task)
        text = self._content(plan, speaker, task, rng, names, st)
        if plan.addressee and plan.addressee in names and plan.act in {
                "question", "disagree", "press", "on_task_claim", "explain"} and rng.bernoulli(0.3):
            text = f"{names[plan.addressee].lower()} {text}"
        return text

    def _realism_content(self, plan: ContentPlan, s: StudentState, task: Task, rng: RNG,
                         names: dict[str, str], st) -> str | None:
        """Closing synthesis, interface answers, off-task replies, greetings that fit the time of day,
        and first fact mentions framed as contributions (never a bare fragment)."""
        from . import realism
        F = _L(task, "FILLER", FILLER)
        if plan.closing == "propose" and plan.partial:
            items = established_lines(task, plan.ideas, rng)
            if not items:
                return rng.choice(_L(task, "CLOSING_PARTIAL_EMPTY", CLOSING_PARTIAL_EMPTY))
            return rng.choice(_L(task, "CLOSING_PARTIAL_LEADS", CLOSING_PARTIAL_LEADS)) + join_items(task, items, rng)
        if plan.closing == "propose":
            # 0.5.3: what the group established in the chat (plan.ideas), paraphrased, joined in varied ways
            items = established_lines(task, plan.ideas, rng) if plan.ideas else [k.correct_claims[0] for k in task.kcs]
            return rng.choice(_L(task, "CLOSING_PROPOSE_LEADS", CLOSING_PROPOSE_LEADS)) + join_items(task, items, rng) \
                + rng.choice(_L(task, "CLOSING_PROPOSE_TAILS", CLOSING_PROPOSE_TAILS))
        if plan.closing == "agree" and plan.concedes:     # giving in at the very end: the reason, then the okay
            return concede_line(task, plan, rng, names, st) + rng.choice(
                _L(task, "CLOSING_CONCEDE_TAILS", CLOSING_CONCEDE_TAILS))
        if plan.closing == "agree" and plan.holds_out:
            return closing_hedge_line(task, rng, st)
        if plan.closing == "agree" and plan.brief:
            return rng.choice(fresh_forms(_L(task, "CLOSING_AGREE_BRIEF", CLOSING_AGREE_BRIEF), st, cap=1))
        if plan.closing == "agree":
            return closing_agree_line(task, rng, st)
        if plan.nudge:
            bank = _L(task, "NUDGE_LINES", NUDGE_LINES)
            if realism.is_dyad(st):
                bank = [b for b in bank if not realism.plural_address(b, task)] or bank
            return rng.choice(fresh_forms(bank, st, cap=1))
        if plan.step_move:
            n = (getattr(st, "step_idx", 0) or 0) + 1
            return rng.choice(fresh_forms(_L(task, "STEP_MOVE_LINES", STEP_MOVE_LINES), st, cap=1)).format(n=n)
        if plan.draft is not None:
            return draft_line(task, plan, s, rng, st)
        if plan.after_hedge:  # 0.5.3, each wording once per session
            return rng.choice(fresh_forms(_L(task, "HEDGE_FOLLOWUPS", HEDGE_FOLLOWUPS), st, cap=1))
        if plan.tag:      # each wording once per session (0.5.2)
            return rng.choice(fresh_forms(_L(task, "TAG_LINES", TAG_LINES), st, cap=1))
        if plan.rationale and plan.misconception_id:
            return rationale_line(task, plan, s, rng, st)
        if plan.unsure:
            am = st.messages[plan.anchor_turn] if st is not None and plan.anchor_turn is not None and \
                plan.anchor_turn < len(getattr(st, "messages", [])) else None
            pressed = am is not None and am.plan is not None and am.plan.act == "press"
            bank = _L(task, "UNSURE_LINES", UNSURE_LINES) if pressed else _L(task, "PING_NO_LINES", PING_NO_LINES)
            return rng.choice(fresh_forms(bank, st, cap=1))
        if plan.concedes:
            return concede_line(task, plan, rng, names, st)
        if plan.step_q:
            q = step_question(task, plan.kc)
            return rng.choice(_L(task, "STEP_Q_LEADS", STEP_Q_LEADS)) + q
        if plan.closing == "adjust" and plan.fact_ids:
            body = _L(task, "EXISTING_LEAD_RE", _EXISTING_LEAD_RE).sub("", rng.choice(task.fact(plan.fact_ids[0]).chat))
            return rng.choice(_L(task, "CLOSING_ADJUST_LEADS", CLOSING_ADJUST_LEADS)) + body
        if plan.act == "interface" and plan.intent and not plan.answers_intent and plan.draft is None:
            return decorate(pick_len(intent_lines(plan.intent, task), plan.target_len, rng), rng, task, "INTERFACE", 0.35)
        if plan.act == "interface" and plan.answers_intent and plan.intent:
            return rng.choice(_L(task, "INTERFACE_ANSWERS", INTERFACE_ANSWERS).get(plan.intent, ["acho que é assim mesmo"]))
        if plan.act == "off_task" and plan.anchor_turn is not None and plan.target_len > 1:
            said = {fold_text(m.text) for m in (getattr(st, "messages", None) or [])}
            replies = [x for x in _L(task, "OFFTASK_REPLY", OFFTASK_REPLY) if fold_text(x) not in said]
            if replies and rng.bernoulli(0.6):
                return rng.choice(replies)        # (a reply line is not reused verbatim in a session)
            return off_task_line(plan, rng, st, task)
        if plan.act == "greeting":
            tod = st.context.get("time_of_day") if st is not None else None
            opts = list(F["greeting"]) + list(_L(task, "GREETING_BY_PERIOD", GREETING_BY_PERIOD).get(tod, []))
            opts = [o for o in opts if realism.ok_line(o, st, task)] or ["oi"]
            return pick_len(fresh_forms(opts, st), plan.target_len, rng)
        if (plan.act in ("on_task_claim", "explain") and plan.claim == "fact" and plan.fact_ids
                and not plan.reference and not plan.family):
            body = _L(task, "EXISTING_LEAD_RE", _EXISTING_LEAD_RE).sub("", rng.choice(task.fact(plan.fact_ids[0]).chat))
            return rng.choice(_L(task, "FACT_LEADS_FRAMED", FACT_LEADS_FRAMED)) + body
        return None

    # -- content selection ---------------------------------------------------------
    def _content(self, plan: ContentPlan, s: StudentState, task: Task, rng: RNG,
                 names: dict[str, str] | None = None, st=None) -> str:
        names = names or {}
        F = _L(task, "FILLER", FILLER)
        a = plan.act
        if a in ("on_task_claim", "explain", "disagree"):
            prefix = rng.choice(_L(task, "DISAGREE_PREFIX", DISAGREE_PREFIX)) if a == "disagree" else ""
            if plan.claim == "misconception" and plan.misconception_id:
                tmpl = task.misconception(plan.misconception_id).wrong_answer_template
                if plan.pushback:  # defends the belief once, not verbatim: another wording + a "but"
                    said = {fold_text(m.text) for m in (st.messages if st is not None else []) if m.speaker == s.id}
                    other = [t for t in tmpl if fold_text(t) not in said and not any(fold_text(t) in x for x in said)]
                    body = rng.choice(other or tmpl)
                    return join_lead(rng.choice(_L(task, "PUSHBACK_LEADS", PUSHBACK_LEADS)), body)
                body = rng.choice(tmpl)
                return prefix + body
            if plan.claim == "fact" and plan.fact_ids:
                own = plan.ref_by == s.id if plan.ref_by else plan.fact_ids[0] in s.shared_facts
                peer = names.get(plan.ref_by) if plan.ref_by and plan.ref_by != s.id else None
                # "like i said" only when restating after a dispute (0.5.1); else a plain "yeah, ..."
                body = (reference_line(task, plan.fact_ids[0], rng, own=own and plan.restate,
                                       peer=None if own else peer)
                        if plan.reference else fact_line(task, plan.fact_ids[0], rng))
                if plan.family == "counterexample" and plan.addresses_misconception:
                    cx = task.misconception(plan.addresses_misconception).counterexample
                    body = rng.choice(cx) if cx else body
                return prefix + body
            if plan.claim == "correct" and plan.kc:
                claims = task.kc(plan.kc).correct_claims
                body = claims[plan.claim_idx] if plan.claim_idx is not None else rng.choice(claims)
                if plan.family == "counterexample" and plan.addresses_misconception:
                    cx = task.misconception(plan.addresses_misconception).counterexample
                    body = rng.choice(cx) if cx else body
                return prefix + (rng.choice(_L(task, "CLAIM_PREFIX", CLAIM_PREFIX)) if not prefix else "") + body
            if plan.claim == "hedge":
                step_hint = {"kc": plan.kc}
                return rng.choice(_L(task, "HEDGE_PREFIX", HEDGE_PREFIX)) + _hedge(task, step_hint["kc"], rng)
            return pick_len(fresh_forms(F["idk"], st), plan.target_len, rng)
        if a == "question":
            if plan.asked_fact:
                return task.fact(plan.asked_fact).ask
            if plan.intent:
                return rng.choice(intent_lines(plan.intent, task))
            return rng.choice(F["question_generic"])
        if a == "interface" and plan.intent:
            return pick_len(intent_lines(plan.intent, task), plan.target_len, rng)
        if a == "press":
            return rng.choice(F["press"])
        if a == "refusal":
            return pick_len(fresh_forms(F["refusal"], st), plan.target_len, rng)
        return pick_len(fresh_forms(F.get(a, F["backchannel"]), st), plan.target_len, rng)


def off_task_line(plan: ContentPlan, rng: RNG, st=None, task=None) -> str:
    """A line of the plan's off-task topic, preferring lines nobody has used this session."""
    from .offtask import lines
    from .repetition import too_similar
    if plan.target_len <= 1:
        return rng.choice(_L(task, "FILLER", FILLER)["off_task_atom"])
    opts = lines(plan.topic, task)
    from . import realism
    if st is not None:
        opts = [o for o in opts if realism.ok_line(o, st, task)] or opts
    if st is not None:
        used = [m.text for m in st.messages if m.kind == "student" and m.plan is not None
                and m.plan.act == "off_task"]
        unused = [o for o in opts if not any(too_similar(o, u) for u in used)]
        opts = unused or opts
    line = pick_len(opts, plan.target_len, rng) if rng.bernoulli(0.5) else rng.choice(opts)
    return decorate(line, rng, task, "OFFTASK")


# 0.5.3: off-task / interface lines get a seeded interjection or tail now and then, so sessions with different
# seeds rarely type the very same line
OFFTASK_PREFIXES = ["bro ", "omg ", "ugh ", "ok but ", "honestly ", "lowkey "]
OFFTASK_SUFFIXES = [" lol", " ugh", " fr", " tbh", " smh"]
INTERFACE_PREFIXES = ["wait ", "um ", "hey ", "uh "]
INTERFACE_SUFFIXES = [" lol", " ??", " help"]


def decorate(line: str, rng: RNG, task=None, kind: str = "OFFTASK", p: float = 0.45) -> str:
    if not line or len(line.split()) < 2 or not rng.bernoulli(p):
        return line
    pre = _L(task, f"{kind}_PREFIXES", globals()[f"{kind}_PREFIXES"])
    suf = _L(task, f"{kind}_SUFFIXES", globals()[f"{kind}_SUFFIXES"])
    if rng.bernoulli(0.5):
        w = rng.choice(pre)
        return line if fold_text(line).startswith(fold_text(w)) else w + line
    t = rng.choice(suf)
    return line if fold_text(line).endswith(fold_text(t)) else line.rstrip("?!. ") + t


def pick_len(options: list[str], target: int, rng: RNG) -> str:
    """Choose among the options whose word count is closest to the target length."""
    best = min(abs(len(o.split()) - target) for o in options)
    return rng.choice([o for o in options if abs(len(o.split()) - target) == best])


# a hedge is only ever the wording of a HEDGE plan (never a fallback for a content act)
HEDGES = ["its something like that but idk", "not sure about this one", "we need to look at the material",
          "something with the numbers?", "i kinda get it but not really", "wait let me read it again"]
SAFE_QUESTIONS = ["what do you guys think?", "wait what do you think it is?", "so what do we put for this one?"]


def _hedge(task: Task, kc: str | None, rng: RNG) -> str:
    """Content-free hedge: signals partial engagement without asserting any KC content."""
    return rng.choice(_L(task, "HEDGES", HEDGES))


# --------------------------------------------------------------------------------
# deterministic noiser (lowercase, dropped final punctuation); typos are added by sim/realism.polish
# (at most one realistic typo per message)
# --------------------------------------------------------------------------------
def noise(text: str, s: StudentState, rng: RNG, st=None) -> str:
    if s.traits.lowercase:
        text = text.lower()
    if rng.bernoulli(0.6):
        text = text.rstrip(".!")
    return text


def join_lead(lead: str, body: str) -> str:
    """lead + body without a doubled phrase ("i still think " + "i think its 80" -> "i still think its 80")."""
    lw, bw = lead.split(), body.split()
    for k in range(min(3, len(bw)), 0, -1):
        if [w.lower() for w in bw[:k]] == [w.lower().strip(",") for w in lw[-k:]]:
            return lead + " ".join(bw[k:])
    for k in range(min(3, len(bw)), 1, -1):   # "eu acho que " + "acho que ..." (a suffix of the lead)
        if [w.lower() for w in bw[:k]] == [w.lower() for w in lw[len(lw) - k:]]:
            return lead + " ".join(bw[k:])
    if len(bw) > 2 and lw and bw[0].lower() in ("i", "eu") and bw[1].lower() == lw[-1].lower():
        return lead + " ".join(bw[2:])          # "i still think " + "i think its 80"
    # 0.5.3: "eu continuo achando que " + "acho que é so..." / "i still think " + "i guess its..."
    low = body.lower()
    if lw and lw[-1].lower() in ("que", "think", "that"):
        for pre in ("eu acho que ", "acho que ", "acho q ", "i think ", "i guess ", "i think that "):
            if low.startswith(pre) and len(bw) > len(pre.split()) + 1:
                return lead + " ".join(bw[len(pre.split()):])
    return lead + body


def fold_text(text: str) -> str:
    from ..models.task import fold
    return " ".join(fold(text).split())


def fresh_forms(options: list[str], st=None, cap: int = 2) -> list[str]:
    """Filler wordings used fewer than `cap` times this session (all of them if none is left): the
    same "sla" / "ok" is not typed over and over."""
    if st is None or not getattr(st, "messages", None):
        return options
    used: dict[str, int] = {}
    for m in st.messages:
        k = fold_text(m.text)
        used[k] = used.get(k, 0) + 1
    fresh = [o for o in options if used.get(fold_text(o), 0) < cap]
    return fresh or options


def idea_lines(task: Task, ideas: list[str]) -> list[str]:
    """One chat phrasing per established key idea (forced closing): a correct claim of its KC, else a
    card fact."""
    out = []
    for ki in task.all_key_ideas():
        if ki.id not in ideas:
            continue
        if ki.claims:
            line = task.kc(ki.kc).correct_claims[ki.claims[0]]
        elif ki.facts:
            line = _L(task, "EXISTING_LEAD_RE", _EXISTING_LEAD_RE).sub("", task.fact(ki.facts[0]).chat[0])
        else:
            continue
        if line not in out:
            out.append(line)
    return out


def reason_clause(task: Task, text: str, names: dict | None = None, max_words: int = 16) -> str:
    """The gist of a refuting line, for a concession to quote (0.5.3): the line without its address, leads
    ("pera nao, mas", "no meu diz que", "é como ana falou") and question mark, cut at a clause end."""
    from .realism import SPLIT_DANGLING, fold
    lg = "pt-BR" if getattr(task, "language", "en") == "pt-BR" else "en"
    first = {fold(n).split()[0] for n in (names or {}).values() if n}
    words = text.strip().split()
    while words and fold(words[0]).strip(",:") in first:
        words.pop(0)
    t = " ".join(words)
    rx = _L(task, "REASON_LEAD_RE", REASON_LEAD_RE)
    for _ in range(6):
        new = rx.sub("", t, count=1).strip(" ,")
        if new == t:
            break
        t = new
    t = t.rstrip("?!. ")
    ws = t.split()
    for c in _L(task, "REASON_CUTS", (r"\s+and (?:it|they) (?:doesnt|isnt|dont)\b",)):
        mm = re.search(c, fold(t))
        if mm and mm.start() > 0 and len(t[:mm.start()].split()) >= 4:
            ws = t[:mm.start()].split()
            break
    if len(ws) > max_words:
        head = " ".join(ws[:max_words])
        if "," in head and len(head[:head.rindex(",")].split()) >= 4:
            ws = head[:head.rindex(",")].split()
        else:
            ws = ws[:max_words]
    while len(ws) > 3 and (fold(ws[-1]).strip(",") in SPLIT_DANGLING[lg] or not re.search(r"[a-z0-9]", fold(ws[-1]))):
        ws.pop()
    return " ".join(ws).rstrip(",")


def concede_line(task: Task, plan, rng: RNG, names: dict | None = None, st=None) -> str:
    """A concession that names the reason (0.5.3): the refuting line it answers, quoted briefly; the plain bank
    when there is none to quote."""
    msgs = getattr(st, "messages", None) or []
    ref = plan.reason_turn if plan.reason_turn is not None else plan.anchor_turn
    r = ""
    if ref is not None and 0 <= ref < len(msgs):
        text = msgs[ref].text
        nxt = msgs[ref + 1] if ref + 1 < len(msgs) else None
        if nxt is not None and nxt.speaker == msgs[ref].speaker and nxt.plan is not None and nxt.plan.split_tail \
                and not nxt.plan.typo_fix:
            text = text + " " + nxt.text
        r = reason_clause(task, text, names)
    if len(r.split()) < 3:
        return rng.choice(fresh_forms(_L(task, "CONCEDE_LINES", CONCEDE_LINES), st, cap=1))
    rx_if = _L(task, "REASON_IF_RE", REASON_IF_RE)
    bank = _L(task, "CONCEDE_REASON_IF", CONCEDE_REASON_IF) if rx_if.search(r) else \
        _L(task, "CONCEDE_REASON", CONCEDE_REASON)
    used = {fold_text(m.text)[:12] for m in msgs}
    forms = [b for b in bank if fold_text(b.format(r=""))[:12] not in used] or bank
    return rng.choice(forms).format(r=r)


def established_lines(task: Task, ideas: list[str], rng: RNG, cap: int = 5) -> list[str]:
    """One paraphrase per established key idea (0.5.3): a correct claim of the idea or one of its card facts'
    chat phrasings, drawn per session; near-identical items dropped; at most `cap`, every activity's KC kept."""
    out: list[str] = []
    kcs: list[str] = []
    rx = _L(task, "EXISTING_LEAD_RE", _EXISTING_LEAD_RE)
    for ki in task.all_key_ideas():
        if ki.id not in ideas:
            continue
        opts = [task.kc(ki.kc).correct_claims[i] for i in ki.claims]
        opts += [_draft_body(rx.sub("", c)) for f in ki.facts for c in task.fact(f).chat]
        opts = [o for o in opts if o and not any(_overlap(o, x) >= 0.6 for x in out)]
        if opts:
            out.append(rng.choice(opts))
            kcs.append(ki.kc)
    while len(out) > cap:     # drop from the KC with most items (each activity keeps at least one point)
        from collections import Counter
        c = Counter(kcs)
        k = max(c, key=lambda x: c[x])
        i = max(j for j, x in enumerate(kcs) if x == k)
        out.pop(i)
        kcs.pop(i)
    return out


def _overlap(a: str, b: str) -> float:
    ta, tb = set(fold_text(a).split()), set(fold_text(b).split())
    return len(ta & tb) / min(len(ta), len(tb)) if ta and tb else 0.0


def join_items(task: Task, items: list[str], rng: RNG) -> str:
    j = rng.choice(_L(task, "CLOSING_JOINS", CLOSING_JOINS))
    return j.join(items)


def closing_agree_line(task: Task, rng: RNG, st=None) -> str:
    """An okay to the closing proposal composed from an ack, a verdict and (sometimes) a "send it" (0.5.3), none
    of whose parts another closing response of this session used."""
    used = set()
    msgs = getattr(st, "messages", None) or []
    for i in (getattr(st, "closing_msgs", None) or [])[1:]:
        if i < len(msgs):
            used |= set(fold_text(msgs[i].text).replace(",", " ").split())
    def fresh(bank):
        ok = [b for b in bank if not b or not (set(fold_text(b).split()) & used)]
        return ok or bank
    ack = rng.choice(fresh(_L(task, "CLOSING_ACKS", CLOSING_ACKS)))
    verdict = rng.choice(fresh(_L(task, "CLOSING_VERDICTS", CLOSING_VERDICTS)))
    send = rng.choice(fresh(_L(task, "CLOSING_SENDS", CLOSING_SENDS)))
    form = rng.randint(0, 3)
    if form == 0:
        parts = [ack, verdict]
    elif form == 1:
        parts = [ack, verdict, send]
    elif form == 2:
        parts = [verdict, send]
    else:
        parts = [ack, send or verdict]
    return ", ".join(p for p in parts if p)


def closing_hedge_line(task: Task, rng: RNG, st=None) -> str:
    """0.5.3: a holder who never gave in okays the final answer with a hedge ("ta, pode ser, mas eu ainda acho
    que nao é bem assim"), composed so no two are alike in a session."""
    used = set()
    msgs = getattr(st, "messages", None) or []
    for i in (getattr(st, "closing_msgs", None) or [])[1:]:
        if i < len(msgs):
            used.add(fold_text(msgs[i].text))
    oks, buts = _L(task, "CLOSING_HEDGE_OK", CLOSING_HEDGE_OK), _L(task, "CLOSING_HEDGE_BUT", CLOSING_HEDGE_BUT)
    for _ in range(8):
        line = f"{rng.choice(oks)}, {rng.choice(buts)}"
        if not any(fold_text(line.split(", ", 1)[1]) in u for u in used):
            return line
    return line
