"""Realism controls: code-side rules every chat runs under. Everything here is deterministic given
the session's "realism" RNG stream:

* session context, fixed once: time of day (manha / tarde / noite), setting ("online": every
  student at their own computer, no physical classroom), per-student laughter propensity;
* final-line checks on the realized text, across ALL speakers and the whole session:
  near-duplicates (token Jaccard / character 4-gram containment, typo- and laughter-robust),
  role words (tutor, bot, AI...), "@" mentions and names of participants who have not written yet,
  physical-classroom talk in an online chat, greetings / meals of another time of day;
* polish: laughter capped by trait and session frequency (never repeatedly as an opener, almost
  never on an on-task content statement) and realistic, sparse typos (at most one per message,
  most messages clean; missing accent, chat abbreviation, doubled / dropped letter, rarely an
  adjacent-key slip; domain terms almost never touched). The LLM is NOT asked for typos.
"""
from __future__ import annotations

import re
import unicodedata

from ..models.rng import RNG

PERIODS = ("manha", "tarde", "noite")
NEAR_DUP_JACCARD = 0.6
NEAR_DUP_CONTAIN = 0.8
NEAR_DUP_MIN_TOKENS = 4
CONTENT_ACTS = {"on_task_claim", "explain", "disagree", "press"}


def _L(task, name, default):
    from .. import lang
    return lang.attr(task, name, default)


# ------------------------------------------------------------------ context
def init_context(st, cfg, rng: RNG) -> None:
    """Fix the session context once."""
    tod = cfg.time_of_day
    if tod is not None:
        tod = fold(tod)
        if tod not in PERIODS:
            raise ValueError(f"time_of_day must be one of {PERIODS}, not {cfg.time_of_day!r}")
    else:
        tod = rng.choice(list(PERIODS))
    st.context = {"time_of_day": tod, "setting": cfg.setting,
                  "offtask_rate": cfg.offtask_rate, "laughter_cap": cfg.laughter_cap,
                  "laughter": {s.id: round(rng.beta(2.0, 4.0), 3) for s in st.students}}


def fold(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


# ------------------------------------------------------------------ laughter
LAUGH_RE = re.compile(r"(?<![a-z])(?:k{2,}|rs{1,3}|(?:ha){2,}h?|(?:he){2,}|lol|lmao+|lmfao)(?![a-z])", re.I)


def has_laugh(text: str) -> bool:
    return bool(LAUGH_RE.search(text))


def strip_laugh(text: str) -> str:
    out = LAUGH_RE.sub("", text)
    out = re.sub(r"\s+([,.!?])", r"\1", out)
    out = re.sub(r"^[\s,.!]+|[\s,]+$", "", out)
    return re.sub(r"\s{2,}", " ", out).strip()


def starts_with_laugh(text: str) -> bool:
    m = LAUGH_RE.search(text.strip())
    return bool(m and m.start() == 0)


# ------------------------------------------------------------------ near-duplicates
def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", fold(LAUGH_RE.sub(" ", text)))]


def _grams(text: str, n: int = 4) -> set[str]:
    s = " ".join(_tokens(text))
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def similarity(a: str, b: str) -> tuple[float, float]:
    ta, tb = set(_tokens(a)), set(_tokens(b))
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 1.0
    ga, gb = _grams(a), _grams(b)
    cont = len(ga & gb) / len(ga) if ga and gb else 0.0   # share of the CANDIDATE found in the earlier line
    return jac, cont


def whole_lines(st) -> list:
    """The chat's messages with each split thought (0.5.2: a line typed as two messages) joined back into
    one line, so repetition checks compare whole statements."""
    from types import SimpleNamespace
    msgs = list(st.messages)
    tails = {}
    msgs = [m for m in msgs if not (m.plan is not None and m.plan.typo_fix)]    # ("*superficie")
    for i, m in enumerate(msgs):
        if m.plan is not None and m.plan.split_tail:
            j = next((k for k in range(i - 1, -1, -1) if msgs[k].speaker == m.speaker), None)
            if j is not None:
                tails[j] = m
    out = []
    for i, m in enumerate(msgs):
        if m.plan is not None and m.plan.split_tail and any(t is m for t in tails.values()):
            continue
        text = m.text + (" " + tails[i].text if i in tails else "")
        out.append(SimpleNamespace(id=m.id, speaker=m.speaker, kind=m.kind, plan=m.plan, text=text, t=m.t))
    return out


def near_dup(text: str, st, names: dict | None = None, plan=None) -> dict | None:
    """Session-wide near-duplicate against every earlier line (any speaker; split thoughts joined). A
    concession may quote the reason it gives in (its anchor, 0.5.3)."""
    toks = _tokens(text)
    if len(toks) < NEAR_DUP_MIN_TOKENS:
        return None
    strip_names = {fold(n) for n in (names or {}).values()}
    core = " ".join(t for t in toks if t not in strip_names)
    quoted = set()
    if plan is not None and plan.concedes and plan.anchor_turn is not None:
        quoted = {plan.anchor_turn}
        # the anchor's split tail, too
        quoted |= {m.id for m in st.messages[plan.anchor_turn + 1:plan.anchor_turn + 3]
                   if m.plan is not None and m.plan.split_tail}
    for m in reversed(whole_lines(st)):
        if m.id in quoted:
            continue
        if len(_tokens(m.text)) < NEAR_DUP_MIN_TOKENS:
            continue
        other = " ".join(t for t in _tokens(m.text) if t not in strip_names)
        jac, cont = similarity(core, other)
        if jac >= NEAR_DUP_JACCARD or cont >= NEAR_DUP_CONTAIN:
            return {"reason": "near_duplicate_session", "prev": m.text, "prev_id": m.id,
                    "jaccard": round(jac, 3), "containment": round(cont, 3)}
    return None


# ------------------------------------------------------------------ world consistency
PHYSICAL_RE_EN = re.compile(r"\b(teacher|classroom|recess|cafeteria|canteen|next door|uniform|the room|in here)\b", re.I)
PERIOD_WORDS_EN = {
    "manha": r"\b(good afternoon|good evening|good night|dinner|tonight)\b",
    "tarde": r"\b(good morning|good evening|good night|breakfast|dinner|tonight)\b",
    "noite": r"\b(good morning|good afternoon|breakfast|lunch)\b",
}
ROLE_WORDS_EN = r"\b(facilitator\w*|tutor\w*|teacher bot|the bot|a bot|chatbot|robot|the ai|an ai|artificial intelligence)\b"


def participant_mention_violation(text: str, st, task) -> dict | None:
    """Students never use role words (tutor, facilitator, bot, AI), never "@"-mention anyone, and never
    name an external participant who has not written in the chat yet."""
    low = fold(text)
    if "@" in text:
        return {"reason": "participant_mention", "what": "@", "prev": text}
    m = re.search(_L(task, "ROLE_WORDS_RE", ROLE_WORDS_EN), low)
    if m:
        return {"reason": "participant_mention", "what": m.group(0), "prev": text}
    for pid, d in (getattr(st, "participants", None) or {}).items():
        if d.get("first_spoke_t") is None and re.search(rf"\b{re.escape(fold(d['name']))}\b", low):
            return {"reason": "participant_mention", "what": "silent_participant", "prev": text}
    return None


def world_violation(text: str, st, task) -> dict | None:
    ctx = st.context
    low = fold(text)
    hit = participant_mention_violation(text, st, task)
    if hit is not None:
        return hit
    if ctx.get("setting", "online") == "online":
        rx = _L(task, "PHYSICAL_RE", PHYSICAL_RE_EN)
        if rx.search(low):
            return {"reason": "physical_classroom", "prev": text}
    pw = _L(task, "PERIOD_WORDS", PERIOD_WORDS_EN).get(ctx.get("time_of_day"))
    if pw and re.search(pw, low):
        return {"reason": "wrong_time_of_day", "prev": text}
    return None


OWN_REPEAT_OVERLAP = 0.75     # share of the shorter line's content words (numbers included) found in the other
OWN_REPEAT_OVERLAP_ASKED = 0.9   # a restatement the student was asked for / that answers a dispute: other words
OWN_REPEAT_MIN_WORDS = 3


def _permitted_restatement(plan, st, sid: str) -> bool:
    if plan is None:
        return False
    if plan.restate or plan.answers_ping or plan.replies_to_external:
        return True
    a = plan.anchor_turn
    if a is not None and 0 <= a < len(st.messages):
        am = st.messages[a]
        return am.speaker != sid and am.plan is not None and (
            am.plan.act in ("question", "press", "disagree") or bool(am.plan.addresses_misconception))
    return False


def own_content_repeat(text: str, st, sid: str, task, names: dict | None = None, plan=None) -> dict | None:
    """0.5.2: a student states each fact once. Same speaker, whole session: a line whose content words
    (stemmed, numbers included) are mostly those of one of their earlier lines is a repeat, unless the
    restatement was asked for or answers a dispute -- and then it must use other words."""
    from .content import _match, content_words
    if plan is not None and (plan.closing is not None or plan.draft is not None or plan.tag or plan.opener
                             or plan.concedes):
        return None
    # (a) a card fact the speaker already stated, tacked onto a line whose plan does not voice it
    from .content import plan_ideas
    own = [m for m in whole_lines(st) if m.speaker == sid and m.kind == "student" and not (
        m.plan is not None and (m.plan.closing is not None or m.plan.draft is not None))]
    said = {f for m in own for f in task.facts_mentioned(m.text)} | {
        f for m in own if m.plan is not None and m.plan.claim == "fact" for f in m.plan.fact_ids}
    mine = set(plan.fact_ids) if plan is not None else set()
    if plan is not None:
        ideas = plan_ideas(task, plan)
        mine |= {f for ki in task.all_key_ideas() if ki.id in ideas for f in ki.facts}
        if plan.claim == "correct" and plan.kc and plan.claim_idx is not None:
            mine |= set(task.facts_mentioned(task.kc(plan.kc).correct_claims[plan.claim_idx]))
        if plan.family == "counterexample" and plan.addresses_misconception:
            mine |= {f for c in task.misconception(plan.addresses_misconception).counterexample
                     for f in task.facts_mentioned(c)}
    extra = (set(task.facts_mentioned(text)) - mine) & said
    if extra and (plan is None or plan.claim != "misconception"):
        prev = next((m for m in reversed(own) if set(task.facts_mentioned(m.text)) & extra), own[-1] if own else None)
        return {"reason": "own_content_repeat", "prev": prev.text if prev else "", "prev_id": prev.id if prev else None,
                "facts": sorted(extra)}
    # (b) the same content words as one of the speaker's earlier lines (numbers included)
    drop = {fold(n) for n in (names or {}).values()}
    cw = {w for w in content_words(text, task.language) if w not in drop}
    if len(cw) < OWN_REPEAT_MIN_WORDS:
        return None
    from .content import same_claim
    permitted = _permitted_restatement(plan, st, sid)
    for m in reversed(whole_lines(st)):
        if m.speaker != sid or m.kind != "student" or (m.plan is not None and (
                m.plan.closing is not None or m.plan.draft is not None)):
            continue
        ow = {w for w in content_words(m.text, task.language) if w not in drop}
        if len(ow) < OWN_REPEAT_MIN_WORDS:
            continue
        ov = len(_match(cw, ow)) / min(len(cw), len(ow))
        # an asked-for restatement may repeat the plan's OWN content (in other words), nothing else
        same = permitted and m.plan is not None and plan is not None and (
            same_claim(task, m.plan, plan) or bool(set(m.plan.fact_ids) & set(plan.fact_ids)))
        if ov >= (OWN_REPEAT_OVERLAP_ASKED if same else OWN_REPEAT_OVERLAP):
            return {"reason": "own_content_repeat", "prev": m.text, "prev_id": m.id, "overlap": round(ov, 3)}
    return None


# 0.5.3: a thought is typed as two messages only at a clause boundary: before a connector that opens a new
# clause ("... | entao ...", "... | por isso ...", "... | tipo ...", "... | mas ..."), never before a bare
# "e" / "and" (a coordinated verb phrase: "a amilase desnatura | e para de funcionar"), never at an accented
# word that folds onto a connector ("é" is not "e"), and never leaving a dangling word at the end of the head
# (a relative pronoun / preposition / article / copula: "... em maltose que" | "é um açucar").
SPLIT_CONNECTORS = {"pt-BR": ("entao", "por isso", "tipo", "mas", "porque", "pq", "so que", "ai"),
                    "en": ("so", "because", "cause", "thats why", "but", "like")}
SPLIT_DANGLING = {
    "pt-BR": frozenset("""que de do da dos das em no na nos nas num numa a o as os um uma uns umas pra pro pras pros para
        com por pelo pela pelos pelas e ou se mais muito mt é eh ta tá esta está sao são tem ter cuja cujo onde quando
        como qual quais quem ao aos à às sem entre sobre ate até meu minha seu sua nosso nossa tipo""".split()),
    "en": frozenset("""the a an of to that which who whom whose with for in on at by from and or is are was were be
        been more very than as into about its their his her my your our like so""".split()),
}
SPLIT_BAD_START = {"pt-BR": frozenset({"é", "eh", "que", "de", "do", "da", "e", "ou", "ta", "tá", "sao", "são", "cuja",
                                       "cujo", "mais", "menos", "nao", "não", "tipo"}),
                   "en": frozenset({"is", "are", "was", "that", "which", "who", "of", "to", "and", "or", "than", "not"})}
SPLIT_MIN_HEAD = 4      # words before the cut
SPLIT_MIN_TAIL = 2      # words after it


def _bare(w: str) -> str:
    return w.lower().strip(",.;:!?\"'()")


def split_point(text: str, task) -> tuple[str, str | None]:
    """Where a content line can be typed as two messages (0.5.3): before a clause connector, with >= 4
    words before and >= 2 after, the head carrying the line's content and not ending on a dangling word.
    Accents count: "é" never matches the connector "e" (which is not a split point anyway)."""
    words = text.split()
    if len(words) < SPLIT_MIN_HEAD + SPLIT_MIN_TAIL:
        return text, None
    lg = "pt-BR" if getattr(task, "language", "en") == "pt-BR" else "en"
    cons = SPLIT_CONNECTORS[lg]
    dangling = SPLIT_DANGLING[lg]
    cands = []
    for i in range(SPLIT_MIN_HEAD, len(words) - SPLIT_MIN_TAIL + 1):
        raw = _bare(words[i])
        if raw != fold(raw) and raw not in ("então", "aí", "só"):
            continue                         # an accented word ("é", "já") is never a connector
        w = fold(raw)
        two = fold(" ".join(_bare(x) for x in words[i:i + 2]))
        comma = words[i - 1].endswith(",") and raw.isalpha() and raw not in SPLIT_BAD_START[lg] \
            and raw not in dangling and len(words) - i >= 3
        if not (w in cons or two in cons or comma):
            continue
        if w == "tipo" and len(words) - i < 3:
            continue                         # "... tipo isso": a tag, not a new clause
        last = _bare(words[i - 1])
        if last in dangling or fold(last) in dangling or not last:
            continue
        cands.append(i)
    from .content import content_words
    kp = set(task.facts_mentioned(text)) | set(task.content_kcs(text))

    def keeps(i):     # the first message carries the content: >= 2 content words and the line's keyphrases
        head = " ".join(words[:i])
        if len(content_words(head, task.language)) < 2:
            return False
        return not kp or bool(set(task.facts_mentioned(head)) | set(task.content_kcs(head)))
    cands = [i for i in cands if keeps(i)]
    if not cands:
        return text, None
    i = min(cands, key=lambda k: abs(k - len(words) / 2))
    head = " ".join(words[:i]).rstrip(",")
    tail = " ".join(words[i:])
    if len(tail.split()) < SPLIT_MIN_TAIL:
        return text, None
    return head, tail


def bad_split(head: str, tail: str, task) -> bool:
    """A cut mid-clause (the audit's definition): the head ends on a dangling word or the tail starts with one."""
    lg = "pt-BR" if getattr(task, "language", "en") == "pt-BR" else "en"
    hw, tw = [_bare(w) for w in head.split()], [_bare(w) for w in tail.split()]
    bad_start = SPLIT_BAD_START[lg] - {"tipo", "nao", "não", "not"}
    if tw and tw[0].startswith("*"):
        return False                 # a typo correction ("*superficie"), not a split
    return bool((hw and hw[-1] in SPLIT_DANGLING[lg]) or (tw and tw[0] in bad_start))


# ------------------------------------------------------------------ 0.5.3: address, leads, re-asks, closings
PLURAL_ADDRESS_RE_EN = re.compile(r"\b(?:guys|everyone|everybody|anyone|anybody|nobody|no one|you all|y ?all)\b")
DYAD_REWRITES_EN = [
    (r"\b(hi|hey|hello|yo|morning|good morning|good afternoon|good evening)\s+(?:guys|everyone|all)\b", r"\1 {name}"),
    (r"^(?:guys|ok guys|hey guys)[,!]?\s+", "{name} "), (r"[,\s]+(?:guys|everyone)\s*(?=[?!.]*$)", ""),
    (r"\byou guys\b", "you"), (r"\bdoes anyone know\b", "do you know"), (r"\banyone know\b", "do you know"),
    (r"\bno one knows\b", "you dont know"), (r"\bnobody knows\b", "you dont know"), (r"\bwho else is\b", "are you"),
    (r"^(hello\?*\s*)anyone\b", r"\1{name}"), (r"^anyone\s*\?", "{name}?"), (r"\banyone\b", "you"),
    (r"\banybody\b", "you"), (r"\beveryone\b", "you"), (r"\bguys\b", "{name}"), (r"\byou all\b", "you"),
]
LEAD_WORDS_EN = ("wait", "guys", "so", "ok so", "look", "hold on", "um", "like", "ok")
REPEAT_QUESTION_OVERLAP = 0.5


def is_dyad(st) -> bool:
    """Only two people in the chat (two students, no external participant who wrote)."""
    roster = getattr(st, "roster", None)
    if not roster or len(roster) != 2:
        return False
    try:
        return not st.spoken_externals()
    except AttributeError:
        return True


def plural_address(text: str, task) -> bool:
    """Plural address ("galera", "gente", "vcs", "alguem", "guys", "anyone"); "a gente" (= we) is not one."""
    return bool(_L(task, "PLURAL_ADDRESS_RE", PLURAL_ADDRESS_RE_EN).search(fold(text)))


def singularize(text: str, task, partner: str) -> str:
    """A dyad's line addresses the one partner: plural address -> singular / the partner's name (0.5.3)."""
    if not plural_address(text, task):
        return text
    out = text
    name = (partner or "").lower()
    for rx, rep in _L(task, "DYAD_REWRITES", DYAD_REWRITES_EN):
        out = re.sub(rx, rep.replace("{name}", name), out, flags=re.I)
    out = re.sub(r"\s{2,}", " ", out).strip(" ,")
    return out if len(out.split()) >= 1 else text


def strip_repeated_lead(text: str, opener: str, task) -> str:
    """The content line after a one-word opener never repeats the lead ("olha" | "olha, pelo que eu li..." ->
    "pelo que eu li..."): a leading lead word of the bank, or any word of the opener, is dropped."""
    bank = {fold(w) for w in _L(task, "LEAD_WORDS", LEAD_WORDS_EN)}
    leads = bank | {" ".join(_tokens(opener))} if " ".join(_tokens(opener)) in bank else bank
    words = text.split()
    k = 0
    while k < len(words) - 2:
        two = fold(" ".join(words[k:k + 2])).strip(",.!?:")
        one = fold(words[k]).strip(",.!?:")
        if two in leads:
            k += 2
        elif one in leads:
            k += 1
        else:
            break
    return " ".join(words[k:]).lstrip(" ,") if k else text


def _content_q(text: str, task) -> set[str]:
    from .content import content_words
    return set(content_words(text, task.language))


def question_repeat(text: str, st, sid: str, task, plan=None) -> dict | None:
    """0.5.3: the same student never asks an equivalent question twice (content words of the shorter one mostly
    shared); a follow-up may come back to an unanswered question once, at least 2 minutes later."""
    if plan is None or plan.act != "question" or plan.tag or plan.nudge or plan.step_q or plan.closing or plan.draft \
            or getattr(plan, "step_move", False):
        return None
    if "?" not in text and not plan.asked_fact and not plan.intent:
        return None
    qw = _content_q(text, task)
    if len(qw) < 2:
        return None
    from .content import _match
    for m in reversed(st.messages):
        if m.speaker != sid or m.plan is None or m.plan.act != "question" or m.plan.tag or m.plan.nudge:
            continue
        ow = _content_q(m.text, task)
        if len(ow) < 2:
            continue
        if len(_match(qw, ow)) / min(len(qw), len(ow)) >= REPEAT_QUESTION_OVERLAP:
            if plan.follow_up and st.t - m.t >= 120 and not any(
                    x.speaker == sid and x.plan is not None and x.plan.follow_up and x.plan.anchor_turn == plan.anchor_turn
                    for x in st.messages):
                return None
            return {"reason": "question_repeat", "prev": m.text, "prev_id": m.id}
    return None


def closing_repeat(text: str, st, task, plan=None) -> dict | None:
    """0.5.3: no two closing responses of a session share a phrase (laughter / typos aside)."""
    if plan is None or plan.closing not in ("agree", "adjust") or plan.brief:
        return None
    mine = " ".join(_tokens(strip_laugh(text)))
    mw = set(mine.split())
    for i in getattr(st, "closing_msgs", [])[1:]:
        if i >= len(st.messages):
            continue
        m = st.messages[i]
        if m.plan is not None and m.plan.brief:
            continue
        other = " ".join(_tokens(strip_laugh(m.text)))
        ow = set(other.split())
        if mine == other or (mw and ow and len(mw & ow) / min(len(mw), len(ow)) >= 0.6):
            return {"reason": "closing_repeat", "prev": m.text, "prev_id": m.id}
    return None


def offtask_named(text: str, st, sid: str, plan=None, names: dict | None = None) -> dict | None:
    """0.5.3: an off-task remark out of the blue (not a reply) is general, never addressed to a classmate by
    name with a premise nobody gave ("feliz aniversario, pedro")."""
    if plan is None or plan.act != "off_task" or plan.anchor_turn is not None or not names:
        return None
    low = fold(text)
    for pid, nm in names.items():
        if pid != sid and nm and re.search(rf"\b{re.escape(fold(nm).split()[0])}\b", low):
            return {"reason": "offtask_named", "prev": text}
    return None


def check(text: str, st, sid: str, task, names: dict | None = None, plan=None) -> dict | None:
    hit = (world_violation(text, st, task) or near_dup(text, st, names, plan)
           or own_content_repeat(text, st, sid, task, names, plan))
    if hit is not None:
        return hit
    if is_dyad(st) and plural_address(text, task):
        other = next((x for x in st.roster if x != sid), "")
        return {"reason": "plural_address", "prev": text, "name": (names or {}).get(other, other)}
    return (question_repeat(text, st, sid, task, plan) or closing_repeat(text, st, task, plan)
            or offtask_named(text, st, sid, plan, names))


def rephrase_note(hit: dict, task) -> str:
    notes = _L(task, "REALISM_NOTES", REALISM_NOTES_EN)
    return notes.get(hit["reason"], notes["near_duplicate_session"]).format(prev=hit.get("prev", ""),
                                                                            name=hit.get("name", ""))


REALISM_NOTES_EN = {
    "plural_address": ("REWRITE: there are only the two of you in this chat: talk to {name} (singular: you), never "
                       "'guys', 'everyone' or 'anyone'."),
    "question_repeat": ("DO NOT ASK AGAIN: you already asked this (\"{prev}\"). Say something else, or react "
                        "briefly."),
    "closing_repeat": ("SAY IT DIFFERENTLY: someone already answered the proposal like that (\"{prev}\"); okay it in "
                       "your own different words, briefly."),
    "offtask_named": "REWRITE: an off-topic remark to the whole chat, without addressing a classmate by name.",
    "own_content_repeat": ("DO NOT REPEAT YOURSELF: you already said this (\"{prev}\"). Say something else, or only "
                           "react briefly; never state the same fact again."),
    "near_duplicate_session": ("SAY IT DIFFERENTLY: your draft was almost the same as an earlier message in the "
                               "chat (\"{prev}\"). Say something with clearly different words, or shorter."),
    "participant_mention": ("REWRITE: talk only to the people who have written in this chat, by name, without '@'; "
                            "there is no teacher, tutor or AI here."),
    "physical_classroom": "REWRITE: this is an online chat, everyone at their own computer; no classroom or teacher in the room.",
    "wrong_time_of_day": "REWRITE: keep to the time of day of this session (see SESSION CONTEXT).",
}


def ok_line(text: str, st, task) -> bool:
    """Template-bank filter (greetings, off-task lines): consistent with the session context."""
    return st is None or world_violation(text, st, task) is None


# ------------------------------------------------------------------ polish: laughter + typos
LAUGHS_EN = ["lol", "haha", "lmao"]
LAUGH_ACTS = {"off_task", "agree", "greeting"}


def _add_laughter(text: str, plan, s, st, task, rng: RNG, info: dict) -> str:
    """Social lines (off-task, agreeing, greeting) sometimes carry a laugh, under the session cap and
    the student's propensity (the pilot chats laugh in ~6% of messages; fillers are capped, so the
    laughter that used to come with them is attached here)."""
    if has_laugh(text) or plan.act not in LAUGH_ACTS or plan.opener:
        return text
    stu = [m for m in st.messages if m.kind == "student"]
    share = sum(has_laugh(m.text) for m in stu) / max(1, len(stu))
    if share >= min(st.context.get("laughter_cap", 0.15), 0.08):
        return text
    if rng.random() >= 0.5 * st.context.get("laughter", {}).get(s.id, 0.3):
        return text
    info["laughter"] = "added"
    return f"{text} {rng.choice(_L(task, 'LAUGHS', LAUGHS_EN))}"


def polish(text: str, plan, s, st, task, rng: RNG) -> tuple[str, dict]:
    info: dict = {}
    text = _laughter(text, plan, s, st, task, rng, info)
    if "laughter" not in info:
        text = _add_laughter(text, plan, s, st, task, rng, info)
    typo = _typo(text, plan, s, task, rng, info)
    if typo != text and world_violation(typo, st, task) is not None:   # a typo never creates a violation
        info.pop("typo", None)
        return text, info
    return typo, info


def _content_statement(plan) -> bool:
    return plan.act in CONTENT_ACTS or plan.closing is not None or plan.draft is not None or (
        plan.claim in ("correct", "fact", "misconception") and plan.act != "agree")


def _laughter(text: str, plan, s, st, task, rng: RNG, info: dict) -> str:
    if not has_laugh(text):
        return text
    stu = [m for m in st.messages if m.kind == "student"]
    share = sum(has_laugh(m.text) for m in stu) / max(1, len(stu))
    trait = st.context.get("laughter", {}).get(s.id, 0.3)
    cap = st.context.get("laughter_cap", 0.15)
    allowed = share < cap and rng.random() < trait
    if _content_statement(plan) and rng.random() < 0.9:
        allowed = False
    core = strip_laugh(text)
    if not core:  # a pure-laughter message
        if allowed:
            return text
        bank = [b for b in _L(task, "FILLER", {}).get("backchannel", ["ok", "hmm"]) if not has_laugh(b)] or ["ok"]
        info["laughter"] = "replaced"
        return rng.choice(bank)
    if not allowed:
        info["laughter"] = "stripped"
        return core
    if starts_with_laugh(text) and any(starts_with_laugh(m.text) for m in stu[-5:]):
        m = LAUGH_RE.search(text.strip())
        info["laughter"] = "moved_to_end"
        return f"{core} {m.group(0)}"
    return text


ABBREV_PT = {"voce": "vc", "você": "vc", "voces": "vcs", "vocês": "vcs", "também": "tbm", "tambem": "tb",
             "porque": "pq", "por que": "pq", "não": "n", "que": "q", "está": "ta", "tá": "ta", "beleza": "blz",
             "verdade": "vdd", "muito": "mt", "hoje": "hj", "mesmo": "msm", "quando": "qnd"}
ABBREV_EN = {"you": "u", "because": "bc", "though": "tho", "really": "rly", "okay": "ok", "don't": "dont",
             "please": "pls", "your": "ur", "people": "ppl", "thanks": "thx"}
KEYS = {"a": "s", "s": "a", "e": "r", "r": "e", "i": "o", "o": "i", "n": "m", "m": "n", "t": "r", "u": "i", "d": "s"}


def _domain_terms(task) -> set[str]:
    out = set()
    for f in task.facts:
        for k in f.keyphrases:
            out.update(w for w in re.findall(r"[a-z0-9]+", fold(k)) if len(w) > 3)
    for k in task.kcs:
        for kp in k.keyphrases:
            out.update(w for w in re.findall(r"[a-z0-9]+", fold(kp)) if len(w) > 3)
    return out


def _typo(text: str, plan, s, task, rng: RNG, info: dict) -> str:
    words = text.split(" ")
    p = min(0.3, 2.5 * s.traits.typo_rate)
    if len(words) < 3:
        p *= 0.5
    if not rng.bernoulli(p):
        return text
    domain = _domain_terms(task)
    abbrev = _L(task, "TYPO_ABBREV", ABBREV_EN)
    kinds = ["accent", "abbrev", "double", "drop", "key"]
    weights = [4.0, 3.0, 1.0, 1.0, 0.3]
    for _ in range(6):
        k = rng.weighted(kinds, weights)
        idx = list(range(len(words)))
        rng.shuffle(idx)
        for i in idx:
            w = words[i]
            core = re.sub(r"[^\wÀ-ÿ']", "", w)
            if not core:
                continue
            if fold(core) in domain and rng.random() < 0.9:
                continue
            new = _apply(k, core, abbrev, rng)
            if new and new != core:
                words[i] = w.replace(core, new, 1)
                info["typo"] = k
                info["typo_word"] = [core, new]
                return " ".join(words)
    return text


def _apply(kind: str, w: str, abbrev: dict, rng: RNG) -> str | None:
    if kind == "accent":
        f = "".join(c for c in unicodedata.normalize("NFD", w) if unicodedata.category(c) != "Mn")
        return f if f != w else None
    if kind == "abbrev":
        return abbrev.get(w.lower())
    if not w.isalpha() or len(w) < 5:
        return None
    i = rng.randint(1, len(w) - 2)
    if kind == "double":
        return w[:i] + w[i] + w[i:]
    if kind == "drop":
        return w[:i] + w[i + 1:]
    c = w[i].lower()
    return w[:i] + KEYS[c] + w[i + 1:] if c in KEYS else None


# ------------------------------------------------------------------ framing leads (0.5.0)
# How a student introduces something from their material ("mine says...", "no meu card diz que...").
# One lead form may be used at most cfg.lead_form_max times per session, and the word "card" (which
# exposes the jigsaw mechanism) at most cfg.card_lead_max times; a capped lead is dropped (or swapped
# for an unused natural one), and a capped "card" becomes a neutral word.
LEAD_PREFIX_EN = r"^(?:(?:ok|so|wait|like|yeah|hmm|oh|and|but)[,\s]+)*"
LEAD_FORMS_EN = [
    ("card", r"(?:(?:on|in)\s+)?(?:my|the)\s+card(?:\s+(?:says|has))?(?:\s+that)?\s*[:,]?\s*"),
    ("mine_says", r"(?:on\s+)?mine\s+says(?:\s+that)?\s+"),
    ("sheet", r"my\s+sheet\s+(?:has|says)(?:\s+that)?\s*"),
    ("it_says", r"(?:it\s+says(?:\s+here)?|says\s+here)(?:\s+that)?\s*"),
    ("apparently", r"apparently,?\s*"),
    ("i_read", r"i\s+read\s+that\s+"),
    ("material", r"the\s+material\s+says(?:\s+that)?\s+"),
    ("has_to_do", r"i\s+think\s+it\s+has\s+to\s+do\s+with\s+this:?\s*"),
    ("from_what_i_read", r"from\s+what\s+i\s+read,?\s*"),
]
CARD_REPLACE_EN = [(r"\b(on|in) my card\b", r"\1 mine"), (r"\bmy card\b", "my sheet"), (r"\bthe card\b", "the sheet"),
                   (r"\bcards?\b", "sheet")]
NATURAL_LEADS_EN = ["i read that ", "it says here ", "apparently ", "from what i read, ", "the material says ",
                    "i think it has to do with this: "]
NEUTRAL_LEAD_EN = "i think "   # not a framing lead: used when every natural lead is at its cap


def _fold_keep(text: str) -> str:
    """Lowercase, accents stripped, SAME LENGTH as `text` (indices map back to the original)."""
    out = []
    for c in text:
        lc = c.lower()
        base = unicodedata.normalize("NFD", lc)[0] if lc else lc
        out.append(base if len(lc) == 1 else c)
    return "".join(out)


def lead_form(text: str, task) -> tuple[str, int] | None:
    """(form key, end index of the lead in `text`) of the framing lead the line opens with, if any."""
    low = _fold_keep(text)
    pre = re.match(_L(task, "LEAD_PREFIX", LEAD_PREFIX_EN), low)
    start = pre.end() if pre else 0
    for key, rx in _L(task, "LEAD_FORMS", LEAD_FORMS_EN):
        m = re.compile(rx).match(low, start)
        if m:
            return key, m.end()
    return None


def has_card(text: str) -> bool:
    return bool(re.search(r"\bcards?\b", fold(text)))


def cap_framing(text: str, st, task, cfg, rng: RNG) -> tuple[str, dict | None]:
    stu = [m for m in st.messages if m.kind == "student"]
    lf = lead_form(text, task)
    info: dict = {}
    if lf is not None:
        key, end = lf
        used = sum(1 for m in stu if (lead_form(m.text, task) or ("",))[0] == key)
        cap = cfg.card_lead_max if key == "card" else cfg.lead_form_max
        if used >= cap:
            pre = re.match(_L(task, "LEAD_PREFIX", LEAD_PREFIX_EN), _fold_keep(text))
            head = text[:pre.end()] if pre else ""
            body = text[end:].lstrip(" ,:")
            def uses(l):
                k = (lead_form(l + "x", task) or ("",))[0]
                return sum(1 for m in stu if (lead_form(m.text, task) or ("",))[0] == k) if k else 0
            fresh = [l for l in _L(task, "NATURAL_LEADS", NATURAL_LEADS_EN)
                     if (lead_form(l + "x", task) or ("",))[0] not in ("card", key) and uses(l) < cfg.lead_form_max]
            lead = rng.choice(fresh) if fresh else _L(task, "NEUTRAL_LEAD", NEUTRAL_LEAD_EN)
            new = (head + lead + body).strip()
            if len(new.split()) >= 2:
                info = {"lead": key, "draft": text}
                text = new
    if has_card(text) and sum(1 for m in stu if has_card(m.text)) >= cfg.card_lead_max:
        draft = text
        for rx, rep in _L(task, "CARD_REPLACE", CARD_REPLACE_EN):
            text = re.sub(rx, rep, text, flags=re.I)
        if text != draft:
            info.setdefault("draft", draft)
            info["card"] = True
    if info:
        info["final"] = text
        return text, info
    return text, None


# ------------------------------------------------------------------ content-free lines (0.5.0)
def content_free_text(text: str, task) -> bool:
    """A line with no content: pure laughter, or (laughter aside) one of the filler / short-reply
    template lines ("ok", "sla", "eu também kkk")."""
    core = " ".join(_tokens(strip_laugh(text)))
    if not core:
        return True
    from .realizer import FILLER, OFFTASK_REPLY
    F = _L(task, "FILLER", FILLER)
    bank = [l for k in ("backchannel", "agree", "idk", "refusal", "off_task_atom") for l in F.get(k, [])]
    bank += _L(task, "OFFTASK_REPLY", OFFTASK_REPLY)
    return core in {" ".join(_tokens(strip_laugh(l))) for l in bank}
