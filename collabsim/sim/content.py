"""Code-side content helpers shared by the planner, the realizers, the engine and the updater.
Everything here reads the code plans and the visible chat; no LLM.

* Visible facts. A card fact is VISIBLE to a student if it was mentioned (by plan or by
  keyphrase) in one of the last `VISIBLE_ECHO_WINDOW` messages, written by a peer or by an
  external participant. Echoing a visible fact is not a leak; the speaker demonstrably read it, so
  the engine adds it to their known facts when they use it.
* Leak triggers. A keyphrase of a fact the speaker does not hold is classified as
  `visible_echo` (the fact is visible in the chat), `own_material` (the keyphrase also occurs in a
  card fact the speaker holds, e.g. f_rate's "more surface"), or `true_leak` (nobody said it and the
  speaker does not hold it). Only true leaks are blocked.
* Restatement keys and key ideas: a claim's identity is the set of task key ideas it
  covers plus its raw key (fact id / KC correct-claim index / misconception id) and its correctness.
"""
from __future__ import annotations

import re as _re

from ..models.message import ContentPlan, Message
from ..models.session import SessionState
from ..models.student import StudentState
from ..models.task import Task, fold

VISIBLE_ECHO_WINDOW = 12   # messages (students + external participants), most recent first


def visible_messages(st: SessionState, sid: str, window: int = VISIBLE_ECHO_WINDOW,
                     before: int | None = None) -> list[Message]:
    msgs = st.messages if before is None else st.messages[:before]
    return msgs[-window:]


def message_facts(task: Task, m: Message) -> set[str]:
    out = set(task.facts_mentioned(m.text))
    if m.plan is not None:
        out |= set(m.plan.fact_ids)
    return out


def visible_fact_sources(task: Task, st: SessionState | None, sid: str,
                         window: int = VISIBLE_ECHO_WINDOW) -> dict[str, Message]:
    """fact id -> the most recent visible message (peer or external participant) that mentioned it."""
    if st is None:
        return {}
    out: dict[str, Message] = {}
    for m in visible_messages(st, sid, window):
        if m.speaker == sid:
            continue
        for f in message_facts(task, m):
            out[f] = m
    return out


def last_mention(task: Task, st: SessionState, fid: str, exclude: str | None = None) -> Message | None:
    """Most recent message (any distance) mentioning fact `fid`, optionally excluding a speaker."""
    for m in reversed(st.messages):
        if m.speaker != exclude and fid in message_facts(task, m):
            return m
    return None


def _own_material(task: Task, s: StudentState, plan: ContentPlan | None) -> str:
    held = set(s.known_facts) | set(plan.fact_ids if plan else [])
    return " ".join([fold(task.fact(f).text) for f in held] + [fold(c) for f in held for c in task.fact(f).chat])


def leak_triggers(task: Task, s: StudentState, plan: ContentPlan | None, text: str,
                  st: SessionState | None = None) -> list[dict]:
    """Every fact keyphrase in `text` of a fact the speaker does not hold (and the plan does not
    voice), classified as true_leak / visible_echo / own_material, with diagnostics."""
    low = fold(text)
    lang = task.language
    allowed = set(s.known_facts) | set(plan.fact_ids if plan else [])
    vis = visible_fact_sources(task, st, s.id)
    own = _own_material(task, s, plan)
    out = []
    for f in task.facts:
        if f.id in allowed:
            continue
        hits = [fold(k) for k in f.keyphrases if fold(k) in low]
        if not hits:
            continue
        src = vis.get(f.id)
        said = last_mention(task, st, f.id, exclude=s.id) if st is not None else None
        rec = {"fact_id": f.id, "kc": f.kc, "keyphrases": hits,
               "speaker_exposed": f.id in s.known_facts,
               "said_in_chat": None if said is None else {
                   "msg": said.id, "speaker": said.speaker, "t": round(said.t, 2),
                   "turns_ago": len(st.messages) - said.id - 1,
                   "speaker_was_reader": s.id in (said.labels.get("readers") or [])}}
        if src is not None:
            rec["type"] = "visible_echo"
        elif all(k in own for k in hits):
            rec["type"] = "own_material"
        elif all(not asserts_in(text, k, f, lang) for k in hits):
            rec["type"] = "question"   # asked, not asserted (allowed)
        else:
            rec["type"] = "true_leak"
        out.append(rec)
    return out


# ------------------------------------------------------------------ questions
# A question does not assert a fact, so a keyphrase inside a question is not a leak, unless the
# question PRESUPPOSES the fact. Heuristic (simple by design, documented in the README):
#   interrogative sentence = ends with "?" or starts with a question word / auxiliary
#     (what, how, why, is there, does, do, can, should, ...), and is not a tag question
#     ("..., right?", "..., isnt it?", "..., yeah?": those assert and ask for confirmation) and not
#     uptalk (declarative word order + "?": "i think its body temp?", "maybe it denatures?");
#   presupposes fact f = it contains a keyphrase of f AND a causal/explanatory frame
#     (why / how come / since / because / what makes), e.g. "why does it denature above 60?";
#     or it spells the fact out with >= 2 distinct keyphrases of f ("does it denature above 60?").
# "is there a best temperature for it?" (one keyphrase, no causal frame) -> not a leak.
QUESTION_START = ("what", "whats", "what's", "how", "why", "is there", "are there", "is it", "is that",
                  "are", "does", "do", "did", "can", "could", "should", "would", "will", "where", "wheres",
                  "when", "which", "who", "wait what", "so what", "so how", "so why", "hmm what")


def sentences(text: str) -> list[str]:
    """Split a chat line into sentences/clauses; '?' stays attached to its sentence."""
    parts = _re.split(r"(?<=[?.!])\s+|\s*\n\s*", text.strip())
    return [p.strip() for p in parts if p.strip()]


_TAG_Q = _re.compile(r"(?:,|\s)\s*(?:right|yeah|ya|no|isnt it|is it|dont they|doesnt it|correct|ok)\s*\?+\s*$")
_CAUSAL = _re.compile(r"\b(?:why|how come|since|because|cuz|cause|bc|what makes)\b")
# declarative word order + "?" (uptalk/hedged assertion: "i think its body temp?", "maybe it denatures?")
_UPTALK = _re.compile(r"^(?:so\s+|like\s+|ok\s+|wait\s+|yeah\s+)*(?:i think|i guess|i bet|maybe|probably|its|it|it's|"
                      r"they|thats|that's|that|because|cuz|cause|bc|the enzyme|the amylase|enzymes|amylase)\b")
_ADDRESS = _re.compile(r"^@?[a-z]+,\s+")  # "zoe, is there ..."


def _lp(lang, name, default):
    from .. import lang as _lang
    return _lang.attr(lang, name, default)


def is_interrogative(sentence: str, lang: str | None = None) -> bool:
    """`lang`: language code (None = English); text is accent-folded first (no-op for English)."""
    s = _re.sub(r"^[^a-z0-9@]+", "", fold(sentence).strip())
    s = _lp(lang, "ADDRESS", _ADDRESS).sub("", s)
    if _lp(lang, "TAG_Q", _TAG_Q).search(s):
        return False
    if s.endswith("?"):
        return not _lp(lang, "UPTALK", _UPTALK).match(s)
    return any(s == w or s.startswith(w + " ") for w in _lp(lang, "QUESTION_START", QUESTION_START))


def presupposes(sentence: str, fact, lang: str | None = None) -> bool:
    low = fold(sentence)
    hits = {k for k in fact.keyphrases if fold(k) in low}
    if not hits:
        return False
    return bool(_lp(lang, "CAUSAL", _CAUSAL).search(low)) or len(hits) >= 2


def asserts_in(text: str, keyphrase: str, fact, lang: str | None = None) -> bool:
    """True if some sentence of `text` containing `keyphrase` asserts it: a declarative sentence,
    or a question that presupposes `fact`."""
    found = False
    kp = fold(keyphrase)
    for sent in sentences(text):
        if kp not in fold(sent):
            continue
        found = True
        if not is_interrogative(sent, lang) or presupposes(sent, fact, lang):
            return True
    return not found  # keyphrase spans a sentence break: be conservative (treat as asserted)


def true_leaks(task: Task, s: StudentState, plan: ContentPlan | None, text: str,
               st: SessionState | None = None) -> list[str]:
    return [r["fact_id"] for r in leak_triggers(task, s, plan, text, st) if r["type"] == "true_leak"]


def learn_visible_echoes(task: Task, st: SessionState, s: StudentState, plan: ContentPlan, text: str) -> list[str]:
    """A student who uses a visible fact demonstrably read it -> it becomes known."""
    learned = []
    for r in leak_triggers(task, s, plan, text, st):
        if r["type"] == "visible_echo" and r["fact_id"] not in s.known_facts:
            s.known_facts.append(r["fact_id"])
            s.kcs[r["kc"]].evidence_seen.append(r["fact_id"])
            learned.append(r["fact_id"])
    return learned


# ------------------------------------------------------------------ claims
def claim_voiceable(task: Task, s: StudentState, kc: str, idx: int, st: SessionState | None) -> bool:
    """A KC correct claim is voiceable by `s` if its wording carries no true leak."""
    text = task.kc(kc).correct_claims[idx]
    return not true_leaks(task, s, None, text, st)


def effective_claim_idx(plan: ContentPlan) -> int | None:
    if plan.claim != "correct" or plan.kc is None or plan.family == "counterexample":
        return None
    return plan.claim_idx if plan.claim_idx is not None else 0


def plan_ideas(task: Task, plan: ContentPlan) -> set[str]:
    """Key ideas voiced with correct content by this plan."""
    if plan.is_correct is not True or plan.claim not in ("fact", "correct"):
        return set()
    return task.ideas_of(plan.kc, plan.fact_ids, effective_claim_idx(plan))


def raw_claim_key(plan: ContentPlan) -> tuple | None:
    if plan.claim == "fact" and plan.fact_ids:
        return ("fact", plan.fact_ids[0], True)
    if plan.claim == "correct" and plan.kc:
        if plan.family == "counterexample":
            return ("cx", plan.addresses_misconception, True)
        return ("claim", plan.kc, effective_claim_idx(plan), True)
    if plan.claim == "misconception" and plan.misconception_id:
        return ("m", plan.misconception_id, False)
    return None


def same_claim(task: Task, a: ContentPlan, b: ContentPlan) -> bool:
    """Same claim = same correctness and (same raw key, or overlapping key ideas)."""
    ka, kb = raw_claim_key(a), raw_claim_key(b)
    if ka is None or kb is None or ka[-1] != kb[-1]:
        return False
    return ka == kb or bool(plan_ideas(task, a) & plan_ideas(task, b))


def own_statements(task: Task, st: SessionState, sid: str, plan: ContentPlan) -> list[Message]:
    """Earlier messages in which `sid` stated the same claim."""
    return [m for m in st.messages if m.speaker == sid and m.plan is not None and same_claim(task, m.plan, plan)]


# ------------------------------------------------------------------ attribution

SELF_REF_RE = _re.compile(r"\b(?:yeah,?\s+)?(?:like|as)\s+i\s+(?:said|mentioned|was saying)(?:\s+(?:before|earlier|already))?"
                          r"|\bi\s+(?:already\s+)?said\s+(?:it|that|this)(?:\s+(?:already|before))?"
                          r"|\bagain,", _re.I)


def _plan_kcs(task: Task, plan: ContentPlan) -> set[str]:
    out = {plan.kc} if plan.kc else set()
    out |= {task.fact(f).kc for f in plan.fact_ids}
    return out


def said_before(task: Task, st: SessionState, sid: str, plan: ContentPlan) -> bool:
    """Did `sid` state this content (same claim, or for content-less plans anything on the KC) earlier?"""
    if raw_claim_key(plan) is not None:
        if own_statements(task, st, sid, plan):
            return True
        if plan.fact_ids and plan.fact_ids[0] in st.student(sid).shared_facts:
            return True
        return False
    kcs = _plan_kcs(task, plan)
    return any(m.speaker == sid and m.plan is not None and m.plan.claim != "none"
               and (_plan_kcs(task, m.plan) & kcs) for m in st.messages)


def credit_source(task: Task, st: SessionState, fid: str, sid: str) -> Message | None:
    """0.5.3: whom a back-reference to fact `fid` by `sid` credits -- the most recent STATEMENT of it by someone
    else (a question that names it is not saying it), skipping a line that was itself quoting `sid` (no circular
    credit); None if only `sid` really said it."""
    quoted = None
    for m in reversed(st.messages):
        if m.speaker == sid or m.plan is None or m.plan.act == "question" or m.plan.concedes or m.plan.closing \
                or m.plan.draft:
            continue      # (a concession quoting it, or a write-up, is not where it came from)
        if fid not in message_facts(task, m):
            continue
        if m.plan.reference and m.plan.ref_by == sid:
            continue
        if m.plan.reference:          # someone quoting someone else: prefer whoever said it first
            quoted = quoted or m
            continue
        return m
    return quoted


def peer_author(task: Task, st: SessionState, sid: str, plan: ContentPlan) -> str | None:
    """The most recent OTHER chat member (student or external participant) who stated this content (a question
    is not a statement; a line quoting `sid` is not its source)."""
    if plan.ref_by and plan.ref_by != sid and (plan.ref_by in st.roster or st.is_external(plan.ref_by)):
        return plan.ref_by
    for m in reversed(st.messages):
        if m.speaker == sid or m.plan is None or m.plan.act == "question" or (m.plan.reference and m.plan.ref_by == sid) \
                or m.plan.concedes or m.plan.closing or m.plan.draft:
            continue
        if (raw_claim_key(plan) is not None and same_claim(task, m.plan, plan)) or \
                (plan.fact_ids and set(plan.fact_ids) & message_facts(task, m)):
            return m.speaker
    return None


def fix_self_attribution(task: Task, st: SessionState, sid: str, plan: ContentPlan, text: str,
                         names: dict[str, str]) -> tuple[str, dict | None]:
    """Rewrite a false "like i said" (the speaker never said it): credit the peer who did, else drop it."""
    m = _lp(task.language, "SELF_REF_RE", SELF_REF_RE).search(text)
    if m is None or said_before(task, st, sid, plan):
        return text, None
    peer = peer_author(task, st, sid, plan)
    credit = _lp(task.language, "credit_peer", lambda n: f"like {n} said")
    repl = credit(names[peer].lower()) if peer and peer in names else _lp(task.language, "CREDIT_NONE", "yeah")
    if fold(m.group(0)).startswith(_lp(task.language, "AGAIN_PREFIX", "again")):
        repl = _lp(task.language, "CREDIT_AGAIN", "yeah,")
    out = (text[:m.start()] + repl + text[m.end():]).strip()
    out = _re.sub(r"\s{2,}", " ", out)
    return out, {"matched": m.group(0), "peer": peer, "draft": text, "final": out}


CREDIT_PEER_RE_EN = [r"\blike (?P<name>[a-z]+) said\b,?", r"\b(?P<name>[a-z]+)'?s point\b,?",
                     r"\b(?P<name>[a-z]+) (?:already )?said (?:it|that)\b,?", r"\bsame as (?P<name>[a-z]+)\b,?",
                     r"\b(?P<name>[a-z]+) is right\b,?", r"\bwhat (?P<name>[a-z]+) said\b,?"]


def credited(task: Task, st: SessionState, pid: str, sid: str, plan: ContentPlan, rest: str) -> bool:
    """Did `pid` actually say what `sid` credits them with (0.5.3)? A statement by `pid` sharing the plan's facts
    / claim or most of the line's content words, not itself quoting `sid`; for a credit with no content of its
    own ("acho que davi ta certo nessa"), a content statement by `pid` in the last 12 messages."""
    lg = task.language
    rw = content_words(rest, lg) - {fold(n) for n in st.display_names().values()}
    rf = set(task.facts_mentioned(rest)) | set(plan.fact_ids)
    recent = {m.id for m in st.messages[-12:]}
    for m in st.messages:
        if m.speaker != pid or m.plan is None or m.plan.act == "question" and m.plan.draft is None:
            continue
        if m.plan.reference and m.plan.ref_by == sid:
            continue          # circular: they were quoting the speaker
        mf = message_facts(task, m)
        if raw_claim_key(plan) is not None and raw_claim_key(m.plan) is not None and same_claim(task, m.plan, plan):
            return True
        if rf & mf:
            return True
        mw = content_words(m.text, lg)
        if len(rw) >= 2 and len(_match(rw, mw)) / len(rw) >= 0.4:
            return True
        if len(rw) < 2 and not rf and m.id in recent and (m.plan.claim in ("fact", "correct", "misconception")
                                                          or m.plan.draft == "propose" or m.plan.closing == "propose"):
            return True
    return False


def fix_peer_attribution(task: Task, st: SessionState, sid: str, plan: ContentPlan, text: str,
                         names: dict[str, str]) -> tuple[str, dict | None]:
    """0.5.3: "como malu disse X" only if Malu said X (or close): otherwise the credit goes to whoever did say
    it, or is dropped; one credit per line ("é como caio falou, o que caio disse, ..." -> the first one)."""
    from .realism import _fold_keep
    text0 = text
    text, dup = _dedupe_credits(task, text)
    if dup:
        out, fix = fix_peer_attribution(task, st, sid, plan, text, names)
        return out, {**(fix or {}), "reason": "double_credit", "draft": text0, "final": out}
    low = _fold_keep(text)
    first = {fold(n).split()[0]: pid for pid, n in names.items() if n}
    for rx in _lp(task.language, "CREDIT_PEER_RE", CREDIT_PEER_RE_EN):
        m = _re.search(rx, low)
        if m is None:
            continue
        pid = first.get(m.group("name"))
        if pid is None or pid == sid:
            continue
        rest = text[:m.start()] + text[m.end():]
        if credited(task, st, pid, sid, plan, rest):
            return text, None
        real = peer_author(task, st, sid, plan.model_copy(update={"ref_by": None}))
        if real is not None and real != pid and real in names and credited(task, st, real, sid, plan, rest):
            a, b = m.span("name")
            out = text[:a] + names[real].lower() + text[b:]
        else:
            out = (text[:m.start()] + text[m.end():]).strip(" ,")
            out = _re.sub(r"\s{2,}", " ", out).strip()
            if len(out.split()) < 2:
                out = _lp(task.language, "CREDIT_NONE", "yeah") + " " + out if out else \
                    _lp(task.language, "CREDIT_NONE", "yeah")
        return out, {"matched": m.group(0), "credited": pid, "peer": real, "reason": "false_peer_credit",
                     "draft": text, "final": out}
    return text, None


def _dedupe_credits(task: Task, text: str) -> tuple[str, bool]:
    """Drop every credit phrase after the first one of a line."""
    from .realism import _fold_keep
    spans = []
    low = _fold_keep(text)
    for rx in _lp(task.language, "CREDIT_PEER_RE", CREDIT_PEER_RE_EN):
        for m in _re.finditer(rx, low):
            spans.append(m.span())
    spans = sorted(set(spans))
    keep = []
    for a, b in spans:
        if keep and a < keep[-1][1]:
            continue
        keep.append((a, b))
    if len(keep) < 2:
        return text, False
    out = text
    for a, b in reversed(keep[1:]):
        out = out[:a] + out[b:]
    out = _re.sub(r"\s{2,}", " ", out).replace(" ,", ",").strip(" ,")
    out = _re.sub(r",\s*,", ",", out)
    return out, True


def strip_self_reference(task: Task, st: SessionState, sid: str, plan: ContentPlan, text: str
                         ) -> tuple[str, dict | None]:
    """0.5.1: a TRUE "like i said" / "como eu disse" is kept only when the speaker restates their point
    after someone disputed it (plan.restate); otherwise the phrase is dropped."""
    if plan.restate:
        return text, None
    m = _lp(task.language, "SELF_REF_RE", SELF_REF_RE).search(text)
    if m is None:
        return text, None
    out = (text[:m.start()] + text[m.end():]).strip()
    out = _re.sub(r"^[\s,.;:-]+", "", out)
    out = _re.sub(r"\s{2,}", " ", out).strip()
    if len(out.split()) < 2:
        return text, None
    return out, {"matched": m.group(0), "reason": "self_reference_not_disputed", "draft": text, "final": out}


# ------------------------------------------------------------------ LLM-flag evidence
# The Haiku check must return, for each leak flag, the item id and `quote`: the exact words of the
# draft that ASSERT the item. Code then keeps a flag only if the evidence holds up:
#   (a) the quote is present and is a substring of the draft (both normalized: lowercase, no
#       punctuation/apostrophes, single spaces);
#   (b) the sentence(s) holding the quote are not plain questions (see `is_interrogative`), unless
#       they presuppose the item (`presupposes`, same heuristic as the code-side check);
#   (c) the quote carries some of the item's content: a content word (stopwords and task-generic words
#       removed, light stemming) shared with the item's text, chat lines or keyphrases;
# and the same allowances as the code-side check apply, with KC-level ids ("kc_enzyme")
# mapped to the KC's facts and correct claims:
#   own_material - the speaker holds a card fact of the item's KC and every item word in the quote
#                  is in their own material (or in the public task text);
#   visible_echo - the item (fact) is visible in the speaker's window, or every item word in the
#                  quote was said by someone else in the visible window (or is public task text);
#   can_state    - the speaker can state the item's KC and the quote's item words are covered by
#                  the KC's correct claims (they may say a correct claim they know).
# Only flags that survive are true leaks; the rest are logged as `llm_flag_rejected` with a reason.
_STOPWORDS = frozenset("""
a an the and or but so if then than i im ive id me my we us our you your u ur he she they them their it its
is are was were be been being am do does did doing done have has had having can could should would will
wont dont doesnt didnt isnt arent wasnt not no yes yeah ya ok okay lol like just really very kinda sorta
gonna wanna gotta to of in on at for with from by as about into onto over under up down out off this that
thats these those there theres here what whats how why when where who which whom tho though also too
some any all more most much many lot lots thing things stuff guys guy anyone someone something anything
one get gets got go goes going say says said said think thinks know knows mean means right well oh hmm um
wait now still even again because cuz cause bc since let lets make makes made
""".split())


def norm(text: str) -> str:
    t = fold(text).replace("'", "").replace("’", "")
    t = _re.sub(r"[^a-z0-9?]+", " ", t)
    return _re.sub(r"\s+", " ", t.replace("?", " ")).strip()


def _stem(w: str) -> str:
    for suf in ("ation", "ing", "est", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w[:-1] if w.endswith("e") and len(w) > 3 else w


def content_words(text: str, lang: str | None = None) -> set[str]:
    stop, stem = _lp(lang, "STOPWORDS", _STOPWORDS), _lp(lang, "stem", _stem)
    return {stem(w) for w in norm(text).split() if w not in stop and (len(w) > 1 or w.isdigit())}


def _match(a: set[str], b: set[str]) -> set[str]:
    """Words of `a` matched in `b` (equal stems, or a shared prefix of >= 5 letters)."""
    out = set()
    for x in a:
        if x in b or any(len(x) >= 5 and len(y) >= 5 and (x.startswith(y) or y.startswith(x)) for y in b):
            out.add(x)
    return out


def _fact_content(f, lang: str | None = None) -> set[str]:
    return content_words(" ".join([f.text, *f.chat, *f.keyphrases]), lang)


def generic_words(task: Task) -> set[str]:
    """Words shared by many of the task's facts (>= max(3, half)): they identify no item."""
    from collections import Counter
    c = Counter(w for f in task.facts for w in _fact_content(f, task.language))
    thr = max(3, (len(task.facts) + 1) // 2)
    return {w for w, n in c.items() if n >= thr}


def item_facts(task: Task, item_id: str) -> list:
    if item_id in {f.id for f in task.facts}:
        return [task.fact(item_id)]
    return [f for f in task.facts if f.kc == item_id]


def item_kc(task: Task, item_id: str) -> str | None:
    if item_id in {f.id for f in task.facts}:
        return task.fact(item_id).kc
    return item_id if item_id in {k.id for k in task.kcs} else None


def item_content(task: Task, item_id: str) -> set[str]:
    lg = task.language
    words = set().union(*[_fact_content(f, lg) for f in item_facts(task, item_id)]) if item_facts(task, item_id) else set()
    if item_id in {k.id for k in task.kcs}:
        k = task.kc(item_id)
        words |= content_words(" ".join([k.name, *k.correct_claims, *k.keyphrases]), lg)
    return words - generic_words(task)


def _quote_sentences(draft: str, quote_n: str) -> list[str]:
    out = []
    for sent in sentences(draft):
        sn = norm(sent)
        if sn and (sn in quote_n or quote_n in sn or set(sn.split()) & set(quote_n.split())):
            out.append(sent)
    return out or [draft]


def validate_llm_flag(task: Task, s: StudentState, plan: ContentPlan | None, draft: str, item_id: str | None,
                      quote: str | None, st: SessionState | None = None,
                      dont_ids: set[str] | None = None) -> tuple[bool, str]:
    """(keep, reason). keep=True: a true leak backed by evidence. Otherwise reason says why the
    flag is discarded (no_quote, quote_not_in_draft, unknown_item, not_in_do_not_know, interrogative,
    no_item_content, own_material, visible_echo, can_state)."""
    kc = item_kc(task, item_id or "")
    if kc is None:
        return False, "unknown_item"
    if dont_ids is not None and item_id not in dont_ids:
        return False, "not_in_do_not_know"
    if not quote or not str(quote).strip():
        return False, "no_quote"
    qn, dn = norm(str(quote)), norm(draft)
    if not qn or qn not in dn:
        return False, "quote_not_in_draft"
    facts = item_facts(task, item_id)
    lg = task.language
    sents = _quote_sentences(draft, qn)
    if all(is_interrogative(x, lg) for x in sents):
        k = task.kc(kc)
        causal = _lp(lg, "CAUSAL", _CAUSAL)
        pres = any(presupposes(x, f, lg) for x in sents for f in facts) or (
            item_id == kc and any(causal.search(fold(x)) and any(fold(p) in fold(x) for p in k.keyphrases)
                                  for x in sents))
        if not pres:
            return False, "interrogative"
    ov = _match(content_words(qn, lg), item_content(task, item_id))
    if not ov:
        return False, "no_item_content"
    public = content_words(task.public_text(), lg)
    held = set(s.known_facts) | set(plan.fact_ids if plan else [])
    own_kc = [task.fact(f) for f in held if task.fact(f).kc == kc]
    if own_kc:
        own = set().union(*[_fact_content(f, lg) for f in own_kc])
        if _match(ov, own) and not (ov - _match(ov, own) - _match(ov, public)):
            return False, "own_material"
    if st is not None:
        vis_src = visible_fact_sources(task, st, s.id)
        if item_id in vis_src:
            return False, "visible_echo"
        vis_words = set()
        for m in visible_messages(st, s.id):
            if m.speaker != s.id:
                vis_words |= content_words(m.text, lg)
        if _match(ov, vis_words) and not (ov - _match(ov, vis_words) - _match(ov, public)):
            return False, "visible_echo"
    if s.kcs.get(kc) is not None and s.kcs[kc].can_state:
        claims = content_words(" ".join(task.kc(kc).correct_claims), lg)
        if not (ov - _match(ov, claims) - _match(ov, public)):
            return False, "can_state"
    return True, "true_leak"
