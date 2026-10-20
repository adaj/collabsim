"""Code-level coherence of challenges, disagreements and belief lines (0.5.2).

The 0.5.1 real-Haiku review found "refutations" carrying an unrelated fact, "discordo" in front of a
restatement of what the other student had just said, and belief lines that also stated their own
refutation ("acho que a enzima nao se gasta, ela funciona como um combustivel que vai sendo
consumido"). The planner now builds every challenge from content that refutes the targeted belief
(planner._challenge_plan); this module holds the text-level checks applied to every final line,
whichever realizer produced it:

* `contradiction(task, plan, text)`: a belief line must not contain the belief's refutation (a
  refuting fact's keyphrase or a refuting claim's KC keyphrase), and a refuting line must not assert
  the belief (the misconception's own keyphrases, matched negation-aware, so "nao se gasta" does not
  hit "se gasta"). Counterexamples are exempt (they quote the belief on purpose).
* `opposes(st, sid, plan)`: may this turn be phrased as a disagreement? Only a challenge of a belief
  someone ELSE voiced (and the speaker does not hold), or a holder defending their belief.
* `disagree_marker(text, task)` / `strip_disagree(text, task)`: "discordo", "acho que nao", "nada a
  ver", a leading "nao," / "no," / "nah", "i disagree"... (language pack `DISAGREE_MARKER_RE`).
"""
from __future__ import annotations

import re

from ..models.task import fold

NEGATIONS = frozenset({"nao", "n", "nunca", "nem", "not", "isnt", "dont", "doesnt", "never", "wont", "cant",
                       "arent", "aint"})
_WORD = re.compile(r"[a-z0-9/:.+]+")

_NOT_DISAGREE_PT = (r"(?!sei\b|to\b|tou\b|tenho\b|tinha\b|entendi\b|lembro\b|faco\b|achei\b|consigo\b|sabia\b|ta\b|"
                    r"e\b|era\b|vi\b|estudei\b|aguento\b|muito\b|mt\b|da\b|deu\b|precisa\b|sai\b|vou\b|quero\b|"
                    r"gosto\b|vejo\b|aguentando\b|acho\b)")
DISAGREE_MARKER_RE = {
    "pt-BR": re.compile(
        r"\bdiscord\w*|\bacho que (?:nao|n)\b(?:,|\s+" + _NOT_DISAGREE_PT + r"|$)|\bnada a ver\b|\bnao concordo\b"
        r"|^\W*(?:\w+,?\s+)?(?:hmm |pera |mas )?nao(?:[,!]?$|[,!]?\s+" + _NOT_DISAGREE_PT + ")"),
    "en": re.compile(
        r"\bi disagree\b|\bi dont think so\b|\bdont agree\b|\bthats wrong\b"
        r"|^\W*(?:\w+,?\s+)?(?:wait |hmm |but )?(?:no|nah|nope)(?:[,!]?$|[,!]?\s+"
        r"(?!idea\b|clue\b|way\b|one\b|problem\b|worries\b|spoilers?\b))"),
}
# leading disagreement leads that strip_disagree removes (folded text)
_LEAD_PT = re.compile(r"^\W*(?:(?:hmm|pera|mas|entao)\s+)*(?:nao|n|nada a ver|acho que nao|discordo(?: um pouco)?"
                      r"(?: d[oa] \w+)?|nao concordo(?: com \w+)?)\s*[,.!:]?\s*")
_LEAD_EN = re.compile(r"^\W*(?:(?:hmm|wait|but|so)\s+)*(?:no|nah|nope|i dont think so|i disagree(?: with \w+)?)"
                      r"\s*[,.!:]?\s*")


def _lang(task) -> str:
    return "pt-BR" if getattr(task, "language", "en") == "pt-BR" else "en"


def _words(text: str) -> list[str]:
    return _WORD.findall(fold(text))


def phrase_hits(text: str, phrases, negation_aware: bool = True, whole_word: bool = False) -> list[str]:
    """Phrases found in `text` (accent-folded, token-aligned at the start). negation_aware: a hit right
    after a negation word (within 3 tokens) does not count, unless the phrase itself starts with one.
    whole_word: the phrase must also end at a word boundary ("more lemon" does not hit "more lemony")."""
    low = " " + " ".join(_words(text)) + " "
    out = []
    for p in phrases:
        fp = " ".join(_words(p))
        if not fp:
            continue
        rx = r"(?<![a-z0-9])" + re.escape(fp) + (r"(?![a-z0-9])" if whole_word else "")
        for m in re.finditer(rx, low):
            if negation_aware and fp.split()[0] not in NEGATIONS:
                if any(w in NEGATIONS for w in low[:m.start()].split()[-3:]):
                    continue
            out.append(p)
            break
    return out


def belief_hits(task, mid: str, text: str) -> list[str]:
    """Words of `text` that ASSERT misconception `mid` (its keyphrases, negation-aware)."""
    return phrase_hits(text, task.misconception(mid).keyphrases, negation_aware=True, whole_word=True)


def refutation_hits(task, mid: str, text: str) -> list[str]:
    """Words of `text` that state the refutation of `mid`."""
    return phrase_hits(text, task.refuting_phrases(mid), negation_aware=False)


def is_challenge(plan) -> bool:
    """A turn that takes up someone's belief: a disagreement, an explanation or a press addressing it."""
    return bool(plan.addresses_misconception) and plan.closing is None and plan.draft is None


def refutes(task, plan) -> bool:
    """Does the challenge's CONTENT refute the belief it addresses? (press / counterexample: yes)."""
    mid = plan.addresses_misconception
    if not mid or plan.act == "press" or plan.family in ("press_for_reasoning", "counterexample"):
        return True
    if plan.claim == "fact" and plan.fact_ids:
        return plan.fact_ids[0] in task.refuting_facts(mid)
    if plan.claim == "correct" and plan.kc:
        return plan.claim_idx is None or plan.claim_idx in task.refuting_claims(mid)
    return plan.claim in ("none", "hedge")


def contradiction(task, plan, text: str) -> dict | None:
    """A belief line that states its refutation, or a refuting line that asserts the belief."""
    if plan.claim == "misconception" and plan.misconception_id:
        h = refutation_hits(task, plan.misconception_id, text)
        if h:
            return {"reason": "belief_states_refutation", "misconception": plan.misconception_id, "hits": h}
    mid = plan.addresses_misconception
    if mid and plan.claim in ("fact", "correct") and plan.family != "counterexample" and plan.closing is None:
        h = belief_hits(task, mid, text)
        if h:
            return {"reason": "refutation_asserts_belief", "misconception": mid, "hits": h}
    return None


def opposes(st, sid: str, plan) -> bool:
    """May this turn be worded as a disagreement? A challenge of a belief another chat member voiced
    (the anchor) that the speaker does not hold, or a holder defending their own belief."""
    if plan.claim == "misconception" and plan.misconception_id and (plan.pushback or plan.anchor_turn is not None):
        return True
    mid = plan.addresses_misconception
    if not mid or plan.anchor_turn is None or not (0 <= plan.anchor_turn < len(st.messages)):
        return False
    am = st.messages[plan.anchor_turn]
    if am.speaker == sid or am.plan is None:
        return False
    if not (am.plan.claim == "misconception" or am.plan.misconception_id == mid):
        return False
    try:
        s = st.student(sid)
    except StopIteration:
        return True
    m = s.misconception(mid)
    return not (m is not None and m.active)


def disagree_marker(text: str, task) -> bool:
    from .. import lang
    rx = lang.attr(task, "DISAGREE_MARKER_RE", None) or DISAGREE_MARKER_RE[_lang(task)]
    return bool(rx.search(fold(text)))


def strip_disagree(text: str, task) -> str:
    """The line without its leading disagreement lead ("nao, ..." -> "...")."""
    rx = _LEAD_PT if _lang(task) == "pt-BR" else _LEAD_EN
    t = text
    for _ in range(2):
        m = rx.match(fold(t))
        if not m or not m.group(0).strip():
            break
        t = t[m.end():]
    t = re.sub(r"^\W+", "", t).strip()
    return t if len(t.split()) >= 2 else text


def coherence_issue(task, st, sid: str, plan, text: str) -> dict | None:
    """The code checks on a final line: contradiction, a disagreement marker without opposition, and a
    disagreement whose content is already in the line it answers ("discordo, pra mim <the same>")."""
    c = contradiction(task, plan, text)
    if c is not None:
        return c
    if plan.act in ("off_task", "idk", "refusal", "backchannel", "greeting") or not disagree_marker(text, task):
        return None
    if not opposes(st, sid, plan):
        return {"reason": "disagree_without_opposition"}
    mid = plan.addresses_misconception
    if mid and plan.anchor_turn is not None and plan.family != "counterexample":
        mine = set(refutation_hits(task, mid, text))
        theirs = set(refutation_hits(task, mid, st.messages[plan.anchor_turn].text))
        if mine & theirs:
            return {"reason": "disagree_restating", "hits": sorted(mine & theirs)}
    return None
