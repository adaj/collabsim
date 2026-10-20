"""Deterministic near-duplicate guard.

A candidate student line is TOO SIMILAR to an earlier one if, on normalized tokens, the set
Jaccard is >= .7 or the difflib ratio is >= .85, against any of the last `WINDOW` student
messages. Lines with fewer than `MIN_TOKENS` tokens ("ok", "lol", "anyone?") are exempt, since
students in the pilot chats repeat such atoms constantly; but no student ever sends the exact same string twice
in a row. Only WORDING is affected: the content plan (what the updater reads) never changes.
"""
from __future__ import annotations

import difflib
import re

JACCARD_MAX = 0.7
RATIO_MAX = 0.85
WINDOW = 15
MIN_TOKENS = 3
REPHRASE = ("SAY IT DIFFERENTLY: your draft was almost the same as an earlier message in the chat "
            "(\"{prev}\"). Express the same content with clearly different words, or shorter.")


def norm_tokens(text: str) -> list[str]:
    from ..models.task import fold  # accent-folded (no-op for English): pt-BR words stay whole
    return re.findall(r"[a-z0-9]+", fold(text))


def norm(text: str) -> str:
    return " ".join(norm_tokens(text))


def similarity(a: str, b: str) -> tuple[float, float]:
    ta, tb = set(norm_tokens(a)), set(norm_tokens(b))
    jac = len(ta & tb) / len(ta | tb) if ta | tb else 1.0
    sm = difflib.SequenceMatcher(None, norm(a), norm(b), autojunk=False)
    # exact ratio only when the cheap upper bounds reach the threshold; otherwise the (upper-bound)
    # quick ratio, which is already below RATIO_MAX, so threshold decisions are unchanged
    bound = sm.real_quick_ratio()
    if bound >= RATIO_MAX:
        bound = sm.quick_ratio()
        if bound >= RATIO_MAX:
            return jac, sm.ratio()
    return jac, bound


def _too_similar(a: str, b: str) -> tuple[bool, float, float]:
    jac, ratio = similarity(a, b)
    return jac >= JACCARD_MAX or ratio >= RATIO_MAX, jac, ratio


def check(text: str, st, sid: str, window: int = WINDOW) -> dict | None:
    """Returns a hit record (reason, matched text, scores) or None."""
    stu = [m for m in st.messages if m.kind == "student"]
    own = [m for m in stu if m.speaker == sid]
    if own and norm(own[-1].text) == norm(text):
        return {"reason": "own_repeat_in_a_row", "prev": own[-1].text, "prev_id": own[-1].id}
    if len(norm_tokens(text)) < MIN_TOKENS:
        return None
    for m in reversed(stu[-window:]):
        if len(norm_tokens(m.text)) < MIN_TOKENS:
            continue
        bad, jac, ratio = _too_similar(text, m.text)
        if bad:
            return {"reason": "near_duplicate", "prev": m.text, "prev_id": m.id,
                    "jaccard": round(jac, 3), "ratio": round(ratio, 3)}
    return None


def max_similarity(text: str, st, window: int = WINDOW) -> float:
    """Largest Jaccard / full difflib ratio against the window (used only to rank leftovers)."""
    stu = [m for m in st.messages if m.kind == "student"][-window:]
    best = 0.0
    for m in stu:
        ta, tb = set(norm_tokens(text)), set(norm_tokens(m.text))
        jac = len(ta & tb) / len(ta | tb) if ta | tb else 1.0
        best = max(best, jac, difflib.SequenceMatcher(None, norm(text), norm(m.text), autojunk=False).ratio())
    return best


def too_similar(a: str, b: str) -> bool:
    return _too_similar(a, b)[0]
