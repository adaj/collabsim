"""Structural chat statistics shared by the pilot calibration and simulator output.

All functions take a list of *sessions*; a session is a list of message dicts with
keys ``speaker`` (str), ``t`` (seconds from session start, float), ``text`` (str),
``is_external`` (bool: not a student).  Only student messages enter the statistics.
"""
from __future__ import annotations

import re
import statistics
from typing import Iterable

IDK_RE = re.compile(r"\b(sla|sei la|sei lá|nao sei|não sei|n sei|nao faco ideia|não faço ideia|nao faço ideia|sei nada|"
                    r"idk|no idea|i dont know|i don't know|no clue)\b", re.IGNORECASE)


def gini(values: Iterable[float]) -> float:
    xs = sorted(float(v) for v in values)
    n = len(xs)
    if n == 0 or sum(xs) == 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(xs))
    return (2 * cum) / (n * sum(xs)) - (n + 1) / n


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def words(text: str) -> int:
    return len(text.split())


def session_stats(msgs: list[dict], roster: list[str] | None = None, lurk_min: float = 5.0) -> dict:
    stu = [m for m in msgs if not m["is_external"]]
    speakers = roster if roster is not None else sorted({m["speaker"] for m in stu})
    counts = {s: 0 for s in speakers}
    for m in stu:
        counts[m["speaker"]] = counts.get(m["speaker"], 0) + 1
    gaps = [b["t"] - a["t"] for a, b in zip(stu, stu[1:])]
    same = [a["speaker"] == b["speaker"] for a, b in zip(stu, stu[1:])]
    start = msgs[0]["t"] if msgs else 0.0
    first = {}
    for m in stu:
        first.setdefault(m["speaker"], m["t"])
    lurkers = [s for s in speakers if first.get(s, float("inf")) - start >= lurk_min * 60]
    cf = [is_backchannel(m["text"]) or bool(IDK_RE.search(m["text"])) for m in stu]
    run = best = 0
    for c in cf:
        run = run + 1 if c else 0
        best = max(best, run)
    return {
        "n_student_msgs": len(stu),
        "msgs_per_student": list(counts.values()),
        "gini": gini(counts.values()),
        "words": [words(m["text"]) for m in stu],
        "gaps": gaps,
        "same_speaker": same,
        "n_students": len(speakers),
        "n_lurkers": len(lurkers),
        "minutes": (stu[-1]["t"] - stu[0]["t"]) / 60.0 if len(stu) > 1 else 0.0,
        "n_idk": sum(1 for m in stu if IDK_RE.search(m["text"])),
        "n_backchannel": sum(1 for m in stu if is_backchannel(m["text"])),
        "max_content_free_run": best,
    }


def aggregate(sessions: list[list[dict]], rosters: list[list[str]] | None = None) -> dict:
    per = [session_stats(s, rosters[i] if rosters else None) for i, s in enumerate(sessions)]
    allw = [w for p in per for w in p["words"]]
    allg = [g for p in per for g in p["gaps"]]
    alls = [x for p in per for x in p["same_speaker"]]
    mps = [x for p in per for x in p["msgs_per_student"]]
    n_stu = sum(p["n_student_msgs"] for p in per)
    return {
        "n_sessions": len(per),
        "msgs_per_session_median": statistics.median(p["n_student_msgs"] for p in per),
        "msgs_per_session_mean": statistics.mean(p["n_student_msgs"] for p in per),
        "msgs_per_student_median": statistics.median(mps),
        "msgs_per_student_per_10min_median": statistics.median(
            x / max(p["minutes"], 1.0) * 10.0 for p in per for x in p["msgs_per_student"]),
        "msgs_per_student_max": max(mps),
        "gini_median": statistics.median(p["gini"] for p in per),
        "words_median": statistics.median(allw),
        "words_p90": quantile(allw, 0.9),
        "one_word_share": sum(1 for w in allw if w <= 1) / len(allw),
        "gap_median_s": quantile(allg, 0.5),
        "gap_p25_s": quantile(allg, 0.25),
        "gap_p75_s": quantile(allg, 0.75),
        "gap_p90_s": quantile(allg, 0.9),
        "gap_p95_s": quantile(allg, 0.95),
        "gap_p99_s": quantile(allg, 0.99),
        "gap_max_s": max(allg) if allg else float("nan"),
        "session_minutes_median": statistics.median(p["minutes"] for p in per),
        "p_same_speaker": sum(alls) / len(alls) if alls else float("nan"),
        "lurker_share": sum(p["n_lurkers"] for p in per) / sum(p["n_students"] for p in per),
        "idk_share": sum(p["n_idk"] for p in per) / n_stu if n_stu else 0.0,
        "backchannel_share": sum(p["n_backchannel"] for p in per) / n_stu if n_stu else 0.0,
        "max_content_free_run_max": max(p["max_content_free_run"] for p in per),
    }


STAT_KEYS = [
    "msgs_per_session_median", "msgs_per_student_median", "msgs_per_student_per_10min_median", "gini_median",
    "words_median",
    "one_word_share", "gap_median_s", "gap_p90_s", "p_same_speaker", "lurker_share",
]
# 0.5.1: the tail of the silence distribution (sets the group-silence cap) and session length
# (the default pace reproduces it); computed with leave-2-out ranges like the others
TAIL_KEYS = ["gap_p95_s", "gap_p99_s", "gap_max_s", "session_minutes_median"]
# filler statistics (0.5.0): pooled pilot values are upper bounds for the simulator, not bands
FILLER_KEYS = ["idk_share", "backchannel_share", "max_content_free_run_max"]


# ---------------------------------------------------------------------------------------------
# Answered-question rate (implementation-level calibration target, not a metric)
# ---------------------------------------------------------------------------------------------
# A transparent, language-neutral heuristic applied identically to the pilot (pt-BR) and simulated
# transcripts:
#   question  = a student message containing '?', not a greeting, not addressed to a non-student;
#   answered  = within the next ANSWER_WINDOW student messages, a DIFFERENT student sends a message
#               with >= ANSWER_MIN_WORDS words that is neither a question ('?') nor a backchannel.
ANSWER_WINDOW = 4
ANSWER_MIN_WORDS = 3
GREETING_RE = re.compile(r"^\s*(oi+|ol[aá]|salve|e\s?a[ií]|eae|bom dia|boa (tarde|noite)|hi+|hey|hello|yo|"
                         r"good (morning|evening|afternoon))\b", re.IGNORECASE)
BACKCHANNEL_TOKENS = {
    "ok", "okk", "okay", "lol", "lmao", "lmaooo", "haha", "hahaha", "kk", "kkk", "kkkk", "rs", "sim", "s", "yes",
    "yeah", "yep", "ya", "n", "nao", "não", "no", "nah", "verdade", "true", "concordo", "agree", "same", "hmm",
    "hm", "ah", "oh", "ata", "entendi", "blz", "beleza", "isso", "right", "nice", "show", "massa", "top", "fr",
    "tbm", "também", "too", "me", "eu", "i", "it", "is", "that", "thats", "e", "é", "mesmo", "makes", "sense",
    "boa", "pois", "kkkkk", "kkkkkk", "omg", "wow", "uau", "ss", "aham", "uhum"}
LAUGH_RE = re.compile(r"^(k{2,}|(ha){2,}h?|(rs){1,}|lo+l|lma+o+|ja{2,})$", re.IGNORECASE)


def is_backchannel(text: str) -> bool:
    toks = re.findall(r"[\wáéíóúãõâêôç]+", text.lower())
    return bool(toks) and all(t in BACKCHANNEL_TOKENS or LAUGH_RE.match(t) for t in toks)


def is_peer_question(text: str, addressed_out=None) -> bool:
    """A question among the students: not a greeting, and not addressed to a non-student (a message
    naming one of the session's external speakers, or using "@")."""
    if "?" not in text or GREETING_RE.match(text):
        return False
    return not (addressed_out is not None and addressed_out.search(text))


def _external_rx(msgs: list[dict]):
    names = sorted({m["speaker"] for m in msgs if m["is_external"]}, key=len, reverse=True)
    return re.compile("|".join([re.escape(n) for n in names] + ["@"]), re.IGNORECASE)


def is_answer(text: str) -> bool:
    return len(text.split()) >= ANSWER_MIN_WORDS and "?" not in text and not is_backchannel(text)


def answered_questions(msgs: list[dict]) -> tuple[int, int]:
    """(n peer questions, n answered) in one session (message dicts as above)."""
    stu = [m for m in msgs if not m["is_external"]]
    out = _external_rx(msgs)
    n = ans = 0
    for i, m in enumerate(stu):
        if not is_peer_question(m["text"], out):
            continue
        n += 1
        nxt = stu[i + 1:i + 1 + ANSWER_WINDOW]
        ans += any(x["speaker"] != m["speaker"] and is_answer(x["text"]) for x in nxt)
    return n, ans


def answered_question_rate(sessions: list[list[dict]]) -> dict:
    tot = [answered_questions(s) for s in sessions]
    n = sum(t[0] for t in tot)
    a = sum(t[1] for t in tot)
    return {"n_questions": n, "n_answered": a, "rate": a / n if n else float("nan")}


# ---------------------------------------------------------------------------------------------
# Same-speaker runs and top-speaker share (reported rows)
# ---------------------------------------------------------------------------------------------
def run_lengths(msgs: list[dict]) -> list[int]:
    """Lengths of runs of consecutive STUDENT messages by one speaker (external messages skipped)."""
    stu = [m for m in msgs if not m["is_external"]]
    runs: list[int] = []
    for i, m in enumerate(stu):
        if i and stu[i - 1]["speaker"] == m["speaker"]:
            runs[-1] += 1
        else:
            runs.append(1)
    return runs


def top_speaker_share(msgs: list[dict]) -> float:
    stu = [m for m in msgs if not m["is_external"]]
    if not stu:
        return 0.0
    c: dict[str, int] = {}
    for m in stu:
        c[m["speaker"]] = c.get(m["speaker"], 0) + 1
    return max(c.values()) / len(stu)


RUN_KEYS = ["run_len_p95", "run_len_max", "max_run_median", "top_speaker_share_median"]


def run_stats(sessions: list[list[dict]]) -> dict:
    per = [run_lengths(s) for s in sessions]
    allr = [r for p in per for r in p]
    return {"run_len_p95": quantile(allr, 0.95), "run_len_max": max(allr) if allr else 0,
            "max_run_median": statistics.median(max(p) if p else 0 for p in per),
            "top_speaker_share_median": statistics.median(top_speaker_share(s) for s in sessions)}


# Pilot sessions were time-boxed by design while simulated sessions end when the task is complete.
# 0.5.1: the default pace ("pilot") reproduces the pilot session length, so messages per student is
# checked again; messages per session depends on the group-size mix and stays reported only.
REPORTED_ONLY_STATS = {
    "msgs_per_session_median": "reported only: it depends on the group-size mix (pilot 2-4 students, simulated "
                               "3/4 by seed)",
}
