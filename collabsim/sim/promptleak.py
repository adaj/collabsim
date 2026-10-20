"""Prompt-label leak detector.

A final student line must never carry meta-instructions or plan labels from the realizer prompt
("ask the group: ...", "say: ...", "SAY THIS", "TARGET_LEN", "[on_task_claim]", "kc_temp", "(e.g. ...",
"<name>", a leading "Ethan:" speaker label, ...). Applied to every final line, whichever realizer
produced it (LLM draft, regenerated draft, template, fallback); a hit is regenerated or replaced by
another variant, and logged. Wording only: the content plan is untouched.
"""
from __future__ import annotations

import re

_LABEL_WORDS = (r"ask(?: the group| everyone| them| \w+)?|say(?: this)?|share(?: this)?|tell (?:the group|them|everyone)|"
                r"mention|answer|reply|respond|state(?: your belief)?|push back|disagree(?: with \w+)?|"
                r"agree(?: with \w+)?|restate(?: in your own words)?|refer back|point back|follow[- ]?up|"
                r"question|claim|fact|belief|hedge|act|message|output|content|response|line|nudge")
# "<label>:" at the start of the line (optionally after a name / filler), e.g. "ask the group: ..."
LEADING_LABEL_RE = re.compile(rf"^\W*(?:\w+[,:]?\s+)?(?:{_LABEL_WORDS})\s*:\s*", re.I)
PATTERNS = {
    "instruction_label": re.compile(rf"(?:^|[.!?]\s+)(?:{_LABEL_WORDS})\s*:\s*\S", re.I),
    "prompt_header": re.compile(r"\b(SAY THIS|SAY IT DIFFERENTLY|TARGET_LEN|YOU KNOW|YOU BELIEVE|YOU DO NOT KNOW|"
                                r"YOU ARE|MOOD|LAST MESSAGES|STRICT|ACT|CONTENT|ALLOWED THIS TURN|JSON)\s*:", re.I),
    "say_this": re.compile(r"\bsay this( content)?\b", re.I),
    "plan_label": re.compile(r"\[(on_task_claim|question|agree|disagree|explain|off_task|interface|"
                             r"backchannel|refusal|greeting|idk|press)\]|\b(on_task_claim|off_task|"
                             r"misconception_id|fact_ids|target_len|content_plan)\b", re.I),
    "task_ids": re.compile(r"\b(kc|f|m)_[a-z]{3,}\b"),
    "instruction_phrase": re.compile(r"\b(in your own (casual )?words|no content of your own|do not repeat|"
                                     r"do not announce|very short follow-up|refer back to it briefly|"
                                     r"you could start with|without stating any answer|as a teen would)\b", re.I),
    "example_marker": re.compile(r"\(e\.g\.|\be\.g\.,?\s*'|<name>|\{name\}|\{peer\}", re.I),
    # 0.5.2: second-person directives addressed to the WRITER of the line (never something a student
    # types to classmates): "you believe this and say it as your own opinion: ...", "the idea you use"
    "instruction_directive": re.compile(
        r"\byou (?:believe (?:this|that)|disagree with|push back on|were convinced|are unsure|said this yourself|"
        r"could start with|can start with|hold (?:this|the) belief)\b|\bsay (?:it|this|that) as (?:your|an?) "
        r"|\b(?:as )?your own (?:opinion|belief|words|casual words)\b|\bthe (?:idea|point|question|gist) you "
        r"|\bdefend (?:it|this|your)\b|\bwith a reason of your own\b|\bget (?:this|the) idea across\b|"
        r"\bgive this counterexample\b|\bwithout (?:adding|stating) (?:any )?(?:content|answer)\b|\bno new content\b|"
        r"\bvery short\b|\bwriting only\b|\bonly the (?:question|message)\b|\bshort, plausible\b|"
        r"\bin one or two (?:words|sentences)\b|\bthe chat has gone quiet\b|\bnobody answered\b|"
        r"\boff-task remark\b|\ba short (?:reaction|remark|nudge|follow-up)\b|\bthe (?:last|previous) message\b|"
        r"\bdo not (?:repeat|announce|add|copy)\b|\bnever (?:'like i said'|copy)\b|\bcrediting (?:them|\w+)\b|"
        r"\bthe activity question you are on\b|\bwhy they think that\b|"
        r"^\W*(?:say|write|ask|tell|mention|answer|reply|respond|state|restate|defend|agree with|disagree with|give|use|"
        r"suggest|propose|greet|decline|okay the|confirm|repeat|explain to)\b(?! (?:so|it|me|that again)\b)"
        r"(?!\s+what\W*$)|\bone short line about\b|\bnothing new\b|\bask the others to\b|\byou are supposed to\b|"
        r"\bin 1-2 sentences\b|\btime is almost up: suggest\b|\banswer with your reason\b|\basked why you think\b|"
        r"\b(?:your belief|the reason|the idea|the point|the question)\s*:|\bthinks this \(wrong\)|"
        r"\bshowing why it is wrong\b|\bit must speak against\b|\basked you; say\b|\bno content\)", re.I),
}

# pt-BR prompt labels: the detector is language-agnostic (English and pt-BR
# patterns both always apply), so no language needs to be threaded through.
from ..lang import pt_br as _pt  # noqa: E402

PT_LEADING_LABEL_RE = re.compile(rf"^\W*(?:\w+[,:]?\s+)?(?:{_pt.PROMPT_LEAK_LABEL_WORDS})\s*:\s*", re.I)
PT_PATTERNS = {
    "instruction_label": re.compile(rf"(?:^|[.!?]\s+)(?:{_pt.PROMPT_LEAK_LABEL_WORDS})\s*:\s*\S", re.I),
    "prompt_header": re.compile(rf"\b(?:{_pt.PROMPT_LEAK_HEADERS})\s*:", re.I),
    "instruction_phrase": re.compile(rf"\b(?:{_pt.PROMPT_LEAK_PHRASES})\b", re.I),
    "example_marker": re.compile(r"\(ex\.?:|\(por exemplo|\(tipo\s*'", re.I),
    "instruction_directive": re.compile(rf"(?:{_pt.PROMPT_LEAK_DIRECTIVES})", re.I),
}


def _folded(text: str) -> str:
    from ..models.task import fold
    return fold(text)


def hits(text: str, names: dict[str, str] | None = None, speaker: str | None = None) -> list[str]:
    """Names of the prompt-label patterns found in `text` (empty = clean). pt-BR patterns run on the
    accent-folded text too (a model may drop the accents of "você acredita")."""
    out = [k for k, rx in PATTERNS.items() if rx.search(text)]
    ft = _folded(text)
    out += [k for k, rx in PT_PATTERNS.items() if k not in out and (rx.search(text) or rx.search(ft))]
    if (LEADING_LABEL_RE.search(text) or PT_LEADING_LABEL_RE.search(text)) and "instruction_label" not in out:
        out.append("instruction_label")
    if names and speaker in names and re.match(rf"^\s*{re.escape(names[speaker])}\s*:", text, re.I):
        out.append("speaker_label")
    return out


def strip(text: str, names: dict[str, str] | None = None, speaker: str | None = None) -> str:
    """Last-resort cleanup: drop leading labels / speaker tags and example markers."""
    t = text
    if names and speaker in names:
        t = re.sub(rf"^\s*{re.escape(names[speaker])}\s*:\s*", "", t, flags=re.I)
    for _ in range(3):
        t2 = PT_LEADING_LABEL_RE.sub("", LEADING_LABEL_RE.sub("", t))
        if t2 == t:
            break
        t = t2
    t = PATTERNS["instruction_label"].sub(lambda m: m.group(0)[-1], t)
    t = re.sub(r"\(e\.g\.[^)]*\)?|<name>|\{name\}|\{peer\}", "", t, flags=re.I)
    t = PATTERNS["prompt_header"].sub("", t)
    t = PT_PATTERNS["prompt_header"].sub("", t)
    t = PT_PATTERNS["instruction_label"].sub(lambda m: m.group(0)[-1], t)
    t = PATTERNS["say_this"].sub("", t)
    t = re.sub(r"\[(\w+)\]", "", t)
    return re.sub(r"\s{2,}", " ", t).strip(" :-") or "hmm"


# ------------------------------------------------------------------ instruction echo (0.5.2)
def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _folded(text))


def instruction_frame(instruction: str, payloads=()) -> list[str]:
    """The instruction's own words: the instruction with its content payloads (the belief, the claim,
    the fact gist, the question...) removed, as folded tokens with a break marker at each removal."""
    t = _folded(instruction)
    for p in sorted((x for x in payloads if x), key=len, reverse=True):
        t = t.replace(_folded(p), " | ")
    out = []
    for chunk in t.split("|"):
        out += _toks(chunk) + ["|"]
    return out


def echo_hits(text: str, instruction: str, payloads=(), n: int = 4) -> list[str]:
    """Runs of >= n consecutive words of the instruction frame (or the whole frame chunk, if shorter
    but at least 3 words) that reappear in `text`: the model copied the instruction itself."""
    frame = instruction_frame(instruction, payloads)
    tt = " " + " ".join(_toks(text)) + " "
    found = []
    chunks, cur = [], []
    for w in frame:
        if w == "|":
            if cur:
                chunks.append(cur)
            cur = []
        else:
            cur.append(w)
    if cur:
        chunks.append(cur)
    for c in chunks:
        if 3 <= len(c) < n:
            if " " + " ".join(c) + " " in tt:
                found.append(" ".join(c))
            continue
        for i in range(len(c) - n + 1):
            g = " " + " ".join(c[i:i + n]) + " "
            if g in tt:
                found.append(g.strip())
                break
    return found
