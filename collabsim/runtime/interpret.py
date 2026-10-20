"""External participants' messages -> a ContentPlan.

An external participant (a person or a bot) types free text into the simulated group chat. ONE
Haiku call per message ("interpret") maps the text to a ContentPlan-like annotation: act, KC refs, fact ids mentioned
(only facts of the task), correctness, misconception voiced / challenged, addressee. Code then
validates every field against the task and the chat (unknown ids are dropped, the challenged
misconception must have been voiced recently, a correct claim is credited to a key idea only when
its wording overlaps one of the KC's correct claims) and builds the plan. The normal state-update
rules then apply to it as to a peer turn: students may learn from the participant or disagree.
Every participant is read the same way, whoever they are.

Participants have no hidden state. When interpretation fails (API error, unparsable or empty JSON),
the message is an OPAQUE SOCIAL turn: a backchannel (or a content-free question if it ends with
"?"), which moves nobody's knowledge.

`heuristic_interpretation` is the code-only interpretation (keyphrases, misconception wording,
counterexample wording, names); it is the offline MockLLM's answer and the `template` backend's
interpreter.
"""
from __future__ import annotations

import json
import re

from ..llm.client import LLMClient, LLMRequest
from ..llm.config import MAX_TOKENS, ModelConfig
from ..models.message import ContentPlan
from ..models.session import SessionState
from ..models.task import Task, fold

ACTS = ("on_task_claim", "question", "agree", "disagree", "explain", "off_task", "interface",
        "backchannel", "greeting", "idk", "press", "refusal")
GREET_RE = re.compile(r"^\W*(oi+e?|ola|opa|salve|iae|eai|e ai|bom dia|boa tarde|boa noite|hi+|hey+|hello|yo)\b")
AGREE_RE = re.compile(r"^\W*(concordo|isso|verdade|vdd|exato|sim|faz sentido|agree|yes|true|same|makes sense)\b")
IDK_RE = re.compile(r"\b(nao sei|n sei|sei la|sla|idk|no idea|i dont know|dont know)\b")


def _names_to_ids(st: SessionState) -> dict[str, str]:
    return {fold(name): sid for sid, name in st.display_names().items()}


def addressee_of(text: str, st: SessionState | dict, sender: str) -> str | None:
    """The chat member a message names (students and external participants, longest name first, never
    the writer). Writing someone's name is how a participant addresses them. `st`: the session state,
    or an {id: display name} dict."""
    low = fold(text)
    names = st if isinstance(st, dict) else st.display_names()
    for pid, name in sorted(names.items(), key=lambda kv: -len(kv[1])):
        if pid != sender and name and re.search(rf"@?\b{re.escape(fold(name))}\b", low):
            return pid
    return None


def _recent_voiced(st: SessionState, mid: str, hid: str, window: int = 6):
    """Most recent turn (not the writer's) voicing misconception `mid` in the last `window` turns."""
    for m in reversed(st.recent_turns(window)):
        if m.speaker != hid and m.plan is not None and m.plan.claim == "misconception" \
                and m.plan.misconception_id == mid:
            return m
    return None


def _best_claim(task: Task, kc: str, text: str) -> int | None:
    from ..sim.content import content_words
    tw = content_words(text, task.language)
    best, idx = 0, None
    for i, c in enumerate(task.kc(kc).correct_claims):
        ov = len(tw & content_words(c, task.language))
        if ov > best:
            best, idx = ov, i
    return idx


def heuristic_interpretation(task: Task, st: SessionState, text: str, sender: str) -> dict:
    """Code-only reading of a participant's message, in the same JSON shape the Haiku call returns."""
    hid = sender
    from ..sim.content import is_interrogative, sentences
    low = fold(text)
    facts = task.facts_mentioned(text)
    voiced = task.misconceptions_expressed(text)
    challenged = task.misconceptions_addressed(text)
    kcs = task.content_kcs(text)
    question = text.strip().endswith("?") or any(is_interrogative(x, task.language) for x in sentences(text))
    to = addressee_of(text, st, hid)
    names = st.display_names()
    out = {"act": "backchannel", "kc": None, "facts": [], "is_correct": None, "voices_misconception": None,
           "challenges_misconception": None, "addressee": names.get(to) if to else None,
           "asks_about_fact": None, "claim_idx": None}
    if voiced:
        m = task.misconception(voiced[0])
        out.update(act="on_task_claim", kc=m.kc, voices_misconception=m.id, is_correct=False)
    elif question:
        out.update(act="question", kc=kcs[0] if kcs else None, asks_about_fact=facts[0] if facts else None)
        if challenged:
            out.update(act="press", challenges_misconception=challenged[0], kc=task.misconception(challenged[0]).kc)
    elif facts or kcs or challenged:
        kc = task.fact(facts[0]).kc if facts else (kcs[0] if kcs else task.misconception(challenged[0]).kc)
        out.update(act="disagree" if challenged else "on_task_claim", kc=kc, facts=facts, is_correct=True,
                   challenges_misconception=challenged[0] if challenged else None)
    elif GREET_RE.search(low):
        out["act"] = "greeting"
    elif IDK_RE.search(low):
        out["act"] = "idk"
    elif AGREE_RE.search(low):
        out["act"] = "agree"
    return out


def build_plan(task: Task, st: SessionState, raw: dict, text: str, hid: str) -> tuple[ContentPlan, list[str]]:
    """Validate an interpretation (LLM or heuristic) against the task and the chat; build the plan.
    Returns (plan, notes) where notes list every field that was dropped or corrected."""
    notes: list[str] = []
    fact_ids = {f.id for f in task.facts}
    kc_ids = {k.id for k in task.kcs}
    m_ids = {m.id for m in task.misconceptions}
    act = raw.get("act") if raw.get("act") in ACTS else None
    if act is None:
        notes.append(f"act {raw.get('act')!r} unknown")
        act = "question" if text.strip().endswith("?") else "backchannel"
    facts = []
    for f in raw.get("facts") or []:
        (facts.append(f) if f in fact_ids and f not in facts else notes.append(f"fact {f!r} not in task"))
    kc = raw.get("kc") if raw.get("kc") in kc_ids else None
    if raw.get("kc") is not None and kc is None:
        notes.append(f"kc {raw.get('kc')!r} not in task")
    voiced = raw.get("voices_misconception") if raw.get("voices_misconception") in m_ids else None
    challenged = raw.get("challenges_misconception") if raw.get("challenges_misconception") in m_ids else None
    asked = raw.get("asks_about_fact") if raw.get("asks_about_fact") in fact_ids else None
    correct = raw.get("is_correct") if isinstance(raw.get("is_correct"), bool) else None
    names = _names_to_ids(st)
    addr_raw = raw.get("addressee")
    addressee = None
    if isinstance(addr_raw, str) and addr_raw.strip():
        a = fold(addr_raw.strip().lstrip("@"))
        addressee = names.get(a)
        if addressee is None and a in names.values():
            addressee = a
        if addressee == hid:
            addressee = None
    plan = ContentPlan(act=act, addressee=addressee)
    if voiced is not None:
        m = task.misconception(voiced)
        plan = ContentPlan(act="explain" if act == "explain" else "on_task_claim", kc=m.kc, claim="misconception",
                           misconception_id=voiced, is_correct=False, addressee=addressee)
    elif act == "question" and challenged is None:
        plan = ContentPlan(act=act, kc=kc or (task.fact(asked).kc if asked else None), asked_fact=asked,
                           addressee=addressee)
    elif act in ("on_task_claim", "explain", "disagree", "press") or facts or challenged:
        if facts and kc is None:
            kc = task.fact(facts[0]).kc
        if challenged is not None and kc is None:
            kc = task.misconception(challenged).kc
        if act in ("question", "agree", "backchannel", "off_task", "greeting", "idk", "refusal", "interface"):
            act = "press" if act == "question" else "on_task_claim"
        if facts and correct is not False:
            plan = ContentPlan(act=act if act != "press" else "disagree", kc=kc, claim="fact", fact_ids=facts,
                               is_correct=True, addressee=addressee)
        elif kc is not None and correct is True:
            idx = raw.get("claim_idx")
            n = len(task.kc(kc).correct_claims)
            if not (isinstance(idx, int) and 0 <= idx < n):
                idx = _best_claim(task, kc, text)
            if idx is None:   # correct, but matches no correct claim: counts, credits no key idea
                plan = ContentPlan(act=act, kc=kc, claim="none", is_correct=True, addressee=addressee)
            else:
                plan = ContentPlan(act=act, kc=kc, claim="correct", claim_idx=idx, is_correct=True, addressee=addressee)
        elif kc is not None and correct is False:
            plan = ContentPlan(act=act if act != "press" else "on_task_claim", kc=kc, claim="none",
                               is_correct=False, addressee=addressee)
        elif act == "press" and challenged is not None:
            plan = ContentPlan(act="press", kc=kc, addressee=addressee)
        else:
            plan = ContentPlan(act="backchannel" if act != "question" else "question", addressee=addressee)
        if challenged is not None:
            src = _recent_voiced(st, challenged, hid)
            if src is None:
                notes.append(f"challenged misconception {challenged} was not voiced recently")
            else:
                cx = task.misconception(challenged).counterexample
                fam = ("counterexample" if any(c for c in cx) and challenged in task.misconceptions_addressed(text)
                       else "evidence_from_material" if plan.fact_ids
                       else "press_for_reasoning" if plan.act == "press" else "peer_explanation")
                plan.addresses_misconception = challenged
                plan.anchor_turn = src.id
                plan.family = fam
                plan.addressee = plan.addressee or src.speaker
                if plan.act == "on_task_claim":
                    plan.act = "disagree"
                if plan.kc is None:
                    plan.kc = task.misconception(challenged).kc
    elif act in ("agree", "backchannel"):
        anchor = next((m for m in reversed(st.recent_turns(3)) if m.speaker != hid), None)
        plan = ContentPlan(act=act, anchor_turn=anchor.id if anchor else None,
                           kc=anchor.plan.kc if anchor is not None and anchor.plan is not None else None,
                           addressee=addressee)
    plan.target_len = max(1, min(60, len(text.split())))
    return plan, notes


def opaque_plan(text: str) -> ContentPlan:
    """Fallback: an opaque social turn (moves nobody's knowledge)."""
    return ContentPlan(act="question" if text.strip().endswith("?") else "backchannel",
                       target_len=max(1, min(60, len(text.split()))))


# ------------------------------------------------------------------ Haiku interpretation
def interpret_prefix(task: Task) -> str:
    from ..llm.prompts import task_block
    from .. import lang
    L = lang.of(task)
    kcs = "\n".join(f"- {k.id}: {k.name}. Correct ideas: " + " | ".join(f"[{i}] {c}" for i, c in enumerate(k.correct_claims))
                    for k in task.kcs)
    facts = "\n".join(f"- {f.id} ({f.kc}): {f.text}" for f in task.facts)
    mis = "\n".join(f"- {m.id} ({m.kc}): {m.wrong_belief} Typical wording: " + " | ".join(m.wrong_answer_template)
                    for m in task.misconceptions)
    return f"""You annotate ONE chat message written by a participant who joined a small-group student chat \
in a research simulator of collaborative learning. The chat and the task are in {getattr(L, 'label', 'English')}. \
You never write chat messages yourself; you return ONE JSON object, nothing else.

{task_block(task)}

KNOWLEDGE COMPONENTS (KCs) and their correct ideas (indexed):
{kcs}
FACTS (printed on the students' material cards):
{facts}
MISCONCEPTIONS (wrong beliefs some simulated students hold):
{mis}

ACT LABELS (what the message does):
- on_task_claim: states something about the task content; explain: gives a reason or explanation;
- question: asks something; press: asks someone to justify what they said ("why do you think that?");
- agree / disagree: agrees with / challenges someone; greeting; idk / refusal: does not know, declines;
- off_task: talk not about the activity; interface: talk about the activity screen;
- backchannel: a short reaction with no content ("ok", "lol", "hmm").

Read the MESSAGE in the context of the RECENT CHAT and return:
{{"act": one of on_task_claim | question | agree | disagree | explain | off_task | interface | backchannel | greeting | idk | press | refusal,
 "kc": KC id the message is about, or null,
 "facts": [ids of FACTS the message states or paraphrases; only ids listed above],
 "is_correct": true if the content it asserts is correct for the task, false if wrong, null if it asserts nothing,
 "claim_idx": index of the correct idea of "kc" it expresses, or null,
 "voices_misconception": misconception id the message itself asserts, or null,
 "challenges_misconception": misconception id (voiced by someone in the RECENT CHAT) that the message argues against or questions, or null,
 "addressee": the name of the chat member it is addressed to (anyone listed in PARTICIPANTS), or null for the whole group,
 "asks_about_fact": fact id a question asks about, or null}}
Rules: a question asserts nothing (is_correct null) unless it takes a fact for granted. Greetings, jokes, \
complaints and talk about the screen are social acts with kc null. Only use ids that appear above; \
if unsure, prefer null."""


def interpret_request(model: str, task: Task, st: SessionState, text: str, sender: str) -> LLMRequest:
    from ..llm.prompts import task_ref
    names = st.display_names()
    lines = [f"{names.get(m.speaker, m.speaker)}: {m.text}" for m in st.messages[-8:]]
    roster = ", ".join(s.name for s in st.students)
    others = ", ".join(d["name"] for p, d in st.participants.items() if p != sender)
    suffix = (f"JOB: interpret\nPARTICIPANTS: simulated students {roster}"
              + (f"; other participants {others}" if others else "")
              + f"; the writer is {names.get(sender, sender)}\nRECENT CHAT:\n" + ("\n".join(lines) or "-")
              + f"\nMESSAGE ({names.get(sender, sender)}): {text}\nJSON:")
    return LLMRequest(model=model, purpose="interpret", prefix=interpret_prefix(task), suffix=suffix,
                      temperature=0.0, max_tokens=MAX_TOKENS["interpret"], json_mode=True,
                      meta={"task_id": task_ref(task), "state": st.model_dump(mode="json"), "text": text,
                            "sender": sender})


def mock_interpret(req) -> str:
    """Offline MockLLM handler: the code heuristic, returned as the model's JSON."""
    from ..llm.mock import _task
    m = req.meta
    st = SessionState.model_validate(m["state"])
    return json.dumps(heuristic_interpretation(_task(m["task_id"]), st, m["text"], m["sender"]))


class Interpreter:
    """client=None -> code heuristic only (template backend); else one Haiku call per message."""

    def __init__(self, client: LLMClient | None, models: ModelConfig | None = None):
        self.client = client
        self.models = models or (ModelConfig.from_env() if client is not None else None)
        self.log: list[dict] = []

    def __call__(self, task: Task, st: SessionState, text: str, sender: str) -> ContentPlan:
        from ..llm.client import GracefulStop
        rec = {"msg": len(st.messages), "text": text, "sender": sender}
        try:
            if self.client is None:
                raw, rec["source"] = heuristic_interpretation(task, st, text, sender), "heuristic"
            else:
                raw = self.client.complete_json(interpret_request(self.models.judge, task, st, text, sender))
                rec["source"] = "llm"
            if not isinstance(raw, dict) or not raw:
                raise ValueError("empty or unparsable interpretation")
            plan, notes = build_plan(task, st, raw, text, sender)
            rec.update(raw={k: v for k, v in raw.items() if not k.startswith("_")}, notes=notes)
        except GracefulStop:
            raise
        except Exception as e:  # noqa: BLE001  (robust fallback: opaque social turn)
            plan = opaque_plan(text)
            rec.update(source="fallback", error=f"{type(e).__name__}: {e}"[:200])
        rec["plan"] = plan.model_dump()
        self.log.append(rec)
        return plan
