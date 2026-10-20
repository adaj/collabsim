"""LLM realizer + leakage guard + slippage audit, behind the same `Realizer` interface as the
template realizer. Frugal by design, without changing what is measured:

1. DETERMINISTIC ACTS ARE TEMPLATED (no LLM). Turns with no epistemic content are worded by the
   template realizer from chat-style fillers: greetings, backchannels / acks / laughter ("ok",
   "lol", "hahaha", "agree", "same"), idk / refusals, and one-word atoms of any non-content act.
   Why: in the pilot corpus ~20% of student messages are a single word, and these lines are drawn
   from a tiny closed set; an LLM adds nothing but cost and variance. None of them carries KC
   content, so the updater (which reads the code plan), the leakage measures and the slippage
   audit are unaffected. Content acts (claims, explanations, disagreements, questions, press) and
   open-ended social lines (off-task chatter, interface talk) are ALWAYS worded by Haiku, even at
   one word, so lexical realism is still measured on LLM text.
2. Content turns (claims, explanations, disagreements, questions, press, hedges): Haiku writes
   the message from the code ContentPlan.
3. GUARD: the code keyphrase check runs first. A code-detected leak triggers one STRICT
   regeneration, then template fallback. The Haiku check still runs on every LLM-worded content
   turn (the spec places a Haiku classifier in the pipeline; keyphrases cannot catch
   paraphrases), but it is MERGED with the slippage audit into ONE call (JOB "check").
4. TRUNCATION: a draft cut by the output cap (stop_reason "max_tokens") is regenerated once with a
   "shorter" note, else trimmed back to its last complete sentence / word.
5. Deterministic noiser.
Student state ALWAYS updates from the plan; the check is guard + audit only, logged.
"""
from __future__ import annotations

from ..models.message import ContentPlan
from ..models.rng import RNG
from ..models.student import StudentState
from ..models.task import Task
from ..sim import promptleak
from ..sim.content import (last_mention, leak_triggers, norm, true_leaks, validate_llm_flag,
                            visible_fact_sources)
from ..sim.realizer import TemplateRealizer, noise, normalize_chat, strip_card_announcement
from .. import lang
from .client import LLMClient
from .config import ModelConfig
from .prompts import TRUNCATED_NOTE, check_request, coherence_request, history, knowledge, realizer_request, say_this
from ..sim import coherence

STRICT = ("STRICT: your previous draft mentioned something you do NOT know. Rewrite using ONLY the "
          "content above; if unsure, hedge or say idk.")
PROMPT_LEAK_NOTE = ("CLEAN: your previous draft contained instruction text or a label (like 'ask the group:' or "
                    "'say:'). Write ONLY the chat message itself, exactly as the student would type it.")
ON_TASK = {"on_task_claim", "explain", "disagree"}
TEMPLATED_ACTS = frozenset({"greeting", "backchannel", "agree", "idk", "refusal"})
CONTENT_ACTS = frozenset({"on_task_claim", "explain", "disagree", "question", "press"})
# content-free last-resort lines per plan act (no KC content, no prompt labels)
MINIMAL_ACT_LINES = {
    "question": ["anyone know?", "any ideas?", "hmm what do we think?", "does anyone know?"],
    "press": ["wait why?", "how do you know?", "why tho?"],
    "disagree": ["hmm not sure about that", "wait idk about that", "hmm idk"],
    "explain": ["hmm not sure how to say it", "idk how to explain it"],
    "on_task_claim": ["hmm not sure", "idk tbh"],
    "_other": ["hmm", "ok", "not sure"],
}


COHERENCE_NOTE = ("COHERENT: your previous draft did not hang together ({reason}). Write the line again so it does "
                  "exactly what SAY THIS CONTENT asks: a belief line states only the belief (never the idea that "
                  "refutes it); a challenge speaks against the belief it answers, with the content given, and never "
                  "repeats what the other person said.")
INSTRUCTION_ECHO_N = 5      # words in a row copied from the instruction frame = an instruction echo


def needs_coherence(plan: ContentPlan) -> bool:
    """Challenge / disagreement / belief lines (0.5.2): the only lines the Haiku coherence check sees."""
    if plan.closing is not None or plan.draft is not None or plan.concedes or plan.tag or plan.opener:
        return False
    if plan.claim == "misconception" and plan.misconception_id:
        return True
    if plan.act == "disagree":
        return True
    return bool(plan.addresses_misconception) and plan.act != "press" and plan.claim in ("fact", "correct")


_PAYLOADS: dict[int, tuple] = {}


def instruction_payloads(task: Task) -> list[str]:
    """Content strings an instruction may carry (beliefs, claims, facts, gists, questions, ...): removed
    from the instruction before the echo check, so repeating the CONTENT is never an echo."""
    hit = _PAYLOADS.get(id(task))
    if hit is not None and hit[0] is task:
        return hit[1]
    from ..sim.realizer import fact_gist, step_question
    out = [task.title, task.prompt]
    for st_ in task.steps:
        out += [st_.title, st_.instruction]
        try:
            out.append(step_question(task, next(g.kcs[0] for g in task.gold if g.step == st_.id)))
        except StopIteration:
            pass
    for f in task.facts:
        out += [f.text, f.ask, *f.chat, fact_gist(task, f.id)]
    for k in task.kcs:
        out += [k.name, *k.correct_claims]
    for m in task.misconceptions:
        out += [m.wrong_belief, *m.wrong_answer_template, *m.counterexample, *m.because]
    out = [x for x in dict.fromkeys(out) if x]
    _PAYLOADS[id(task)] = (task, out)
    return out


LONG_NOTE = ("SHORT: your draft was too long and sounded like the material; write it like a student types "
             "in chat, in your own words, at most {n} words.")
CONTENT_MAX_WORDS = 25


def too_long(plan: ContentPlan, text: str, cap: int = CONTENT_MAX_WORDS) -> bool:
    """A content line (fact / claim / belief, not the closing synthesis) longer than `cap` words."""
    if plan.closing is not None or plan.draft == "propose":
        return False
    return plan.claim in ("fact", "correct", "misconception") and len(text.split()) > cap


def is_templated(plan: ContentPlan) -> bool:
    # 0.5.1: openers, afterthoughts, one-word okays, nudges and the short write-up turns are template
    # lines (no Haiku call)
    if plan.opener or plan.tag or plan.brief or plan.nudge or plan.split_tail or plan.draft in ("agree", "typed") \
            or plan.concedes or plan.holds_out or plan.step_move or plan.after_hedge or plan.typo_fix:
        return True
    if plan.closing == "agree":
        return False     # 0.5.3: the okays to the final answer are worded by the model (no fixed bank lines)
    if plan.act in CONTENT_ACTS or plan.claim is not None and plan.claim != "none":
        return False
    if plan.act in TEMPLATED_ACTS:
        return True
    return plan.target_len <= 1


_SENT_END = __import__("re").compile(r"[.!?…](?=\s|$)")
_VOCAB: dict[int, tuple] = {}


def _vocab(task) -> set[str]:
    """Words the chat is made of (task texts, template banks, few-shots), accent-folded."""
    hit = _VOCAB.get(id(task))
    if hit is not None and hit[0] is task:
        return hit[1]
    import re as _re
    from ..models.task import fold
    from ..sim.realizer import FILLER
    texts = [task.title, task.prompt] + [x.title + " " + x.instruction for x in task.steps]
    for f in task.facts:
        texts += [f.text, f.ask, *f.chat]
    for k in task.kcs:
        texts += [k.name, *k.correct_claims]
    for m in task.misconceptions:
        texts += [m.wrong_belief, *m.wrong_answer_template, *m.counterexample]
    texts += [l for v in lang.attr(task, "FILLER", FILLER).values() for l in v]
    words = {w for t in texts for w in _re.findall(r"[a-z]+", fold(t))}
    _VOCAB[id(task)] = (task, words)
    return words


def looks_cut(text: str, task) -> bool:
    """A line whose last word looks cut off ("...continua quebr"): it ends without punctuation, and
    its last word is not a word of the chat vocabulary (or of the line itself) but begins one."""
    import re as _re
    from ..models.task import fold
    t = text.rstrip()
    if not t or t[-1] in ".!?…)\"'" or t[-1].isdigit():
        return False
    words = _re.findall(r"[a-z]+", fold(t))
    if len(words) < 3:
        return False
    last = words[-1]
    if len(last) < 3:
        return False
    vocab = _vocab(task) | set(words[:-1])
    return last not in vocab and any(w.startswith(last) and len(w) > len(last) for w in vocab)


def trim_truncated(text: str) -> str:
    """A line cut by the output cap, trimmed back to complete words: to its last sentence end if that
    keeps >= 3 words, else to its last clause (comma), else without its (probably cut) last word."""
    t = text.rstrip()
    ends = [m.end() for m in _SENT_END.finditer(t)]
    if ends and len(t[:ends[-1]].split()) >= 3:
        return t[:ends[-1]].strip()
    if "," in t and len(t[:t.rindex(",")].split()) >= 3:
        return t[:t.rindex(",")].strip()
    words = t.split()
    return " ".join(words[:-1]) if len(words) > 1 else t


class LLMRealizer:
    def __init__(self, client: LLMClient, models: ModelConfig | None = None, use_llm_guard: bool = True,
                 audit: bool = True, template_deterministic: bool = True):
        self.client = client
        self.models = models or ModelConfig.from_env()
        self.use_llm_guard = use_llm_guard
        self.audit = audit
        self.template_deterministic = template_deterministic
        self.max_content_words = CONTENT_MAX_WORDS     # the engine sets it from SimConfig.content_max_words
        self.coherence_check = True                     # 0.5.2 Haiku coherence check on challenge / belief lines
        self.fallback = TemplateRealizer()
        self.log: list[dict] = []

    # -------------------------------------------------------------- guard
    def leaked_facts(self, task: Task, s: StudentState, plan: ContentPlan, text: str, st=None) -> list[str]:
        """TRUE leaks only: a keyphrase of a fact the speaker does not hold, that
        nobody made visible in the recent chat and that is not part of the speaker's own material."""
        return true_leaks(task, s, plan, text, st)

    def _check(self, task: Task, s: StudentState, plan: ContentPlan, text: str, context: str, st=None) -> tuple[dict, list]:
        """Merged Haiku guard + audit. Facts visible in the recent chat are ALLOWED. Returns
        the JSON and the DO-NOT-KNOW items (id, text) so flagged indices can be logged by id."""
        vis = visible_fact_sources(task, st, s.id)
        dont_items = [(f.id, f.text) for f in task.facts
                      if f.id not in s.known_facts and f.id not in plan.fact_ids and f.id not in vis]
        dont_items += [(k, task.kc(k).name) for k, v in s.kcs.items() if v.p < 0.6]
        unknown = [f.id for f in task.facts if f.id not in s.known_facts and f.id not in plan.fact_ids
                   and f.id not in vis]
        allowed = [task.fact(f).text for f in plan.fact_ids]
        allowed += [f"{task.fact(f).text} (already said in the chat)" for f in vis if f not in plan.fact_ids]
        req = check_request(self.models.student, task, s, text, allowed, dont_items, unknown, context)
        return self.client.complete_json(req), dont_items

    @staticmethod
    def _llm_items(check: dict, dont_items: list) -> list[dict]:
        """Parse flags: {"id", "quote"} objects; bare indices / ids carry no quote
        and are therefore rejected by the evidence rule."""
        texts = dict(dont_items)
        out = []
        for i in check.get("leaked_items", []) or []:
            quote = None
            if isinstance(i, dict):
                quote, i = i.get("quote"), i.get("id", i.get("item"))
            if isinstance(i, int) and not isinstance(i, bool) and 0 <= i < len(dont_items):
                iid = dont_items[i][0]
            else:
                iid = None if i is None else str(i)
            out.append({"id": iid, "text": texts.get(iid), "quote": quote})
        return out

    @staticmethod
    def _vet_flags(task, s, plan, draft, items, dont_items, st) -> tuple[list[dict], list[dict]]:
        """Evidence rule + own_material / visible_echo / can_state on LLM flags."""
        dont_ids = {d[0] for d in dont_items}
        kept, rejected = [], []
        for it in items:
            ok, why = validate_llm_flag(task, s, plan, draft, it["id"], it["quote"], st, dont_ids)
            (kept if ok else rejected).append({**it, "reason": why})
        return kept, rejected

    def final_ok(self, task: Task, s: StudentState, plan: ContentPlan, text: str, names=None, st=None,
                 rejected=()) -> bool:
        """The checks every final line must pass: no code-side true leak, no prompt labels, and not a
        draft the guard already rejected (normalized comparison)."""
        return (bool(text and text.strip()) and not self.leaked_facts(task, s, plan, text, st)
                and not promptleak.hits(text, names, s.id) and norm(text) not in {norm(r) for r in rejected})

    def minimal_line(self, task: Task, s: StudentState, plan: ContentPlan, rng: RNG, names=None, st=None,
                     rejected=()) -> str:
        """Content-free line consistent with the plan act, from a safe bank (last resort)."""
        lines = lang.attr(task, "MINIMAL_ACT_LINES", MINIMAL_ACT_LINES)
        bank = lines.get(plan.act, lines["_other"])
        ok = [c for c in bank if self.final_ok(task, s, plan, c, names, st, rejected)]
        return rng.choice(ok) if ok else "hmm"

    def _safe_fallback(self, task: Task, s: StudentState, plan: ContentPlan, rng: RNG, names, st=None,
                       rejected=()) -> str:
        """Plan-consistent template wording that passes the final checks:
        questions fall back to the plan's question bank, facts to the templated fact line, claims to
        a voiceable claim of the plan's KC. Never a rejected draft; if no variant is safe, a minimal
        content-free line for the plan act."""
        def ok(c):
            return self.final_ok(task, s, plan, c, names, st, rejected)
        good = [c for c in self.fallback.candidates(plan, s, task, names, st) if ok(c)]
        if good:
            return rng.choice(good)
        alt = self.fallback.safe_line(plan, s, task, rng, names, st, ok=ok)
        return alt if ok(alt) else self.minimal_line(task, s, plan, rng, names, st, rejected)

    # -------------------------------------------------------------- realize
    def variant(self, plan: ContentPlan, speaker: StudentState, task: Task, rng: RNG,
                names: dict[str, str], st=None) -> str:
        """Template wording that passes the leak check (repetition guard's last resort)."""
        return noise(self._safe_fallback(task, speaker, plan, rng, names, st), speaker, rng, st)

    def _draft(self, task, st, speaker, plan, names, rep, rng, strict="", rephrase="", rec: dict | None = None) -> str:
        req = realizer_request(self.models.student, task, st, speaker, plan, names, replicate_idx=rep,
                               strict=strict, rephrase=rephrase)
        resp = self.client.complete(req)
        text = resp.text.strip().strip('"')
        if resp.stop_reason == "max_tokens" or looks_cut(text, task):
            # cut by the output cap (e.g. "...continua quebr"): one regeneration asking for a short,
            # complete sentence (a different request, so not a cache replay), else trimmed
            note = lang.attr(task, "TRUNCATED_NOTE", TRUNCATED_NOTE)
            req2 = realizer_request(self.models.student, task, st, speaker, plan, names, replicate_idx=rep,
                                    strict="\n".join(x for x in (strict, note) if x), rephrase=rephrase)
            resp2 = self.client.complete(req2)
            info = {"draft": text, "regenerated": True, "stop_reason": resp.stop_reason}
            if resp2.stop_reason == "max_tokens" or looks_cut(resp2.text.strip().strip('"'), task):
                text = trim_truncated(resp2.text.strip().strip('"'))
                info["resolution"] = "trimmed"
            else:
                text = resp2.text.strip().strip('"')
                info["resolution"] = "regenerated"
            info["final"] = text
            if rec is not None:
                rec.setdefault("truncated", []).append(info)
                rec["extra_calls"] = rec.get("extra_calls", 0) + 1
        if too_long(plan, text, self.max_content_words):
            # 0.5.1: a content line in student register, not a textbook sentence: one regeneration asking
            # for a shorter line in their own words, else the template wording of the same plan
            note = lang.attr(task, "LONG_NOTE", LONG_NOTE).format(n=min(plan.target_len + 3, self.max_content_words))
            req3 = realizer_request(self.models.student, task, st, speaker, plan, names, replicate_idx=rep,
                                    strict="\n".join(x for x in (strict, note) if x), rephrase=rephrase)
            resp3 = self.client.complete(req3)
            info = {"draft": text, "words": len(text.split()), "cap": self.max_content_words}
            t3 = resp3.text.strip().strip('"')
            if too_long(plan, t3, self.max_content_words) or looks_cut(t3, task):
                text = self.fallback.content(plan, speaker, task, rng, names, st)
                info["resolution"] = "template"
            else:
                text, info["resolution"] = t3, "regenerated"
            info["final"] = text
            if rec is not None:
                rec.setdefault("too_long", []).append(info)
                rec["extra_calls"] = rec.get("extra_calls", 0) + 1
        text = strip_card_announcement(text, rng, task)
        return normalize_chat(text, rng, task)

    def realize(self, plan: ContentPlan, speaker: StudentState, task: Task, rng: RNG,
                names: dict[str, str], st=None, rephrase: str = "") -> str:
        assert st is not None, "LLMRealizer needs the session state for history"
        rec = {"msg": len(st.messages), "speaker": speaker.id, "act": plan.act, "attempts": 0}
        if self.template_deterministic and is_templated(plan):
            rec.update(templated=True, pre_guard_leak=False, post_guard_leak=False)
            self.log.append(rec)
            return self.fallback.realize(plan, speaker, task, rng, names, st)
        rep = len(st.messages)
        ctx = history(st, names, n=4, viewer=speaker.id, task=task)
        text = self._draft(task, st, speaker, plan, names, rep, rng, rephrase=rephrase, rec=rec)
        rec["attempts"] = 1 + rec.get("extra_calls", 0)   # + a "too long" regeneration (0.5.1)
        rec["triggers"], rec["rejected_drafts"] = [], []
        instruction = say_this(task, plan, names, variant=rep + len(speaker.name), speaker=speaker.id)

        def inspect(draft: str, stage: str) -> tuple[dict, dict]:
            """Code leak triggers + prompt labels first; the Haiku check only on a code-clean draft."""
            trig = leak_triggers(task, speaker, plan, draft, st)
            for t in trig:
                rec["triggers"].append({**t, "stage": stage, "detected_by": "code"})
            code = [t for t in trig if t["type"] == "true_leak"]
            scaf = promptleak.hits(draft, names, speaker.id)
            echo = promptleak.echo_hits(draft, instruction, instruction_payloads(task) + list(names.values()),
                                        n=INSTRUCTION_ECHO_N)
            if echo:
                scaf = scaf + ["instruction_echo"]
                rec.setdefault("instruction_echo", []).append({"stage": stage, "draft": draft, "echo": echo})
            check, items = {}, []
            if not code and not scaf and (self.use_llm_guard or self.audit):
                check, dont_items = self._check(task, speaker, plan, draft, ctx, st)
                if check.get("leak") and self.use_llm_guard:
                    items, rejected = self._vet_flags(task, speaker, plan, draft, self._llm_items(check, dont_items),
                                                      dont_items, st)
                    for it in rejected:
                        rec["triggers"].append({"type": "llm_flag_rejected", "stage": stage, "detected_by": "llm",
                                                "fact_id": it["id"], "item_text": it["text"], "quote": it["quote"],
                                                "reason": it["reason"]})
                    for it in items:
                        src = last_mention(task, st, it["id"], exclude=speaker.id) if it["id"] in {f.id for f in task.facts} else None
                        rec["triggers"].append({"type": "llm_flag", "stage": stage, "detected_by": "llm",
                                                "fact_id": it["id"], "item_text": it["text"], "quote": it["quote"],
                                                "speaker_exposed": it["id"] in speaker.known_facts,
                                                "said_in_chat": None if src is None else {
                                                    "msg": src.id, "speaker": src.speaker, "t": round(src.t, 2)}})
            bad = {"code": code, "llm": items, "prompt_leak": scaf}
            if bad["code"] or bad["llm"] or bad["prompt_leak"]:
                rec["rejected_drafts"].append({"stage": stage, "text": draft,
                                               "leak_triggers": [{k: t[k] for k in ("fact_id", "kc", "keyphrases", "type")}
                                                                 for t in code] + [{"fact_id": i["id"], "type": "llm_flag",
                                                                                     "item_text": i["text"]} for i in bad["llm"]],
                                               "prompt_leak": scaf})
            return bad, check

        bad, check = inspect(text, "first")
        leak1 = bool(bad["code"] or bad["llm"])
        rec["pre_guard_leak"] = leak1
        if leak1:
            rec.update(pre_guard_items=[t["fact_id"] for t in bad["code"]] or [i["id"] for i in bad["llm"]],
                       detected_by="code" if bad["code"] else "llm")
        if bad["prompt_leak"]:
            rec["prompt_leak"] = bad["prompt_leak"]
        if leak1 or bad["prompt_leak"]:
            rec["attempts"] = 2
            note = lang.attr(task, "STRICT", STRICT) if leak1 else lang.attr(task, "PROMPT_LEAK_NOTE", PROMPT_LEAK_NOTE)
            text2 = self._draft(task, st, speaker, plan, names, rep, rng, strict=note, rephrase=rephrase, rec=rec)
            rec["attempts"] = 2 + rec.get("extra_calls", 0)
            bad2, check2 = inspect(text2, "regenerated")
            if bad2["code"] or bad2["llm"] or bad2["prompt_leak"]:
                text = self._safe_fallback(task, speaker, plan, rng, names, st,
                                           rejected=[d["text"] for d in rec["rejected_drafts"]])
                rec.update(fallback=True, regen_items=[t["fact_id"] for t in bad2["code"]] or [i["id"] for i in bad2["llm"]],
                           fallback_reason="leak" if (bad2["code"] or bad2["llm"]) else "prompt_leak",
                           fallback_text=text)
                check = {}
            else:
                text, check = text2, check2
        if self.use_llm_guard and self.coherence_check and needs_coherence(plan) and not rec.get("fallback"):
            text, check = self._cohere(task, st, speaker, plan, names, rep, rng, text, check, rec, inspect, rephrase)
        final_leak = self.leaked_facts(task, speaker, plan, text, st)
        if final_leak:  # by construction this never happens; logged, then replaced
            rec["final_true_leak"] = {"text": text, "facts": final_leak}
            text, check = self.minimal_line(task, speaker, plan, rng, names, st), {}
        rec["post_guard_leak"] = bool(self.leaked_facts(task, speaker, plan, text, st))
        assert not rec["post_guard_leak"], f"post-guard true leak: {text!r}"
        if self.audit and check:
            rec["audit"] = self.slippage(plan, check)
        self.log.append(rec)
        return noise(text or "ok", speaker, rng, st)

    # -------------------------------------------------------------- coherence (0.5.2)
    @staticmethod
    def coherence_target(st, speaker: StudentState, plan: ContentPlan) -> str:
        """The line a challenge / belief line answers: its anchor, else the last line by someone else."""
        if plan.anchor_turn is not None and 0 <= plan.anchor_turn < len(st.messages):
            return st.messages[plan.anchor_turn].text
        m = next((m for m in reversed(st.messages) if m.speaker != speaker.id), None)
        return m.text if m is not None else ""

    def _coherent_fallback(self, task, s, plan, rng, names, st, rejected) -> str:
        """Template wording of the plan that passes every final check and the code coherence check: for a
        challenge the refuting fact / claim the planner chose, for a belief its template line."""
        def ok(c):
            return self.final_ok(task, s, plan, c, names, st, rejected) and \
                coherence.coherence_issue(task, st, s.id, plan, c) is None
        good = [c for c in self.fallback.candidates(plan, s, task, names, st) if ok(c)]
        if good:
            return rng.choice(good)
        alt = self.fallback.safe_line(plan, s, task, rng, names, st, ok=ok)
        return alt if ok(alt) else self.minimal_line(task, s, plan, rng, names, st, rejected)

    def _cohere(self, task, st, speaker, plan, names, rep, rng, text, check, rec, inspect, rephrase=""):
        """Code coherence check, then ONE Haiku coherence call (JSON {coherent, reason}); an incoherent
        line is regenerated once with the reason (and checked again), else the template line."""
        target = self.coherence_target(st, speaker, plan)
        info = {"checks": 0, "target": target}
        rec["coherence"] = info
        code = coherence.coherence_issue(task, st, speaker.id, plan, text)
        if code is None:
            v = self.client.complete_json(coherence_request(self.models.student, task, plan, target, text,
                                                            replicate_idx=rep))
            info["checks"] += 1
            if v.get("coherent", True) is not False:
                info["verdict"] = "coherent"
                return text, check
            reason = str(v.get("reason") or "incoherent")[:200]
            info["first"] = {"by": "llm", "reason": reason, "draft": text}
        else:
            reason = code["reason"].replace("_", " ")
            info["first"] = {"by": "code", "reason": code["reason"], "draft": text}
        note = lang.attr(task, "COHERENCE_NOTE", COHERENCE_NOTE).format(reason=reason)
        n0 = rec.get("extra_calls", 0)
        text2 = self._draft(task, st, speaker, plan, names, rep, rng, strict=note, rephrase=rephrase, rec=rec)
        rec["attempts"] = rec.get("attempts", 1) + 1 + rec.get("extra_calls", 0) - n0
        bad2, check2 = inspect(text2, "coherence")
        ok2 = not (bad2["code"] or bad2["llm"] or bad2["prompt_leak"]) and \
            coherence.coherence_issue(task, st, speaker.id, plan, text2) is None
        if ok2:
            v2 = self.client.complete_json(coherence_request(self.models.student, task, plan, target, text2,
                                                             replicate_idx=rep + 1))
            info["checks"] += 1
            ok2 = v2.get("coherent", True) is not False
        if ok2:
            info.update(verdict="regenerated", final=text2)
            return text2, check2
        rejected = [d["text"] for d in rec.get("rejected_drafts", [])] + [text, text2]
        fb = self._coherent_fallback(task, speaker, plan, rng, names, st, rejected)
        info.update(verdict="template", final=fb)
        return fb, {}

    # -------------------------------------------------------------- audit
    @staticmethod
    def slippage(plan: ContentPlan, j: dict) -> dict:
        planned_on_task = plan.act in ON_TASK and plan.claim in ("correct", "fact", "misconception")
        real_m = j.get("expresses_misconception")
        out = {"plan_act": plan.act, "real_act": j.get("act"), "plan_m": plan.misconception_id,
               "real_m": real_m, "plan_correct": plan.is_correct, "real_correct": j.get("is_correct")}
        out["slip_misconception"] = (plan.misconception_id or None) != (real_m or None) and planned_on_task
        out["slip_correctness"] = planned_on_task and plan.is_correct is not None and \
            j.get("is_correct") is not None and plan.is_correct != j.get("is_correct")
        out["slip"] = bool(out["slip_misconception"] or out["slip_correctness"])
        return out

    def trigger_counts(self) -> dict:
        """Leak-guard triggers over all drafts, by type: true_leak (blocked), llm_flag (Haiku check,
        blocked), visible_echo, own_material and question (allowed, not leaks), and llm_flag_rejected
        (Haiku flags without valid evidence, discarded)."""
        out = {"true_leak": 0, "llm_flag": 0, "visible_echo": 0, "own_material": 0, "question": 0,
               "llm_flag_rejected": 0}
        for r in self.log:
            for t in r.get("triggers", []):
                out[t["type"]] = out.get(t["type"], 0) + 1
        return out

    def summary(self) -> dict:
        n = len(self.log) or 1
        llm = [r for r in self.log if not r.get("templated")]
        nl = len(llm) or 1
        aud = [r["audit"] for r in self.log if "audit" in r]
        return {"turns": len(self.log), "templated_turns": len(self.log) - len(llm), "llm_turns": len(llm),
                "pre_guard_leak_rate": sum(r.get("pre_guard_leak", False) for r in self.log) / n,
                "post_guard_leak_rate": sum(r.get("post_guard_leak", False) for r in self.log) / n,
                "pre_guard_leak_rate_llm_turns": sum(r.get("pre_guard_leak", False) for r in llm) / nl,
                "regenerated": sum(r.get("attempts", 0) > 1 for r in self.log),
                "fallbacks": sum(r.get("fallback", False) for r in self.log),
                "fallback_rate_llm_turns": sum(r.get("fallback", False) for r in llm) / nl,
                "slippage_rate": (sum(a["slip"] for a in aud) / len(aud)) if aud else None,
                "leak_triggers_by_type": self.trigger_counts(),
                "fallback_reasons": {k: sum(r.get("fallback_reason") == k for r in self.log)
                                     for k in ("leak", "prompt_leak")},
                "prompt_leak_hits": sum(bool(r.get("prompt_leak")) or bool(r.get("final_prompt_leak")) for r in self.log),
                "truncated_drafts": sum(len(r.get("truncated", [])) for r in self.log),
                "repetition_guard_hits": sum("repetition_guard" in r for r in self.log),
                "coherence_checks": sum(r.get("coherence", {}).get("checks", 0) for r in self.log),
                "coherence_regenerated": sum(r.get("coherence", {}).get("verdict") == "regenerated" for r in self.log),
                "coherence_fallbacks": sum(r.get("coherence", {}).get("verdict") == "template" for r in self.log),
                "instruction_echo_hits": sum(bool(r.get("instruction_echo")) for r in self.log),
                "repetition_guard_regenerations": sum(r.get("repetition_guard", {}).get("regenerated", False)
                                                      for r in self.log)}
