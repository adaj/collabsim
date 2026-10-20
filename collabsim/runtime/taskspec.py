"""Task library, TaskSpec parsing, validation and Haiku drafting.

A TaskSpec is the JSON form of `collabsim.models.task.Task` (one language). A bilingual spec is
{"languages": {"pt-BR": <TaskSpec>, "en": <TaskSpec>}} with the same ids in every variant.

`validate_spec` = schema (pydantic) + consistency checks the simulator relies on:
  errors   - schema errors; jigsaw with < 2 roles; a misconception no card refutes (no `refutes` entry
             naming it, or the refuting facts are on no card); a misconception refuted by EVERY role's
             card (no student could ever hold it); a fact or KC without keyphrases (the leak guard and
             fact detection need them); a fact's chat lines not containing one of its keyphrases; a
             fact on no card;
  warnings - steps without explicit key ideas (one idea per gold KC is then assumed); a KC correct
             claim containing another KC's fact keyphrases; a keyphrase that occurs in the public task
             text (it would leak through the prompt); no misconceptions; keys of older task files this
             version ignores (quiz, transfer) and flip families it no longer has (dropped).
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from ..lang import SUPPORTED, normalize
from ..llm.client import LLMClient, LLMRequest
from ..llm.config import MAX_TOKENS, ModelConfig
from ..models.task import Task, fold, list_tasks, load_task, task_languages


DEFAULT_TASK = "sci-enzyme-01"   # used when a session config names neither task_id nor task


# ------------------------------------------------------------------ library
def library() -> list[dict]:
    out = []
    for tid in list_tasks():
        langs = task_languages(tid)
        titles = {lg: load_task(tid, language=lg).title for lg in langs}
        out.append({"id": tid, "title": titles.get("pt-BR") or titles.get("en"), "titles": titles,
                    "languages": langs, "domain": load_task(tid, language=langs[0]).domain,
                    "n_roles": len(load_task(tid, language=langs[0]).jigsaw)})
    return out


def resolve_task(task_id: str | None, spec: dict | None, language: str) -> Task:
    """The Task a session runs: a library task in `language`, or a posted TaskSpec (single-language,
    or {"languages": {...}} from which the `language` variant is taken). Raises ValueError."""
    lang = normalize(language)
    if spec is not None:
        if "languages" in spec and isinstance(spec["languages"], dict):
            variants = {normalize(k): v for k, v in spec["languages"].items()}
            if lang not in variants:
                raise ValueError(f"task spec has no {lang!r} variant (has {sorted(variants)})")
            spec = dict(variants[lang])
        spec = dict(spec)
        spec.setdefault("language", lang)
        if normalize(spec["language"]) != lang:
            raise ValueError(f"task spec language {spec['language']!r} != session language {lang!r}")
        spec["language"] = lang
        rep = validate_spec(spec)
        if not rep["valid"]:
            raise ValueError("invalid task spec: " + "; ".join(rep["errors"][:8]))
        return Task.model_validate(spec)
    if not task_id:
        task_id = DEFAULT_TASK
    if task_id not in list_tasks():
        raise ValueError(f"unknown task_id {task_id!r} (library: {list_tasks()})")
    if lang not in task_languages(task_id):
        raise ValueError(f"task {task_id!r} has no {lang!r} version (has {task_languages(task_id)})")
    return load_task(task_id, language=lang)


# ------------------------------------------------------------------ validation
def _schema_errors(e: ValidationError) -> list[str]:
    out = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err.get("loc", ()))
        msg = err.get("msg", "")
        out.append(f"{loc}: {msg}" if loc else msg)
    return out


def validate_spec(spec: Any) -> dict:
    if isinstance(spec, dict) and "languages" in spec and isinstance(spec["languages"], dict):
        reps = {k: validate_spec({**v, "language": v.get("language", k)}) for k, v in spec["languages"].items()}
        errors = [f"[{k}] {e}" for k, r in reps.items() for e in r["errors"]]
        warnings = [f"[{k}] {w}" for k, r in reps.items() for w in r["warnings"]]
        ids = {k: r.get("ids") for k, r in reps.items() if r["valid"]}
        if len({json.dumps(v, sort_keys=True) for v in ids.values()}) > 1:
            errors.append("language variants must share the same fact / KC / misconception / step ids")
        return {"valid": not errors, "errors": errors, "warnings": warnings, "languages": sorted(reps)}
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(spec, dict):
        return {"valid": False, "errors": ["task spec must be a JSON object"], "warnings": []}
    lang = spec.get("language", "en")
    try:
        normalize(lang)
    except ValueError as e:
        errors.append(str(e))
    from ..models.task import FLIP_TYPES, LEGACY_TASK_KEYS
    for k in LEGACY_TASK_KEYS:
        if k in spec:
            warnings.append(f"key {k!r} is not used by this version and is ignored")
    for m in spec.get("misconceptions") or []:
        if isinstance(m, dict) and isinstance(m.get("flip_by"), list):
            old = [x for x in m["flip_by"] if x not in FLIP_TYPES]
            if old:
                warnings.append(f"misconception {m.get('id')}: flip_by values {old} are not used by this version (dropped)")
    try:
        t = Task.model_validate({**spec, "language": normalize(lang) if not errors else "en"})
    except ValidationError as e:
        return {"valid": False, "errors": errors + _schema_errors(e), "warnings": warnings}
    except (AssertionError, ValueError) as e:
        return {"valid": False, "errors": errors + [str(e)], "warnings": warnings}
    on_card = {f for fs in t.jigsaw.values() for f in fs}
    for f in t.facts:
        if f.id not in on_card:
            errors.append(f"fact {f.id} is on no jigsaw card")
        if not f.keyphrases:
            errors.append(f"fact {f.id} has no keyphrases (leak guard / fact detection need them)")
        elif not any(f.id in t.facts_mentioned(c) for c in f.chat):
            errors.append(f"fact {f.id}: no chat line contains one of its keyphrases")
    for k in t.kcs:
        if not k.keyphrases and not any(f.keyphrases for f in t.facts if f.kc == k.id):
            errors.append(f"KC {k.id} has no keyphrases (neither its own nor its facts')")
        for c in k.correct_claims:
            other = [f for f in t.facts_mentioned(c) if t.fact(f).kc != k.id]
            if other:
                warnings.append(f"KC {k.id} correct claim {c!r} carries keyphrases of other-KC facts {other}")
    for m in t.misconceptions:
        refuters = [f for f, ms in t.refutes.items() if m.id in ms]
        if not refuters:
            errors.append(f"misconception {m.id} is refuted by no card fact (add it to `refutes`)")
        elif not set(refuters) & on_card:
            errors.append(f"misconception {m.id}: its refuting facts {refuters} are on no card")
        else:
            holders = [r for r, fs in t.jigsaw.items() if m.id not in t.refuted_by(fs)]
            if not holders:
                errors.append(f"misconception {m.id} is refuted by every role's card (no student can hold it)")
        if not any(m.id in t.misconceptions_expressed(w) for w in m.wrong_answer_template):
            warnings.append(f"misconception {m.id}: wrong_answer_template wording is not detected as voicing it")
        if not m.counterexample:
            warnings.append(f"misconception {m.id} has no counterexample")
    public = t.public_text()
    for f in t.facts:
        for kp in f.keyphrases:
            if fold(kp) in public:
                warnings.append(f"fact {f.id} keyphrase {kp!r} occurs in the public task text")
    for g in t.gold:
        if not g.key_ideas:
            warnings.append(f"step {g.step} has no key_ideas (one idea per gold KC is assumed)")
    if not t.misconceptions:
        warnings.append("no misconceptions: students will have nothing to be wrong about")
    ids = {"facts": sorted(f.id for f in t.facts), "kcs": sorted(k.id for k in t.kcs),
           "misconceptions": sorted(m.id for m in t.misconceptions), "steps": [s.id for s in t.steps]}
    return {"valid": not errors, "errors": errors, "warnings": warnings, "ids": ids, "language": t.language,
            "n_roles": len(t.jigsaw)}


# ------------------------------------------------------------------ Haiku drafting
SCHEMA_GUIDE = """TaskSpec JSON schema (all ids lowercase snake_case; every list non-empty unless noted):
{
 "id": str, "language": "<LANG>", "domain": str, "title": str, "prompt": str (public instructions),
 "steps": [{"id": "s1", "title": str, "instruction": str}],                       # 2-4 steps
 "kcs": [{"id": "kc_x", "name": str, "keyphrases": [str], "correct_claims": [str, str]}],  # one KC per step
 "facts": [{"id": "f_x", "kc": "kc_x", "text": str, "chat": [str, str], "ask": str, "keyphrases": [str]}],  # 2 per KC
 "jigsaw": {"A": ["f_..", "f_.."], "B": [...], "C": [...]},                        # N_ROLES roles; each fact on exactly one card
 "refutes": {"f_x": ["m_y"]},                                                      # fact -> misconceptions it contradicts
 "gold": [{"step": "s1", "kcs": ["kc_x"], "answer": str,
           "key_ideas": [{"id": "s1_idea", "kc": "kc_x", "facts": ["f_x"], "claims": [0]}]}],
 "misconceptions": [{"id": "m_x", "kc": "kc_x", "trigger": "s1", "wrong_belief": str,
                     "wrong_answer_template": [str, str], "flip_by": ["counterexample", "evidence_from_material",
                     "press_for_reasoning", "peer_explanation"],
                     "counterexample": [str]}]                                     # one per KC
}
Rules the simulator checks in code:
- keyphrases are short lowercase markers (2-4 words) of the answer content; EVERY fact's "chat" lines must
  contain one of that fact's keyphrases verbatim; keyphrases must NOT appear in title/prompt/steps.
- "chat" lines are how a teenager would say the fact in a chat (casual, lowercase); "ask" is how a student
  WITHOUT the fact would ask about it (a question that does not give the answer away).
- correct_claims are chat-register correct statements of the KC; wrong_answer_template lines voice the
  misconception in chat register; counterexample lines are chat-register challenges to it.
- every misconception must be refuted (via "refutes") by at least one card fact, and must NOT be refuted by
  every card (some role must be able to hold it).
- key_ideas: each lists the card facts and/or correct-claim indices that express it."""


def draft_request(model: str, description: str, language: str, n_roles: int = 3) -> LLMRequest:
    lang = normalize(language)
    from .. import lang as L
    label = getattr(L.pack(lang), "label", "English")
    prefix = ("You design small-group jigsaw collaborative-learning tasks for a research simulator of student "
              "chats. You output ONE JSON object (a TaskSpec), nothing else.\n\n"
              + SCHEMA_GUIDE.replace("<LANG>", lang).replace("N_ROLES", str(n_roles)))
    suffix = (f"Write a TaskSpec in {label} (\"language\": \"{lang}\"), with exactly {n_roles} jigsaw roles, for "
              f"this activity description:\n{description}\nJSON:")
    return LLMRequest(model=model, purpose="task_draft", prefix=prefix, suffix=suffix, temperature=0.0,
                      max_tokens=MAX_TOKENS["task_draft"], json_mode=True,
                      meta={"description": description, "language": lang, "n_roles": n_roles})


def draft_task(client: LLMClient, description: str, language: str, n_roles: int = 3,
               models: ModelConfig | None = None) -> dict:
    """Haiku drafts a TaskSpec; it is validated and RETURNED FOR REVIEW (never auto-used)."""
    models = models or ModelConfig.from_env()
    resp = client.complete(draft_request(models.judge, description, language, n_roles))
    draft = client.parse_json(resp.text) or None
    rep = validate_spec(draft) if draft else {"valid": False, "errors": ["the model returned no parsable JSON"],
                                              "warnings": []}
    return {"draft": draft, "validation": rep, "truncated": resp.stop_reason == "max_tokens",
            "usage": resp.usage.model_dump()}


def mock_draft(req) -> str:
    """Offline stand-in: the library science task in the requested language, re-titled with the
    description (deterministic, valid)."""
    m = req.meta
    lang = m["language"] if m["language"] in SUPPORTED else "en"
    t = load_task("sci-enzyme-01", language=lang).model_dump(mode="json")
    import hashlib
    t["id"] = "draft-" + hashlib.sha256((m["description"] or "").encode()).hexdigest()[:6]
    t["title"] = (m["description"] or "draft")[:80]
    return json.dumps(t, ensure_ascii=False)
