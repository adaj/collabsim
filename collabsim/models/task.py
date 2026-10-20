"""Task schema: steps, KCs, card facts, jigsaw roles, gold key ideas and misconceptions."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

FlipType = Literal["counterexample", "peer_explanation", "evidence_from_material", "press_for_reasoning"]
FLIP_TYPES: tuple[str, ...] = ("counterexample", "peer_explanation", "evidence_from_material", "press_for_reasoning")
# keys of older task files that this version ignores (validate_spec warns about them)
LEGACY_TASK_KEYS = ("quiz", "transfer")

TASK_DIR = Path(__file__).resolve().parents[1] / "tasks"
_STOP = {"i", "think", "its", "it", "the", "a", "so", "is", "to", "and", "but", "that", "thats", "just",
         # pt-BR (accent-folded): "eu acho que e so pra ..." (simulator service, 2026-10-18)
         "eu", "acho", "que", "e", "o", "os", "as", "um", "uma", "de", "do", "da", "pra", "para", "mas",
         "entao", "tipo", "ne", "isso", "so"}


def fold(text: str) -> str:
    """Lowercase + strip accents (no-op for ASCII English): every keyphrase / wording match in the
    task model runs on folded text, so pt-BR keyphrases may be written with or without accents."""
    import unicodedata
    t = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in t if not unicodedata.combining(c))


def _tokens(text: str) -> list[str]:
    import re
    return [w for w in re.findall(r"[a-z0-9/:.]+", fold(text).replace("'", "")) if w not in _STOP]


class KC(BaseModel):
    id: str
    name: str
    correct_claims: list[str] = Field(min_length=1, description="chat-register correct statements")
    keyphrases: list[str] = Field(default_factory=list, description="lowercase answer-content markers (leak detection)")


class Fact(BaseModel):
    id: str
    kc: str
    text: str
    chat: list[str] = Field(min_length=1, description="chat-register phrasings")
    ask: str = Field(description="how a student who lacks it would ask")
    keyphrases: list[str] = Field(default_factory=list, description="lowercase content markers (leak detection)")


class Step(BaseModel):
    id: str
    title: str
    instruction: str


class KeyIdea(BaseModel):
    """A key idea of a gold step (rule 6): it counts as voiced when a student states
    one of its card facts, or one of the listed correct claims of its KC, with correct content."""
    id: str
    kc: str
    facts: list[str] = Field(default_factory=list)
    claims: list[int] = Field(default_factory=list, description="indices into the KC's correct_claims")


class GoldStep(BaseModel):
    step: str
    kcs: list[str]
    answer: str
    key_ideas: list[KeyIdea] = Field(default_factory=list,
                                     description="empty = one idea per gold KC (any of its facts or claims)")


class MisconceptionSpec(BaseModel):
    id: str
    kc: str
    trigger: str = Field(description="step id at which the belief is voiced")
    wrong_belief: str
    wrong_answer_template: list[str] = Field(min_length=1)
    flip_by: list[FlipType] = Field(min_length=1)
    counterexample: list[str] = Field(default_factory=list, description="chat-register counterexamples")
    # 0.5.2 coherence fields (all optional; derived when absent)
    keyphrases: list[str] = Field(default_factory=list, description=(
        "lowercase markers of ASSERTING the belief (matched negation-aware: 'nao se gasta' does not hit "
        "'se gasta'); used by the contradiction check"))
    refuting_claims: list[int] | None = Field(default=None, description=(
        "indices into the KC's correct_claims that refute this belief (None: claims that mention a "
        "refuting fact's keyphrase, else all of them)"))
    because: list[str] = Field(default_factory=list, description=(
        "chat-register reasons a holder gives when asked why (invented; 'pq quanto mais usa mais gasta')"))

    @field_validator("flip_by", mode="before")
    @classmethod
    def _drop_legacy_families(cls, v):
        """Older task files list families this version no longer has; they are dropped (validate_spec
        warns). If nothing is left, any peer challenge can flip the belief."""
        if not isinstance(v, list):
            return v
        kept = [x for x in v if x in FLIP_TYPES]
        return kept or list(FLIP_TYPES)


class Task(BaseModel):
    """One task in ONE language (`language`). A bilingual library task is two Task files with the same
    ids (tasks/<id>.yaml = en, tasks/pt-BR/<id>.yaml = pt-BR). Jigsaw roles are any number (>= 2) of
    named cards; students cycle through them (groups of 2-10)."""
    id: str
    language: str = "en"
    domain: str
    title: str
    prompt: str
    steps: list[Step] = Field(min_length=1)
    kcs: list[KC] = Field(min_length=1)
    facts: list[Fact] = Field(min_length=1)
    jigsaw: dict[str, list[str]] = Field(description="role name -> card fact ids (>= 2 roles)")
    gold: list[GoldStep]
    misconceptions: list[MisconceptionSpec] = Field(default_factory=list)
    refutes: dict[str, list[str]] = Field(default_factory=dict,
                                          description="fact id -> misconception ids the fact contradicts")

    @model_validator(mode="after")
    def _check(self) -> "Task":
        kc_ids = {k.id for k in self.kcs}
        step_ids = {s.id for s in self.steps}
        fact_ids = {f.id for f in self.facts}
        m_ids = {m.id for m in self.misconceptions}
        for f in self.facts:
            assert f.kc in kc_ids, f"fact {f.id} unknown kc {f.kc}"
        assert len(self.jigsaw) >= 2, "jigsaw needs >= 2 roles"
        for role, fids in self.jigsaw.items():
            assert fids, f"jigsaw role {role} has no facts"
            for fid in fids:
                assert fid in fact_ids, f"jigsaw {role} unknown fact {fid}"
        assert {g.step for g in self.gold} == step_ids, "gold must cover every step"
        for g in self.gold:
            assert set(g.kcs) <= kc_ids
            for ki in g.key_ideas:
                assert ki.kc in g.kcs, f"key idea {ki.id}: kc {ki.kc} not a gold kc of {g.step}"
                assert all(self.fact(f).kc == ki.kc for f in ki.facts), f"key idea {ki.id}: fact of another kc"
                n = len(self.kc(ki.kc).correct_claims)
                assert all(0 <= i < n for i in ki.claims), f"key idea {ki.id}: bad claim index"
                assert ki.facts or ki.claims, f"key idea {ki.id} is empty"
        for m in self.misconceptions:
            assert m.kc in kc_ids and m.trigger in step_ids, f"bad misconception {m.id}"
        for fid, mids in self.refutes.items():
            assert fid in fact_ids, f"refutes: unknown fact {fid}"
            assert set(mids) <= m_ids, f"refutes: unknown misconception in {fid}"
        return self

    # -- helpers -------------------------------------------------------------
    @property
    def roles(self) -> list[str]:
        return list(self.jigsaw)

    def role_facts(self, role: str) -> list[str]:
        return list(self.jigsaw[role])

    def kc(self, kc_id: str) -> KC:
        return next(k for k in self.kcs if k.id == kc_id)

    def fact(self, fid: str) -> Fact:
        return next(f for f in self.facts if f.id == fid)

    def misconception(self, mid: str) -> MisconceptionSpec:
        return next(m for m in self.misconceptions if m.id == mid)

    def refuted_by(self, fact_ids) -> set[str]:
        """Misconception ids contradicted by any of these facts."""
        return {m for f in fact_ids for m in self.refutes.get(f, [])}

    def refuting_facts(self, mid: str) -> list[str]:
        """Card facts that contradict misconception `mid` (from `refutes`)."""
        return [f for f, ms in self.refutes.items() if mid in ms]

    def refuting_claims(self, mid: str) -> list[int]:
        """Correct-claim indices (of the belief's KC) whose content refutes misconception `mid`."""
        m = self.misconception(mid)
        claims = self.kc(m.kc).correct_claims
        if m.refuting_claims is not None:
            return [i for i in m.refuting_claims if 0 <= i < len(claims)]
        kps = [fold(k) for f in self.refuting_facts(mid) for k in self.fact(f).keyphrases]
        hit = [i for i, c in enumerate(claims) if any(k in fold(c) for k in kps)]
        return hit or list(range(len(claims)))

    def refuting_phrases(self, mid: str) -> list[str]:
        """Keyphrases that state the refutation of `mid`: the refuting facts' keyphrases, plus the KC
        keyphrases that occur in a refuting claim."""
        m = self.misconception(mid)
        out = [k for f in self.refuting_facts(mid) for k in self.fact(f).keyphrases]
        claims = [fold(self.kc(m.kc).correct_claims[i]) for i in self.refuting_claims(mid)]
        out += [k for k in self.kc(m.kc).keyphrases if any(fold(k) in c for c in claims)]
        return list(dict.fromkeys(out))

    def gold_for(self, step_id: str) -> GoldStep:
        return next(g for g in self.gold if g.step == step_id)

    def key_ideas(self, step_id: str) -> list[KeyIdea]:
        g = self.gold_for(step_id)
        if g.key_ideas:
            return list(g.key_ideas)
        return [KeyIdea(id=f"{step_id}:{k}", kc=k, facts=[f.id for f in self.facts if f.kc == k],
                        claims=list(range(len(self.kc(k).correct_claims)))) for k in g.kcs]

    def all_key_ideas(self) -> list[KeyIdea]:
        return [ki for s in self.steps for ki in self.key_ideas(s.id)]

    def ideas_of(self, kc: str | None, fact_ids=(), claim_idx: int | None = None) -> set[str]:
        """Key ideas covered by correct content: card facts and/or correct claim `claim_idx` of `kc`."""
        out = set()
        for ki in self.all_key_ideas():
            if any(f in ki.facts for f in fact_ids):
                out.add(ki.id)
            if claim_idx is not None and ki.kc == kc and claim_idx in ki.claims:
                out.add(ki.id)
        return out

    def public_text(self) -> str:
        return fold(" ".join([self.title, self.prompt] + [s.title + " " + s.instruction for s in self.steps]))

    def content_kcs(self, text: str) -> list[str]:
        """KCs whose answer content (fact or KC keyphrases) appears in `text` (code, no LLM)."""
        low = fold(text)
        out = []
        for k in self.kcs:
            phrases = list(k.keyphrases) + [p for f in self.facts if f.kc == k.id for p in f.keyphrases]
            if any(fold(p) in low for p in phrases):
                out.append(k.id)
        return out

    def misconceptions_expressed(self, text: str, threshold: float = 0.7) -> list[str]:
        """Misconceptions whose wrong-answer wording is (mostly) contained in `text`."""
        tt = set(_tokens(text))
        out = []
        for m in self.misconceptions:
            for w in m.wrong_answer_template:
                wt = set(_tokens(w))
                if wt and len(wt & tt) / len(wt) >= threshold:
                    out.append(m.id)
                    break
        return out

    def misconceptions_addressed(self, text: str, threshold: float = 0.7) -> list[str]:
        """Misconceptions whose counterexample wording is (mostly) contained in `text`
        (code-side content check used to detect MISALIGNED interventions)."""
        tt = set(_tokens(text))
        out = []
        for m in self.misconceptions:
            for c in m.counterexample:
                ct = set(_tokens(c))
                if ct and len(ct & tt) / len(ct) >= threshold:
                    out.append(m.id)
                    break
        return out

    def facts_mentioned(self, text: str) -> list[str]:
        low = fold(text)
        return [f.id for f in self.facts if any(fold(k) in low for k in f.keyphrases)]

    def public_view(self) -> dict:
        """The public part of the task (no gold, misconceptions or facts)."""
        return {"id": self.id, "title": self.title, "instructions": self.prompt,
                "steps": [s.model_dump() for s in self.steps]}


def load_task(task_id: str, task_dir: Path = TASK_DIR, language: str = "en") -> Task:
    """English tasks live in tasks/<id>.yaml; other languages in tasks/<lang>/<id>.yaml."""
    from ..lang import normalize
    lang = normalize(language)
    path = task_dir / f"{task_id}.yaml" if lang == "en" else task_dir / lang / f"{task_id}.yaml"
    t = Task.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if t.language != lang:
        raise ValueError(f"task file {path} declares language {t.language!r}, expected {lang!r}")
    return t


def task_languages(task_id: str, task_dir: Path = TASK_DIR) -> list[str]:
    from ..lang import SUPPORTED
    out = []
    for lang in SUPPORTED:
        p = task_dir / f"{task_id}.yaml" if lang == "en" else task_dir / lang / f"{task_id}.yaml"
        if p.exists():
            out.append(lang)
    return out


def list_tasks(task_dir: Path = TASK_DIR) -> list[str]:
    return sorted(p.stem for p in task_dir.glob("*.yaml"))
