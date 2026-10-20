"""Model policy, prices and per-role limits.

Every LLM role uses claude-haiku-5-5. The allowlist below is enforced when a ModelConfig, an
LLMRequest or a backend call is constructed with any other model. Env overrides
(COLLABSIM_STUDENT_MODEL, COLLABSIM_JUDGE_MODEL) may only select allowed models.

Prices from platform.claude.com/docs/en/about-claude/pricing (Haiku 5.5, <=100k-token tier).
"""
from __future__ import annotations

import os

from pydantic import BaseModel, field_validator

ALLOWED_MODELS: frozenset[str] = frozenset({"claude-haiku-5-5"})
DEFAULT_MODEL = "claude-haiku-5-5"


class ModelNotAllowed(ValueError):
    """Raised for any non-allowlisted model. Inherits ValueError so pydantic validators surface it."""


def check_model(model: str) -> str:
    if model not in ALLOWED_MODELS:
        raise ModelNotAllowed(
            f"model {model!r} is not allowed: this project uses claude-haiku-5-5 for every role "
            f"(allowed: {sorted(ALLOWED_MODELS)}).")
    return model


class Price(BaseModel):
    input: float
    cache_write_5m: float
    cache_read: float
    output: float
    batch_discount: float = 0.5


PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(input=0.10, cache_write_5m=0.125, cache_read=0.01, output=0.50),
}

# Tight output caps per role (tokens). Truncations (stop_reason == "max_tokens") are counted by the
# usage meter; a truncated chat line is regenerated or trimmed (llm/realizer.py).
MAX_TOKENS: dict[str, int] = {
    "realize": 100,       # one chat line <= ~30 words (pt-BR needs more tokens per word than English)
    "check": 130,         # merged leakage guard + slippage audit JSON (~45 tokens typical; + quoted evidence per flag)
    "coherence": 60,      # 0.5.2 coherence check JSON on challenge / belief lines ({"coherent", "reason"} ~25 tokens)
    "interpret": 150,     # one external participant's message -> ContentPlan-like JSON (~60-110 tokens)
    "task_draft": 6000,   # a whole draft TaskSpec JSON (~2.5-4.5k tokens)
    "ping": 16,           # connectivity check
}

# Minimum cacheable prefix for Haiku 5.5 (tokens). Shorter prefixes are silently not cached.
MIN_CACHEABLE_TOKENS = 512


class ModelConfig(BaseModel):
    student: str = DEFAULT_MODEL   # realizer, guard/audit
    judge: str = DEFAULT_MODEL     # interpretation of external participants' messages, task drafts

    @field_validator("student", "judge")
    @classmethod
    def _allowed(cls, v: str) -> str:
        return check_model(v)

    def __init__(self, **kw):
        for v in kw.values():  # raise ModelNotAllowed directly (not wrapped in a ValidationError)
            check_model(v)
        super().__init__(**kw)

    @classmethod
    def from_env(cls) -> "ModelConfig":
        return cls(student=check_model(os.environ.get("COLLABSIM_STUDENT_MODEL", DEFAULT_MODEL)),
                   judge=check_model(os.environ.get("COLLABSIM_JUDGE_MODEL", DEFAULT_MODEL)))

    def manifest(self) -> dict:
        return {**self.model_dump(), "policy": "haiku-only (allowlist enforced)"}


def price_for(model: str) -> Price:
    return PRICES[check_model(model)]


# Haiku 5.5 request settings (platform migration guide, verified 2026-10-17):
# - sampling parameters are not sent (temperature must be 1 if present; top_k always rejected);
# - adaptive thinking is on by default and counts toward max_tokens -> request it disabled for the
#   short, capped calls this project makes.
THINKING: dict | None = {"type": "disabled"}


def request_extras() -> dict:
    return {"thinking": THINKING} if THINKING else {}
