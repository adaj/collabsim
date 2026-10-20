"""Hidden student state. Never shown in the chat; exported only in the research record."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Role = str  # jigsaw role name (any number of roles)


class KCState(BaseModel):
    p: float = Field(ge=0.0, le=1.0)
    produced_explanation: bool = False
    evidence_seen: list[str] = Field(default_factory=list)

    @property
    def can_state(self) -> bool:
        return self.p >= 0.6

    @property
    def can_apply(self) -> bool:
        return self.p >= 0.8 and self.produced_explanation


class MisconceptionState(BaseModel):
    id: str
    kc: str
    active: bool = True
    strength: float = Field(ge=0.0, le=1.0)
    voiced: bool = False
    flipped_at: float | None = None
    relapses: int = 0


class Traits(BaseModel):
    talkativeness: float
    latency_scale: float
    burstiness: float
    off_task: float
    lurk_until_min: float
    deference_peer: float
    interface_confusion: float
    verbosity: float
    typo_rate: float
    lowercase: bool


class Affect(BaseModel):
    engagement: float = 0.8
    frustration: float = 0.0
    confidence: dict[str, float] = Field(default_factory=dict)


class StudentState(BaseModel):
    id: str
    name: str
    role: Role
    kcs: dict[str, KCState]
    misconceptions: list[MisconceptionState] = Field(default_factory=list)
    jigsaw_facts: list[str]
    known_facts: list[str]
    shared_facts: list[str] = Field(default_factory=list)
    traits: Traits
    affect: Affect = Field(default_factory=Affect)
    last_progress_t: float = 0.0
    off_task_topics: list[str] = Field(default_factory=list)  # personal chatter topics (wording only)

    def active_misconception(self, kc: str) -> MisconceptionState | None:
        return next((m for m in self.misconceptions if m.kc == kc and m.active), None)

    def misconception(self, mid: str) -> MisconceptionState | None:
        return next((m for m in self.misconceptions if m.id == mid), None)
