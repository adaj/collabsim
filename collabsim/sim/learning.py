"""Learning parameters: KC learning, exposure, misconception flips and affect (state rules 1-6).

These are code defaults (one model, no external files). A misconception flips under TARGETED
conditions only if the challenge's family is in that misconception's `flip_by` list; challenges
anchored to a different misconception are MISALIGNED (they entrench); anything else touching the KC
is GENERIC. Every turn, whoever wrote it (a simulated student or an external participant), is read
through the same rules; the reader's `deference_peer` trait scales its influence.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class PFlip(BaseModel):
    targeted_a: float = 0.45   # p_targeted = a * (1 - strength) + b
    targeted_b: float = 0.15
    misaligned: float = 0.02
    generic: float = 0.03


class LearningParams(BaseModel):
    # KC learning (rule 3)
    t_received_correct: float = 0.10
    t_self_produced: float = 0.20
    t_incorrect: float = 0.0
    can_state: float = 0.6
    can_apply: float = 0.8
    active_misconception_p_cap: float = 0.55   # mastery cannot reach can_state while the misconception is active
    # exposure (rule 2)
    read_min_engagement: float = 0.3
    read_p_intercept: float = 0.6
    read_p_slope: float = 0.4
    # misconception flips (rule 4)
    p_flip: PFlip = Field(default_factory=PFlip)
    engagement_exponent: float = 0.5
    addressee_multiplier: float = 1.0
    non_addressee_multiplier: float = 0.5
    flip_kc_bonus: float = 0.15
    targeted_nonflip_strength_delta: float = -0.10
    misaligned_entrench_delta: float = 0.05
    relapse_p: float = 0.02
    anchor_window: int = 3
    # affect (rule 5)
    engagement_decay_per_min: float = 0.01
    engagement_replied_bonus: float = 0.05
    frustration_unanswered: float = 0.10
    unanswered_window_turns: int = 4

    def p_targeted(self, strength: float) -> float:
        return self.p_flip.targeted_a * (1.0 - strength) + self.p_flip.targeted_b
