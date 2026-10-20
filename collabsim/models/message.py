"""Chat messages and code-side content plans."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Fine-grained planner acts. The classification labels are ACT_LABEL values; `greeting` and `idk`
# are kept separate because the pilot act mix reports them.
Act = Literal["on_task_claim", "question", "agree", "disagree", "explain", "off_task",
              "interface", "backchannel", "refusal", "greeting", "idk", "press"]

ACT_LABEL: dict[str, str] = {
    "on_task_claim": "on_task_claim", "question": "question", "agree": "agree",
    "disagree": "disagree", "explain": "explain", "off_task": "off_task",
    "interface": "interface", "backchannel": "backchannel",
    "refusal": "refusal", "greeting": "off_task", "idk": "refusal", "press": "question",
}

# acts whose content can move a misconception (others never trigger rule 4)
EPISTEMIC_ACTS = {"on_task_claim", "explain", "disagree", "press"}
ON_TASK_CONTENT_ACTS = {"on_task_claim", "explain", "disagree"}
# acts that carry no task content (fillers); capped by the planner (0.5.0)
CONTENT_FREE_ACTS = {"backchannel", "agree", "idk", "refusal"}

Claim = Literal["correct", "misconception", "fact", "hedge", "none"]
MessageKind = Literal["student", "external"]


class ContentPlan(BaseModel):
    """What a turn says, decided in code (ground truth for the updater). External participants'
    messages carry the plan built from their interpretation."""
    act: Act
    kc: str | None = None
    claim: Claim = "none"
    misconception_id: str | None = None   # misconception EXPRESSED by this turn
    addresses_misconception: str | None = None  # misconception this turn CHALLENGES
    fact_ids: list[str] = Field(default_factory=list)   # facts MENTIONED
    asked_fact: str | None = None                      # fact a question asks about
    revoice: bool = False                              # restating a peer's content
    substantive: bool = False                          # substantive reply to an address
    is_correct: bool | None = None
    anchor_turn: int | None = None        # message id this turn responds to
    addressee: str | None = None          # student / participant id, or None (= group)
    family: str | None = None             # flip family if it challenges a belief
    replies_to_external: bool = False     # reply to an external participant's address
    target_len: int = 4
    topic: str | None = None              # off-task topic (wording only)
    follow_up: bool = False               # follow-up on an earlier unanswered question
    reference: bool = False               # short back-reference to a fact already said
    ref_by: str | None = None             # who said it before: the speaker (own) or a peer
    claim_idx: int | None = None          # which of the KC's correct_claims a "correct" claim voices
    intent: str | None = None             # interface / procedural intent key
    answers_intent: bool = False          # answers a peer's interface question (intent = its key)
    closing: str | None = None            # closing synthesis: "propose" | "agree" | "adjust"
    continuation: bool = False            # continues the speaker's own previous message
    pushback: bool = False                # defends a challenged belief once, with a reason (0.5.0)
    opener: bool = False                  # one-word lead-in ("pera", "gente") before one's own contribution (0.5.0)
    partial: bool = False                 # closing proposal under time / turn pressure (task incomplete; 0.5.0)
    ideas: list[str] = Field(default_factory=list)  # key ideas a partial closing proposal sums up (0.5.0)
    nudge: bool = False                   # a short nudge to the group after a long silence ("e ai gente?"; 0.5.1)
    step_q: bool = False                  # asks the group the current activity question (0.5.1)
    restate: bool = False                 # restates one's own point after it was disputed (0.5.1)
    brief: bool = False                   # a one-word okay to the closing proposal ("blz"; 0.5.1)
    # writing up the current activity's answer (0.5.1): "propose" (the hypothesis text) | "agree" |
    # "confidence" (the confidence bar) | "typed" (the proposer typed / sent it)
    draft: str | None = None
    concedes: str | None = None           # gives up this (own) misconception after being challenged (0.5.1)
    tag: bool = False                     # a short afterthought to one's own statement ("ne?", "faz sentido?"; 0.5.1)
    rationale: bool = False               # a holder's reason for their belief, answering "pq vc acha isso?" (0.5.2)
    answers_ping: bool = False            # answers a classmate who asked them by name / pressed them (0.5.2)
    split_tail: str | None = None         # the second half of one's own thought, typed as a separate message (0.5.2)
    unsure: bool = False                  # answers a press / ping saying they are not sure (no content; 0.5.2)
    reason_turn: int | None = None        # the refuting message a concession gives in to, quoted (0.5.3)
    holds_out: bool = False               # okays the final answer with a hedge, belief unresolved (0.5.3)
    step_move: bool = False               # marks moving on after a solved step ("bora pra 2?"; 0.5.3)
    after_hedge: bool = False             # a short hedge after one's own statement ("acho", "se eu entendi direito"; 0.5.3)
    typo_fix: bool = False                # "*superficie": corrects a misspelling of one's previous line (0.5.3)


class Message(BaseModel):
    id: int
    t: float
    speaker: str                         # student id (s1..) or external participant id (e1..)
    kind: MessageKind
    text: str
    addressee: str | None = None
    plan: ContentPlan | None = None      # hidden
    labels: dict = Field(default_factory=dict)  # rule-1 output
