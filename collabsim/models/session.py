"""Full mutable session state (everything except RNG streams)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from .message import Message
from .student import StudentState


class PendingQuestion(BaseModel):
    msg_id: int
    asker: str
    kc: str | None = None
    turns_waited: int = 0


class OpenAddress(BaseModel):
    """A message from an external participant directed at a student (by name) that the student has
    not responded to yet."""
    msg_id: int
    t: float
    source: str                     # external participant id (e1..)
    anchor_turn: int | None = None
    turns_waited: int = 0
    floor_decided: bool = False


class PlannedEvent(BaseModel):
    t: float
    speaker: str
    forced: bool = False      # the group-silence cap was reached: this turn must say something (0.5.1)


class SessionState(BaseModel):
    session_id: str
    task_id: str
    seed: int
    t: float = 0.0
    students: list[StudentState]
    messages: list[Message] = Field(default_factory=list)
    n_steps: int = 3
    step_idx: int = 0
    step_started_t: float = 0.0
    step_started_msg: int = -1         # id of the message that solved the previous step (0.5.1)
    step_correct: dict[str, int] = Field(default_factory=dict)  # kc -> correct claims this step
    step_speakers: list[str] = Field(default_factory=list)      # speakers of those claims
    steps_solved: list[dict] = Field(default_factory=list)
    ideas_voiced: dict[str, int] = Field(default_factory=dict)  # key idea id -> first msg voicing it (rule 6)
    last_progress_t: float = 0.0
    last_progress_msg: int = -1        # id of the last message that moved the task forward (0.5.0)
    addressed_at: dict[str, float] = Field(default_factory=dict)
    open_address: dict[str, OpenAddress] = Field(default_factory=dict)
    pending_questions: list[PendingQuestion] = Field(default_factory=list)
    interface_until: dict[str, float] = Field(default_factory=dict)
    greeted: list[str] = Field(default_factory=list)
    last_affect_t: float = 0.0
    student_turns: int = 0
    skipped_turns: int = 0             # planned turns the student skipped (nothing to say; 0.5.0)
    quiet_until: dict[str, float] = Field(default_factory=dict)  # a student who skipped stays quiet until then
    pending_contribution: dict | None = None  # {"speaker", "plan"}: content that follows the speaker's opener
    planned: PlannedEvent | None = None
    timeline: list[dict] = Field(default_factory=list)
    flip_log: list[dict] = Field(default_factory=list)
    slippage_log: list[dict] = Field(default_factory=list)
    done: bool = False
    solved_at: float | None = None     # sim-seconds when the last step was solved
    ended_reason: str | None = None    # complete | time_cap | turn_cap
    # external participants (humans or bots), id -> {"name", "avatar", "first_spoke_t"}. They post
    # messages of kind "external" whose plan comes from an interpretation of their text; they have
    # NO hidden state. The code never branches on a participant's name or avatar.
    participants: dict[str, dict] = Field(default_factory=dict)
    # realism: session context fixed once (time of day, setting, per-student laughter) and the
    # closing-synthesis phase (None -> "propose" -> "confirm" -> "done")
    context: dict = Field(default_factory=dict)
    closing: str | None = None
    closing_msgs: list[int] = Field(default_factory=list)
    closing_forced: bool = False       # closing started by time / turn pressure (task incomplete)
    closing_responders: list[str] = Field(default_factory=list)  # who still responds to the proposal (0.5.1)
    # 0.5.1 conflict: misconceptions slated to surface, [{"student", "m", "step"}] (sim/conflict.py)
    slates: list[dict] = Field(default_factory=list)

    # -- helpers -------------------------------------------------------------
    def student(self, sid: str) -> StudentState:
        return next(s for s in self.students if s.id == sid)

    def is_student(self, sid: str | None) -> bool:
        return sid is not None and any(s.id == sid for s in self.students)

    def is_external(self, pid: str | None) -> bool:
        return pid is not None and pid in self.participants

    def spoken_externals(self) -> list[str]:
        """External participants who have written in the chat (in join order)."""
        return [p for p, d in self.participants.items() if d.get("first_spoke_t") is not None]

    def display_names(self) -> dict[str, str]:
        return {**{s.id: s.name for s in self.students}, **{p: d["name"] for p, d in self.participants.items()}}

    @property
    def roster(self) -> list[str]:
        return [s.id for s in self.students]

    def message(self, mid: int) -> Message:
        return self.messages[mid]

    def recent_turns(self, n: int, before: int | None = None) -> list[Message]:
        """The last n chat turns (students and external participants)."""
        msgs = self.messages if before is None else self.messages[:before]
        return msgs[-n:] if n > 0 else []

    def student_msgs(self) -> list[Message]:
        """Simulated students' messages only (counts, caps and shares)."""
        return [m for m in self.messages if m.kind == "student"]

    def recent_student_msgs(self, n: int) -> list[Message]:
        return self.student_msgs()[-n:] if n > 0 else []

    @property
    def task_complete(self) -> bool:
        return self.step_idx >= self.n_steps

    @property
    def last_message_t(self) -> float:
        return self.messages[-1].t if self.messages else 0.0
