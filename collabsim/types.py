"""Public data types: SimEvent (one chat event), SessionConfig (how a session is built),
ParticipantSpec (an external participant) and TaskSpec (a task definition; the same model the
engine runs)."""
from __future__ import annotations

from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

from . import lang as _lang
from .models.task import Task as TaskSpec
from .sim.group import MAX_STUDENTS, MIN_STUDENTS

__all__ = ["SimEvent", "SessionConfig", "ParticipantSpec", "TaskSpec", "MAX_SPEED", "MAX_PARTICIPANTS",
           "EventKind", "Pacing"]

MAX_SPEED = 20.0
MAX_PARTICIPANTS = 8
EventKind = Literal["student", "external", "system"]
Pacing = Literal["realtime", "manual"]


class SimEvent(BaseModel):
    """One message (or system notice) in a session's chat, in release order.

    seq          1, 2, 3, ... per session (poll with after=<last seq seen>)
    id           "<session id>:<seq>" (globally unique, stable)
    t_sim        sim seconds since the session started
    ts_ms        wall-clock epoch milliseconds when the event was released / posted
    speaker      student id ("s1".."s10"), external participant id ("e1", "e2", ... in join order) or "system"
    display_name the name to show (student name, participant name, "system")
    kind         "student" | "external" | "system"
    text         the chat text (for system events: a readable notice, pt-BR or en)
    addressee    id of the chat member the message names, if any
    code         system events only: "ended" | "error"
    reason       system events only: ended -> complete | time_cap | turn_cap | stopped | idle;
                 error -> billing | auth | rate_limit | connection | refused | internal
    """
    model_config = ConfigDict(frozen=True)

    seq: int
    id: str
    t_sim: float
    ts_ms: int
    speaker: str
    display_name: str
    kind: EventKind
    text: str
    addressee: str | None = None
    code: str | None = None
    reason: str | None = None

    def to_dict(self) -> dict:
        return self.model_dump()

    @classmethod
    def from_raw(cls, ev: dict) -> "SimEvent":
        return cls(**{k: ev.get(k) for k in cls.model_fields if k in ev})


def fold_name(name: str) -> str:
    import unicodedata
    return "".join(c for c in unicodedata.normalize("NFKD", name.strip().lower()) if not unicodedata.combining(c))


class ParticipantSpec(BaseModel):
    """An external participant (a person or a bot) who writes in the chat with Session.say. The
    simulator treats every participant the same way: their messages count only through what they say."""
    name: str = Field(min_length=1, max_length=40)
    avatar: str | None = None

    @field_validator("name")
    @classmethod
    def _strip(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("participant name must not be empty")
        return v


class SessionConfig(BaseModel):
    """Everything `Simulator.new_session` accepts (keyword arguments map 1:1 onto these fields).

    task        library task id ("sci-enzyme-01", "math-ratio-01"), a TaskSpec, or a TaskSpec dict
                (single-language, or {"languages": {"pt-BR": {...}, "en": {...}}})
    traits      "mixed" (= pilot-calibrated priors) or a preset name ("chatty", "quiet", "focused",
                "distracted", "deferent", "stubborn"), or a dict
                {"preset": ..., "overrides": {trait: value}, "per_student": {"0": {trait: value}}}
    participants   external participants who may write in the chat (Session.say): names or
                   {"name", "avatar"} (ParticipantSpec), up to 8, names unique (accent/case-folded); a
                   student whose name clashes is renamed. More can join later (Session.add_participant,
                   or say() with a new sender name).
    pacing      "realtime": a background thread releases turns at sim time / speed after start();
                "manual": the caller advances with step() / run_to_end()
    offtask_rate   target share of off-task student turns (default .15, 0..0.5); off-task turns are
                   spaced, never two in a row by one student, and form short threads
    time_of_day    "manha" | "tarde" | "noite" (also morning / afternoon / evening); None: drawn once
                   per session from the seed. Greetings and chatter stay consistent with it.
    pace        "pilot" (default): sessions as long as the pilot chats (~30-38 min, 12-19 messages per
                student; each activity stays open for a minimum time); "brisk": a short demo (~15-20 min:
                shorter dwell, quicker turns, fewer restatements per idea)
    target_minutes  overrides the pace's target session length (minutes; None = the pace's)
    max_minutes     sim-minutes time cap (None = the pace's: pilot 45, brisk 25)
    conflict    how many held misconceptions surface and get discussed: "low" (only when they come up,
                maybe none), "normal" (default: at least 1, typically 2, voiced by a student who holds
                it and challenged by a peer), "high" (every held belief, stronger and more persistent;
                an activity stays open until the holder changes their mind: the group may stall)
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    n_students: int = Field(default=4, ge=MIN_STUDENTS, le=MAX_STUDENTS)
    task: Union[str, TaskSpec, dict, None] = "sci-enzyme-01"
    language: str = _lang.SERVICE_DEFAULT
    seed: int | None = None
    pacing: Pacing = "realtime"
    speed: float = Field(default=1.0, gt=0, le=MAX_SPEED, description="sim seconds per wall second")
    traits: Union[str, dict, None] = "mixed"
    participants: list[Union[str, ParticipantSpec]] = Field(default_factory=list)
    max_minutes: float | None = Field(default=None, ge=1.0, le=180.0,
                                      description="sim-minutes time cap (None: the pace's, pilot 45 / brisk 25)")
    pace: Literal["pilot", "brisk"] = "pilot"
    target_minutes: float | None = Field(default=None, ge=3.0, le=120.0)
    conflict: Literal["low", "normal", "high"] = "normal"
    end_on_complete: bool = True
    max_student_turns: int | None = Field(default=None, ge=1)
    offtask_rate: float = Field(default=0.15, ge=0.0, le=0.5,
                                description="target share of off-task student turns (0..0.5)")
    time_of_day: Literal["manha", "tarde", "noite"] | None = None

    @field_validator("time_of_day", mode="before")
    @classmethod
    def _tod_ok(cls, v):
        if v is None:
            return None
        import unicodedata
        f = "".join(c for c in unicodedata.normalize("NFD", str(v).strip().lower()) if unicodedata.category(c) != "Mn")
        return {"morning": "manha", "afternoon": "tarde", "evening": "noite", "night": "noite"}.get(f, f)

    @field_validator("participants", mode="before")
    @classmethod
    def _participants_ok(cls, v):
        if v is None:
            return []
        if not isinstance(v, (list, tuple)):
            raise ValueError("participants must be a list of names or {name, avatar}")
        out = [ParticipantSpec(name=x) if isinstance(x, str) else
               (x if isinstance(x, ParticipantSpec) else ParticipantSpec.model_validate(x)) for x in v]
        if len(out) > MAX_PARTICIPANTS:
            raise ValueError(f"at most {MAX_PARTICIPANTS} participants")
        seen = set()
        for p in out:
            k = fold_name(p.name)
            if k in seen:
                raise ValueError(f"duplicate participant name {p.name!r}")
            seen.add(k)
        return out

    def participant_specs(self) -> list[ParticipantSpec]:
        return [p if isinstance(p, ParticipantSpec) else ParticipantSpec(name=p) for p in self.participants]

    @field_validator("language")
    @classmethod
    def _lang_ok(cls, v: str) -> str:
        return _lang.normalize(v)

    def traits_spec(self) -> dict:
        t: Any = self.traits
        if t is None or t == "mixed":
            return {"preset": "default"}
        if isinstance(t, str):
            return {"preset": t}
        return dict(t)
