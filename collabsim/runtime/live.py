"""One live simulated group chat: the engine wrapped for real-time use (internal; the public
wrapper is collabsim.simulator.Session).

The engine (sim/engine.Simulation) is the state-based simulator: code decides who speaks, when and
what (content plans, hidden state, pilot-calibrated turn-taking); Haiku only words turns.
`LiveSession` wraps it for real-time use:

* PACING. Sim time maps to wall time as  t_sim = anchor_sim + (wall - anchor_wall) * speed  (frozen
  while paused). The next student turn is PRE-GENERATED (planned, worded by Haiku, state-updated) as
  soon as the previous one is released, and held in `pending` until wall time reaches its sim time,
  so release is on time even though wording takes a second or two. `typing` lists the speaker of the
  pending turn shortly before its release. Manual mode releases each turn as soon as it is generated
  (the caller drives it with step() / run_to_end()). Before start() the realtime clock is frozen at 0.
* EXTERNAL MESSAGES (any participant who is not a simulated student: a person typing in a UI, or a
  bot relayed by it) are stamped with the current sim time and become events at once (their `seq`
  is returned to the poster). Before the engine integrates one, any pre-generated, unreleased turn is
  DISCARDED by restoring the snapshot taken before it was generated: that turn was planned without
  seeing the new message. The engine then inserts the message at its sim time (after one Haiku
  interpretation call) and plans the next turn from there. A discarded turn's
  Haiku calls are paid but unused (counted in the session's usage).
* THREADS. `pump()` does the heavy work (generation, interpretation); it is serialized by the `_work`
  lock (the realtime runner thread, or the caller in manual mode). Readers and posters only take the
  short `_ev` lock, so polling never waits on an LLM call.
* ERRORS. An API refusal (billing / credit / spend limit / auth, persistent 429/529 after backoff),
  a connection failure, or any internal error ends the session with status "error" and a readable
  message (collabsim.errors.friendly_message, with the underlying exception chain); released events
  are kept and nothing is raised.
"""
from __future__ import annotations

import threading
import time
from typing import Callable

from ..errors import friendly_message
from ..models.message import Message

TYPING_LEAD_MAX_S = 8.0   # sim seconds before release that the speaker shows as typing


def _mask(text: str) -> str:
    try:
        from ..runio import mask
        return mask(text)
    except Exception:  # noqa: BLE001
        return text


class LiveSession:
    def __init__(self, session_id: str, sim, *, mode: str = "realtime", speed: float = 1.0, interpreter=None,
                 clock: Callable[[], float] = time.monotonic,
                 wall_ms: Callable[[], int] = lambda: int(time.time() * 1000), client=None, config: dict | None = None):
        self.id = session_id
        self.sim = sim
        self.mode = mode
        self.speed = float(speed)
        self.interpreter = interpreter
        self.clock = clock
        self.wall_ms = wall_ms
        self.client = client
        self.config = config or {}
        self.events: list[dict] = []
        self.status = "running"
        self.error: str | None = None
        self.error_kind: str | None = None
        self.ended_reason: str | None = None
        self.pending: dict | None = None
        self.discarded_turns = 0
        self._seq = 0
        self._queue: list[dict] = []
        self._gen_hint: tuple[str, float] | None = None
        self._stop = False
        self._paused = False
        self._anchor_wall = clock()
        self._anchor_sim = 0.0
        self.started = mode != "realtime"
        self._ev = threading.RLock()
        self._work = threading.Lock()
        self.last_access = clock()
        self.created_ms = wall_ms()
        self.interpretations: list[dict] = []
        self._participants: dict[str, tuple[str, str | None]] = {}

    # ------------------------------------------------------------------ clock
    def start(self) -> None:
        """Start the realtime clock (sim time 0 = now). Idempotent."""
        with self._ev:
            if not self.started:
                self._anchor_wall = self.clock()
                self._anchor_sim = 0.0
                self.started = True

    def sim_now(self) -> float:
        if self.mode != "realtime":
            return max([e["t_sim"] for e in self.events[-1:]] or [0.0])
        if self._paused or not self.started:
            return self._anchor_sim
        return self._anchor_sim + (self.clock() - self._anchor_wall) * self.speed

    def touch(self) -> None:
        self.last_access = self.clock()

    @property
    def active(self) -> bool:
        return self.status in ("running", "paused")

    # ------------------------------------------------------------------ events
    def _display(self, speaker: str) -> str:
        if speaker in self._participants:
            return self._participants[speaker][0]
        return self.sim.names.get(speaker, speaker)

    def _kind(self, speaker: str) -> str:
        return "student" if self.sim.state.is_student(speaker) else "external"

    def _push(self, ev: dict) -> dict:
        self._seq += 1
        ev = {"seq": self._seq, "id": f"{self.id}:{self._seq}", **ev}
        self.events.append(ev)
        return ev

    def _emit_msg(self, m: Message) -> None:
        addr = m.addressee
        self._push({"t_sim": round(m.t, 2), "ts_ms": self.wall_ms(), "speaker": m.speaker,
                    "display_name": self._display(m.speaker), "kind": self._kind(m.speaker), "text": m.text,
                    "addressee": addr})

    def _emit_system(self, code: str, text: str, **extra) -> None:
        self._push({"t_sim": round(self.sim_now() if self.mode == "realtime" else self.sim.state.t, 2),
                    "ts_ms": self.wall_ms(), "speaker": "system", "display_name": "system", "kind": "system",
                    "text": text, "addressee": None, "code": code, **extra})

    def events_after(self, after: int = 0, limit: int = 500) -> dict:
        self.touch()
        self.release_due()
        with self._ev:
            evs = [e for e in self.events if e["seq"] > after][:limit]
            nxt = evs[-1]["seq"] if evs else max(after, 0)
            out = {"events": evs, "next_after": nxt, "status": self.public_status, "typing": self.typing()}
            if self.error:
                out["error"] = self.error
            return out

    @property
    def public_status(self) -> str:
        if self.status in ("error", "done"):
            return self.status
        if self._paused:
            return "paused"
        return "running" if self.started else "ready"

    def typing(self) -> list[str]:
        with self._ev:
            if not self.active or self._paused or self.mode != "realtime" or not self.started:
                return []
            now = self.sim_now()
            p = self.pending
            if p is not None:
                words = len(p["msgs"][0].text.split())
                lead = min(TYPING_LEAD_MAX_S, 1.5 + 0.4 * words)
                return [p["speaker"]] if now >= p["t"] - lead else []
            if self._gen_hint is not None and now >= self._gen_hint[1] - 3.0:
                return [self._gen_hint[0]]
            return []

    # ------------------------------------------------------------------ control
    def control(self, action: str, speed: float | None = None) -> dict:
        self.touch()
        with self._ev:
            if action == "pause":
                if self.active and not self._paused:
                    self._anchor_sim = self.sim_now()
                    self._paused = True
                    self.status = "paused"
            elif action == "resume":
                if self.active and self._paused:
                    self._anchor_wall = self.clock()
                    self._paused = False
                    self.status = "running"
            elif action == "speed":
                if speed is None or not (0 < float(speed) <= 20):
                    raise ValueError("speed must be in (0, 20]")
                self._anchor_sim = self.sim_now()
                self._anchor_wall = self.clock()
                self.speed = float(speed)
            elif action == "stop":
                if self.active:
                    self._stop = True
                    self.pending = None
                    self._queue = []
                    self._finish("stopped")
            else:
                raise ValueError(f"unknown action {action!r} (pause | resume | stop | speed)")
            return {"status": self.public_status, "speed": self.speed, "t_sim": round(self.sim_now(), 2)}

    def _finish(self, reason: str | None) -> None:
        if not self.active:
            return
        self.status = "done"
        self.ended_reason = reason or "done"
        self._emit_system("ended", self._end_text(self.ended_reason), reason=self.ended_reason)

    def _end_text(self, reason: str) -> str:
        pt = getattr(self.sim.task, "language", "en") == "pt-BR"
        txt = {"complete": ("a tarefa foi concluída", "the task is complete"),
               "time_cap": ("o tempo da sessão acabou", "the session time is up"),
               "turn_cap": ("limite de mensagens da sessão atingido", "the session message limit was reached"),
               "stopped": ("sessão encerrada", "session stopped"),
               "idle": ("sessão encerrada por inatividade", "session closed after inactivity")}.get(reason, (reason, reason))
        return txt[0] if pt else txt[1]

    def _fail(self, exc: BaseException) -> None:
        kind, message = friendly_message(exc, getattr(self.sim.task, "language", "en"))
        with self._ev:
            if self.status == "error":
                return
            self.pending = None
            self.status = "error"
            self.error = _mask(message)
            self.error_kind = kind
            self.ended_reason = "error"
            self._emit_system("error", self.error, reason=kind)

    # ------------------------------------------------------------------ external messages
    def submit(self, sender: str, text: str, ts_ms: int | None = None) -> dict:
        """A message by external participant `sender` (an id from the roster). Becomes an event
        immediately; integrated by the next pump."""
        self.touch()
        if sender not in self._participants:
            raise ValueError(f"unknown participant {sender!r}")
        self.release_due()
        with self._ev:
            if not self.active:
                return {"seq": None, "accepted": False, "status": self.public_status}
            t = max([self.sim_now()] + [e["t_sim"] for e in self.events[-1:]])
            from .interpret import addressee_of
            names = {**{x.id: x.name for x in self.sim.state.students},
                     **{pid: v[0] for pid, v in self._participants.items()}}
            target = addressee_of(text, names, sender)
            ev = self._push({"t_sim": round(t, 2), "ts_ms": int(ts_ms) if ts_ms is not None else self.wall_ms(),
                             "speaker": sender, "display_name": self._display(sender), "kind": "external",
                             "text": text, "addressee": target})
            self._queue.append({"sender": sender, "text": text, "t": t, "seq": ev["seq"]})
            return {"seq": ev["seq"], "accepted": True, "t_sim": round(t, 2), "status": self.public_status}

    def _apply_external(self, ext: dict) -> None:
        st = self.sim.state
        plan = self.interpreter(self.sim.task, st, ext["text"], ext["sender"])
        msg = self.sim.inject_external(ext["sender"], ext["text"], plan, t=ext["t"])
        rec = dict(self.interpreter.log[-1]) if getattr(self.interpreter, "log", None) else {}
        rec.update(seq=ext["seq"], msg=msg.id)
        self.interpretations.append(rec)

    # ------------------------------------------------------------------ generation / release
    def _realizer_log(self):
        return getattr(self.sim.realizer, "log", None)

    def add_participant(self, pid: str, name: str, avatar: str | None = None) -> None:
        """Register a participant at once (never waits on a generation); the engine learns about them
        at the next pump (the generation thread is the only one that touches the engine)."""
        with self._ev:
            self._participants[pid] = (name, avatar)

    def _sync_participants(self) -> None:
        for pid, (name, avatar) in list(self._participants.items()):
            if pid not in self.sim.state.participants:
                self.sim.add_participant(pid, name, avatar)

    def _restore(self, p: dict) -> None:
        self.sim.restore(p["snap"])
        self._sync_participants()   # joined after the snapshot: kept
        log = self._realizer_log()
        if log is not None and p.get("log0") is not None:
            del log[p["log0"]:]
        self.discarded_turns += 1

    def _generate(self) -> None:
        sim = self.sim
        st = sim.state
        snap = sim.snapshot()
        log = self._realizer_log()
        log0 = len(log) if log is not None else None
        n0 = len(st.messages)
        with self._ev:
            self._gen_hint = (st.planned.speaker, st.planned.t) if st.planned is not None else None
        guard = 0
        while not sim.state.done and len(sim.state.messages) == n0 and guard < 50:
            sim.step()
            guard += 1
        new = list(sim.state.messages[n0:])
        with self._ev:
            self._gen_hint = None
            p = {"snap": snap, "msgs": new, "t": new[0].t if new else None,
                 "speaker": new[0].speaker if new else None, "log0": log0}
            if self._stop:
                return
            if self._queue and new:          # an external message arrived meanwhile: this turn is stale
                self._restore(p)
                return
            if new:
                self.pending = p

    def release_due(self) -> None:
        with self._ev:
            p = self.pending
            if p is None or not self.active:
                return
            if self.mode != "realtime" or (self.started and not self._paused and self.sim_now() >= p["t"]):
                for m in p["msgs"]:
                    self._emit_msg(m)
                self.pending = None

    def pump(self, generate: bool = True) -> bool:
        """One unit of work (integrate queued external messages, then pre-generate the next turn
        unless generate=False); returns True if it did something. Never raises."""
        if not self.active:
            return False
        with self._work:
            try:
                return self._pump(generate)
            except Exception as e:  # noqa: BLE001  (API refusals, connection errors, internal bugs)
                self._fail(e)
            return False

    def _pump(self, generate: bool = True) -> bool:
        did = False
        self._sync_participants()
        with self._ev:
            queue, self._queue = self._queue, []
            if queue and self.pending is not None:
                p, self.pending = self.pending, None
                self._restore(p)
        for ext in queue:
            if self._stop:
                return True
            self._apply_external(ext)
            did = True
        if self._stop or not self.active:
            return did
        if generate and self.pending is None and not self.sim.state.done:
            with self._ev:
                waiting = bool(self._queue)
            if not waiting:
                self._generate()
                did = True
        self.release_due()
        with self._ev:
            if self.pending is None and self.sim.state.done and not self._queue and self.active:
                self._finish(self.sim.state.ended_reason)
                did = True
        return did

    def run_sync(self, max_iter: int = 100_000) -> None:
        """Run to completion without pacing (manual mode / tests)."""
        n = 0
        while self.active and n < max_iter:
            self.pump()
            n += 1

    # ------------------------------------------------------------------ reports
    def roster(self) -> list[dict]:
        with self._ev:
            out = roster_of(self.sim)
            have = {r["id"] for r in out}
            for pid, (name, avatar) in self._participants.items():   # joined, not yet synced
                if pid not in have:
                    out.append({"id": pid, "name": name, "avatar": avatar, "ring": None, "role": None,
                                "kind": "external", "spoken": False})
            return out

    def usage(self) -> dict:
        if self.client is None:
            return {"calls": 0, "usd": 0.0, "est_usd": 0.0}
        t = self.client.meter.totals()
        return {"calls": int(t["calls"]), "local_cache_hits": int(t["local_cache_hits"]),
                "input_tokens": int(t["input_tokens"]), "cache_read_tokens": int(t["cache_read_tokens"]),
                "cache_write_tokens": int(t["cache_write_tokens"]), "output_tokens": int(t["output_tokens"]),
                "usd": round(t["usd"], 6), "est_usd": round(t["est_usd"], 6),
                "backend": getattr(self.client.backend, "name", "?")}

    def summary(self) -> dict:
        st = self.sim.state
        self.touch()
        return {"session_id": self.id, "status": self.public_status, "ended_reason": self.ended_reason,
                "error": self.error, "error_kind": self.error_kind, "task_id": self.sim.task.id, "language": self.sim.task.language,
                "roster": self.roster(), "participants": [r for r in self.roster() if r["kind"] == "external"],
                "pacing": {"mode": self.mode, "speed": self.speed},
                "t_sim": round(self.sim_now(), 2), "n_events": len(self.events),
                "student_turns": st.student_turns, "steps_solved": [s["step"] for s in st.steps_solved],
                "n_steps": st.n_steps, "discarded_pregenerated_turns": self.discarded_turns,
                "usage": self.usage(), "config": self.config}


def roster_of(sim) -> list[dict]:
    """Students first (avatar = i % 6, the UI has 6 images; ring = i // 6, a color index so repeated
    images differ), then external participants in join order (avatar as declared)."""
    out = []
    for i, s in enumerate(sim.state.students):
        out.append({"id": s.id, "name": s.name, "avatar": i % 6, "ring": i // 6, "role": s.role, "kind": "student"})
    for pid, d in sim.state.participants.items():
        out.append({"id": pid, "name": d["name"], "avatar": d.get("avatar"), "ring": None, "role": None,
                    "kind": "external", "spoken": d.get("first_spoke_t") is not None})
    return out
