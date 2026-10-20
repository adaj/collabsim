"""Public API: `Simulator` (one per app) and `Session` (one per simulated group chat).

    from collabsim import Simulator
    sim = Simulator(api_key="sk-ant-...")             # or backend="template" / "mock" offline
    s = sim.new_session(n_students=4, task="sci-enzyme-01", language="pt-BR",
                        participants=["Adelson", {"name": "Bot", "avatar": "bot.png"}])
    s.start()                                         # realtime: a background thread paces the chat
    out = s.poll(after=0)                             # {"events": [SimEvent...], "next_after", "status", "typing"}
    s.say("acho que a temperatura importa", sender="Adelson")
    s.say("Ana, pode explicar melhor?", sender="Bot")  # addressing a student = writing their name
    s.close()

Every external participant (a person typing in a UI, or a bot relayed by it) is treated the same
way: their messages are interpreted into content plans and affect the students only through what
they say. The code never branches on a participant's name or avatar.

Thread safety: every public method may be called from any thread (e.g. concurrent Gradio
callbacks). `poll`, `say`, `add_participant` and the pacing controls never wait on an LLM call in
realtime mode; generation and message interpretation run in the session's own background thread. Errors in that thread (API billing/auth/connection, internal bugs) never propagate: the
session ends with status "error", a readable message (`poll()["error"]`) and an "error" system
event, and every event released so far is kept.
"""
from __future__ import annotations

import secrets
import threading
import time
import uuid
from typing import Callable, Literal

from . import lang as _lang
from .errors import BackendUnavailable, CapacityError, describe_error, diagnose, friendly_message
from .llm.client import AnthropicBackend, GracefulStop, LLMClient, MockLLM
from .llm.config import DEFAULT_MODEL, check_model
from .models.task import fold, list_tasks, load_task, task_languages
from .runtime.interpret import Interpreter
from .runtime.live import LiveSession, roster_of
from .runtime.presets import PRESETS, TraitError, apply_traits, preset_list
from .runtime.taskspec import DEFAULT_TASK, draft_task, library, resolve_task, validate_spec
from .sim.config import SimConfig
from .sim.group import MAX_STUDENTS, MIN_STUDENTS, NAMES
from .types import MAX_PARTICIPANTS, MAX_SPEED, ParticipantSpec, SessionConfig, SimEvent, TaskSpec, fold_name

Backend = Literal["llm", "mock", "template"]
MAX_TEXT = 2000


class Simulator:
    """Factory and registry of live sessions; holds the (shared, thread-safe) LLM backend.

    backend: "llm" (Anthropic API, Haiku only; needs api_key or ANTHROPIC_API_KEY),
             "mock" (the full LLM pipeline against a deterministic offline stand-in; tests/demos),
             "template" (no LLM at all: template wording, heuristic interpretation of participants' messages).
    cache_dir: optional persistent response cache (identical requests are replayed, not re-paid).
    student_model: only "claude-haiku-5-5" is allowed (allowlist enforced for every role).
    max_sessions: running/paused/ready sessions at once (CapacityError beyond).
    max_student_turns: hard cap per session.
    idle_ttl_s: a realtime session nobody polled or posted to for this long is stopped
                (protects against abandoned browser tabs burning API credit).
    """

    def __init__(self, api_key: str | None = None, backend: Backend = "llm", cache_dir: str | None = None,
                 student_model: str = DEFAULT_MODEL, max_sessions: int = 8, max_student_turns: int = 200,
                 *, idle_ttl_s: float = 1800.0, lurk_cap_minutes: float = 10.0,
                 unspoken_boost_per_min: float = 1.0, tick_s: float = 0.1, request_timeout_s: float = 30.0,
                 clock: Callable[[], float] = time.monotonic, wall_ms: Callable[[], int] | None = None,
                 backend_factory: Callable[[], object] | None = None):
        if backend not in ("llm", "mock", "template"):
            raise ValueError(f"backend must be 'llm', 'mock' or 'template', not {backend!r}")
        self.student_model = check_model(student_model)
        self.backend = backend
        self.cache_dir = cache_dir
        self.max_sessions = int(max_sessions)
        self.max_student_turns = int(max_student_turns)
        self.idle_ttl_s = float(idle_ttl_s)
        self.lurk_cap_minutes = lurk_cap_minutes
        self.unspoken_boost_per_min = unspoken_boost_per_min
        self.tick_s = float(tick_s)
        self.clock = clock
        self.wall_ms = wall_ms or (lambda: int(time.time() * 1000))
        self._backend_factory = backend_factory
        self._lock = threading.Lock()
        self._sessions: dict[str, Session] = {}
        self._shared = None
        if backend == "llm" and backend_factory is None:
            try:
                self._shared = AnthropicBackend(api_key=api_key, timeout=request_timeout_s)
            except RuntimeError as e:
                raise BackendUnavailable(str(e)) from None
            except ImportError as e:  # pragma: no cover
                raise BackendUnavailable(f"the 'anthropic' package is required for backend='llm': {e}") from None
            except Exception as e:  # noqa: BLE001  (e.g. an unreadable SSL_CERT_FILE)
                raise BackendUnavailable(f"could not create the Anthropic client: {describe_error(e)}") from e

    # ------------------------------------------------------------------ backend
    def _backend(self):
        if self._backend_factory is not None:
            return self._backend_factory()
        if self.backend == "mock":
            return MockLLM()
        if self.backend == "llm":
            return self._shared
        return None

    def _new_client(self) -> LLMClient | None:
        """One client per session (own usage meter) over the shared backend."""
        b = self._backend()
        return None if b is None else LLMClient(b, self.cache_dir if self.backend == "llm" else None)

    def check_connection(self) -> dict:
        """One cheap request (a 1-token Haiku message, ~US$0.00001) to verify network, key and billing.
        Returns {"ok", "backend", "model", "latency_s", "error", "error_kind", "hint"}; never raises."""
        out = {"ok": False, "backend": self.backend, "model": None, "latency_s": None,
               "error": None, "error_kind": None, "hint": None}
        if self.backend != "llm" or self._shared is None:
            out.update(ok=True, model=None if self.backend == "template" else "mock")
            return out
        from .llm.config import request_extras
        t0 = time.monotonic()
        try:
            c = self._shared._client.with_options(max_retries=1)
            r = c.messages.create(model=self.student_model, max_tokens=1,
                                  messages=[{"role": "user", "content": "ok"}], **request_extras())
            out.update(ok=True, model=getattr(r, "model", self.student_model))
        except Exception as e:  # noqa: BLE001
            from .llm.client import _is_connection_error, _status, _stop_kind, classify_error
            kind = classify_error(e)
            ek = {"stop": _stop_kind(e), "retry": "rate_limit", "connection": "connection"}.get(kind, "refused")
            stop = GracefulStop(str(e), kind=ek, detail=describe_error(e),
                                hint=diagnose(e) if _is_connection_error(e) else None)
            _, msg = friendly_message(stop, "pt-BR")
            out.update(error=msg, error_kind=ek, hint=stop.hint, status_code=_status(e),
                       exception=describe_error(e))
        out["latency_s"] = round(time.monotonic() - t0, 3)
        return out

    # ------------------------------------------------------------------ tasks
    def options(self) -> dict:
        """What a UI needs to build its controls."""
        return {"tasks": [{"id": t["id"], "title": t["title"], "titles": t["titles"], "languages": t["languages"],
                           "domain": t["domain"], "n_roles": t["n_roles"]} for t in library()],
                "default_task": DEFAULT_TASK, "languages": list(_lang.SUPPORTED), "default_language": "pt-BR",
                "n_students": {"min": MIN_STUDENTS, "max": MAX_STUDENTS, "default": 4,
                               "calibrated": {"min": 2, "max": 4}},
                "pacing": ["realtime", "manual"], "speed": {"min": 0.1, "max": MAX_SPEED, "default": 1.0},
                "pace": ["pilot", "brisk"], "default_pace": "pilot",
                "conflict": ["low", "normal", "high"], "default_conflict": "normal",
                "trait_presets": [{"id": "mixed", "description": "pilot-calibrated priors (a natural mix)",
                                   "multipliers": {}}] + [p for p in preset_list() if p["id"] != "default"],
                "participants": {"max": MAX_PARTICIPANTS}, "max_student_turns": self.max_student_turns,
                "max_sessions": self.max_sessions, "backend": self.backend, "avatars": 6,
                "model": self.student_model if self.backend != "template" else None}

    def list_tasks(self, language: str | None = None) -> list[dict]:
        """Library tasks [{id, title, languages, domain, n_roles}], optionally only those in `language`."""
        lg = _lang.normalize(language) if language else None
        out = []
        for t in library():
            if lg and lg not in t["languages"]:
                continue
            out.append({"id": t["id"], "title": t["titles"].get(lg) if lg else t["title"],
                        "languages": t["languages"], "domain": t["domain"], "n_roles": t["n_roles"]})
        return out

    def get_task(self, task_id: str, language: str = "pt-BR") -> TaskSpec:
        """A library task in one language (raises KeyError if unknown / not in that language)."""
        lg = _lang.normalize(language)
        if task_id not in list_tasks():
            raise KeyError(f"unknown task {task_id!r} (library: {list_tasks()})")
        if lg not in task_languages(task_id):
            raise KeyError(f"task {task_id!r} has no {lg!r} version (has {task_languages(task_id)})")
        return load_task(task_id, language=lg)

    def validate_task(self, spec: dict | TaskSpec) -> dict:
        """{"valid", "errors", "warnings", ...} for a TaskSpec (dict or model); never raises."""
        if isinstance(spec, TaskSpec):
            spec = spec.model_dump(mode="json")
        return validate_spec(spec)

    def draft_task(self, description: str, language: str = "pt-BR", n_roles: int | None = None) -> dict:
        """Haiku drafts a TaskSpec from a free-text activity description; it is validated and returned
        FOR REVIEW (pass `draft` as task= to new_session once you are happy with it). One call,
        ~US$0.003. backend 'mock'/'template': a deterministic stand-in draft. Never raises: on an API
        failure returns {"draft": None, "error": <readable message>, ...}."""
        lg = _lang.normalize(language)
        n = int(n_roles or 3)
        if not (2 <= n <= 6):
            raise ValueError("n_roles must be between 2 and 6")
        client = self._new_client() or LLMClient(MockLLM())
        try:
            out = draft_task(client, description, lg, n)
        except Exception as e:  # noqa: BLE001
            kind, msg = friendly_message(e, lg)
            return {"draft": None, "validation": {"valid": False, "errors": [msg], "warnings": []},
                    "error": msg, "error_kind": kind, "usage": self._client_usage(client)}
        out["usage"] = self._client_usage(client)
        out["error"] = None
        return out

    @staticmethod
    def _client_usage(client: LLMClient) -> dict:
        t = client.meter.totals()
        return {"calls": int(t["calls"]), "input_tokens": int(t["input_tokens"]),
                "output_tokens": int(t["output_tokens"]), "usd": round(t["usd"], 6),
                "est_usd": round(t["est_usd"], 6)}

    # ------------------------------------------------------------------ sessions
    def _active_count(self) -> int:
        return sum(1 for s in self._sessions.values() if s._live.active)

    def reap(self) -> list[str]:
        """Stop and forget sessions nobody polled or posted to for idle_ttl_s; returns their ids.
        (A forgotten Session object stays usable: record(), usage(), events().)"""
        now = self.clock()
        with self._lock:
            old = [k for k, s in self._sessions.items() if now - s._live.last_access > self.idle_ttl_s]
        for k in old:
            s = self.get_session(k)
            if s is not None:
                if s._live.active:
                    s._stop_idle()
                s.close()
        return old

    def sessions(self) -> list["Session"]:
        with self._lock:
            return list(self._sessions.values())

    def get_session(self, session_id: str) -> "Session | None":
        with self._lock:
            return self._sessions.get(session_id)

    def new_session(self, config: SessionConfig | None = None, **kwargs) -> "Session":
        """Build a session (not started). Keyword arguments are SessionConfig fields. Raises
        ValueError (bad config / task), CapacityError (too many sessions)."""
        if config is None:
            cfg = SessionConfig(**kwargs)
        elif kwargs:
            cfg = SessionConfig(**{**{k: getattr(config, k) for k in SessionConfig.model_fields}, **kwargs})
        else:
            cfg = config
        self.reap()
        with self._lock:
            if self._active_count() >= self.max_sessions:
                raise CapacityError(f"too many concurrent sessions (max {self.max_sessions}); close one first")
        task = self._resolve(cfg)
        seed = cfg.seed if cfg.seed is not None else secrets.randbelow(2**31)
        cap = min(cfg.max_student_turns or self.max_student_turns, self.max_student_turns)
        extra = {k: v for k, v in (("session_minutes", cfg.max_minutes), ("target_minutes", cfg.target_minutes))
                 if v is not None}
        scfg = SimConfig(pace=cfg.pace, conflict=cfg.conflict, max_student_turns=cap,
                         end_on_complete=cfg.end_on_complete, lurk_cap_minutes=self.lurk_cap_minutes,
                         unspoken_boost_per_min=self.unspoken_boost_per_min, offtask_rate=cfg.offtask_rate,
                         time_of_day=cfg.time_of_day, **extra)
        client = self._new_client()
        from .sim.factory import build_session
        sid = uuid.uuid4().hex[:12]
        engine = build_session(task, seed, realizer=self.backend, client=client, n_students=cfg.n_students, cfg=scfg)
        engine.state.session_id = sid
        try:
            traits = apply_traits(engine.state.students, cfg.traits_spec())
        except TraitError as e:
            raise ValueError(str(e)) from None
        for st in engine.state.students:   # the lurk cap holds after presets / overrides too
            st.traits.lurk_until_min = min(st.traits.lurk_until_min, self.lurk_cap_minutes)
        specs = cfg.participant_specs()
        _dedupe_names(engine, [p.name for p in specs], task)
        engine.initial_students = [st.model_dump() for st in engine.state.students]
        live = LiveSession(sid, engine, mode=cfg.pacing, speed=cfg.speed,
                           interpreter=Interpreter(client), clock=self.clock, wall_ms=self.wall_ms, client=client,
                           config={**cfg.model_dump(mode="json", exclude={"task"}), "seed": seed, "task_id": task.id,
                                   "language": _lang.code_of(task), "traits": traits,
                                   "custom_task": not isinstance(cfg.task, str) and cfg.task is not None,
                                   "backend": self.backend, "model": self.student_model if client else None,
                                   "max_student_turns": cap})
        for i, p in enumerate(specs, 1):
            live.add_participant(f"e{i}", p.name, p.avatar)
        live._sync_participants()
        sess = Session(self, live)
        with self._lock:
            if self._active_count() >= self.max_sessions:
                raise CapacityError(f"too many concurrent sessions (max {self.max_sessions}); close one first")
            self._sessions[sid] = sess
        return sess

    def _resolve(self, cfg: SessionConfig):
        t = cfg.task
        if isinstance(t, TaskSpec):
            return resolve_task(None, t.model_dump(mode="json"), cfg.language)
        if isinstance(t, dict):
            return resolve_task(None, t, cfg.language)
        return resolve_task(t or DEFAULT_TASK, None, cfg.language)

    def _forget(self, sid: str) -> None:
        with self._lock:
            self._sessions.pop(sid, None)

    def close(self) -> None:
        """Stop and close every session (e.g. at app shutdown)."""
        for s in self.sessions():
            s.close()


def _dedupe_names(engine, names: list[str], task) -> None:
    """A student whose name clashes with a participant's (accent/case-folded) gets another name."""
    pool = _lang.attr(task, "NAMES", NAMES)
    taken = {fold_name(n) for n in names}
    used = {fold(s.name) for s in engine.state.students} | taken
    for s in engine.state.students:
        if fold_name(s.name) in taken:
            s.name = next(n for n in pool if fold(n) not in used)
            used.add(fold(s.name))
            engine.names[s.id] = s.name


class Session:
    """One simulated group chat. Create with Simulator.new_session; see the module docstring."""

    def __init__(self, simulator: Simulator, live: LiveSession):
        self._simulator = simulator
        self._live = live
        self._thread: threading.Thread | None = None
        self._halt = threading.Event()
        self._ctl = threading.Lock()

    # ------------------------------------------------------------------ info
    @property
    def id(self) -> str:
        return self._live.id

    @property
    def roster(self) -> list[dict]:
        """[{id, name, avatar, ring, role, kind}]: students (avatar = i % 6, ring = i // 6, a color
        index so repeated avatar images differ; kind "student"), then the external participants in
        join order (ids e1, e2, ...; avatar as declared; kind "external"; "spoken": has written)."""
        return self._live.roster()

    @property
    def participants(self) -> list[dict]:
        """The external participants' roster entries."""
        return [r for r in self.roster if r["kind"] == "external"]

    @property
    def task(self) -> TaskSpec:
        return self._live.sim.task

    @property
    def seed(self) -> int:
        return self._live.config["seed"]

    @property
    def pacing(self) -> str:
        return self._live.mode

    @property
    def status(self) -> str:
        """"ready" (realtime, not started) | "running" | "paused" | "done" | "error"."""
        return self._live.public_status

    @property
    def error(self) -> str | None:
        return self._live.error

    @property
    def ended_reason(self) -> str | None:
        return self._live.ended_reason

    # ------------------------------------------------------------------ realtime
    def start(self) -> "Session":
        """Realtime: start the sim clock and the background thread that pre-generates the next turn and
        releases it when wall >= start + t_sim / speed. Manual: no-op. Idempotent."""
        with self._ctl:
            if self._live.mode != "realtime":
                return self
            self._live.start()
            if self._thread is None and self._live.active:
                self._thread = threading.Thread(target=self._run, name=f"collabsim-{self.id}", daemon=True)
                self._thread.start()
        return self

    def _run(self) -> None:
        live = self._live
        while not self._halt.is_set() and live.active:
            try:
                if self._simulator.clock() - live.last_access > self._simulator.idle_ttl_s:
                    self._stop_idle()
                    break
                did = live.pump()
            except Exception:  # noqa: BLE001  (pump never raises; belt and braces)
                did = False
            if not did:
                self._halt.wait(self._simulator.tick_s)

    def _stop_idle(self) -> None:
        live = self._live
        with live._ev:
            if live.active:
                live._stop = True
                live.pending = None
                live._queue = []
                live._finish("idle")

    def poll(self, after: int = 0, limit: int = 500) -> dict:
        """Non-blocking. {"events": [SimEvent with seq > after], "next_after": int, "status": str,
        "typing": [speaker ids about to post], "error": str (only when status == "error")}.
        Polling also keeps the session alive (idle_ttl_s)."""
        out = self._live.events_after(after, limit)
        out["events"] = [SimEvent.from_raw(e) for e in out["events"]]
        return out

    def events(self) -> list[SimEvent]:
        """Every event so far (does not count as activity for idle_ttl_s)."""
        with self._live._ev:
            return [SimEvent.from_raw(e) for e in self._live.events]

    # ------------------------------------------------------------------ participants
    def _event(self, seq: int | None) -> SimEvent | None:
        if seq is None:
            return None
        with self._live._ev:
            for e in reversed(self._live.events):
                if e["seq"] == seq:
                    return SimEvent.from_raw(e)
        return None

    @staticmethod
    def _clean(text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ValueError("empty message")
        return text[:MAX_TEXT]

    def _resolve_sender(self, sender: str | None, create: bool = True) -> str:
        """A participant id from a name or an id (accent/case-folded). None: the only declared
        participant. An unknown name joins the chat (auto-join)."""
        parts = self.participants
        if sender is None:
            if len(parts) != 1:
                raise ValueError(f"say() needs sender= when the session has {len(parts)} participants")
            return parts[0]["id"]
        if not isinstance(sender, str) or not sender.strip():
            raise ValueError("sender must be a participant name or id")
        key = fold_name(sender)
        for p in parts:
            if key in (fold_name(p["id"]), fold_name(p["name"])):
                return p["id"]
        for r in self.roster:
            if r["kind"] == "student" and key in (fold_name(r["id"]), fold_name(r["name"])):
                raise ValueError(f"{sender!r} is a simulated student; participants cannot speak as students")
        if not create:
            raise ValueError(f"unknown participant {sender!r}")
        return self.add_participant(sender)["id"]

    def add_participant(self, name: str, avatar: str | None = None) -> dict:
        """A participant joins (also mid-session); returns their roster entry. Raises ValueError for an
        empty / duplicate name, a student's name, or more than 8 participants."""
        spec = ParticipantSpec(name=name, avatar=avatar)
        with self._ctl:
            parts = self.participants
            key = fold_name(spec.name)
            if any(key == fold_name(p["name"]) for p in parts):
                raise ValueError(f"participant {spec.name!r} already in the chat")
            if any(r["kind"] == "student" and key == fold_name(r["name"]) for r in self.roster):
                raise ValueError(f"{spec.name!r} is the name of a simulated student")
            if len(parts) >= MAX_PARTICIPANTS:
                raise ValueError(f"at most {MAX_PARTICIPANTS} participants")
            pid = f"e{len(parts) + 1}"
            self._live.add_participant(pid, spec.name, spec.avatar)
        if self._live.mode != "realtime":
            self._live.pump(generate=False)
        return next(p for p in self.participants if p["id"] == pid)

    def say(self, text: str, sender: str | None = None, *, ts: int | None = None) -> SimEvent | None:
        """External participant `sender` (a name or an id; None if exactly one participant was
        declared; an unknown name joins the chat) posts `text` (ts: optional wall epoch ms). The event
        is created at once (returned); the message is then interpreted (one Haiku call -> a validated
        content plan: facts, claims, misconception voiced/challenged, addressee; fallback: an opaque
        social turn) and integrated into the students' hidden state, in the background thread
        (realtime) or before returning (manual). To address a student, write their name. Returns None
        if the session already ended. Raises ValueError for a missing/ambiguous sender or a
        student's name."""
        pid = self._resolve_sender(sender)
        r = self._live.submit(pid, self._clean(text), ts_ms=ts)
        if self._live.mode != "realtime":
            self._live.pump(generate=False)
        return self._event(r.get("seq"))

    # ------------------------------------------------------------------ controls
    def pause(self) -> dict:
        return self._live.control("pause")

    def resume(self) -> dict:
        return self._live.control("resume")

    def set_speed(self, speed: float) -> dict:
        if not (0 < float(speed) <= MAX_SPEED):
            raise ValueError(f"speed must be in (0, {MAX_SPEED:g}]")
        return self._live.control("speed", float(speed))

    def stop(self) -> dict:
        """End the chat (status "done", reason "stopped"); events, record and usage stay available."""
        out = self._live.control("stop")
        self._halt.set()
        return out

    def close(self, timeout: float = 5.0) -> None:
        """Stop, join the background thread and remove the session from the Simulator."""
        if self._live.active:
            self._live.control("stop")
        self._halt.set()
        t = self._thread
        if t is not None and t is not threading.current_thread():
            t.join(timeout)
        self._simulator._forget(self.id)

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ------------------------------------------------------------------ manual pacing
    def _require_manual(self, what: str) -> None:
        if self._live.mode == "realtime" and self._thread is not None and self._thread.is_alive():
            raise RuntimeError(f"{what}() is for pacing='manual' (this realtime session is running; stop() it first)")

    def step(self) -> list[SimEvent]:
        """Manual pacing: generate and release the next student turn now; returns the new events
        (the turn, plus an "ended"/"error" system event when the session ends). [] once ended."""
        self._require_manual("step")
        live = self._live
        if live.mode == "realtime":
            live.mode = "manual"
        n0 = len(live.events)
        guard = 0
        while live.active and guard < 200:
            live.pump()
            guard += 1
            with live._ev:
                if any(e["kind"] == "student" for e in live.events[n0:]):
                    break
        with live._ev:
            return [SimEvent.from_raw(e) for e in live.events[n0:]]

    def run_to_end(self, max_iter: int = 100_000) -> list[SimEvent]:
        """Run the whole chat synchronously (no pacing); returns the events released by this call."""
        self._require_manual("run_to_end")
        live = self._live
        if live.mode == "realtime":
            live.mode = "manual"
        n0 = len(live.events)
        live.run_sync(max_iter)
        with live._ev:
            return [SimEvent.from_raw(e) for e in live.events[n0:]]

    # ------------------------------------------------------------------ reports
    def record(self) -> dict:
        """Full research record: transcript with code content plans, hidden student state (initial,
        timeline, final), misconception flips, guard logs, interpretations of participants' messages,
        every event, config and usage. Waits for an in-flight generation to finish (consistent snapshot)."""
        live = self._live
        with live._work:
            rec = live.sim.result()
            rec["interpretations"] = list(live.interpretations)
            with live._ev:
                rec["events"] = list(live.events)
            rec["summary"] = live.summary()
        return rec

    def usage(self) -> dict:
        """{"calls", "local_cache_hits", "input_tokens", "cache_read_tokens", "cache_write_tokens",
        "output_tokens", "usd", "est_usd", "backend"} for this session (template backend: zeros; mock:
        calls and tokens counted, `usd` 0, `est_usd` = those tokens at Haiku rates)."""
        u = self._live.usage()
        if "input_tokens" not in u:
            u = {"calls": 0, "local_cache_hits": 0, "input_tokens": 0, "cache_read_tokens": 0,
                 "cache_write_tokens": 0, "output_tokens": 0, "usd": 0.0, "est_usd": 0.0, "backend": "template"}
        return u

    def summary(self) -> dict:
        return self._live.summary()

    def __repr__(self) -> str:
        return f"<collabsim.Session {self.id} {self.status} task={self.task.id} n={len(self._live.sim.state.students)}>"
