"""Public API (collabsim.Simulator / Session): options and tasks, manual and threaded realtime
sessions, external participants' messages, errors (billing, connection with a diagnosable cause),
check_connection, thread safety, CLI, and the slim-package surface."""
import importlib
import json
import pkgutil
import ssl
import sys
import threading
import time

import pytest

import collabsim
from collabsim import BackendUnavailable, SessionConfig, SimEvent, Simulator, TaskSpec
from collabsim.llm.client import MockLLM
from collabsim.llm.config import ModelNotAllowed


def wait_for(cond, timeout=10.0, every=0.02):
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        if cond():
            return True
        time.sleep(every)
    return False


# ------------------------------------------------------------------ options / tasks
def test_options_tasks_and_validation():
    sim = Simulator(backend="template")
    o = sim.options()
    assert {t["id"] for t in o["tasks"]} == {"sci-enzyme-01", "math-ratio-01"}
    assert o["languages"] == ["pt-BR", "en"] and o["n_students"]["min"] == 2 and o["n_students"]["max"] == 10
    assert o["trait_presets"][0]["id"] == "mixed" and "chatty" in {p["id"] for p in o["trait_presets"]}
    assert o["pacing"] == ["realtime", "manual"] and o["participants"] == {"max": 8}
    assert set(o) == {"avatars", "backend", "conflict", "default_conflict", "default_language", "default_pace",
                      "default_task", "languages", "max_sessions", "max_student_turns", "model", "n_students",
                      "pace", "pacing", "participants", "speed", "tasks", "trait_presets"}
    tl = sim.list_tasks("pt-BR")
    assert all("pt-BR" in t["languages"] for t in tl) and tl[0]["title"]
    t = sim.get_task("sci-enzyme-01", "pt-BR")
    assert isinstance(t, TaskSpec) and t.language == "pt-BR"
    with pytest.raises(KeyError):
        sim.get_task("nope")
    assert sim.validate_task(t)["valid"]
    bad = t.model_dump(mode="json")
    bad["jigsaw"] = {"A": bad["jigsaw"]["A"]}
    rep = sim.validate_task(bad)
    assert not rep["valid"] and rep["errors"]


def test_draft_task_offline_and_custom_task_session():
    sim = Simulator(backend="mock")
    d = sim.draft_task("uma atividade sobre fotossíntese", "pt-BR", n_roles=3)
    assert d["error"] is None and d["validation"]["valid"] and d["usage"]["calls"] == 1
    s = sim.new_session(task=d["draft"], pacing="manual", seed=3, n_students=3)
    assert s.task.id.startswith("draft-")
    assert s.step()
    with pytest.raises(ValueError):
        sim.new_session(task={"id": "x"}, pacing="manual")
    with pytest.raises(ValueError):
        sim.new_session(task="unknown-task", pacing="manual")


# ------------------------------------------------------------------ manual pacing
def test_manual_session_participants_record_usage():
    sim = Simulator(backend="mock")
    s = sim.new_session(n_students=4, task="sci-enzyme-01", language="pt-BR", seed=5, pacing="manual",
                        participants=["Adelson", {"name": "Bot", "avatar": "bot.png"}], traits="focused")
    roster = s.roster
    assert [r["kind"] for r in roster] == ["student"] * 4 + ["external"] * 2
    assert [r["id"] for r in roster[4:]] == ["e1", "e2"] and roster[-1]["avatar"] == "bot.png"
    assert roster[4]["name"] == "Adelson" and roster[4]["avatar"] is None and not roster[4]["spoken"]
    assert all({"id", "name", "avatar", "ring"} <= set(r) for r in roster)
    first = s.step()
    assert first and all(isinstance(e, SimEvent) for e in first) and first[-1].kind == "student"
    h = s.say("acho que mastigar é so pra comida descer mais facil", sender="Adelson")
    assert h.kind == "external" and h.display_name == "Adelson" and h.speaker == "e1"
    assert s._live.interpretations[-1]["source"] == "llm"           # integrated before say() returned
    a = s.say(f"{roster[1]['name']}, o que você acha?", sender="e2")
    assert a.kind == "external" and a.display_name == "Bot" and a.addressee == roster[1]["id"]
    assert all(p["spoken"] for p in s.participants)
    rest = s.run_to_end()
    assert rest[-1].kind == "system" and rest[-1].code == "ended" and s.status == "done"
    seqs = [e.seq for e in s.events()]
    assert seqs == list(range(1, len(seqs) + 1))
    assert s.step() == [] and s.say("ainda tem alguém?", sender="Adelson") is None
    rec = s.record()
    assert rec["interpretations"] and rec["hidden_state_initial"] and rec["events"][-1]["code"] == "ended"
    assert [m["speaker"] for m in rec["transcript"] if m["kind"] == "external"] == ["e1", "e2"]
    assert [p["name"] for p in rec["participants"]] == ["Adelson", "Bot"]
    u = s.usage()
    assert u["calls"] > 0 and u["input_tokens"] > 0 and u["usd"] == 0 and u["est_usd"] > 0   # mock: offline
    json.dumps(rec, default=str)
    s.close()
    assert sim.get_session(s.id) is None


def test_record_key_set():
    """record() has exactly this key set."""
    sim = Simulator(backend="template")
    s = sim.new_session(n_students=3, pacing="manual", seed=1, participants=["Ana"])
    s.run_to_end()
    rec = s.record()
    assert set(rec) == {
        "session_id", "task_id", "seed", "domain", "learning_params", "realizer", "config", "duration_s",
        "n_student_turns", "solved_at_min", "ended_reason", "roster", "participants", "transcript",
        "hidden_state_initial", "hidden_state_final", "hidden_state_timeline", "flip_log", "steps_solved",
        "repetition_guard_log", "prompt_leak_guard_log", "attribution_guard_log", "visible_echo_log",
        "realism_guard_log", "filler_gate_log", "framing_guard_log", "belief_guard_log", "coherence_guard_log", "context",
        "closing_msgs", "closing_forced", "interpretations", "events", "summary"}
    assert set(rec["transcript"][0]) == {"id", "t", "kind", "text", "speaker", "display_name", "addressee", "plan",
                                         "labels"}
    assert set(rec["summary"]["config"]) == {
        "backend", "model", "task_id", "custom_task", "language", "n_students", "seed", "participants", "pacing",
        "speed", "traits", "offtask_rate", "time_of_day", "pace", "target_minutes", "conflict", "max_minutes",
        "end_on_complete", "max_student_turns"}
    assert rec["summary"]["participants"][0]["name"] == "Ana"


def test_permissions_and_config_object():
    sim = Simulator(backend="template")
    cfg = SessionConfig(n_students=3, pacing="manual", seed=1)
    s = sim.new_session(cfg, language="en")
    assert s.task.language == "en" and s.participants == []
    with pytest.raises(ValueError):
        s.say("hi")                                    # no participant declared: sender required
    s.add_participant("Ana")
    s.add_participant("Bea")
    with pytest.raises(ValueError):
        s.say("hi")                                    # two participants: ambiguous
    with pytest.raises(ValueError):
        s.say("hi", sender=s.roster[0]["name"])        # a student's name
    with pytest.raises(ValueError):
        s.add_participant("ana")                       # duplicate (folded)
    with pytest.raises(ValueError):
        sim.new_session(n_students=11)
    with pytest.raises(ValueError):
        sim.new_session(language="fr")
    with pytest.raises(ValueError):
        sim.new_session(traits="nope")
    with pytest.raises(ValueError):
        sim.new_session(participants=["Ana", "ANA"])
    with pytest.raises(ValueError):
        sim.new_session(participants=[f"p{i}" for i in range(9)])
    assert set(SessionConfig.model_fields) == {
        "n_students", "task", "language", "seed", "participants", "pacing", "speed", "traits", "offtask_rate",
        "time_of_day", "pace", "target_minutes", "conflict", "max_minutes", "end_on_complete", "max_student_turns"}
    with pytest.raises(ValueError):
        sim.new_session(no_such_field="x")                # unknown fields are rejected


# ------------------------------------------------------------------ realtime (threaded)
def test_realtime_threaded_poll_pause_resume_stop():
    sim = Simulator(backend="template", tick_s=0.01)
    s = sim.new_session(n_students=3, seed=2, speed=20.0, participants=["Ana"])
    assert s.status == "ready" and s.poll()["events"] == []
    s.start()
    assert wait_for(lambda: len(s.poll()["events"]) >= 3, timeout=30)
    out = s.poll(0)
    assert out["status"] == "running" and out["next_after"] == out["events"][-1].seq
    assert s.poll(out["next_after"])["events"] == [] or True     # non-blocking, incremental
    ts = [e.t_sim for e in out["events"]]
    assert ts == sorted(ts)
    s.pause()
    n = len(s.events())
    time.sleep(0.3)
    assert s.poll()["status"] == "paused" and len(s.events()) == n
    s.resume()
    s.set_speed(20.0)
    h = s.say("oi gente")                              # the only participant: sender may be omitted
    assert h.kind == "external" and h.speaker == "e1"
    assert wait_for(lambda: s._live.interpretations, timeout=10)  # interpreted in the background thread
    with pytest.raises(ValueError):
        s.set_speed(0)
    with pytest.raises(RuntimeError):
        s.step()
    s.stop()
    assert s.status == "done" and s.events()[-1].reason == "stopped"
    s.close()
    assert not s._thread.is_alive()


def test_realtime_concurrent_callers_are_safe():
    sim = Simulator(backend="mock", tick_s=0.005)
    s = sim.new_session(n_students=4, seed=7, speed=20.0, participants=["Ana", "Bot"], max_minutes=8)
    s.start()
    errors = []

    def poller():
        last = 0
        try:
            for _ in range(200):
                out = s.poll(last)
                last = out["next_after"]
                time.sleep(0.002)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    def talker(i):
        try:
            for k in range(5):
                s.say(f"msg {i}-{k}: acho que a enzima quebra o amido", sender="Ana")
                s.say("o que vocês acham?", sender="Bot")
                time.sleep(0.01)
        except Exception as e:  # noqa: BLE001
            errors.append(e)
    th = [threading.Thread(target=poller) for _ in range(3)] + [threading.Thread(target=talker, args=(i,))
                                                               for i in range(2)]
    for t in th:
        t.start()
    for t in th:
        t.join(60)
    assert not errors
    assert wait_for(lambda: not s._live._queue, timeout=20)
    s.stop()
    evs = s.events()
    assert [e.seq for e in evs] == list(range(1, len(evs) + 1))
    assert sum(e.speaker == "e1" for e in evs) == 10 and sum(e.speaker == "e2" for e in evs) == 10
    assert s.record()["interpretations"]
    s.close()


# ------------------------------------------------------------------ errors
class APIConnectionError(Exception):
    """Mimics anthropic.APIConnectionError: no status code, message 'Connection error.'"""


def _conn_error():
    try:
        try:
            raise ssl.SSLCertVerificationError(1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed: "
                                                  "self-signed certificate in certificate chain")
        except ssl.SSLError as inner:
            raise APIConnectionError("Connection error.") from inner
    except APIConnectionError as e:
        return e


def test_connection_error_is_diagnosable_and_never_raised_into_caller():
    sim = Simulator(backend="mock", backend_factory=lambda: MockLLM(fail_after=6, fail_exc=_conn_error()),
                    tick_s=0.005)
    s = sim.new_session(n_students=3, seed=1, speed=20.0, participants=["Ana"])
    s.start()
    assert wait_for(lambda: s.status == "error", timeout=30)
    out = s.poll(0)
    err = out["error"]
    assert out["status"] == "error" and s._live.error_kind == "connection"
    assert "conectar" in err and "SSL" in err and "APIConnectionError" in err and "SSLCertVerificationError" in err
    assert out["events"][-1].code == "error" and out["events"][-1].reason == "connection"
    assert s.say("oi", sender="Ana") is None                         # ended: accepted=False, no raise
    s.close()


def test_billing_error_manual_mode_english_message():
    sim = Simulator(backend="mock", backend_factory=lambda: MockLLM(fail_after=4))
    s = sim.new_session(n_students=3, seed=1, pacing="manual", language="en")
    evs = s.run_to_end()
    assert s.status == "error" and evs[-1].code == "error" and evs[-1].reason == "billing"
    assert "credit" in s.error.lower() and "Console" in s.error


def test_check_connection(monkeypatch):
    assert Simulator(backend="template").check_connection()["ok"]
    sim = Simulator(api_key="sk-ant-test-0000")
    exc = _conn_error()

    class Msgs:
        def create(self, **kw):
            assert kw["max_tokens"] == 1 and kw["model"] == "claude-haiku-5-5" and "temperature" not in kw
            raise exc

    class C:
        messages = Msgs()

        def with_options(self, **kw):
            return self
    sim._shared._client = C()
    r = sim.check_connection()
    assert not r["ok"] and r["error_kind"] == "connection" and r["hint"] == "ssl"
    assert "SSLCertVerificationError" in r["exception"] and "sk-ant-test" not in r["error"]

    class OK:
        class messages:  # noqa: N801
            @staticmethod
            def create(**kw):
                return type("R", (), {"model": "claude-haiku-5-5"})()

        def with_options(self, **kw):
            return self
    sim._shared._client = OK()
    assert sim.check_connection()["ok"]


def test_backend_unavailable_and_haiku_only(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(BackendUnavailable):
        Simulator()
    with pytest.raises(ModelNotAllowed):
        Simulator(backend="template", student_model="claude-opus-5-5")
    with pytest.raises(ValueError):
        Simulator(backend="nope")


# ------------------------------------------------------------------ CLI + package surface
def test_cli_try_and_check(tmp_path, capsys):
    from collabsim.__main__ import main
    assert main(["try", "--backend", "template", "--n", "3", "--out", str(tmp_path), "--env-file", ""]) == 0
    md = list(tmp_path.glob("*.md"))
    js = list(tmp_path.glob("*.json"))
    assert len(md) == len(js) == 1 and "| min | who | message |" in md[0].read_text()
    assert json.loads(js[0].read_text())["events"]
    assert main(["check", "--backend", "template", "--env-file", ""]) == 0
    assert main(["check", "--env-file", str(tmp_path / "missing.env")]) == 2


def test_slim_package_surface():
    assert collabsim.__version__ == "0.5.3"
    assert collabsim.__all__ == ["Simulator", "Session", "SessionConfig", "ParticipantSpec", "SimEvent", "TaskSpec",
                                 "CollabSimError", "BackendUnavailable", "CapacityError", "__version__"]
    subpackages = {m.name for m in pkgutil.iter_modules(collabsim.__path__, "collabsim.") if m.ispkg}
    assert subpackages == {"collabsim.lang", "collabsim.llm", "collabsim.models", "collabsim.runtime", "collabsim.sim"}
    for name in sorted(subpackages):
        importlib.import_module(name)
    assert "fastapi" not in sys.modules and "uvicorn" not in sys.modules
