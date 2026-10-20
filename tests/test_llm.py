"""LLM layer (offline; mock backend only)."""
import json

import pytest

from collabsim.llm import LLMClient, LLMRealizer, LLMRequest, MockLLM, ModelConfig, make_client
from collabsim.llm.client import AnthropicBackend
from collabsim.llm.prompts import fewshots, realizer_prefix
from collabsim.models import load_task
from collabsim.sim import Simulation
from collabsim.sim.factory import build_session


def _req(**kw):
    d = dict(model="claude-haiku-5-5", purpose="echo", prefix="P" * 4000, suffix="hello", temperature=1.0)
    d.update(kw)
    return LLMRequest(**d)


def test_hash_cache_on_disk(tmp_path):
    calls = []
    client = LLMClient(MockLLM({"echo": lambda r: calls.append(1) or f"x{r.replicate_idx}"}), tmp_path)
    a = client.complete(_req())
    b = client.complete(_req())
    c = client.complete(_req(replicate_idx=1))
    d = client.complete(_req(temperature=0.0))
    assert a.text == b.text and b.cached and not a.cached
    assert c.text == "x1" and not c.cached and not d.cached
    assert len(calls) == 3 and list(tmp_path.rglob("*.json"))
    client2 = LLMClient(MockLLM({"echo": lambda r: "never"}), tmp_path)  # new process, same cache
    assert client2.complete(_req()).text == a.text


def test_cache_aware_usage_metering():
    client = LLMClient(MockLLM({"echo": lambda r: "ok"}))
    client.complete(_req())
    client.complete(_req(suffix="again"))
    client.complete(_req(prefix="short", suffix="x"))
    s = client.meter.summary()["echo@claude-haiku-5-5"]
    assert s["cache_write_tokens"] == 1000 and s["cache_read_tokens"] == 1000
    assert s["usd"] == 0 and s["est_usd"] > 0          # mock is offline: free, but priced as an estimate


def test_model_config_pinned_and_env_override(monkeypatch):
    from collabsim.llm.config import ModelNotAllowed
    m = ModelConfig()
    assert m.student == m.judge == "claude-haiku-5-5" and set(ModelConfig.model_fields) == {"student", "judge"}
    monkeypatch.setenv("COLLABSIM_STUDENT_MODEL", "claude-haiku-5-5")
    assert ModelConfig.from_env().student == "claude-haiku-5-5"
    monkeypatch.setenv("COLLABSIM_JUDGE_MODEL", "claude-sonnet-5-5")
    with pytest.raises(ModelNotAllowed, match="haiku"):
        ModelConfig.from_env()


def test_haiku_only_allowlist_everywhere():
    from collabsim.llm.config import ModelNotAllowed, price_for
    for bad in ("claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-4-5", "gpt-4"):
        with pytest.raises(ModelNotAllowed):
            LLMRequest(model=bad, purpose="x", suffix="hi")
        with pytest.raises(ModelNotAllowed):
            ModelConfig(student=bad)
        with pytest.raises(ModelNotAllowed):
            price_for(bad)
    with pytest.raises(ModelNotAllowed):
        MockLLM().complete(LLMRequest.model_construct(model="claude-sonnet-5-5", purpose="x", suffix="hi",
                                                      prefix="", temperature=0, max_tokens=5, replicate_idx=0,
                                                      json_mode=False, meta={}))


def test_anthropic_backend_requires_key_and_marks_cache_control(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        AnthropicBackend()
    sent = {}

    class FakeMessages:
        def create(self, **kw):
            sent.update(kw)
            class U: input_tokens = 3; output_tokens = 2; cache_creation_input_tokens = 900; cache_read_input_tokens = 0
            class B: type = "text"; text = "hey"
            class R: content = [B()]; usage = U()
            return R()

    be = AnthropicBackend.__new__(AnthropicBackend)
    be._client = type("C", (), {"messages": FakeMessages()})()
    r = be.complete(_req())
    assert r.text == "hey" and r.usage.cache_creation_input_tokens == 900
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}  # 4000-char prefix >= 512 tokens
    assert len(sent["system"]) == 1 and sent["model"] == "claude-haiku-5-5"
    be.complete(_req(prefix="short"))
    assert sent["system"][0]["text"].startswith("short")  # breakpoint harmless below the minimum


def test_fewshots_and_prefix():
    fs = fewshots()
    assert len(fs) == 30 and all(f["en"] and "pt" not in f for f in fs)
    p = realizer_prefix(load_task("sci-enzyme-01"))
    assert "SAY THIS CONTENT" in p and "STYLE EXAMPLES" in p
    assert all(f["en"] in p for f in fs)


def test_state_updates_from_plan_not_text():
    """The state rules read the plans, never the wording: replaying a (mock LLM) session's plans with
    any other text gives the identical hidden-state timeline."""
    b = build_session("sci-enzyme-01", 5, realizer="mock")
    b.run()

    def replay(text_of):
        a = Simulation("sci-enzyme-01", 5)
        a._started = True
        for m in b.state.messages:
            a.inject_student(m.speaker, m.plan, text_of(m), delay=max(0.0, m.t - a.state.t))
        return a.state.timeline
    same = replay(lambda m: m.text)
    other = replay(lambda m: "lorem ipsum")
    assert same == other and len(same) == len(b.state.messages)
    assert [t["students"] for t in same] == [t["students"] for t in other]

def test_leakage_guard_regenerate_and_fallback():
    seen = []

    def always_leak(req):
        seen.append(req.meta.get("strict"))
        f = req.meta["unknown_facts"][0] if req.meta["unknown_facts"] else None
        task = load_task(req.meta["task_id"])
        return "ok " + (task.fact(f).chat[0] if f else "")

    client = make_client("mock", handlers={"realize": always_leak})
    sim = build_session("sci-enzyme-01", 3, realizer="mock", client=client)
    sim.run(max_events=40)
    log = sim.realizer.log
    leaked = [r for r in log if r["pre_guard_leak"]]
    assert leaked and all(r["attempts"] == 2 and r.get("fallback") for r in leaked)
    assert all(not r["post_guard_leak"] for r in log)
    assert True in seen  # STRICT regeneration was requested


def test_llm_guard_catches_paraphrase_and_merged_check_is_one_call():
    """A leak the keyphrase check misses is caught by the merged Haiku check (guard+audit, 1 call)."""
    from collabsim.llm import mock

    # paraphrases with no keyphrase (the code check cannot see them); D1 v4: the flag carries a quote
    para = {"f_denature": "if its super hot the shape gets wrecked and it just stops",
            "f_optimum": "human spit stuff is fastest around 37", "f_surface": "smaller bits expose way more area to juices",
            "f_rate": "the breakdown only happens on the outer layer of the bits",
            "f_amylase": "spit turns starch into sugar", "f_reuse": "one molecule acts on loads of substrate"}

    def para_realize(req):
        if req.meta.get("strict") or not req.meta["unknown_facts"]:
            return mock.realize(req)
        return para[req.meta["unknown_facts"][0]]

    def para_leak(req):
        fid = next((f for f, t in para.items() if t == req.meta["text"]), None)
        if fid is None:
            return mock.check(req)
        return json.dumps({**json.loads(mock.check(req)), "leak": True,
                           "leaked_items": [{"id": fid, "quote": req.meta["text"]}]})

    client = make_client("mock", handlers={"check": para_leak, "realize": para_realize})
    sim = build_session("sci-enzyme-01", 3, realizer="mock", client=client)
    sim.run(max_events=60)
    llm_turns = [r for r in sim.realizer.log if not r.get("templated")]
    by = [r.get("detected_by") for r in llm_turns if r["pre_guard_leak"]]
    assert llm_turns and by.count("llm") >= len(by) - by.count("code") and "llm" in by
    assert all(not r["post_guard_leak"] for r in llm_turns)
    summ = client.meter.summary()
    assert "guard@claude-haiku-5-5" not in summ and "classify_turn@claude-haiku-5-5" not in summ
    assert summ["check@claude-haiku-5-5"]["calls"] >= len(llm_turns)


def test_deterministic_acts_are_templated():
    from collabsim.llm.realizer import TEMPLATED_ACTS
    sim = build_session("sci-enzyme-01", 6, realizer="mock")
    sim.run()
    log = sim.realizer.log
    tpl = [r for r in log if r.get("templated")]
    assert tpl and all(r["act"] in TEMPLATED_ACTS or r["attempts"] == 0 for r in tpl)
    realize_calls = sim.client.meter.summary()["realize@claude-haiku-5-5"]["calls"]
    assert realize_calls == sum(r["attempts"] + r.get("repetition_guard", {}).get("extra_realize_calls", 0)
                                + r.get("realism_extra_realize_calls", 0)
                                for r in log + getattr(sim.realizer, "skipped_log", []))
    assert len(tpl) / len(log) > 0.2  # a sizeable share of turns needs no LLM (fillers are capped since 0.5.0)


def test_slippage_audit_logged():
    sim = build_session("math-ratio-01", 2, realizer="mock")
    sim.run()
    s = sim.realizer.summary()
    assert s["turns"] > 0 and s["slippage_rate"] is not None
    assert all("audit" in r for r in sim.realizer.log if not r.get("templated") and not r.get("fallback"))


def test_haiku_lock_backend(monkeypatch):
    """Backend refuses non-Haiku even for a request built without validation."""
    from collabsim.llm.config import ALLOWED_MODELS, ModelNotAllowed
    assert ALLOWED_MODELS == frozenset({"claude-haiku-5-5"})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-0000")
    be = AnthropicBackend()
    bad = LLMRequest.model_construct(model="claude-opus-5-5", purpose="x", suffix="hi", prefix="",
                                     temperature=0, max_tokens=5, replicate_idx=0, json_mode=False, meta={})
    with pytest.raises(ModelNotAllowed):
        be._params(bad)
