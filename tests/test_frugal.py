"""Frugal by design, without changing what is measured."""
from __future__ import annotations

import re

import pytest

from collabsim.llm.client import AnthropicBackend, LLMClient, LLMRequest, MockLLM, approx_tokens
from collabsim.llm.config import MAX_TOKENS, MIN_CACHEABLE_TOKENS
from collabsim.llm.realizer import CONTENT_ACTS, is_templated
from collabsim.models.message import ContentPlan
from collabsim.sim.factory import build_session


class FakeSDK:
    """Fake anthropic client that behaves like the API's prompt cache: a system block carrying
    cache_control is cached only if it is >= the model minimum (tokens estimated at 4 chars),
    and only an exact repeat of the prefix up to the breakpoint is a read."""

    def __init__(self):
        self.seen: set[str] = set()
        self.sent: list[dict] = []
        outer = self

        class Messages:
            def create(self, **kw):
                outer.sent.append(kw)
                sys_ = kw.get("system") or []
                pre = "".join(b["text"] for b in sys_)
                marked = any("cache_control" in b for b in sys_)
                pt = approx_tokens(pre)
                cw = cr = 0
                inp = approx_tokens(kw["messages"][0]["content"])
                if marked and pt >= MIN_CACHEABLE_TOKENS:
                    if pre in outer.seen:
                        cr = pt
                    else:
                        cw = pt
                        outer.seen.add(pre)
                else:
                    inp += pt
                purpose_json = "JSON" in kw["messages"][0]["content"]

                class U:
                    input_tokens = inp
                    output_tokens = 5
                    cache_creation_input_tokens = cw
                    cache_read_input_tokens = cr

                class B:
                    type = "text"
                    text = '{"answer": "A", "type": "silent", "leak": false, "act": "backchannel"}' \
                        if purpose_json else "ok"

                class R:
                    content = [B()]
                    usage = U()
                    model = kw["model"]
                    stop_reason = "end_turn"
                return R()

        self.messages = Messages()


def _fake_backend():
    be = AnthropicBackend.__new__(AnthropicBackend)
    be._client = FakeSDK()
    return be


def test_every_production_prefix_is_cacheable_and_read_after_warmup():
    """(c) Each role's stable prefix meets the 512-token minimum, carries cache_control, and the
    second call with that prefix is a cache READ (verified against a fake SDK)."""
    be = _fake_backend()
    client = LLMClient(be)
    for task in ("sci-enzyme-01", "math-ratio-01"):
        sim = build_session(task, 1, realizer="llm", client=client)
        sim.run(max_events=80)
    sent = be._client.sent
    purposes = set()
    for kw in sent:
        sys_ = kw["system"]
        assert len(sys_) == 1 and sys_[0]["cache_control"] == {"type": "ephemeral"}
        assert approx_tokens(sys_[0]["text"]) >= MIN_CACHEABLE_TOKENS, sys_[0]["text"][:80]
        # nothing per-call (names of the current speaker, history, timestamps) before the breakpoint
        assert "CHAT SO FAR" not in sys_[0]["text"] and "RECENT CHAT" not in sys_[0]["text"]
    t = client.meter.totals()
    assert t["cache_read_tokens"] > 5 * t["cache_write_tokens"] > 0
    for k, v in client.meter.summary().items():
        if not k.startswith("_"):
            purposes.add(k.split("@")[0])
            if v["calls"] >= 2:
                assert v["cache_read_tokens"] > 0, k
    assert {"realize", "check"} <= purposes


def test_check_and_interpret_prefixes_are_cacheable():
    """The per-turn check prefix and the interpretation prefix stay above the cache minimum in both
    languages (and for both library tasks), so each is written once and read afterwards."""
    from collabsim.llm.prompts import check_request, judge_prefix
    from collabsim.models import load_task
    from collabsim.runtime.interpret import interpret_prefix
    for tid in ("sci-enzyme-01", "math-ratio-01"):
        for lg in ("en", "pt-BR"):
            task = load_task(tid, language=lg)
            assert approx_tokens(judge_prefix(task)) >= MIN_CACHEABLE_TOKENS, (tid, lg)
            assert approx_tokens(interpret_prefix(task)) >= MIN_CACHEABLE_TOKENS, (tid, lg)
            sim = build_session(task, 1, realizer="mock")
            s = sim.state.students[0]
            c = check_request("claude-haiku-5-5", task, s, "hi", [], [], [], "-")
            assert c.prefix == judge_prefix(task) and c.system_blocks()[0]["cache_control"] == {"type": "ephemeral"}


def test_merged_check_one_call_per_llm_turn():
    """(a) guard + slippage audit + turn classification: ONE call per LLM-realized student turn."""
    client = LLMClient(MockLLM())
    sim = build_session("math-ratio-01", 2, realizer="mock", client=client)
    sim.run()
    log = sim.realizer.log
    s = client.meter.summary()
    llm_turns = [r for r in log if not r.get("templated")]
    assert s["check@claude-haiku-5-5"]["calls"] <= len(llm_turns) * 2
    assert s["check@claude-haiku-5-5"]["calls"] >= len([r for r in llm_turns if r.get("detected_by") != "code"])
    assert not any(k.startswith(("guard@", "classify_turn@")) for k in s)


@pytest.mark.parametrize("act,tl,expect", [
    ("greeting", 3, True), ("backchannel", 1, True), ("agree", 4, True), ("idk", 2, True),
    ("refusal", 1, True), ("off_task", 1, True),
    ("off_task", 6, False), ("interface", 7, False), ("on_task_claim", 1, False), ("question", 1, False),
    ("explain", 1, False), ("press", 1, False), ("disagree", 1, False)])
def test_templated_act_policy(act, tl, expect):
    """(b) closed-set, content-free acts and one-word atoms are templated; content acts never are."""
    p = ContentPlan(act=act)
    p.target_len = tl
    assert is_templated(p) is expect


def test_content_turns_always_llm_and_templated_turns_have_no_content():
    sim = build_session("sci-enzyme-01", 5, realizer="mock")
    sim.run()
    by_msg = {m.id: m for m in sim.state.messages}
    for r in sim.realizer.log:
        m = by_msg.get(r["msg"])
        # 0.5.1 short template lines with no content: afterthoughts ("ne isso?"), nudges, openers,
        # one-word okays and the short write-up turns ("pode colocar assim")
        # 0.5.2: the second half of a split line (already worded with the first) is not a new realization
        short = m.plan.tag or m.plan.nudge or m.plan.opener or m.plan.brief or m.plan.draft in ("agree", "typed") \
            or bool(m.plan.split_tail) or m.plan.step_move or m.plan.after_hedge
        # 0.5.3: "bora pra 2?" and a short hedge after one's own line ("acho") are template lines too
        if r.get("templated"):
            assert (r["act"] not in CONTENT_ACTS or short) and m.plan.claim == "none" and not m.plan.fact_ids
        if r["act"] in CONTENT_ACTS and not short:
            assert not r.get("templated")


def test_max_tokens_tight_per_role():
    """(d) every request uses its role's cap (<=160), never the old 200/80/20 defaults."""
    caps = []
    orig = MockLLM.complete

    def spy(self, req):
        caps.append((req.purpose, req.max_tokens))
        return orig(self, req)
    MockLLM.complete = spy
    try:
        client = LLMClient(MockLLM())
        sim = build_session("sci-enzyme-01", 3, realizer="mock", client=client)
        sim.run()
    finally:
        MockLLM.complete = orig
    for purpose, mt in caps:
        assert mt == MAX_TOKENS[purpose] and mt <= 160
    assert client.meter.totals()["truncated"] == 0
