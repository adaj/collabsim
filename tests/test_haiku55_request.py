"""Haiku 5.5 request shape (platform migration guide): no sampling params, thinking disabled,
refusals yield empty text."""
from types import SimpleNamespace

from collabsim.llm.client import AnthropicBackend, LLMRequest


def _backend():
    b = AnthropicBackend.__new__(AnthropicBackend)
    b.name = "anthropic"
    b.sleep = lambda s: None
    return b


def test_params_have_no_sampling_and_thinking_disabled():
    p = _backend()._params(LLMRequest(model="claude-haiku-5-5", purpose="x", suffix="hi", temperature=0.0, max_tokens=16))
    assert not {"temperature", "top_p", "top_k"} & set(p)
    assert p["thinking"] == {"type": "disabled"}


def test_refusal_returns_empty_text():
    b = _backend()
    resp = SimpleNamespace(content=[SimpleNamespace(type="text", text="I can't")], model="claude-haiku-5-5",
                           stop_reason="refusal",
                           usage=SimpleNamespace(input_tokens=1, output_tokens=1, cache_creation_input_tokens=0,
                                                 cache_read_input_tokens=0))
    b._client = SimpleNamespace(messages=SimpleNamespace(create=lambda **k: resp))
    r = b.complete(LLMRequest(model="claude-haiku-5-5", purpose="x", suffix="hi", max_tokens=16))
    assert r.text == "" and r.stop_reason == "refusal"
