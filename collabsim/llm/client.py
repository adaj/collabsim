"""LLM client: Haiku-only backends, persistent on-disk response cache, cache-aware usage meter and
graceful stop on billing / spend-limit errors.

* `LLMRequest`: cached `prefix` (system block with cache_control) + uncached `suffix`. The model
  is checked against the allowlist when the request is constructed.
* `AnthropicBackend`: real SDK calls (key from ANTHROPIC_API_KEY); SDK retries transient errors;
  billing / credit / permission / rate-limit errors after retries raise `GracefulStop`.
* `MockLLM`: deterministic offline backend routed by `purpose` (collabsim.llm.mock).
* `LLMClient`: every call goes through it. Responses are cached by
  sha256(model, prefix, suffix, temperature, max_tokens, replicate_idx) and written atomically,
  so interrupted work is never re-paid and reruns replay exactly.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Protocol

from pydantic import BaseModel, Field

from .config import request_extras, MIN_CACHEABLE_TOKENS, check_model, price_for


class GracefulStop(RuntimeError):
    """The API refused further work (credit / spend limit / permission / persistent rate limit /
    connection failure after retries). A live session catches it, keeps every released event and
    ends with status "error". `kind`: billing | auth | rate_limit | connection | refused;
    `detail`: the underlying exception chain (collabsim.errors.describe_error); `hint`: for
    connection errors, ssl | proxy | dns | timeout | refused | reset (or None)."""

    def __init__(self, message: str, kind: str = "refused", detail: str | None = None, hint: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.detail = detail
        self.hint = hint


class FakeAPIError(Exception):
    """Mimics anthropic.APIStatusError (status_code + message) for offline tests."""

    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class LLMRequest(BaseModel):
    model: str
    purpose: str
    prefix: str = ""
    suffix: str
    temperature: float = 1.0
    max_tokens: int = 60
    replicate_idx: int = 0
    json_mode: bool = False
    meta: dict = Field(default_factory=dict)  # structured context for the mock (never sent)

    def __init__(self, **kw):
        check_model(kw.get("model", ""))  # raises ModelNotAllowed before any other validation
        super().__init__(**kw)

    def key(self) -> str:
        blob = json.dumps([self.model, self.prefix, self.suffix, self.temperature, self.max_tokens,
                           self.replicate_idx, self.json_mode], ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()

    def system_blocks(self) -> list[dict]:
        """One system block holding the whole stable prefix (+ JSON instruction), with the cache
        breakpoint on it. Nothing variable precedes the breakpoint."""
        if not self.prefix:
            return []
        text = self.prefix + ("\n\nRespond with a single JSON object and nothing else." if self.json_mode else "")
        # Always mark the breakpoint: below the model's minimum the API simply does not cache (no
        # error, no surcharge). Every production prefix is built to exceed MIN_CACHEABLE_TOKENS
        # (asserted by tests/test_frugal.py).
        return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


class Usage(BaseModel):
    input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0


class LLMResponse(BaseModel):
    text: str
    usage: Usage = Field(default_factory=Usage)
    cached: bool = False
    backend: str = ""
    model: str = ""            # model reported by the API
    stop_reason: str = ""
    batch: bool = False
    latency_s: float = 0.0


class Backend(Protocol):
    name: str

    def complete(self, req: LLMRequest) -> LLMResponse: ...


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token), used by the mock and the dry-run estimator."""
    return max(1, len(text) // 4) if text else 0


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ backends
# Errors that mean "the API will not do more work for this key right now". They stop the run
# GRACEFULLY (everything done is kept; rerun the same command to resume). There is no spend cap
# in this package: spend control lives in the user's Console workspace limit.
STOP_HINTS = ("credit", "billing", "spend", "usage limit", "api usage limits", "quota", "balance", "payment")
RETRY_STATUS = (429, 529)              # rate limit / overloaded: extra backoff, then stop gracefully
STOP_STATUS = (401, 402, 403)          # auth / payment / permission
BACKOFF_S = (15.0, 45.0, 90.0)         # after the SDK's own retries (max_retries) are exhausted
CONNECTION_BACKOFF_S = (2.0, 5.0)      # connection failures: short extra retries, then a diagnosable error


def _status(e) -> int | None:
    st = getattr(e, "status_code", None)
    if st is None and getattr(e, "response", None) is not None:
        st = getattr(e.response, "status_code", None)
    return st


def _msg(e) -> str:
    from ..runio import mask  # local import: runio imports this module
    return mask(str(getattr(e, "message", "") or e))[:300]


def _is_connection_error(e) -> bool:
    name = type(e).__name__
    return _status(e) is None and ("Connection" in name or "Timeout" in name)


def classify_error(e) -> str:
    """'stop' (graceful stop now) | 'retry' (back off, then graceful stop) | 'connection' | 'raise'."""
    if isinstance(e, GracefulStop):
        return "stop"
    st = _status(e)
    low = _msg(e).lower()
    if st in STOP_STATUS or (st == 400 and any(h in low for h in STOP_HINTS)):
        return "stop"
    if st in RETRY_STATUS or "overloaded" in low or "rate limit" in low:
        return "retry"
    if _is_connection_error(e):
        return "connection"
    return "raise"


def _stop_kind(e) -> str:
    low = _msg(e).lower()
    if any(h in low for h in STOP_HINTS) or _status(e) == 402:
        return "billing"
    return "auth" if _status(e) in (401, 403) else "refused"


def call_with_backoff(fn, sleep=time.sleep, backoff=BACKOFF_S, log=print,
                      connection_backoff=CONNECTION_BACKOFF_S):
    from ..errors import describe_error, diagnose
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            kind = classify_error(e)
            if kind == "stop":
                if isinstance(e, GracefulStop):
                    raise
                raise GracefulStop(f"API refused the request ({_status(e)}): {_msg(e)}", kind=_stop_kind(e),
                                   detail=describe_error(e)) from e
            waits = backoff if kind == "retry" else connection_backoff
            if kind in ("retry", "connection") and attempt < len(waits):
                log(f"  API {'busy/rate-limited' if kind == 'retry' else 'connection error'} ({_status(e)}); "
                    f"backing off {waits[attempt]:.0f}s (retry {attempt + 1}/{len(waits)})")
                sleep(waits[attempt])
                attempt += 1
                continue
            if kind == "retry":
                raise GracefulStop(f"API still rate-limited/unavailable after retries ({_status(e)}): "
                                   f"{_msg(e)}", kind="rate_limit", detail=describe_error(e)) from e
            if kind == "connection":
                detail = describe_error(e)
                raise GracefulStop(f"API connection failed after retries: {detail}", kind="connection",
                                   detail=detail, hint=diagnose(e)) from e
            raise


class AnthropicBackend:
    name = "anthropic"
    sleep = staticmethod(time.sleep)

    def __init__(self, api_key: str | None = None, max_retries: int = 2, timeout: float = 30.0,
                 log: Callable[[str], None] | None = None):
        import anthropic  # imported lazily so the package works without it

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set (pass api_key=..., set the env var, or use "
                               "backend='mock'/'template' offline)")
        self._anthropic = anthropic
        self._client = anthropic.Anthropic(api_key=key, max_retries=max_retries, timeout=timeout)
        self.log = log or (lambda m: None)

    def _params(self, req: LLMRequest) -> dict:
        # Haiku 5.5 (migration guide): no temperature/top_p/top_k (non-default values are a 400);
        # adaptive thinking is on by default and its tokens count toward max_tokens, so our short,
        # tightly capped calls request thinking off (THINKING in config). req.temperature is kept only
        # as part of the cache key for backwards-compatible hashing; it is never sent.
        p = {"model": check_model(req.model), "max_tokens": req.max_tokens,
             "messages": [{"role": "user", "content": req.suffix}]}
        p.update(request_extras())
        sb = req.system_blocks()
        if sb:
            p["system"] = sb
        return p

    @staticmethod
    def _usage(u) -> Usage:
        return Usage(input_tokens=u.input_tokens, output_tokens=u.output_tokens,
                     cache_creation_input_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0,
                     cache_read_input_tokens=getattr(u, "cache_read_input_tokens", 0) or 0)

    def complete(self, req: LLMRequest) -> LLMResponse:
        t0 = time.monotonic()
        params = self._params(req)
        r = call_with_backoff(lambda: self._client.messages.create(**params), sleep=self.sleep,
                              log=getattr(self, "log", print))
        text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        if (getattr(r, "stop_reason", "") or "") == "refusal":
            text = ""  # safeguard refusal: callers treat empty text as a fallback (logged via stop_reason)
        return LLMResponse(text=text, backend=self.name, model=getattr(r, "model", ""), usage=self._usage(r.usage),
                           latency_s=round(time.monotonic() - t0, 3), stop_reason=getattr(r, "stop_reason", "") or "")


Handler = Callable[[LLMRequest], str]


class MockLLM:
    """Deterministic offline backend. `handlers[purpose](req) -> text`. Usage is simulated as if
    the prefix were cached after its first use (cache-aware counting). Like the real API, an output
    longer than `max_tokens` is cut off (stop_reason "max_tokens")."""
    name = "mock"

    def __init__(self, handlers: dict[str, Handler] | None = None, fail_after: int | None = None,
                 fail_status: int = 400, fail_exc: BaseException | None = None):
        from .mock import default_handlers
        self.handlers = default_handlers()
        self.handlers.update(handlers or {})
        self._seen_prefix: set[str] = set()
        self.calls = 0                # backend calls actually served (== calls a real run would pay)
        self.fail_after = fail_after  # tests: raise a fake billing error after N served calls
        self.fail_status = fail_status
        self.fail_exc = fail_exc        # tests: raise this exception instead (e.g. a fake connection error)

    def _maybe_fail(self) -> None:
        if self.fail_after is not None and self.calls >= self.fail_after:
            if self.fail_exc is not None:
                exc = self.fail_exc
                call_with_backoff(lambda: (_ for _ in ()).throw(exc), sleep=lambda s: None, log=lambda m: None)
            e = FakeAPIError(self.fail_status, "Your credit balance is too low to access the Anthropic API. "
                                               "Please go to Plans & Billing to upgrade or purchase credits.")
            call_with_backoff(lambda: (_ for _ in ()).throw(e), sleep=lambda s: None, log=lambda m: None)

    def complete(self, req: LLMRequest) -> LLMResponse:
        check_model(req.model)
        self._maybe_fail()
        self.calls += 1
        h = self.handlers.get(req.purpose)
        if h is None:
            raise KeyError(f"MockLLM has no handler for purpose {req.purpose!r}")
        text = h(req)
        sb = req.system_blocks()
        pt = approx_tokens(sb[0]["text"]) if sb else 0
        ph = hashlib.sha256((req.model + (sb[0]["text"] if sb else "")).encode()).hexdigest()
        if not sb or pt < MIN_CACHEABLE_TOKENS:  # the API does not cache short prefixes
            u = Usage(input_tokens=pt + approx_tokens(req.suffix))
        elif ph in self._seen_prefix:
            u = Usage(cache_read_input_tokens=pt, input_tokens=approx_tokens(req.suffix))
        else:
            self._seen_prefix.add(ph)
            u = Usage(cache_creation_input_tokens=pt, input_tokens=approx_tokens(req.suffix))
        stop = "end_turn"
        if approx_tokens(text) > req.max_tokens:   # cut off at the output cap, like the real API
            text, stop = text[:req.max_tokens * 4], "max_tokens"
        u.output_tokens = min(req.max_tokens, approx_tokens(text))
        return LLMResponse(text=text, usage=u, backend=self.name, model=req.model, stop_reason=stop)


# ------------------------------------------------------------------ metering
def response_usd(req: LLMRequest, resp: LLMResponse) -> float:
    u = resp.usage
    p = price_for(req.model)
    f = p.batch_discount if resp.batch else 1.0
    return f * (u.input_tokens * p.input + u.cache_creation_input_tokens * p.cache_write_5m
                + u.cache_read_input_tokens * p.cache_read + u.output_tokens * p.output) / 1e6


OFFLINE_BACKENDS = frozenset({"mock"})


class UsageMeter:
    """Two views. PAID (`usd`, token counts): what this process actually sent to the API (local
    cache hits are free). UNIT (`unit_*`, `reported_models`): every DISTINCT response this meter
    saw, paid now or replayed from the on-disk cache, i.e. the true cost and API-reported model of
    the work, invariant under interruption and resume.

    An offline backend (mock) costs nothing: its `usd` / `unit_usd` stay 0 while calls and tokens are
    counted. `est_usd` prices the same tokens at the model's rates on every backend (for the API
    backend it equals `usd`), i.e. what the work would cost against the API."""

    def __init__(self):
        self.by: dict[str, dict] = defaultdict(lambda: defaultdict(float))
        self.models_seen: set[str] = set()
        self.reported_models: dict[str, int] = defaultdict(int)
        self._unit_keys: set[str] = set()

    def add(self, req: LLMRequest, resp: LLMResponse, cached_locally: bool) -> None:
        d = self.by[f"{req.purpose}@{req.model}"]
        d["calls"] += 1
        if resp.model:
            self.models_seen.add(resp.model)
        k = req.key()
        if k not in self._unit_keys:
            self._unit_keys.add(k)
            self.reported_models[resp.model or "<missing>"] += 1
            u = resp.usage
            d["unit_calls"] += 1
            d["unit_usd"] += 0.0 if resp.backend in OFFLINE_BACKENDS else response_usd(req, resp)
            d["unit_cache_read_tokens"] += u.cache_read_input_tokens
            d["unit_cache_write_tokens"] += u.cache_creation_input_tokens
            d["truncated"] += 1 if resp.stop_reason == "max_tokens" else 0
        if cached_locally:
            d["local_cache_hits"] += 1
            return
        u = resp.usage
        d["batch_calls"] += 1 if resp.batch else 0
        d["latency_s"] += resp.latency_s
        d["input_tokens"] += u.input_tokens
        d["cache_write_tokens"] += u.cache_creation_input_tokens
        d["cache_read_tokens"] += u.cache_read_input_tokens
        d["output_tokens"] += u.output_tokens
        cost = response_usd(req, resp)
        d["est_usd"] += cost
        d["usd"] += 0.0 if resp.backend in OFFLINE_BACKENDS else cost

    def total_usd(self) -> float:
        return sum(v.get("usd", 0.0) for v in self.by.values())

    def totals(self) -> dict:
        keys = ("calls", "local_cache_hits", "batch_calls", "truncated", "input_tokens", "cache_write_tokens",
                "cache_read_tokens", "output_tokens", "usd", "est_usd", "latency_s",
                "unit_calls", "unit_usd", "unit_cache_read_tokens", "unit_cache_write_tokens")
        return {k: sum(v.get(k, 0) for v in self.by.values()) for k in keys}

    def summary(self) -> dict:
        out = {k: dict(v) for k, v in self.by.items()}
        out["_total_usd"] = round(self.total_usd(), 6)
        out["_models_seen"] = sorted(self.models_seen)
        out["_reported_models"] = dict(self.reported_models)
        return out


# ------------------------------------------------------------------ cached client
class LLMClient:
    """What the simulator talks to. All LLM calls go through `complete`."""

    def __init__(self, backend: Backend, cache_dir: Path | str | None = None):
        self.backend = backend
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meter = UsageMeter()

    def _path(self, key: str) -> Path:
        return self.cache_dir / key[:2] / f"{key}.json"

    def cached(self, req: LLMRequest) -> LLMResponse | None:
        if not self.cache_dir:
            return None
        p = self._path(req.key())
        if p.exists():
            try:
                resp = LLMResponse.model_validate_json(p.read_text())
            except Exception:  # noqa: BLE001  (corrupt partial file: treat as missing)
                return None
            resp.cached = True
            return resp
        return None

    def _store(self, req: LLMRequest, resp: LLMResponse) -> None:
        if self.cache_dir:
            atomic_write(self._path(req.key()), resp.model_dump_json())

    def complete(self, req: LLMRequest) -> LLMResponse:
        hit = self.cached(req)
        if hit is not None:
            self.meter.add(req, hit, cached_locally=True)
            return hit
        resp = self.backend.complete(req)
        self.meter.add(req, resp, cached_locally=False)
        self._store(req, resp)
        return resp

    @staticmethod
    def parse_json(text: str) -> dict:
        try:
            s, e = text.index("{"), text.rindex("}") + 1
            return json.loads(text[s:e])
        except (ValueError, json.JSONDecodeError):
            return {}

    def complete_json(self, req: LLMRequest) -> dict:
        req.json_mode = True
        return self.parse_json(self.complete(req).text)


DEFAULT_CACHE_DIR = Path(".cache/llm")  # persistent: whatever was paid for is never paid again


def make_client(backend: str = "mock", cache_dir: Path | str | None = None,
                handlers: dict[str, Handler] | None = None) -> LLMClient:
    if backend == "mock":
        # test hook: COLLABSIM_MOCK_FAIL_AFTER=N makes the mock raise a fake billing error (HTTP 400,
        # "credit balance is too low") after N served calls, to exercise graceful stop + resume.
        fa = os.environ.get("COLLABSIM_MOCK_FAIL_AFTER")
        return LLMClient(MockLLM(handlers, fail_after=int(fa) if fa else None), cache_dir)
    if backend in ("llm", "anthropic"):
        return LLMClient(AnthropicBackend(), cache_dir or DEFAULT_CACHE_DIR)
    raise ValueError(f"unknown backend {backend!r}")
