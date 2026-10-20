"""Error types and human-readable (pt-BR / en) messages for API failures.

`describe_error` walks an exception's cause chain (``__cause__`` / ``__context__``), so an
``anthropic.APIConnectionError("Connection error.")`` is reported together with what actually
failed underneath it (``httpx.ConnectError`` -> ``ssl.SSLCertVerificationError`` ...), and
`diagnose` turns that chain into a short hint (ssl / proxy / dns / timeout / refused).
"""
from __future__ import annotations


class CollabSimError(RuntimeError):
    """Base class of the errors this package raises in the CALLER's thread."""


class BackendUnavailable(CollabSimError):
    """The LLM backend cannot be created (e.g. no API key for backend='llm')."""


class CapacityError(CollabSimError):
    """Too many concurrent sessions (Simulator(max_sessions=...))."""


def _mask(text: str) -> str:
    from .runio import mask
    return mask(text)


def _qualname(e: BaseException) -> str:
    mod = type(e).__module__ or ""
    root = mod.split(".")[0]
    return type(e).__name__ if root in ("builtins", "") else f"{root}.{type(e).__name__}"


def error_chain(e: BaseException, max_depth: int = 8) -> list[tuple[str, str]]:
    """[(qualified class name, message)] from the outermost exception down to the root cause."""
    out: list[tuple[str, str]] = []
    seen: set[int] = set()
    cur: BaseException | None = e
    while cur is not None and id(cur) not in seen and len(out) < max_depth:
        seen.add(id(cur))
        msg = str(getattr(cur, "message", "") or cur or "").strip()
        out.append((_qualname(cur), _mask(msg)[:300]))
        nxt = cur.__cause__
        if nxt is None and not cur.__suppress_context__:
            nxt = cur.__context__
        cur = nxt
    return out


def describe_error(e: BaseException) -> str:
    """'anthropic.APIConnectionError: Connection error. <- httpx.ConnectError: [SSL: ...] ...'"""
    parts, last = [], None
    for name, msg in error_chain(e):
        txt = f"{name}: {msg}" if msg else name
        if txt != last:
            parts.append(txt)
        last = txt
    return " <- ".join(parts)


HINTS = (  # (hint, substrings looked for in the lowercased chain)
    ("ssl", ("certificate_verify_failed", "ssl:", "sslerror", "sslcertverification", "certificate verify",
             "self signed", "self-signed", "unable to get local issuer")),
    ("proxy", ("proxyerror", "proxy", " 407", "tunnel connection failed")),
    ("dns", ("name or service not known", "nodename nor servname", "getaddrinfo", "temporary failure in name",
             "name resolution", "no address associated", "gaierror")),
    ("timeout", ("timeout", "timed out")),
    ("refused", ("connection refused", "connectionrefused")),
    ("reset", ("connection reset", "remoteprotocolerror", "server disconnected", "connectionreset")),
)


def diagnose(e: BaseException) -> str | None:
    low = describe_error(e).lower()
    for hint, needles in HINTS:
        if any(n in low for n in needles):
            return hint
    return None


HINT_TEXT = {
    "pt-BR": {"ssl": "falha de certificado SSL (proxy corporativo/antivírus interceptando HTTPS? ajuste SSL_CERT_FILE "
                     "ou REQUESTS_CA_BUNDLE para o certificado da rede)",
              "proxy": "falha no proxy HTTP(S) (verifique HTTPS_PROXY / HTTP_PROXY / NO_PROXY)",
              "dns": "falha de DNS: o nome api.anthropic.com não foi resolvido (sem internet ou DNS bloqueado)",
              "timeout": "tempo de conexão esgotado (rede lenta, firewall ou proxy bloqueando)",
              "refused": "conexão recusada (firewall ou proxy local desligado)",
              "reset": "conexão interrompida pelo servidor ou pela rede"},
    "en": {"ssl": "SSL certificate failure (corporate proxy/antivirus intercepting HTTPS? point SSL_CERT_FILE or "
                  "REQUESTS_CA_BUNDLE at the network's CA certificate)",
           "proxy": "HTTP(S) proxy failure (check HTTPS_PROXY / HTTP_PROXY / NO_PROXY)",
           "dns": "DNS failure: api.anthropic.com could not be resolved (offline or DNS blocked)",
           "timeout": "connection timed out (slow network, firewall or proxy blocking)",
           "refused": "connection refused (firewall, or a local proxy that is not running)",
           "reset": "connection reset by the server or the network"},
}

KIND_TEXT = {
    "pt-BR": {"billing": "A API da Anthropic recusou a requisição: crédito ou limite de gastos esgotado. "
                         "Verifique o saldo e os limites do workspace no Console da Anthropic.",
              "auth": "A API da Anthropic recusou a chave (inválida, revogada ou sem permissão).",
              "rate_limit": "A API da Anthropic está sobrecarregada ou limitando requisições, mesmo após novas "
                            "tentativas. Tente de novo em alguns minutos.",
              "connection": "Não foi possível conectar à API da Anthropic.",
              "refused": "A API da Anthropic recusou a requisição.",
              "internal": "Erro interno do simulador."},
    "en": {"billing": "The Anthropic API refused the request: credit or spend limit exhausted. Check the balance "
                      "and workspace limits in the Anthropic Console.",
           "auth": "The Anthropic API rejected the key (invalid, revoked or without permission).",
           "rate_limit": "The Anthropic API is overloaded or rate-limiting, even after retries. Try again in a few "
                         "minutes.",
           "connection": "Could not connect to the Anthropic API.",
           "refused": "The Anthropic API refused the request.",
           "internal": "Internal simulator error."},
}


def friendly_message(e: BaseException, language: str = "pt-BR") -> tuple[str, str]:
    """(kind, readable message) for an exception raised while a session was generating turns."""
    from .llm.client import GracefulStop
    lg = "pt-BR" if language == "pt-BR" else "en"
    kind = getattr(e, "kind", None) if isinstance(e, GracefulStop) else None
    if isinstance(e, GracefulStop):
        kind = kind or "refused"
        detail = getattr(e, "detail", None) or describe_error(e)
    else:
        kind = "connection" if ("Connection" in type(e).__name__ or "Timeout" in type(e).__name__) else "internal"
        detail = describe_error(e)
    msg = KIND_TEXT[lg].get(kind, KIND_TEXT[lg]["internal"])
    hint = getattr(e, "hint", None) or (diagnose(e) if kind == "connection" else None)
    if hint and hint in HINT_TEXT[lg]:
        msg += (" Provável causa: " if lg == "pt-BR" else " Likely cause: ") + HINT_TEXT[lg][hint] + "."
    msg += (" Detalhe técnico: " if lg == "pt-BR" else " Technical detail: ") + detail
    return kind, _mask(msg)[:900]
