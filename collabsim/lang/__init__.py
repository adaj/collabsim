"""Language packs.

Everything the simulator WRITES or PARSES in a natural language lives in a pack: template banks,
off-task topics, names, interface / procedural intents, question detection for the leak guard,
attribution phrases ("like i said" / "como eu disse"), realizer prompt text and few-shots.
Code decides content in a language-neutral way (ids, plans); a pack only words or parses it.

    from collabsim import lang
    L = lang.of(task)          # the pack of task.language
    L.FILLER["greeting"]

Packs: "en" and "pt-BR" (the native language of the pilot study the structural targets come from;
few-shots are invented lines in the students' register). The default for a Simulator session is
pt-BR; Task.language defaults to "en".
"""
from __future__ import annotations

from importlib import import_module

SUPPORTED: tuple[str, ...] = ("pt-BR", "en")
SERVICE_DEFAULT = "pt-BR"
_ALIASES = {"pt": "pt-BR", "pt-br": "pt-BR", "pt_br": "pt-BR", "ptbr": "pt-BR", "pt-pt": "pt-BR",
            "en": "en", "en-us": "en", "en-gb": "en", "en_us": "en", "english": "en", "portugues": "pt-BR"}
_MODULES = {"en": "collabsim.lang.en", "pt-BR": "collabsim.lang.pt_br"}
_CACHE: dict = {}


class UnsupportedLanguage(ValueError):
    pass


def normalize(code: str | None, default: str = "en") -> str:
    if code is None or not str(code).strip():
        return default
    c = str(code).strip()
    if c in SUPPORTED:
        return c
    n = _ALIASES.get(c.lower())
    if n is None:
        raise UnsupportedLanguage(f"unsupported language {code!r} (supported: {list(SUPPORTED)})")
    return n


def pack(code: str | None):
    c = normalize(code)
    if c not in _CACHE:
        _CACHE[c] = import_module(_MODULES[c])
    return _CACHE[c]


def of(task) -> object:
    """The pack of a Task (or of anything with a `language` attribute); English if none."""
    return pack(getattr(task, "language", None) or "en")


def attr(obj, name: str, default):
    """`name` from the pack of `obj` (a Task, a language code, or None = English), or `default`.
    The English pack defines (almost) nothing: English wording stays in the modules that use it and
    is passed here as the default."""
    p = pack(obj) if (obj is None or isinstance(obj, str)) else of(obj)
    return getattr(p, name, default)


def code_of(obj) -> str:
    return normalize(obj) if (obj is None or isinstance(obj, str)) else normalize(getattr(obj, "language", None))
