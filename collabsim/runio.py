"""Small I/O helpers: API-key masking, atomic JSON writes and a KEY=VALUE env-file loader.

`mask` hides anything resembling an Anthropic key (sk-ant-...) in logs, error messages and written
files. `load_env_file` loads only the known keys from a .env file; values are never printed.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .llm.client import atomic_write

KEY_RE = re.compile(r"sk-ant-[A-Za-z0-9_\-]{4,}")
ENV_KEYS = ("ANTHROPIC_API_KEY", "COLLABSIM_STUDENT_MODEL", "COLLABSIM_JUDGE_MODEL")


def mask(text: str) -> str:
    return KEY_RE.sub("sk-ant-***", str(text))


def atomic_write_json(path: Path | str, obj, indent: int | None = 1) -> None:
    atomic_write(Path(path), mask(json.dumps(obj, indent=indent, ensure_ascii=False, default=str)))


def parse_env_file(path: Path | str) -> dict[str, str]:
    """Minimal KEY=VALUE parser: blank lines and # comments skipped, optional `export `,
    optional matching single/double quotes."""
    out: dict[str, str] = {}
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
            v = v[1:-1]
        elif " #" in v:
            v = v.split(" #", 1)[0].rstrip()
        if k:
            out[k] = v
    return out


def load_env_file(path: Path | str | None, required: bool = True) -> list[str]:
    """Load the known keys from an env file into os.environ (file wins over the shell).
    Returns the NAMES loaded (never the values). A missing file raises FileNotFoundError when
    `required`, else is ignored."""
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        if required:
            raise FileNotFoundError(f"env file {p} not found")
        return []
    vals = parse_env_file(p)
    loaded = []
    for k in ENV_KEYS:
        if vals.get(k):
            os.environ[k] = vals[k]
            loaded.append(k)
    return loaded
