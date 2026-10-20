"""Build a simulation with a chosen realizer backend.

realizer: "template" (no LLM) | "mock" (LLM pipeline with deterministic MockLLM) | "llm" (Anthropic API).
"""
from __future__ import annotations

from pathlib import Path


def build_session(task, seed: int, realizer: str = "template", client=None, cache_dir: Path | str | None = None,
                  n_students: int | None = None, cfg=None, params=None):
    from .engine import Simulation
    if realizer == "template":
        return Simulation(task, seed, cfg=cfg, n_students=n_students, params=params)
    from ..llm import LLMRealizer, make_client
    if client is None:
        client = make_client("mock" if realizer == "mock" else "llm", cache_dir)
    sim = Simulation(task, seed, cfg=cfg, n_students=n_students, realizer=LLMRealizer(client), params=params)
    sim.client = client
    return sim
