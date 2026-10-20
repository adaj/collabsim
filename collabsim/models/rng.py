"""Seeded, serialisable RNG streams.

Each purpose (group generation, scheduler, realizer, updater, latency) gets its own
stream derived from the session seed, so that a draw in one stream (e.g. the wording) never
perturbs another (e.g. the scheduler).
State is JSON-serialisable for snapshot()/restore().
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Sequence, TypeVar

T = TypeVar("T")

STREAMS = ("group", "sched", "realize", "update", "latency")


def derive_seed(seed: int, name: str) -> int:
    h = hashlib.sha256(f"{seed}:{name}".encode()).digest()
    return int.from_bytes(h[:8], "big")


class RNG:
    """Thin wrapper over random.Random with the distributions the spec needs."""

    def __init__(self, seed: int | None = None):
        self._r = random.Random(seed)

    # -- state -------------------------------------------------------------
    def get_state(self) -> list[Any]:
        v, internal, gauss = self._r.getstate()
        return [v, list(internal), gauss]

    def set_state(self, state: list[Any]) -> None:
        v, internal, gauss = state
        self._r.setstate((v, tuple(internal), gauss))

    # -- distributions -----------------------------------------------------
    def random(self) -> float:
        return self._r.random()

    def uniform(self, a: float, b: float) -> float:
        return self._r.uniform(a, b)

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def gauss(self, mu: float, sigma: float) -> float:
        return self._r.gauss(mu, sigma)

    def lognormal(self, mu: float, sigma: float) -> float:
        return math.exp(self._r.gauss(mu, sigma))

    def beta(self, a: float, b: float) -> float:
        return self._r.betavariate(a, b)

    def bernoulli(self, p: float) -> bool:
        return self._r.random() < p

    def choice(self, seq: Sequence[T]) -> T:
        return seq[self._r.randrange(len(seq))]

    def weighted(self, items: Sequence[T], weights: Sequence[float]) -> T:
        tot = float(sum(weights))
        if tot <= 0:
            return self.choice(items)
        u = self._r.random() * tot
        acc = 0.0
        for it, w in zip(items, weights):
            acc += w
            if u < acc:
                return it
        return items[-1]

    def sample(self, seq: Sequence[T], k: int) -> list[T]:
        return self._r.sample(list(seq), k)

    def shuffle(self, seq: list[T]) -> None:
        self._r.shuffle(seq)


class RNGBank:
    """Named independent streams derived from one seed."""

    def __init__(self, seed: int):
        self.seed = seed
        self.streams = {n: RNG(derive_seed(seed, n)) for n in STREAMS}

    def __getitem__(self, name: str) -> RNG:
        return self.streams[name]

    def get_state(self) -> dict[str, list[Any]]:
        return {n: r.get_state() for n, r in self.streams.items()}

    def set_state(self, state: dict[str, list[Any]]) -> None:
        for n, s in state.items():
            self.streams[n].set_state(s)
