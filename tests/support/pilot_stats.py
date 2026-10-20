"""Compute the structural calibration targets from the private pilot corpus (a TSV with columns
Group, Username, Timestamp, Message). The corpus is NOT distributed and is never used to train
models: point the env var COLLABSIM_PILOT_TSV at a local copy to recompute the targets; without it
only the aggregate data/pilot_targets.json is available (load_targets()) and the tests that need
the corpus skip.

Usage:
    COLLABSIM_PILOT_TSV=/path/to/corpus.tsv python -m tests.support.pilot_stats [--out PATH] [--leave-out g1,g2]

Writes the pooled stats and leave-2-groups-out ranges (min/max over all C(n,2)
subsets) to data/pilot_targets.json. These are *calibration* targets (structural only), never
validation.

Non-student lines: a username that writes in at least half of the groups is not a student (students
belong to one group); its lines are excluded from every statistic. Set
COLLABSIM_PILOT_EXCLUDE to a comma-separated list of usernames to override the rule.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
from datetime import datetime
from pathlib import Path

from .structural import FILLER_KEYS, RUN_KEYS, STAT_KEYS, TAIL_KEYS, aggregate, answered_question_rate, run_stats

DATA = Path(__file__).resolve().parent / "data"
ENV_VAR = "COLLABSIM_PILOT_TSV"
EXCLUDE_VAR = "COLLABSIM_PILOT_EXCLUDE"
DEFAULT_OUT = DATA / "pilot_targets.json"
SOURCE_LABEL = "aggregate realism targets from a private pilot corpus (not distributed)"


def pilot_tsv() -> Path | None:
    """Path of the private pilot corpus from $COLLABSIM_PILOT_TSV, or None when unset / missing."""
    p = os.environ.get(ENV_VAR)
    return Path(p) if p and Path(p).is_file() else None


def require_pilot_tsv() -> Path:
    """The corpus path, or pytest.skip when it is not available (tests only)."""
    p = pilot_tsv()
    if p is None:
        import pytest
        pytest.skip("pilot corpus not available (set COLLABSIM_PILOT_TSV)")
    return p


def _resolve(tsv: Path | None) -> Path:
    tsv = tsv or pilot_tsv()
    if tsv is None:
        raise FileNotFoundError(f"pilot corpus not available: set {ENV_VAR} to a local copy of the TSV")
    return Path(tsv)


def load_targets() -> dict:
    """The aggregate calibration targets (distributed; no corpus needed)."""
    return json.loads(DEFAULT_OUT.read_text())


def _rows(tsv: Path) -> list[dict]:
    with open(tsv, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def non_students(rows: list[dict]) -> set[str]:
    env = os.environ.get(EXCLUDE_VAR)
    if env is not None:
        return {u.strip() for u in env.split(",") if u.strip()}
    groups: dict[str, set[str]] = {}
    for r in rows:
        groups.setdefault(r["Username"], set()).add(r["Group"])
    n = len({r["Group"] for r in rows})
    return {u for u, gs in groups.items() if 2 * len(gs) >= n and len(gs) > 1}


def load_pilot(tsv: Path | None = None, min_msgs: int = 5) -> dict[str, list[dict]]:
    """Groups keyed by anonymous ids g1..gN (real group ids sorted, trivial groups dropped)."""
    rows = _rows(_resolve(tsv))
    excluded = non_students(rows)
    groups: dict[str, list[dict]] = {}
    for row in rows:
        ts = datetime.strptime(row["Timestamp"].strip(), "%Y-%m-%d %H:%M:%S")
        groups.setdefault(row["Group"], []).append(
            {"speaker": row["Username"], "ts": ts, "text": row["Message"],
             "is_external": row["Username"] in excluded})
    out = {}
    kept = sorted(g for g, msgs in groups.items() if len(msgs) >= min_msgs)  # trivial group dropped
    for i, g in enumerate(kept, 1):
        msgs = groups[g]
        msgs.sort(key=lambda m: m["ts"])
        t0 = msgs[0]["ts"]
        out[f"g{i}"] = [{"speaker": m["speaker"], "t": (m["ts"] - t0).total_seconds(),
                         "text": m["text"], "is_external": m["is_external"]} for m in msgs]
    return out


def compute_targets(tsv: Path | None = None, leave_out: list[str] | None = None) -> dict:
    groups = load_pilot(tsv)
    keep = {g: s for g, s in groups.items() if g not in set(leave_out or [])}
    pooled = aggregate(list(keep.values()))
    names = sorted(keep, key=lambda g: int(g[1:]))
    l2o: dict[str, dict] = {k: {"min": float("inf"), "max": float("-inf")} for k in STAT_KEYS + FILLER_KEYS + TAIL_KEYS}
    if len(names) > 3:
        for drop in itertools.combinations(names, 2):
            sub = aggregate([keep[g] for g in names if g not in drop])
            for k in STAT_KEYS + FILLER_KEYS + TAIL_KEYS:
                l2o[k]["min"] = min(l2o[k]["min"], sub[k])
                l2o[k]["max"] = max(l2o[k]["max"], sub[k])
        aq = [answered_question_rate([keep[g] for g in names if g not in drop])["rate"]
              for drop in itertools.combinations(names, 2)]
        l2o["answered_question_rate"] = {"min": min(aq), "max": max(aq)}
        for k in RUN_KEYS:
            vals = [run_stats([keep[g] for g in names if g not in drop])[k]
                    for drop in itertools.combinations(names, 2)]
            l2o[k] = {"min": min(vals), "max": max(vals)}
    aqr = answered_question_rate(list(keep.values()))
    pooled["answered_question_rate"] = aqr["rate"]
    pooled["answered_question_n"] = aqr["n_questions"]
    pooled.update(run_stats(list(keep.values())))
    for k in l2o:
        l2o[k]["width"] = l2o[k]["max"] - l2o[k]["min"]
    return {"source": SOURCE_LABEL, "groups": names, "left_out": leave_out or [],
            "pooled": pooled, "leave_2_out_range": l2o,
            "note": "structural calibration targets only (pt-BR pilot data, aggregates); not validation."}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tsv", type=Path, default=None, help=f"defaults to ${ENV_VAR}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--leave-out", default="", help="comma-separated group ids (g1..g7) to hold out")
    a = ap.parse_args(argv)
    lo = [g for g in a.leave_out.split(",") if g]
    res = compute_targets(a.tsv, lo)
    a.out.write_text(json.dumps(res, indent=2))
    for k in STAT_KEYS + FILLER_KEYS + TAIL_KEYS + ["answered_question_rate"] + RUN_KEYS:
        r = res["leave_2_out_range"][k]
        print(f"{k:26s} {res['pooled'][k]:8.3f}   L2O [{r['min']:.3f}, {r['max']:.3f}]")


if __name__ == "__main__":
    main()
