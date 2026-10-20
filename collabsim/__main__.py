"""Command line: a quick whole-chat run, a small batch, and an API connection check.

    python -m collabsim try   --env-file .env --n 4 --language pt-BR [--task sci-enzyme-01] [--seed 1]
                              [--backend llm|mock|template] [--out trysim] [--max-minutes M]
                              [--pace pilot|brisk] [--conflict low|normal|high]
    python -m collabsim batch --env-file .env --seeds 1,2,3,4 --n 2,4,6 --language pt-BR
                              [--task sci-enzyme-01] [--pace pilot|brisk] [--conflict ...]
                              [--backend llm|mock|template] [--out trysim] [--max-sessions 12] [--yes]
    python -m collabsim check --env-file .env

`try` writes <out>/<stamp>.md (readable transcript) and <stamp>.json (full research record) and
prints usage/cost. `batch` runs every (n, seed) pair sequentially into <out>/<stamp>/<n>-<seed>.md
/.json plus summary.md (per-session stats and totals); it prints a cost estimate first and asks
for confirmation (or --yes). The API key is read from --env-file (or ANTHROPIC_API_KEY) and never
printed. Exit codes: 0 ok, 1 session/connection error, 2 bad arguments / no key, 3 not confirmed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def _load_env(path: str | None) -> None:
    from .runio import load_env_file
    if path:
        load_env_file(path, required=path != ".env")


def _simulator(backend: str):
    from . import BackendUnavailable, Simulator
    try:
        return Simulator(backend=backend)
    except BackendUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(2) from None


def cmd_try(a) -> int:
    _load_env(a.env_file)
    from .runio import atomic_write_json, mask
    sim = _simulator(a.backend)
    s = sim.new_session(n_students=a.n, task=a.task, language=a.language, seed=a.seed, pacing="manual",
                        max_minutes=a.max_minutes, pace=a.pace, conflict=a.conflict)
    t0 = time.time()
    print(f"running {a.task} [{a.language}] n={a.n} seed={s.seed} backend={a.backend} ...", flush=True)
    events = s.run_to_end()
    rec = s.record()
    u = s.usage()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    atomic_write_json(out / f"{stamp}.json", rec)
    cfg = {"n_students": a.n, "task": a.task, "language": a.language, "seed": s.seed, "backend": a.backend,
           "pace": a.pace, "conflict": a.conflict}
    (out / f"{stamp}.md").write_text(mask(transcript_md(f"collabsim try {stamp}", cfg, s, events)), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f}s: {len(events)} events, status={s.status}, ended={s.ended_reason}, "
          f"calls={u['calls']}, usd={u['usd']:.4f}")
    if s.error:
        print(f"error: {s.error}", file=sys.stderr)
    print(f"transcript: {out / (stamp + '.md')}")
    s.close()
    return 1 if s.status == "error" else 0


def transcript_md(title: str, cfg: dict, s, events) -> str:
    u = s.usage()
    lines = [f"# {title}", "", f"config: `{json.dumps(cfg, ensure_ascii=False)}`",
             f"status: {s.status} | ended: {s.ended_reason} | events: {len(events)} | "
             f"calls: {u['calls']} | usd: {u['usd']:.4f}",
             f"roster: {', '.join(r['name'] for r in s.roster)}", ""]
    if s.error:
        lines += [f"**error:** {s.error}", ""]
    lines += ["| min | who | message |", "|---|---|---|"]
    for e in events:
        lines.append(f"| {e.t_sim / 60:.1f} | {e.display_name} | {e.text.replace('|', '/')} |")
    return "\n".join(lines) + "\n"


def _ints(text: str, name: str) -> list[int]:
    try:
        out = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise SystemExit(f"error: --{name} must be a comma-separated list of integers, got {text!r}") from None
    if not out:
        raise SystemExit(f"error: --{name} is empty")
    return out


# Per Haiku call, from the real-Haiku runs (0.5.0: 52 calls, US$0.0068 for a 30-message chat): the
# estimate counts the turns that need Haiku in a free template dry run of the same sessions.
USD_PER_CALL = 0.00013
CALLS_PER_LLM_TURN = 2.7          # realize + check, plus regenerations (mock pipeline: 2.74 calls per Haiku turn)
ESTIMATE_MARGIN = 1.3
# 0.5.2: the Haiku coherence check on challenge / disagreement / belief lines (one call, ~1.2 with the
# occasional regeneration + re-check)
COHERENCE_CALLS_PER_TURN = 1.2


def estimate(a, pairs) -> tuple[float, list[dict]]:
    """Dry run (template backend, same seeds / sizes / pace): turns that would call Haiku -> US$."""
    from .llm.realizer import is_templated, needs_coherence
    from .models.task import load_task
    from .sim.config import SimConfig
    from .sim.engine import Simulation
    task = load_task(a.task, language=a.language)
    rows, total = [], 0.0
    for n, seed in pairs:
        kw = {"session_minutes": a.max_minutes} if a.max_minutes else {}
        sim = Simulation(task, seed, cfg=SimConfig(pace=a.pace, conflict=a.conflict, **kw), n_students=n)
        sim.run()
        stu = [m for m in sim.state.messages if m.kind == "student"]
        llm = sum(1 for m in stu if not is_templated(m.plan))
        coh = sum(1 for m in stu if not is_templated(m.plan) and needs_coherence(m.plan))
        usd = (llm * CALLS_PER_LLM_TURN + coh * COHERENCE_CALLS_PER_TURN) * USD_PER_CALL * ESTIMATE_MARGIN
        rows.append({"n": n, "seed": seed, "messages": len(stu), "llm_turns": llm, "coherence_turns": coh, "usd": usd})
        total += usd
    return total, rows


def session_stats(rec: dict, usage: dict) -> dict:
    """Per-session stats for the batch summary (computed from the research record)."""
    from .sim.conflict import voiced_and_discussed
    from .models.session import SessionState
    tr = [m for m in rec["transcript"] if m["kind"] == "student"]
    allm = rec["transcript"]
    n_st = len(rec["roster"])
    gaps = [b["t"] - a["t"] for a, b in zip(allm, allm[1:])]
    plans = [m.get("plan") or {} for m in tr]
    idk = sum(1 for p in plans if p.get("act") in ("idk", "refusal"))
    off = sum(1 for p in plans if p.get("act") == "off_task")
    voiced: dict = {}
    for m in tr:
        p = m.get("plan") or {}
        if p.get("claim") == "misconception" and p.get("misconception_id"):
            voiced.setdefault((m["speaker"], p["misconception_id"]), m["id"])
    disc = sum(1 for (sid, mid), first in voiced.items()
               if any(x["speaker"] != sid and (x.get("plan") or {}).get("addresses_misconception") == mid
                      for x in allm if x["id"] > first))
    k = max(1, len(tr))
    return {"n": n_st, "messages": len(tr), "minutes": round(rec["duration_s"] / 60.0, 1),
            "msgs_per_student": round(len(tr) / max(1, n_st), 1), "max_gap_s": round(max(gaps or [0.0])),
            "voiced": len(voiced), "discussed": disc, "idk_share": round(idk / k, 3), "offtask_share": round(off / k, 3),
            "ended": rec["ended_reason"], "calls": int(usage.get("calls", 0)), "usd": round(float(usage.get("usd", 0.0)), 4)}


def cmd_batch(a) -> int:
    _load_env(a.env_file)
    from .runio import atomic_write_json, mask
    seeds, sizes = _ints(a.seeds, "seeds"), _ints(a.n, "n")
    pairs = [(n, seed) for n in sizes for seed in seeds]
    if len(pairs) > a.max_sessions:
        print(f"error: {len(pairs)} sessions requested, more than --max-sessions {a.max_sessions} "
              f"(raise it explicitly to run more)", file=sys.stderr)
        return 2
    est, rows = estimate(a, pairs)
    print(f"batch: {len(pairs)} sessions ({a.task} [{a.language}], n={sizes}, seeds={seeds}, pace={a.pace}, "
          f"conflict={a.conflict}, backend={a.backend}), run one after another")
    if a.backend == "llm":
        print(f"estimated cost: ~US${est:.2f} (dry run: {sum(r['llm_turns'] for r in rows)} Haiku turns, "
              f"{sum(r['messages'] for r in rows)} student messages; margin x{ESTIMATE_MARGIN})")
    else:
        print(f"backend {a.backend}: no API cost")
    if not a.yes:
        try:
            ok = input("proceed? [y/N] ").strip().lower() in ("y", "yes", "s", "sim")
        except EOFError:
            ok = False
        if not ok:
            print("not confirmed (use --yes to skip the question)")
            return 3
    sim = _simulator(a.backend)
    out = Path(a.out) / time.strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    stats, failed = [], 0
    for i, (n, seed) in enumerate(pairs, 1):
        t0 = time.time()
        s = sim.new_session(n_students=n, task=a.task, language=a.language, seed=seed, pacing="manual",
                            max_minutes=a.max_minutes, pace=a.pace, conflict=a.conflict)
        events = s.run_to_end()
        rec, u = s.record(), s.usage()
        cfg = {"n_students": n, "task": a.task, "language": a.language, "seed": seed, "backend": a.backend,
               "pace": a.pace, "conflict": a.conflict}
        atomic_write_json(out / f"{n}-{seed}.json", rec)
        (out / f"{n}-{seed}.md").write_text(mask(transcript_md(f"collabsim batch {n}-{seed}", cfg, s, events)),
                                            encoding="utf-8")
        st = session_stats(rec, u)
        st["seed"], st["status"] = seed, s.status
        stats.append(st)
        failed += s.status == "error"
        print(f"[{i}/{len(pairs)}] n={n} seed={seed}: {st['messages']} msgs, {st['minutes']} min, "
              f"ended={st['ended']}, calls={st['calls']}, usd={st['usd']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if s.error:
            print(f"  error: {s.error}", file=sys.stderr)
        s.close()
    cols = ["n", "seed", "messages", "minutes", "msgs_per_student", "max_gap_s", "voiced", "discussed",
            "idk_share", "offtask_share", "ended", "calls", "usd"]
    lines = [f"# collabsim batch {out.name}", "",
             f"task {a.task} [{a.language}], pace {a.pace}, conflict {a.conflict}, backend {a.backend}; "
             f"estimate before the run: US${est:.2f}", "",
             "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for st in stats:
        lines.append("| " + " | ".join(str(st[c]) for c in cols) + " |")
    tot_msgs = sum(x["messages"] for x in stats)
    lines += ["", f"**totals**: {len(stats)} sessions, {tot_msgs} student messages, "
              f"{sum(x['calls'] for x in stats)} calls, US${sum(x['usd'] for x in stats):.4f}; "
              f"ended {dict((e, sum(1 for x in stats if x['ended'] == e)) for e in sorted({x['ended'] for x in stats}))}; "
              f"max gap {max(x['max_gap_s'] for x in stats)} s; "
              f"misconceptions voiced {sum(x['voiced'] for x in stats)}, discussed {sum(x['discussed'] for x in stats)}"]
    (out / "summary.md").write_text(mask("\n".join(lines) + "\n"), encoding="utf-8")
    print(f"summary: {out / 'summary.md'} | total usd={sum(x['usd'] for x in stats):.4f}")
    return 1 if failed else 0


def cmd_check(a) -> int:
    _load_env(a.env_file)
    sim = _simulator(a.backend)
    r = sim.check_connection()
    if r["ok"]:
        lat = f" latency={r['latency_s']}s" if r["latency_s"] is not None else ""
        print(f"ok: backend={r['backend']} model={r['model']}{lat}")
        return 0
    print(f"FAILED ({r['error_kind']}, hint={r['hint']}): {r['error']}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m collabsim", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("try", help="run one whole chat and write a transcript")
    t.add_argument("--env-file", default=".env")
    t.add_argument("--backend", default="llm", choices=["llm", "mock", "template"])
    t.add_argument("--n", type=int, default=4)
    t.add_argument("--language", default="pt-BR")
    t.add_argument("--task", default="sci-enzyme-01")
    t.add_argument("--seed", type=int, default=1)
    t.add_argument("--max-minutes", type=float, default=None, help="time cap (default: the pace's)")
    t.add_argument("--pace", default="pilot", choices=["pilot", "brisk"])
    t.add_argument("--conflict", default="normal", choices=["low", "normal", "high"])
    t.add_argument("--out", default="trysim")
    b = sub.add_parser("batch", help="run a small batch of chats (sequentially) with a summary")
    b.add_argument("--env-file", default=".env")
    b.add_argument("--backend", default="llm", choices=["llm", "mock", "template"])
    b.add_argument("--seeds", default="1,2,3,4")
    b.add_argument("--n", default="2,4,6", help="comma-separated group sizes")
    b.add_argument("--language", default="pt-BR")
    b.add_argument("--task", default="sci-enzyme-01")
    b.add_argument("--pace", default="pilot", choices=["pilot", "brisk"])
    b.add_argument("--conflict", default="normal", choices=["low", "normal", "high"])
    b.add_argument("--max-minutes", type=float, default=None, help="time cap (default: the pace's)")
    b.add_argument("--max-sessions", type=int, default=12, help="refuse larger batches unless raised")
    b.add_argument("--out", default="trysim")
    b.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    c = sub.add_parser("check", help="verify the API key / network with one cheap request")
    c.add_argument("--env-file", default=".env")
    c.add_argument("--backend", default="llm", choices=["llm", "mock", "template"])
    a = ap.parse_args(argv)
    try:
        return {"try": cmd_try, "batch": cmd_batch, "check": cmd_check}[a.cmd](a)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
