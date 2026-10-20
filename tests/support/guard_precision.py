"""Guard-precision harness: replays leak-guard flags offline against the current
code rules, without any LLM call.

Fixture: JSONL, one flag per line: task, speaker, role, known_facts, kcs_p, act, draft, kind
("llm_flag": a Haiku check flag, validated by `validate_llm_flag`; "code": the code keyphrase
check), flag_id, quote (llm flags; where the original review did not log quotes the full draft is
used and `quote_assumed` is set), visible (others' messages in the speaker's window) and expect
("reject" = false positive the guard must discard, "keep" = true leak it must still block).

    python -m tests.support.guard_precision tests/fixtures/d1_v4_guard_precision.jsonl
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from collabsim.models import ContentPlan, Message, load_task
from collabsim.sim.content import true_leaks, validate_llm_flag


def _state(rec: dict):
    from collabsim.sim import Simulation
    sim = Simulation(rec["task"], 0, n_students=4)
    st = sim.state
    s = st.student(rec["speaker"])
    s.role, s.known_facts, s.jigsaw_facts, s.shared_facts = rec["role"], list(rec["known_facts"]), list(rec["known_facts"]), []
    for k, p in rec.get("kcs_p", {}).items():
        s.kcs[k].p = p
    st.messages = []
    for i, m in enumerate(rec.get("visible", [])):
        st.messages.append(Message(id=i, t=5.0 * (i + 1), speaker=m["speaker"], kind="student", text=m["text"]))
    return sim.task, st, s


def replay(rec: dict) -> dict:
    task, st, s = _state(rec)
    plan = ContentPlan(act=rec.get("act", "on_task_claim"))
    if rec["kind"] == "code":
        leak = rec["flag_id"] in true_leaks(task, s, plan, rec["draft"], st)
        keep, reason = leak, "true_leak" if leak else "not_a_true_leak"
    else:
        keep, reason = validate_llm_flag(task, s, plan, rec["draft"], rec["flag_id"], rec.get("quote"), st)
    return {"case": rec["case"], "kind": rec["kind"], "expect": rec["expect"], "decision": "keep" if keep else "reject",
            "reason": reason, "ok": (keep and rec["expect"] == "keep") or (not keep and rec["expect"] == "reject")}


def evaluate(path: Path) -> dict:
    recs = [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]
    rows = [replay(r) for r in recs]
    fp = [r for r in rows if r["expect"] == "reject"]
    tp = [r for r in rows if r["expect"] == "keep"]
    return {"rows": rows, "n": len(rows), "all_ok": all(r["ok"] for r in rows),
            "false_positives_rejected": sum(r["decision"] == "reject" for r in fp), "false_positives": len(fp),
            "true_leaks_kept": sum(r["decision"] == "keep" for r in tp), "true_leaks": len(tp)}


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    res = evaluate(Path(argv[0] if argv else "tests/fixtures/d1_v4_guard_precision.jsonl"))
    for r in res["rows"]:
        print(f"{'ok ' if r['ok'] else 'BAD'} {r['case']:<36} {r['kind']:<8} expect={r['expect']:<6} "
              f"-> {r['decision']:<6} ({r['reason']})")
    print(f"false positives rejected {res['false_positives_rejected']}/{res['false_positives']}; "
          f"true leaks kept {res['true_leaks_kept']}/{res['true_leaks']}")


if __name__ == "__main__":
    main()
