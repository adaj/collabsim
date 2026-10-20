"""Over 100 template sessions in the configuration a Simulator session runs (no participants), the
structural statistics land within the pilot ranges.

Bounds are LOOSE on purpose (calibration, not validation): each statistic must fall in the pilot
leave-2-groups-out range widened by a tolerance; most must be inside the unwidened range. A
comparison table is printed (run with -s to see it).
"""
import pytest

from tests.support.pilot_stats import compute_targets, load_targets, require_pilot_tsv
from tests.support.structural import FILLER_KEYS, REPORTED_ONLY_STATS, STAT_KEYS, aggregate
from collabsim.models import load_task
from collabsim.sim import Simulation, transcript_for_stats

N_SIMS = 100
# relative widening of the pilot leave-2-out range (absolute floor for shares)
TOL = {"msgs_per_session_median": 0.35, "msgs_per_student_median": 0.35, "msgs_per_student_per_10min_median": 0.35,
       "gini_median": 0.35,
       "words_median": 0.25, "one_word_share": 0.35, "gap_median_s": 0.35, "gap_p90_s": 0.35,
       "p_same_speaker": 0.20, "lurker_share": 0.6}
TASKS = ("sci-enzyme-01", "math-ratio-01")


def session_sims(language: str = "pt-BR", n: int = N_SIMS) -> list[Simulation]:
    out = []
    for seed in range(n):
        task = load_task(TASKS[seed % 2], language=language)
        sim = Simulation(task, seed, n_students=(3, 4)[(seed // 2) % 2])
        sim.run()
        out.append(sim)
    return out


@pytest.fixture(scope="module")
def sim_sessions():
    sims = session_sims()
    return [transcript_for_stats(s.state) for s in sims], [s.state.roster for s in sims]


@pytest.fixture(scope="module")
def sim_stats(sim_sessions):
    return aggregate(*sim_sessions)


def test_answered_question_rate_reported(sim_sessions):
    """Peers answer content questions: the rate stays at or above the pilot band (holders of the
    asked fact are boosted), and below answering almost everything."""
    from tests.support.structural import answered_question_rate
    tg = load_targets()
    b = tg["leave_2_out_range"]["answered_question_rate"]
    r = answered_question_rate(sim_sessions[0])
    print(f"\nanswered_question_rate  pilot {tg['pooled']['answered_question_rate']:.3f} "
          f"[{b['min']:.3f}, {b['max']:.3f}]  sim {r['rate']:.3f} (n={r['n_questions']})")
    # with fillers capped (0.5.0) almost every line after a question has content, which the lenient
    # heuristic reads as an answer; reported, loosely bounded
    assert b["min"] - 0.05 <= r["rate"] <= 0.97


def test_pilot_targets_file_is_current():
    """Needs the private corpus ($COLLABSIM_PILOT_TSV); skipped otherwise."""
    saved = load_targets()
    fresh = compute_targets(require_pilot_tsv())
    assert saved["pooled"] == pytest.approx(fresh["pooled"])
    for k, r in fresh["leave_2_out_range"].items():
        assert saved["leave_2_out_range"][k] == pytest.approx(r), k
    assert saved["groups"] == fresh["groups"] and saved["source"] == fresh["source"]
    assert len(fresh["groups"]) == 7  # trivial 2-message group dropped


def test_leave_out_option():
    """Needs the private corpus ($COLLABSIM_PILOT_TSV); skipped otherwise."""
    full = compute_targets(require_pilot_tsv())
    lo = compute_targets(leave_out=full["groups"][:2])
    assert len(lo["groups"]) == 5 and lo["left_out"] == full["groups"][:2]


def test_targets_file_holds_only_aggregates():
    """Pooled statistics and leave-2-groups-out ranges; nothing per group."""
    from tests.support.structural import RUN_KEYS, TAIL_KEYS
    tg = load_targets()
    assert set(tg) == {"source", "groups", "left_out", "pooled", "leave_2_out_range", "note"}
    assert set(tg["leave_2_out_range"]) == set(STAT_KEYS + FILLER_KEYS + TAIL_KEYS + RUN_KEYS) | {"answered_question_rate"}
    assert set(tg["pooled"]) >= set(tg["leave_2_out_range"])
    assert all(set(r) == {"min", "max", "width"} for r in tg["leave_2_out_range"].values())
    assert "private pilot corpus" in tg["source"]


def test_existing_stats_inside_strict_bands(sim_stats):
    """Every structural stat (session length excepted: reported only) is inside its unwidened band."""
    tg = load_targets()
    out = [k for k in STAT_KEYS if k not in REPORTED_ONLY_STATS
           and not tg["leave_2_out_range"][k]["min"] <= sim_stats[k] <= tg["leave_2_out_range"][k]["max"]]
    assert not out, {k: sim_stats[k] for k in out}


def test_filler_stats_within_pilot_upper_bounds(sim_stats):
    """"Don't know" lines and backchannels no more frequent than in the pilot (upper end of its
    leave-2-out range, + a small margin), and never more than 2 content-free messages in a row."""
    tg = load_targets()
    l2o = tg["leave_2_out_range"]
    print(f"\nidk share {sim_stats['idk_share']:.3f} (pilot {tg['pooled']['idk_share']:.3f}, max "
          f"{l2o['idk_share']['max']:.3f}); backchannel share {sim_stats['backchannel_share']:.3f} "
          f"(pilot {tg['pooled']['backchannel_share']:.3f}, max {l2o['backchannel_share']['max']:.3f}); "
          f"max content-free run {sim_stats['max_content_free_run_max']}")
    assert sim_stats["idk_share"] <= l2o["idk_share"]["max"] + 0.02
    assert sim_stats["backchannel_share"] <= l2o["backchannel_share"]["max"] + 0.03
    assert sim_stats["max_content_free_run_max"] <= tg["pooled"]["max_content_free_run_max"] + 1
    assert set(FILLER_KEYS) <= set(sim_stats)


def test_structural_stats_within_pilot_ranges(sim_stats):
    tg = load_targets()
    rows = []
    fails = []
    for k in STAT_KEYS:
        lo, hi = tg["leave_2_out_range"][k]["min"], tg["leave_2_out_range"][k]["max"]
        tol = TOL[k] * max(abs(lo), abs(hi), 0.1)
        ok = lo - tol <= sim_stats[k] <= hi + tol
        inside = lo <= sim_stats[k] <= hi
        rows.append(f"{k:26s} {tg['pooled'][k]:8.3f}  [{lo:7.3f}, {hi:7.3f}]  {sim_stats[k]:8.3f}  "
                    f"{'in' if inside else ('ok' if ok else 'FAIL')}")
        if not ok and k not in REPORTED_ONLY_STATS:
            fails.append(k)
    print("\nstat                       pilot pooled  pilot leave-2-out     sim(100)")
    print("\n".join(rows))
    assert not fails, f"outside loose pilot bounds: {fails}"


def test_session_length_and_silence_tail(sim_stats):
    """0.5.1: the default pace gives pilot-like session lengths (reported against the pilot's leave-2-out
    range, checked loosely: 25-40 min), the silence tail's 95th percentile is inside the pilot band, and
    no silence is longer than the cap (so the pilot's p99 / max are deliberately not reproduced)."""
    from collabsim.sim.config import SimConfig
    tg = load_targets()
    l2o = tg["leave_2_out_range"]
    print(f"\nsession minutes {sim_stats['session_minutes_median']:.1f} (pilot {tg['pooled']['session_minutes_median']:.1f} "
          f"[{l2o['session_minutes_median']['min']:.1f}, {l2o['session_minutes_median']['max']:.1f}]); gap p95 "
          f"{sim_stats['gap_p95_s']:.0f}s (pilot [{l2o['gap_p95_s']['min']:.0f}, {l2o['gap_p95_s']['max']:.0f}]), p99 "
          f"{sim_stats['gap_p99_s']:.0f}s, max {sim_stats['gap_max_s']:.0f}s")
    assert 25.0 <= sim_stats["session_minutes_median"] <= 40.0
    assert l2o["gap_p95_s"]["min"] <= sim_stats["gap_p95_s"] <= l2o["gap_p95_s"]["max"]
    assert sim_stats["gap_max_s"] <= SimConfig().max_silence_s + 1e-6
