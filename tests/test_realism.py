"""Realism rules: no formulaic card announcements, anti-repetition, non-contradictory misconception
sampling, few hedges, peer responsiveness, fact dedupe, no self-agreement, monologue cap, closing.
"""
import re

import pytest

from collabsim.llm import LLMClient, LLMRealizer, MockLLM
from collabsim.llm.mock import default_handlers
from collabsim.llm.prompts import realizer_request
from collabsim.models import ContentPlan, RNG, load_task
from collabsim.sim import Simulation
from collabsim.sim.factory import build_session
from collabsim.sim.realizer import TemplateRealizer, strip_card_announcement

TASKS = ("sci-enzyme-01", "math-ratio-01")
SHARE_RE = re.compile(r"share (from|frm|form) my card", re.I)


@pytest.fixture(scope="module")
def template_sims():
    sims = []
    for seed in range(100):
        sim = Simulation(TASKS[seed % 2], seed)
        sim.run()
        sims.append(sim)
    return sims


# ------------------------------------------------------------------ Fix 1: prompt artifact
def test_fix1_no_share_from_card_in_template_lines(template_sims):
    for sim in template_sims:
        for m in sim.state.messages:
            assert not SHARE_RE.search(m.text), m.text


def test_fix1_no_share_from_card_in_mock_llm_lines():
    for seed in range(6):
        sim = build_session(TASKS[seed % 2], seed, realizer="mock")
        sim.run()
        for m in sim.state.messages:
            assert not SHARE_RE.search(m.text), m.text


@pytest.mark.parametrize("tid", TASKS)
def test_fix1_realizer_prompt_never_contains_phrase(tid):
    task = load_task(tid)
    sim = Simulation(tid, 3)
    st = sim.state
    s = st.students[0]
    for i in range(12):
        for f in task.facts:
            for plan in (ContentPlan(act="on_task_claim", kc=f.kc, claim="fact", fact_ids=[f.id], is_correct=True),
                         ContentPlan(act="disagree", kc=f.kc, claim="fact", fact_ids=[f.id], is_correct=True,
                                     family="evidence_from_material", addressee=st.students[1].id)):
                req = realizer_request("claude-haiku-5-5", task, st, s, plan, sim.names, replicate_idx=i)
                full = req.prefix + "\n" + req.suffix
                assert not re.search(r"share\s+(from|frm|form)\s+(my|your)\s+card", full, re.I)


def test_fix1_template_and_fallback_fact_wording_varied():
    task = load_task("sci-enzyme-01")
    sim = Simulation("sci-enzyme-01", 1)
    s = sim.state.students[0]
    plan = ContentPlan(act="on_task_claim", kc="kc_enzyme", claim="fact", fact_ids=["f_reuse"], is_correct=True)
    tr = TemplateRealizer()
    rng = RNG(5)
    lines = {tr.content(plan, s, task, rng, sim.names) for _ in range(80)}
    assert len(lines) >= 6 and not any(SHARE_RE.search(x) for x in lines)
    llm = LLMRealizer(LLMClient(MockLLM(default_handlers())))
    for _ in range(40):
        assert not SHARE_RE.search(llm._safe_fallback(task, s, plan, rng, sim.names))


def test_fix1_code_safety_net_strips_announcement():
    rng = RNG(0)
    for raw in ("share from my card: enzymes are catalysts", "share form my card: amylase works best at 37",
                "ok share frm my card - it isnt used up", "Share from the card: x"):
        out = strip_card_announcement(raw, rng)
        assert not re.search(r"share\s+(from|frm|form)", out, re.I), out
        assert out


def test_fix1_llm_realizer_cleans_model_output():
    h = default_handlers()
    h["realize"] = lambda req: "share form my card: enzymes are catalysts so not used up"
    client = LLMClient(MockLLM(h))
    sim = build_session("sci-enzyme-01", 2, realizer="mock", client=client)
    sim.run(max_events=60)
    for m in sim.state.messages:
        assert not SHARE_RE.search(m.text), m.text


# ------------------------------------------------------------------ Fix 2: anti-repetition
from collabsim.sim import repetition as rep  # noqa: E402


def test_fix2_never_same_string_twice_in_a_row(template_sims):
    for sim in template_sims:
        last = {}
        for m in sim.state.messages:
            if m.kind != "student":
                continue
            assert last.get(m.speaker) != rep.norm(m.text), (sim.state.session_id, m.id, m.text)
            last[m.speaker] = rep.norm(m.text)


def test_fix2_no_verbatim_off_task_reuse_and_persistence(template_sims):
    off_off = [0, 0]
    for sim in template_sims:
        stu = [m for m in sim.state.messages if m.kind == "student"]
        seen = set()
        for m in stu:
            if m.plan.act == "off_task" and len(rep.norm_tokens(m.text)) >= rep.MIN_TOKENS:
                assert rep.norm(m.text) not in seen, m.text
                seen.add(rep.norm(m.text))
                assert m.plan.topic is not None
        for a, b in zip(stu, stu[1:]):
            if a.plan.act == "off_task":
                off_off[0] += 1
                off_off[1] += b.plan.act == "off_task"
    p = off_off[1] / off_off[0]
    # off-task turns form short threads (<= offtask_thread_max) under a share cap, which keeps
    # P(off|off) below the pilot's .37 (0.4.2 sessions: .24); reported, loosely bounded
    print(f"\nP(off|off) = {p:.3f} (pilot .37)")
    assert 0.15 <= p <= 0.47


def test_fix2_topics_assigned_and_bank_large():
    from collabsim.sim.offtask import TOPICS, TOPICS_PER_STUDENT
    assert len(TOPICS) >= 20 and all(len(v[1]) >= 5 for v in TOPICS.values())
    sim = Simulation("sci-enzyme-01", 4)
    for s in sim.state.students:
        assert len(s.off_task_topics) == TOPICS_PER_STUDENT


def test_fix2_same_content_question_not_reasked(template_sims):
    from collabsim.sim.planner import REASK_WINDOW
    n_follow = 0
    for sim in template_sims:
        stu = [m for m in sim.state.messages if m.kind == "student"]
        for i, m in enumerate(stu):
            p = m.plan
            if p.act != "question" or not p.asked_fact:
                continue
            if p.follow_up:
                n_follow += 1
                assert len(m.text.split()) <= 6, m.text
                continue
            for x in stu[max(0, i - REASK_WINDOW):i]:
                assert not (x.plan.act == "question" and x.plan.asked_fact == p.asked_fact
                            and not x.plan.follow_up), (sim.state.session_id, m.text)
    assert n_follow > 0


def test_fix2_near_duplicates_rare_and_logged(template_sims):
    n = dup = hits = 0
    for sim in template_sims:
        hits += sum(r["type"] == "repetition_guard" for r in sim.state.slippage_log)
        stu = [m for m in sim.state.messages if m.kind == "student"]
        for i, m in enumerate(stu):
            if len(rep.norm_tokens(m.text)) < rep.MIN_TOKENS:
                continue
            n += 1
            win = [x for x in stu[max(0, i - rep.WINDOW):i] if len(rep.norm_tokens(x.text)) >= rep.MIN_TOKENS]
            dup += any(rep.too_similar(m.text, x.text) for x in win)
        res = sim.result()
        assert res["repetition_guard_log"] == [r for r in sim.state.slippage_log if r["type"] == "repetition_guard"]
    print(f"\nresidual near-duplicate share (template realizer) = {dup / n:.3f}; guard hits = {hits}")
    assert hits > 0 and dup / n < 0.10


def test_fix2_guard_regenerates_llm_line_and_logs():
    h = default_handlers()
    base = h["realize"]

    def stuck(req):  # a model that keeps producing the same off-task line unless told to rephrase
        if req.meta["plan"]["act"] == "off_task" and not req.meta.get("rephrase"):
            return "ok but seriously my snack is calling me"
        return base(req)
    h["realize"] = stuck
    sim = build_session("sci-enzyme-01", 1, realizer="mock", client=LLMClient(MockLLM(h)))
    sim.run()
    stu = [m for m in sim.state.messages if m.kind == "student"]
    idx = [i for i, m in enumerate(stu) if "snack is calling me" in m.text]
    assert idx and all(b - a > rep.WINDOW for a, b in zip(idx, idx[1:]))  # never inside the window
    guarded = [r for r in sim.realizer.log if "repetition_guard" in r]
    assert guarded and any(r["repetition_guard"]["regenerated"] for r in guarded)
    assert sim.realizer.summary()["repetition_guard_hits"] == len(guarded)
    assert len(sim.realizer.log) == sim.state.student_turns  # one log record per turn (skipped turns: none)


def test_fix2_realizer_prompt_carries_own_lines_and_off_task_topics():
    from collabsim.models import Message
    task = load_task("sci-enzyme-01")
    sim = Simulation("sci-enzyme-01", 3)
    st = sim.state
    s, o = st.students[0], st.students[1]
    for i, (sid, txt) in enumerate([(s.id, "my snack is right here"), (o.id, "did anyone see the game"),
                                    (s.id, "right guys?")]):
        st.messages.append(Message(id=i, t=i * 10.0, speaker=sid, kind="student", text=txt,
                                   plan=ContentPlan(act="off_task" if i < 2 else "backchannel", topic="snack")))
    req = realizer_request("claude-haiku-5-5", task, st, s, ContentPlan(act="off_task", topic="music"), sim.names)
    own = next(l for l in req.suffix.splitlines() if l.startswith("YOUR OWN LAST MESSAGES"))
    off = next(l for l in req.suffix.splitlines() if l.startswith("RECENT OFF-TOPIC"))
    assert "my snack is right here" in own and "right guys?" in own and "did anyone" not in own
    assert "did anyone see the game" in off and "music" in req.suffix


# ------------------------------------------------------------------ Fix 3: contradictory sampling
from collabsim.sim.group import generate_group  # noqa: E402


@pytest.mark.parametrize("tid", TASKS)
def test_fix3_refutes_mapping_filled(tid):
    task = load_task(tid)
    assert task.refutes
    for m in task.misconceptions:  # every misconception is refutable by some card fact
        assert m.id in task.refuted_by([f.id for f in task.facts])


@pytest.mark.parametrize("tid", TASKS)
def test_fix3_no_self_refuted_misconception_property(tid):
    task = load_task(tid)
    n_m = 0
    for seed in range(1500):
        for n in (3, 4):
            g = generate_group(task, RNG(seed), n_students=n)
            assert any(s.misconceptions for s in g)
            for s in g:
                own = task.refuted_by(s.jigsaw_facts)
                peers = task.refuted_by([f for o in g if o.id != s.id for f in o.jigsaw_facts])
                for m in s.misconceptions:
                    n_m += 1
                    assert m.id not in own, (seed, n, s.name, m.id)
                    assert m.id in peers, (seed, n, s.name, m.id)
    assert n_m > 1500


def test_fix3_engine_groups_respect_refutes(template_sims):
    for sim in template_sims:
        task = sim.task
        for s in sim.initial_students:
            own = task.refuted_by(s["jigsaw_facts"])
            assert not any(m["id"] in own for m in s["misconceptions"])


# ------------------------------------------------------------------ Fix 4: hedge spam
HEDGE_TOKENS = {"idk", "not", "sure", "maybe", "no", "idea", "i", "dont", "know", "clue", "honestly", "tbh",
                "guess", "hmm"}


def _pure_hedge(text: str) -> bool:
    toks = rep.norm_tokens(text)
    return bool(toks) and all(t in HEDGE_TOKENS for t in toks) and any(t in ("idk", "sure", "maybe", "know", "clue")
                                                                      for t in toks)


def test_fix4_hedges_capped_at_pilot_dont_know_share(template_sims):
    n = plan_h = text_h = 0
    for sim in template_sims:
        for m in sim.state.messages:
            if m.kind != "student":
                continue
            n += 1
            plan_h += m.plan.claim == "hedge" or m.plan.act in ("idk", "refusal")
            text_h += _pure_hedge(m.text)
    print(f"\nhedge/idk share: plans {plan_h / n:.3f}, pure-hedge lines {text_h / n:.3f} (pilot ~.04, cap .06)")
    assert plan_h / n <= 0.06 and text_h / n <= 0.06


def test_fix4_no_hedge_claim_without_misconception(template_sims):
    """A content-free on-task hedge is only planned for a student who holds an active
    misconception on that KC (free acts) or as a reply (an address / a question)."""
    for sim in template_sims:
        for m in sim.state.messages:
            p = m.plan
            if m.kind == "student" and p.claim == "hedge" and not p.replies_to_external and p.anchor_turn is None:
                hidden = next(t for t in sim.state.timeline if t["msg"] == m.id)["students"][m.speaker]
                assert any(v > 0 for k, v in hidden["misc"].items()
                           if sim.task.misconception(k).kc == p.kc), (sim.state.session_id, m.id)


# ------------------------------------------------------------------ Fix 5: peer responsiveness
from collabsim.models import Message  # noqa: E402
from collabsim.sim.scheduler import Scheduler, can_answer  # noqa: E402


def _q_state(asked_fact="f_reuse"):
    sim = Simulation("sci-enzyme-01", 11, n_students=3)
    st = sim.state
    asker = next(s for s in st.students if asked_fact not in s.known_facts)
    holder = next(s for s in st.students if asked_fact in s.jigsaw_facts)
    other = next(s for s in st.students if s.id not in (asker.id, holder.id))
    holder.misconceptions = []
    holder.affect.engagement = other.affect.engagement = 0.9
    other.kcs["kc_enzyme"].p = 0.2
    plan = ContentPlan(act="question", kc="kc_enzyme", asked_fact=asked_fact)
    msg = Message(id=0, t=100.0, speaker=asker.id, kind="student", text="does the enzyme get used up?", plan=plan)
    st.messages.append(msg)
    st.t = 100.0
    sim.updater.apply(st, msg, sim.rngs["update"])
    return sim, st, asker, holder, other


def test_fix5_holder_gets_hazard_boost_and_answers_with_fact():
    sim, st, asker, holder, other = _q_state()
    assert can_answer(st, holder.id, st.messages[0]) == "fact" and can_answer(st, other.id, st.messages[0]) is None
    sch = Scheduler(sim.cfg.model_copy(update={"unspoken_boost_per_min": None, "stall_holder_boost": 0}))
    base = holder.traits.talkativeness * holder.affect.engagement   # not written yet: raw talkativeness
    assert sch.hazard(st, holder.id, 101.0) > 1.5 * base
    assert sch.hazard(st, other.id, 101.0) == pytest.approx(other.traits.talkativeness * other.affect.engagement)
    answers = 0
    for k in range(200):
        p = sim.planner._answer_peer_question(st, holder, RNG(k))
        if p is not None:
            answers += 1
            assert p.claim == "fact" and p.fact_ids == ["f_reuse"] and p.anchor_turn == 0 and p.addressee == asker.id
    assert 0.6 < answers / 200 < 0.85  # p_answer_holder * (.5 + .5 * engagement) = .76
    assert all(sim.planner._answer_peer_question(st, other, RNG(k)) is None for k in range(50))


def test_fix5_engagement_modulates_answering():
    sim, st, asker, holder, other = _q_state()
    holder.affect.engagement = 0.1
    low = sum(sim.planner._answer_peer_question(st, holder, RNG(k)) is not None for k in range(300))
    holder.affect.engagement = 1.0
    high = sum(sim.planner._answer_peer_question(st, holder, RNG(k)) is not None for k in range(300))
    assert low < high


def test_fix5_content_questions_answered_more_by_holders(template_sims):
    n = a = 0
    for sim in template_sims:
        stu = [m for m in sim.state.messages if m.kind == "student"]
        for i, m in enumerate(stu):
            if m.plan.act == "question" and m.plan.asked_fact and any(
                    m.plan.asked_fact in s.jigsaw_facts for s in sim.state.students if s.id != m.speaker):
                n += 1
                a += any(x.plan.anchor_turn == m.id and x.speaker != m.speaker for x in stu[i + 1:i + 7])
    print(f"\ncontent questions with a holder answered within 6 turns: {a / n:.3f} (n={n})")
    assert a / n >= 0.5


# ====================================================================================
# fact dedupe, no self-agreement, monologue cap, closing
def _announcements(sim):
    """(same-student repeats, group repeats in the session) of fact ANNOUNCEMENTS (short
    back-references and revoicing excluded): a fact is announced once per session."""
    DEDUPE_WINDOW = 10_000
    stu = [m for m in sim.state.messages if m.kind == "student"]
    own, grp = 0, 0
    said: dict[tuple[str, str], int] = {}
    for i, m in enumerate(stu):
        p = m.plan
        if not (p.claim == "fact" and p.fact_ids) or p.reference or p.revoice:
            continue
        f = p.fact_ids[0]
        own += (m.speaker, f) in said
        grp += any(x.plan.fact_ids and f in x.plan.fact_ids for x in stu[max(0, i - DEDUPE_WINDOW):i])
        said[(m.speaker, f)] = i
    return own, grp


def test_v2_fix1_no_repeated_fact_announcements(template_sims):
    per = [_announcements(sim) for sim in template_sims]
    own = [o for o, _ in per]
    grp = [g for _, g in per]
    print(f"\nrepeated own fact announcements/session mean {sum(own) / len(own):.3f}; "
          f"group repeats in window {sum(grp) / len(grp):.3f}")
    assert sum(own) / len(own) <= 0.02 and max(own) <= 1
    assert sum(grp) / len(grp) <= 0.02


def test_v2_fix1_references_are_short_and_marked(template_sims):
    n = 0
    for sim in template_sims:
        for m in sim.state.messages:
            if m.kind == "student" and m.plan.reference:
                n += 1
                assert m.plan.fact_ids and len(m.text.split()) <= 20
    assert n > 0  # answering a pending question / pointing back still happens


def test_v2_fix1_misconception_not_revoiced_unless_challenged(template_sims):
    from collabsim.sim.planner import ContentPlanner
    bad = 0
    for sim in template_sims:
        st = sim.state
        for m in st.messages:
            p = m.plan
            if m.kind != "student" or p.claim != "misconception" or p.replies_to_external:
                continue
            a = st.messages[p.anchor_turn].plan if p.anchor_turn is not None else None
            if a is not None and a.act == "question":
                continue  # answering a pending question about it
            # replay the rule on the transcript prefix
            pre = st.model_copy(update={"messages": st.messages[:m.id]})
            if not ContentPlanner.may_voice(pre, st.student(m.speaker), p.misconception_id):
                bad += 1
    assert bad == 0


def test_v2_fix1_reference_prompt_says_do_not_announce():
    task = load_task("sci-enzyme-01")
    from collabsim.llm.prompts import say_this
    f = task.facts[0].id
    txt = say_this(task, ContentPlan(act="explain", kc=task.facts[0].kc, claim="fact", fact_ids=[f],
                                     is_correct=True, reference=True), {})
    assert "already said" in txt and "do not announce" in txt


def test_v2_fix2_no_self_agreement(template_sims):
    n = bad = 0
    for sim in template_sims:
        msgs = sim.state.messages
        for m in msgs:
            if m.kind != "student" or m.plan.act not in ("agree", "backchannel") or m.plan.opener:
                continue
            n += 1
            vis = msgs[:m.id]
            stu = [x for x in msgs[:m.id] if x.kind == "student"]
            right_after_own = bool(vis) and vis[-1].speaker == m.speaker
            last2_own = len(stu) >= 2 and all(x.speaker == m.speaker for x in stu[-2:]) and not m.plan.replies_to_external
            no_peer = not any(x.speaker != m.speaker for x in stu[-3:])
            bad += right_after_own or last2_own or no_peer
    print(f"\nagree/backchannel turns {n}, self-reactions {bad}")
    assert n > 150 and bad == 0


def test_v2_fix2_rule_unit(make_state, task, params, cfg):
    from tests.conftest import add_student
    from collabsim.sim.planner import ContentPlanner
    st = make_state()
    a, b = st.roster[0], st.roster[1]
    add_student(st, a, ContentPlan(act="on_task_claim", kc="kc_surface", claim="correct", is_correct=True))
    assert ContentPlanner.is_self_reaction(st, a, ContentPlan(act="agree"))
    assert not ContentPlanner.is_self_reaction(st, b, ContentPlan(act="agree"))
    assert not ContentPlanner.is_self_reaction(st, a, ContentPlan(act="question"))
    pl = ContentPlanner(task, params, cfg)
    for i in range(40):  # the planner never returns a self-reaction for the last speaker (None = skips)
        p = pl.plan(st, a, RNG(i))
        assert p is None or p.act not in ("agree", "backchannel"), p


def test_v2_fix3_monologue_cap(template_sims):
    import json
    from tests.support.structural import run_lengths, run_stats
    from tests.support.pilot_stats import load_targets
    from collabsim.sim import transcript_for_stats
    from collabsim.sim.config import SimConfig
    tg = load_targets()
    cap = SimConfig().max_run_hard_cap
    assert cap <= tg["pooled"]["run_len_max"]  # hard cap within the pilot max run
    sessions = [transcript_for_stats(sim.state) for sim in template_sims]
    assert max(max(run_lengths(s)) for s in sessions) <= cap
    rs = run_stats(sessions)
    print(f"\nrun p95 {rs['run_len_p95']} (pilot {tg['pooled']['run_len_p95']}), max-run median "
          f"{rs['max_run_median']} (pilot {tg['pooled']['max_run_median']}), top-speaker share "
          f"{rs['top_speaker_share_median']:.3f} (pilot {tg['pooled']['top_speaker_share_median']:.3f})")
    assert rs["run_len_p95"] <= tg["leave_2_out_range"]["run_len_p95"]["max"]
    assert rs["max_run_median"] <= tg["leave_2_out_range"]["max_run_median"]["max"]


def test_v2_fix3_hazard_decays_with_run():
    from collabsim.sim.scheduler import Scheduler
    from collabsim.sim.config import SimConfig
    cfg = SimConfig()
    sim = Simulation("sci-enzyme-01", 3)
    sim.run(max_events=60)
    st = sim.state
    who, n = Scheduler.current_run(st)
    stu = [m for m in st.messages if m.kind == "student"]
    assert who == stu[-1].speaker and n >= 1
    assert cfg.burst_decay < 1.0


def test_v2_fix5_session_ends_after_closing(template_sims):
    """A completed task ends with the closing synthesis (within the capped wrap-up window); an
    incomplete one wraps up near the time / turn cap (closing_forced) instead of fading out."""
    from collabsim.sim.config import SimConfig
    cfg = SimConfig()
    reasons = {}
    for sim in template_sims:
        st, res = sim.state, sim.result()
        assert res["ended_reason"] in ("complete", "time_cap", "turn_cap")
        assert st.closing == "done" and len(st.closing_msgs) >= 2
        reasons[res["ended_reason"]] = reasons.get(res["ended_reason"], 0) + 1
        if len(st.steps_solved) == len(sim.task.steps):
            assert res["solved_at_min"] == pytest.approx(st.steps_solved[-1]["t"] / 60, abs=1e-3)
            last = max(m.t for m in st.messages)
            assert last <= st.solved_at + cfg.closing_wrap_up_max_minutes * 60 + 1e-6
            assert res["ended_reason"] == "complete"
        else:
            assert res["solved_at_min"] is None and res["ended_reason"] != "complete"
            assert st.closing_forced and st.closing == "done", (st.session_id, st.closing)
    print(f"\nended_reason counts: {reasons}")
    assert reasons.get("complete", 0) > 50


def test_v2_fix5_no_closing_without_end_on_complete():
    from collabsim.sim.config import SimConfig
    sim = Simulation("sci-enzyme-01", 3, SimConfig(end_on_complete=False, closing_headroom_min=0,
                                                   closing_headroom_turns=0))
    sim.run()
    assert sim.state.ended_reason in ("time_cap", "turn_cap")


PT_RE = re.compile(r"k{3,}", re.I)


def test_v2_fix4_no_portuguese_markers_in_banks_and_prompts():
    from collabsim.llm.prompts import fewshots, realizer_prefix
    from collabsim.sim.offtask import TOPICS
    from collabsim.sim.realizer import FILLER, PT_MARKERS_RE
    lines = [l for v in FILLER.values() for l in v] + [l for _, ls in TOPICS.values() for l in ls]
    lines += [f["en"] for f in fewshots()]
    for l in lines:
        assert not PT_RE.search(l) and not PT_MARKERS_RE.search(l), l
    for tid in TASKS:
        pre = realizer_prefix(load_task(tid))
        assert not PT_RE.search(pre) and not PT_MARKERS_RE.search(pre)
        assert "English only" in pre


def test_v2_fix4_no_portuguese_in_generated_lines(template_sims):
    from collabsim.sim.realizer import PT_MARKERS_RE
    for sim in template_sims:
        for m in sim.state.messages:
            if m.kind == "student":
                assert not PT_RE.search(m.text) and not PT_MARKERS_RE.search(m.text), m.text


def test_v2_fix4_model_output_laughter_rewritten():
    from collabsim.sim.realizer import PT_MARKERS_RE
    h = default_handlers()
    h["realize"] = lambda req: "kkkkk enzymes are not used up rs"
    client = LLMClient(MockLLM(h))
    sim = build_session("sci-enzyme-01", 2, realizer="mock", client=client)
    sim.run(max_events=60)
    texts = [m.text for m in sim.state.messages if m.kind == "student"]
    assert texts and not any(PT_RE.search(t) or PT_MARKERS_RE.search(t) for t in texts)
