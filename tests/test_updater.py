"""One unit test per update rule, plus the targeted/misaligned/generic analytic-vs-Monte-Carlo
check. External participants' turns go through exactly the same rules as a peer's."""
import math

import pytest

from collabsim.models import ContentPlan, RNG

from .conftest import add_external, add_student

MISC = ContentPlan(act="on_task_claim", kc="kc_surface", claim="misconception",
                   misconception_id="m_swallow_only", is_correct=False)


# ------------------------------------------------------------------ rule 1
def _press(target: str, anchor: int, mid: str = "m_swallow_only", kc: str = "kc_surface") -> ContentPlan:
    """A press for reasoning on a voiced misconception, anchored to its utterance."""
    return ContentPlan(act="press", kc=kc, anchor_turn=anchor, addressee=target, family="press_for_reasoning",
                       addresses_misconception=mid)


def test_rule1_labels_from_plan_for_students_and_externals(make_state, updater):
    st = make_state()
    m = add_student(st, "s1", MISC)
    lab = updater.rule1_classify(st, m)
    assert lab["act"] == "on_task_claim" and lab["expresses"] == "m_swallow_only"
    assert st.student("s1").misconception("m_swallow_only").voiced
    e = add_external(st, "e1", _press("s1", m.id))
    lab = updater.rule1_classify(st, e)
    assert lab["act"] == "question" and lab["family"] == "press_for_reasoning"
    assert lab["anchor"] == m.id and lab["addressed_misconception"] == "m_swallow_only"


# ------------------------------------------------------------------ rule 2
def test_rule2_exposure(make_state, updater):
    st = make_state()
    st.student("s3").affect.engagement = 0.1  # below read threshold
    m = add_student(st, "s2", ContentPlan(act="on_task_claim", kc="kc_surface", claim="fact",
                                         fact_ids=["f_rate"], is_correct=True))
    lab = updater.rule1_classify(st, m)
    readers = updater.rule2_exposure(st, m, lab, RNG(0))
    assert readers == ["s1"]
    assert "f_rate" in st.student("s1").known_facts
    assert "f_rate" not in st.student("s3").known_facts or "f_rate" in st.student("s3").jigsaw_facts
    assert "f_rate" in st.student("s2").shared_facts


# ------------------------------------------------------------------ rule 3
def test_rule3_kc_learning(make_state, updater):
    st = make_state()
    st.student("s1").misconceptions = []
    m = add_student(st, "s2", ContentPlan(act="explain", kc="kc_enzyme", claim="correct", is_correct=True))
    updater.apply(st, m, RNG(0))
    assert st.student("s2").kcs["kc_enzyme"].p == pytest.approx(0.3 + 0.7 * 0.20)
    assert st.student("s2").kcs["kc_enzyme"].produced_explanation
    for sid in ("s1", "s3"):
        assert st.student(sid).kcs["kc_enzyme"].p == pytest.approx(0.3 + 0.7 * 0.10)
    m = add_student(st, "s3", ContentPlan(act="on_task_claim", kc="kc_temp", claim="misconception",
                                         misconception_id="m_hotter_faster", is_correct=False))
    updater.apply(st, m, RNG(0))
    assert st.student("s1").kcs["kc_temp"].p == pytest.approx(0.3)  # T=0 for incorrect


def test_rule3_cap_while_misconception_active(make_state, updater):
    st = make_state()
    st.student("s1").kcs["kc_surface"].p = 0.54
    m = add_student(st, "s2", ContentPlan(act="explain", kc="kc_surface", claim="correct", is_correct=True))
    st.student("s1").affect.engagement = 1.0
    updater.rule3_learning(st, m, updater.rule1_classify(st, m), ["s1"])
    assert st.student("s1").kcs["kc_surface"].p == pytest.approx(updater.th.active_misconception_p_cap)


# ------------------------------------------------------------------ rule 4
def _nudge(st, sid="s1"):
    return add_external(st, "e1", _press(sid, 0))


def test_rule4_targeted_condition_and_flip_effects(make_state, updater):
    st = make_state()
    updater.apply(st, add_student(st, "s1", MISC), RNG(0))
    a = _nudge(st)
    lab = updater.rule1_classify(st, a)
    assert updater.condition(st, a, lab, "m_swallow_only") == "targeted"
    # force a flip with an RNG that returns 0
    class Zero(RNG):
        def random(self): return 0.0
    p0 = st.student("s1").kcs["kc_surface"].p
    ev = updater.rule4_misconceptions(st, a, lab, ["s1"], Zero())
    assert ev[0]["flipped"] and not st.student("s1").misconception("m_swallow_only").active
    assert ev[0]["source"] == "external"
    assert st.student("s1").kcs["kc_surface"].p == pytest.approx(p0 + 0.15)


def _misaligned(st):
    """s2 voices a DIFFERENT misconception; a turn on kc_surface anchored to it is misaligned for s1."""
    v = add_student(st, "s2", ContentPlan(act="on_task_claim", kc="kc_enzyme", claim="misconception",
                                          misconception_id="m_enzyme_used_up", is_correct=False))
    return add_external(st, "e1", ContentPlan(act="disagree", kc="kc_surface", claim="correct", claim_idx=0,
                                              is_correct=True, anchor_turn=v.id, addressee="s1",
                                              family="peer_explanation"))


def test_rule4_nonflip_and_entrench(make_state, updater):
    class One(RNG):
        def random(self): return 0.999999
    st = make_state()
    updater.apply(st, add_student(st, "s1", MISC), RNG(0))
    a = _nudge(st)
    updater.rule4_misconceptions(st, a, updater.rule1_classify(st, a), ["s1"], One())
    assert st.student("s1").misconception("m_swallow_only").strength == pytest.approx(0.6)
    a = _misaligned(st)
    lab = updater.rule1_classify(st, a)
    assert updater.condition(st, a, lab, "m_swallow_only") == "misaligned"
    updater.rule4_misconceptions(st, a, lab, ["s1"], One())
    assert st.student("s1").misconception("m_swallow_only").strength == pytest.approx(0.65)


def test_rule4_anchor_window_and_addressee_multiplier(make_state, updater):
    st = make_state()
    updater.apply(st, add_student(st, "s1", MISC), RNG(0))
    for _ in range(3):  # push the misconception utterance out of the 3-turn window
        updater.apply(st, add_student(st, "s2", ContentPlan(act="backchannel")), RNG(0))
    a = _nudge(st)
    lab = updater.rule1_classify(st, a)
    assert updater.condition(st, a, lab, "m_swallow_only") == "generic"
    # non-addressee gets x0.5
    st2 = make_state()
    updater.apply(st2, add_student(st2, "s1", MISC), RNG(0))
    a2 = add_external(st2, "e1", _press("s2", 0))
    lab2 = updater.rule1_classify(st2, a2)
    r = st2.student("s1")
    p_non = updater.flip_probability(st2, a2, r, "targeted", 0.7, "s2")
    p_addr = updater.flip_probability(st2, a2, r, "targeted", 0.7, "s1")
    assert p_non == pytest.approx(0.5 * p_addr)
    assert updater.condition(st2, a2, lab2, "m_swallow_only") == "targeted"


def test_rule4_peer_correction_targeted(make_state, updater):
    st = make_state()
    updater.apply(st, add_student(st, "s1", MISC), RNG(0))
    m = add_student(st, "s2", ContentPlan(act="disagree", kc="kc_surface", claim="fact", fact_ids=["f_surface"],
                                         is_correct=True, anchor_turn=0, addressee="s1",
                                         family="evidence_from_material",
                                         addresses_misconception="m_swallow_only"))
    assert updater.condition(st, m, updater.rule1_classify(st, m), "m_swallow_only") == "targeted"


def test_family_outside_flip_by_is_generic(make_state, updater):
    """counterexample is NOT in m_swallow_only.flip_by: an anchored counterexample is only generic."""
    st = make_state()
    updater.apply(st, add_student(st, "s1", MISC), RNG(0))
    a = add_external(st, "e1", ContentPlan(act="disagree", kc="kc_surface", claim="correct", is_correct=True,
                                           anchor_turn=0, addressee="s1", family="counterexample",
                                           addresses_misconception="m_swallow_only"))
    assert updater.condition(st, a, updater.rule1_classify(st, a), "m_swallow_only") == "generic"


def test_external_and_peer_have_the_same_influence(make_state, updater):
    """The same plan has the same flip probability whoever wrote it (no identity-based rule)."""
    st = make_state()
    updater.apply(st, add_student(st, "s1", MISC), RNG(0))
    plan = _press("s1", 0)
    peer = add_student(st, "s2", plan)
    ext = add_external(st, "e1", plan)
    r = st.student("s1")
    assert updater.flip_probability(st, peer, r, "targeted", 0.7, "s1") == \
        updater.flip_probability(st, ext, r, "targeted", 0.7, "s1")


# ------------------------------------------------------------------ rule 5
def test_rule5_affect(make_state, updater):
    st = make_state(engagement=0.8)
    # unanswered question -> frustration
    q = add_student(st, "s2", ContentPlan(act="question", kc="kc_surface"), t=60.0)
    updater.apply(st, q, RNG(0))
    for i in range(5):
        updater.apply(st, add_student(st, "s3", ContentPlan(act="backchannel"), t=70.0 + i), RNG(0))
    assert st.student("s2").affect.frustration == pytest.approx(0.10)
    # an external participant's question is a pending question too (answered -> removed)
    eq = add_external(st, "e1", ContentPlan(act="question", kc="kc_surface", asked_fact="f_rate"), t=75.0)
    updater.apply(st, eq, RNG(0))
    assert any(p.msg_id == eq.id for p in st.pending_questions)
    updater.apply(st, add_student(st, "s1", ContentPlan(act="explain", kc="kc_surface", claim="fact",
                                                        fact_ids=["f_rate"], is_correct=True, anchor_turn=eq.id),
                                  t=76.0), RNG(0))
    assert not any(p.msg_id == eq.id for p in st.pending_questions)
    # engagement decays without progress
    e0 = st.student("s1").affect.engagement
    updater.apply(st, add_student(st, "s3", ContentPlan(act="backchannel"), t=76.0 + 600), RNG(0))
    assert st.student("s1").affect.engagement == pytest.approx(e0 - 0.01 * 600 / 60, abs=1e-6)


# ------------------------------------------------------------------ rule 6
def test_rule6_step_progress(make_state, updater):
    st = make_state()
    st.student("s1").misconceptions = []
    good = lambda: ContentPlan(act="on_task_claim", kc="kc_surface", claim="correct", is_correct=True)
    for sid in ("s1", "s1", "s1"):
        updater.apply(st, add_student(st, sid, good()), RNG(0))
    assert st.step_idx == 0  # 3 claims but only 1 speaker
    updater.apply(st, add_student(st, "s2", good()), RNG(0))
    assert st.step_idx == 1 and st.steps_solved[0]["step"] == "s1"
    # off-step KC does not count
    updater.apply(st, add_student(st, "s2", good()), RNG(0))
    assert st.step_correct == {}


def test_rule6_external_turn_counts_and_challenges(make_state, updater):
    """An external participant stating a key idea counts toward the step; challenging a voiced
    misconception unblocks it (rule 6 (c))."""
    st = make_state()
    updater.apply(st, add_student(st, "s1", MISC), RNG(0))
    assert updater.blocking_misconceptions(st, "s1")
    updater.apply(st, add_external(st, "e1", _press("s1", 0)), RNG(0))
    assert not updater.blocking_misconceptions(st, "s1")
    e = add_external(st, "e1", ContentPlan(act="on_task_claim", kc="kc_surface", claim="fact", fact_ids=["f_surface"],
                                           is_correct=True))
    updater.apply(st, e, RNG(0))
    assert "s1_surface_area" in st.ideas_voiced and st.step_correct["kc_surface"] == 1


# ------------------------------------- analytic flip probability vs Monte Carlo
def _flip_rate(make_state, updater, cond: str, n: int) -> tuple[float, float]:
    flips = 0
    p_an = None
    for i in range(n):
        st = make_state(engagement=1.0)  # read prob = 1 -> s1 always reads
        updater.apply(st, add_student(st, "s1", MISC, t=0.0), RNG(10_000 + i))
        if cond == "targeted":
            a = _nudge(st)
        elif cond == "misaligned":
            st.messages.append(st.messages[0].model_copy(update={"id": 1, "speaker": "s2", "plan": ContentPlan(
                act="on_task_claim", kc="kc_enzyme", claim="misconception", misconception_id="m_enzyme_used_up",
                is_correct=False)}))
            a = add_external(st, "e1", ContentPlan(act="disagree", kc="kc_surface", claim="correct", is_correct=True,
                                                   anchor_turn=1, family="peer_explanation"))
        else:
            a = add_external(st, "e1", ContentPlan(act="on_task_claim", kc="kc_surface", claim="correct",
                                                   is_correct=True))
        a.t = 1.0
        lab = updater.apply(st, a, RNG(i))
        ev = [e for e in lab["flips"] if e["student"] == "s1"]
        assert ev and ev[0]["cond"] == cond
        flips += ev[0]["flipped"]
        p_an = ev[0]["p"]
    return flips / n, p_an


def test_flip_probability_analytic_vs_monte_carlo(make_state, updater):
    th = updater.th
    eng = 1.0 ** th.engagement_exponent
    analytic = {"targeted": th.p_targeted(0.7) * eng, "misaligned": th.p_flip.misaligned * eng,
                "generic": th.p_flip.generic * eng}
    n = 3000
    mc = {}
    print("\ncond        analytic   MC(n=%d)" % n)
    for cond in ("targeted", "generic", "misaligned"):
        rate, p_logged = _flip_rate(make_state, updater, cond, n)
        mc[cond] = rate
        se = math.sqrt(analytic[cond] * (1 - analytic[cond]) / n)
        print(f"{cond:10s}  {analytic[cond]:.4f}    {rate:.4f}")
        assert p_logged == pytest.approx(analytic[cond], abs=1e-4)
        assert abs(rate - analytic[cond]) < 4 * se + 1e-3
    # common random numbers (same RNG(i) per condition) -> MC ordering is monotone in p
    assert mc["targeted"] > mc["generic"] > mc["misaligned"]


def test_engagement_exponent(make_state, updater):
    st = make_state(engagement=0.64)
    m = add_external(st, "e1", ContentPlan(act="explain", kc="kc_surface", claim="correct", is_correct=True))
    p = updater.flip_probability(st, m, st.student("s1"), "generic", 0.7, None)
    assert p == pytest.approx(updater.th.p_flip.generic * 0.8)
