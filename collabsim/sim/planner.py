"""Act sampling + content planning: decides IN CODE what a student turn says (which fact /
misconception / correct claim / idk), whom it answers and whom it addresses. The realizer only
words it.

People a student may address or answer: the other students, and the external participants who
have already written in the chat (st.spoken_externals()). Nothing here depends on who an external
participant is: their turns are read through their plans, like a peer's.
"""
from __future__ import annotations

import math

from ..models.message import CONTENT_FREE_ACTS, EPISTEMIC_ACTS, ContentPlan, Message
from ..models.rng import RNG
from ..models.session import SessionState
from ..models.student import StudentState
from ..models.task import Task
from . import intents
from .config import SimConfig
from .content import claim_voiceable, credit_source, last_mention, own_statements, raw_claim_key
from .learning import LearningParams
from .offtask import FRESH_TOPIC_MASS, TOPIC_FAMILY, TOPIC_IDS, TOPICS_NOT_ONLINE
from .scheduler import can_answer, stalled
from . import realism

REASK_WINDOW = 10        # turns: a follow-up only for a question asked this recently
P_FOLLOW_UP = 0.6        # a re-ask becomes a short follow-up ("anyone?", "<peer> do you know?") or another act

REACTION_ACTS = {"agree", "backchannel"}
OFF_TASK_MEAN = 2 / 11
INTERFACE_MEAN = 2 / 14


def is_content_free(plan: ContentPlan) -> bool:
    """A turn that carries no task content (idk / refusal / backchannel / agree / hedge, or a
    one-word off-task atom like "mds")."""
    return plan.closing is None and plan.draft is None and plan.concedes is None and not plan.opener and not plan.split_tail and (plan.act in CONTENT_FREE_ACTS or plan.claim == "hedge" or (
        plan.act == "off_task" and (plan.target_len <= 1 or (plan.anchor_turn is not None and plan.target_len <= 3))))


class ContentPlanner:
    def __init__(self, task: Task, params: LearningParams, cfg: SimConfig):
        self.task = task
        self.th = params
        self.cfg = cfg

    # ------------------------------------------------------------------ entry
    _no_new_subject = False

    def plan(self, st: SessionState, sid: str, rng: RNG, forced: bool = False) -> ContentPlan | None:
        """The student's next turn, or None when they have nothing to say (a blocked filler with no
        content to replace it: the turn is skipped). While planning a same-speaker continuation (not
        right after a greeting), the fallback acts never open a new subject. `forced` (the group-silence
        cap was reached, 0.5.1): never None -- content if there is any, else the activity question or
        a short nudge."""
        pend = st.pending_contribution
        if pend is not None:
            st.pending_contribution = None
            if pend["speaker"] == sid:     # the content that follows the opener just sent
                return ContentPlan.model_validate(pend["plan"])
        last = st.recent_turns(1)
        self._no_new_subject = bool(last and last[-1].speaker == sid and last[-1].plan is not None
                                    and last[-1].plan.act != "greeting")
        try:
            plan = self._plan(st, sid, rng)
            if forced and (plan is None or (is_content_free(plan) and not plan.opener)):
                plan = self._forced_move(st, st.student(sid), rng)   # (0.5.3: never a filler / off-task reply)
        finally:
            self._no_new_subject = False
        plan = self._maybe_opener(st, sid, plan, rng)
        self._maybe_afterthought(st, sid, plan, rng)
        return plan

    def _maybe_afterthought(self, st: SessionState, sid: str, plan: ContentPlan | None, rng: RNG) -> None:
        """After a statement of one's own (a fact, a claim, a belief), sometimes a short follow-up
        ("ne?", "faz sentido?") by the same student, a few seconds later."""
        if plan is None or plan.opener or st.pending_contribution is not None or len(st.roster) < 2:
            return
        if plan.claim not in ("fact", "correct", "misconception") or plan.reference or plan.closing or plan.draft:
            return
        if plan.act not in ("on_task_claim", "explain", "disagree"):
            return
        from .scheduler import Scheduler
        who, run = Scheduler.current_run(st)
        if who == sid and run + 2 > self.cfg.max_run_hard_cap:
            return          # the follow-up would make the same-speaker run longer than the cap (0.5.3)
        if rng.bernoulli(self.cfg.p_afterthought) and self.tag_allowed(st, sid):
            tag = ContentPlan(act="question", tag=True, kc=plan.kc, target_len=2)
            st.pending_contribution = {"speaker": sid, "plan": tag.model_dump(), "after": True}
        elif rng.bernoulli(self.cfg.p_after_hedge) and self.after_hedge_allowed(st, sid):
            # 0.5.3: or a short hedge that is not a question ("acho", "se eu entendi direito")
            h = ContentPlan(act="explain", after_hedge=True, target_len=2)    # (no KC: it carries no content)
            st.pending_contribution = {"speaker": sid, "plan": h.model_dump(), "after": True}

    def after_hedge_allowed(self, st: SessionState, sid: str) -> bool:
        """Hedge follow-ups: at most after_hedge_share_cap of the session's content claims, never after two
        consecutive claims of the same speaker, each wording once."""
        stu = st.student_msgs()
        hs = [m for m in stu if m.plan is not None and m.plan.after_hedge]
        claims = sum(1 for m in stu if self.content_claim(m.plan)) + 1
        if len(hs) + 1 > max(1, math.floor(self.cfg.after_hedge_share_cap * claims)):
            return False
        from .. import lang
        from .realizer import HEDGE_FOLLOWUPS
        if len(hs) >= len(lang.attr(self.task, "HEDGE_FOLLOWUPS", HEDGE_FOLLOWUPS)):
            return False
        own = [m for m in stu if m.speaker == sid and (self.content_claim(m.plan) or (
            m.plan is not None and (m.plan.tag or m.plan.after_hedge)))]
        return not (own and own[-1].plan is not None and (own[-1].plan.tag or own[-1].plan.after_hedge))

    @staticmethod
    def content_claim(plan: ContentPlan | None) -> bool:
        return (plan is not None and plan.claim in ("fact", "correct", "misconception") and not plan.reference
                and plan.closing is None and plan.draft is None and plan.act in ("on_task_claim", "explain", "disagree"))

    def tag_allowed(self, st: SessionState, sid: str) -> bool:
        """0.5.2 cap on afterthought / tag questions ("vcs concordam?", "ne isso?"): at most tag_share_cap
        of the session's content claims (this one included), never after two consecutive claims of the same
        speaker, and each wording once (the template draws an unused one)."""
        stu = st.student_msgs()
        tags = [m for m in stu if m.plan is not None and m.plan.tag]
        claims = sum(1 for m in stu if self.content_claim(m.plan)) + 1
        if len(tags) + 1 > max(1, math.floor(self.cfg.tag_share_cap * claims)):
            return False
        from .. import lang
        from .realizer import TAG_LINES
        if len(tags) >= len(lang.attr(self.task, "TAG_LINES", TAG_LINES)):
            return False
        own = [m for m in stu if m.speaker == sid and (self.content_claim(m.plan) or (m.plan is not None and m.plan.tag))]
        return not (own and own[-1].plan is not None and own[-1].plan.tag)

    def _maybe_opener(self, st: SessionState, sid: str, plan: ContentPlan | None, rng: RNG) -> ContentPlan | None:
        """Sometimes a contribution comes as two messages: a one-word opener, then the content."""
        if plan is None or plan.closing is not None or plan.reference or plan.continuation or plan.pushback:
            return plan
        if plan.claim not in ("fact", "correct", "misconception") or plan.act not in ("on_task_claim", "explain"):
            return plan
        if not any(m.speaker != sid for m in st.messages[-6:]) or st.messages[-1].speaker == sid \
                or not rng.bernoulli(self.cfg.p_opener):
            return plan
        if self.content_free_run(st) >= self.cfg.max_content_free_run:
            return plan       # "mano" after two fillers would read as a third one
        if not self.opener_allowed(st, sid):
            return plan       # 0.5.3: lead-word-only messages ("pera", "galera", "olha") are rare
        st.pending_contribution = {"speaker": sid, "plan": plan.model_dump()}
        # a one-word opener ("pera") or a short lead ("pera achei aqui"), half and half (0.5.1)
        return ContentPlan(act="backchannel", opener=True,
                           target_len=1 if rng.bernoulli(self.cfg.p_opener_one_word) else 3)

    def opener_allowed(self, st: SessionState, sid: str) -> bool:
        """0.5.3 cap on lead-word-only messages ("pera" / "galera" / "gente olha isso", then the content): about
        one per opener_every student messages in the session, at most opener_per_student_max per student, and none
        in the group's last 3 messages."""
        stu = st.student_msgs()
        ops = [m for m in stu if m.plan is not None and m.plan.opener]
        if len(stu) + 1 < self.cfg.opener_every // 2:
            return False
        if len(ops) + 1 > math.ceil((len(stu) + 1) / self.cfg.opener_every):
            return False
        if sum(1 for m in ops if m.speaker == sid) >= self.cfg.opener_per_student_max:
            return False
        return not any(m.plan is not None and m.plan.opener for m in stu[-3:])

    def _plan(self, st: SessionState, sid: str, rng: RNG) -> ContentPlan | None:
        s = st.student(sid)
        kc = self.current_kc(st)

        if st.closing is None:
            if st.task_complete and self.cfg.end_on_complete:
                st.closing = "propose"
            elif self.needs_wrap_up(st):
                st.closing, st.closing_forced = "propose", True
        if st.closing in ("propose", "confirm"):
            cp = self._closing_plan(st, s, rng)
            if cp is not None:
                return self._finish(cp, s, rng, st)

        def done(p: ContentPlan) -> ContentPlan:
            p = self._realism_gate(st, s, kc, p, rng)
            p = self._enforce(st, s, kc, p, rng)
            p = self._no_self_reaction(st, s, kc, p, rng)
            p = self._enforce(st, s, kc, p, rng, final=True)
            q = self._realism_gate(st, s, kc, p, rng)
            if q is not p or q.act != p.act:
                p = self._enforce(st, s, kc, q, rng, final=True)
            p = self._window_gate(st, s, kc, p, rng)
            p = self._misconception_gate(st, s, kc, p, rng)
            p = self._stance_gate(st, s, kc, p, rng)
            p = self._burst_gate(st, s, kc, p, rng)
            p = self._finish(p, s, rng, st)          # length first: a one-word off-task atom is a filler
            q = self._filler_gate(st, s, kc, p, rng)
            return q if q is None or q is p else self._finish(q, s, rng, st)

        if sid not in st.greeted:
            st.greeted.append(sid)
            if rng.bernoulli(self.cfg.p_greeting_first_msg) and sid not in st.open_address:
                return self._finish(ContentPlan(act="greeting"), s, rng, st)   # (asked by name: answers instead)
        mv = self._step_move(st, s, rng)
        if mv is not None:
            return self._finish(mv, s, rng, st)
        oa = st.open_address.pop(sid, None)
        if oa is not None and oa.source in st.roster:
            plan = self._respond_to_peer(st, s, oa, kc, rng)
            if plan is not None:
                return done(plan)
        elif oa is not None:
            plan = self._respond_to_address(st, s, oa, kc, rng)
            if plan is not None:
                return done(plan)
        gp = st.open_address.get("group")
        if gp is not None and rng.bernoulli(self.cfg.p_group_prompt_uptake):
            st.open_address.pop("group", None)
            plan = self._substantive(st, s, kc, rng, anchor=gp.msg_id, direct=True)
            plan.replies_to_external = True
            return done(plan)
        plan = self._answer_peer_question(st, s, rng)
        if plan is None:
            plan = self._answer_interface(st, s, rng)
        if plan is None and kc is not None:
            plan = self._concede(st, s, kc, rng)
        if plan is None and kc is not None:
            plan = self._slated_voice(st, s, kc, rng)
        if plan is None and kc is not None:
            plan = self._discuss_slate(st, s, rng)
        if plan is None and kc is not None:
            plan = self._draft_move(st, s, kc, rng)
        if plan is None and kc is not None and stalled(st, self.cfg) and rng.bernoulli(self.cfg.p_progress_move):
            plan = self._progress_move(st, s, kc, rng)
        if plan is None:
            plan = self._challenge(st, s, rng, p=self.cfg.p_peer_correction)
        if plan is None:
            plan = self._free_act(st, s, kc, rng)
        return done(plan)

    def _step_move(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan | None:
        """0.5.3: right after an activity was solved, someone sometimes marks moving on ("bora pra 2?", "essa ja
        foi, proxima"): once per solved step, within its first few student messages."""
        if not st.steps_solved or st.task_complete or st.closing is not None:
            return None
        if st.steps_solved[-1].get("msg") != st.step_started_msg:
            return None
        since = [m for m in st.student_msgs() if m.id > st.step_started_msg]
        if len(since) >= 3 or any(m.plan is not None and m.plan.step_move for m in since):
            return None
        if st.open_address.get(s.id) is not None or (st.messages and st.messages[-1].speaker == s.id):
            return None
        if not rng.bernoulli(self.cfg.p_step_move):
            return None
        return ContentPlan(act="question", step_move=True)

    def needs_wrap_up(self, st: SessionState) -> bool:
        """Near the time cap or the turn cap the group wraps up with what it established."""
        cfg = self.cfg
        return (st.t >= cfg.session_minutes * 60.0 - min(cfg.closing_headroom_min * 60.0, cfg.session_minutes * 9.0)
                or st.student_turns >= cfg.max_student_turns - self.turn_headroom())

    def turn_headroom(self) -> int:
        return max(2, min(self.cfg.closing_headroom_turns, self.cfg.max_student_turns // 6))

    # ------------------------------------------- code-side content enforcement
    def _enforce(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan, rng: RNG,
                 final: bool = False) -> ContentPlan:
        """(1) a correct claim voices one of the KC's correct_claims that carries no true leak for this
        speaker (claim_idx, decided here, not by the LLM); (2) a student re-states the same claim at
        most once more, and only when answering a question / challenge; when challenging a
        misconception it is never re-stated (press instead); (3) a back-reference records who said it
        before (`ref_by`); (4) interface / procedural questions carry a capped intent key. A replacement
        act may itself need these rules, so they are re-applied until the plan is stable."""
        for _ in range(4):
            new = self._enforce_once(st, s, kc, plan, rng, final)
            if new is plan:
                return plan
            plan = new
        if self._violates(st, s, plan):
            plan = self._carry(plan, ContentPlan(act="idk", kc=kc))
        return plan

    def _violates(self, st: SessionState, s: StudentState, plan: ContentPlan) -> bool:
        if plan.claim == "correct" and plan.kc and plan.family != "counterexample" and plan.claim_idx is None:
            return True
        if self._needs_intent(plan):
            return True
        if raw_claim_key(plan) is not None and plan.claim != "misconception":
            n = len(own_statements(self.task, st, s.id, plan))
            return n >= 2 or (n == 1 and not self._direct(st, s, plan))
        return False

    def _enforce_once(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan, rng: RNG,
                      final: bool = False) -> ContentPlan:
        task = self.task
        if plan.claim == "correct" and plan.kc and plan.family != "counterexample" and plan.claim_idx is None:
            pool = (task.refuting_claims(plan.addresses_misconception) if plan.addresses_misconception
                    else range(len(task.kc(plan.kc).correct_claims)))
            ok = [i for i in pool if claim_voiceable(task, s, plan.kc, i, st)]
            new = [i for i in ok if not own_statements(task, st, s.id, plan.model_copy(update={"claim_idx": i}))]
            if ok:
                plan.claim_idx = rng.choice(new or ok)
            elif plan.addresses_misconception and plan.anchor_turn is not None:
                return self._press_instead(plan)
            else:
                return self._downgrade(st, s, plan, rng)
        if raw_claim_key(plan) is not None and plan.claim != "misconception":
            n = len(own_statements(task, st, s.id, plan))
            if n >= 1 and plan.addresses_misconception and plan.act in ("disagree", "explain"):
                # challenging a misconception never re-states a claim or fact the student already made
                # (back-references still recite the content): a press for reasoning instead
                return ContentPlan(act="press", kc=plan.kc, anchor_turn=plan.anchor_turn,
                                   addressee=plan.addressee, family="press_for_reasoning",
                                   addresses_misconception=plan.addresses_misconception)
            if n >= 2 or (n == 1 and not self._direct(st, s, plan)):
                return self._other_claim(st, s, plan, rng)
        if plan.claim == "fact" and plan.reference and plan.fact_ids:
            f = plan.fact_ids[0]
            if f in s.shared_facts:
                plan.ref_by = s.id
                # "como eu disse" only when restating after someone disputed it (0.5.1); otherwise
                # a plain short restatement
                plan.restate = self.disputed(st, s, f)
                # 0.5.2: a student announces a fact once; they restate it only when directly asked or
                # disputed, and at most once more
                said = sum(1 for m in st.messages if m.speaker == s.id and m.plan is not None and f in m.plan.fact_ids)
                if said >= 2 or not (plan.restate or self._direct(st, s, plan)):
                    if plan.addresses_misconception and plan.anchor_turn is not None:
                        return self._press_instead(plan)
                    return self._carry(plan, self._alternative(st, s, plan.kc, rng))
            else:
                src = credit_source(task, st, f, s.id)     # 0.5.3: a statement, never a circular credit
                plan.ref_by = src.speaker if src is not None else None
        if self._needs_intent(plan) and self._no_new_subject:
            return self._carry(plan, ContentPlan(act="idk", kc=kc))   # continuation: no new confusion
        if self._needs_intent(plan):
            opts = intents.allowed(st, s.id, plan.act, self.cfg)
            if opts:
                plan.intent = rng.choice(opts)
                return plan
            alt = self._alternative(st, s, kc, rng, exclude=("interface", "question"))
            if final and alt.act in REACTION_ACTS and self.is_self_reaction(st, s.id, alt):
                alt = self._off_task(st, s, rng)
            return self._carry(plan, alt)
        return plan

    def _resolve_intent(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan, rng: RNG,
                        final: bool = False) -> ContentPlan:
        return self._enforce(st, s, kc, plan, rng, final)

    @staticmethod
    def _needs_intent(plan: ContentPlan) -> bool:
        return not plan.intent and not plan.nudge and not plan.step_q and not plan.tag and not plan.step_move and (
            plan.act == "interface" or (
            plan.act == "question" and not plan.asked_fact and not plan.follow_up))

    @staticmethod
    def _carry(old: ContentPlan, new: ContentPlan) -> ContentPlan:
        new.replies_to_external = old.replies_to_external
        if old.replies_to_external and new.addressee is None:
            new.addressee = old.addressee
        return new

    @staticmethod
    def _press_instead(plan: ContentPlan) -> ContentPlan:
        """A challenge with nothing (new) that refutes the belief: ask the holder for their reason."""
        return ContentPlan(act="press", kc=plan.kc, anchor_turn=plan.anchor_turn, addressee=plan.addressee,
                           family="press_for_reasoning", addresses_misconception=plan.addresses_misconception,
                           replies_to_external=plan.replies_to_external)

    def _coherent(self, st: SessionState, s: StudentState, plan: ContentPlan) -> ContentPlan:
        """0.5.2 code rule: a disagreement only when the speaker opposes a claim someone else made (a belief
        voiced by another chat member that the speaker does not hold); a challenge carries content that
        refutes the belief it addresses (else a press for the holder's reason)."""
        from . import coherence
        if plan.closing is not None or plan.draft is not None:
            return plan
        if plan.addresses_misconception and not coherence.refutes(self.task, plan):
            st.slippage_log.append({"type": "coherence_gate", "msg": len(st.messages), "speaker": s.id,
                                    "reason": "challenge_not_refuting", "act": plan.act, "claim": plan.claim,
                                    "fact_ids": list(plan.fact_ids), "claim_idx": plan.claim_idx})
            if plan.anchor_turn is not None:
                return self._press_instead(plan)
            plan.addresses_misconception, plan.family = None, None
        if plan.act == "disagree" and not coherence.opposes(st, s.id, plan):
            st.slippage_log.append({"type": "coherence_gate", "msg": len(st.messages), "speaker": s.id,
                                    "reason": "disagree_without_opposition", "claim": plan.claim})
            plan.act = "explain" if plan.anchor_turn is not None else "on_task_claim"
            if plan.claim in ("none", "hedge") and plan.anchor_turn is None:
                plan.act = "question" if plan.claim == "none" else "on_task_claim"
        return plan

    def _downgrade(self, st: SessionState, s: StudentState, plan: ContentPlan, rng: RNG) -> ContentPlan:
        """Nothing (new) to assert: a non-claim act drawn by the act priors instead."""
        return self._carry(plan, self._alternative(st, s, plan.kc, rng))

    def _direct(self, st: SessionState, s: StudentState, plan: ContentPlan) -> bool:
        """Answering an address, a question / challenge, or challenging a misconception."""
        if plan.replies_to_external or (plan.act == "disagree" and plan.addresses_misconception):
            return True
        if plan.claim == "misconception" and plan.misconception_id and self.may_voice(st, s, plan.misconception_id):
            return True  # challenged since the last voicing: defending it answers the challenge
        if plan.anchor_turn is not None and plan.anchor_turn < len(st.messages):
            am = st.messages[plan.anchor_turn]
            if am.speaker != s.id and am.plan is not None and (
                    am.plan.act in ("question", "press", "disagree") or am.plan.addresses_misconception):
                return True
        return False

    def _other_claim(self, st: SessionState, s: StudentState, plan: ContentPlan, rng: RNG) -> ContentPlan:
        """A capped re-statement becomes another claim the speaker has not made yet, else a non-claim act."""
        task = self.task
        kc = plan.kc
        if plan.addresses_misconception and plan.anchor_turn is not None:
            return self._press_instead(plan)     # a challenge never swaps in unrelated content (0.5.2)
        if kc is not None and plan.claim in ("correct", "fact"):
            for i in range(len(task.kc(kc).correct_claims)):
                if plan.claim == "correct" and plan.family != "counterexample" and claim_voiceable(task, s, kc, i, st):
                    cand = plan.model_copy(update={"claim_idx": i})
                    if not own_statements(task, st, s.id, cand):
                        return cand
            for f in self.fresh_facts(st, s, kc):
                cand = plan.model_copy(update={"claim": "fact", "fact_ids": [f], "reference": False,
                                               "claim_idx": None, "is_correct": True,
                                               "family": "evidence_from_material" if plan.family else None})
                if not own_statements(task, st, s.id, cand):
                    return cand
        return self._downgrade(st, s, plan, rng)

    # ------------------------------------------- no self-agreement
    @staticmethod
    def is_self_reaction(st: SessionState, sid: str, plan: ContentPlan) -> bool:
        """An agree / backchannel ("true", "makes sense", "lol") is a reaction to SOMEONE ELSE's
        recent message: never right after the speaker's own previous message, never when the last
        2 turns are both the speaker's (unless it answers an address that came after them), and only
        if someone else spoke in the last 3 turns."""
        if plan.act not in REACTION_ACTS:
            return False
        if st.messages and st.messages[-1].speaker == sid:
            return True
        last = st.recent_turns(3)
        if not any(m.speaker != sid for m in last):
            return True
        return (not plan.replies_to_external and len(last) >= 2 and all(m.speaker == sid for m in last[-2:]))

    def _no_self_reaction(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan,
                          rng: RNG) -> ContentPlan:
        if not self.is_self_reaction(st, s.id, plan):
            return plan
        alt = self._alternative(st, s, kc, rng, exclude=("backchannel",) if kc is None else ("backchannel", "off_task"))
        if alt.act in REACTION_ACTS:
            alt = self._off_task(st, s, rng)
        return self._carry(plan, alt)

    def current_kc(self, st: SessionState) -> str | None:
        if st.task_complete:
            return None
        return self.task.gold_for(self.task.steps[st.step_idx].id).kcs[0]

    def peers(self, st: SessionState, sid: str) -> list[str]:
        """Who `sid` may address: the other students and the external participants who have spoken."""
        return [x for x in st.roster + st.spoken_externals() if x != sid]

    # ------------------------------------------------------------ free acts
    def _free_act(self, st: SessionState, s: StudentState, kc: str | None, rng: RNG) -> ContentPlan:
        p_off = self._offtask_p(st, s)
        if rng.random() < min(0.9, p_off):
            return self._off_task(st, s, rng)
        w = dict(self.cfg.act_prior)
        w["interface"] = w.get("interface", 0.0) * s.traits.interface_confusion / INTERFACE_MEAN * 0.7
        if kc is None:
            for a in ("on_task_claim", "explain", "disagree", "press", "question", "idk"):
                w[a] = 0.0
        act = rng.weighted(list(w), list(w.values()))
        if act == "on_task_claim":
            return self._claim(st, s, kc, rng)
        if act == "explain":
            if s.kcs[kc].p >= self.th.can_state and s.active_misconception(kc) is None:
                return ContentPlan(act="explain", kc=kc, claim="correct", is_correct=True)
            return self._claim(st, s, kc, rng)
        if act in ("disagree", "press"):
            plan = self._challenge(st, s, rng, p=1.0, prefer=act)
            return plan or self._claim(st, s, kc, rng)
        if act == "agree":
            anchor = self._agreeable(st, s)
            if anchor is None:
                return self._alternative(st, s, kc, rng, exclude=("off_task",))
            return ContentPlan(act="agree", anchor_turn=anchor.id, kc=anchor.plan.kc)
        if act == "question":
            return self._question(st, s, kc, rng)
        if act == "interface":
            lo, hi = self.cfg.interface_block_s
            st.interface_until[s.id] = st.t + rng.uniform(lo, hi)
            return ContentPlan(act="interface")
        if act == "idk":
            return ContentPlan(act="idk", kc=kc)
        return ContentPlan(act="backchannel")

    # ------------------------------------------------------- questions
    def _question(self, st: SessionState, s: StudentState, kc: str | None, rng: RNG,
                  addressee: str | None = None) -> ContentPlan:
        """A content question; a content question is asked once per session (then short follow-ups
        or a different act)."""
        if addressee is not None and not self.ping_ok(st, s.id, addressee):
            addressee = None          # 0.5.2 ping caps (one per classmate, never a silent lurker)
        missing = [f.id for f in self.task.facts if f.kc == kc and self.askable(st, s, f.id)]
        if not (missing and rng.bernoulli(0.7)):
            return ContentPlan(act="question", kc=kc, addressee=addressee)
        recent = st.recent_turns(REASK_WINDOW)
        asked = {m.plan.asked_fact: m for m in recent if m.plan is not None and m.plan.act == "question"
                 and m.plan.asked_fact and not m.plan.follow_up}
        followed = {m.plan.asked_fact for m in recent if m.plan is not None and m.plan.follow_up}
        ever = set(asked) | {m.plan.asked_fact for m in st.messages if m.plan is not None
                             and m.plan.act == "question" and m.plan.asked_fact}
        fresh = [f for f in missing if f not in ever]
        if fresh:
            return ContentPlan(act="question", kc=kc, asked_fact=rng.choice(fresh), addressee=addressee)
        missing = [f for f in missing if f in asked]
        if not missing:
            return self._alternative(st, s, kc, rng, exclude=("question",))
        f = rng.choice(missing)
        if f not in followed and rng.bernoulli(P_FOLLOW_UP):
            to = addressee if addressee is not None and self.ping_ok(st, s.id, addressee) else None
            if to is None and rng.bernoulli(0.5):
                to = self.ping_target(st, s.id, f, rng)
            return ContentPlan(act="question", kc=kc, asked_fact=f, follow_up=True,
                               anchor_turn=asked[f].id, addressee=to)
        return self._alternative(st, s, kc, rng, exclude=("question",))

    def askable(self, st: SessionState, s: StudentState, fid: str) -> bool:
        """0.5.3: a content question about card fact `fid` only if its answer is still open for the asker: they do
        not hold it, nobody stated it in the chat yet, and they did not themselves just assert its answer -- a
        correct claim of its KC, a card fact of the same key idea, or the belief it refutes ("acho que as enzimas
        se gastam" -> never "enzima acaba quando ela trabalha?")."""
        from .content import message_facts
        task = self.task
        if fid in s.known_facts:
            return False
        if any(m.plan is not None and m.plan.act != "question" and fid in message_facts(task, m) for m in st.messages):
            return False
        fk = task.fact(fid).kc
        same_idea = {g for ki in task.all_key_ideas() if fid in ki.facts for g in ki.facts}
        for m in st.messages:
            p = m.plan
            if m.speaker != s.id or p is None or p.closing or p.draft:
                continue
            if p.claim == "correct" and p.kc == fk and p.is_correct is not False:
                return False
            if p.claim == "fact" and set(p.fact_ids) & same_idea:
                return False
            if p.claim == "misconception" and p.misconception_id and fid in task.refuting_facts(p.misconception_id):
                return False
        return True

    # ------------------------------------------------------- 0.5.2 name pings
    def ping_cap(self, st: SessionState) -> int:
        """Peer name pings per session ("julia vc sabe?"): 3 for groups up to 4, scaled by size above."""
        n = len(st.roster)
        return self.cfg.ping_cap_base if n <= 4 else int(self.cfg.ping_cap_base * n / 4)

    @staticmethod
    def pings(st: SessionState) -> list[Message]:
        return [m for m in st.messages if m.kind == "student" and m.plan is not None and m.plan.act == "question"
                and m.plan.addressee in st.roster and not m.plan.tag]

    def ping_ok(self, st: SessionState, sid: str, to: str) -> bool:
        """A classmate may be pinged by name: not oneself, at most once per target and ping_cap per
        session, and only someone who has written at least twice (never a silent lurker). External
        participants who wrote are not capped here."""
        if to == sid:
            return False
        if to not in st.roster:
            return to in st.spoken_externals()
        done = self.pings(st)
        if len(done) >= self.ping_cap(st) or any(m.plan.addressee == to for m in done):
            return False
        return sum(1 for m in st.messages if m.speaker == to) >= self.cfg.ping_min_messages

    def ping_target(self, st: SessionState, sid: str, fid: str | None, rng: RNG,
                    holders_only: bool = False) -> str | None:
        """Whom to ask by name about `fid`: a classmate who holds it (preferred) and may be pinged."""
        ok = [x for x in st.roster if self.ping_ok(st, sid, x)]
        holders = [x for x in ok if fid is not None and fid in st.student(x).known_facts]
        if holders:
            return rng.choice(holders)
        if holders_only or not ok:
            return None
        return rng.choice(ok)

    def _respond_to_peer(self, st: SessionState, s: StudentState, oa, kc: str | None, rng: RNG) -> ContentPlan | None:
        """A classmate asked this student by name ("lara vc sabe?") or pressed them for a reason ("pq vc
        acha isso?"): they answer (0.5.2) -- a press with p_answer_press (by engagement), a ping at the
        pilot reply rate; else the address lapses (someone who holds it may still answer the question)."""
        if not (0 <= oa.msg_id < len(st.messages)):
            return None
        am = st.messages[oa.msg_id]
        p = am.plan
        if p is None:
            return None
        if p.act == "press" and p.addresses_misconception:
            return self._answer_press(st, s, am, rng)
        if not rng.bernoulli(self.cfg.p_reply_when_addressed):
            return None
        base = dict(anchor_turn=am.id, addressee=am.speaker, answers_ping=True)
        if p.asked_fact and p.asked_fact in s.known_facts:
            f = p.asked_fact
            return ContentPlan(act="explain", kc=self.task.fact(f).kc, claim="fact", fact_ids=[f], is_correct=True,
                               reference=not self.fact_fresh(st, s, f), **base)
        akc = p.kc or kc
        if akc is not None and s.kcs[akc].can_state and s.active_misconception(akc) is None:
            return ContentPlan(act="explain", kc=akc, claim="correct", is_correct=True, **base)
        return ContentPlan(act="idk", kc=akc, unsure=True, **base)    # "no meu nao tem isso"

    def _answer_press(self, st: SessionState, s: StudentState, am: Message, rng: RNG) -> ContentPlan | None:
        """Asked why they hold a belief: the holder gives a reason from the belief's rationale (`because`),
        or, with no push-back left, gives in; a student who no longer holds it says they are not sure."""
        mid = am.plan.addresses_misconception
        if not rng.bernoulli(min(1.0, self.cfg.p_answer_press * (0.6 + 0.4 * s.affect.engagement))):
            return None
        base = dict(anchor_turn=am.id, addressee=am.speaker, answers_ping=True)
        m = s.misconception(mid)
        if m is not None and m.active:
            prior = len(self.voicings(st, s.id, mid))
            if prior <= self.cfg.max_pushbacks:
                return ContentPlan(act="explain", kc=m.kc, claim="misconception", misconception_id=mid,
                                   is_correct=False, pushback=prior > 0, rationale=True, **base)
        # no push-back left, or no longer holding it: gives in only to a refuting move (quoting it, 0.5.3);
        # a press alone is no reason -- they stay unsure
        ref = None
        if mid in self.open_beliefs(st, s.id) and self.may_concede(st, s, mid):  # a refuting move in the window
            ref = self.refuting_move(st, s.id, mid, since=self.voicings(st, s.id, mid)[0].id)
        if ref is not None:
            return self.concession_plan(st, s, mid, ref, answers_ping=True)
        # a reply that keeps their stance open ("sei la, foi so oq eu pensei"): an answer, not a filler (0.5.3)
        return ContentPlan(act="explain", unsure=True, **base)

    def _alternative(self, st: SessionState, s: StudentState, kc: str | None, rng: RNG,
                     exclude: tuple[str, ...] = ()) -> ContentPlan:
        """A non-claim act drawn by the act priors (question / ask-peer / backchannel / interface /
        off-task); used when a student has nothing of their own to assert."""
        p = self.cfg.act_prior
        off_w = sum(p.values()) * self.cfg.p_off_base / (1 - self.cfg.p_off_base)
        # 0.5.1: scaled by the session's off-task target (default .15), so the target also steers the
        # turns that have nothing else to say
        r = float(st.context.get("offtask_rate", self.cfg.offtask_rate))
        off_w *= min(3.0, r / 0.15)
        w = {"question": p["question"] * 2 / 3, "ask_peer": p["question"] / 3, "backchannel": p["backchannel"],
             "interface": p["interface"] * s.traits.interface_confusion / INTERFACE_MEAN,
             "off_task": off_w * s.traits.off_task / OFF_TASK_MEAN}
        if self._no_new_subject:
            exclude = tuple(exclude) + ("interface", "off_task", "backchannel")
        for a in exclude:
            w.pop(a, None)
            if a == "question":
                w.pop("ask_peer", None)
        if not w:
            return ContentPlan(act="idk", kc=kc)
        act = rng.weighted(list(w), list(w.values()))
        if act == "off_task":
            return self._off_task(st, s, rng)
        if act == "interface":
            lo, hi = self.cfg.interface_block_s
            st.interface_until[s.id] = st.t + rng.uniform(lo, hi)
            return ContentPlan(act="interface")
        if act in ("question", "ask_peer") and kc is not None:
            peers = self.peers(st, s.id)
            to = rng.choice(peers) if act == "ask_peer" and peers else None
            return self._question(st, s, kc, rng, addressee=to)
        return ContentPlan(act="backchannel")

    def _claim(self, st: SessionState, s: StudentState, kc: str, rng: RNG) -> ContentPlan:
        m = s.active_misconception(kc)
        if m is not None and self.may_voice(st, s, m.id) and rng.bernoulli(m.strength):
            prior = len(self.voicings(st, s.id, m.id))
            if prior <= self.cfg.max_pushbacks:     # a re-voicing is always a push-back (never a 3rd time)
                return ContentPlan(act="on_task_claim", kc=kc, claim="misconception",
                                   misconception_id=m.id, is_correct=False, pushback=prior > 0)
        # content-level dedupe: only facts nobody stated this session are announced
        unshared = self.fresh_facts(st, s, kc)
        if unshared and rng.bernoulli(0.6):
            return ContentPlan(act="on_task_claim", kc=kc, claim="fact", fact_ids=[unshared[0]],
                               is_correct=True)
        if m is None and s.kcs[kc].p >= self.th.can_state and rng.bernoulli(0.8):
            return ContentPlan(act="on_task_claim", kc=kc, claim="correct", is_correct=True)
        known = unshared
        if known and m is None and rng.bernoulli(0.5):
            return ContentPlan(act="on_task_claim", kc=kc, claim="fact", fact_ids=[rng.choice(known)],
                               is_correct=True)
        if m is None:
            # no new fact to voice and no relevant active misconception on this step -> no
            # content-free "idk maybe" claim; a question / ask-peer / backchannel / interface /
            # off-task act drawn by the act priors instead
            return self._alternative(st, s, kc, rng)
        if not self.may_voice(st, s, m.id):  # belief already voiced, unchallenged
            return self._alternative(st, s, kc, rng)
        return ContentPlan(act="on_task_claim", kc=kc, claim="hedge", is_correct=None)

    # --------------------------------------------------- peer interaction
    def _can_correct(self, s: StudentState, kc: str) -> tuple[bool, list[str]]:
        facts = [f for f in s.known_facts if self.task.fact(f).kc == kc]
        ok = s.active_misconception(kc) is None and (s.kcs[kc].p >= self.th.can_state or bool(facts))
        return ok, facts

    def _challenge_plan(self, st: SessionState, s: StudentState, m: Message, rng: RNG,
                        prefer: str | None = None) -> ContentPlan | None:
        """A challenge of the belief voiced in `m`, built from content that REFUTES that belief (0.5.2):
        a refuting card fact the speaker holds (announced if nobody said it yet; pointed back to if a
        classmate said it; never re-recited if the speaker said it), a correct claim that refutes it
        (task.refuting_claims), the belief's counterexample, or a press for the holder's reason. Never an
        unrelated true fact about the same KC."""
        kc = m.plan.kc
        ok, _facts = self._can_correct(s, kc)
        if not ok:
            return None
        task = self.task
        mid = m.plan.misconception_id
        spec = task.misconception(mid)
        refuting = set(task.refuting_facts(mid))
        facts = [f for f in s.known_facts if f in refuting]
        fresh = [f for f in facts if self.fact_fresh(st, s, f)]
        by_peer = [f for f in facts if f not in s.shared_facts and not self.fact_fresh(st, s, f)]
        claims = []
        if s.kcs[kc].p >= self.th.can_state:
            claims = [i for i in task.refuting_claims(mid) if claim_voiceable(task, s, kc, i, st)
                      and not own_statements(task, st, s.id, ContentPlan(act="explain", kc=kc, claim="correct",
                                                                           claim_idx=i))]
        options = []
        if "peer_explanation" in spec.flip_by and claims:
            options.append(("peer_explanation", 3.0))
        if fresh or by_peer:
            options.append(("evidence_from_material", 2.0))
        if spec.counterexample and not any(x.speaker == s.id and x.plan is not None and x.plan.family == "counterexample"
                                           and x.plan.addresses_misconception == mid for x in st.messages):
            options.append(("counterexample", 1.0))
        if not (m.plan.pushback or m.plan.rationale):     # a push-back already gave its reason: no "mas pq?"
            options.append(("press_for_reasoning", 1.5 if prefer == "press" else 0.7))
        if not options:
            return None
        fam = rng.weighted([o[0] for o in options], [o[1] for o in options])
        base = dict(kc=kc, anchor_turn=m.id, addressee=m.speaker, family=fam, addresses_misconception=mid)
        if fam == "press_for_reasoning":
            return ContentPlan(act="press", **base)
        if fam == "evidence_from_material":
            # a fact already said is only pointed back to ("like ana said ..."), never re-announced
            f = rng.choice(fresh) if fresh else rng.choice(by_peer)
            return ContentPlan(act="disagree", claim="fact", fact_ids=[f], is_correct=True,
                               reference=not fresh, **base)
        if fam == "counterexample":
            return ContentPlan(act="disagree", claim="correct", is_correct=True, **base)
        return ContentPlan(act="disagree" if rng.bernoulli(0.5) else "explain", claim="correct", is_correct=True,
                           claim_idx=rng.choice(claims), **base)

    def _challenge(self, st: SessionState, s: StudentState, rng: RNG, p: float,
                   prefer: str | None = None) -> ContentPlan | None:
        """Correction of a misconception voiced in the last `anchor_window` turns (by a peer or an
        external participant; addressed to whoever voiced it)."""
        for m in reversed(st.recent_turns(self.th.anchor_window)):
            if m.speaker == s.id or m.plan is None or m.plan.claim != "misconception":
                continue
            if any(x.speaker == s.id and x.plan is not None and x.plan.anchor_turn == m.id
                   for x in st.messages[m.id + 1:]):
                continue      # one challenge per voicing and challenger (0.5.1: no "mas pq?" x3)
            ok, _ = self._can_correct(s, m.plan.kc)
            if not ok or not rng.bernoulli(p):
                return None
            return self._challenge_plan(st, s, m, rng, prefer)
        return None

    def _answer_peer_question(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan | None:
        """Responsiveness: a student who can answer a pending content question (holds the asked fact /
        can state the KC) answers it with probability modulated by engagement (higher if the question
        named them); generic questions keep the old rule. The asker may be a peer or an external
        participant."""
        eng = 0.5 + 0.5 * s.affect.engagement
        for q in reversed(st.pending_questions):
            if q.asker == s.id:
                continue
            qm = st.messages[q.msg_id]
            if qm.plan is None:
                continue
            how = can_answer(st, s.id, qm)
            if how is not None:
                p = self.cfg.p_answer_named if qm.plan.addressee == s.id else self.cfg.p_answer_holder
                if not rng.bernoulli(p * eng):
                    continue
                f = qm.plan.asked_fact
                if how == "fact":  # answering a pending question about it: a short reference if already said
                    return ContentPlan(act="explain", kc=self.task.fact(f).kc, claim="fact", fact_ids=[f],
                                       is_correct=True, anchor_turn=q.msg_id, addressee=q.asker,
                                       reference=not self.fact_fresh(st, s, f))
                return ContentPlan(act="explain", kc=qm.plan.kc, claim="correct", is_correct=True,
                                   anchor_turn=q.msg_id, addressee=q.asker)
            if qm.plan.asked_fact is None and qm.plan.kc and rng.bernoulli(0.3):
                return self._substantive(st, s, qm.plan.kc, rng, anchor=q.msg_id, direct=True)
            return None
        return None

    def _agreeable(self, st: SessionState, s: StudentState) -> Message | None:
        """0.5.3: the line an "agree" ("concordo", "é isso mesmo") okays -- the last epistemic line by someone else,
        except a belief the speaker does not hold, or a press / challenge aimed at the speaker's own belief."""
        for m in reversed(st.recent_turns(self.th.anchor_window)):
            p = m.plan
            if m.speaker == s.id or p is None or p.act not in EPISTEMIC_ACTS:
                continue
            if p.claim == "misconception" and p.misconception_id:
                mine = s.misconception(p.misconception_id)
                if mine is None or not mine.active:
                    return None
                return m
            if p.addresses_misconception and p.addressee == s.id:
                return None
            if p.act == "press":
                return None
            return m
        return None

    def _last_epistemic(self, st: SessionState, exclude: str | None = None) -> Message | None:
        for m in reversed(st.recent_turns(self.th.anchor_window)):
            if m.plan and m.plan.act in EPISTEMIC_ACTS and m.speaker != exclude:
                return m
        return None

    # ----------------------------------------------- reply to an address (any external participant)
    def _respond_to_address(self, st, s: StudentState, oa, kc, rng: RNG) -> ContentPlan | None:
        if not rng.bernoulli(self.cfg.p_reply_when_addressed):
            return None  # ignores the address (pilot: 5/13)
        am = st.messages[oa.msg_id] if oa.msg_id < len(st.messages) else None
        anchor = oa.anchor_turn if oa.anchor_turn is not None else oa.msg_id
        akc = am.plan.kc if am is not None and am.plan is not None and am.plan.kc else kc
        if not rng.bernoulli(self.cfg.p_substantive_reply) or akc is None:
            act = rng.weighted(["idk", "agree", "backchannel", "refusal"], [3, 2, 1, 1])
            plan = ContentPlan(act=act, kc=akc, anchor_turn=oa.msg_id)
        elif am is not None and am.plan is not None and am.plan.act == "question" and can_answer(st, s.id, am):
            f = am.plan.asked_fact
            if can_answer(st, s.id, am) == "fact":
                plan = ContentPlan(act="explain", kc=self.task.fact(f).kc, claim="fact", fact_ids=[f],
                                   is_correct=True, anchor_turn=oa.msg_id, reference=not self.fact_fresh(st, s, f))
            else:
                plan = ContentPlan(act="explain", kc=am.plan.kc, claim="correct", is_correct=True,
                                   anchor_turn=oa.msg_id)
        else:
            plan = self._substantive(st, s, akc, rng, anchor=anchor, direct=True)
        plan.substantive = plan.act not in CONTENT_FREE_ACTS
        plan.replies_to_external = True
        plan.addressee = oa.source
        return plan

    def _substantive(self, st, s: StudentState, kc: str | None, rng: RNG, anchor: int | None,
                     direct: bool = True) -> ContentPlan:
        """`direct`: answering a question or prompt -- they may re-voice a belief or point back to a
        fact they already said; otherwise the dedupe rules apply."""
        if kc is None:
            return ContentPlan(act="backchannel")
        m = s.active_misconception(kc)
        prior = len(self.voicings(st, s.id, m.id)) if m is not None else 0
        if m is not None and (direct or self.may_voice(st, s, m.id)) and prior <= self.cfg.max_pushbacks:
            # defends / re-voices the wrong belief (a re-voicing is a push-back)
            return ContentPlan(act="explain", kc=kc, claim="misconception", misconception_id=m.id,
                               is_correct=False, anchor_turn=anchor, pushback=prior > 0)
        if m is not None:  # already voiced, not challenged since: no re-voicing
            return self._alternative(st, s, kc, rng)
        if s.kcs[kc].p >= self.th.can_state:
            return ContentPlan(act="explain", kc=kc, claim="correct", is_correct=True, anchor_turn=anchor)
        fresh = self.fresh_facts(st, s, kc)
        facts = fresh or ([f for f in s.known_facts if self.task.fact(f).kc == kc] if direct else [])
        if facts:
            return ContentPlan(act="explain", kc=kc, claim="fact", fact_ids=[rng.choice(facts)],
                               is_correct=True, anchor_turn=anchor, reference=not fresh)
        if not direct and s.known_facts:  # had something, already said: no content-free hedge
            return self._alternative(st, s, kc, rng)
        return ContentPlan(act="on_task_claim", kc=kc, claim="hedge", anchor_turn=anchor)

    # ------------------------------------------------- content dedupe
    @staticmethod
    def fact_fresh(st: SessionState, s: StudentState, fid: str) -> bool:
        """A fact may be ANNOUNCED only if this student never stated it and nobody stated it this
        session (otherwise at most a short back-reference)."""
        if fid in s.shared_facts:
            return False
        return not any(m.plan is not None and fid in m.plan.fact_ids for m in st.messages)

    def fresh_facts(self, st: SessionState, s: StudentState, kc: str) -> list[str]:
        return [f for f in s.known_facts if self.task.fact(f).kc == kc and self.fact_fresh(st, s, f)]

    @staticmethod
    def may_voice(st: SessionState, s: StudentState, mid: str) -> bool:
        """A misconception the student already voiced is re-voiced only after it was challenged
        (a later turn addressing it, at them or at the group)."""
        last = None
        for m in reversed(st.messages):
            if (m.speaker == s.id and m.plan is not None
                    and m.plan.claim == "misconception" and m.plan.misconception_id == mid):
                last = m.id
                break
        if last is None:
            return True
        for m in st.messages[last + 1:]:
            if (m.speaker != s.id and m.plan is not None and m.plan.addresses_misconception == mid
                    and m.plan.addressee in (s.id, None)):
                return True
        return False

    @staticmethod
    def voicings(st: SessionState, sid: str, mid: str) -> list[Message]:
        return [m for m in st.messages if m.speaker == sid and m.plan is not None
                and m.plan.claim == "misconception" and m.plan.misconception_id == mid]

    # ------------------------------------------------------------ 0.5.0 gates
    def _misconception_gate(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan,
                            rng: RNG) -> ContentPlan:
        """A student voices a misconception once; after it was challenged they either flip (rule 4),
        push back ONCE with a reason (plan.pushback), or go quiet on it. Never a third time."""
        if plan.claim != "misconception" or not plan.misconception_id or plan.rationale:
            return plan          # (a reason given when asked "pq?" is the answer to that question, 0.5.2)
        prior = self.voicings(st, s.id, plan.misconception_id)
        if not prior:
            return plan
        if len(prior) <= self.cfg.max_pushbacks and self.may_voice(st, s, plan.misconception_id) and \
                rng.bernoulli(self.cfg.p_pushback):
            plan.pushback = True
            return plan
        alt = self._alternative(st, s, kc, rng, exclude=("off_task",))
        return self._resolve_intent(st, s, kc, self._carry(plan, alt), rng, final=True)

    def content_free_run(self, st: SessionState) -> int:
        """How many of the last student messages in a row carry no content (by plan or by text)."""
        run = 0
        for m in reversed(st.student_msgs()):
            if (m.plan is not None and is_content_free(m.plan) and not m.plan.opener) or realism.content_free_text(
                    m.text, self.task):
                run += 1
            else:
                break
        return run

    def filler_blocked(self, st: SessionState, s: StudentState, plan: ContentPlan) -> str | None:
        """Why a content-free turn may not happen now (None = allowed). Pilot aggregates: few idk
        lines, few backchannels, never more than 2 content-free messages in a row."""
        if not is_content_free(plan):
            return None
        cfg = self.cfg
        stu = st.student_msgs()
        n = len(stu) + 1
        last = st.messages[-1] if st.messages else None
        if last is not None and last.speaker == s.id:
            return "own_continuation"          # e.g. "how do i send it?" -> "no idea" by the same student
        run = self.content_free_run(st)
        if run >= cfg.max_content_free_run:
            return "content_free_run"
        if plan.replies_to_external or plan.act == "off_task" or plan.answers_ping:
            return None   # a short reply to an address is fine (pilot uptake); off-task has its own rate
        if plan.act in ("idk", "refusal"):
            idk = [m for m in stu if m.plan is not None and m.plan.act in ("idk", "refusal")]
            if len(idk) + 1 > max(1, math.floor(cfg.idk_share_cap * n)):
                return "idk_share"
            if sum(m.speaker == s.id for m in idk) >= cfg.idk_per_student_max:
                return "idk_per_student"
        if sum(1 for m in stu[-5:] if m.plan is not None and is_content_free(m.plan)) >= 3:
            return "content_free_window"       # at most 3 content-free lines in any 6 (0.5.2)
        cf = [m for m in stu if m.plan is not None and is_content_free(m.plan) and m.plan.act != "off_task"]
        if len(cf) + 1 > max(2, math.floor(cfg.content_free_share_cap * n)):
            return "content_free_share"
        return None

    def _filler_gate(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan,
                     rng: RNG) -> ContentPlan:
        why = self.filler_blocked(st, s, plan)
        if why is None:
            return plan
        alt = self._substitute(st, s, kc, rng)
        st.slippage_log.append({"type": "filler_gate", "msg": len(st.messages), "speaker": s.id,
                                "act": plan.act, "reason": why, "new_act": alt.act if alt is not None else None})
        if alt is None:
            return None   # nothing to say: the student stays quiet this time (the turn is skipped)
        alt = self._enforce(st, s, kc, self._carry(plan, alt), rng, final=True)
        alt = self._misconception_gate(st, s, kc, alt, rng)
        alt = self._finish(alt, s, rng, st)
        if self.filler_blocked(st, s, alt) is not None:
            return None
        return alt

    @staticmethod
    def is_content_free_plan(plan: ContentPlan) -> bool:
        return is_content_free(plan)

    def substitute(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan | None:
        """A content plan for a turn whose wording failed the final checks (under the usual gates), or None."""
        kc = self.current_kc(st)
        last = st.recent_turns(1)
        self._no_new_subject = bool(last and last[-1].speaker == s.id and last[-1].plan is not None
                                    and last[-1].plan.act != "greeting")
        try:
            alt = self._substitute(st, s, kc, rng)
            if alt is None:
                return None
            alt = self._realism_gate(st, s, kc, alt, rng)
            alt = self._enforce(st, s, kc, alt, rng, final=True)
            alt = self._misconception_gate(st, s, kc, alt, rng)
            if self.filler_blocked(st, s, alt) is not None:
                return None
            return self._finish(alt, s, rng, st)
        finally:
            self._no_new_subject = False

    def _substitute(self, st: SessionState, s: StudentState, kc: str | None, rng: RNG,
                    allow_offtask: bool = True) -> ContentPlan | None:
        """A turn with content instead of a blocked filler: a slated belief, a progress move, a claim, a
        content question, else an off-task remark if one is allowed. Under the information-pacing
        window (0.5.1) a new statement is skipped in favour of the next option."""
        if kc is not None:
            v = (self._slated_voice(st, s, kc, rng, urgent=True) or self._discuss_slate(st, s, rng, urgent=True)
                 or self._concede(st, s, kc, rng, urgent=True))
            if v is not None:
                return v
            d = self._draft_move(st, s, kc, rng, urgent=True)
            if d is not None:
                return d
            p = self._progress_move(st, s, kc, rng)
            if p is not None and not self.window_blocked(st, s, p):
                return p
            c = self._claim(st, s, kc, rng)
            if c.act in ("on_task_claim", "explain", "disagree", "press") and c.claim != "hedge" \
                    and not self.window_blocked(st, s, c):
                return c
            q = self._question(st, s, kc, rng)
            if q.act == "question" and (q.asked_fact or q.follow_up):
                return q
        if allow_offtask and self._offtask_ok(st, s)[0] and not self._no_new_subject:
            o = self._off_task(st, s, rng)
            if o.act == "off_task":
                return o
        return None

    def missing_ideas(self, st: SessionState):
        if st.task_complete:
            return []
        step = self.task.steps[st.step_idx]
        return [ki for ki in self.task.key_ideas(step.id) if ki.id not in st.ideas_voiced]

    def _progress_move(self, st: SessionState, s: StudentState, kc: str, rng: RNG) -> ContentPlan | None:
        """When the group stalls: challenge a voiced, unchallenged misconception blocking the step; share
        a held card fact of a missing key idea; state a missing idea's correct claim; ask about a
        missing fact (or ask a holder by name)."""
        task = self.task
        step = task.steps[st.step_idx]
        from .updater import StateUpdater
        blocking = StateUpdater(task, self.th, self.cfg).blocking_misconceptions(st, step.id)
        for b in blocking:
            if b["student"] == s.id:
                continue
            m = st.messages[b["voiced_msg"]]
            plan = self._challenge_plan(st, s, m, rng)
            if plan is not None:
                return plan
        from . import conflict
        for b in conflict.undiscussed(st, step.id):
            if b["student"] == s.id or b["voiced"] >= len(st.messages):
                continue
            plan = self._challenge_plan(st, s, st.messages[b["voiced"]], rng)
            if plan is not None:
                return plan
        missing = self.missing_ideas(st)
        held = [f for ki in missing for f in ki.facts if f in s.known_facts]
        held = list(dict.fromkeys(held))
        if held:
            f = rng.choice(held)
            return ContentPlan(act="on_task_claim", kc=task.fact(f).kc, claim="fact", fact_ids=[f], is_correct=True,
                               reference=not self.fact_fresh(st, s, f))
        for ki in missing:
            if ki.claims and s.kcs[ki.kc].p >= self.th.can_state and s.active_misconception(ki.kc) is None:
                ok = [i for i in ki.claims if claim_voiceable(task, s, ki.kc, i, st)]
                if ok:
                    return ContentPlan(act="on_task_claim", kc=ki.kc, claim="correct", claim_idx=rng.choice(ok),
                                       is_correct=True)
        lacking = [f for ki in missing for f in ki.facts if self.askable(st, s, f)]
        if lacking:
            asked = {m.plan.asked_fact for m in st.messages if m.plan is not None and m.plan.act == "question"
                     and m.plan.asked_fact}
            fresh = [f for f in lacking if f not in asked]
            if fresh:
                f = rng.choice(fresh)
                return ContentPlan(act="question", kc=task.fact(f).kc, asked_fact=f)
            f = rng.choice(lacking)
            first = next(m for m in st.messages if m.plan is not None and m.plan.asked_fact == f)
            recent = [m for m in st.messages[-6:] if m.plan is not None and m.plan.asked_fact == f]
            to = self.ping_target(st, s.id, f, rng, holders_only=True) if not recent else None
            if to is not None:
                return ContentPlan(act="question", kc=task.fact(f).kc, asked_fact=f, follow_up=True,
                                   anchor_turn=first.id, addressee=to)
        if not missing and StateUpdater(task, self.th, self.cfg).content_ready(st, step.id):
            return None       # only the pace's dwell time is left: no restating for its own sake (0.5.1)
        if not missing:
            c = self._claim(st, s, kc, rng)
            if c.act in ("on_task_claim", "explain") and c.claim in ("fact", "correct"):
                return c
            # every idea is on the table but the group has not converged on it yet: back up a
            # point someone else made ("yeah like ana said, ...")
            gold = set(task.gold_for(step.id).kcs)
            said = [f for f in s.known_facts if task.fact(f).kc in gold and f not in s.shared_facts
                    and not self.fact_fresh(st, s, f)]
            if said and s.active_misconception(task.fact(said[0]).kc) is None:
                f = rng.choice(said)
                return ContentPlan(act="explain", kc=task.fact(f).kc, claim="fact", fact_ids=[f], is_correct=True,
                                   reference=True)
        return None

    # ------------------------------------------------------------ 0.5.1 pacing / conflict
    @staticmethod
    def announces(plan: ContentPlan) -> bool:
        """A NEW content statement: a card fact or a correct claim stated (not pointed back to)."""
        return (plan.claim in ("fact", "correct") and not plan.reference and plan.closing is None
                and plan.act in ("on_task_claim", "explain", "disagree") and not plan.revoice)

    def window_blocked(self, st: SessionState, s: StudentState, plan: ContentPlan) -> bool:
        """Information pacing: at most one new statement per student per fact_window_s seconds or
        fact_window_turns student turns, unless they were asked directly (a question / challenge / an
        address) or are taking up a misconception."""
        if not self.announces(plan) or self._direct(st, s, plan):
            return False
        stu = st.student_msgs()
        last = next((m for m in reversed(stu) if m.speaker == s.id and m.plan is not None
                     and self.announces(m.plan)), None)
        if last is None:
            return False
        if st.t - last.t < self.cfg.fact_window_s:
            return True
        return any(m.id == last.id for m in stu[-self.cfg.fact_window_turns:])

    def _window_gate(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan,
                     rng: RNG) -> ContentPlan:
        if not self.window_blocked(st, s, plan):
            return plan
        st.slippage_log.append({"type": "pacing_gate", "msg": len(st.messages), "speaker": s.id,
                                "act": plan.act, "claim": plan.claim, "fact_ids": list(plan.fact_ids)})
        if kc is not None and rng.bernoulli(0.5):
            q = self._question(st, s, kc, rng)
            if q.act == "question" and (q.asked_fact or q.follow_up):
                return self._carry(plan, q)
        alt = self._alternative(st, s, kc, rng)
        if self.announces(alt):
            alt = ContentPlan(act="backchannel")
        return self._enforce(st, s, kc, self._carry(plan, alt), rng, final=True)

    def disputed(self, st: SessionState, s: StudentState, fid: str) -> bool:
        """Has someone disputed `s`'s statement of fact `fid` (a challenge / question at them, or a
        misconception voiced on its topic) since they made it?"""
        kc = self.task.fact(fid).kc
        first = next((m for m in st.messages if m.speaker == s.id and m.plan is not None and fid in m.plan.fact_ids),
                     None)
        if first is None:
            return False
        for m in st.messages[first.id + 1:]:
            p = m.plan
            if m.speaker == s.id or p is None:
                continue
            if p.claim == "misconception" and (p.kc == kc or p.misconception_id and
                                               self.task.misconception(p.misconception_id).kc == kc):
                return True
            if p.act in ("disagree", "press") and p.anchor_turn is not None and p.anchor_turn < len(st.messages):
                am = st.messages[p.anchor_turn]
                if am.speaker == s.id and am.plan is not None and fid in am.plan.fact_ids:
                    return True
        return False

    def _slated_voice(self, st: SessionState, s: StudentState, kc: str, rng: RNG,
                      urgent: bool = False) -> ContentPlan | None:
        """A belief slated to surface on this step (sim/conflict.py), voiced by its holder once the step
        is underway: with p_slated_voice per turn, or at once when the step only waits for it."""
        from . import conflict
        step = self.task.steps[st.step_idx].id
        mine = [sl for sl in conflict.pending(st, step) if sl["student"] == s.id]
        if not mine:
            return None
        since = sum(1 for m in st.student_msgs() if m.id > st.step_started_msg)
        if since < self.cfg.slated_voice_after_turns and not urgent:
            return None
        from .updater import StateUpdater
        waiting = StateUpdater(self.task, self.th, self.cfg)._content_ready(st, step)
        if not (urgent or waiting or stalled(st, self.cfg) or rng.bernoulli(self.cfg.p_slated_voice)):
            return None
        m = s.misconception(mine[0]["m"])
        return ContentPlan(act="on_task_claim", kc=m.kc, claim="misconception", misconception_id=m.id,
                           is_correct=False)

    def step_msgs(self, st: SessionState) -> list[Message]:
        return [m for m in st.messages[st.step_started_msg + 1:] if m.plan is not None]

    def _draft_move(self, st: SessionState, s: StudentState, kc: str, rng: RNG,
                    urgent: bool = False) -> ContentPlan | None:
        """Writing up the activity's answer once its content is on the table (0.5.1): someone proposes
        the hypothesis text ("vou colocar: eu acho que ..."), one or two others okay it or talk about
        the confidence bar, and the proposer says they typed it. Only while the step is open (the
        pace's dwell time); never new content."""
        from .updater import StateUpdater
        step = self.task.steps[st.step_idx]
        if not StateUpdater(self.task, self.th, self.cfg).content_ready(st, step.id):
            return None
        if any(s.active_misconception(k) is not None for k in self.task.gold_for(step.id).kcs):
            return None       # a student who still holds a wrong belief on it does not write up / okay the answer
        msgs = self.step_msgs(st)
        prop = next((m for m in msgs if m.plan.draft == "propose"), None)
        if prop is None:
            if st.messages and st.messages[-1].speaker == s.id:
                return None       # not as a third line in a row after one's own statement
            if not (urgent or stalled(st, self.cfg) or rng.bernoulli(0.5)):
                return None
            ideas = [k.id for k in self.task.key_ideas(step.id) if k.id in st.ideas_voiced]
            facts = [f for k in self.task.key_ideas(step.id) if k.id in ideas for f in k.facts]
            return ContentPlan(act="explain", kc=kc, draft="propose", ideas=ideas, fact_ids=facts, is_correct=True)
        replies = [m for m in msgs if m.id > prop.id and m.plan.draft in ("agree", "confidence")]
        typed = any(m.plan.draft == "typed" for m in msgs)
        if s.id == prop.speaker:
            if replies and not typed and (urgent or rng.bernoulli(0.6)):
                return ContentPlan(act="interface", draft="typed", anchor_turn=prop.id)
            return None
        if any(m.speaker == s.id for m in replies) or len(replies) >= 2 or st.messages[-1].speaker == s.id:
            return None
        if not (urgent or rng.bernoulli(0.7)):
            return None
        if not any(m.plan.draft == "confidence" for m in replies) and rng.bernoulli(0.5):
            return ContentPlan(act="interface", draft="confidence", kc=kc, anchor_turn=prop.id)
        return ContentPlan(act="agree", draft="agree", kc=kc, anchor_turn=prop.id, addressee=prop.speaker)

    def _concede(self, st: SessionState, s: StudentState, kc: str, rng: RNG,
                 urgent: bool = False) -> ContentPlan | None:
        """A holder who voiced a belief gives in ("ah, se ela acabasse o biscoito parava de ficar doce, faz
        sentido") only in answer to a REFUTING move -- a refuting fact / claim / the counterexample, addressed to
        them, to the group or to that belief -- in the last concede_window student turns (0.5.3; a press, a "mano
        kkk" or an unrelated claim is no reason), and the line quotes that reason. Active beliefs: after being
        challenged and with no push-back left (or two voicings); a belief the hidden state already dropped
        (a flip on a peer's turn) that was voiced and never given up in the chat: as soon as there is such a
        move. Under conflict="high" active beliefs persist: only after every push-back was spent, two explicit
        challenges and a stalled group, and not always."""
        if st.messages and st.messages[-1].speaker == s.id:
            return None
        high = self.cfg.conflict == "high"
        for mid in self.open_beliefs(st, s.id):
            m = s.misconception(mid)
            if m is None:
                continue
            prior = self.voicings(st, s.id, mid)
            ref = self.refuting_move(st, s.id, mid, since=prior[0].id)
            if ref is None and high and m.active and stalled(st, self.cfg):
                # conflict="high": a group stuck on the belief goes back to the argument made against it
                ref = self.refuting_move(st, s.id, mid, since=prior[0].id, window=10 ** 6)
            if ref is None:
                continue
            if not m.active:          # dropped in the hidden state, still standing in the chat
                if urgent or rng.bernoulli(0.6):
                    return self.concession_plan(st, s, mid, ref)
                continue
            if m.kc != kc:
                continue
            spent = len(prior) > self.cfg.max_pushbacks
            if high and not spent and stalled(st, self.cfg) and self.may_voice(st, s, mid) and rng.bernoulli(0.5):
                # conflict="high": a stuck group hears the holder defend the belief again (they stay dissenting)
                return ContentPlan(act="on_task_claim", kc=m.kc, claim="misconception", misconception_id=mid,
                                   is_correct=False, pushback=True)
            if high:
                n_chal = sum(1 for x in st.messages[prior[0].id + 1:] if x.speaker != s.id and x.plan is not None
                             and x.plan.addresses_misconception == mid)
                if not (spent and n_chal >= 2 and stalled(st, self.cfg) and rng.bernoulli(0.3)):
                    continue
            elif not (spent or len(prior) >= 2):
                continue
            if high or urgent or stalled(st, self.cfg) or rng.bernoulli(0.5):
                return self.concession_plan(st, s, mid, ref)
        return None

    # ------------------------------------------------------------ 0.5.3 stance changes
    def may_concede(self, st: SessionState, s: StudentState, mid: str) -> bool:
        """Under conflict="high" an ACTIVE belief is given up only after every push-back was spent and two explicit
        challenges (0.5.1); a belief the hidden state already dropped may always be given up in the chat."""
        m = s.misconception(mid)
        if m is None or not m.active or self.cfg.conflict != "high":
            return True
        prior = self.voicings(st, s.id, mid)
        n_chal = sum(1 for x in st.messages[prior[0].id + 1:] if x.speaker != s.id and x.plan is not None
                     and x.plan.addresses_misconception == mid) if prior else 0
        return len(prior) > self.cfg.max_pushbacks and n_chal >= 2

    def concession_plan(self, st: SessionState, s: StudentState, mid: str, ref: Message, **kw) -> ContentPlan:
        kc = self.task.misconception(mid).kc
        return ContentPlan(act="agree", kc=kc, concedes=mid, anchor_turn=ref.id, reason_turn=ref.id,
                           addressee=ref.speaker if ref.speaker in self.peers(st, s.id) else None, **kw)

    def refuting_move(self, st: SessionState, sid: str, mid: str, since: int | None = None,
                      window: int | None = None) -> Message | None:
        """The most recent message of the last `window` student turns (default cfg.concede_window), after
        `since`, by someone else, whose CONTENT refutes belief `mid` (a refuting card fact, a refuting correct
        claim, the counterexample; never a press), addressed to `sid`, to the group, or at that belief."""
        task = self.task
        window = self.cfg.concede_window if window is None else window
        stu = st.student_msgs()
        lo = stu[-window].id if len(stu) >= window else 0
        if since is not None:
            lo = max(lo, since + 1)
        for x in reversed(st.messages[lo:]):
            p = x.plan
            if x.speaker == sid or p is None:
                continue
            if not (p.addressee in (sid, None) or p.addresses_misconception == mid):
                continue
            if self.refutes_belief(p, mid):
                return x
        return None

    def refutes_belief(self, p: ContentPlan, mid: str) -> bool:
        task = self.task
        if p.act == "press" or p.family == "press_for_reasoning" or p.closing or p.draft or p.concedes:
            return False
        if p.act == "question" or p.claim not in ("fact", "correct") or p.is_correct is False:
            return False
        if p.addresses_misconception == mid:
            return True
        if p.claim == "fact" and set(p.fact_ids) & set(task.refuting_facts(mid)):
            return True
        m = task.misconception(mid)
        return p.claim == "correct" and p.kc == m.kc and p.claim_idx in task.refuting_claims(mid)

    def step_kcs(self, st: SessionState) -> set[str]:
        """KCs of the activity under discussion and of the next one (0.5.3)."""
        out = set()
        for i in (st.step_idx, st.step_idx + 1):
            if 0 <= i < len(self.task.steps):
                out |= set(self.task.gold_for(self.task.steps[i].id).kcs)
        return out

    def _burst_gate(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan,
                    rng: RNG) -> ContentPlan:
        """0.5.3 dialogue-state rules on a content plan: (a) one burst (a run of one's own messages) never mixes a
        claim with a belief on the same KC, nor a belief with a question about it; (b) a belief is voiced only
        for the activity under discussion or the next one, unless it answers a challenge of it."""
        if plan.closing is not None or plan.draft is not None or plan.concedes or plan.split_tail:
            return plan
        run = []
        for m in reversed(st.messages):
            if m.speaker != s.id:
                break
            run.append(m)
        task = self.task

        def kc_of(p):
            if p.claim == "misconception" and p.misconception_id:
                return task.misconception(p.misconception_id).kc
            if p.asked_fact:
                return task.fact(p.asked_fact).kc
            if p.claim == "fact" and p.fact_ids:
                return task.fact(p.fact_ids[0]).kc
            return p.kc
        why = None
        pk = kc_of(plan)
        mine_belief = plan.claim == "misconception" and plan.misconception_id
        mine_claim = plan.claim in ("fact", "correct") and plan.act in ("on_task_claim", "explain", "disagree")
        mine_q = plan.act == "question" and plan.asked_fact
        for m in run:
            p = m.plan
            if p is None or kc_of(p) != pk or pk is None:
                continue
            if mine_belief and p.claim in ("fact", "correct") and p.act in ("on_task_claim", "explain", "disagree"):
                why = "belief_after_own_claim"
            elif mine_belief and p.act == "question" and p.asked_fact:
                why = "belief_after_own_question"
            elif (mine_claim or mine_q) and p.claim == "misconception":
                why = "claim_or_question_after_own_belief"
        if why is None and mine_belief and pk not in self.step_kcs(st) and not (plan.rationale or plan.pushback):
            am = st.messages[plan.anchor_turn] if plan.anchor_turn is not None and plan.anchor_turn < len(st.messages) \
                else None
            if not (am is not None and am.plan is not None and am.plan.addresses_misconception == plan.misconception_id):
                why = "belief_off_step"
        if why is None:
            return plan
        st.slippage_log.append({"type": "burst_gate", "msg": len(st.messages), "speaker": s.id, "reason": why,
                                "act": plan.act, "claim": plan.claim})
        alt = self._alternative(st, s, kc, rng, exclude=("question",))
        return self._enforce(st, s, kc, self._carry(plan, alt), rng, final=True)

    def open_beliefs(self, st: SessionState, sid: str) -> list[str]:
        """Beliefs `sid` voiced in the chat and never gave up there (no concession line)."""
        out = []
        for m in st.messages:
            if m.speaker != sid or m.plan is None:
                continue
            if m.plan.claim == "misconception" and m.plan.misconception_id and m.plan.misconception_id not in out:
                out.append(m.plan.misconception_id)
            if m.plan.concedes and m.plan.concedes in out:
                out.remove(m.plan.concedes)
        return out

    def _stance_gate(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan,
                     rng: RNG) -> ContentPlan:
        """0.5.3: a student who voiced a belief and never gave it up in the chat does not switch sides silently:
        stating content that refutes it (or okaying a line that does) becomes a concession quoting the refuting
        move if there is one in the window, else a different, non-claim act."""
        if plan.closing is not None or plan.draft is not None or plan.concedes or plan.split_tail:
            return plan
        for mid in self.open_beliefs(st, s.id):
            if plan.claim == "misconception" and plan.misconception_id == mid:
                continue
            agree_ref = None
            if plan.act == "agree" and plan.anchor_turn is not None and 0 <= plan.anchor_turn < len(st.messages):
                am = st.messages[plan.anchor_turn]
                if am.plan is not None and self.refutes_belief(am.plan, mid):
                    agree_ref = am
            if not (agree_ref is not None or self.refutes_belief(plan, mid) or (
                    plan.act == "disagree" and plan.addresses_misconception == mid)):
                continue
            last = self.voicings(st, s.id, mid)[0]
            ref = agree_ref or self.refuting_move(st, s.id, mid, since=last.id)
            if ref is not None and (not self.may_concede(st, s, mid) or (st.messages and st.messages[-1].speaker == s.id)):
                ref = None
            st.slippage_log.append({"type": "stance_gate", "msg": len(st.messages), "speaker": s.id, "m": mid,
                                    "act": plan.act, "claim": plan.claim, "concede": ref is not None})
            if ref is not None:
                return self._carry(plan, self.concession_plan(st, s, mid, ref))
            alt = self._alternative(st, s, kc, rng, exclude=("off_task",))
            return self._enforce(st, s, kc, self._carry(plan, alt), rng, final=True)
        return plan

    def _discuss_slate(self, st: SessionState, s: StudentState, rng: RNG, urgent: bool = False) -> ContentPlan | None:
        """A belief slated to surface was voiced but nobody took it up yet: someone who can correct it
        does (whenever it was said in this step)."""
        from . import conflict
        step = self.task.steps[st.step_idx].id
        for b in conflict.undiscussed(st, step):
            if b["student"] == s.id or b["voiced"] >= len(st.messages):
                continue
            if not (urgent or rng.bernoulli(0.6)):
                return None
            return self._challenge_plan(st, s, st.messages[b["voiced"]], rng)
        return None

    def _forced_move(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan:
        """The group has been silent for max_silence_s: someone says something, content first."""
        kc = self.current_kc(st)
        self._no_new_subject = False
        alt = self._substitute(st, s, kc, rng, allow_offtask=False)
        if alt is not None:
            alt = self._enforce(st, s, kc, alt, rng, final=True)
            alt = self._misconception_gate(st, s, kc, alt, rng)
            if self.filler_blocked(st, s, alt) is None and not self.window_blocked(st, s, alt):
                st.slippage_log.append({"type": "silence_cap", "msg": len(st.messages), "speaker": s.id,
                                        "act": alt.act})
                return self._finish(alt, s, rng, st)
        return self.nudge_plan(st, s, rng)

    def nudge_plan(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan:
        """The activity question (once per step), else a short nudge to the group."""
        kc = self.current_kc(st)
        if kc is not None and not any(m.plan is not None and m.plan.step_q and m.id > st.step_started_msg
                                      for m in st.messages):
            return self._finish(ContentPlan(act="question", kc=kc, step_q=True), s, rng, st)
        return self._finish(ContentPlan(act="question", kc=kc, nudge=True), s, rng, st)

    # ------------------------------------------------------------ length
    def _finish(self, plan: ContentPlan, s: StudentState, rng: RNG, st: SessionState | None = None) -> ContentPlan:
        if st is not None:
            # 0.5.3 dialogue-state rules on every path (substitutes, forced moves and fallbacks included)
            kc = self.current_kc(st)
            plan = self._stance_gate(st, s, kc, plan, rng)
            plan = self._burst_gate(st, s, kc, plan, rng)
            plan = self._coherent(st, s, plan)
        if rng.random() < self.cfg.p_one_word:
            n = 1
        else:
            n = rng.lognormal(self.cfg.len_mu, self.cfg.len_sigma) * s.traits.verbosity
        plan.target_len = int(max(1, min(self.cfg.max_words, round(n))))
        # a content statement is a contribution with minimal framing, never a bare fragment
        if plan.closing == "propose":
            plan.target_len = max(plan.target_len, 16)
        elif plan.concedes:
            plan.target_len = min(max(plan.target_len, 3), 8)
        elif plan.brief or plan.tag:
            plan.target_len = min(plan.target_len, 3)
        elif plan.draft == "propose":
            plan.target_len = max(plan.target_len, 10)
        elif plan.draft is not None:
            plan.target_len = min(max(plan.target_len, 3), 8)
        elif plan.nudge:
            plan.target_len = min(max(plan.target_len, 2), 6)
        elif plan.step_q:
            plan.target_len = max(plan.target_len, 6)
        elif plan.claim in ("fact", "correct", "misconception") and not plan.reference:
            # student register: a content line is a short sentence, longer for wordier students but
            # rarely past ~20 words (0.5.1: no textbook-length lines)
            cap = int(round(min(self.cfg.content_max_words - 5, max(9.0, self.cfg.content_len_cap * s.traits.verbosity))))
            plan.target_len = min(max(plan.target_len, 7), cap)
        elif plan.act == "question" and plan.asked_fact and not plan.follow_up:
            plan.target_len = max(plan.target_len, 5)
        elif plan.answers_intent or plan.closing in ("agree", "adjust"):
            plan.target_len = max(plan.target_len, 4)
        elif plan.act == "off_task" and plan.anchor_turn is not None:
            plan.target_len = min(max(plan.target_len, 2), 8)
        return plan

    # ------------------------------------------------------------ realism rules
    def _realism_gate(self, st: SessionState, s: StudentState, kc: str | None, plan: ContentPlan,
                      rng: RNG) -> ContentPlan:
        """A same-speaker continuation is a coherent continuation: no new off-task topic or interface
        confusion right after one's own message; a content follow-up is marked as a continuation."""
        stu = st.recent_turns(1)
        cont = bool(stu) and stu[-1].speaker == s.id
        if cont:
            prev = stu[-1].plan
            after_greeting = prev is not None and prev.act == "greeting"
            if plan.act == "off_task" or (not after_greeting and (
                    plan.act == "interface" or (plan.act == "question" and plan.intent))):
                plan = self._carry(plan, self._continuation(st, s, kc, rng))
            if prev is not None and prev.act != "greeting" and plan.act in (
                    "on_task_claim", "explain", "disagree", "question", "press") and (
                    prev.kc is None or plan.kc in (None, prev.kc)):
                plan.continuation = True
        return plan

    def fallback_plan(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan:
        """A different, content-free plan when no wording of the planned turn passes the realism checks."""
        anchor = self._agreeable(st, s)
        p = ContentPlan(act="agree", anchor_turn=anchor.id, kc=anchor.plan.kc) if anchor else ContentPlan(act="idk")
        if self.is_self_reaction(st, s.id, p):
            p = ContentPlan(act="idk", kc=self.current_kc(st))
        return self._finish(p, s, rng, st)

    def _continuation(self, st: SessionState, s: StudentState, kc: str | None, rng: RNG) -> ContentPlan:
        if kc is None:
            return ContentPlan(act="idk")
        plan = self._claim(st, s, kc, rng)
        if plan.act in ("off_task", "interface", "backchannel") or plan.intent or (
                plan.act == "question" and not plan.asked_fact):
            asked = {m.plan.asked_fact for m in st.messages if m.speaker == s.id and m.plan is not None
                     and m.plan.asked_fact}
            missing = [f.id for f in self.task.facts if f.kc == kc and self.askable(st, s, f.id) and f.id not in asked]
            if missing:
                return ContentPlan(act="question", kc=kc, asked_fact=rng.choice(missing))
            return ContentPlan(act="idk", kc=kc)
        return plan

    def _non_offtask(self, st: SessionState, s: StudentState, kc: str | None, rng: RNG) -> ContentPlan:
        """An on-task replacement for a gated off-task turn (never off-task, never a reaction)."""
        if kc is None:
            return ContentPlan(act="idk")
        if rng.bernoulli(0.5):
            c = self._claim(st, s, kc, rng)
            if c.act in ("on_task_claim", "explain", "question", "disagree", "press"):
                return c
        alt = self._alternative(st, s, kc, rng, exclude=("off_task", "backchannel", "interface"))
        if alt.act == "off_task" or alt.act in REACTION_ACTS:
            return ContentPlan(act="idk", kc=kc)
        return alt

    def _offtask_stats(self, st: SessionState) -> tuple[list[Message], list[Message]]:
        stu = st.student_msgs()
        return stu, [m for m in stu if m.plan is not None and m.plan.act == "off_task"]

    def _offtask_ok(self, st: SessionState, s: StudentState) -> tuple[bool, Message | None]:
        """(allowed, thread message to reply to). Rules: target share offtask_rate; never two off-task
        turns in a row by the same speaker, nor right after one's own message; at least
        offtask_min_spacing student turns between off-task turns, except a peer's reply in a short
        thread (at most offtask_thread_max turns)."""
        r = float(st.context.get("offtask_rate", self.cfg.offtask_rate))
        if r <= 0 or st.closing is not None:      # (0.5.3: not while the answer is being written up)
            return False, None
        stu, offs = self._offtask_stats(st)
        if stu and stu[-1].speaker == s.id:
            return False, None
        own = [m for m in stu if m.speaker == s.id]
        last = offs[-1] if offs else None
        since = sum(1 for m in stu if last is not None and m.id > last.id)
        share_next = (len(offs) + 1) / (len(stu) + 1)
        if last is not None and last.speaker != s.id and since <= 1 and last.plan.target_len > 1:
            thread = [m for m in offs if m.plan.topic == last.plan.topic and m.id >= last.id - 8]
            if len(thread) < self.cfg.offtask_thread_max and share_next <= r + 0.08 and not (
                    own and own[-1].plan is not None and own[-1].plan.act == "off_task"
                    and own[-1].plan.topic != last.plan.topic):
                return True, last
        if own and own[-1].plan is not None and own[-1].plan.act == "off_task" and \
                any(m.id == own[-1].id for m in stu[-4:]):
            return False, None          # (0.5.1: only a recent one; a chatty student may drift off again later)
        spacing = max(1, min(self.cfg.offtask_min_spacing, round(1.0 / r) - 2))   # a higher target, denser
        if last is not None and since < spacing:
            return False, None
        if share_next > r + 0.03:
            return False, None
        return True, None

    def _offtask_p(self, st: SessionState, s: StudentState) -> float:
        ok, reply = self._offtask_ok(st, s)
        if not ok:
            return 0.0
        if reply is not None:
            return self.cfg.p_offtask_reply
        r = float(st.context.get("offtask_rate", self.cfg.offtask_rate))
        stu, offs = self._offtask_stats(st)
        share = len(offs) / max(1, len(stu))
        trait = min(2.5, max(0.3, s.traits.off_task / OFF_TASK_MEAN))
        deficit = min(3.0, max(0.3, 1.0 + 2.0 * (r - share) / max(r, 1e-9)))
        return min(0.85, 4.5 * r * trait * deficit)     # 0.5.1: 3.2 -> 4.5 (more structured turns now)

    def _off_task(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan:
        """An off-task turn: a reply in the current short thread, or a NEW topic (never one the group
        already used this session, nor one of the same family; no physical-classroom topics in an
        online chat); if not allowed, an on-task turn instead."""
        kc = self.current_kc(st)
        ok, reply = self._offtask_ok(st, s)
        if not ok:
            return self._non_offtask(st, s, kc, rng)
        if reply is not None:
            return ContentPlan(act="off_task", topic=reply.plan.topic, anchor_turn=reply.id, addressee=reply.speaker)
        used = {m.plan.topic for m in st.messages if m.plan is not None and m.plan.topic}
        fams = {TOPIC_FAMILY.get(t, t) for t in used}
        banned = set(TOPICS_NOT_ONLINE) if st.context.get("setting", self.cfg.setting) == "online" else set()
        cands = [t for t in TOPIC_IDS if t not in used and TOPIC_FAMILY.get(t, t) not in fams and t not in banned]
        if not cands:
            return self._non_offtask(st, s, kc, rng)
        own = set(s.off_task_topics or [])
        w = [1.0 if t in own else FRESH_TOPIC_MASS / max(1, len(cands)) for t in cands]
        return ContentPlan(act="off_task", topic=rng.weighted(cands, w))

    def _answer_interface(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan | None:
        """Someone may answer a recent, unanswered interface question (once)."""
        for m in reversed(st.recent_turns(6)):
            p = m.plan
            if m.speaker == s.id or p is None or p.act != "interface" or not p.intent or p.answers_intent:
                continue
            if p.intent not in intents.INTERFACE:
                return None
            if any(x.plan is not None and x.plan.answers_intent and x.plan.anchor_turn == m.id for x in st.messages):
                return None
            if rng.bernoulli(self.cfg.p_answer_interface):
                return ContentPlan(act="interface", intent=p.intent, answers_intent=True, anchor_turn=m.id,
                                   addressee=m.speaker)
            return None
        return None

    def _closing_plan(self, st: SessionState, s: StudentState, rng: RNG) -> ContentPlan | None:
        """Closing synthesis, decided in code (0.5.3): someone who took part proposes the group answer from
        what the group actually ESTABLISHED in the chat (the key ideas voiced, in the activities' order --
        paraphrased, possibly partial); then SOMEONE ELSE okays it (in their own words), or okays it and adds one
        detail they hold that the proposal does not have. A student who voiced a belief and never gave it up
        does not just okay it: they give in quoting the reason (if a refuting move came after their belief and
        the hidden state dropped it), else okay it with a hedge ("ta, pode ser, mas eu ainda acho...")."""
        if st.closing == "propose":
            if any(m.active and m.voiced for m in s.misconceptions) and any(
                    not any(m.active and m.voiced for m in x.misconceptions) for x in st.students if x.id != s.id):
                return None    # not the one still arguing for a wrong belief (0.5.1)
            if self.open_beliefs(st, s.id) and any(not self.open_beliefs(st, x.id) for x in st.students if x.id != s.id):
                return None    # nor someone who voiced one and never gave it up in the chat (0.5.3)
            mine = sum(1 for m in st.student_msgs() if m.speaker == s.id and self.content_claim(m.plan))
            others = [x for x in st.students if x.id != s.id and sum(
                1 for m in st.student_msgs() if m.speaker == x.id and self.content_claim(m.plan)) >= 2]
            spoke = sum(1 for m in st.student_msgs() if m.speaker == s.id)
            if mine == 0 and spoke < 4 and others and not st.closing_forced and rng.bernoulli(0.5):
                return None    # someone who argued the content usually writes it up (0.5.3)
            ideas = self.established_ideas(st)
            facts = [f for ki in self.task.all_key_ideas() if ki.id in ideas for f in ki.facts]
            if st.closing_forced:
                return ContentPlan(act="explain", closing="propose", fact_ids=list(dict.fromkeys(facts)),
                                   is_correct=True, ideas=ideas, partial=True)
            return ContentPlan(act="explain", closing="propose", fact_ids=list(dict.fromkeys(facts)),
                               is_correct=True, ideas=ideas)
        prop = st.messages[st.closing_msgs[0]] if st.closing_msgs else None
        if prop is None or prop.speaker == s.id:
            return None
        if st.closing_responders and s.id not in st.closing_responders:
            return None
        base = dict(anchor_turn=prop.id, addressee=prop.speaker)
        for mid in self.open_beliefs(st, s.id):
            m = s.misconception(mid)
            last = self.voicings(st, s.id, mid)[0]
            ref = self.refuting_move(st, s.id, mid, since=last.id, window=10 ** 6)
            if (m is None or not m.active) and ref is not None:
                return ContentPlan(act="agree", closing="agree", concedes=mid, reason_turn=ref.id,
                                   kc=self.task.misconception(mid).kc, **base)
            return ContentPlan(act="agree", closing="agree", holds_out=True, **base)
        spoke = sum(1 for m in st.messages if m.speaker == s.id)
        if spoke <= 1 or s.affect.engagement < 0.45:
            # someone who barely took part just okays it ("blz")
            return ContentPlan(act="agree", closing="agree", brief=True, **base)
        covered = set(prop.plan.fact_ids if prop.plan else []) | set(self.task.facts_mentioned(prop.text))
        if prop.plan is not None:
            covered |= {f for ki in self.task.all_key_ideas() if ki.id in prop.plan.ideas for f in ki.facts}
        held = [f for f in s.known_facts if f not in covered and not set(self.task.facts_mentioned(
            " ".join(self.task.fact(f).chat))) & covered]
        adjusted = any(st.messages[i].plan is not None and st.messages[i].plan.closing == "adjust"
                       for i in st.closing_msgs[1:])
        if held and not adjusted and rng.bernoulli(0.3):
            return ContentPlan(act="explain", closing="adjust", fact_ids=[rng.choice(held)], is_correct=True, **base)
        return ContentPlan(act="agree", closing="agree", **base)

    def established_ideas(self, st: SessionState) -> list[str]:
        """Key ideas the group voiced in the chat (0.5.3), in the activities' order; the correct version of a
        belief that was discussed is there because its refuting idea was voiced."""
        task = self.task
        order = {ki.id: i for i, ki in enumerate(task.all_key_ideas())}
        ideas = dict(st.ideas_voiced)
        # a belief the group discussed and someone gave up: its correct version goes in the answer too
        for m in st.messages:
            mid = m.plan.concedes if m.plan is not None else None
            if not mid:
                continue
            kc = task.misconception(mid).kc
            rf, rc = set(task.refuting_facts(mid)), set(task.refuting_claims(mid))
            ki = next((k for k in task.all_key_ideas() if k.kc == kc and (set(k.facts) & rf or set(k.claims) & rc)), None)
            if ki is not None and ki.id not in ideas:
                ideas[ki.id] = m.id
        return sorted(ideas, key=lambda k: (order.get(k, 99), ideas[k]))


def expected_len(cfg: SimConfig) -> float:
    return cfg.p_one_word + (1 - cfg.p_one_word) * math.exp(cfg.len_mu + cfg.len_sigma ** 2 / 2)
