"""State updater (rules 1-6). Applied after EVERY message, a simulated student's or an external
participant's. Every message carries a code ContentPlan (students: the planner's; external
participants: the validated interpretation of their text), and the rules read only the plan.

Flip conditions (rule 4):
  TARGETED   reader holds active m AND message anchored (referenced turn within the last
             `anchor_window` turns) to an utterance expressing m AND the challenge's family is
             in m.flip_by.
  MISALIGNED message anchored to an utterance expressing a *different* misconception.
  GENERIC    message touches m's KC but is neither of the above.
  p = p_flip[cond] * engagement**exp * (addressee 1.0 | non-addressee 0.5) * deference_peer
"""
from __future__ import annotations

from typing import Literal

from ..models.message import ACT_LABEL, EPISTEMIC_ACTS, ON_TASK_CONTENT_ACTS, Message
from ..models.rng import RNG
from ..models.session import PendingQuestion, SessionState
from ..models.student import StudentState
from ..models.task import Task
from .config import SimConfig
from . import conflict
from .content import plan_ideas
from .learning import LearningParams

Condition = Literal["targeted", "misaligned", "generic"]


class StateUpdater:
    def __init__(self, task: Task, params: LearningParams, cfg: SimConfig):
        self.task = task
        self.th = params
        self.cfg = cfg

    # =====================================================================
    def apply(self, st: SessionState, msg: Message, rng: RNG) -> dict:
        """Apply rules 1-6 for one message (already appended to st.messages)."""
        self._affect_time(st, msg.t)                       # rule 5 (time-based part)
        labels = self.rule1_classify(st, msg)              # rule 1
        readers = self.rule2_exposure(st, msg, labels, rng)  # rule 2
        self.rule3_learning(st, msg, labels, readers)      # rule 3
        flips = self.rule4_misconceptions(st, msg, labels, readers, rng)  # rule 4
        flips += self.concession(st, msg)
        self.rule5_affect(st, msg, labels, readers)        # rule 5
        solved = self.rule6_progress(st, msg, labels, rng)  # rule 6
        labels.update({"readers": readers, "flips": flips, "step_solved": solved})
        msg.labels = labels
        st.timeline.append(self.snapshot_hidden(st, msg))
        return labels

    # ---------------------------------------------------------------- rule 1
    def rule1_classify(self, st: SessionState, msg: Message) -> dict:
        p = msg.plan
        assert p is not None
        if p.claim == "misconception" and p.misconception_id and st.is_student(msg.speaker):
            ms = st.student(msg.speaker).misconception(p.misconception_id)
            if ms is not None:
                ms.voiced = True
                conflict.mark_voiced(st, msg.speaker, p.misconception_id, msg.id)
        return {"act": ACT_LABEL[p.act], "kc_refs": [p.kc] if p.kc else [],
                "facts_mentioned": list(p.fact_ids), "is_correct": p.is_correct,
                "addressee": p.addressee, "expresses": p.misconception_id,
                "family": p.family, "anchor": p.anchor_turn,
                "addressed_misconception": p.addresses_misconception}

    # ---------------------------------------------------------------- rule 2
    def rule2_exposure(self, st: SessionState, msg: Message, labels: dict, rng: RNG) -> list[str]:
        readers: list[str] = []
        for s in st.students:
            if s.id == msg.speaker:
                continue
            e = s.affect.engagement
            if e < self.th.read_min_engagement:
                continue
            if rng.random() < self.th.read_p_intercept + self.th.read_p_slope * e:
                readers.append(s.id)
        for sid in readers:
            s = st.student(sid)
            for f in labels["facts_mentioned"]:
                if f not in s.known_facts:
                    s.known_facts.append(f)
                    kc = self.task.fact(f).kc
                    s.kcs[kc].evidence_seen.append(f)
        if st.is_student(msg.speaker):  # external participants have no hidden state
            sp = st.student(msg.speaker)
            for f in labels["facts_mentioned"]:
                if f not in sp.shared_facts:
                    sp.shared_facts.append(f)
        return readers

    # ---------------------------------------------------------------- rule 3
    def rule3_learning(self, st: SessionState, msg: Message, labels: dict, readers: list[str]) -> None:
        if not labels["kc_refs"]:
            return
        kc = labels["kc_refs"][0]
        correct = labels["is_correct"]
        p = msg.plan
        if p.act not in ON_TASK_CONTENT_ACTS or correct is None:
            return
        T_recv = self.th.t_received_correct if correct else self.th.t_incorrect
        if correct and (p.act == "explain" or p.revoice) and st.is_student(msg.speaker):
            sp = st.student(msg.speaker)
            self._learn(sp, kc, self.th.t_self_produced)
            sp.kcs[kc].produced_explanation = True
        for sid in readers:
            self._learn(st.student(sid), kc, T_recv)

    def _learn(self, s: StudentState, kc: str, T: float) -> None:
        if T <= 0:
            return
        k = s.kcs[kc]
        k.p = k.p + (1 - k.p) * T
        if s.active_misconception(kc) is not None:
            k.p = min(k.p, self.th.active_misconception_p_cap)

    # ---------------------------------------------------------------- rule 4
    def condition(self, st: SessionState, msg: Message, labels: dict, m_id: str) -> Condition:
        """Decide TARGETED / MISALIGNED / GENERIC for misconception m_id (code)."""
        anchor = labels.get("anchor")
        expressed = None
        if anchor is not None and self._anchored(st, msg, anchor):
            am = st.messages[anchor]
            expressed = am.plan.misconception_id if am.plan else None
        if expressed == m_id and self._eligible(labels.get("family"), m_id):
            return "targeted"
        if expressed is not None and expressed != m_id:
            return "misaligned"
        return "generic"

    def _anchored(self, st: SessionState, msg: Message, anchor: int) -> bool:
        window = [m.id for m in st.recent_turns(self.th.anchor_window, before=msg.id)]
        return anchor in window

    def _eligible(self, family: str | None, m_id: str) -> bool:
        return family is not None and family in self.task.misconception(m_id).flip_by

    def flip_probability(self, st: SessionState, msg: Message, reader: StudentState,
                         cond: Condition, strength: float, addressee: str | None) -> float:
        th = self.th
        base = {"targeted": th.p_targeted(strength), "misaligned": th.p_flip.misaligned,
                "generic": th.p_flip.generic}[cond]
        p = base * reader.affect.engagement ** th.engagement_exponent
        is_addr = addressee is None or addressee == reader.id
        p *= th.addressee_multiplier if is_addr else th.non_addressee_multiplier
        p *= reader.traits.deference_peer
        return min(1.0, max(0.0, p))

    def rule4_misconceptions(self, st: SessionState, msg: Message, labels: dict,
                             readers: list[str], rng: RNG) -> list[dict]:
        kc_refs = set(labels["kc_refs"])
        if not kc_refs:
            return []
        if msg.plan.act not in EPISTEMIC_ACTS:
            return []
        if msg.plan.claim == "misconception":
            return []  # voicing a misconception does not challenge anyone
        events = []
        keep = conflict.protected(st)     # a slated belief is not dissolved by talk before it surfaces
        for sid in readers:
            r = st.student(sid)
            for m in r.misconceptions:
                if not m.active or m.kc not in kc_refs or (sid, m.id) in keep:
                    continue
                cond = self.condition(st, msg, labels, m.id)
                p = self.flip_probability(st, msg, r, cond, m.strength, labels.get("addressee"))
                u = rng.random()
                flipped = u < p
                ev = {"msg": msg.id, "t": msg.t, "student": sid, "m": m.id, "cond": cond,
                      "p": round(p, 4), "flipped": flipped, "source": msg.kind}
                if flipped:
                    m.active = False
                    m.flipped_at = msg.t
                    k = r.kcs[m.kc]
                    k.p = min(1.0, k.p + self.th.flip_kc_bonus)
                elif cond == "targeted":
                    m.strength = max(0.0, m.strength + self.th.targeted_nonflip_strength_delta)
                elif cond == "misaligned":
                    m.strength = min(1.0, m.strength + self.th.misaligned_entrench_delta)
                events.append(ev)
        st.flip_log.extend(events)
        return events

    def concession(self, st: SessionState, msg: Message) -> list[dict]:
        """0.5.1: a student who says they were convinced (plan.concedes) drops that belief."""
        p = msg.plan
        if not p.concedes or not st.is_student(msg.speaker):
            return []
        m = st.student(msg.speaker).misconception(p.concedes)
        if m is None or not m.active:
            return []
        m.active = False
        m.flipped_at = msg.t
        k = st.student(msg.speaker).kcs[m.kc]
        k.p = min(1.0, k.p + self.th.flip_kc_bonus)
        ev = {"msg": msg.id, "t": msg.t, "student": msg.speaker, "m": m.id, "cond": "concede", "p": 1.0,
              "flipped": True, "source": msg.kind}
        st.flip_log.append(ev)
        return [ev]

    def relapse(self, st: SessionState, rng: RNG) -> None:
        """Relapse check, once per task step."""
        for s in st.students:
            for m in s.misconceptions:
                if not m.active and m.flipped_at is not None and rng.random() < self.th.relapse_p:
                    m.active = True
                    m.relapses += 1
                    s.kcs[m.kc].p = min(s.kcs[m.kc].p, self.th.active_misconception_p_cap)
                    st.flip_log.append({"t": st.t, "student": s.id, "m": m.id, "relapse": True})

    # ---------------------------------------------------------------- rule 5
    def _affect_time(self, st: SessionState, t: float) -> None:
        dt_min = max(0.0, t - st.last_affect_t) / 60.0
        st.last_affect_t = t
        if t - st.last_progress_t > 60.0:
            for s in st.students:
                s.affect.engagement = max(0.0, s.affect.engagement - self.th.engagement_decay_per_min * dt_min)

    def rule5_affect(self, st: SessionState, msg: Message, labels: dict, readers: list[str]) -> None:
        th = self.th
        p = msg.plan
        # replied-to bonus
        if p.anchor_turn is not None:
            am = st.messages[p.anchor_turn]
            if am.speaker != msg.speaker and st.is_student(am.speaker):
                tgt = st.student(am.speaker)
                tgt.affect.engagement = min(1.0, tgt.affect.engagement + th.engagement_replied_bonus)
            for q in list(st.pending_questions):
                if q.msg_id == p.anchor_turn and q.asker != msg.speaker:
                    st.pending_questions.remove(q)
        # unanswered questions age by one turn
        for q in list(st.pending_questions):
            if q.asker == msg.speaker and q.msg_id != msg.id:
                continue
            q.turns_waited += 1
            if q.turns_waited > th.unanswered_window_turns:
                if st.is_student(q.asker):
                    a = st.student(q.asker)
                    a.affect.frustration = min(1.0, a.affect.frustration + th.frustration_unanswered)
                st.pending_questions.remove(q)
        if p.act == "question" and not p.step_move:      # ("bora pra 2?" is not a question to answer)
            st.pending_questions.append(PendingQuestion(msg_id=msg.id, asker=msg.speaker, kc=p.kc))

    # ---------------------------------------------------------------- rule 6
    def rule6_progress(self, st: SessionState, msg: Message, labels: dict, rng: RNG) -> str | None:
        """Task progress. The current step is solved when, at a message,
          (a) its gold KCs have >= min_correct_claims_per_kc correct claims since the step started,
              from >= min_distinct_speakers chat members;
          (b) every KEY IDEA of the step has been stated correctly by someone in the group chat
              (code check on the plan content: a card fact or a correct claim covering the idea);
          (c) no student holds an ACTIVE misconception tied to the step (its KC or trigger step) that
              they voiced in the group chat and that was never challenged or refuted since (a turn
              addressing it, or a turn stating a fact that refutes it). A misconception that was never
              voiced does not block."""
        if st.task_complete:
            return None
        p = msg.plan
        if st.closing_forced:
            return None        # wrapping up under time pressure: no step is solved any more (0.5.1)
        new_idea = False
        for idea in plan_ideas(self.task, p):
            if idea not in st.ideas_voiced:
                st.ideas_voiced[idea] = msg.id
                new_idea = True
        step = self.task.steps[st.step_idx]
        gold = self.task.gold_for(step.id)
        if p.act in ON_TASK_CONTENT_ACTS and p.is_correct and p.kc is not None and p.kc in gold.kcs:
            st.step_correct[p.kc] = st.step_correct.get(p.kc, 0) + 1
            if msg.speaker not in st.step_speakers:
                st.step_speakers.append(msg.speaker)
            st.last_progress_t = msg.t
            st.last_progress_msg = msg.id
        if new_idea or p.addresses_misconception:
            st.last_progress_msg = msg.id      # stall detection only (last_progress_t keeps its meaning)
        if not self.step_ready(st, step.id, msg.t):
            return None
        st.steps_solved.append({"step": step.id, "t": msg.t, "msg": msg.id,
                                "key_ideas": {k.id: st.ideas_voiced[k.id] for k in self.task.key_ideas(step.id)
                                              if k.id in st.ideas_voiced}})
        st.step_idx += 1
        st.step_started_t = msg.t
        st.step_started_msg = msg.id
        st.last_progress_msg = msg.id
        st.step_correct = {}
        st.step_speakers = []
        self.relapse(st, rng)
        return step.id

    def step_ready(self, st: SessionState, step_id: str, t: float | None = None) -> bool:
        """Rule 6 (a)-(c), plus (0.5.1) (d) the step has been open for at least the pace's dwell time,
        (e) every belief slated to surface on it has been voiced (and, by (c), challenged), and under
        conflict="high" (f) no voiced belief tied to it is still held."""
        if not self.content_ready(st, step_id):
            return False
        t = st.t if t is None else t
        if t - st.step_started_t < self.cfg.step_dwell_s(len(self.task.steps)):
            return False
        return True

    def content_ready(self, st: SessionState, step_id: str) -> bool:
        """Every content condition of rule 6 (all but the dwell time)."""
        if (conflict.pending(st, step_id) or conflict.undiscussed(st, step_id)) and st.t - st.step_started_t < (
                self.cfg.step_dwell_s(len(self.task.steps)) + self.cfg.slate_grace_s):
            return False
        if self.cfg.conflict_resolution_required and self.unresolved_misconceptions(st, step_id):
            return False
        return self._content_ready(st, step_id)

    def unresolved_misconceptions(self, st: SessionState, step_id: str) -> list[dict]:
        """Voiced beliefs tied to the step whose holder still holds them (conflict="high")."""
        gold = self.task.gold_for(step_id)
        out = []
        for s in st.students:
            for m in s.misconceptions:
                spec = self.task.misconception(m.id)
                if m.active and m.voiced and (m.kc in gold.kcs or spec.trigger == step_id):
                    out.append({"student": s.id, "m": m.id})
        return out

    def _content_ready(self, st: SessionState, step_id: str) -> bool:
        gold = self.task.gold_for(step_id)
        if not (all(st.step_correct.get(k, 0) >= self.cfg.min_correct_claims_per_kc for k in gold.kcs)
                and len(st.step_speakers) >= min(self.cfg.min_distinct_speakers, len(st.students))):
            return False
        if self.cfg.require_key_ideas and self.missing_key_ideas(st, step_id):
            return False
        if self.cfg.block_unchallenged_misconception and self.blocking_misconceptions(st, step_id):
            return False
        return True

    def missing_key_ideas(self, st: SessionState, step_id: str) -> list[str]:
        return [k.id for k in self.task.key_ideas(step_id) if k.id not in st.ideas_voiced]

    def blocking_misconceptions(self, st: SessionState, step_id: str) -> list[dict]:
        """Active, group-voiced, never-challenged misconceptions tied to the step (rule 6 (c))."""
        gold = self.task.gold_for(step_id)
        out = []
        for s in st.students:
            for m in s.misconceptions:
                if not m.active:
                    continue
                spec = self.task.misconception(m.id)
                if m.kc not in gold.kcs and spec.trigger != step_id:
                    continue
                last = next((x for x in reversed(st.messages) if x.speaker == s.id and x.plan is not None
                             and x.plan.claim == "misconception" and x.plan.misconception_id == m.id), None)
                if last is not None and not self.challenged_since(st, s.id, m.id, last.id):
                    out.append({"student": s.id, "m": m.id, "voiced_msg": last.id})
        return out

    def challenged_since(self, st: SessionState, sid: str, mid: str, since: int) -> bool:
        refuters = {f for f, ms in self.task.refutes.items() if mid in ms}
        for x in st.messages[since + 1:]:
            if x.speaker == sid or x.plan is None:
                continue
            if x.plan.addresses_misconception == mid:
                return True
            if x.plan.is_correct and x.plan.claim in ("fact", "correct"):
                facts = set(x.plan.fact_ids)
                for ki in self.task.all_key_ideas():
                    if ki.id in plan_ideas(self.task, x.plan):
                        facts |= set(ki.facts)
                if facts & refuters:
                    return True
        return False

    # ---------------------------------------------------------------- timeline
    @staticmethod
    def snapshot_hidden(st: SessionState, msg: Message) -> dict:
        return {"msg": msg.id, "t": round(msg.t, 2), "step_idx": st.step_idx,
                "students": {s.id: {
                    "p": {k: round(v.p, 4) for k, v in s.kcs.items()},
                    "misc": {m.id: (round(m.strength, 3) if m.active else 0.0) for m in s.misconceptions},
                    "eng": round(s.affect.engagement, 3), "frus": round(s.affect.frustration, 3),
                    "n_facts": len(s.known_facts)}
                    for s in st.students}}
