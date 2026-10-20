"""Continuous-time turn-taking (simulated students only; external participants post on their own).

lambda_i = base * talkativeness * engagement * g_i(t), with
  g = 0 before lurk_until; x3 if addressed (decays over 30 s); x0.5 during interface work.
Gap to the next student message is lognormal (pilot-fitted), scaled by
(Lambda_ref / Lambda) ** gap_rate_exponent; the part of a gap beyond 60 s of silence is
compressed by the x1.5 silence boost.

Bursts: "x burstiness if last speaker" is implemented as a *probability*: with probability
burstiness_last (trait mean .37 = pilot P(same speaker)) the last speaker keeps the floor and the
gap comes from the shorter burst distribution; otherwise the next speaker is drawn among the
others proportional to lambda_i.  (A raw multiplier < 1 on the last speaker's hazard would push
P(same) far below the pilot value.)
"""
from __future__ import annotations

import math

from ..models.rng import RNG
from ..models.session import PlannedEvent, SessionState
from .config import SimConfig


def can_answer(st: SessionState, sid: str, qmsg) -> str | None:
    """'fact' if the student holds the fact a content question asks about, 'kc' if they can
    state the question's KC and hold no misconception there, else None. Generic questions
    ("what do we do?") are not content questions."""
    p = qmsg.plan
    if p is None or p.act != "question" or not p.asked_fact:
        return None
    s = st.student(sid)
    if p.asked_fact in s.known_facts:
        return "fact"
    if p.kc and s.kcs[p.kc].can_state and s.active_misconception(p.kc) is None:
        return "kc"
    return None


def stalled(st: SessionState, cfg: SimConfig) -> bool:
    """No task progress (a counted correct claim, a new key idea, a challenge) for cfg.stall_turns
    student turns or cfg.stall_s sim seconds (0.5.0 progress pressure)."""
    if st.task_complete or st.closing is not None:
        return False
    since = sum(1 for m in st.messages[st.last_progress_msg + 1:] if st.is_student(m.speaker))
    ref_t = max(st.last_progress_t, st.step_started_t,
                st.messages[st.last_progress_msg].t if 0 <= st.last_progress_msg < len(st.messages) else 0.0)
    return since >= cfg.stall_turns or (since >= 2 and st.t - ref_t >= cfg.stall_s)


def missing_idea_facts(st: SessionState, task=None) -> set[str]:
    """Card facts of the current step's key ideas nobody has voiced yet (set on the state by the engine)."""
    return set(st.context.get("missing_facts", []))


def holds_missing_fact(st: SessionState, sid: str) -> bool:
    miss = missing_idea_facts(st)
    return bool(miss) and any(f in miss for f in st.student(sid).known_facts)


class Scheduler:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg

    def hazard(self, st: SessionState, sid: str, t: float) -> float:
        s = st.student(sid)
        if t < s.traits.lurk_until_min * 60.0 or t < st.quiet_until.get(sid, -1.0):
            return 0.0
        g = 1.0
        ta = st.addressed_at.get(sid)
        if ta is not None and t >= ta:
            g *= 1.0 + (self.cfg.addressed_boost - 1.0) * math.exp(-(t - ta) / self.cfg.addressed_decay_s)
        if st.interface_until.get(sid, -1.0) > t:
            g *= self.cfg.interface_factor
        if self.cfg.unspoken_boost_per_min:
            t0 = s.traits.lurk_until_min * 60.0
            if t > t0 and not any(m.speaker == sid for m in st.messages):
                g *= 1.0 + self.cfg.unspoken_boost_per_min * (t - t0) / 60.0
        if self.cfg.stall_holder_boost and stalled(st, self.cfg) and holds_missing_fact(st, sid):
            g *= self.cfg.stall_holder_boost
        if self.cfg.stall_holder_boost and sid in st.context.get("slate_holders", ()) and stalled(st, self.cfg):
            g *= self.cfg.stall_holder_boost    # a belief slated to surface on this step (0.5.1)
        tq = self.answerable_question_t(st, sid)
        if tq is not None and t >= tq:
            g *= 1.0 + (self.cfg.answer_boost - 1.0) * s.affect.engagement * math.exp(
                -(t - tq) / self.cfg.answer_decay_s)
        # a student who has not written yet keeps their raw talkativeness (everybody says something)
        spoke = any(m.speaker == sid for m in st.messages)
        talk = self.talk(st, s) if spoke else s.traits.talkativeness
        return talk * max(0.05, s.affect.engagement) * g

    def talk(self, st: SessionState, s) -> float:
        """Talkativeness sharpened by talk_exponent, renormalised so the group's mean is unchanged
        (who talks changes, not how much the group talks)."""
        k = self.cfg.talk_exponent
        if k == 1.0:
            return s.traits.talkativeness
        xs = [x.traits.talkativeness for x in st.students]
        return s.traits.talkativeness ** k * (sum(xs) / sum(x ** k for x in xs))

    @staticmethod
    def answerable_question_t(st: SessionState, sid: str) -> float | None:
        """Time of the newest pending content question this student can answer."""
        for q in reversed(st.pending_questions):
            if q.asker == sid:
                continue
            if can_answer(st, sid, st.messages[q.msg_id]):
                return st.messages[q.msg_id].t
        return None

    @staticmethod
    def current_run(st: SessionState) -> tuple[str | None, int]:
        """(speaker, length) of the current run of consecutive student turns by one speaker
        (external participants' messages are skipped)."""
        who, n = None, 0
        for m in reversed(st.messages):
            if not st.is_student(m.speaker):
                continue
            if who is None:
                who = m.speaker
            elif m.speaker != who:
                break
            n += 1
        return who, n

    @staticmethod
    def content_in_run(st: SessionState) -> int:
        """Content turns in the current same-speaker run (openers and content-free turns do not count)."""
        from .planner import is_content_free
        who, n = None, 0
        for m in reversed(st.messages):
            if not st.is_student(m.speaker):
                continue
            if who is None:
                who = m.speaker
            elif m.speaker != who:
                break
            if m.plan is not None and not m.plan.opener and not m.plan.split_tail and m.plan.act != "greeting" \
                    and not m.plan.after_hedge and not m.plan.unsure and not is_content_free(m.plan):
                n += 1
        return n

    def last_student_speaker(self, st: SessionState) -> str | None:
        for m in reversed(st.messages):
            if st.is_student(m.speaker):
                return m.speaker
        return None

    def plan_next(self, st: SessionState, now: float, rng: RNG) -> PlannedEvent | None:
        """Sample (time, speaker) of the next student message given state at `now`, under the
        group-silence cap (0.5.1)."""
        ev = self._plan_next(st, now, rng)
        return ev if ev is None else self.cap_silence(st, now, ev, rng)

    def cap_silence(self, st: SessionState, now: float, ev: PlannedEvent, rng: RNG) -> PlannedEvent:
        """Never more than cfg.max_silence_s of group silence while the chat runs: if the next message
        would come later, someone who is already in the chat speaks at the cap (a forced turn: it says
        something, preferably with content). Prefers the planned speaker unless they spoke last."""
        cap = self.cfg.max_silence_s
        if not cap or not st.messages or st.closing == "done":
            return ev
        limit = st.last_message_t + cap
        if ev.t <= limit:
            return ev
        # someone breaks the silence somewhere in the last fifth before the cap (no spike at the cap)
        t = max(st.last_message_t + cap * rng.uniform(0.8, 1.0), now)
        ready = [i for i in st.roster if st.student(i).traits.lurk_until_min * 60.0 <= t]
        if not ready:
            return ev                      # nobody has arrived yet
        last = self.last_student_speaker(st)
        if ev.speaker in ready and (ev.speaker != last or len(ready) == 1):
            who = ev.speaker
        else:
            others = [i for i in ready if i != last] or ready
            w = [st.student(i).traits.talkativeness * max(0.05, st.student(i).affect.engagement) for i in others]
            who = rng.weighted(others, w)
        if who == last and self.current_run(st)[1] >= self.cfg.max_run_hard_cap and len(st.roster) > 1:
            # alone in the chat for a while: the first of the others arrives instead of a longer monologue
            nxt = min((i for i in st.roster if i != last), key=lambda i: st.student(i).traits.lurk_until_min)
            st.student(nxt).traits.lurk_until_min = min(st.student(nxt).traits.lurk_until_min, t / 60.0)
            who = nxt
        return PlannedEvent(t=t, speaker=who, forced=True)

    def _plan_next(self, st: SessionState, now: float, rng: RNG) -> PlannedEvent | None:
        cfg = self.cfg
        ids = st.roster
        lam = {i: self.hazard(st, i, now) for i in ids}
        if sum(lam.values()) <= 0:
            # everyone lurking: wake up at the earliest lurk end
            wake = min(st.student(i).traits.lurk_until_min * 60.0 for i in ids)
            if wake <= now:
                return None
            return self._plan_next(st, wake + 1e-6, rng)
        pend = st.pending_contribution
        if pend is not None and pend["speaker"] in lam:
            # the content right after one's own opener ("pera" ... "acho que ..."), or an afterthought
            gap = rng.lognormal(math.log(8.0 if pend.get("after") else 5.0), 0.6)
            return PlannedEvent(t=now + gap, speaker=pend["speaker"])
        # the student an external participant addressed takes the floor (decided once per address)
        last, run = self.current_run(st)
        capped = last is not None and len(ids) > 1 and (
            run >= cfg.max_run_hard_cap or self.content_in_run(st) >= cfg.max_content_run)
        if st.closing == "propose" and run >= 2 and len(ids) > 1:
            capped = True       # the synthesis is not a third message in a row
        if st.closing == "confirm" and st.closing_msgs and len(ids) > 1:
            last = st.messages[st.closing_msgs[0]].speaker   # someone ELSE confirms the proposed answer
            capped = True
        for sid, oa in st.open_address.items():
            if (sid in lam and lam[sid] > 0 and oa.turns_waited == 0
                    and not oa.floor_decided and not (capped and sid == last)):
                oa.floor_decided = True
                if rng.random() < cfg.p_addressee_takes_floor:
                    gap = rng.lognormal(math.log(cfg.burst_gap_median_s * 1.5), cfg.burst_gap_sigma)
                    return PlannedEvent(t=now + gap, speaker=sid)
        others_addressed = any(st.addressed_at.get(i, -1e9) > now - cfg.addressed_decay_s
                               for i in ids if i != last)
        burst_p = 0.0
        if last is not None and lam.get(last, 0) > 0:
            # soft cap: the keep-the-floor probability decays with every consecutive turn
            # (burst_decay ** (run - 1)); hard cap at max_run_hard_cap
            burst_p = (min(0.95, st.student(last).traits.burstiness * cfg.burst_scale)
                       * (0.5 if others_addressed else 1.0) * cfg.burst_decay ** max(0, run - 1))
            if capped:
                burst_p = 0.0
            # a student keeps the floor only to CONTINUE a thought; after a reaction, an idk or an
            # off-task remark there is nothing coherent to add, and after their own question the
            # others are the ones to answer (0.5.0: no "how do i send it?" -> "no idea" by one student)
            lm = next((m for m in reversed(st.messages) if st.is_student(m.speaker)), None)
            if lm is not None and lm.plan is not None and (lm.plan.act in (
                    "off_task", "idk", "refusal", "backchannel", "agree")
                    or lm.plan.claim == "hedge"):
                burst_p = 0.0
            elif lm is not None and lm.plan is not None and lm.plan.act in ("question", "press") \
                    and lm.plan.addressee in st.roster and lm.plan.addressee != lm.speaker and not lm.plan.tag:
                burst_p = 0.0      # (0.5.3) asked a classmate by name / pressed them: they wait for the answer
            elif lm is not None and lm.plan is not None and lm.plan.act in ("question", "interface"):
                burst_p *= cfg.burst_after_question
            if st.messages and not st.is_student(st.messages[-1].speaker):
                burst_p *= cfg.burst_after_external   # someone else just wrote: the floor is open
        others = {i: v for i, v in lam.items() if i != last and v > 0}
        if st.closing == "confirm" and st.closing_responders:
            # the students still to respond to the proposal take the floor, one after another
            real_last = self.last_student_speaker(st)
            resp = [i for i in st.closing_responders if i in lam and i != last]
            resp = [i for i in resp if i != real_last] or resp
            here = [i for i in resp if st.student(i).traits.lurk_until_min * 60.0 <= now]
            pool = here or sorted(resp, key=lambda i: st.student(i).traits.lurk_until_min)[:1]
            if pool:
                others = {i: max(lam.get(i, 0.0), 0.2) for i in pool}
                burst_p = 0.0
        if not others and burst_p == 0.0 and last is not None:
            # the others are quiet for now (they had nothing to say) and the last speaker has nothing
            # coherent to add: the floor waits for the first of them to be ready again
            waits = [st.quiet_until[i] for i in ids if i != last and st.quiet_until.get(i, -1.0) > now
                     and st.student(i).traits.lurk_until_min * 60.0 <= st.quiet_until[i]]
            if waits:
                return self._plan_next(st, min(waits) + 1e-6, rng)
        if capped and not others:
            # everyone else is lurking: the floor passes to the earliest-waking other student
            # (lurk end is enforced below) instead of a monologue beyond the cap
            nxt = min((i for i in ids if i != last), key=lambda i: st.student(i).traits.lurk_until_min)
            others = {nxt: 1.0}
        if last is not None and (rng.random() < burst_p or not others):
            speaker = last
            gap = rng.lognormal(math.log(cfg.burst_gap_median_s), cfg.burst_gap_sigma)
        else:
            pool = others if others else lam
            speaker = rng.weighted(list(pool), list(pool.values()))
            total = sum(lam.values())
            gap = rng.lognormal(math.log(cfg.gap_median_s), cfg.gap_sigma)
            gap *= (cfg.lambda_ref / total) ** cfg.gap_rate_exponent
        if st.closing in ("propose", "confirm"):
            # the group is wrapping up; the proposal / confirmation follows promptly
            gap = min(gap, 90.0, rng.lognormal(math.log(12.0), 0.6))
        since_last = now - st.last_message_t
        threshold = max(0.0, cfg.silence_boost_after_s - since_last)
        if gap > threshold:
            gap = threshold + (gap - threshold) / cfg.silence_boost
        # a lurker cannot speak before lurk end
        t = max(now + gap, st.student(speaker).traits.lurk_until_min * 60.0)
        return PlannedEvent(t=t, speaker=speaker)
