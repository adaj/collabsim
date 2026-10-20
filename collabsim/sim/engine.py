"""Event-driven session engine: scheduler -> planner -> realizer -> guards -> updater.

The engine plans and words the simulated students' turns. External participants (people or bots
joining the chat through the public API) post with inject_external(); their messages carry the
validated interpretation of their text as a plan, and the same state rules apply to them as to a
peer's turn. The code never branches on who a participant is.

snapshot()/restore() capture the full SessionState and every RNG stream, so a restored session
replays bit-identically.
"""
from __future__ import annotations

import copy

from ..models.message import ContentPlan, Message
from ..models.rng import RNG, RNGBank, derive_seed
from ..models.session import OpenAddress, SessionState
from ..models.task import Task, load_task
from . import conflict, realism
from .config import SimConfig
from .group import generate_group
from .learning import LearningParams
from .planner import ContentPlanner
from .realizer import Realizer, TemplateRealizer
from .scheduler import Scheduler
from .updater import StateUpdater

OPEN_ADDRESS_TURNS = 6     # an address left unanswered this many turns is dropped
ADDRESS_ACTS = {"question", "press", "disagree", "explain", "on_task_claim", "interface"}


class Simulation:
    def __init__(self, task: Task | str, seed: int, cfg: SimConfig | None = None,
                 realizer: Realizer | None = None, n_students: int | None = None,
                 session_id: str | None = None, params: LearningParams | None = None):
        self.task = load_task(task) if isinstance(task, str) else task
        self.params = params or LearningParams()
        self.cfg = cfg or SimConfig()
        self.rngs = RNGBank(seed)
        self.realizer = realizer or TemplateRealizer()
        if hasattr(self.realizer, "max_content_words"):
            self.realizer.max_content_words = self.cfg.content_max_words
        self.planner = ContentPlanner(self.task, self.params, self.cfg)
        self.scheduler = Scheduler(self.cfg)
        self.updater = StateUpdater(self.task, self.params, self.cfg)
        students = generate_group(self.task, self.rngs["group"], self.cfg, n_students)
        sid = session_id or f"{self.task.id}__s{seed}"
        self.state = SessionState(session_id=sid, task_id=self.task.id, seed=seed, students=students,
                                  n_steps=len(self.task.steps))
        self.rngs.streams["realism"] = RNG(derive_seed(seed, "realism"))  # own stream: other streams unchanged
        # 0.5.1: conflict top-up / slates and the closing responders draw from their own streams
        self.rngs.streams["conflict"] = RNG(derive_seed(seed, "conflict"))
        self.rngs.streams["closing"] = RNG(derive_seed(seed, "closing"))
        conflict.top_up(self.task, students, self.cfg, self.rngs["conflict"])
        self.state.slates = conflict.make_slates(self.task, students, self.cfg, self.rngs["conflict"])
        realism.init_context(self.state, self.cfg, self.rngs["realism"])
        self._update_missing()
        self.initial_students = [s.model_dump() for s in students]
        self.names = {s.id: s.name for s in students}
        self._started = False

    # =============================================================== snapshot
    def snapshot(self) -> dict:
        snap = {"state": self.state.model_dump(mode="json"), "rng": self.rngs.get_state(),
                "started": self._started, "names": dict(self.names)}
        return copy.deepcopy(snap)

    def restore(self, snap: dict) -> None:
        snap = copy.deepcopy(snap)
        self.state = SessionState.model_validate(snap["state"])
        self.rngs.set_state({k: v for k, v in snap["rng"].items() if k in self.rngs.streams})
        self._started = snap["started"]
        if "names" in snap:
            self.names = snap["names"]

    # =============================================================== running
    @property
    def end_t(self) -> float:
        return self.cfg.session_minutes * 60.0

    @property
    def stop_t(self) -> float:
        """Time cap; or, once the closing synthesis is done, now; or the end of the capped wrap-up
        window after the task was completed."""
        st = self.state
        if st.closing == "done":
            return min(self.end_t, st.t)
        if st.closing == "confirm" and st.closing_responders:
            return self.end_t        # a proposal being answered is not cut off by the wrap-up window (0.5.2)
        if self.cfg.end_on_complete and st.solved_at is not None:
            return min(self.end_t, st.solved_at + self.cfg.closing_wrap_up_max_minutes * 60.0)
        return self.end_t

    def run(self, max_events: int | None = None) -> SessionState:
        n = 0
        while not self.state.done:
            self.step()
            n += 1
            if max_events is not None and n >= max_events:
                break
        return self.state

    def step(self) -> None:
        st = self.state
        if not self._started:
            self._started = True
            st.planned = self.scheduler.plan_next(st, st.t, self.rngs["sched"])
            return
        self._wrap_up_in_time()
        planned = st.planned
        stop = self.stop_t
        t = planned.t if planned is not None else None
        if st.closing == "done":
            t = None
        if t is None or t > stop or st.student_turns >= self.cfg.max_student_turns:
            st.done = True
            if st.student_turns >= self.cfg.max_student_turns and t is not None and t <= stop:
                st.ended_reason = "turn_cap"
            elif st.closing_forced:      # wrapped up under time / turn pressure (even if the last step got done meanwhile)
                near_turns = st.student_turns >= self.cfg.max_student_turns - self.planner.turn_headroom()
                st.ended_reason = "turn_cap" if near_turns else "time_cap"
            elif stop < self.end_t:
                st.ended_reason = "complete"
            else:
                st.ended_reason = "time_cap"
            st.t = min(max(st.t, t if t is not None else st.t), stop)
            return
        t0 = st.t
        st.t = t
        if self._student_turn(planned.speaker, forced=planned.forced) is None:
            # nothing to say: that student stays quiet until then, and the floor is re-drawn from the
            # previous moment (a skipped turn does not eat the group's time)
            st.quiet_until[planned.speaker] = t
            st.t = t0
            nxt = self.scheduler.plan_next(st, t0, self.rngs["sched"])
            if nxt is None or nxt.t > t:
                st.t = t
                nxt = self.scheduler.plan_next(st, t, self.rngs["sched"])
            st.planned = nxt
            return
        st.planned = self.scheduler.plan_next(st, st.t, self.rngs["sched"])

    def _wrap_up_in_time(self) -> None:
        """The closing synthesis must fit before the time cap: if the next turn would only come after
        the wrap-up point (a long silence near the end), it is brought forward to it."""
        st, cfg = self.state, self.cfg
        if st.closing is not None or st.task_complete or st.planned is None:
            return
        headroom = min(cfg.closing_headroom_min * 60.0, cfg.session_minutes * 9.0)
        wrap_t = self.end_t - headroom
        if st.planned.t <= wrap_t + headroom / 3 or st.t >= self.end_t - 60.0:
            return
        t = max(st.t + 1.0, wrap_t)
        ready = [x for x in st.students if x.traits.lurk_until_min * 60.0 <= t]
        if not ready:
            return
        who = st.planned.speaker if any(x.id == st.planned.speaker for x in ready) else \
            max(ready, key=lambda x: x.traits.talkativeness).id
        st.planned = st.planned.model_copy(update={"t": t, "speaker": who})

    # =============================================================== students
    def _student_turn(self, sid: str, forced: bool = False) -> Message | None:
        st = self.state
        plan = self.planner.plan(st, sid, self.rngs["sched"], forced=forced)
        if plan is None:            # nothing to say: the student stays quiet (logged by the planner)
            st.skipped_turns += 1
            return None
        log = getattr(self.realizer, "log", None)
        n0 = len(log) if log is not None else 0
        if plan.split_tail:       # the rest of a line that already passed every check (0.5.2)
            text = plan.split_tail
            if log is not None:   # one realizer record per turn (a template line, no call)
                log.append({"msg": len(st.messages), "speaker": sid, "act": plan.act, "attempts": 0,
                            "templated": True, "split_tail": True, "pre_guard_leak": False, "post_guard_leak": False})
        else:
            text = self._finalize(plan, sid, self._word(plan, sid))
            plan, text = self._realism_final(plan, sid, text)
            if forced and plan is not None and self.planner.is_content_free_plan(plan) and not plan.opener:
                plan = None          # a forced turn is never a filler (the nudge below, 0.5.3)
        if plan is not None and not (plan.closing or plan.opener or plan.replies_to_external or plan.nudge
                                     or plan.draft) and \
                realism.content_free_text(text, self.task) and \
                self.planner.content_free_run(st) >= self.cfg.max_content_free_run:
            plan = None             # the wording came out as one more filler after a run of them: quiet
            st.slippage_log.append({"type": "filler_gate", "msg": len(st.messages), "speaker": sid,
                                    "reason": "content_free_run_text", "draft": text, "new_act": None})
        after_opener = bool(st.messages) and st.messages[-1].speaker == sid and \
            st.messages[-1].plan is not None and st.messages[-1].plan.opener
        if plan is None and (forced or after_opener):
            # the silence cap was reached, or the student already typed "pera ...": a short nudge / the
            # activity question at least (never a lone opener)
            plan, text = self._nudge(sid)
        pend = st.pending_contribution
        if pend is not None and pend.get("after") and pend["speaker"] == sid and (
                plan is None or plan.claim not in ("fact", "correct", "misconception")):
            st.pending_contribution = None    # no afterthought to a statement that was not sent
        if plan is None:            # no wording passes the checks and nothing else to say: quiet
            st.skipped_turns += 1
            if log is not None:   # kept aside (their calls were paid), not as a turn record
                if not hasattr(self.realizer, "skipped_log"):
                    self.realizer.skipped_log = []
                self.realizer.skipped_log.extend(log[n0:])
                del log[n0:]
            return None
        if log is not None and len(log) > n0 + 1:   # one realizer record per turn (the last one, + extras)
            recs = log[n0:]
            final = dict(recs[-1])
            final["pre_guard_leak"] = any(r.get("pre_guard_leak", False) for r in recs)
            # realize calls of the earlier records (realism-guard regeneration), counted, not lost
            final["realism_extra_realize_calls"] = sum(
                r.get("attempts", 0) + r.get("repetition_guard", {}).get("extra_realize_calls", 0) for r in recs[:-1])
            for r in recs[:-1]:
                if "repetition_guard" in r and "repetition_guard" not in final:
                    final["repetition_guard"] = {**r["repetition_guard"], "extra_realize_calls": 0}
            log[n0:] = [final]
        text = self._address_and_lead(plan, sid, text)
        typo = getattr(self, "_last_typo", None) if not plan.split_tail else None
        self._last_typo = None
        text = self._maybe_split(plan, sid, text)
        self._maybe_typo_fix(plan, sid, text, typo)
        msg = Message(id=len(st.messages), t=st.t, speaker=sid, kind="student", text=text,
                      addressee=plan.addressee, plan=plan)
        st.messages.append(msg)
        st.student_turns += 1
        if plan.addressee and plan.addressee in st.roster:
            st.addressed_at[plan.addressee] = st.t
        self._age_addresses()
        if plan.addressee in st.roster and plan.addressee != sid and (
                (plan.act == "question" and not plan.tag and not plan.nudge)
                or (plan.act == "press" and plan.addresses_misconception)):
            # 0.5.2: a classmate asked by name / pressed for a reason is expected to answer (planner._respond_to_peer)
            st.open_address[plan.addressee] = OpenAddress(msg_id=msg.id, t=st.t, source=sid,
                                                          anchor_turn=plan.anchor_turn)
        self.updater.apply(st, msg, self.rngs["update"])
        if st.task_complete and st.solved_at is None:
            st.solved_at = st.t
        self._update_missing()
        self._closing_step(msg)
        return msg

    def _nudge(self, sid: str):
        """A forced turn whose content could not be worded: a short nudge to the group (template)."""
        st, s, rng = self.state, self.state.student(sid), self.rngs["realize"]
        plan = self.planner.nudge_plan(st, s, rng)
        tr = TemplateRealizer()
        text = tr.realize(plan, s, self.task, rng, self.names, st=st)
        for _ in range(6):
            if realism.check(text, st, sid, self.task, self.names) is None:
                break
            text = tr.realize(plan, s, self.task, rng, self.names, st=st)
        st.slippage_log.append({"type": "silence_cap", "msg": len(st.messages), "speaker": sid, "final": text})
        return plan, text

    def _age_addresses(self) -> None:
        st = self.state
        for oa in st.open_address.values():
            oa.turns_waited += 1
        st.open_address = {k: v for k, v in st.open_address.items() if v.turns_waited <= OPEN_ADDRESS_TURNS}

    def _update_missing(self) -> None:
        """Card facts of the current step's unvoiced key ideas (the scheduler's stall boost reads it)."""
        st = self.state
        st.context["missing_facts"] = sorted({f for ki in self.planner.missing_ideas(st) for f in ki.facts})
        # 0.5.1: holders of a belief slated to surface on this step (the scheduler boosts them)
        step = self.task.steps[st.step_idx].id if not st.task_complete else None
        st.context["slate_holders"] = sorted({sl["student"] for sl in conflict.pending(st, step)}) if step else []
        und = conflict.undiscussed(st, step) if step else []
        st.context["slate_holders"] += sorted({s.id for s in st.students for b in und if b["student"] != s.id
                                               and self.planner._can_correct(s, s.misconception(b["m"]).kc
                                                                             if s.misconception(b["m"]) else
                                                                             self.task.misconception(b["m"]).kc)[0]})

    # =============================================================== closing synthesis
    def _closing_step(self, msg: Message) -> None:
        st = self.state
        if st.closing is None:
            if self.cfg.end_on_complete and st.task_complete:
                st.closing = "propose"
            return
        p = msg.plan
        if p is not None and p.closing == "propose" and st.closing == "propose":
            st.closing, st.closing_msgs = "confirm", [msg.id]
            st.closing_responders = self._closing_responders(msg)
        elif p is not None and p.closing in ("agree", "adjust") and st.closing == "confirm":
            st.closing_msgs.append(msg.id)
            if msg.speaker in st.closing_responders:
                st.closing_responders.remove(msg.speaker)
            if not st.closing_responders:
                st.closing = "done"

    def _closing_responders(self, prop: Message) -> list[str]:
        """Who answers the closing proposal (0.5.1): with 3 or more students, at least
        cfg.closing_min_responders others (agree / adjust / a small objection), and every other student
        who is in the chat with P = their engagement; with 2 students, the other one."""
        st, rng = self.state, self.rngs["closing"]
        others = [s for s in st.students if s.id != prop.speaker]
        if not others:
            return []
        here = [s for s in others if s.traits.lurk_until_min * 60.0 <= st.t] or others
        need = 1 if len(st.students) < 3 else min(self.cfg.closing_min_responders, len(here))
        pool = list(here)
        chosen: list[str] = []
        # 0.5.3: a student still holding a belief they voiced (never given up in the chat) is less eager to
        # confirm the final answer (and hedges when they do)
        holds = {x.id for x in pool if self.planner.open_beliefs(st, x.id)}
        for _ in range(need):
            w = [(x.affect.engagement * x.traits.talkativeness + 0.05) * (0.3 if x.id in holds else 1.0) for x in pool]
            pick = rng.weighted(pool, w)
            pool.remove(pick)
            chosen.append(pick.id)
        chosen += [x.id for x in pool if rng.bernoulli(min(1.0, x.affect.engagement) * (0.5 if x.id in holds else 1.0))]
        return chosen

    # =============================================================== final-line checks
    def _realism_final(self, plan, sid: str, text: str):
        """Final-line realism checks (session-wide near-duplicates across all speakers, role words /
        "@" / names of participants who have not written yet, physical classroom / wrong time of
        day): one regeneration with a note, then template variants, then a different (content-free)
        plan; then laughter / typo polish."""
        from .. import lang
        st, s, rng = self.state, self.state.student(sid), self.rngs["realize"]
        hit = realism.check(text, st, sid, self.task, self.names, plan)
        if hit is not None:
            info = {"type": "realism_guard", "msg": len(st.messages), "speaker": sid, "act": plan.act,
                    "reason": hit["reason"], "first_draft": text}
            new = self.realizer.realize(plan, s, self.task, rng, self.names, st=st,
                                        rephrase=realism.rephrase_note(hit, self.task))
            new = self._finalize(plan, sid, new)
            if realism.check(new, st, sid, self.task, self.names, plan) is None:
                text, info["resolution"] = new, "regenerated"
            else:
                for _ in range(6):
                    v = self.realizer.variant(plan, s, self.task, rng, self.names, st=st)
                    if realism.check(v, st, sid, self.task, self.names, plan) is None:
                        text, info["resolution"] = v, "template_variant"
                        break
                else:
                    v = self._credit_line(plan, s, hit) if plan.claim in ("correct", "fact", "misconception") else None
                    if plan.closing in ("agree", "adjust"):
                        # the answer to the closing proposal stays one: a short okay instead
                        plan = plan.model_copy(update={"closing": "agree", "brief": True, "fact_ids": [],
                                                       "act": "agree", "target_len": 1})
                        v = TemplateRealizer().realize(plan, s, self.task, rng, self.names, st=st)
                        text, info["resolution"] = v, "closing_brief"
                    elif v is not None and realism.check(v, st, sid, self.task, self.names, plan) is None:
                        # the content (a correct claim / fact the plan voices) was already said: the turn
                        # credits the peer instead of reciting it again; the plan is kept for the updater
                        text, info["resolution"] = v, "credit_peer"
                        plan.revoice = True
                    else:
                        # another plan: content first (a progress move, a claim, a question) under the
                        # same gates; a content-free one only if the filler caps allow it; else the
                        # student stays quiet (never an unbounded run of "idk" lines)
                        alt = self.planner.substitute(st, s, rng)
                        fb = self.planner.fallback_plan(st, s, rng)
                        if alt is None and self.planner.filler_blocked(st, s, fb) is None:
                            alt = fb
                        plan = alt
                if info.get("resolution") is None and plan is None:
                    info["resolution"], info["final"] = "skipped", None
                    st.slippage_log.append(info)
                    return None, None
                if info.get("resolution") is None:
                    tr = TemplateRealizer()
                    v = text
                    for _ in range(6):
                        v = tr.realize(plan, s, self.task, rng, self.names, st=st)
                        if realism.check(v, st, sid, self.task, self.names, plan) is None:
                            break
                    else:  # a short atom (below the near-duplicate length) of the fallback act
                        if self.planner.is_content_free_plan(plan):
                            F = lang.attr(self.task, "FILLER", {})
                            atoms = [a for a in F.get(plan.act, []) + F.get("backchannel", [])
                                     if len(realism._tokens(a)) < realism.NEAR_DUP_MIN_TOKENS] or ["ok"]
                            from .realizer import fresh_forms
                            v = rng.choice(fresh_forms(atoms, st))
                        else:
                            info["resolution"], info["final"] = "skipped", None
                            st.slippage_log.append(info)
                            return None, None
                    text, info["resolution"] = v, "different_plan"
                    info["new_act"] = plan.act
            if info.get("resolution") in ("different_plan", "template_variant", "regenerated"):
                text, fr = realism.cap_framing(text, st, self.task, self.cfg, rng)   # the new line too (0.5.1)
                if fr:
                    st.slippage_log.append({"type": "framing_guard", "msg": len(st.messages), "speaker": sid, **fr})
            info["final"] = text
            st.slippage_log.append(info)
        text, pol = realism.polish(text, plan, s, st, self.task, self.rngs["realism"])
        if pol:
            st.slippage_log.append({"type": "realism_polish", "msg": len(st.messages), "speaker": sid, **pol})
        self._last_typo = pol.get("typo_word") if pol and pol.get("typo") in ("double", "drop", "key") else None
        text = self._not_own_repeat(plan, sid, text)
        return plan, text

    def _not_own_repeat(self, plan, sid: str, text: str) -> str:
        """Never the exact same line twice in a row by one student (whatever step produced it)."""
        from . import repetition as rep
        from .. import lang
        from .realizer import FILLER
        own = next((m for m in reversed(self.state.messages) if m.speaker == sid), None)
        if own is None or rep.norm(own.text) != rep.norm(text):
            return text
        rng = self.rngs["realize"]
        F = lang.attr(self.task, "FILLER", FILLER)
        opts = [o for o in F.get(plan.act, []) + F.get("backchannel", []) if rep.norm(o) != rep.norm(text)]
        if len(rep.norm_tokens(text)) < rep.MIN_TOKENS and opts:
            new = rng.choice(opts)
        else:
            new = rng.choice(lang.attr(self.task, "REPEAT_PREFIXES", ["wait ", "so ", "ok ", "like "])) + text
        self.state.slippage_log.append({"type": "repetition_guard", "msg": len(self.state.messages), "speaker": sid,
                                        "act": plan.act, "reason": "own_repeat_in_a_row", "prev": own.text,
                                        "prev_id": own.id, "first_draft": text, "regenerated": False,
                                        "resolution": "final_variant", "final": new})
        return new

    def _credit_line(self, plan, s, hit: dict) -> str | None:
        from .realizer import CREDIT_LINES
        from .. import lang
        prev = self.state.messages[hit["prev_id"]] if hit.get("prev_id") is not None else None
        if prev is None or prev.speaker == s.id or prev.speaker not in self.names:
            return None
        if prev.plan is not None and (prev.plan.ref_by == s.id or prev.plan.act == "question"):
            return None       # 0.5.3: they were quoting this student (no circular credit) / only asking
        if any(m.speaker == s.id and m.plan is not None and m.plan.revoice for m in self.state.messages):
            return None       # one credit line per student and session
        if plan.claim == "misconception":
            # a holder siding with a peer who voiced the same belief ("eu tbm acho isso...")
            if prev.plan is None or prev.plan.misconception_id != plan.misconception_id:
                return None
            from .realizer import BELIEF_CREDIT_LINES
            bank = lang.attr(self.task, "BELIEF_CREDIT_LINES", BELIEF_CREDIT_LINES)
            return self.rngs["realize"].choice(bank).format(peer=self.names[prev.speaker].lower())
        bank = lang.attr(self.task, "CREDIT_LINES", CREDIT_LINES)
        return self.rngs["realize"].choice(bank).format(peer=self.names[prev.speaker].lower())

    def _word(self, plan, sid: str) -> str:
        """Realize a plan under the deterministic repetition guard (sim/repetition.py): a line too
        similar to one of the last 15 turns is regenerated once with "say it differently", else
        replaced by another template variant. Wording only; every hit is logged."""
        from . import repetition as rep
        st, s, rng = self.state, self.state.student(sid), self.rngs["realize"]
        log = getattr(self.realizer, "log", None)
        n0 = len(log) if log is not None else 0
        text = self.realizer.realize(plan, s, self.task, rng, self.names, st=st)
        hit = rep.check(text, st, sid)
        if hit is None:
            return text
        info = {"type": "repetition_guard", "msg": len(st.messages), "speaker": sid, "act": plan.act,
                **hit, "first_draft": text, "regenerated": True}
        cands = [text]
        from .. import lang
        text = self.realizer.realize(plan, s, self.task, rng, self.names, st=st,
                                     rephrase=lang.attr(self.task, "REPHRASE", rep.REPHRASE).format(prev=hit["prev"]))
        cands.append(text)
        if rep.check(text, st, sid) is None:
            info["resolution"] = "regenerated"
        else:
            info["resolution"] = "unresolved"
            for _ in range(8):
                v = self.realizer.variant(plan, s, self.task, rng, self.names, st=st)
                cands.append(v)
                if rep.check(v, st, sid) is None:
                    text, info["resolution"] = v, "template_variant"
                    break
            if info["resolution"] == "unresolved":
                ok = [c for c in cands if (rep.check(c, st, sid) or {}).get("reason") != "own_repeat_in_a_row"]
                if ok:
                    text = min(ok, key=lambda c: rep.max_similarity(c, st))
                else:  # never the exact same string twice in a row by one student
                    text = rng.choice(lang.attr(self.task, "REPEAT_PREFIXES", ["wait ", "so ", "ok ", "like "])) + cands[-1]
        info["final"] = text
        st.slippage_log.append(info)
        if log is not None and len(log) > n0:
            recs = log[n0:]
            # one record per turn: the leak-guard fields of the LAST realize call (its draft is the
            # one the audit labels), plus the guard's own record; extra calls are counted there
            final = dict(recs[-1])
            final["pre_guard_leak"] = any(r.get("pre_guard_leak", False) for r in recs)
            if hasattr(self.realizer, "leaked_facts"):
                final["post_guard_leak"] = bool(self.realizer.leaked_facts(self.task, s, plan, text, st))
            final["repetition_guard"] = {k: info[k] for k in ("reason", "prev_id", "resolution", "regenerated")}
            final["repetition_guard"]["extra_realize_calls"] = sum(r.get("attempts", 0) for r in recs[:-1])
            if "audit" in final and info["resolution"] != "regenerated":
                final["audit_of"] = "llm_draft"  # final wording is a template variant of the same plan
            log[n0:] = [final]
        return text

    def _finalize(self, plan, sid: str, text: str) -> str:
        """Checks on EVERY final line, whichever realizer produced it:
        prompt leak: prompt labels ("ask the group:", "say:", plan labels, ...) -> another variant of the
               same plan, else stripped; logged;
        belief leak: a line that voices a misconception the plan does not voice -> a template wording
               of the plan (the planner decides when a belief is voiced, never the wording);
        framing: one framing lead ("mine says...", "no meu card diz que...") at most a few times per
               session, the word "card" at most once;
        attribution: a false "like i said" (the speaker never said it) credits whoever did, else is dropped;
        visible echo: a fact the speaker echoed from the visible chat becomes known (they read it)."""
        from . import promptleak
        from .content import fix_peer_attribution, fix_self_attribution, learn_visible_echoes, strip_self_reference
        st, s, rng = self.state, self.state.student(sid), self.rngs["realize"]
        log = getattr(self.realizer, "log", None)
        rec = log[-1] if log and log[-1].get("msg") == len(st.messages) else None
        h = promptleak.hits(text, self.names, sid)
        if h:
            info = {"type": "prompt_leak_guard", "msg": len(st.messages), "speaker": sid, "act": plan.act,
                    "hits": h, "draft": text}
            for _ in range(8):
                v = self.realizer.variant(plan, s, self.task, rng, self.names, st=st)
                if not promptleak.hits(v, self.names, sid):
                    text, info["resolution"] = v, "template_variant"
                    break
            else:
                text, info["resolution"] = promptleak.strip(text, self.names, sid), "stripped"
            info["final"] = text
            st.slippage_log.append(info)
            if rec is not None:
                rec["final_prompt_leak"] = {k: info[k] for k in ("hits", "draft", "resolution")}
        if plan.claim != "misconception":
            voiced = [m for m in self.task.misconceptions_expressed(text) if s.misconception(m) is not None]
            if voiced:
                new = TemplateRealizer().realize(plan, s, self.task, rng, self.names, st=st)
                if set(voiced) - set(self.task.misconceptions_expressed(new)):   # the wording added a belief
                    st.slippage_log.append({"type": "belief_guard", "msg": len(st.messages), "speaker": sid,
                                            "draft": text, "misconceptions": voiced, "final": new})
                    text = new
        text = self._coherence_guard(plan, sid, text, rec)
        text, fr = realism.cap_framing(text, st, self.task, self.cfg, rng)
        if fr:
            st.slippage_log.append({"type": "framing_guard", "msg": len(st.messages), "speaker": sid, **fr})
        text, fix = fix_self_attribution(self.task, st, sid, plan, text, self.names)
        if fix is None:
            text, fix = strip_self_reference(self.task, st, sid, plan, text)
        if fix is None:
            text, fix = fix_peer_attribution(self.task, st, sid, plan, text, self.names)
        if fix is not None:  # "like i said" only for the speaker's own earlier statement
            st.slippage_log.append({"type": "attribution_guard", "msg": len(st.messages), "speaker": sid, **fix})
            if rec is not None:
                rec["attribution_fix"] = fix
        if hasattr(self.realizer, "leaked_facts"):  # no final line carries a true leak
            leak = self.realizer.leaked_facts(self.task, s, plan, text, st)
            if leak:
                st.slippage_log.append({"type": "final_true_leak", "msg": len(st.messages), "speaker": sid,
                                        "text": text, "facts": leak})
                if rec is not None:
                    rec["final_true_leak"] = {"text": text, "facts": leak}
                text = self.realizer.minimal_line(self.task, s, plan, rng, self.names, st, rejected=[text])
            if rec is not None:
                rec["post_guard_leak"] = bool(self.realizer.leaked_facts(self.task, s, plan, text, st))
        learned = learn_visible_echoes(self.task, st, s, plan, text)
        if learned:
            st.slippage_log.append({"type": "visible_echo_learned", "msg": len(st.messages), "speaker": sid,
                                    "facts": learned})
            if rec is not None:
                rec["visible_echo_learned"] = learned
        return text

    def _address_and_lead(self, plan, sid: str, text: str) -> str:
        """0.5.3: in a dyad every line addresses the one partner (no "galera" / "vcs" / "alguem"); the content
        line after a one-word opener does not repeat the lead ("olha" | "olha, pelo que eu li ...")."""
        st = self.state
        new = text
        if realism.is_dyad(st):
            other = next((x for x in st.roster if x != sid), "")
            new = realism.singularize(new, self.task, self.names.get(other, other))
        prev = st.messages[-1] if st.messages else None
        if prev is not None and prev.speaker == sid and prev.plan is not None and prev.plan.opener and not plan.opener:
            new = realism.strip_repeated_lead(new, prev.text, self.task)
        if new != text:
            st.slippage_log.append({"type": "address_guard", "msg": len(st.messages), "speaker": sid,
                                    "draft": text, "final": new})
        return new or text

    def _maybe_typo_fix(self, plan, sid: str, text: str, typo) -> None:
        """0.5.3: a misspelled word (a doubled / dropped / slipped letter) is sometimes corrected by the same student
        a few seconds later ("*superficie"), as students do in chats."""
        st = self.state
        if not typo or st.pending_contribution is not None or plan.closing is not None or len(st.roster) < 2:
            return
        if plan.act not in ("on_task_claim", "explain", "disagree", "question", "press", "interface") or \
                self.planner.is_content_free_plan(plan):
            return              # (a filler or an off-task remark is not worth correcting)
        orig, bad = typo
        if bad not in text or len(orig) < 5 or not self.rngs["realism"].bernoulli(self.cfg.p_typo_fix):
            return
        who, run = self.scheduler.current_run(st)
        if who == sid and run + 2 > self.cfg.max_run_hard_cap:
            return
        from . import promptleak
        if promptleak.hits(f"*{orig.lower()}", self.names, sid):
            return
        tp = ContentPlan(act="explain", split_tail=f"*{orig.lower()}", typo_fix=True, continuation=True, target_len=1)
        st.pending_contribution = {"speaker": sid, "plan": tp.model_dump(), "after": True}

    def _maybe_split(self, plan, sid: str, text: str) -> str:
        """0.5.2 split thoughts: a content line is sometimes typed as two messages by the same student
        ("a amilase quebra o amido" / "entao fica doce"), cut at a comma or a connector. The first half
        carries the plan; the second follows a few seconds later (pending_contribution, a template turn).
        Keeps P(same speaker) in its pilot band now that tag questions are capped."""
        st = self.state
        if (st.pending_contribution is not None or plan.closing is not None or plan.draft is not None
                or plan.claim not in ("fact", "correct", "misconception") or plan.reference or plan.split_tail
                or len(st.roster) < 2 or not self.rngs["realism"].bernoulli(self.cfg.p_split_thought)):
            return text
        who, run = self.scheduler.current_run(st)
        if who == sid and run + 2 > self.cfg.max_run_hard_cap:
            return text           # the second half would make the run longer than the cap
        # 0.5.3: rarer -- each student splits at most split_per_student_max thoughts per session
        mine = sum(1 for m in st.messages if m.speaker == sid and m.plan is not None and m.plan.split_tail
                   and not m.plan.typo_fix)
        if mine >= self.cfg.split_per_student_max:
            return text
        head, tail = realism.split_point(text, self.task)
        if not tail or realism.bad_split(head, tail, self.task):
            return text
        tp = ContentPlan(act="explain", split_tail=tail, continuation=True,
                         target_len=len(tail.split()))
        st.pending_contribution = {"speaker": sid, "plan": tp.model_dump(), "after": True}
        st.slippage_log.append({"type": "split_thought", "msg": len(st.messages), "speaker": sid, "head": head,
                                "tail": tail})
        return head

    def _coherence_guard(self, plan, sid: str, text: str, rec: dict | None = None) -> str:
        """0.5.2 code coherence on EVERY final line (sim/coherence.py): a belief line that states its own
        refutation, a refuting line that asserts the belief, "discordo" / "acho que nao" without a claim
        the speaker opposes, or a disagreement repeating its target -> the disagreement lead is dropped, or
        another template wording of the same plan (whose content the planner chose to refute the belief)."""
        from . import coherence, promptleak
        st, s, rng = self.state, self.state.student(sid), self.rngs["realize"]
        issue = coherence.coherence_issue(self.task, st, sid, plan, text)
        if issue is None:
            return text
        info = {"type": "coherence_guard", "msg": len(st.messages), "speaker": sid, "act": plan.act, **issue,
                "draft": text}

        def ok(c):
            return bool(c) and coherence.coherence_issue(self.task, st, sid, plan, c) is None and \
                not promptleak.hits(c, self.names, sid)
        new = None
        if issue["reason"] in ("disagree_without_opposition", "disagree_restating"):
            c = coherence.strip_disagree(text, self.task)
            if c != text and ok(c):
                new, info["resolution"] = c, "lead_dropped"
        if new is None:
            tr = TemplateRealizer()
            for _ in range(8):
                c = tr.realize(plan, s, self.task, rng, self.names, st=st)
                if ok(c):
                    new, info["resolution"] = c, "template_variant"
                    break
        if new is None:
            c = tr.safe_line(plan, s, self.task, rng, self.names, st, ok=ok)
            new, info["resolution"] = (c, "safe_line") if ok(c) else (text, "unresolved")
        info["final"] = new
        st.slippage_log.append(info)
        if rec is not None:
            rec["coherence_guard"] = {k: info[k] for k in ("reason", "resolution")}
        return new

    # =============================================================== external participants
    def add_participant(self, pid: str, name: str, avatar: str | None = None) -> dict:
        """Register an external participant (no hidden state). Students may address or name them only
        after they have written in the chat."""
        d = self.state.participants.setdefault(pid, {"name": name, "avatar": avatar, "first_spoke_t": None})
        self.names[pid] = d["name"]
        return d

    def inject_external(self, pid: str, text: str, plan, t: float | None = None) -> Message:
        """An external participant's message, inserted at sim time `t` (>= the current clock). `plan` is
        the code-validated interpretation of the text (runtime.interpret); the normal state-update rules
        then apply to it like to a peer turn (students may learn from it, be challenged by it, or
        disagree). External turns do not count toward max_student_turns."""
        st = self.state
        if pid not in st.participants:
            self.add_participant(pid, pid)
        st.t = max(st.t, t if t is not None else st.t)
        st.pending_contribution = None   # planned before this message: dropped (the opener stands alone)
        msg = Message(id=len(st.messages), t=st.t, speaker=pid, kind="external", text=text,
                      addressee=plan.addressee, plan=plan)
        st.messages.append(msg)
        d = st.participants[pid]
        if d.get("first_spoke_t") is None:
            d["first_spoke_t"] = st.t
        self._age_addresses()
        to = plan.addressee
        if to and to in st.roster:
            st.addressed_at[to] = st.t
            if plan.act in ADDRESS_ACTS:
                st.open_address[to] = OpenAddress(msg_id=msg.id, t=st.t, source=pid, anchor_turn=plan.anchor_turn)
        elif to is None and plan.act in ("question", "press") and not plan.asked_fact:
            st.open_address["group"] = OpenAddress(msg_id=msg.id, t=st.t, source=pid)
        self.updater.apply(st, msg, self.rngs["update"])
        if st.task_complete and st.solved_at is None:
            st.solved_at = st.t
        self._update_missing()
        self._closing_step(msg)
        st.planned = self.scheduler.plan_next(st, st.t, self.rngs["sched"])
        return msg

    def inject_student(self, sid: str, plan, text: str, delay: float = 5.0) -> Message:
        """Insert a student message with a given plan and wording (test hook)."""
        st = self.state
        st.t += delay
        st.pending_contribution = None   # planned before this message: dropped (as for an external message)
        msg = Message(id=len(st.messages), t=st.t, speaker=sid, kind="student", text=text,
                      addressee=plan.addressee, plan=plan)
        st.messages.append(msg)
        st.student_turns += 1
        if plan.addressee in st.roster:
            st.addressed_at[plan.addressee] = st.t
        self.updater.apply(st, msg, self.rngs["update"])
        if st.task_complete and st.solved_at is None:
            st.solved_at = st.t
        self._update_missing()
        st.planned = self.scheduler.plan_next(st, st.t, self.rngs["sched"])
        return msg

    def run_student_turns(self, k: int, max_events: int = 10_000) -> list[Message]:
        start = self.state.student_turns
        n0 = len(self.state.messages)
        n = 0
        while not self.state.done and self.state.student_turns < start + k and n < max_events:
            self.step()
            n += 1
        return self.state.messages[n0:]

    # =============================================================== results
    def result(self) -> dict:
        st = self.state
        log = st.slippage_log
        return {
            "session_id": st.session_id, "task_id": st.task_id, "seed": st.seed, "domain": self.task.domain,
            "learning_params": self.params.model_dump(mode="json"),
            "realizer": type(self.realizer).__name__,
            "config": self.cfg.model_dump(mode="json"),
            "duration_s": round(st.t, 2), "n_student_turns": st.student_turns,
            "solved_at_min": round(st.solved_at / 60.0, 3) if st.solved_at is not None else None,
            "ended_reason": st.ended_reason,
            "roster": [{"id": s.id, "name": s.name, "role": s.role} for s in st.students],
            "participants": [{"id": p, "name": d["name"], "avatar": d.get("avatar"),
                              "first_spoke_t": d.get("first_spoke_t")} for p, d in st.participants.items()],
            "transcript": [{"id": m.id, "t": round(m.t, 2), "speaker": m.speaker,
                            "display_name": self.names.get(m.speaker, m.speaker),
                            "kind": m.kind, "text": m.text, "addressee": m.addressee,
                            "plan": m.plan.model_dump() if m.plan else None,
                            "labels": {k: v for k, v in m.labels.items() if k != "flips"}}
                           for m in st.messages],
            "hidden_state_initial": self.initial_students,
            "hidden_state_final": [s.model_dump() for s in st.students],
            "hidden_state_timeline": st.timeline,
            "flip_log": st.flip_log,
            "steps_solved": st.steps_solved,
            "repetition_guard_log": [r for r in log if r.get("type") == "repetition_guard"],
            "prompt_leak_guard_log": [r for r in log if r.get("type") == "prompt_leak_guard"],
            "attribution_guard_log": [r for r in log if r.get("type") == "attribution_guard"],
            "visible_echo_log": [r for r in log if r.get("type") == "visible_echo_learned"],
            "realism_guard_log": [r for r in log if r.get("type") == "realism_guard"],
            "filler_gate_log": [r for r in log if r.get("type") == "filler_gate"],
            "framing_guard_log": [r for r in log if r.get("type") == "framing_guard"],
            "belief_guard_log": [r for r in log if r.get("type") == "belief_guard"],
            "coherence_guard_log": [r for r in log if r.get("type") in ("coherence_guard", "coherence_gate")],
            "context": st.context, "closing_msgs": st.closing_msgs, "closing_forced": st.closing_forced,
        }


def transcript_for_stats(st: SessionState) -> list[dict]:
    """Message dicts for the structural statistics (external participants' turns are flagged)."""
    return [{"speaker": m.speaker, "t": m.t, "text": m.text, "is_external": m.kind != "student"}
            for m in st.messages]


def run_session(task: str, seed: int, cfg: SimConfig | None = None, n_students: int | None = None) -> Simulation:
    sim = Simulation(task, seed, cfg, n_students=n_students)
    sim.run()
    return sim
