"""Simulator (behavioural) configuration: turn-taking, act prior, replies to an address, length
model, task progress and the realism controls. Distinct from the learning parameters
(sim/learning.py). Defaults are the configuration a `Simulator` session runs ("pilot-calibrated":
structural statistics are checked against aggregate targets from a private pilot corpus)."""
from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Pace = Literal["pilot", "brisk"]
Conflict = Literal["low", "normal", "high"]

# 0.5.1 pacing presets. "pilot" (default) reproduces the pilot sessions' length (median ~34 min,
# 12-19 messages per student); "brisk" is a short demo session (~15-20 min). A preset only fills the
# fields the caller did not set explicitly.
PACE_PRESETS: dict[str, dict] = {
    "pilot": {"target_minutes": 36.0, "session_minutes": 48.0},
    # brisk: shorter dwell, quicker turns, fewer restatements per idea and faster progress pressure
    "brisk": {"target_minutes": 14.0, "session_minutes": 25.0, "gap_median_s": 22.0, "burst_gap_median_s": 9.0,
              "min_correct_claims_per_kc": 2, "p_afterthought": 0.4, "stall_turns": 4, "stall_s": 90.0},
}
# 0.5.1 conflict presets: how many held misconceptions are SLATED to surface (voiced by a student who
# holds it, then discussed) and how persistent they are.
#   low    - 0.5.0 behaviour: a belief comes up only if its holder happens to claim on that step
#   normal - at least 1, typically 2 voiced and challenged per session (default)
#   high   - every held belief surfaces; beliefs are stronger, pushed back on up to twice, and a step
#            stays blocked until the holder changes their mind (the group may stall)
CONFLICT_PRESETS: dict[str, dict] = {
    "low": {"slated_misconceptions": 0, "conflict_resolution_required": False, "max_pushbacks": 1},
    "normal": {"slated_misconceptions": 2, "conflict_resolution_required": False, "max_pushbacks": 1},
    "high": {"slated_misconceptions": 99, "conflict_resolution_required": True, "max_pushbacks": 2,
             "p_pushback": 0.85, "slate_grace_s": 900.0},
}


class SimConfig(BaseModel):
    # ---------------------------------------------------------------- 0.5.1 presets
    pace: Pace = "pilot"
    conflict: Conflict = "normal"
    # session stop: sim-minutes time cap and a safety cap on student turns (pace preset: pilot 45)
    session_minutes: float = 48.0
    max_student_turns: int = 200
    # when every task step is solved the group runs its closing synthesis and the chat ends
    end_on_complete: bool = True
    # gap model: lognormal; bursts (same speaker) use a shorter distribution
    gap_median_s: float = 50.0         # 0.5.0: 22 -> 32; 0.5.1: 50 with silence_boost 2.2 (calibration_051)
    gap_sigma: float = 1.3
    burst_gap_median_s: float = 12.0   # 0.5.0: 8 -> 10; 0.5.1: 14; 0.5.3: 12
    burst_gap_sigma: float = 0.95
    # monologue cap: P(keep the floor) x burst_decay ** (run - 1); hard cap on the same-speaker run
    burst_decay: float = 0.7
    burst_scale: float = 4.0      # re-centres P(same speaker) inside its pilot band (0.5.0: 1.35 -> 3.0; 0.5.1: 4.0)
    max_run_hard_cap: int = 4     # 0.5.0: 3 -> 4 (pilot max run 7, p95 3)
    burst_after_question: float = 1.0     # x P(keep the floor) right after one's own question (never a filler)
    burst_after_external: float = 0.3     # x P(keep the floor) right after an external participant's message
    gap_rate_exponent: float = 0.5        # gap scaled by (Lambda_ref / Lambda) ** exponent
    lambda_ref: float = 3.0
    silence_boost_after_s: float = 60.0   # hazard x1.5 after >60 s silence
    silence_boost: float = 2.2          # 0.5.1: 1.5 -> 2.2 (long silences are rarer than mid-length ones)
    addressed_boost: float = 3.0
    addressed_decay_s: float = 30.0
    interface_factor: float = 0.5
    interface_block_s: tuple[float, float] = (30.0, 90.0)
    student_latency_sigma: float = 0.35
    # act prior (pilot act mix); weights, normalised at sampling time
    act_prior: dict[str, float] = Field(default_factory=lambda: {
        "on_task_claim": 17.0, "explain": 5.0, "agree": 8.0, "disagree": 3.0, "press": 3.0,
        "question": 15.0, "interface": 9.5, "backchannel": 8.6, "idk": 4.0,
    })
    p_off_base: float = 0.13
    p_off_given_off: float = 0.37
    p_greeting_first_msg: float = 0.7
    # a student addressed by name by an external participant (pilot-calibrated: 8/13 reply, about half
    # of the replies substantive); the same for every participant, whoever they are
    p_reply_when_addressed: float = 8 / 13
    p_substantive_reply: float = 0.5
    p_group_prompt_uptake: float = 0.35
    # P(the addressee takes the next floor); calibrated to the pilot "addressee replies within 2 student
    # messages" = 9/13 (0.5.0: .30 -> .60 with the longer same-speaker runs; tests/test_participants.py)
    p_addressee_takes_floor: float = 0.60
    # length model: lognormal(ln 4, .75) * verbosity; one-word atoms; hard cap
    len_mu: float = math.log(4.0)
    len_sigma: float = 0.75
    p_one_word: float = 0.2
    max_words: int = 30
    # task progress (rule 6): correct claims per gold KC needed since step start
    min_correct_claims_per_kc: int = 3
    min_distinct_speakers: int = 2
    # a step also needs every key idea voiced correctly, and no voiced, never-challenged active
    # misconception tied to it (rule 6; updater.rule6_progress)
    require_key_ideas: bool = True
    block_unchallenged_misconception: bool = True
    # misconception voicing / peer correction
    p_peer_correction: float = 0.5
    # peer responsiveness: a student who can answer a pending content question gets a hazard boost
    # 1 + (answer_boost - 1) * engagement * exp(-dt / answer_decay_s) and answers with probability
    # p_answer_holder * (0.5 + 0.5 * engagement) (p_answer_named if the question named them)
    answer_boost: float = 2.5
    answer_decay_s: float = 45.0
    p_answer_holder: float = 0.8
    p_answer_named: float = 0.9
    # cap on the initial lurk (pilot prior: 20% lurk U(5, 30) min). None = unchanged.
    lurk_cap_minutes: float | None = 10.0
    # a student who has not spoken yet gets hazard x (1 + this * minutes since their lurk ended), so in
    # large groups (the pilot covers 2-4) nobody stays silent all session. None = off.
    unspoken_boost_per_min: float | None = 1.0
    # ---------------------------------------------------------------- realism
    offtask_rate: float = Field(default=0.15, ge=0.0, le=0.5)   # target share of off-task student turns
    offtask_min_spacing: int = 3        # student turns between off-task turns (a thread reply is exempt)
    offtask_thread_max: int = 3         # off-task turns in one thread (topic + replies)
    time_of_day: str | None = None      # "manha" | "tarde" | "noite"; None: drawn once per session
    setting: str = "online"             # "online": no physical classroom / teacher-in-the-room talk
    laughter_cap: float = 0.15          # max share of student messages with laughter
    closing_wrap_up_max_minutes: float = 5.0  # completion: closing synthesis must happen within this
    # ---------------------------------------------------------------- 0.5.0 realism controls
    # fillers (pilot aggregates: "don't know" lines ~2%, backchannels ~7% of student messages, never
    # more than 2 content-free messages in a row)
    idk_share_cap: float = 0.05         # session share of idk / refusal turns
    idk_per_student_max: int = 2        # idk / refusal turns per student per session
    content_free_share_cap: float = 0.16  # session share of idk / refusal / backchannel / agree / hedge turns
    max_content_free_run: int = 2       # consecutive content-free turns, group-wide
    filler_form_max: int = 2            # uses of one filler wording per session
    # interface / procedural confusion: at most 1 + n // 2 interface and 2 + n // 2 confusions in all
    interface_cap_base: int = 1
    confusion_cap_base: int = 2
    confusion_cap_max: int = 8
    p_answer_interface: float = 0.8
    # progress pressure: the group is STALLED after stall_turns student turns or stall_s sim seconds
    # without task progress; a stalled turn becomes a progress move with p_progress_move
    stall_turns: int = 5
    stall_s: float = 150.0
    p_progress_move: float = 0.85
    stall_holder_boost: float = 2.0     # hazard x this for a student holding a missing key-idea fact
    # headroom: near the time / turn cap the group wraps up with what it established
    closing_headroom_min: float = 5.0
    closing_headroom_turns: int = 8
    # framing leads ("no meu card diz que..."): uses per session of one lead form; card leads in all
    lead_form_max: int = 2
    card_lead_max: int = 1
    # a contribution (a fact, a claim, a belief) is sometimes typed as two messages: a one-word opener
    # ("pera", "gente", "olha") and then the content, by the same student (pilot chats fragment this way)
    p_opener: float = 0.8        # 0.5.1: .6 -> .8; some openers are short leads ("pera achei aqui")
    p_opener_one_word: float = 0.8
    # 0.5.3: lead-word-only messages are rare: about one per opener_every student messages, at most
    # opener_per_student_max per student; each wording once per session, and the content line never repeats
    # the lead ("olha" | "olha, pelo que eu li..." -> "pelo que eu li...")
    opener_every: int = 15
    opener_per_student_max: int = 3
    # misconceptions: a student voices one at most once, plus at most one reasoned push-back after
    # it was challenged (or goes quiet on it)
    p_pushback: float = 0.5
    # ---------------------------------------------------------------- 0.5.1 realism controls
    # pacing: a step is solved no earlier than min_step_dwell_frac * target_minutes / n_steps after it
    # started (the pilot groups took ~10 min per activity); None/0 = no dwell
    target_minutes: float = 36.0
    min_step_dwell_frac: float = 0.65
    # group silence: never more than max_silence_s between two messages while the chat is running
    # (pilot gap p95 131 s, p99 285 s; 0.5.0 produced 8-minute dead silences). When the cap is hit
    # someone speaks, preferably with content. None = off.
    max_silence_s: float | None = 150.0
    # who talks most: hazard uses talkativeness ** talk_exponent (the pilot groups were dominated by
    # one or two students: top-speaker share ~.54, Gini ~.24)
    talk_exponent: float = 3.0
    # a content statement is sometimes followed by a short afterthought by the same student ("ne?",
    # "faz sentido?"): split thoughts keep P(same speaker) in its pilot band without consecutive facts
    p_afterthought: float = 0.7
    # information pacing: at most one NEW content statement (a card fact / a correct claim) per student
    # per fact_window_s sim seconds or fact_window_turns student turns, unless directly asked
    fact_window_s: float = 90.0
    fact_window_turns: int = 3
    # same-speaker runs: at most this many content turns in one run (openers and short follow-ups
    # do not count)
    max_content_run: int = 2
    # content lines: target length capped by register (verbosity trait); a realized content line
    # longer than content_max_words is regenerated shorter (LLM) or re-templated
    content_len_cap: float = 16.0
    content_max_words: int = 25
    # conflict (filled by the preset; see CONFLICT_PRESETS)
    slated_misconceptions: int = 2
    conflict_resolution_required: bool = False
    max_pushbacks: int = 1
    p_slated_voice: float = 0.6        # P(voice a slated belief) on a turn of its holder once the step is underway
    slated_voice_after_turns: int = 2  # student turns into the step before a slated belief comes up
    slate_grace_s: float = 360.0       # a slate that has not surfaced blocks the step at most this long past the dwell
    # closing: besides the proposer, at least this many others respond (n >= 3); every other present
    # student also responds with P = engagement
    closing_min_responders: int = 2
    # ---------------------------------------------------------------- 0.5.2 coherence controls
    # afterthought / tag questions ("vcs concordam?", "ne isso?"): at most this share of the session's
    # content claims (each wording once, never on two consecutive claims of one student)
    tag_share_cap: float = 0.15
    # peer name pings ("julia vc sabe?"): at most ping_cap_base per session for groups up to 4 (scaled by
    # n / 4 above), one per target, only to a student who has written at least ping_min_messages times;
    # the pinged student answers at the pilot reply rate (p_reply_when_addressed)
    ping_cap_base: int = 3
    ping_min_messages: int = 2
    # a holder asked "pq vc acha isso?" answers (a reason from the belief's `because`, giving in, or
    # unsure) with P = p_answer_press x (.6 + .4 x engagement)
    p_answer_press: float = 0.95
    # split thoughts: a content line with a comma / connector is sometimes typed as two messages by the same
    # student (keeps P(same speaker) in the pilot band with fewer tag questions)
    p_split_thought: float = 0.9
    # 0.5.3: only at a clause boundary (before "entao" / "por isso" / "tipo" / "mas" / "so" / "because"),
    # never mid-clause, and at most this many per student per session
    split_per_student_max: int = 3
    # a holder gives in only to a refuting move (fact / claim / counterexample, not a press) addressed to them, to
    # the group or at that belief, within this many student turns; the concession quotes that reason
    concede_window: int = 8
    # right after an activity is solved someone marks moving on ("bora pra 2?") with this P per turn, within the
    # new activity's first 3 student messages (once per step)
    p_step_move: float = 0.2
    # an off-task remark gets a reply (a short thread) from the next or second-next speaker with this P
    # (0.5.2: .45, next speaker only)
    p_offtask_reply: float = 0.55
    # a misspelled word is corrected by the same student a few seconds later ("*superficie") with this P
    p_typo_fix: float = 0.85
    # after one's own statement, when no tag question comes: a short hedge that is not a question ("acho", "se eu
    # entendi direito") with this P, at most after_hedge_share_cap of the session's content claims
    p_after_hedge: float = 0.8
    after_hedge_share_cap: float = 0.25

    @model_validator(mode="before")
    @classmethod
    def _presets(cls, data):
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for key, table in (("pace", PACE_PRESETS), ("conflict", CONFLICT_PRESETS)):
            name = data.get(key) or cls.model_fields[key].default
            if name not in table:
                raise ValueError(f"{key} must be one of {sorted(table)}, not {name!r}")
            for k, v in table[name].items():
                data.setdefault(k, v)
        return data

    def step_dwell_s(self, n_steps: int) -> float:
        """Minimum time a step stays open (sim seconds)."""
        if not self.min_step_dwell_frac or not self.target_minutes:
            return 0.0
        return self.min_step_dwell_frac * self.target_minutes * 60.0 / max(1, n_steps)
