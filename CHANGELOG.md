# Changelog

## 0.5.3 (2026-10-20): first public release

CollabSim simulates small-group student chats (2–10 students) working on a jigsaw task, as a Python
package a UI imports directly. Code decides every epistemic event (turn-taking, what each student knows or
believes, what each turn says, when a misconception changes); Claude Haiku (`claude-haiku-5-5`) only words
the turns.

### Features
- **State-based simulation**: jigsaw cards, KC mastery, misconceptions and affect per student; content
  plans per turn; learning rules for when a held belief is given up (a refuting fact, claim or
  counterexample, with the reason quoted in the concession).
- **Pilot-calibrated structure**: turn-taking, message lengths, gaps, same-speaker runs, lurking and
  filler shares are calibrated against aggregate statistics of a private pilot corpus of pt-BR student
  chats (groups of 2–4); only aggregates ship (`tests/support/data/pilot_targets.json`).
- **Guards**: an evidence-required leak guard (students never say what they do not know), plus
  repetition, prompt-label, framing, belief, attribution and coherence guards, all logged in `record()`.
- **Realism controls**: off-task threads, typos and corrections, split messages at clause boundaries,
  questions and answers, name addressing, dyad-aware address, silences capped at 150 s, minimum time per
  activity, a closing write-up the group okays.
- **External participants**: people or bots relayed by a UI join the chat; their messages are
  interpreted into content plans and reach the students only through what they say.
- **Backends**: `llm` (Anthropic API, Haiku only), `mock` (the full LLM pipeline against a deterministic
  offline stand-in; free), `template` (no LLM at all). Optional persistent response cache.
- **Languages**: pt-BR (default, the calibration language) and English. Two library tasks
  (`sci-enzyme-01`, `math-ratio-01`); custom tasks as TaskSpec dicts, with validation and Haiku drafting.
- **CLI**: `python -m collabsim check | try | batch` (a batch prints a cost estimate from a free dry run
  before it starts).

### API
- `Simulator(api_key=None, backend="llm", cache_dir=None, ...)`: `options()`, `list_tasks()`,
  `get_task()`, `validate_task()`, `draft_task()`, `new_session(**SessionConfig)`, `check_connection()`,
  `sessions()` / `get_session()` / `reap()` / `close()`.
- `SessionConfig`: `n_students`, `task`, `language`, `seed`, `participants`, `pacing` (`realtime` /
  `manual`), `speed`, `traits`, `offtask_rate`, `time_of_day`, `pace` (`pilot` / `brisk`),
  `target_minutes`, `conflict` (`low` / `normal` / `high`), `max_minutes`, `end_on_complete`,
  `max_student_turns`. Unknown fields raise `ValueError`.
- `Session`: `roster`, `participants`, `start()`, non-blocking `poll(after)`, `say(text, sender)`,
  `add_participant()`, `pause()` / `resume()` / `set_speed()` / `stop()` / `close()`, `step()` /
  `run_to_end()`, `record()` (full research record), `usage()` (calls, tokens, `usd` billed, `est_usd`
  priced at Haiku rates), `events()`.
- `SimEvent` (frozen pydantic model), `TaskSpec`, `ParticipantSpec`; errors `CollabSimError`,
  `BackendUnavailable`, `CapacityError`. Background failures (billing, auth, rate limit, connection)
  end the session with `status == "error"` and a readable message instead of raising.

Earlier versions were internal pre-releases.
