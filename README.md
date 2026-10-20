# CollabSim

A **simulator of small-group student chats** for collaborative learning, as a Python package. A UI
(for example a Gradio app) imports it directly: it creates simulated groups of 2–10 students
working on a jigsaw task, and any number of **participants** (people typing in the UI, or bots the
UI relays) can join the chat and write in it.

**Code** decides every epistemic event: who speaks and when (pilot-calibrated turn-taking), what
each student knows or believes (jigsaw cards, KC mastery, misconceptions, affect), what a turn says
(a content plan), and whether a misconception flips. **Claude Haiku** (`claude-haiku-5-5`, the only
allowed model; no sampling parameters, thinking disabled) only words the turns. An
evidence-required leak guard stops students from saying what they do not know.

Every participant is treated the same way: their messages are interpreted into content plans and
reach the students only through what they say, exactly like a classmate's turn. The code never
branches on a participant's name or avatar.

Version 0.5.3. Languages: pt-BR (default, the calibration language) and English. No HTTP server.

## Install

```bash
pip install "git+https://github.com/adaj/collabsim@v0.5.3"
```

Python ≥ 3.11. Dependencies: `pydantic`, `pyyaml`, `anthropic`. Task and language data ship in the wheel.

## Quick start

```python
from collabsim import Simulator

sim = Simulator(api_key="sk-ant-...")          # or ANTHROPIC_API_KEY; backend="template" / "mock" offline
print(sim.check_connection())                  # {"ok": True, "model": "claude-haiku-5-5", ...}

s = sim.new_session(n_students=4, task="sci-enzyme-01", language="pt-BR",
                    participants=["Adelson", {"name": "Bot", "avatar": "bot.png"}])
print(s.roster)          # students s1..s4, then {"id": "e1", "name": "Adelson", "kind": "external", ...}, e2
s.start()                # realtime: a background thread paces the chat

out = s.poll(after=0)    # {"events": [SimEvent, ...], "next_after": 3, "status": "running", "typing": ["s2"]}
s.say("acho que mastigar é só pra comida descer mais fácil", sender="Adelson")
s.say("Lara, você poderia explicar melhor?", sender="Bot")   # writing a student's name addresses them
s.pause(); s.resume(); s.set_speed(5)
s.stop()
record = s.record()      # full research record (hidden state, plans, flips, interpretations)
print(s.usage())         # {"calls": ..., "input_tokens": ..., "output_tokens": ..., "usd": ..., "est_usd": ...}
s.close()
```

Offline, synchronous (tests, notebooks):

```python
sim = Simulator(backend="template")            # no LLM at all; "mock" = the LLM pipeline, deterministic, $0
s = sim.new_session(n_students=3, pacing="manual", seed=1, participants=["Ana"])
s.step()                 # -> [SimEvent] the next student turn, now
s.say("oi gente")        # the only participant: sender may be omitted
events = s.run_to_end()  # the rest of the chat
```

CLI (the key comes from `--env-file` or `ANTHROPIC_API_KEY` and is never printed):

```bash
python -m collabsim check --env-file .env                      # one 1-token Haiku call: key, network, billing
python -m collabsim try --env-file .env --n 4 --language pt-BR # one whole chat -> trysim/<stamp>.md / .json
python -m collabsim try --backend mock --n 4 --language pt-BR  # offline
python -m collabsim try --env-file .env --n 4 --pace brisk --conflict high
# a small real-Haiku batch, one session after another: prints a cost estimate and asks y/N (--yes skips);
# writes trysim/<stamp>/<n>-<seed>.md/.json and summary.md (per-session stats and totals); at most
# --max-sessions (12) sessions unless raised
python -m collabsim batch --env-file .env --seeds 1,2,3,4 --n 2,4,6 --language pt-BR
```

## API

### `Simulator`

```python
Simulator(api_key=None, backend="llm", cache_dir=None, student_model="claude-haiku-5-5",
          max_sessions=8, max_student_turns=200, *, idle_ttl_s=1800, lurk_cap_minutes=10,
          unspoken_boost_per_min=1.0, tick_s=0.1, request_timeout_s=30)
```

| method | returns |
|---|---|
| `options()` | tasks (id, title, languages, …), languages, `n_students` range (2–10; calibrated 2–4), pacing modes, speed range, trait presets, `participants` (`{"max": 8}`), caps |
| `list_tasks(language=None)` / `get_task(id, language)` | library task list / a `TaskSpec` |
| `validate_task(spec)` | `{"valid", "errors", "warnings", ...}` (schema and consistency checks; never raises) |
| `draft_task(description, language, n_roles=None)` | Haiku drafts a TaskSpec **for review**: `{"draft", "validation", "usage", "error"}` (~US$0.003; never raises) |
| `new_session(config=None, **fields)` | a `Session` (not started) |
| `check_connection()` | `{"ok", "model", "latency_s", "error", "error_kind", "hint", ...}` (never raises) |
| `sessions()` / `get_session(id)` / `reap()` / `close()` | registry; `reap()` stops sessions idle > `idle_ttl_s` |

`backend`: `"llm"` (Anthropic API; raises `BackendUnavailable` without a key), `"mock"` (the full LLM
pipeline against a deterministic offline stand-in), `"template"` (template wording, heuristic
interpretation of participants' messages). `cache_dir` enables a persistent response cache
(identical requests replay for free). `max_sessions` counts running/paused/ready sessions
(`CapacityError`).

### `SessionConfig` (the fields of `new_session`)

Unknown fields are rejected (`ValueError`).

| field | default | |
|---|---|---|
| `n_students` | 4 | 2–10 |
| `task` | `"sci-enzyme-01"` | library id (`"sci-enzyme-01"`, `"math-ratio-01"`), a `TaskSpec`, or a TaskSpec dict (one language, or `{"languages": {"pt-BR": {...}, "en": {...}}}`) |
| `language` | `"pt-BR"` | `"pt-BR"` or `"en"` |
| `seed` | random | same seed + config = same group (names, cards, misconceptions, traits) |
| `participants` | `[]` | external participants: names or `{"name", "avatar"}` (`ParticipantSpec`), up to 8, names unique (accent/case-folded). Ids `e1`, `e2`, … in join order. A student whose name clashes is renamed. More can join later |
| `pacing` | `"realtime"` | `"realtime"` (background thread after `start()`) or `"manual"` (`step()` / `run_to_end()`) |
| `speed` | 1.0 | sim seconds per wall second, (0, 20] |
| `traits` | `"mixed"` | `"mixed"` (pilot-calibrated priors), a preset (`chatty`, `quiet`, `focused`, `distracted`, `deferent`, `stubborn`) or `{"preset", "overrides": {trait: v}, "per_student": {"0": {trait: v}}}` |
| `offtask_rate` | 0.15 | target share of off-task student turns, 0–0.5 (off-task turns are spaced, never two in a row by one student, form short threads, and a topic is not reused in a session) |
| `time_of_day` | `None` | `"manha"` / `"tarde"` / `"noite"` (or morning / afternoon / evening); `None` = drawn from the seed. Greetings and chatter stay consistent with it |
| `pace` | `"pilot"` | `"pilot"`: as long as the pilot chats (median ~33 min in the calibration configuration, 12–19 messages per student; each activity stays open at least ~7.8 min); `"brisk"`: a ~15–20 min demo |
| `target_minutes` | `None` | overrides the pace's target length (pilot 36, brisk 14); an activity stays open at least .65 × target / n_steps |
| `conflict` | `"normal"` | `"low"`: a held misconception comes up only if its holder happens to claim on that activity (maybe none); `"normal"`: at least 1, typically 2, voiced by a student who holds it and challenged by a peer before the activity is solved; `"high"`: every held belief, stronger, pushed back on up to twice, and an activity stays open until the holder gives it up (the group may stall) |
| `max_minutes`, `end_on_complete`, `max_student_turns` | pace's (pilot 48, brisk 25), True, None | `max_minutes` is the sim-minutes time cap; near it the group wraps up |

### `Session`

| member | |
|---|---|
| `roster` | `[{id, name, avatar, ring, role, kind}]`: students first (`kind` `"student"`, avatar = i % 6, ring = i // 6 so repeated images get a different ring color), then participants (`kind` `"external"`, avatar as declared, `"spoken"`: has written) |
| `participants` | the participants' roster entries |
| `task`, `seed`, `status`, `error`, `ended_reason`, `pacing` | `status`: `ready` (realtime, not started) · `running` · `paused` · `done` · `error` |
| `start()` | realtime: start the clock and the thread that pre-generates the next turn and releases it when wall ≥ start + t_sim / speed. Idempotent |
| `poll(after=0)` | **non-blocking** `{"events": [SimEvent], "next_after", "status", "typing", "error"?}`; also keeps the session alive |
| `say(text, sender=None, *, ts=None)` | participant `sender` (a name or an id, folded match; `None` only if exactly one participant is declared; an unknown name joins the chat) posts `text` → `SimEvent` at once. One Haiku call interprets it (facts, claims, misconception voiced or challenged, addressee → a validated content plan; fallback: an opaque social turn) and the normal state rules apply (students may learn from it, challenge it, disagree). A sender matching a student raises `ValueError`. Returns `None` once the session has ended |
| `add_participant(name, avatar=None)` | join mid-session; returns the roster entry (`ValueError`: duplicate, a student's name, more than 8) |
| `pause()` `resume()` `set_speed(x)` `stop()` `close()` | `close()` stops, joins the thread and unregisters |
| `step()` / `run_to_end()` | manual pacing: next student turn now / the whole chat |
| `record()` | transcript with code content plans, hidden state (initial, timeline, final), flips, guard logs, interpretations, participants, events, config, usage |
| `usage()` | `{"calls", "local_cache_hits", "input_tokens", "cache_read_tokens", "cache_write_tokens", "output_tokens", "usd", "est_usd", "backend"}`: `usd` is what was billed (0 for `mock` and `template`, which are offline); `est_usd` prices the same tokens at Haiku rates (equal to `usd` on the `llm` backend; with `mock`, an estimate from its simulated token counts) |
| `events()` | every event so far |

**Addressing.** Writing a student's name ("Lara, …") addresses them: they reply within two student
turns about as often as in the pilot chats (~65%; ~half of the replies substantive), whoever wrote.
Students address or name a participant only after that participant has written; they never use role
words (tutor, bot, AI), never "@", never ask for anyone who is not in the chat.

A participant's message that arrives while a pre-generated turn is waiting rolls that turn back
(snapshot/restore), so the group reacts to the message; the discarded turn's calls are counted.

### `SimEvent`

`seq` (1, 2, … per session), `id` (`"<session>:<seq>"`), `t_sim` (sim seconds), `ts_ms` (wall epoch ms),
`speaker` (`s1`…`s10`, `e1`…`eN`, `system`), `display_name`, `kind` (`student` | `external` | `system`),
`text`, `addressee` (id of the chat member it names, if any), and for system events `code` (`ended` |
`error`) and `reason` (ended: `complete` | `time_cap` | `turn_cap` | `stopped` | `idle`; error: `billing`
| `auth` | `rate_limit` | `connection` | `refused` | `internal`). Frozen pydantic model; `to_dict()`.

### Errors

Nothing raised in the background thread reaches the caller. An API refusal (billing, credit or spend
limit, auth), a persistent 429/529 after backoff, a connection failure, or an internal error ends the
session with `status == "error"`, a readable message in `poll()["error"]` / `s.error` (pt-BR for pt-BR
sessions) and a final `system` event; every released event is kept. Connection errors carry their
cause chain and a hint, e.g.

> Não foi possível conectar à API da Anthropic. Provável causa: falha de certificado SSL (proxy
> corporativo/antivírus interceptando HTTPS? ajuste SSL_CERT_FILE ou REQUESTS_CA_BUNDLE …). Detalhe
> técnico: anthropic.APIConnectionError: Connection error. <- httpx.ConnectError: [SSL:
> CERTIFICATE_VERIFY_FAILED] … <- ssl.SSLCertVerificationError: …

Hints: `ssl`, `proxy`, `dns`, `timeout`, `refused`, `reset`. Run `sim.check_connection()` (or
`python -m collabsim check`) at app start-up to diagnose before a session fails. Requests time out
after 30 s; connection failures are retried twice (2 s, 5 s) after the SDK's own 2 retries. The
caller's thread only sees `ValueError` (bad config, task or sender), `CapacityError`, and
`BackendUnavailable` (no key, at construction).

## Gradio integration

- **One `Simulator` per app**, created at import time (`Simulator(api_key=os.environ["ANTHROPIC_API_KEY"])`);
  it is thread-safe and shares one HTTP client across sessions. Call `check_connection()` once at
  start-up and show its `error` if `ok` is false.
- **One session per browser session**, with the participants it will have:
  `sim.new_session(participants=[user_name, {"name": "Bot", "avatar": "bot.png"}])`. Keep the
  session id in `gr.State` and look it up with `sim.get_session(sid)` (or keep the `Session` object).
  Create it in the "start" callback and call `s.start()`.
- **`gr.Timer(0.5)` → `s.poll(after=last_seq)`**: append `out["events"]`, store `out["next_after"]`, show
  `out["typing"]` as "X está digitando…", and stop the timer when `out["status"]` is `done` or `error`
  (show `out["error"]`). `poll` never blocks on the LLM.
- **Submit → `s.say(text, sender=user_name)`**; **a bot's reply → `s.say(reply, sender="Bot")`** (run
  the bot on the visible events and relay what it writes; to address a student, the bot writes their
  name). Both return immediately in realtime; the interpretation runs in the session thread.
- **Styling**: `event.kind == "external"` for participants, `event.speaker` / `display_name` to tell
  them apart; `roster[i]["avatar"]` (0–5 for students; as declared for participants) and `["ring"]`.
- **Cleanup**: `s.close()` when the user leaves (e.g. `gr.State(..., delete_callback=...)` or
  `demo.unload` in recent Gradio). Sessions nobody polls for `idle_ttl_s` (30 min) are stopped anyway,
  so abandoned tabs stop spending.
- `s.record()` at the end for research logs (JSON-serializable with `default=str`).

## Costs

Priced with Haiku 5.5 rates ($0.10 in, $0.125 cache write, $0.01 cache read, $0.50 out per MTok). The
latest real-Haiku session (0.5.0, 4 students, 30 messages) used 52 calls for US$0.0068, about
US$0.00013 per call. A 0.5.1 default-pace session is about twice as long (n=4: ~50–60 messages,
~40% of them template lines that need no call, ~2.7 calls per Haiku turn): **≈ US$0.01–0.015 per
session of ~32 min** (≈ $0.02–0.03 per hour of chat). `python -m collabsim batch` prints an estimate
from a free dry run before it starts (the suggested 12-session batch: ≈ US$0.14, printed with a ×1.3
margin as ≈ US$0.18). Plus ≈ $0.0001 per participant message and about one
turn's cost per pre-generated turn discarded because a message arrived; a task draft ≈ $0.003. There
is no budget logic in the package: set a spend limit in the Anthropic Console workspace. 0.5.2 adds a
coherence check on challenge / belief lines (~5–8 calls per session, ≈ +US$0.0003–0.0006, +5–8%). 0.5.3
words the closing okays with the model (+1–3 calls per session).

## Realism notes

- Turn-taking, message lengths, gaps, same-speaker runs, lurking and filler shares are calibrated
  against aggregate statistics of a private pilot corpus of real pt-BR student chats (groups of
  2–4); the test suite checks the engine's structural statistics stay inside the pilot
  leave-2-groups-out bands (`tests/test_structure.py`).
- pt-BR is the native register: the realizer style guide is pt-BR and its few-shots are invented lines
  written in the students' register (lowercase, few accents, "kkkk", "sla", "tbm"); English output goes
  through an English-only filter.
- Students only voice what their state supports: the leak guard (code keyphrases + a Haiku check that
  must quote evidence) regenerates or falls back to a template when a line would reveal an unseen card
  fact; repetition, prompt-label, framing, belief and self-attribution guards are logged in `record()`.
- **Fillers are capped** (0.5.0): "don't know" turns ≤ 5% of a session and ≤ 2 per student,
  content-free turns ≤ 16%, never more than 2 content-free messages in a row, a filler wording at most
  twice, and never a filler right after one's own message (no "como eu envio?" → "nao faço ideia").
  A student with nothing to add stays quiet instead.
- **Interface confusions** are capped by group size (1 + n//2 interface questions, 2 + n//2
  confusions in all; each once per session) and usually someone answers.
- **Progress**: when the group stalls (5 turns or 150 s without progress) a student who holds a
  missing key-idea fact shares it, someone states a missing idea, asks about it, or challenges a
  blocking misconception; near the time / turn cap the group wraps up with what it established (a
  closing synthesis, `record()["closing_forced"]`) instead of fading out.
- **Framing**: one way of introducing the material ("aqui diz que…", "li que…") at most twice per
  session; the word "card" at most once.
- **Misconceptions**: a student voices a belief once; after a challenge they flip (learning rules),
  push back once with a reason in other words, or go quiet on it; a line never voices a belief the
  plan does not (the wording is overridden).
- **Truncation**: `max_tokens` 100 for chat lines; a line cut by the cap (or ending in a cut-off word)
  is regenerated once with a "shorter" note, else trimmed back to complete words.
- A contribution is sometimes typed as two messages ("pera" … "acho que …"), or followed by a short
  afterthought ("ne isso?"), as in the pilot chats.
- **Silences** (0.5.1): never more than 150 s between two messages while the chat runs (pilot gap p95
  131 s); at the cap someone in the chat speaks, with content if they have any, else the activity
  question or a short nudge.
- **Pace** (0.5.1): each activity stays open for a minimum time (default pace: ~7.8 min for 3
  activities), so a group does not finish a 3-activity task in 15 minutes; once the content is on the
  table they write up the answer ("vou colocar: eu acho que ..."), okay it and talk about the
  confidence bar. `pace="brisk"` for demos.
- **Conflict** (0.5.1): by default 1–2 held misconceptions are slated to surface: the student who
  holds it voices it once the activity is underway, a peer challenges it before the activity can be
  solved, and the holder flips, pushes back once, or eventually gives in ("ah entendi, faz sentido").
- **Information pacing** (0.5.1): one new fact or claim per student per 90 s / 3 turns unless asked;
  at most 2 content turns in a same-speaker run; "como eu disse" only when restating after a dispute;
  content lines in student register (≤ 25 words; the LLM gets a short gist of the material, never its
  sentence).
- **Closing** (0.5.1): with 3+ students at least two others answer the proposal (agree or add a
  detail), others with probability by engagement; then the chat ends.
- **Coherence** (0.5.2): a challenge always carries content that refutes the belief it answers (a
  refuting card fact, a refuting correct claim, the counterexample, or "pq vc acha isso?"); "discordo" /
  "acho que nao" only when the speaker opposes a belief someone else voiced; a belief line never states
  its own refutation (task files: `keyphrases`, `refuting_claims`, `because` per misconception). LLM-worded
  challenge / belief lines get one extra Haiku coherence check (JSON); an incoherent line is regenerated
  once, else the template line. Instruction-style text ("voce acredita nisso e fala como opiniao sua")
  never reaches the chat.
- **Tag questions, pings, repeats** (0.5.2): "vcs concordam?" / "ne isso?" after at most 15% of the
  content claims (each wording once); at most 3 name pings per session for n <= 4 (one per classmate,
  never at someone who has barely written), and the pinged student answers at the pilot reply rate; a
  student states each fact once (restated only when asked or disputed, in other words); a holder asked
  "pq?" gives a reason (`because`), or says they are unsure. A content line is sometimes typed as two
  messages, only at a clause boundary ("a amilase quebra o amido" / "por isso fica doce"; never "... em
  maltose que" / "é um açucar"), at most 3 per student.
- **Dialogue state** (0.5.3): a holder gives in only to a refuting move (a refuting fact / claim / the
  counterexample addressed to them, the group or that belief, within 8 student turns), and the concession
  quotes the reason ("ah, se ela acabasse o biscoito ia parar de ficar mais doce, faz sentido"); a holder who
  never gave in okays the final answer with a hedge; no question about what one just asserted (or what the
  chat already answered), no re-asking; credit only for what the credited classmate actually said (no
  circular credit); one burst never mixes a claim with its own belief; a belief is voiced for the activity
  under discussion or the next one.
- **Address and closings** (0.5.3): in dyads no "galera" / "gente" / "vcs" / "alguem" (rewritten to the
  partner, else regenerated); lead-word-only messages ("pera", "galera") about one per 15 messages, the lead
  never repeated in the next line; the final answer sums up what the group established (paraphrased, by
  someone who took part); the okays are worded by the model (composed per session with the template
  backend), never two alike in a session, and an added detail is never one the proposal already has;
  "bora pra 2?" after a solved activity now and then; off-task remarks get a reply more often, are never
  addressed to a classmate out of the blue, and vary across seeds; a misspelled word is sometimes
  corrected ("*superficie"); a short hedge after one's own statement ("acho", "se eu entendi direito").
- **More than 4 students is extrapolation** (structure is tested for n = 2–10; realism is not).
- Participants' messages are interpreted, never verified: a wrong interpretation changes what the
  simulated students learn (logged in `record()["interpretations"]`).
- Sessions live in memory; a process restart loses them.

## Calibration data

The structural targets are **aggregate realism targets from a private pilot corpus of small-group
student chats** (pt-BR, groups of 2–4). The pilot study was conducted under informed consent and
ethics approval. The corpus is **not distributed** and is **not used to train models**; only
aggregate statistics are released: `tests/support/data/pilot_targets.json` (pooled statistics,
leave-2-groups-out ranges, anonymous groups g1..g7), which the structural tests use. The few-shot
style examples are invented.

With a local copy of the corpus (a TSV with columns `Group`, `Username`, `Timestamp`, `Message`), the
tests that recompute the targets run (they skip otherwise), and the JSON can be regenerated:

```bash
export COLLABSIM_PILOT_TSV=/path/to/corpus.tsv   # keep it outside the repo (*.tsv is gitignored)
pytest tests/test_structure.py
python -m tests.support.pilot_stats              # rewrites tests/support/data/pilot_targets.json
```

## Layout

```
collabsim/
  __init__.py  simulator.py  types.py  errors.py  runio.py  __main__.py   public API, CLI
  runtime/     live.py (pacing, rollback, errors) interpret.py (participants' messages) taskspec.py presets.py
  sim/         engine, scheduler, planner, realizer, updater, learning (parameters), guards, groups, off-task
  llm/         client (Haiku-only allowlist, cache, usage meter), realizer, prompts, mock, config
  models/ lang/ tasks/ (en + pt-BR)
tests/         default suite (`pytest`); tests/support = structural statistics + pilot targets (test-only)
```

Development: `pip install -e ".[dev]" && pytest` (≈ 5 min, fully offline).

See `CHANGELOG.md`.

## License

MIT, see [`LICENSE`](LICENSE). Copyright (c) 2026 Adelson de Araujo.
