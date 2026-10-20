"""Coherence audit (0.5.2): scans finished transcripts for the qualitative defects of the 0.5.1
real-Haiku review. Works on plain transcript dicts (``Simulation.result()["transcript"]``: id, t,
speaker, kind, text, addressee, plan) plus a Task, so it can audit runs of any version (the
task's misconception keyphrases / refuting claims come from the current task files; the ids are
stable across versions).

Per session it counts:
  instruction_leak            student lines that read like a realizer instruction ("voce acredita
                              nisso", "fala como opiniao sua", "diga que", "in your own words", ...)
  disagree_no_opposition      lines with a disagreement marker ("discordo", "acho que nao", "nah",
                              "i disagree") whose plan does not oppose a claim the speaker rejects
  disagree_restating          disagreement lines whose content is already in the line they answer
  challenge_unrelated         challenges of a misconception carrying content that does not refute it
  self_contradiction          a belief line that states its refutation, or a refuting line that
                              asserts the belief
  tag_rate                    afterthought / tag questions per content claim (+ consecutive by one
                              speaker, max uses of one form)
  pings / pings_unanswered / pings_to_quiet / ping_target_max
                              peer name pings ("julia vc sabe?"), how many got no answer within 6
                              student turns, how many targeted someone who had spoken at most once
  fact_repeats                one speaker stating the same content (fact / correct claim / key idea)
                              again without being asked or disputed; near_dup_same_speaker: text-level
                              repeats (content tokens, numbers included)
  press_unanswered            "pq vc acha isso?" at a holder that the holder never answers

0.5.3 detectors (real-Haiku batch on 0.5.2 + an independent qualitative review):
  bad_split                   a thought typed as two messages cut mid-clause: the head ends on a dangling
                              word (relative pronoun / preposition / article / "é") or the tail starts with
                              one ("... em maltose que" | "é um açucar"); splits_max_per_student
  lead_only                   a message made only of a lead word ("pera", "galera", "olha", "gente olha
                              isso") followed by the same student's content line; lead_dup: that line opens
                              with the same lead word again ("olha" | "olha, pelo que eu li...")
  flip_unexplained            a holder who voiced a belief concedes / switches stance with no refuting move
                              (addressed to them or the group) in the previous 8 student lines;
                              concede_no_reason: a concession that does not refer to the reason;
                              flip_silent: a holder asserting the refutation without ever conceding;
                              closing_accept_unresolved: a holder who never conceded okays the final answer
                              without a hedge
  assert_then_ask             a question whose answer the asker had just asserted (same KC / key idea, or the
                              refuted belief they voiced); reasked: the same student asking an equivalent
                              question again; asked_answered: asking about a fact already stated in the chat
  plural_dyad                 plural address ("galera", "gente", "vcs", "alguem", "guys", "anyone") in a dyad
  closing_repeat              a closing response repeating an earlier one of the same session (summary also
                              reports the across-seed share of the most common closing phrase)
  false_attribution           "como malu disse ..." where Malu never said it (or was quoting the speaker)
  offtask_threads / offtask_replied, offtask_named (an off-task line out of the blue naming a classmate),
  step_moves ("bora pra 2?"), and across-seed repetition of off-task lines (summary)
"""
from __future__ import annotations

import re
import statistics
import unicodedata

WINDOW = 6   # student turns


def fold(t: str) -> str:
    t = unicodedata.normalize("NFKD", (t or "").lower())
    return "".join(c for c in t if not unicodedata.combining(c))


# instruction-style phrasing addressed to the writer (pt-BR + English), matched on folded text
INSTRUCTION_RE = re.compile(
    r"\bvoce (acredita nisso|discorda de|contesta o que|nao tem certeza|se convenceu|mesmo\(a\) disse|pode comecar)"
    r"|\bfala como opiniao|\bcomo opiniao sua|\bcom suas (proprias )?palavras|\bdiga (que|isso|o mesmo|de outro)"
    r"|\bdefenda (uma vez|com)|\bsem conteudo seu|\bquestionaram sua ideia|\bde este contraexemplo"
    r"|\bpasse esta ideia|\brepita com suas|\bpergunte (pra|isto|pro)|\bescreva (so|a mensagem|uma)"
    r"|\bthe idea you use|\byou believe this|\bsay it as your own|\byour own opinion|\bin your own (casual )?words"
    r"|\byou disagree with|\byou push back on|\bdefend it once|\bgive this counterexample|\bget this idea across"
    r"|\bsay that you|\bwrite only|\bask \w+ why they", re.I)

_NOT_DISAGREE_PT = (r"(?!sei\b|to\b|tou\b|tenho\b|tinha\b|entendi\b|lembro\b|faco\b|achei\b|consigo\b|sabia\b|ta\b|e\b|era\b|"
                    r"vi\b|estudei\b|aguento\b|muito\b|mt\b|da\b|deu\b|precisa\b|sai\b|vou\b|quero\b|gosto\b)")
DISAGREE_RE = {
    "pt-BR": re.compile(
        r"\bdiscord\w*|\bacho que (?:nao|n)\b(?:,|\s+" + _NOT_DISAGREE_PT + r"|$)|\bnada a ver\b|\bnao concordo\b"
        r"|^\W*(?:\w+,?\s+)?(?:hmm |pera |mas )?nao(?:[,!]?$|[,!]?\s+" + _NOT_DISAGREE_PT + ")", re.I),
    "en": re.compile(
        r"\bi disagree\b|\bi dont think so\b|\bdont agree\b|\bthats wrong\b"
        r"|^\W*(?:\w+,?\s+)?(?:wait |hmm |but )?(?:no|nah|nope)(?:[,!]?$|[,!]?\s+(?!idea\b|clue\b|way\b|one\b|problem\b|worries\b))", re.I),
}


def disagree_marker(text: str, language: str) -> bool:
    rx = DISAGREE_RE.get(language, DISAGREE_RE["en"])
    return bool(rx.search(fold(text)))

NEG = {"nao", "n", "nunca", "nem", "not", "isnt", "dont", "doesnt", "never", "wont", "cant", "arent", "no"}


def _toks(text: str) -> list[str]:
    return re.findall(r"[a-z0-9/:.,+]+", fold(text))


def phrase_hits(text: str, phrases, negation_aware: bool = True) -> list[str]:
    """Phrases found in `text` (folded); with negation_aware a hit right after a negation word (within
    3 tokens) does not count, unless the phrase itself starts with one."""
    low = " " + " ".join(re.findall(r"[a-z0-9/:.+]+", fold(text))) + " "
    out = []
    for p in phrases:
        fp = " ".join(re.findall(r"[a-z0-9/:.+]+", fold(p)))
        if not fp:
            continue
        for m in re.finditer(r"(?<![a-z0-9])" + re.escape(fp), low):
            if negation_aware and fp.split()[0] not in NEG:
                before = low[:m.start()].split()[-3:]
                if any(w in NEG for w in before):
                    continue
            out.append(p)
            break
    return out


def refuting_facts(task, mid: str) -> list[str]:
    return [f for f, ms in task.refutes.items() if mid in ms]


def refuting_claims(task, mid: str) -> list[int]:
    m = task.misconception(mid)
    explicit = getattr(m, "refuting_claims", None)
    if explicit:
        return list(explicit)
    return list(range(len(task.kc(m.kc).correct_claims)))


def refuting_phrases(task, mid: str) -> list[str]:
    m = task.misconception(mid)
    out = [k for f in refuting_facts(task, mid) for k in task.fact(f).keyphrases]
    claims = [task.kc(m.kc).correct_claims[i] for i in refuting_claims(task, mid)]
    out += [k for k in task.kc(m.kc).keyphrases if any(fold(k) in fold(c) for c in claims)]
    return list(dict.fromkeys(out))


def belief_phrases(task, mid: str) -> list[str]:
    return list(getattr(task.misconception(mid), "keyphrases", []) or [])


def _content_key(task, p: dict):
    """Key ideas (or a raw key) a content plan states."""
    if p.get("claim") == "fact" and p.get("fact_ids"):
        f = p["fact_ids"][0]
        ideas = {ki.id for ki in task.all_key_ideas() if f in ki.facts}
        return ideas or {("fact", f)}
    if p.get("claim") == "correct" and p.get("kc") and p.get("claim_idx") is not None and p.get("family") != "counterexample":
        ideas = {ki.id for ki in task.all_key_ideas() if ki.kc == p["kc"] and p["claim_idx"] in ki.claims}
        return ideas or {("claim", p["kc"], p["claim_idx"])}
    return None


def _content_tokens(text: str) -> set[str]:
    stop = set("""a o as os um uma e ou mas entao se que de do da no na em pra pro para por com eu ele ela
    vc vcs isso esse essa ai aqui la e eh ta to tipo ne ok acho que sim nao n ja so tbm mt mais
    the a an and or but so if that this it its is are was to of in on for with i you we they
    think like yeah ok just""".split())
    return {w for w in re.findall(r"[a-z0-9]+", fold(text)) if w not in stop and (len(w) > 2 or w.isdigit())}


def audit_session(task, transcript: list[dict], n_students: int | None = None) -> dict:
    msgs = transcript
    stu = [m for m in msgs if m.get("kind") == "student"]
    by_id = {m["id"]: m for m in msgs}
    names = {}
    out = {k: 0 for k in ("lines", "instruction_leak", "disagree_lines", "disagree_no_opposition",
                          "disagree_restating", "challenges", "challenge_unrelated", "self_contradiction",
                          "content_claims", "tags", "tag_consecutive_same_speaker", "tag_form_max", "pings",
                          "pings_unanswered", "pings_to_quiet", "ping_target_max", "fact_repeats",
                          "near_dup_same_speaker", "presses_at_holder", "press_unanswered")}
    out["lines"] = len(stu)
    out["examples"] = []
    tag_forms: dict[str, int] = {}
    last_claim_tagged: dict[str, bool] = {}
    said: dict[str, list] = {}
    said_text: dict[str, list] = {}
    ping_targets: dict[str, int] = {}
    pos = {m["id"]: i for i, m in enumerate(stu)}

    def ex(kind, m):
        if len(out["examples"]) < 40:
            out["examples"].append((kind, round(m["t"] / 60, 1), m["speaker"], m["text"]))

    def opposing(m, p) -> bool:
        if p.get("pushback") or p.get("claim") == "misconception":
            return True   # a holder defending the belief against a challenge
        mid, a = p.get("addresses_misconception"), p.get("anchor_turn")
        if not mid or a is None or a not in by_id:
            return False
        am = by_id[a]
        ap = am.get("plan") or {}
        return am["speaker"] != m["speaker"] and ap.get("claim") == "misconception"

    for i, m in enumerate(stu):
        p = m.get("plan") or {}
        text = m["text"]
        if INSTRUCTION_RE.search(fold(text)):
            out["instruction_leak"] += 1
            ex("instruction_leak", m)
        # ---------------- disagreement
        if p.get("act") not in ("off_task", "idk", "refusal", "backchannel", "greeting") and not p.get("split_tail") and \
                disagree_marker(text, task.language):
            out["disagree_lines"] += 1
            if not opposing(m, p):
                out["disagree_no_opposition"] += 1
                ex("disagree_no_opposition", m)
            a = p.get("anchor_turn")
            mid = p.get("addresses_misconception")
            if mid and a is not None and a in by_id and p.get("family") != "counterexample":
                mine = set(phrase_hits(text, refuting_phrases(task, mid)))
                theirs = set(phrase_hits(by_id[a]["text"], refuting_phrases(task, mid)))
                if mine and mine & theirs:
                    out["disagree_restating"] += 1
                    ex("disagree_restating", m)
        # ---------------- challenges
        mid = p.get("addresses_misconception")
        if mid and p.get("claim") in ("fact", "correct") and p.get("family") != "counterexample":
            out["challenges"] += 1
            bad = False
            if p.get("claim") == "fact" and p.get("fact_ids"):
                bad = p["fact_ids"][0] not in refuting_facts(task, mid)
            elif p.get("claim") == "correct" and p.get("claim_idx") is not None:
                bad = p["claim_idx"] not in refuting_claims(task, mid)
            if bad:
                out["challenge_unrelated"] += 1
                ex("challenge_unrelated", m)
            if phrase_hits(text, belief_phrases(task, mid)):
                out["self_contradiction"] += 1
                ex("self_contradiction(refuting line asserts belief)", m)
        if p.get("claim") == "misconception" and p.get("misconception_id"):
            if phrase_hits(text, refuting_phrases(task, p["misconception_id"]), negation_aware=False):
                out["self_contradiction"] += 1
                ex("self_contradiction(belief line refutes itself)", m)
        # ---------------- tags
        if p.get("tag"):
            out["tags"] += 1
            k = fold(text).strip(" ?!.")
            tag_forms[k] = tag_forms.get(k, 0) + 1
            if last_claim_tagged.get(("prev_tag", m["speaker"])):
                out["tag_consecutive_same_speaker"] += 1
            last_claim_tagged[("prev_tag", m["speaker"])] = True
        is_claim = (p.get("claim") in ("fact", "correct", "misconception") and not p.get("reference")
                    and p.get("act") in ("on_task_claim", "explain", "disagree") and not p.get("closing")
                    and not p.get("draft"))
        if is_claim:
            out["content_claims"] += 1
            # a new untagged claim by this speaker breaks their tag chain only if the next turn is not a tag
            nxt = stu[i + 1] if i + 1 < len(stu) else None
            if not (nxt is not None and nxt["speaker"] == m["speaker"] and (nxt.get("plan") or {}).get("tag")):
                last_claim_tagged[("prev_tag", m["speaker"])] = False
        # ---------------- pings
        to = p.get("addressee")
        if p.get("act") == "question" and p.get("follow_up") and to and to != m["speaker"] \
                and re.fullmatch(r"s\d+", to):
            out["pings"] += 1
            ping_targets[to] = ping_targets.get(to, 0) + 1
            spoke = sum(1 for x in stu[:i] if x["speaker"] == to)
            if spoke <= 1:
                out["pings_to_quiet"] += 1
            nxt = stu[i + 1:i + 1 + WINDOW]
            anchors = {m["id"], p.get("anchor_turn")} - {None}
            answered = any(x["speaker"] == to for x in nxt) or any(
                (x.get("plan") or {}).get("anchor_turn") in anchors and x["speaker"] != m["speaker"] for x in nxt)
            if not answered:
                out["pings_unanswered"] += 1
                ex("ping_unanswered", m)
        # ---------------- press at a holder
        if p.get("act") == "press" and p.get("addresses_misconception") and to and to != m["speaker"]:
            out["presses_at_holder"] += 1
            nxt = [x for x in stu[i + 1:i + 1 + WINDOW] if x["speaker"] == to]
            if not nxt:
                out["press_unanswered"] += 1
                ex("press_unanswered", m)
        # ---------------- same-speaker repeats
        key = _content_key(task, p) if p.get("act") in ("on_task_claim", "explain", "disagree") and not p.get("closing") \
            and not p.get("draft") else None
        if key:
            direct = p.get("restate") or (p.get("anchor_turn") is not None and p["anchor_turn"] in by_id and (
                (by_id[p["anchor_turn"]].get("plan") or {}).get("act") in ("question", "press", "disagree")
                and by_id[p["anchor_turn"]]["speaker"] != m["speaker"]))
            prev = said.setdefault(m["speaker"], [])
            if any(key & k for k in prev) and not direct:
                out["fact_repeats"] += 1
                ex("fact_repeat", m)
            prev.append(key)
        if p.get("claim") in ("fact", "correct") and not p.get("closing") and not p.get("draft"):
            ct = _content_tokens(text)
            prevt = said_text.setdefault(m["speaker"], [])
            a = p.get("anchor_turn")
            asked = p.get("restate") or p.get("answers_ping") or (a is not None and a in by_id and (
                (by_id[a].get("plan") or {}).get("act") in ("question", "press", "disagree")) and by_id[a]["speaker"] != m["speaker"])
            if len(ct) >= 3:
                for q in prevt:
                    # a restatement asked for / after a dispute may repeat the point, in other words (Jaccard)
                    sim_ = len(ct & q) / len(ct | q) if asked else len(ct & q) / min(len(ct), len(q))
                    if sim_ >= (0.8 if asked else 0.75):
                        out["near_dup_same_speaker"] += 1
                        ex("near_dup_same_speaker", m)
                        break
                prevt.append(ct)
    out["tag_form_max"] = max(tag_forms.values(), default=0)
    out["ping_target_max"] = max(ping_targets.values(), default=0)
    out["tag_rate"] = out["tags"] / out["content_claims"] if out["content_claims"] else 0.0
    out.update(audit_053(task, transcript, n_students, ex))
    return out


# ====================================================================== 0.5.3 detectors
DANGLING = {
    "pt-BR": set("""que de do da dos das em no na nos nas num numa a o as os um uma uns umas pra pro pras pros para
        com por pelo pela pelos pelas e ou se mais muito mt é eh ta tá esta está sao são tem ter cuja cujo onde
        quando como qual quais quem ao aos à às sem entre sobre ate até""".split()),
    "en": set("""the a an of to that which who whom whose with for in on at by from and or is are was were be
        been more very than as into about its their his her my your our""".split()),
}
BAD_TAIL_START = {"pt-BR": {"é", "eh", "que", "de", "do", "da", "e", "ou", "ta", "tá", "sao", "são", "cuja", "cujo"},
                  "en": {"is", "are", "was", "that", "which", "who", "of", "to", "and", "or", "than"}}
LEAD_LINES = {
    "pt-BR": ["pera", "galera", "olha", "mano", "tipo", "entao", "gente", "ei", "pera ai", "pera ai gente",
              "gente olha isso", "pera achei aqui", "olha oq diz aqui", "olha so", "olha isso", "tipo assim",
              "entao olha", "gente olha", "galera olha", "mano olha", "ok entao", "entao tipo", "gnte", "gnte olha isso"],
    "en": ["wait", "guys", "so", "ok so", "look", "hold on", "um", "like", "wait i found it", "ok look at this",
           "guys look", "hold on guys", "ok so like", "look at this", "so like"],
}
LEAD_WORDS = {"pt-BR": {"pera", "galera", "olha", "mano", "tipo", "entao", "gente", "ei", "gnte"},
              "en": {"wait", "guys", "so", "look", "hold", "um", "like"}}
PLURAL_RE = {
    "pt-BR": re.compile(r"\b(?:galera|pessoal|vcs|voces|alguem|ninguem|quem mais|gnte)\b"
                        r"|(?<!\ba )(?<!\bpra )(?<!\bda )(?<!\bde )\bgente\b"),
    "en": re.compile(r"\b(?:guys|everyone|everybody|anyone|anybody|nobody|no one|you all|y ?all|you two)\b"),
}
CONCEDE_RE = {
    "pt-BR": re.compile(r"\b(?:ah entendi|entendi agora|faz sentido|vc (?:ta|tem) (?:cert|razao)|voce (?:ta|tem) (?:cert|razao)"
                        r"|nao tinha pensado|tem razao|ta certo entao|me convenceu)\b"),
    "en": re.compile(r"\b(?:makes sense|youre right|you re right|you are right|didnt think of|i get it now|fair enough"
                     r"|ok fair|you convinced me)\b"),
}
HEDGE_RE = {"pt-BR": re.compile(r"\b(?:mas|ainda|sei la|duvida|sla|nao sei|meio)\b"),
            "en": re.compile(r"\b(?:but|still|not sure|idk|kinda|i guess)\b")}
CREDIT_RE = {
    "pt-BR": [re.compile(r"\b(?:como|igual|que|o que|isso que|oq)\s+(?:a |o )?([a-z]+)\s+(?:disse|falou|tinha falado|tinha dito|falou antes)\b"),
              re.compile(r"\b([a-z]+)\s+(?:ja|ja tinha)\s+(?:disse|falou)\b"),
              re.compile(r"\b([a-z]+)\s+(?:ta|esta|tava)\s+cert[oa]\b"),
              re.compile(r"\bconcordo com (?:a |o )?([a-z]+)\b")],
    "en": [re.compile(r"\blike ([a-z]+) said\b"), re.compile(r"\b([a-z]+)s point\b"),
           re.compile(r"\b([a-z]+) (?:already )?said (?:it|that)\b"), re.compile(r"\bsame as ([a-z]+)\b"),
           re.compile(r"\b([a-z]+) is right\b"), re.compile(r"\bagree with ([a-z]+)\b"),
           re.compile(r"\bwhat ([a-z]+) said\b")],
}
CREDIT_STOP = ("acho que nessa nisso isso concordo certo certa tbm tambem mesmo igual verdade e ja falou disse "
               "think right agree same said already point yeah true thats")
STOP_FLIP = set("""ah hmm ok ta faz sentido entendi entao nao tinha pensado nisso verdade vc voce certo razao isso mesmo
    agora acho que mas pse oh true makes sense didnt think of that youre right fair get it now so yeah""".split())


def _lg(task) -> str:
    return "pt-BR" if getattr(task, "language", "en") == "pt-BR" else "en"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", fold(text or ""))


def _words_raw(text: str) -> list[str]:
    return re.findall(r"[a-zà-ÿ0-9]+", (text or "").lower())


def _cw(task, text: str) -> set[str]:
    from collabsim.sim.content import content_words
    return set(content_words(text or "", task.language))


def _refuting_move(task, plan: dict, mid: str) -> bool:
    """Does this plan carry content that refutes misconception `mid` (not a press)?"""
    if not plan:
        return False
    if plan.get("act") == "press" or plan.get("family") == "press_for_reasoning":
        return False
    if plan.get("closing") or plan.get("draft") or plan.get("concedes"):
        return False
    if plan.get("addresses_misconception") == mid and plan.get("claim") in ("fact", "correct"):
        return True
    if plan.get("claim") == "fact" and set(plan.get("fact_ids") or []) & set(refuting_facts(task, mid)):
        return plan.get("is_correct", True) is not False
    m = task.misconception(mid)
    if plan.get("claim") == "correct" and plan.get("kc") == m.kc and plan.get("claim_idx") in refuting_claims(task, mid):
        return True
    return False


def audit_053(task, transcript: list[dict], n_students: int | None = None, ex=None) -> dict:
    lg = _lg(task)
    stu = [m for m in transcript if m.get("kind") == "student"]
    by_id = {m["id"]: m for m in transcript}
    pos = {m["id"]: i for i, m in enumerate(stu)}
    names = {}
    for m in transcript:
        if m.get("display_name"):
            names[fold(m["display_name"]).split()[0]] = m["speaker"]
    speakers = {m["speaker"] for m in stu}
    n = n_students if n_students is not None else len(speakers)
    o = {k: 0 for k in ("splits", "bad_split", "splits_max_per_student", "lead_only", "lead_dup", "concessions",
                        "flip_unexplained", "concede_no_reason", "flip_silent", "closing_responses",
                        "closing_accept_unresolved", "questions", "assert_then_ask", "reasked", "asked_answered",
                        "plural_dyad", "closing_repeat", "credits", "false_attribution", "offtask_threads",
                        "offtask_replied", "offtask_named", "step_moves", "misc_offstep")}
    o["_closing_texts"], o["_offtask_texts"] = [], []
    ex = ex or (lambda kind, m: None)
    P = lambda m: m.get("plan") or {}  # noqa: E731
    lead_set = {" ".join(_tokens(x)) for x in LEAD_LINES[lg]}
    from collabsim.sim.realism import strip_laugh

    def core(text):
        return " ".join(_tokens(strip_laugh(text or "")))

    # ---------------------------------------------------------------- splits
    per = {}
    for i, m in enumerate(stu):
        p = P(m)
        if not p.get("split_tail") or p.get("typo_fix"):
            continue
        head = next((x for x in reversed(stu[:i]) if x["speaker"] == m["speaker"]), None)
        o["splits"] += 1
        per[m["speaker"]] = per.get(m["speaker"], 0) + 1
        hw, tw = _words_raw(head["text"] if head else ""), _words_raw(m["text"])
        if (hw and hw[-1] in DANGLING[lg]) or (tw and tw[0] in BAD_TAIL_START[lg]):
            o["bad_split"] += 1
            ex("bad_split", m)
    o["splits_max_per_student"] = max(per.values(), default=0)
    # ---------------------------------------------------------------- lead-only messages
    for i, m in enumerate(stu[:-1]):
        c = core(m["text"])
        if not c or c not in lead_set:
            continue
        nx = stu[i + 1]
        if nx["speaker"] != m["speaker"] or nx["t"] - m["t"] > 45:
            continue
        o["lead_only"] += 1
        nw = _tokens(nx["text"])
        if nw and nw[0] in LEAD_WORDS[lg] and nw[0] in set(c.split()):
            o["lead_dup"] += 1
            ex("lead_dup", nx)
    # ---------------------------------------------------------------- stance changes
    voiced = {}            # (speaker, mid) -> list of student positions of voicings
    conceded = set()
    for i, m in enumerate(stu):
        p = P(m)
        sp = m["speaker"]
        if p.get("claim") == "misconception" and p.get("misconception_id"):
            voiced.setdefault((sp, p["misconception_id"]), []).append(i)
        mine = [mid for (s_, mid) in voiced if s_ == sp and (s_, mid) not in conceded]
        if not mine:
            continue
        text = fold(m["text"])
        is_closing = p.get("closing") in ("agree", "adjust")
        conc_mid = p.get("concedes")
        anc = by_id.get(p.get("anchor_turn")) if p.get("anchor_turn") is not None else None
        stance = conc_mid is not None or (not is_closing and p.get("act") in ("agree", "backchannel")
                                          and CONCEDE_RE[lg].search(text) is not None and anc is not None
                                          and any(_refuting_move(task, P(anc), mid) for mid in mine))
        if stance and not is_closing:
            mids = [conc_mid] if conc_mid in mine else mine
            o["concessions"] += 1
            lo = max(0, i - 8)
            ok_mid, ref = None, None
            for mid in mids:
                last_v = voiced[(sp, mid)][0]
                for x in reversed(stu[max(lo, last_v + 1):i]):
                    if x["speaker"] == sp:
                        continue
                    xp = P(x)
                    if _refuting_move(task, xp, mid) and (xp.get("addressee") in (None, sp)
                                                          or x["id"] in (p.get("reason_turn"), p.get("anchor_turn"))):
                        ok_mid, ref = mid, x
                        break
                if ok_mid:
                    break
            if ok_mid is None:
                o["flip_unexplained"] += 1
                ex("flip_unexplained", m)
            else:
                mine_w = _cw(task, m["text"]) - _cw(task, " ".join(STOP_FLIP))
                if not (mine_w & _cw(task, ref["text"])):
                    o["concede_no_reason"] += 1
                    ex("concede_no_reason", m)
            for mid in mids:
                conceded.add((sp, mid))
            continue
        if is_closing:
            continue
        for mid in mine:
            if _refuting_move(task, p, mid):
                o["flip_silent"] += 1
                ex("flip_silent", m)
                conceded.add((sp, mid))
                break
    # closing responses
    closing_seen = []
    for m in stu:
        p = P(m)
        if p.get("closing") not in ("agree", "adjust"):
            continue
        o["closing_responses"] += 1
        sp = m["speaker"]
        if p.get("concedes"):
            conceded.add((sp, p["concedes"]))      # gives in at the end, quoting the reason
        open_ = [mid for (s_, mid) in voiced if s_ == sp and (s_, mid) not in conceded]
        if open_ and not HEDGE_RE[lg].search(fold(m["text"])):
            o["closing_accept_unresolved"] += 1
            ex("closing_accept_unresolved", m)
        if p.get("brief"):
            continue
        c = core(m["text"])
        if c in closing_seen:
            o["closing_repeat"] += 1
            ex("closing_repeat", m)
        closing_seen.append(c)
        o["_closing_texts"].append(c)
    # ---------------------------------------------------------------- questions
    own_q: dict[str, list] = {}
    stated = set()
    for i, m in enumerate(stu):
        p = P(m)
        sp = m["speaker"]
        is_q = p.get("act") == "question" and not (p.get("follow_up") or p.get("tag") or p.get("nudge")
                                                    or p.get("step_q") or p.get("step_move"))
        if is_q:
            o["questions"] += 1
            f = p.get("asked_fact")
            qw = _cw(task, m["text"])
            if f:
                fk = task.fact(f).kc
                ideas = {ki.id for ki in task.all_key_ideas() if f in ki.facts}
                if f in stated:
                    o["asked_answered"] += 1
                    ex("asked_answered", m)
                for x in stu[:i]:
                    xp = P(x)
                    if x["speaker"] != sp or xp.get("closing") or xp.get("draft"):
                        continue
                    hit = (xp.get("claim") == "correct" and xp.get("kc") == fk and xp.get("is_correct", True)) or (
                        xp.get("claim") == "fact" and (f in (xp.get("fact_ids") or []) or ideas & {
                            ki.id for ki in task.all_key_ideas() for g in (xp.get("fact_ids") or []) if g in ki.facts})) or (
                        xp.get("claim") == "misconception" and xp.get("misconception_id")
                        and f in refuting_facts(task, xp["misconception_id"]))
                    if hit:
                        o["assert_then_ask"] += 1
                        ex("assert_then_ask", m)
                        break
            elif len(qw) >= 2 and not p.get("intent"):
                for x in stu[max(0, i - 10):i]:
                    if x["speaker"] == sp and P(x).get("claim") in ("fact", "correct", "misconception"):
                        xw = _cw(task, x["text"])
                        if xw and len(qw & xw) / len(qw) >= 0.5:
                            o["assert_then_ask"] += 1
                            ex("assert_then_ask", m)
                            break
            if len(qw) >= 2:
                for q in own_q.get(sp, []):
                    if len(qw & q) / min(len(qw), len(q)) >= 0.5:
                        o["reasked"] += 1
                        ex("reasked", m)
                        break
                own_q.setdefault(sp, []).append(qw)
        if p.get("claim") == "fact" and p.get("is_correct", True) is not False:
            stated |= set(p.get("fact_ids") or [])
        stated |= set(task.facts_mentioned(m["text"])) if p.get("act") != "question" else set()
    # ---------------------------------------------------------------- plural address in a dyad
    if n == 2:
        for m in stu:
            if PLURAL_RE[lg].search(fold(m["text"])):
                o["plural_dyad"] += 1
                ex("plural_dyad", m)
    # ---------------------------------------------------------------- attribution
    for i, m in enumerate(stu):
        text = fold(m["text"])
        sp = m["speaker"]
        for rx in CREDIT_RE[lg]:
            mm = rx.search(text)
            if not mm:
                continue
            who = names.get(mm.group(1))
            if who is None or who == sp:
                continue
            o["credits"] += 1
            rest = rx.sub(" ", text)
            rw = _cw(task, rest) - {fold(k) for k in names} - _cw(task, CREDIT_STOP)
            rf = set(task.facts_mentioned(rest)) | set(P(m).get("fact_ids") or [])
            evid = []
            for x in stu[:i]:
                if x["speaker"] != who:
                    continue
                xp = P(x)
                xw = _cw(task, x["text"])
                xf = set(task.facts_mentioned(x["text"])) | set(xp.get("fact_ids") or [])
                if xp.get("act") == "question" and not xp.get("draft"):
                    continue          # asking about it is not saying it
                mp = P(m)
                if mp.get("claim") == "correct" and xp.get("claim") == "correct" and xp.get("kc") == mp.get("kc") \
                        and xp.get("claim_idx") == mp.get("claim_idx") and mp.get("claim_idx") is not None:
                    evid.append(x)
                    continue
                if mp.get("claim") == "misconception" and mp.get("misconception_id") \
                        and xp.get("misconception_id") == mp.get("misconception_id"):
                    evid.append(x)          # siding with a classmate who voiced the same belief
                    continue
                if rw or rf:
                    if (rf & xf) or (rw and len(rw & xw) / len(rw) >= 0.4):
                        evid.append(x)
                elif (xp.get("claim") in ("fact", "correct", "misconception") or xp.get("draft") == "propose"
                      or xp.get("closing") == "propose") and pos[x["id"]] >= i - 12:
                    evid.append(x)
            spname = fold(next((x.get("display_name", "") for x in stu if x["speaker"] == sp), "")).split()[:1]

            def circular(x):
                xp = P(x)
                return xp.get("ref_by") == sp or bool(spname and re.search(rf"\b{re.escape(spname[0])}\b", fold(x["text"])))
            if not evid or all(circular(x) for x in evid):
                o["false_attribution"] += 1
                ex("false_attribution", m)
            break
    # ---------------------------------------------------------------- off-task threads, transitions
    for i, m in enumerate(stu):
        p = P(m)
        if p.get("step_move"):
            o["step_moves"] += 1
        if p.get("act") != "off_task" or p.get("anchor_turn") is not None or len(_tokens(m["text"])) < 2:
            continue
        o["offtask_threads"] += 1
        o["_offtask_texts"].append(core(m["text"]))
        if any(P(x).get("act") == "off_task" and x["speaker"] != m["speaker"] and P(x).get("anchor_turn") == m["id"]
               for x in stu[i + 1:i + 5]):
            o["offtask_replied"] += 1
        t = fold(m["text"])
        if any(re.search(rf"\b{re.escape(k)}\b", t) for k, v in names.items() if v != m["speaker"] and len(k) > 2):
            o["offtask_named"] += 1
            ex("offtask_named", m)
    return o


RATE_KEYS = [("instruction_leak", "lines"), ("disagree_no_opposition", "disagree_lines"),
             ("disagree_restating", "disagree_lines"), ("challenge_unrelated", "challenges"),
             ("self_contradiction", "lines"), ("tags", "content_claims"), ("pings_unanswered", "pings"),
             ("pings_to_quiet", "pings"), ("press_unanswered", "presses_at_holder"), ("fact_repeats", "lines"),
             ("near_dup_same_speaker", "lines"),
             # 0.5.3
             ("bad_split", "splits"), ("splits", "lines"), ("lead_only", "lines"), ("lead_dup", "lead_only"),
             ("flip_unexplained", "concessions"), ("concede_no_reason", "concessions"), ("flip_silent", "lines"),
             ("closing_accept_unresolved", "closing_responses"), ("assert_then_ask", "questions"),
             ("reasked", "questions"), ("asked_answered", "questions"), ("plural_dyad", "lines"),
             ("closing_repeat", "closing_responses"), ("false_attribution", "credits"),
             ("offtask_replied", "offtask_threads"), ("offtask_named", "offtask_threads"), ("step_moves", "lines")]


def summarize(rows: list[dict]) -> dict:
    """Pooled rates plus per-session means / maxima."""
    tot = {k: sum(r.get(k, 0) for r in rows) for k in rows[0] if isinstance(rows[0][k], int)}
    out = {"sessions": len(rows)}
    for num, den in RATE_KEYS:
        out[f"{num}/{den}"] = round(tot[num] / tot[den], 4) if tot[den] else 0.0
    for k in ("instruction_leak", "disagree_no_opposition", "challenge_unrelated", "self_contradiction", "tags",
              "tag_consecutive_same_speaker", "tag_form_max", "pings", "pings_unanswered", "ping_target_max",
              "pings_to_quiet", "fact_repeats", "near_dup_same_speaker", "press_unanswered"):
        vals = [r[k] for r in rows]
        out[f"{k}_mean"] = round(statistics.mean(vals), 3)
        out[f"{k}_max"] = max(vals)
    out["tag_rate_session_max"] = round(max(r["tag_rate"] for r in rows), 3)
    for k in ("splits_max_per_student", "bad_split", "lead_only", "flip_unexplained", "assert_then_ask", "reasked",
              "plural_dyad", "closing_repeat", "false_attribution"):
        vals = [r.get(k, 0) for r in rows]
        out[f"{k}_mean"] = round(statistics.mean(vals), 3)
        out[f"{k}_max"] = max(vals)
    out.update(across_sessions(rows))
    return out


def across_sessions(rows: list[dict]) -> dict:
    """Across-seed repetition: the share of sessions using the most common closing response, the share of
    closing responses / off-task lines whose exact wording occurs in more than 10% of the sessions."""
    from collections import Counter
    n = len(rows)
    out = {}
    for key, name in (("_closing_texts", "closing"), ("_offtask_texts", "offtask")):
        per = Counter(t for r in rows for t in set(r.get(key, [])))
        lines = [t for r in rows for t in r.get(key, [])]
        out[f"{name}_top_session_share"] = round(max(per.values(), default=0) / n, 3) if n else 0.0
        common = {t for t, c in per.items() if c > max(1, 0.10 * n)}
        out[f"{name}_common_line_share"] = round(sum(t in common for t in lines) / len(lines), 3) if lines else 0.0
        out[f"{name}_distinct_per_line"] = round(len(set(lines)) / len(lines), 3) if lines else 0.0
    return out
