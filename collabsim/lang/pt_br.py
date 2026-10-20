"""Brazilian Portuguese pack (pt-BR): the native language of the pilot study the structural
targets were calibrated on (a private corpus, not distributed). Register follows students' chat style: lowercase, few accents,
abbreviations (vc, pq, oq, tbm, n, sla, vdd, mds, blz), laughter "kkkk". Few-shots are INVENTED
lines written in that register (lang/fewshots_pt.yaml). Every bank here has the same keys / ids as its English original, so
the code-side planner (which decides content by id) is language-neutral.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import yaml

code = "pt-BR"
label = "português do Brasil"

NAMES = ["Ana", "Pedro", "Júlia", "Lucas", "Bia", "Gabriel", "Lara", "Davi", "Malu", "Rafa",
         "Duda", "Caio", "Yasmin", "Enzo", "Luana", "Thiago", "Manu", "Igor"]

# ------------------------------------------------------------------ interface / procedural intents
# same keys as sim/intents.py (the planner picks the key; the pack only words it)
INTERFACE = {
    "save_mechanism": ("perguntar se a resposta ou a barrinha salva sozinha",
                       ["sera que salvou mesmo?", "salvou?", "salva sozinho?", "pera minha resposta salvou?",
                        "se eu mudar depois de enviar ainda salva?"]),
    "slider_meaning": ("perguntar pra que serve a barrinha de confiança",
                       ["a barrinha conta na nota?", "essa barrinha é pra que?", "pra que serve essa barra?", "a barrinha é o quanto a gente tem certeza?"]),
    "where_to_type": ("perguntar onde é pra digitar a resposta",
                      ["é pra escrever onde?", "onde a gente escreve a resposta?", "nao to achando a caixa de texto", "onde eu digito a hipotese?"]),
    "send_how": ("perguntar como envia a resposta",
                 ["onde aperta pra enviar?", "como eu envio?", "é pra apertar no enviar?", "como manda a resposta"]),
    "next_button": ("perguntar como passa pra próxima atividade",
                    ["cade o botao de proxima?", "tem botao de proximo?", "como passa pra proxima pagina?"]),
    "scroll": ("dizer que a página não rola ou perguntar como rola",
               ["a pagina travou aqui", "pera como desce a pagina", "minha tela nao desce kkk"]),
    "find_activity": ("dizer que não acha a atividade na tela",
                      ["sumiu a atividade da minha tela", "cade a atividade", "onde ta a aba da atividade", "ta em qual aba?"]),
    "answer_each": ("perguntar se cada um responde ou se é uma resposta do grupo",
                    ["é uma resposta so pra todo mundo?", "cada um responde ou é uma pro grupo?", "cada um escreve a sua?"]),
}
PROCEDURAL = {
    "what_to_do": ("perguntar o que o grupo tem que fazer agora",
                   ["oq era pra fazer msm?", "e agora?", "a gente faz oq agora"]),
    "what_to_write": ("perguntar o que é pra escrever na resposta",
                      ["é pra escrever o que?", "oq a gente coloca aqui?"]),
    "which_activity": ("perguntar em qual atividade / etapa o grupo está",
                       ["a gente ta em qual atividade?", "ja passou pra proxima?"]),
    "whats_the_answer": ("perguntar pro grupo qual é a resposta",
                         ["entao qual é a resposta?", "e oq vai ai?"]),
    "lost": ("dizer que está perdido / confuso",
             ["ue?", "to perdido kkk", "pera nao entendi"]),
}

# ------------------------------------------------------------------ template banks (sim/realizer.py)
FILLER = {
    "greeting": ["oi", "oie", "opa", "salve", "iae", "eai gente", "oi gente", "boa noite"],
    "backchannel": ["ok", "kkkk", "sim", "kkkkkk", "vdd", "hmm", "blz", "aham", "ah", "show", "kkk"],
    "agree": ["concordo", "é isso mesmo", "isso", "vdd", "faz sentido", "tbm acho", "sim",
              "concordo com isso", "acho que é bem isso"],
    "idk": ["sei la", "nao sei", "sla", "n sei", "nao faço ideia", "hmm nao lembro", "nem ideia", "nao sei bem"],
    "off_task": ["to com fome kkk", "isso ta demorando mt", "que horas acaba a aula", "kkkkkkk",
                 "vcs viram o jogo ontem", "meu lanche ta me chamando", "ja volto",
                 "minha internet ta mt lenta", "kkkkk para", "quem mais ta com sono", "show",
                 "bora pra casa", "esqueci meu carregador", "mds", "que musica é essa kkk"],
    "interface": [l for _k, (_d, ls) in INTERFACE.items() for l in ls],
    "question_generic": [l for _k, (_d, ls) in PROCEDURAL.items() for l in ls],
    "follow_up": ["alguem?", "alguem sabe?", "gente?", "entao ninguem sabe", "oi?? alguem",
                  "ninguem sabe?", "e ai?"],
    "follow_up_to": ["{name} vc sabe?", "{name} oq ta no seu?", "{name} tem isso ai no seu material?",
                     "{name}??", "e vc {name}"],
    "off_task_atom": ["kkkk", "mds", "mano", "kkkkkk", "ja volto", "show", "slk", "ue", "aff", "eita", "vish",
                      "socorro", "nossa"],
    "refusal": ["nao sei", "nao", "sla", "passo", "nem", "n"],
    "press": ["pq?", "como vc sabe?", "pq vc acha isso?", "mas pq?", "oq te faz pensar isso?", "certeza?"],
}

FACT_LEADS = ["", "", "", "no meu diz que ", "aqui diz que ", "aparentemente ", "li que ",
              "no material fala que ", "pelo que eu li, ", "pera aqui ta que "]
EXISTING_LEAD_RE = re.compile(r"^(meu card diz( que)?|o card diz( que)?|no meu diz( que)?|diz aqui( que)?|diz que|"
                              r"aqui diz( que)?|meu card:)\s*", re.I)
CARD_ANNOUNCE_RE = re.compile(
    r"\b(?:vou\s+|deixa eu\s+)?compartilhar\s+(?:do|o que (?:ta|tá|diz) n?o)\s+(?:meu|o|nosso)\s+"
    r"(?:card|cart[aã]o|material|texto)\s*[:,-]?\s*", re.I)
CARD_ANNOUNCE_LEADS = ["", "", "aqui diz que ", "aparentemente ", "li que "]

REF_LEADS = ["como eu disse ", "eu ja falei, ", "é como eu falei ", "de novo, ", "como eu falei antes "]
REF_LEADS_PEER = ["é como {peer} falou ", "isso, o que {peer} disse, ", "{peer} ja falou, ", "como {peer} disse, "]
REF_LEADS_NEUTRAL = ["é, ", "isso, ", "vdd, ", "é isso, "]

LAUGHS = ["kkkk", "kkkkk", "kkk", "kkkkkk"]
OPENERS = ["pera", "gente", "olha", "tipo", "entao", "mano", "galera", "ei"]
HEDGE_PREFIX = ["acho que ", "talvez ", "nao sei mas ", "sei la, "]
CLAIM_PREFIX = ["", "", "", "acho que ", "ok entao ", "entao ", "eu acho que "]
DISAGREE_PREFIX = ["nao ", "nada a ver, ", "acho que nao, ", "pera nao, ", "hmm nao "]
HEDGES = ["é algo assim mas sei la", "nao tenho certeza nessa", "a gente tem que ver o material",
          "tem a ver com os numeros?", "meio que entendi mas nao muito", "pera vou ler de novo"]
SAFE_QUESTIONS = ["oq vcs acham?", "pera oq vcs acham que é?", "entao oq a gente coloca nessa?"]
MINIMAL_ACT_LINES = {
    "question": ["alguem sabe?", "alguma ideia?", "hmm oq a gente acha?", "alguem sabe isso?"],
    "press": ["pera pq?", "como vc sabe?", "pq?"],
    "disagree": ["hmm nao sei nao", "pera sei la isso", "hmm sla"],
    "explain": ["hmm nao sei explicar", "sla como explicar"],
    "on_task_claim": ["hmm nao sei", "sla"],
    "_other": ["hmm", "ok", "nao sei"],
}
REPEAT_PREFIXES = ["pera ", "entao ", "ok ", "tipo "]
MOCK_ALSO = "e tambem"     # MockLLM's deliberate slip ("..., also <fact>")

# English chat slang that must never appear in a pt-BR line (the inverse of english_only)
EN_MARKERS_RE = re.compile(r"\b(lol|lmao+|lmfao|haha(?:ha)*|hehe|idk|omg|bruh|brb|tbh|yeah|yep|nope|"
                           r"guys|wait|okk|lemme|gonna|wanna|dunno|anyone|thx|pls)\b", re.I)
_EN_SUB = {"lol": "kkkk", "lmao": "kkkkkk", "lmfao": "kkkkkk", "hehe": "kkk", "idk": "sla", "omg": "mds",
           "bruh": "mano", "brb": "ja volto", "tbh": "sinceramente", "yeah": "é", "yep": "sim", "nope": "nao",
           "guys": "gente", "wait": "pera", "okk": "ok", "lemme": "deixa eu", "gonna": "vou", "wanna": "quero",
           "dunno": "sei la", "anyone": "alguem", "thx": "vlw", "pls": "pfv"}


def normalize_chat(text: str, rng) -> str:
    """pt-BR only: English chat slang / laughter in a template or model line -> pt-BR ("lol" -> "kkkk").
    Portuguese laughter "kkkk" is RIGHT here (the english_only filter applies to English only)."""
    if not EN_MARKERS_RE.search(text):
        return text

    def sub(m):
        w = m.group(0).lower()
        if w.startswith("haha") or w.startswith("lmao"):
            return rng.choice(["kkkk", "kkkkkk", "kkk"])
        return _EN_SUB.get(w, "")
    return re.sub(r"\s{2,}", " ", EN_MARKERS_RE.sub(sub, text)).strip() or "kkkk"


# ------------------------------------------------------------------ off-task topics (same ids as sim/offtask.py)
TOPICS: dict[str, tuple[str, list[str]]] = {
    "snack": ("comida / fome / um lanche", ["minha barriga ta roncando", "quero um pastel agora", 
        "to morrendo de fome", "alguem mais com fome", "trouxe salgadinho mas n posso comer aqui kkk",
        "oq vcs vao comer no almoço", "preciso de comida urgente"]),
    "game_last_night": ("um jogo de futebol de ontem", ["o gol de ontem foi lindo", "nem acredito no resultado de ontem", 
        "alguem viu o jogo ontem", "aquele jogo ontem foi loucura",
        "mano o juiz ontem foi mt ruim", "fiquei acordado vendo o jogo kkk",
        "perdemos de novo ontem affs"]),
    "videogame": ("um videogame que jogam", ["perdi minha sequencia ontem", "baixei um jogo novo ontem", 
        "alguem jogando a temporada nova", "subi pro diamante ontem", "quem quer jogar depois da aula",
        "meu controle quebrou kkk", "essa atualizacao estragou o jogo"]),
    "music": ("música / uma música / fone", ["descobri uma banda boa", "minha playlist ta repetindo tudo", 
        "to com uma musica na cabeça", "quem mais escuta musica fazendo isso",
        "meu fone descarregou", "esse album novo ta mt bom", "alguem coloca uma musica kkk"]),
    "weekend": ("planos pro fim de semana", ["sabado vou na casa da minha vó", "fds vai chover ne", 
        "oq vcs vao fazer no fds", "nao vejo a hora de chegar sabado", "fds chega logo pfv",
        "alguem vai na festa", "vou so dormir o fds todo"]),
    "tired": ("cansaço / sono", ["meus olhos tao fechando", "acordei atrasado hj", 
        "to com mt sono", "dormi tipo 4 horas", "nao to aguentando de sono", "preciso de um cochilo",
        "pq essa aula é tao cedo"]),
    "bored": ("tédio / a aula demorando", ["o tempo nao passa", "to quase dormindo aqui", 
        "isso ta demorando demais", "quem mais ta entediado", "quanto tempo falta",
        "ja ta acabando?", "essa aula ta durando uma eternidade hj"]),
    "internet": ("internet lenta / o computador", ["o wifi aqui de casa é horrivel", "quase caiu minha conexao", 
        "minha internet ta mt lenta", "meu wifi fica caindo", "o computador ta travando",
        "minha tela apagou do nada", "ta travando pra vcs tbm?"]),
    "phone": ("o celular / bateria / carregador", ["meu celular ta quente demais", "a tela do meu celular rachou", 
        "meu celular ta com 3 por cento", "esqueci meu carregador", "alguem tem carregador",
        "meu celular nao para de vibrar kkk", "o professor pegou meu celular semana passada"]),
    "pet": ("um bicho de estimação", ["meu gato deitou no teclado kkk", "meu cachorro ta latindo pro carteiro", 
        "meu cachorro comeu meu caderno serio", "minha gata ta mt chata ultimamente", "quero um cachorrinho",
        "meu cachorro latiu a noite toda", "a gente pegou um gatinho novo"]),
    "weather": ("o tempo / clima", ["ta abafado demais hj", "parece que vai chover", 
        "ta mt calor aqui", "ta chovendo la fora?", "odeio esse frio", "ta um sol hj",
        "o ar condicionado ta forte demais"]),
    "homework_other": ("dever de outra matéria", ["ainda nao fiz o trabalho de geografia", "a lista de fisica ta enorme", 
        "alguem fez o dever de mat", "tem prova amanha ne", "nao estudei pra historia",
        "a redação é pra hj?", "esqueci o outro dever"]),
    "teacher": ("o professor / a sala", [
        "o professor ta olhando pra gente kkk", "o professor vai voltar?", "pq ta tao barulhento aqui",
        "tem gente gritando na sala do lado", "a professora ta no celular kkk"]),
    "series": ("uma série ou vídeo", ["comecei uma serie nova ontem", "o ultimo episodio me deixou bolado", 
        "vcs viram o episodio novo", "sem spoiler pfv", "essa serie é mt boa",
        "maratonei a temporada toda", "o final foi mt ruim"]),
    "laugh": ("rindo de algo bobo", ["kkkkkkkk nao aguento", "ri sozinho aqui", 
        "kkkkkkkkk", "kkkkk para", "to chorando kkk", "pq isso é tao engraçado", "kkkkkk"]),
    "home": ("ir pra casa / depois da aula", ["hj vou direto pra cama", "minha mae vai me buscar mais tarde", 
        "bora pra casa", "que horas acaba a aula", "quero ir pra casa", "o onibus ta atrasado hj?",
        "quem vai a pé pra casa depois"]),
    "birthday": ("um aniversário (o seu ou de alguém da sua família, nunca de um colega do chat)", ["o aniversario da minha irma é sabado", "vai ter bolo la em casa hj", 
        "hj é aniversario do meu primo", "tem bolo em casa kkk",
        "a gente devia fazer uma festa", "meu aniversario é semana que vem"]),
    "football_team": ("o time do coração", ["meu time ta jogando mal demais", "domingo tem classico", 
        "meu time vai ganhar o campeonato", "vcs torcem pra quem", "a gente joga amanha",
        "aquela contratação foi loucura", "melhor time do brasil kkk"]),
    "recess": ("o intervalo / recreio", [
        "ja é o intervalo?", "nao vejo a hora do intervalo", "quem vai na cantina",
        "bora jogar no intervalo", "quanto tempo é o intervalo hj"]),
    "clothes": ("tênis / roupa", ["meu casaco novo é mt confortavel", "minha meia ta furada kkk", 
        "comprei um tenis novo", "meu tenis ta encharcado", "amei esse moletom", "dia de uniforme affs",
        "esqueci meu casaco"]),
    "social_media": ("um post / trend das redes", ["vi um meme mt bom", "esse video ta em todo lugar", 
        "vcs viram aquele video", "essa trend é mt cringe", "postei um negocio kkk",
        "todo mundo fazendo aquela dancinha", "olha o grupo depois"]),
    "sibling": ("um irmão ou irmã", ["meu irmao pegou meu carregador", "minha irma nao para de cantar", 
        "meu irmao é mt chato", "minha irma pegou meu fone", "meu irmao mais novo quebrou meu celular",
        "tenho que buscar minha irma depois", "meu irmao nao para de me mandar msg"]),
    "random_omg": ("uma exclamação aleatória", [
        "mds", "ue oq", "mano", "nem a pau", "show"]),
}
TOPIC_DEFAULT_DESC = "comida, futebol, tédio, a aula, a internet"

# ------------------------------------------------------------------ question detection (sim/content.py)
# In pt-BR a yes/no question normally keeps declarative word order ("a enzima desnatura acima de 60?"),
# so "?" alone marks a question; tag questions ("..., né?") and hedged assertions ("acho que é 37?")
# still assert. Text is accent-folded before matching.
QUESTION_START = ("o que", "oq", "que", "qual", "quais", "como", "por que", "porque", "pq", "pra que",
                  "para que", "onde", "cade", "quando", "quem", "quanto", "quantos", "quantas", "sera que",
                  "tem como", "da pra", "e pra", "pode", "alguem sabe", "vcs sabem", "voces sabem",
                  "pera o que", "entao o que", "mas o que", "mas por que", "e agora")
TAG_Q = re.compile(r"(?:,|\s)\s*(?:ne|certo|nao e|tipo|ok|sim|ta|neh|nao)\s*\?+\s*$")
CAUSAL = re.compile(r"\b(?:por que|porque|pq|por causa|ja que|como que|o que faz|oq faz)\b")
UPTALK = re.compile(r"^(?:entao\s+|tipo\s+|ok\s+|pera\s+|e\s+)*(?:eu acho|acho|eu acredito|talvez|provavelmente|"
                    r"deve ser|com certeza|na verdade)\b")
ADDRESS = re.compile(r"^@?[a-z]+,\s+")
STOPWORDS = frozenset("""
a o as os um uma uns umas e ou mas entao se que de do da dos das no na nos nas em num numa pra pro pras pros
para por pelo pela com sem ao aos eu tu ele ela nos vos eles elas voce vc vcs voces me te se lhe mim meu minha
meus minhas seu sua seus suas nosso nossa isso isto aquilo esse essa este esta aquele aquela ai aqui la ca
e eh foi era sao ser estar esta estao ta to tava ter tem tinha tenho fazer faz fez ir vai vou vamos
nao n sim ja so tambem tbm muito mt mais menos tipo ne ok kkk kkkk sla sei la acho acha achei
oq pq porque como quando onde qual quem quanto ainda agora depois antes sempre nunca bem mal coisa coisas
gente alguem nada tudo todo toda todos todas outro outra cada mesmo mesma assim pode pra
diz dizer disse falou falar fala card material
""".split())


def stem(w: str) -> str:
    for suf in ("mente", "coes", "cao", "oes", "ando", "endo", "indo", "adas", "ados", "ada", "ado", "ida",
                "ido", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            w = w[: -len(suf)]
            break
    return w[:-1] if w[-1:] in ("a", "o", "e") and len(w) > 3 else w


# ------------------------------------------------------------------ attribution (sim/content.py)
SELF_REF_RE = re.compile(r"\b(?:[eé],?\s+)?como\s+eu\s+(?:disse|falei|tinha falado|tava falando)"
                         r"(?:\s+(?:antes|ja|j[aá]))?|\beu\s+j[aá]\s+(?:disse|falei)(?:\s+(?:isso|antes))?"
                         r"|\bde novo,", re.I)
AGAIN_PREFIX = "de novo"


def credit_peer(name: str) -> str:
    return f"como {name} falou"


CREDIT_NONE = "é"
CREDIT_AGAIN = "é,"

# ------------------------------------------------------------------ repetition guard / realizer notes
REPHRASE = ("DIGA DE OUTRO JEITO: seu rascunho ficou quase igual a uma mensagem anterior do chat "
            "(\"{prev}\"). Diga o mesmo conteúdo com palavras bem diferentes, ou mais curto.")
STRICT = ("ESTRITO: seu rascunho anterior mencionou algo que você NÃO sabe. Reescreva usando SÓ o "
          "conteúdo acima; se estiver em dúvida, enrole ou diga sla.")
PROMPT_LEAK_NOTE = ("LIMPO: seu rascunho anterior tinha texto de instrução ou rótulo (tipo 'pergunte ao grupo:' ou "
                 "'diga:'). Escreva SÓ a mensagem de chat, exatamente como o aluno digitaria.")

# ------------------------------------------------------------------ realizer prompt (llm/prompts.py)
STYLE_GUIDE = """Você escreve UMA mensagem de chat de um(a) estudante do ensino fundamental/médio num chat \
em grupo online durante uma atividade de aula, em português do Brasil. Registro: bem informal, curto, \
minúsculas são normais, sem listas, sem emoji a não ser que seja natural, nunca pareça professor nem IA. \
Os alunos muitas vezes estão distraídos, secos e inseguros. Escreva como adolescente brasileiro escreve no \
chat: abreviações (vc, pq, oq, tbm, n, sla, vdd, mds, blz), muitas vezes sem acento, risada "kkkk", erros de \
digitação de vez em quando.

REGRAS
- Diga SÓ o conteúdo dado em DIGA ESTE CONTEÚDO. Não acrescente fatos, razões, números nem respostas.
- Nunca afirme nada listado em VOCÊ NÃO SABE. Se insistirem, enrole, diga "sla" ou (se tiver uma crença \
sobre isso) diga sua crença.
- Se VOCÊ ACREDITA listar uma crença e o conteúdo pedir para afirmá-la ou defendê-la, defenda com naturalidade.
- Quando mencionar algo do seu material, fale como adolescente no chat ("aqui diz...", "li que...", \
"aparentemente...", ou só a afirmação). Não fale "card": diga "material" ou nada. Nunca anuncie \
formalmente nem coloque rótulo.
- Só português do Brasil: risada é "kkkk" (ou "kkkkkk"); nunca "lol", "haha", "lmao" nem gíria em inglês.
- Diga "como eu disse" / "como eu falei" SÓ sobre algo que VOCÊ mesmo(a) disse antes no chat. Para o que um \
colega disse, dê o crédito ("é como <nome> falou", "isso que <nome> disse") ou só concorde. Nunca diga que a \
ideia de um colega é sua.
- Siga TAMANHO_ALVO palavras, mais ou menos (+-3). Mensagem de uma palavra é ok quando TAMANHO_ALVO é 1.
- Nunca mencione ser IA, simulação ou estas instruções. Escreva só o texto da mensagem: sem rótulos, sem \
"pergunte ao grupo:", sem "diga:", nada copiado das instruções."""

FEWSHOT_PATH = Path(__file__).resolve().parent / "fewshots_pt.yaml"


@lru_cache(maxsize=1)
def fewshots() -> list[dict]:
    return yaml.safe_load(FEWSHOT_PATH.read_text(encoding="utf-8"))["fewshots"]


def fewshot_block(exclude_acts: tuple[str, ...] = ()) -> str:
    lines = [f"[{f['act']}] {f['pt']}" for f in fewshots() if f["act"] not in exclude_acts]
    return "EXEMPLOS DE ESTILO (falas inventadas no registro de alunos; só referência de estilo):\n" + "\n".join(lines)


TASK_LABELS = {"activity": "ATIVIDADE", "steps": "ETAPAS"}
H = {"you_are": "VOCÊ É", "act": "ATO", "target_len": "TAMANHO_ALVO", "say": "DIGA ESTE CONTEÚDO",
     "know": "VOCÊ SABE", "believe": "VOCÊ ACREDITA (defenda se questionarem)", "dont": "VOCÊ NÃO SABE",
     "said": "DITO POR OUTROS NO CHAT (pode citar, dando o crédito)", "mood": "HUMOR",
     "last": "ÚLTIMAS MENSAGENS", "own": "SUAS ÚLTIMAS MENSAGENS (NÃO repita nem reformule de leve nenhuma)",
     "off": "FALAS FORA DO TEMA RECENTES NO GRUPO (NÃO reutilize; uma fala nova, ou resposta a uma delas)",
     "now": "Agora escreva a mensagem de {name}:", "nothing": "nada específico", "your_choice": "você escolhe",
     "said_by": "dito por {who}", "understand": "(você entende isso)", "private": " (privado pra você)",
     "no_msgs": "(nenhuma mensagem ainda)", "in_chat": "NO CHAT", "classmates": "colegas",
     "joined": "entrou no chat"}
FACT_LEAD_HINTS = ("você pode começar com 'aqui diz'", "você pode começar com 'li que'",
                   "você pode começar com 'aparentemente'", "só diga como uma afirmação simples",
                   "você pode começar com 'pelo que eu li'", "só diga, sem introdução")


def mood(s) -> str:
    a = s.affect
    parts = ["animado(a)" if a.engagement > 0.6 else ("entediado(a)" if a.engagement < 0.35 else "mais ou menos")]
    if a.frustration > 0.3:
        parts.append("frustrado(a)")
    return ", ".join(parts)


GIST_NOTE = (" (isso é só a ideia: diga com suas palavras, curto, do jeito que aluno digita; nunca copie o texto "
             "do material)")


def _gist(task, fid: str) -> str:
    from ..sim.realizer import fact_gist
    return fact_gist(task, fid)


def say_this(task, plan, names: dict, variant: int = 0, speaker: str | None = None) -> str:
    """pt-BR version of llm.prompts.say_this (same branches, same content; nothing copyable)."""
    from ..sim import intents
    a = plan.act
    to = names.get(plan.addressee, "") if plan.addressee else ""
    hint = FACT_LEAD_HINTS[variant % len(FACT_LEAD_HINTS)]
    if a == "question" and plan.follow_up:
        if to:
            return (f"a pergunta já foi feita e ninguém respondeu; pergunte direto pra {to} se sabe "
                    "(bem curto), e NÃO repita a pergunta")
        return ("a pergunta já foi feita e ninguém respondeu; uma cutucada bem curta pro grupo, "
                "e NÃO repita a pergunta")
    if a == "off_task":
        from ..sim.offtask import description
        return (f"um comentário fora do tema sobre {description(plan.topic, task)}, pro chat todo (sem chamar um colega "
                "pelo nome)")
    if plan.claim == "misconception" and plan.misconception_id and plan.rationale:
        mm = task.misconception(plan.misconception_id)
        why = mm.because[variant % len(mm.because)] if mm.because else "um motivo seu"
        return (f"{to or 'um colega'} perguntou por que você acha isso; responda com o seu motivo, curto, de boa, com "
                f"outras palavras (só o motivo, nunca a ideia certa). O motivo: {why}. Sua crença: " + mm.wrong_belief)
    if plan.claim == "misconception" and plan.misconception_id:
        if plan.pushback:
            return ("questionaram sua ideia; defenda UMA vez, com um motivo seu e outras palavras (sem repetir "
                    "o que você já disse): " + task.misconception(plan.misconception_id).wrong_belief)
        return ("você acredita nisso e fala como opinião sua: "
                + task.misconception(plan.misconception_id).wrong_belief)
    ref = _reference_instruction(task, plan, names, speaker) if plan.claim == "fact" and plan.reference else None
    claim = (task.kc(plan.kc).correct_claims[plan.claim_idx if plan.claim_idx is not None else 0]
             if plan.claim == "correct" and plan.kc else None)
    if plan.unsure:
        return (f"{to or 'um colega'} te perguntou; diga curto que você não tem certeza / que no seu material não "
                "tem isso (sem conteúdo)")
    if a == "disagree" or plan.family:
        lead = f"você discorda de {to}. " if to else "você contesta o que foi dito. "
        if plan.addresses_misconception:
            lead = (f"{to or 'alguém'} acha isto (errado): {task.misconception(plan.addresses_misconception).wrong_belief} "
                    f"Você {'discorda de' if a == 'disagree' else 'responde pra'} {to or 'essa pessoa'} mostrando por que "
                    "não é assim. ")
        if plan.family == "counterexample" and plan.addresses_misconception:
            cx = task.misconception(plan.addresses_misconception).counterexample
            return lead + "Dê este contraexemplo com suas palavras: " + (cx[0] if cx else "um contraexemplo")
        if plan.family == "press_for_reasoning":
            return f"pergunte pra {to or 'eles'} por que acham isso (sem conteúdo seu)"
        if ref:
            return lead + ref
        if plan.claim == "fact" and plan.fact_ids:
            return (lead + f"Use o que o seu material diz, de boa ({hint}). A ideia: "
                    + _gist(task, plan.fact_ids[0]) + GIST_NOTE)
        if claim:
            return lead + "A ideia que você usa, com suas palavras (ela tem que ir contra essa crença): " + claim
    if ref:
        return ref
    if plan.claim == "fact" and plan.fact_ids:
        return (f"fale isso do seu material com suas palavras, de boa ({hint}). A ideia: "
                + _gist(task, plan.fact_ids[0]) + GIST_NOTE)
    if claim:
        verb = "repita com suas palavras a ideia" if plan.revoice else "passe esta ideia com suas palavras"
        return f"{verb}: {claim}"
    if plan.claim == "hedge":
        return "você não tem certeza; enrole de leve sem dizer nenhuma resposta nem razão"
    if a == "interface":
        return ("uma fala curta sobre a tela da atividade: "
                + (intents.description(plan.intent, task) if plan.intent else "como a interface funciona"))
    if a == "question" and plan.asked_fact:
        return ("pergunte isto pros outros, com suas palavras, escrevendo só a pergunta. "
                "A pergunta: " + task.fact(plan.asked_fact).ask)
    if a == "question":
        return intents.description(plan.intent, task) if plan.intent else "pergunte o que é pra fazer"
    return {
        "greeting": "cumprimente o grupo", "backchannel": "uma reação curta (ok/kkkk/sim)",
        "agree": f"concorde com {to or 'a última mensagem'} sem acrescentar conteúdo",
        "idk": "diga que não sabe", "refusal": "recuse / diga não ou sla",
        "press": f"pergunte pra {to or 'eles'} por que acham isso",
    }.get(a, "uma reação curta")


def _reference_instruction(task, plan, names: dict, speaker: str | None) -> str:
    fact = task.fact(plan.fact_ids[0]).text
    if plan.ref_by and plan.ref_by == speaker and plan.restate:
        return ("você mesmo(a) disse isso antes e questionaram; retome BEM rápido, poucas palavras (tipo começando "
                "com 'como eu disse'); já foi dito, então não anuncie como novidade: " + fact)
    if plan.ref_by and plan.ref_by == speaker:
        return ("você mesmo(a) disse isso antes no chat; repita BEM rápido e direto, poucas palavras, sem "
                "'como eu disse'; não anuncie como novidade: " + fact)
    who = names.get(plan.ref_by) if plan.ref_by else None
    if who:
        return (f"{who} (não você) disse isso antes no chat; concorde / complemente BEM rápido, dando o crédito "
                f"pra {who} (tipo começando com 'é como {who.lower()} falou' ou 'isso que {who.lower()} disse'); "
                "nunca 'como eu disse'; já foi dito, então não anuncie como novidade: " + fact)
    return ("isso já foi dito no chat (não por você); concorde BEM rápido, em poucas palavras, nunca "
            "'como eu disse'; não anuncie como novidade: " + fact)


# ------------------------------------------------------------------ prompt labels (sim/promptleak.py)
PROMPT_LEAK_LABEL_WORDS = (r"pergunte(?: ao grupo| pra todos| pra eles| pro grupo)?|diga(?: isso)?|compartilhe|"
                        r"mencione|responda|discorde(?: de \w+)?|concorde(?: com \w+)?|reformule|"
                        r"cren[cç]a|afirma[cç][aã]o|conte[uú]do|mensagem|sa[ií]da")
PROMPT_LEAK_HEADERS = (r"DIGA ESTE CONTE[UÚ]DO|DIGA DE OUTRO JEITO|TAMANHO_ALVO|VOC[EÊ] SABE|VOC[EÊ] ACREDITA|"
                    r"VOC[EÊ] N[AÃ]O SABE|VOC[EÊ] [EÉ]|HUMOR|[UÚ]LTIMAS MENSAGENS|ESTRITO|LIMPO|ATO|DITO POR OUTROS")
PROMPT_LEAK_PHRASES = (r"com suas (pr[oó]prias )?palavras|sem conte[uú]do seu|n[aã]o repita|n[aã]o anuncie|"
                    r"bem curto e|como adolescente|voc[eê] pode come[cç]ar com|sem dizer nenhuma resposta")
# 0.5.2: second-person directives addressed to the WRITER of the line (matched on accent-folded text):
# "voce acredita nisso e fala como opiniao sua: ...", "diga que ...", "escreva so a pergunta", "a ideia: ..."
PROMPT_LEAK_DIRECTIVES = (
    r"\bvoce (?:acredita (?:nisso|nessa)|discorda d[eoa]|contesta o que|se convenceu|nao tem certeza|"
    r"mesmo\(?a?\)? disse|pode comecar|tem (?:essa|uma) crenca)\b|\bfal[ae] como (?:opiniao|se fosse)|"
    r"\bcomo (?:opiniao|crenca) sua\b|\bopiniao sua\b|\bdiga (?:que|isso|o mesmo|de outro jeito|so)\b|"
    r"\bescreva\b|\bdefenda\b|\bquestionaram sua ideia\b|\bcom um motivo seu\b|\bde este contraexemplo\b|"
    r"\bpasse esta ideia\b|\brepita (?:com|bem)\b|\bpergunte (?:pra|pro|isto|direto|o que)\b|\bfale isso\b|"
    r"\bsem acrescentar\b|\bsem conteudo\b|\bsem puxar assunto\b|\buma (?:cutucada|reacao|fala) (?:bem )?curta\b|"
    r"\bum comentario (?:curto|fora do tema)\b|\bfora do tema sobre\b|\bdando o credito\b|\bnunca 'como eu disse'|"
    r"\bnao anuncie\b|\bcumprimente\b|\brecuse\b|\bconcorde com (?:a ultima|o que|a resposta)\b|"
    r"\bdiga bem curto\b|\bbem curto\)|\b(?:a ideia|a pergunta|o motivo|o detalhe|sua crenca|sua ideia antiga)\s*:|"
    r"\bdo seu material com\b|\buse o que o seu material\b|\bescrevendo so\b|\bo chat ficou parado\b|"
    r"\bninguem respondeu\b|\ba pergunta ja foi feita\b|\bresponda (?:a duvida|ao comentario)\b|"
    r"\bproponha\b|\bsugira\b|\bde um ok\b|\bvoce digitou\b|\bvoce se convenceu\b|\bisso e so a ideia\b|"
    r"\bnunca copie\b|\bdo jeito que aluno\b|^\W*(?:diga|escreva|fale|pergunte|responda|defenda|concorde|discorde|"
    r"repita|passe|mencione|reformule|sugira|proponha|cumprimente|recuse|enrole|perguntar|dizer|falar|responder|"
    r"comentar|concordar|discordar)\b|\buma fala curta\b|\bem 1-2 frases\b|\bpeca confirmacao\b|\bnada novo\b|"
    r"\bacha isto \(errado\)|\bmostrando por que\b|\bperguntou por que voce acha\b|\bresponda com o seu motivo\b|"
    r"\bela tem que ir contra\b|\bte perguntou; diga\b")


# ------------------------------------------------------------------ realism controls (sim/realism.py)
# All lines below are invented for the simulator (not corpus lines).
PHYSICAL_RE = re.compile(r"\bprof(?:essor|essora|a|e)?\b|\bsala(?: de aula| do lado)?\b|\bintervalo\b|\brecreio\b|"
                         r"\bcantina\b|\buniforme\b|\blousa\b|\bcarteira do lado\b")
PERIOD_WORDS = {
    "manha": r"\b(boa tarde|boa noite|jantar|janta|jantei)\b",
    "tarde": r"\b(bom dia|boa noite|jantar|janta|jantei|cafe da manha)\b",
    "noite": r"\b(bom dia|boa tarde|almoc\w*|cafe da manha)\b",
}
PERIOD_NAMES = {"manha": "manhã", "tarde": "tarde", "noite": "noite"}
GREETING_BY_PERIOD = {"manha": ["bom dia", "bom dia gente"], "tarde": ["boa tarde", "boa tarde gente"],
                      "noite": ["boa noite", "boa noite gente"]}
ROLE_WORDS_RE = (r"\b(facilitador\w*|tutor\w*|monitor\w*|o bot|um bot|chatbot|robo\w*|a ia|uma ia|"
                 r"inteligencia artificial)\b")
REALISM_NOTES = {
    "plural_address": ("REESCREVA: no chat são só vocês dois: fale com {name} no singular (vc), nunca 'gente', "
                       "'galera', 'vcs' ou 'alguém'."),
    "question_repeat": "NÃO PERGUNTE DE NOVO: você já perguntou isso (\"{prev}\"). Diga outra coisa, ou reaja curto.",
    "closing_repeat": ("DIGA DE OUTRO JEITO: alguém já respondeu a proposta assim (\"{prev}\"); dê o ok com outras "
                       "palavras, curto."),
    "offtask_named": "REESCREVA: um comentário fora do tema pro chat todo, sem chamar nenhum colega pelo nome.",
    "own_content_repeat": ("NÃO SE REPITA: você já disse isso (\"{prev}\"). Diga outra coisa, ou só reaja curto; "
                           "nunca diga o mesmo fato de novo."),
    "near_duplicate_session": ("DIGA DE OUTRO JEITO: seu rascunho ficou quase igual a uma mensagem anterior do chat "
                               "(\"{prev}\"). Diga com palavras bem diferentes, ou mais curto."),
    "participant_mention": ("REESCREVA: fale só com quem já escreveu neste chat, pelo nome, sem '@'; não existe "
                            "professor, tutor nem IA aqui."),
    "physical_classroom": "REESCREVA: é um chat online, cada um no seu computador; sem sala de aula nem professor por perto.",
    "wrong_time_of_day": "REESCREVA: mantenha o período do dia desta sessão (veja CONTEXTO DA SESSÃO).",
}
TYPO_ABBREV = {"você": "vc", "voce": "vc", "vocês": "vcs", "voces": "vcs", "também": "tbm", "tambem": "tb",
               "porque": "pq", "não": "n", "nao": "n", "que": "q", "está": "ta", "tá": "ta", "beleza": "blz",
               "verdade": "vdd", "muito": "mt", "hoje": "hj", "mesmo": "msm", "quando": "qnd", "para": "pra"}
OFFTASK_REPLY = ["sério?", "nossa eu tbm", "pior que é", "ah eu vi isso", "aff nem fala", "eu também",
                 "mds sim", "nem me fala", "ah é?", "que isso"]
INTERFACE_ANSWERS = {
    "save_mechanism": ["acho que salva sozinho sim", "o meu salvou quando cliquei fora da caixa"],
    "slider_meaning": ["acho que é o quanto a gente tem certeza da resposta", "acho q é tipo o nivel de certeza da hipótese"],
    "where_to_type": ["tem uma caixinha embaixo da pergunta", "é na caixa de texto logo abaixo"],
    "send_how": ["tem um botão de enviar no canto", "é só clicar em enviar embaixo"],
    "next_button": ["acho que aparece depois que a gente responde", "rola até o fim que tem o botão"],
    "scroll": ["tenta clicar na página e rolar de novo", "a minha desceu quando cliquei no meio"],
    "find_activity": ["tá na primeira aba lá em cima", "atualiza a página que aparece"],
    "answer_each": ["acho que é uma resposta pro grupo", "cada um escreve, mas pode ser a mesma"],
}
CLOSING_PROPOSE_LEADS = ["bora fechar a resposta do grupo: ", "então a gente coloca assim: ", "resumindo pro grupo: ",
                         "posso escrever assim pro grupo? ", "vou mandar assim, ok? ", "fica assim entao: ",
                         "acho que a resposta final é: ", "pra fechar: ", "deixa eu juntar tudo: ", "entao fica: ",
                         "juntando o que a gente falou: "]
CLOSING_PROPOSE_TAILS = ["", "", "", " pode ser?", " ta bom assim?", " blz?"]
CLOSING_JOINS = ["; e ", ", e ", ". e tbm ", " + ", ", tbm que "]
# 0.5.3: closing okays composed per session (ack + verdict + send), never two alike in a session
CLOSING_ACKS = ["fechou", "boa", "isso", "show", "beleza", "perfeito", "massa", "otimo", "tranquilo", "uhum",
                "sim sim", "top", "isso ai", "ok", "certinho"]
CLOSING_VERDICTS = ["ficou bom", "ta completo", "acho que é isso", "concordo", "ta certinho", "ficou bem explicado",
                    "é isso mesmo", "faz sentido tudo", "nao falta nada", "tudo certo", "eu tbm acho", "ficou otimo",
                    "ta bem resumido", "foi o que a gente falou"]
CLOSING_SENDS = ["", "", "pode mandar", "manda assim", "envia ai", "bora enviar", "pode enviar", "manda ai",
                 "pode colocar", "manda logo"]
CLOSING_JOIN = "; e "
CLOSING_AGREE = ["fechou, pode mandar assim", "isso, concordo com essa resposta", "boa, ficou completo",
                 "por mim tá ótimo assim"]
CLOSING_ADJUST_LEADS = ["concordo, só coloca também que ", "isso, e acrescenta que "]
CREDIT_LINES = ["é isso que {peer} falou", "{peer} já falou isso, concordo", "isso, igual {peer} disse",
                "acho que {peer} tá certo nessa"]
BELIEF_CREDIT_LINES = ["eu tbm acho isso que {peer} falou", "concordo com {peer} nessa", "pensei a mesma coisa que {peer}"]
CONCEDE_LINES = ["ah entendi, faz sentido", "hmm ok, acho que vc ta certo", "ah ta, entao nao é só isso",
                 "verdade, nao tinha pensado assim", "blz, entendi agora"]
TAG_LINES = ["ne isso?", "vcs concordam?", "faz sentido isso?", "acho q é isso", "foi oq eu entendi",
             "ta certo isso?", "oq vcs acham?"]
OPENERS_LONG = ["pera achei aqui", "olha oq diz aqui", "gente olha isso", "tipo assim", "entao olha", "pera ai gente"]
DRAFT_PROPOSE_LEADS = ["vou colocar: eu acho que ", "posso escrever assim? eu acho que ", "bora escrever essa: eu acho que ",
                       "coloquei: eu acho que "]
DRAFT_JOIN = " e "
DRAFT_AGREE = ["isso, pode colocar assim", "boa, coloca isso", "pode ser assim msm", "ficou bom assim"]
DRAFT_CONFIDENCE = ["coloquei uns {pct}% na confiança", "eu botei {pct} na barrinha", "quanto vcs colocaram de confiança?",
                    "{pct}% de confiança ta bom?"]
DRAFT_TYPED = ["pronto, coloquei", "escrevi ja", "mandei aqui", "coloquei la"]
NUDGE_LINES = ["e ai gente, bora continuar?", "gente cade vcs", "alguem ai? bora fazer", "e ai, oq a gente faz nessa?",
               "gente vamo terminar essa parte", "ei, vcs tao ai?"]
STEP_Q_LEADS = ["gente a pergunta é: ", "entao, ", "pera, ", "gente, "]
CLOSING_AGREE_BRIEF = ["blz", "pode ser", "ok", "fechou", "sim", "bora", "uhum"]
FACT_LEADS_FRAMED = ["aqui diz que ", "li que ", "aparentemente ", "no meu diz que ",
                     "acho que tem a ver com isso: ", "pelo que eu li, ", "no material fala que "]
CLOSING_PARTIAL_LEADS = ["gente o tempo ta acabando, bora mandar o que deu: ", "vamos mandar o que a gente tem: ",
                         "acho melhor mandar assim mesmo: "]
CLOSING_PARTIAL_EMPTY = ["gente o tempo ta acabando, bora mandar o que deu", "vamos mandar assim mesmo o que deu"]
PUSHBACK_LEADS = ["hmm mas eu ainda acho que ", "sei nao, pra mim ", "mas pera, eu continuo achando que ",
                  "mas eu ainda acho que "]
# 0.5.2: answers to "pq vc acha isso?" / a name ping (invented lines)
RATIONALE_LEADS = ["", "", "ah ", "sei la, ", "tipo "]
RATIONALE_FALLBACK = ["sei la, pra mim faz sentido assim", "ah pq foi oq eu sempre ouvi falar"]
UNSURE_LINES = ["foi so oq eu pensei mesmo", "nao tenho certeza na vdd", "hmm agora fiquei na duvida"]
PING_NO_LINES = ["no meu nao tem isso", "nao tenho isso no meu material", "hmm no meu nao fala disso",
                 "sei la, no meu nao aparece"]
COHERENCE_NOTE = ("COERENTE: seu rascunho anterior não fazia sentido ({reason}). Escreva de novo fazendo exatamente o que "
                  "DIGA ESTE CONTEÚDO pede: uma fala de crença diz só a crença (nunca a ideia que a derruba); uma "
                  "contestação fala contra a crença que ela responde, com o conteúdo dado, e nunca repete o que o "
                  "outro disse.")
# framing leads (sim/realism.cap_framing): matched on lowercased, accent-stripped text
LEAD_PREFIX = r"^(?:(?:entao|pera|ok|tipo|ah|e|mas|hmm|olha)[,\s]+)*"
LEAD_FORMS = [
    ("card", r"(?:(?:no|pelo|o|aqui no)\s+)?(?:meu\s+)?card\b(?:\s+(?:diz|fala|ta|tem))?(?:\s+que)?\s*[:,]?\s*"),
    ("no_meu", r"no\s+meu\s+(?:diz|ta|fala|tem)(?:\s+que)?\s*"),
    ("material", r"(?:no|o|meu|pelo)\s+material\s*(?:diz|fala|ta|tem)?(?:\s+que)?\s*[:,]?\s*"),
    ("aqui_diz", r"(?:aqui\s+(?:diz|ta|fala)|diz\s+aqui)(?:\s+que)?\s*"),
    ("diz_que", r"diz\s+que\s+"),
    ("aparentemente", r"aparentemente,?\s*"),
    ("li_que", r"(?:eu\s+)?li\s+que\s+"),
    ("pelo_que_li", r"pelo\s+que\s+eu\s+li,?\s*"),
    ("tem_a_ver", r"acho\s+que\s+tem\s+a\s+ver\s+com\s+isso:?\s*"),
]
CARD_REPLACE = [(r"\bno meu card\b", "no meu material"), (r"\bpelo meu card\b", "pelo que eu li"),
                (r"\bmeu card\b", "meu material"), (r"\bo card\b", "o material"), (r"\bcards?\b", "material")]
NATURAL_LEADS = ["li que ", "aqui diz que ", "aparentemente ", "pelo que eu li, ", "no material fala que ",
                 "acho que tem a ver com isso: "]
NEUTRAL_LEAD = "acho que "
REALISM_STYLE_DROP = ", erros de digitação de vez em quando"
REALISM_CONTEXT = ("CONTEXTO DA SESSÃO: chat online, cada aluno no seu computador (não existe sala física: nada de "
                   "professor falando na sala, colega do lado, intervalo ou cantina). Período do dia: {period}; "
                   "cumprimentos e comentários combinam com esse período. Escreva as palavras sem erro de digitação "
                   "(o sistema cuida disso). Risada (kkkk) só de vez em quando e nunca numa afirmação sobre o conteúdo.")
REALISM_PARTICIPANTS = ("Fale só com quem já escreveu neste chat (veja NO CHAT), pelo nome e sem '@'; não existe "
                        "professor, tutor nem IA aqui. Nunca chame nem mencione mais ninguém.")
REALISM_FORMAT = ("FORMATO: frase completa, com um mínimo de contexto (tipo 'acho que tem a ver com...', "
                  "'li que...'), nunca um fragmento solto.")
LONG_NOTE = ("CURTO: seu rascunho ficou longo e com cara de texto do material; escreva como aluno digita no chat, "
             "com suas palavras, no máximo {n} palavras.")
TRUNCATED_NOTE = "CURTO: sua mensagem anterior ficou longa demais e foi cortada; escreva uma frase completa e curta."
REALISM_CONTINUATION = ("CONTINUAÇÃO: você acabou de mandar a mensagem anterior; continue o MESMO assunto "
                        "(complete o pensamento), sem puxar assunto novo.")


def say_this_realism(task, plan, names: dict) -> str | None:
    """pt-BR wording instructions for the realism-only plans (closing, interface answers, off-task replies)."""
    from ..sim import intents
    from ..sim.offtask import description
    to = names.get(plan.addressee, "") if plan.addressee else ""
    if plan.closing == "propose" and plan.partial:
        from ..sim.realizer import idea_lines
        items = "; ".join(idea_lines(task, plan.ideas))
        if not items:
            return "o tempo está acabando; proponha mandar a resposta do grupo assim mesmo, curto, sem conteúdo novo"
        return ("o tempo está acabando: proponha mandar a resposta do grupo com o que vocês já concluíram, em 1-2 "
                "frases, e peça confirmação. Só isto (nada novo): " + items)
    if plan.closing == "propose":
        from ..sim.realizer import idea_lines
        items = "; ".join(idea_lines(task, plan.ideas) if plan.ideas else [k.correct_claims[0] for k in task.kcs])
        return ("proponha a resposta final do grupo, resumindo em 1-2 frases o que vocês concluíram no chat (com "
                "suas palavras), e peça confirmação. O que o grupo concluiu: " + items)
    if plan.closing == "agree" and plan.brief:
        return f"dê um ok na resposta final que {to or 'o colega'} propôs, em uma ou duas palavras"
    if plan.closing == "agree":
        return (f"dê o ok na resposta final que {to or 'o colega'} propôs, com suas palavras, curto; sem conteúdo novo "
                "e não do jeito que os outros já responderam")
    if plan.concedes:
        return (f"você se convenceu com o que {to or 'o colega'} disse: diga rápido que ele(a) tem razão e que você "
                "entendeu (sem conteúdo novo, sem repetir sua ideia antiga)")
    if plan.tag:
        return ("um complemento bem curto da sua própria mensagem anterior (tipo conferir se os outros concordam), "
                "sem conteúdo novo")
    if plan.draft:
        from ..sim.realizer import idea_lines
        if plan.draft == "propose":
            items = "; ".join(idea_lines(task, plan.ideas)[:2])
            return ("sugira o que o grupo escreve como resposta desta atividade, como hipótese começando com "
                    "'Eu acho', numa frase curta, e pergunte se pode ser. Só este conteúdo: " + items)
        if plan.draft == "confidence":
            return ("um comentário curto sobre a barrinha de confiança dessa resposta (quanto você tem certeza, "
                    "tipo uma porcentagem), ou pergunte quanto os outros colocaram")
        if plan.draft == "typed":
            return "diga bem curto que você digitou / mandou a resposta"
        return f"diga que a resposta que {to or 'o colega'} sugeriu pode ser escrita assim (curto, sem conteúdo novo)"
    if plan.nudge:
        return "o chat ficou parado; uma cutucada curta pro grupo continuar (sem conteúdo, sem perguntar a resposta)"
    if plan.step_q:
        from ..sim.realizer import step_question
        return ("o chat ficou parado; pergunte pro grupo a pergunta da atividade em que vocês estão, curto e com "
                "suas palavras: " + step_question(task, plan.kc))
    if plan.closing == "adjust" and plan.fact_ids:
        return (f"concorde com a resposta de {to or 'o colega'} mas acrescente este detalhe, curto: "
                + task.fact(plan.fact_ids[0]).text)
    if plan.act == "interface" and plan.answers_intent and plan.intent:
        return (f"responda a dúvida de {to or 'um colega'} sobre a tela ({intents.description(plan.intent, task)}): "
                "uma dica curta e plausível")
    if plan.act == "off_task" and plan.anchor_turn is not None:
        return f"responda ao comentário de {to or 'um colega'} sobre {description(plan.topic, task)} (curto, continue a conversa)"
    return None


# ------------------------------------------------------------------ 0.5.3: dyads, leads (sim/realism.py)
# plural address, matched on accent-folded text; "a gente" / "pra gente" / "da gente" (= we / us / our) and
# "muita gente" / "todo mundo" (people in general) are not an address
PLURAL_ADDRESS_RE = re.compile(r"\b(?:galera|pessoal|vcs|voces|alguem|ninguem|quem mais|gnte)\b"
                               r"|(?<!\ba )(?<!\bpra )(?<!\bda )(?<!\bde )(?<!\bmuita )(?<!\bessa )\bgente\b")
DYAD_REWRITES = [
    (r"\b(oi|oie|eai|e ai|iae|boa noite|bom dia|boa tarde|salve|opa)\s+(?:gente|galera|pessoal)\b", r"\1 {name}"),
    (r"^(?:gente|galera|pessoal|gnte)[,!]?\s+(?=\S)", "{name} "), (r"[,\s]+(?:gente|galera|pessoal)\s*(?=[?!.]*$)", ""),
    (r"^(?:gente|galera|gnte)\s*\?+$", "{name}?"),
    (r"\bv(?:c|oc[eê])s concordam\b", "vc concorda"), (r"\bv(?:c|oc[eê])s acham\b", "vc acha"),
    (r"\bv(?:c|oc[eê])s sabem\b", "vc sabe"), (r"\bv(?:c|oc[eê])s viram\b", "vc viu"),
    (r"\bv(?:c|oc[eê])s torcem\b", "vc torce"), (r"\bv(?:c|oc[eê])s v[aã]o\b", "vc vai"),
    (r"\bcad[eê] v(?:c|oc[eê])s\b", "{name}, ta por ai"), (r"\bv(?:c|oc[eê])s (?:t[aã]o|est[aã]o) a[ií]\b", "vc ta por ai"),
    (r"\bv(?:c|oc[eê])s (?:t[aã]o|est[aã]o)\b", "vc ta"), (r"\bv(?:c|oc[eê])s colocaram\b", "vc colocou"),
    (r"\bv(?:c|oc[eê])s entenderam\b", "vc entendeu"), (r"\bv(?:c|oc[eê])s querem\b", "vc quer"),
    (r"\bv(?:c|oc[eê])s fizeram\b", "vc fez"), (r"\bvcs\b", "vc"), (r"\bvoc[eê]s\b", "você"),
    (r"\b(?:entao )?ningu[eé]m sabe\b", "vc nao sabe"), (r"\bningu[eé]m\b", "vc nao"),
    (r"^(oi\?*\s*)algu[eé]m\b", r"\1{name}"), (r"^algu[eé]m\s*\?", "{name}?"),
    (r"\balgu[eé]m sabe\b", "vc sabe"), (r"\balgu[eé]m mais\b", "vc tbm"), (r"\balgu[eé]m (?:jogando|vendo)\b", "vc ta jogando"),
    (r"\balgu[eé]m\b", "vc"), (r"\bquem mais\b", "vc tbm"), (r"\bquem quer\b", "vc quer"),
    (r"\b(?:gente|galera|pessoal)\b(?!\s*$)", "{name}"),
]
LEAD_WORDS = ("pera", "galera", "olha", "mano", "tipo", "entao", "gente", "ei", "gnte", "pera ai", "olha so")
DYAD_NOTE = ("SÓ VOCÊS DOIS: no chat estão só você e {name}: fale com {name} no singular (vc, {name}), nunca "
             "'gente', 'galera', 'vcs' ou 'alguém'.")
# 0.5.3: concessions quote their reason; a holder who never gave in okays the answer with a hedge
CONCEDE_REASON_IF = ["ah, {r}, faz sentido", "hmm verdade, {r}... entendi", "ah ta, {r}, faz sentido mesmo"]
CONCEDE_REASON = ["ah, {r}? faz sentido entao", "hmm é, {r}, nao tinha pensado nisso", "ah ta, {r}, entendi agora",
                  "verdade, {r}, vc ta certo"]
REASON_LEAD_RE = re.compile(
    r"^(?:pera|hmm|n[aã]o|n|nada a ver|acho (?:que|q) n[aã]o|acho (?:que|q) n|mas|ent[aã]o|tipo|ah|ok|olha|[eé]|isso|vdd|ue|"
    r"[eé] como \w+ falou|como \w+ disse|\w+ j[aá] falou|isso,? o que \w+ disse|no meu diz que|aqui diz que|"
    r"diz que|li que|aparentemente|pelo que eu li|no material fala que|acho que tem a ver com isso:?|"
    r"aqui t[aá] que|aqui fala que|no meu t[aá] que|no meu fala que|no material diz que|o material diz que|"
    r"eu acho que|acho que|acho q|pq|porque)[,:]?\s+", re.I)
REASON_IF_RE = re.compile(r"^(?:se|quando)\b", re.I)
REASON_CUTS = (r"\s+e (?:nao|n) (?:para|e|fica)\b",)
CLOSING_CONCEDE_TAILS = [", pode mandar", ", manda assim", ". por mim ok"]
CLOSING_HEDGE_OK = ["ta, pode ser", "pode mandar", "blz, manda", "ok", "pode ser", "ta bom", "manda entao"]
CLOSING_HEDGE_BUT = ["mas eu ainda acho que nao é bem assim", "mas eu continuo meio na duvida", "mas fico com um pé atras",
                     "mas pra mim ainda nao fechou tudo", "mas ainda tenho minhas duvidas", "mas nao to 100% convencido",
                     "mas eu ainda acho meio estranho", "mas pra mim tinha outra coisa ai"]
# 0.5.3: credits to a classmate (sim/content.fix_peer_attribution), on accent-folded text
CREDIT_PEER_RE = [r"\b(?:[eé],?\s+)?(?:como|igual|que|o que|isso que|oq)\s+(?:a |o )?(?P<name>[a-z]+)\s+(?:disse|falou|tinha falado|tinha dito)\b,?",
                  r"\b(?P<name>[a-z]+)\s+j[aá]\s+(?:disse|falou)(?:\s+isso)?\b,?",
                  r"\b(?:acho que\s+)?(?P<name>[a-z]+)\s+(?:ta|tá|esta|está)\s+cert[oa](?:\s+nessa)?\b,?"]
STEP_MOVE_LINES = ["bora pra {n}?", "essa ja foi, proxima", "fechou essa, bora pra proxima", "agora a {n} ne",
                   "proxima entao", "beleza, partiu {n}", "ok essa foi, vamo pra {n}"]
OFFTASK_PREFIXES = ["mano ", "nossa ", "aff ", "serio, ", "cara ", "slk ", "ai ai "]
OFFTASK_SUFFIXES = [" kkk", " affs", " socorro", " mds", " pfv", " kkkkk"]
INTERFACE_PREFIXES = ["pera ", "ue ", "ei ", "hmm "]
INTERFACE_SUFFIXES = [" kkk", " ??", " socorro"]
HEDGE_FOLLOWUPS = ["acho", "eu acho pelo menos", "pelo menos é oq eu acho", "se eu entendi direito",
                   "mas posso ta errado", "acho q sim", "algo assim", "mais ou menos isso", "se nao me engano",
                   "pelo menos no meu ta assim"]
