"""Off-task topic bank: each student gets a few personal topics at group
generation; the planner picks one per off-task turn with decay on topics the group has already
used, so chatter persists (P(off|off) is decided upstream, unchanged) without verbatim re-use.

Each topic: a short description (for the LLM realizer) and several chat-register template lines.
"""
from __future__ import annotations

TOPICS: dict[str, tuple[str, list[str]]] = {
    "snack": ("food / being hungry / a snack", [
        "im so hungry rn", "anyone else starving", "i brought chips but cant eat them here lol",
        "what are u guys having for lunch", "i need food asap"]),
    "game_last_night": ("a sports match from last night", [
        "did anyone see the game last night", "that match yesterday was crazy",
        "bro the ref yesterday was so bad", "i stayed up watching the game lol",
        "we lost again yesterday smh"]),
    "videogame": ("a video game they play", [
        "anyone playing the new season", "i got to diamond yesterday", "who wants to play after class",
        "my controller broke lol", "that update ruined the game"]),
    "music": ("music / a song / headphones", [
        "i have a song stuck in my head", "who else listens to music while doing this",
        "my headphones died", "that new album is so good", "someone put music on lol"]),
    "weekend": ("plans for the weekend", [
        "what are u doing this weekend", "i cant wait for saturday", "weekend cant come fast enough",
        "anyone going to the party", "im just gonna sleep all weekend"]),
    "tired": ("being tired / sleepy", [
        "im so sleepy", "i slept like 4 hours", "can barely keep my eyes open", "i need a nap fr",
        "why is this class so early"]),
    "bored": ("being bored / the class being long", [
        "this is taking forever", "who else is bored", "how much time is left",
        "is it almost over", "this class feels so long today"]),
    "internet": ("slow internet / the computer", [
        "my internet is so slow", "my wifi keeps dropping", "the computer is freezing",
        "my screen went black for a sec", "is it lagging for u guys too"]),
    "phone": ("their phone / battery / charger", [
        "my phone is at 3 percent", "i forgot my charger", "who has a charger",
        "my phone keeps buzzing lol", "teacher took my phone last week"]),
    "pet": ("a pet at home", [
        "my dog ate my homework for real", "my cat is so annoying lately", "i want a puppy",
        "my dog was barking all night", "we got a new kitten"]),
    "weather": ("the weather", [
        "its so hot in here", "is it raining outside", "i hate this cold", "its so sunny today",
        "the ac in here is too strong"]),
    "homework_other": ("homework from another class", [
        "did anyone do the math homework", "we have a test tomorrow right", "i didnt study for history",
        "is the english essay due today", "i forgot the other homework"]),
    "teacher": ("the teacher / the classroom", [
        "the teacher is looking at us lol", "is the teacher coming back", "why is it so loud in here",
        "someone is yelling next door", "the teacher is on her phone lol"]),
    "series": ("a tv series or video", [
        "did u watch the new episode", "no spoilers pls", "that show is so good",
        "i binged the whole season", "the ending was so bad"]),
    "laugh": ("laughing at something silly", [
        "lmaooo", "hahaha stop", "im crying", "why is that so funny", "hahaha"]),
    "home": ("going home / after class", [
        "can we go home", "what time does class end", "i wanna go home", "is the bus late today",
        "who is walking home after"]),
    "birthday": ("a birthday (their own or a family member's, never a classmate's in the chat)", [
        "its my cousins birthday today", "my sisters birthday is on saturday", "theres cake at home lol",
        "we should do a party", "my birthday is next week"]),
    "football_team": ("their favourite team", [
        "my team is gonna win the cup", "who do u guys support", "we play tomorrow",
        "that transfer was crazy", "best team ever lol"]),
    "recess": ("break / recess", [
        "is it recess yet", "cant wait for break", "who is going to the canteen",
        "we should play at break", "how long is break today"]),
    "clothes": ("shoes / clothes", [
        "i got new shoes", "my shoes are soaked", "i love that hoodie", "its uniform day ugh",
        "i forgot my jacket"]),
    "social_media": ("a social media post / trend", [
        "did u see that video", "that trend is so cringe", "i posted something lol",
        "everyone is doing that dance", "check the group chat later"]),
    "sibling": ("a sibling", [
        "my brother is so annoying", "my sister took my headphones", "my little brother broke my phone",
        "i have to pick up my sister later", "my brother wont stop texting me"]),
    "random_omg": ("a random exclamation", [
        "omg", "wait what", "bruh", "no way", "nice"]),
}

TOPIC_IDS = list(TOPICS)
TOPICS_PER_STUDENT = 3
TOPIC_DECAY = 0.3        # weight multiplier per earlier group use of a topic
JOIN_PEER_TOPIC = 1.5    # weight of continuing a peer's recent off-task topic
FRESH_TOPIC_MASS = 0.4   # total weight spread over bank topics not assigned to the student
# realism layer (0.4.1): topics of one family count as the same topic for in-session dedupe
# ("the game last night" and "my team" are one subject); topics that presuppose a physical classroom
# are not used in an online chat
TOPIC_FAMILY = {"game_last_night": "sports", "football_team": "sports", "series": "screen",
                "social_media": "screen", "bored": "bored", "home": "bored", "laugh": "atom", "random_omg": "atom"}
TOPICS_NOT_ONLINE = ("teacher", "recess")


def description(topic: str | None, lang_of=None) -> str:
    """`lang_of`: a Task or language code (None = English); topic ids are the same in every pack."""
    from .. import lang
    topics = lang.attr(lang_of, "TOPICS", TOPICS)
    if topic is None or topic not in topics:
        return lang.attr(lang_of, "TOPIC_DEFAULT_DESC", "food, sports, being bored, the class, the internet")
    return topics[topic][0]


def lines(topic: str | None, lang_of=None) -> list[str]:
    from .. import lang
    topics = lang.attr(lang_of, "TOPICS", TOPICS)
    return topics[topic][1] if topic in topics else [l for _, ls in topics.values() for l in ls]
