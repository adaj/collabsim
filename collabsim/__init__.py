"""CollabSim: a state-based small-group student chat simulator, as a Python package.

Code decides who speaks, when and what (hidden student state: knowledge, misconceptions, affect;
pilot-calibrated turn-taking); Claude Haiku only words the turns. External participants (people or
bots) can join a live chat and write in it.

    from collabsim import Simulator
    sim = Simulator(api_key=...)                       # or Simulator(backend="template")
    s = sim.new_session(n_students=4, task="sci-enzyme-01", language="pt-BR", participants=["Ana"])
    s.start(); s.poll(after=0); s.say("oi gente", sender="Ana"); s.close()

See README.md for the full API.
"""
from .errors import BackendUnavailable, CapacityError, CollabSimError
from .simulator import Session, Simulator
from .types import ParticipantSpec, SessionConfig, SimEvent, TaskSpec

__version__ = "0.5.3"
__all__ = ["Simulator", "Session", "SessionConfig", "ParticipantSpec", "SimEvent", "TaskSpec", "CollabSimError",
           "BackendUnavailable", "CapacityError", "__version__"]
