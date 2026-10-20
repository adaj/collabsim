"""Trait presets and overrides for simulated students.

Traits are sampled per student from the pilot-calibrated priors (sim/group.sample_traits); a preset
then MULTIPLIES some of them for the whole group (applied after sampling, so the group's knowledge,
cards and misconceptions are identical to the same seed without a preset). `overrides` set absolute
trait values for every student, `per_student` for one student by index (0-based):

    {"preset": "chatty"}
    {"preset": "distracted", "overrides": {"typo_rate": 0.0}, "per_student": {"0": {"talkativeness": 2.0}}}
"""
from __future__ import annotations

from ..models.student import StudentState, Traits

PRESETS: dict[str, dict] = {
    "default": {"description": "pilot-calibrated priors, unchanged", "mult": {}},
    "chatty": {"description": "more talkative, slightly longer messages",
               "mult": {"talkativeness": 1.6, "verbosity": 1.2}},
    "quiet": {"description": "less talkative, shorter messages, longer initial lurking",
              "mult": {"talkativeness": 0.6, "verbosity": 0.8, "lurk_until_min": 1.5}},
    "focused": {"description": "little off-task talk or interface confusion",
                "mult": {"off_task": 0.35, "interface_confusion": 0.5}},
    "distracted": {"description": "more off-task talk and interface confusion",
                   "mult": {"off_task": 2.0, "interface_confusion": 1.6}},
    "deferent": {"description": "defer more to others", "mult": {"deference_peer": 1.4}},
    "stubborn": {"description": "defer less to others (misconceptions are harder to flip)",
                 "mult": {"deference_peer": 0.6}},
}
BOUNDED = {"burstiness", "off_task", "deference_peer", "interface_confusion", "typo_rate"}   # probabilities: clipped to [0, 1]
TRAIT_NAMES = [k for k in Traits.model_fields]


class TraitError(ValueError):
    pass


def _set(s: StudentState, name: str, value) -> None:
    if name not in TRAIT_NAMES:
        raise TraitError(f"unknown trait {name!r} (traits: {TRAIT_NAMES})")
    if name == "lowercase":
        s.traits.lowercase = bool(value)
        return
    v = float(value)
    if v < 0:
        raise TraitError(f"trait {name} must be >= 0")
    if name in BOUNDED:
        v = min(1.0, v)
    setattr(s.traits, name, v)


def apply_traits(students: list[StudentState], spec: dict | None) -> dict:
    """Apply {"preset", "overrides", "per_student"} in place; returns the normalized spec."""
    spec = dict(spec or {})
    preset = spec.get("preset") or "default"
    if preset not in PRESETS:
        raise TraitError(f"unknown trait preset {preset!r} (presets: {list(PRESETS)})")
    for s in students:
        for name, f in PRESETS[preset]["mult"].items():
            _set(s, name, getattr(s.traits, name) * f)
    for name, v in (spec.get("overrides") or {}).items():
        for s in students:
            _set(s, name, v)
    for idx, ov in (spec.get("per_student") or {}).items():
        try:
            s = students[int(idx)]
        except (ValueError, IndexError):
            raise TraitError(f"per_student index {idx!r} out of range (0..{len(students) - 1})") from None
        for name, v in (ov or {}).items():
            _set(s, name, v)
    return {"preset": preset, "overrides": spec.get("overrides") or {}, "per_student": spec.get("per_student") or {}}


def preset_list() -> list[dict]:
    return [{"id": k, "description": v["description"], "multipliers": v["mult"]} for k, v in PRESETS.items()]
