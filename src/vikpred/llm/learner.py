"""The user's football-knowledge profile, so the assistant's answers grow with them.

The user picks a level (Beginner / Fan / Analyst); the assistant records concepts
they've shown they understand, concepts it has introduced, and their interests,
through the `update_learner_profile` tool. The profile is a small JSON file on
the user's machine (under data/, which Git ignores).
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from vikpred.config import DATA_DIR

PROFILE_PATH = DATA_DIR / "learner_profile.json"

LEVELS = {
    "beginner": ("Beginner", "New to football analytics. Use plain language, avoid jargon, and explain any "
                             "stat in everyday terms with a quick example."),
    "fan": ("Fan", "Knows the game well and follows the season. Use normal football vocabulary; briefly "
                   "explain advanced stats (EPA, success rate, SHAP) the first time they come up."),
    "analyst": ("Analyst", "Comfortable with advanced metrics and modeling. Be concise and technical: EPA, "
                           "success rate, log-odds impacts, calibration, and Brier scores without hand-holding."),
}
DEFAULT_LEVEL = "fan"
LIMITS = {"known": 60, "learning": 20, "interests": 20, "notes": 10}

TOOL = {
    "name": "update_learner_profile",
    "description": (
        "Record what the user has shown they understand or care about, so future answers match their "
        "football knowledge. Call it only on clear evidence: the user correctly uses or explains a concept "
        "(add to known), you just explained a concept to them for the first time (add to learning), or "
        "they show sustained interest in a topic (add to interests). Use short lowercase concept names "
        "like 'epa', 'success rate', 'shap values', 'cover 2', 'brier score'. Don't mention this tool to "
        "the user."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "known": {"type": "array", "items": {"type": "string"},
                      "description": "Concepts the user has demonstrated they understand."},
            "learning": {"type": "array", "items": {"type": "string"},
                         "description": "Concepts just introduced or explained to the user."},
            "interests": {"type": "array", "items": {"type": "string"},
                          "description": "Topics the user keeps asking about."},
            "note": {"type": "string", "description": "One short observation about how they like answers."},
        },
        "additionalProperties": False,
    },
}


def _norm(items) -> list[str]:
    return [s.strip().lower() for s in items or [] if isinstance(s, str) and s.strip()]


def _merge(existing: list[str], new: list[str], limit: int) -> list[str]:
    """Append new items (deduped, newest last) and keep only the most recent `limit`."""
    out = [x for x in existing if x not in new] + list(dict.fromkeys(new))
    return out[-limit:]


@dataclass
class LearnerProfile:
    level: str = DEFAULT_LEVEL
    known: list[str] = field(default_factory=list)
    learning: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    updated_at: str = ""

    def apply(self, update: dict) -> list[str]:
        """Merge a tool update. Returns human-readable changes."""
        changes = []
        known = _norm(update.get("known"))
        if known:
            self.known = _merge(self.known, known, LIMITS["known"])
            # Mastered concepts graduate out of "learning".
            self.learning = [c for c in self.learning if c not in known]
            changes.append(f"knows: {', '.join(known)}")
        learning = [c for c in _norm(update.get("learning")) if c not in self.known]
        if learning:
            self.learning = _merge(self.learning, learning, LIMITS["learning"])
            changes.append(f"learning: {', '.join(learning)}")
        interests = _norm(update.get("interests"))
        if interests:
            self.interests = _merge(self.interests, interests, LIMITS["interests"])
            changes.append(f"interested in: {', '.join(interests)}")
        note = (update.get("note") or "").strip()
        if note:
            self.notes = _merge(self.notes, [note], LIMITS["notes"])
            changes.append("note added")
        return changes

    def context_message(self) -> str:
        """The profile as the assistant sees it. Only changes when the profile does."""
        name, description = LEVELS.get(self.level, LEVELS[DEFAULT_LEVEL])
        lines = [
            "Learner profile, set by the app (not written by the user):",
            f"- Level: {name}. {description}",
            f"- Concepts they know well: {', '.join(self.known) or 'none recorded yet'}",
            f"- Concepts introduced, still learning: {', '.join(self.learning) or 'none'}",
            f"- Interests: {', '.join(self.interests) or 'none recorded yet'}",
        ]
        if self.notes:
            lines.append(f"- Notes: {' | '.join(self.notes)}")
        lines.append("Pitch your answers to this profile. Briefly explain any concept that isn't in their "
                     "known list the first time it comes up; use known concepts freely without explaining them.")
        return "\n".join(lines)


class LearnerStore:
    """Loads and saves the profile as JSON."""

    def __init__(self, path: Path = PROFILE_PATH):
        self.path = Path(path)
        self.profile = self._load()

    def _load(self) -> LearnerProfile:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return LearnerProfile(**{k: v for k, v in data.items() if k in LearnerProfile.__dataclass_fields__})
        except (FileNotFoundError, json.JSONDecodeError, TypeError):
            return LearnerProfile()

    def save(self) -> None:
        self.profile.updated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(asdict(self.profile), indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def set_level(self, level: str) -> None:
        if level not in LEVELS:
            raise ValueError(f"Unknown level '{level}'. Choose from {list(LEVELS)}")
        if level != self.profile.level:
            self.profile.level = level
            self.save()

    def handle_tool(self, update: dict) -> str:
        """Run the update_learner_profile tool; returns the tool result text."""
        changes = self.profile.apply(update)
        if not changes:
            return "No changes recorded."
        self.save()
        return "Profile updated (" + "; ".join(changes) + ")."

    def reset(self) -> None:
        level = self.profile.level
        self.profile = LearnerProfile(level=level)
        self.save()
