"""Scenario bank data structures and I/O for exp3v2."""

from __future__ import annotations

import json
import logging
import random
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Scenario:
    """A single remediation scenario."""
    id: str                          # e.g. "factual_0001"
    problem: str                     # math problem text
    student_answer: str              # incorrect student answer
    correct_answer: str              # gold correct answer
    misconception_label: str         # e.g. "All square roots are rational"
    misconception_type: str          # factual | procedural | conceptual
    difficulty: str                  # easy | medium | hard
    indicators: list[str] = field(default_factory=list)  # misconception indicator strings
    verification_problem: str = ""   # pre-authored verification problem for the student
    verification_answer: str = ""    # gold correct answer for the verification problem
    topic: str = ""                  # e.g. "algebra", "geometry"
    source: str = ""                 # "mathdial" | "bea2025" | "manual" | "gpt5_variation"
    source_id: str = ""              # original dataset instance ID (if extracted)
    variation_of: str = ""           # parent scenario ID (if GPT-5 generated)
    _review_status: str = "pending"  # pending | accepted | rejected

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        """Create from dict, tolerating extra keys."""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


MISCONCEPTION_TYPES = ["factual", "procedural", "conceptual"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


def load_bank(path: Path | str) -> list[Scenario]:
    """Load scenario bank from JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    scenarios = []
    for item in data:
        scenarios.append(Scenario.from_dict(item))
    return scenarios


def save_bank(scenarios: list[Scenario], path: Path | str) -> None:
    """Save scenario bank to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [s.to_dict() for s in scenarios]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_bank(scenarios: list[Scenario]) -> dict:
    """Validate scenario bank balance and coverage.

    Returns dict with:
        - total: int
        - by_type: dict[str, int]
        - by_difficulty: dict[str, int]
        - by_type_difficulty: dict[str, dict[str, int]]
        - imbalances: list[str] — warning messages for any imbalances
        - valid: bool — True if no critical issues
    """
    by_type: dict[str, int] = {t: 0 for t in MISCONCEPTION_TYPES}
    by_difficulty: dict[str, int] = {d: 0 for d in DIFFICULTY_LEVELS}
    by_type_diff: dict[str, dict[str, int]] = {
        t: {d: 0 for d in DIFFICULTY_LEVELS} for t in MISCONCEPTION_TYPES
    }

    for s in scenarios:
        if s.misconception_type in by_type:
            by_type[s.misconception_type] += 1
        if s.difficulty in by_difficulty:
            by_difficulty[s.difficulty] += 1
        if s.misconception_type in by_type_diff and s.difficulty in by_type_diff[s.misconception_type]:
            by_type_diff[s.misconception_type][s.difficulty] += 1

    imbalances = []
    valid = True

    # Check minimum per type
    for t, count in by_type.items():
        if count < 30:
            imbalances.append(f"Type '{t}' has only {count} scenarios (target: 30+)")
        if count == 0:
            valid = False

    # Check minimum per type x difficulty
    for t in MISCONCEPTION_TYPES:
        for d in DIFFICULTY_LEVELS:
            count = by_type_diff[t][d]
            if count < 10:
                imbalances.append(
                    f"Cell ({t}, {d}) has only {count} scenarios (target: 10+)"
                )

    # Check difficulty balance within each type
    for t in MISCONCEPTION_TYPES:
        counts = [by_type_diff[t][d] for d in DIFFICULTY_LEVELS]
        if counts and max(counts) > 0:
            ratio = min(counts) / max(counts) if max(counts) > 0 else 0
            if ratio < 0.5:
                imbalances.append(
                    f"Type '{t}' has imbalanced difficulties: "
                    f"{dict(zip(DIFFICULTY_LEVELS, counts))}"
                )

    return {
        "total": len(scenarios),
        "by_type": by_type,
        "by_difficulty": by_difficulty,
        "by_type_difficulty": by_type_diff,
        "imbalances": imbalances,
        "valid": valid,
    }


def get_scenarios_for_cell(
    bank: list[Scenario],
    misconception_type: str,
    difficulty: str | None = None,
    n: int = 10,
    seed: int = 42,
) -> list[Scenario]:
    """Sample n scenarios for a given cell from the bank.

    Args:
        bank: Full scenario bank.
        misconception_type: Filter by this type.
        difficulty: Filter by difficulty (None = all difficulties).
        n: Number of scenarios to sample.
        seed: Random seed for reproducibility.

    Returns:
        List of n Scenario objects.
    """
    # Filter
    matching = [
        s for s in bank
        if s.misconception_type == misconception_type
        and (difficulty is None or s.difficulty == difficulty)
        and s._review_status == "accepted"
    ]

    if not matching:
        raise ValueError(
            f"No accepted scenarios for type={misconception_type}, "
            f"difficulty={difficulty}"
        )

    rng = random.Random(seed)

    if n <= len(matching):
        # Sample without replacement
        return rng.sample(matching, n)
    else:
        # Sample with replacement + warning
        warnings.warn(
            f"Requested {n} scenarios but only {len(matching)} available for "
            f"type={misconception_type}, difficulty={difficulty}. "
            f"Sampling with replacement.",
            stacklevel=2,
        )
        return [rng.choice(matching) for _ in range(n)]
