"""Gold remediation control for exp3v2.

Researcher-authored ideal remediation responses injected as tutor turns.
Tests H1 (tutor ability) and H4 (evaluation rigidity).
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from src.evaluation.simulated_student import (
    StudentProfile,
    build_student_system_prompt,
)
from src.experiments.exp3v2.scenario_bank import (
    Scenario,
    MISCONCEPTION_TYPES,
    DIFFICULTY_LEVELS,
)
from src.experiments.exp3v2.evaluator import evaluate_remediation
from src.experiments.exp3v2.run import _assign_profiles, _check_answer_revealed, _adjust_tokens
from src.experiments.exp3v2.prompts import build_verification_prompt_v2
from src.evaluation.metrics import remediation_success_rate, ar_excluded_rsr
from src.utils.common import save_json, ensure_dir, get_timestamp

logger = logging.getLogger(__name__)


@dataclass
class GoldRemediation:
    """A researcher-authored ideal remediation."""
    scenario_id: str             # matches Scenario.id
    misconception_type: str      # factual | procedural | conceptual
    difficulty: str              # easy | medium | hard
    tutor_responses: list[str]   # 1-2 turns of expert remediation
    explanation: str             # pedagogical rationale

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GoldRemediation":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


def load_gold_bank(path: Path | str) -> list[GoldRemediation]:
    """Load gold remediation bank from JSON."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return [GoldRemediation.from_dict(item) for item in data]


def save_gold_bank(gold_bank: list[GoldRemediation], path: Path | str) -> None:
    """Save gold remediation bank to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [g.to_dict() for g in gold_bank]
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def validate_gold_bank(
    gold_bank: list[GoldRemediation],
    scenario_bank: list[Scenario] | None = None,
) -> dict:
    """Validate gold bank completeness.

    Checks:
    - >= 2 entries per type x difficulty cell
    - All scenario_ids reference valid scenarios (if scenario_bank provided)

    Returns dict with cell_counts, missing_cells, invalid_refs, valid (bool).
    """
    from collections import Counter
    cell_counts = Counter((g.misconception_type, g.difficulty) for g in gold_bank)

    missing_cells = []
    for t in MISCONCEPTION_TYPES:
        for d in DIFFICULTY_LEVELS:
            count = cell_counts.get((t, d), 0)
            if count < 2:
                missing_cells.append(f"({t}, {d}): {count} (need >= 2)")

    invalid_refs = []
    if scenario_bank:
        valid_ids = {s.id for s in scenario_bank}
        for g in gold_bank:
            if g.scenario_id not in valid_ids:
                invalid_refs.append(g.scenario_id)

    return {
        "total": len(gold_bank),
        "cell_counts": {f"{t}_{d}": cell_counts.get((t, d), 0)
                        for t in MISCONCEPTION_TYPES for d in DIFFICULTY_LEVELS},
        "missing_cells": missing_cells,
        "invalid_refs": invalid_refs,
        "valid": len(missing_cells) == 0 and len(invalid_refs) == 0,
    }


def run_gold_dialogue(
    scenario: Scenario,
    gold: GoldRemediation,
    student_model,
    profile: StudentProfile,
    judge_model=None,
    judge_cache: dict | None = None,
) -> dict:
    """Run one gold remediation dialogue.

    Injects researcher-authored tutor responses instead of LLM generation.
    Student model still responds normally.
    """
    student_system = build_student_system_prompt(profile)

    conversation_history: list[dict] = []
    student_responses: list[str] = []

    # Initial student message
    initial_msg = (
        f"I solved this problem: {scenario.problem}\n\n"
        f"My answer is: {scenario.student_answer}"
    )
    conversation_history.append({"role": "user", "content": initial_msg})
    student_responses.append(initial_msg)

    # Inject gold tutor responses with student replies
    for turn_idx, gold_response in enumerate(gold.tutor_responses):
        conversation_history.append({"role": "assistant", "content": gold_response})

        # Student responds (except after last gold turn)
        if turn_idx < len(gold.tutor_responses) - 1:
            student_messages = [{"role": "system", "content": student_system}]
            student_messages.extend(conversation_history)
            try:
                student_resp = student_model.generate_with_history(
                    messages=student_messages,
                    temperature=0.7,
                    max_tokens=_adjust_tokens(student_model, 256),
                )
                student_text = student_resp.text.strip() or "[Student empty response]"
            except Exception as e:
                logger.warning(f"Gold student error: {e}")
                student_text = "[Student error]"
            student_responses.append(student_text)
            conversation_history.append({"role": "user", "content": student_text})

    # Verification
    verification_prompt = build_verification_prompt_v2(
        verification_problem=scenario.verification_problem,
        misconception_type=scenario.misconception_type,
    )
    verification_messages = [
        {"role": "system", "content": student_system},
        *conversation_history,
        {"role": "user", "content": verification_prompt},
    ]
    try:
        verif_resp = student_model.generate_with_history(
            messages=verification_messages,
            temperature=0.3,
            max_tokens=_adjust_tokens(student_model, 384),
        )
        verification_answer = verif_resp.text.strip() or ""
    except Exception as e:
        logger.warning(f"Gold verification error: {e}")
        verification_answer = ""

    # Evaluate
    # [FIX] The student solves verification_problem, so use verification_answer as gold.
    # Previously: scenario.correct_answer (the original answer) -> scored against the wrong gold.
    evaluation = evaluate_remediation(
        student_response=verification_answer,
        correct_answer=scenario.verification_answer,
        problem=scenario.verification_problem,
        misconception=scenario.misconception_label,
        misconception_indicators=scenario.indicators,
        judge_model=judge_model,
        cache=judge_cache,
    )

    answer_revealed = _check_answer_revealed(gold.tutor_responses, scenario.correct_answer)

    return {
        "scenario_id": scenario.id,
        "gold_id": gold.scenario_id,
        "misconception_type": scenario.misconception_type,
        "difficulty": scenario.difficulty,
        "profile_knowledge": profile.knowledge_level,
        "profile_cooperation": profile.cooperation,
        "gold_tutor_responses": gold.tutor_responses,
        "student_responses": student_responses,
        "verification_answer": verification_answer,
        "remediation_success": evaluation["official_correct"],
        "tier1_correct": evaluation["tier1_correct"],
        "tier2_correct": evaluation["tier2_correct"],
        "answer_revealed": answer_revealed,
        "ar_excluded_success": evaluation["official_correct"] and not answer_revealed,
        "is_gold": True,
        "error": None,
    }


def run_gold_experiment(
    gold_bank: list[GoldRemediation],
    scenario_bank: list[Scenario],
    student_model,
    judge_model=None,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict:
    """Run gold remediation experiment.

    For each gold scenario x 6 profiles = total dialogues.

    Returns dict with aggregate results and per-dialogue details.
    """
    if output_dir:
        ensure_dir(output_dir)

    # Build scenario lookup
    scenario_map = {s.id: s for s in scenario_bank}

    judge_cache: dict = {}
    all_results = []

    for g_idx, gold in enumerate(gold_bank):
        scenario = scenario_map.get(gold.scenario_id)
        if scenario is None:
            logger.warning(f"Gold scenario_id {gold.scenario_id} not found in bank, skipping")
            continue

        # 6 profiles for this scenario
        dummy_scenarios = [scenario] * 6
        profiles = _assign_profiles(dummy_scenarios, scenario.misconception_type, seed + g_idx)

        for p_idx, profile in enumerate(profiles):
            try:
                result = run_gold_dialogue(
                    scenario=scenario,
                    gold=gold,
                    student_model=student_model,
                    profile=profile,
                    judge_model=judge_model,
                    judge_cache=judge_cache,
                )
            except Exception as e:
                logger.error(f"Gold dialogue error {gold.scenario_id}/{p_idx}: {e}")
                result = {
                    "scenario_id": scenario.id,
                    "gold_id": gold.scenario_id,
                    "misconception_type": scenario.misconception_type,
                    "difficulty": scenario.difficulty,
                    "profile_knowledge": profile.knowledge_level,
                    "profile_cooperation": profile.cooperation,
                    "remediation_success": False,
                    "answer_revealed": False,
                    "ar_excluded_success": False,
                    "is_gold": True,
                    "error": str(e),
                }
            all_results.append(result)

        logger.info(
            f"Gold {g_idx + 1}/{len(gold_bank)}: {gold.scenario_id} "
            f"({gold.misconception_type}/{gold.difficulty})"
        )

    # Aggregate
    pre_correct = [False] * len(all_results)
    post_correct = [r["remediation_success"] for r in all_results]
    ar_flags = [r["answer_revealed"] for r in all_results]

    rsr = remediation_success_rate(pre_correct, post_correct)
    ar_excl = ar_excluded_rsr(pre_correct, post_correct, ar_flags)

    # By type x difficulty
    by_cell = {}
    for t in MISCONCEPTION_TYPES:
        for d in DIFFICULTY_LEVELS:
            cell_results = [r for r in all_results
                           if r["misconception_type"] == t and r["difficulty"] == d]
            if cell_results:
                cell_pre = [False] * len(cell_results)
                cell_post = [r["remediation_success"] for r in cell_results]
                cell_ar = [r["answer_revealed"] for r in cell_results]
                by_cell[f"{t}_{d}"] = {
                    "n": len(cell_results),
                    "rsr": remediation_success_rate(cell_pre, cell_post),
                    "ar_excluded_rsr": ar_excluded_rsr(cell_pre, cell_post, cell_ar),
                }

    summary = {
        "total_dialogues": len(all_results),
        "overall_rsr": rsr,
        "ar_excluded_rsr": ar_excl,
        "by_cell": by_cell,
        "timestamp": get_timestamp(),
    }

    output = {"summary": summary, "results": all_results}

    if output_dir:
        save_json(output, output_dir / "gold_results.json")
        logger.info(f"Gold results saved: RSR={rsr:.3f}, AR-excl={ar_excl:.3f}")

    return output
