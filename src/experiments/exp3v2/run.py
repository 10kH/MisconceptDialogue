"""Experiment 3v2: Redesigned Remediation Strategy Effectiveness.

Key changes from exp3:
- Scenario bank loaded from JSON (not hardcoded)
- Random profile assignment (tiled-shuffle, not i%6)
- LLM-as-judge evaluation (two-tier)
- Difficulty-stratified cells
- Gold remediation mode
- Word-boundary AR detection

Usage:
    python -m src.experiments.exp3v2.run --model qwen-2.5-7b --bank data/exp3v2_scenario_bank.json --seed 42
    python -m src.experiments.exp3v2.run --model phi-4 --bank data/exp3v2_scenario_bank.json --gold --gold_bank data/exp3v2_gold_bank.json
"""

from __future__ import annotations

import argparse
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.evaluation.metrics import (
    remediation_success_rate,
    ar_excluded_rsr,
    rsr_with_ci,
)
from src.evaluation.simulated_student import (
    StudentProfile,
    build_student_system_prompt,
)
from src.utils.common import (
    set_seed,
    save_json,
    ensure_dir,
    get_timestamp,
    RESULTS_DIR,
)
from src.experiments.exp3v2.prompts import (
    STRATEGIES,
    MISCONCEPTION_TYPES,
    build_remediation_prompt,
    build_tutor_system_prompt,
    build_verification_prompt_v2,
)
from src.experiments.exp3v2.scenario_bank import (
    Scenario,
    load_bank,
    get_scenarios_for_cell,
    DIFFICULTY_LEVELS,
)
from src.experiments.exp3v2.evaluator import evaluate_remediation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token adjustment (replicated from exp3 for independence)
# ---------------------------------------------------------------------------

_REASONING_PREFIXES = ("gpt-5", "o1", "o3")

def _adjust_tokens(model, base: int) -> int:
    """Increase token budget for reasoning models that consume tokens on thinking."""
    model_name = getattr(model, "model_name", "")
    if any(model_name.startswith(p) for p in _REASONING_PREFIXES):
        return max(base * 4, 4096)
    # Check HF reasoning models
    _REASONING_MODELS = {"Qwen/QwQ-32B"}
    if model_name in _REASONING_MODELS or any(m in model_name for m in _REASONING_MODELS):
        return max(base * 4, 4096)
    return base


# ---------------------------------------------------------------------------
# Profile assignment (KEY FIX: tiled-shuffle instead of i%6)
# ---------------------------------------------------------------------------

def _assign_profiles(
    scenarios: list[Scenario],
    misconception_type: str,
    seed: int,
) -> list[StudentProfile]:
    """Balanced random assignment using tiled-shuffle.

    Guarantees each of 6 profiles appears between floor(N/6) and ceil(N/6) times.
    This fixes the deterministic i%6 mapping that caused the ceiling effect in exp3.

    Args:
        scenarios: List of scenarios to assign profiles to.
        misconception_type: The cell's misconception type (passed to StudentProfile).
        seed: Random seed for reproducibility.

    Returns:
        List of StudentProfile, one per scenario.
    """
    assert misconception_type in MISCONCEPTION_TYPES, (
        f"Invalid misconception_type: {misconception_type}"
    )

    rng = random.Random(seed)
    profiles = [
        StudentProfile(k, c, misconception_type, "", "")
        for k in ["low", "medium", "high"]
        for c in ["cooperative", "resistant"]
    ]

    n = len(scenarios)
    n_tiles = (n + len(profiles) - 1) // len(profiles)
    assignments = []
    for _ in range(n_tiles):
        tile = list(profiles)
        rng.shuffle(tile)
        assignments.extend(tile)

    return assignments[:n]


# ---------------------------------------------------------------------------
# Answer-revealed detection (improved with word boundaries)
# ---------------------------------------------------------------------------

def _check_answer_revealed(
    tutor_responses: list[str],
    correct_answer: str,
) -> bool:
    """Check if tutor revealed the answer using word-boundary matching.

    Improved over exp3's raw substring matching to avoid false positives
    like "3" matching "13".
    """
    if not correct_answer:
        return False

    all_tutor_text = " ".join(tutor_responses).lower()
    answer_lower = correct_answer.strip().lower()
    pattern = r'\b' + re.escape(answer_lower) + r'\b'
    return bool(re.search(pattern, all_tutor_text))


# ---------------------------------------------------------------------------
# Core dialogue runner
# ---------------------------------------------------------------------------

def run_remediation_dialogue(
    scenario: Scenario,
    strategy: str,
    tutor_model,
    student_model,
    profile: StudentProfile,
    judge_model=None,
    judge_cache: dict | None = None,
    min_turns: int = 3,
    max_turns: int = 5,
) -> dict:
    """Run one full multi-turn remediation dialogue.

    Args:
        scenario: The scenario to run.
        strategy: Teaching strategy to use.
        tutor_model: Model acting as tutor.
        student_model: Model acting as simulated student.
        profile: Student profile for this dialogue.
        judge_model: Optional OpenAIModel for Tier 2 evaluation.
        judge_cache: Optional cache dict for judge evaluations.
        min_turns: Minimum dialogue turns.
        max_turns: Maximum dialogue turns.

    Returns:
        dict with full dialogue data and evaluation results.
    """
    student_system = build_student_system_prompt(profile)
    tutor_system = build_tutor_system_prompt()

    # Suppress meta-reasoning for models that leak analysis into output
    # Meta-reasoning suppression removed (no longer needed for 5-model set)

    conversation_history: list[dict] = []
    tutor_responses: list[str] = []
    student_responses: list[str] = []

    # Initial student message
    initial_student_msg = (
        f"I solved this problem: {scenario.problem}\n\n"
        f"My answer is: {scenario.student_answer}"
    )
    conversation_history.append({"role": "user", "content": initial_student_msg})
    student_responses.append(initial_student_msg)

    for turn in range(max_turns):
        # --- Tutor turn ---
        tutor_prompt = build_remediation_prompt(
            problem=scenario.problem,
            student_answer=scenario.student_answer,
            misconception_label=scenario.misconception_label,
            misconception_type=scenario.misconception_type,
            strategy=strategy,
            conversation_history=conversation_history,
        )

        try:
            tutor_resp = tutor_model.generate(
                prompt=tutor_prompt,
                system_prompt=tutor_system,
                temperature=0.2,
                max_tokens=_adjust_tokens(tutor_model, 2048),
            )
            tutor_text = tutor_resp.text.strip()
        except Exception as e:
            logger.warning(f"Tutor error turn {turn}: {e}")
            tutor_text = "[Tutor error]"

        tutor_responses.append(tutor_text)
        conversation_history.append({"role": "assistant", "content": tutor_text})

        # --- Student turn (not on last turn) ---
        if turn < max_turns - 1:
            student_messages = [{"role": "system", "content": student_system}]
            student_messages.extend(conversation_history)

            try:
                student_resp = student_model.generate_with_history(
                    messages=student_messages,
                    temperature=0.7,
                    max_tokens=_adjust_tokens(student_model, 2048),
                )
                student_text = student_resp.text.strip()
                if not student_text:
                    student_text = "[Student empty response]"
            except Exception as e:
                logger.warning(f"Student error turn {turn}: {e}")
                student_text = "[Student error]"

            student_responses.append(student_text)
            conversation_history.append({"role": "user", "content": student_text})

    # --- Verification ---
    verification_prompt_text = build_verification_prompt_v2(
        verification_problem=scenario.verification_problem,
        misconception_type=scenario.misconception_type,
    )
    verification_messages = [
        {"role": "system", "content": student_system},
        *conversation_history,
        {"role": "user", "content": verification_prompt_text},
    ]
    try:
        verif_resp = student_model.generate_with_history(
            messages=verification_messages,
            temperature=0.3,
            max_tokens=_adjust_tokens(student_model, 2048),
        )
        verification_answer = verif_resp.text.strip()
        if not verification_answer:
            verification_answer = ""
    except Exception as e:
        logger.warning(f"Verification error: {e}")
        verification_answer = ""

    # --- Evaluate with two-tier system ---
    # [FIX] 학생은 verification_problem을 풀고 있으므로, 평가 시에도
    # verification_answer/verification_problem을 gold로 사용해야 한다.
    # 이전에는 원문 문제의 정답(scenario.correct_answer)을 넘겨
    # 102/106 시나리오에서 잘못된 gold로 채점하는 버그가 있었다.
    evaluation = evaluate_remediation(
        student_response=verification_answer,
        correct_answer=scenario.verification_answer,
        problem=scenario.verification_problem,
        misconception=scenario.misconception_label,
        misconception_indicators=scenario.indicators,
        judge_model=judge_model,
        cache=judge_cache,
    )

    # --- Check answer revealed ---
    answer_revealed = _check_answer_revealed(tutor_responses, scenario.correct_answer)

    return {
        "scenario_id": scenario.id,
        "strategy": strategy,
        "misconception_type": scenario.misconception_type,
        "difficulty": scenario.difficulty,
        "scenario_source": scenario.source,
        "profile_knowledge": profile.knowledge_level,
        "profile_cooperation": profile.cooperation,
        "num_turns": len(tutor_responses),
        "tutor_responses": tutor_responses,
        "student_responses": student_responses,
        "verification_answer": verification_answer,
        "remediation_success": evaluation["official_correct"],
        "tier1_correct": evaluation["tier1_correct"],
        "tier2_correct": evaluation["tier2_correct"],
        "judge_reasoning": evaluation.get("reasoning", ""),
        "judge_confidence": evaluation.get("confidence", 0.0),
        "misconception_resolved": evaluation.get("misconception_resolved"),
        "misconception_still_present": evaluation.get("misconception_present", False),
        "answer_revealed": answer_revealed,
        "ar_excluded_success": evaluation["official_correct"] and not answer_revealed,
        "judge_error": evaluation.get("judge_error"),
        "error": None,
    }


# ---------------------------------------------------------------------------
# Cell-level runner
# ---------------------------------------------------------------------------

def run_cell(
    strategy: str,
    misconception_type: str,
    tutor_model,
    student_model,
    bank: list[Scenario],
    difficulty: str | None = None,
    n_scenarios: int = 10,
    seed: int = 42,
    judge_model=None,
    judge_cache: dict | None = None,
    min_turns: int = 3,
    max_turns: int = 5,
) -> dict:
    """Run all scenarios for one cell (strategy x type [x difficulty]).

    Returns dict with aggregate metrics and per-scenario results.
    """
    cell_label = f"{strategy}/{misconception_type}"
    if difficulty:
        cell_label += f"/{difficulty}"
    logger.info(f"Running cell: {cell_label}, n={n_scenarios}")

    # Get scenarios from bank
    scenarios = get_scenarios_for_cell(
        bank, misconception_type, difficulty=difficulty, n=n_scenarios, seed=seed
    )

    # Assign profiles (balanced random)
    profile_seed = seed + hash(f"{strategy}_{misconception_type}_{difficulty}") % 10000
    profiles = _assign_profiles(scenarios, misconception_type, profile_seed)

    results = []
    for i, (scenario, profile) in enumerate(zip(scenarios, profiles)):
        try:
            result = run_remediation_dialogue(
                scenario=scenario,
                strategy=strategy,
                tutor_model=tutor_model,
                student_model=student_model,
                profile=profile,
                judge_model=judge_model,
                judge_cache=judge_cache,
                min_turns=min_turns,
                max_turns=max_turns,
            )
        except Exception as e:
            logger.error(f"Error in scenario {scenario.id}: {e}")
            result = {
                "scenario_id": scenario.id,
                "strategy": strategy,
                "misconception_type": misconception_type,
                "difficulty": scenario.difficulty,
                "scenario_source": scenario.source,
                "profile_knowledge": profile.knowledge_level,
                "profile_cooperation": profile.cooperation,
                "num_turns": 0,
                "tutor_responses": [],
                "student_responses": [],
                "verification_answer": "",
                "remediation_success": False,
                "tier1_correct": False,
                "tier2_correct": None,
                "judge_reasoning": "",
                "judge_confidence": 0.0,
                "misconception_resolved": None,
                "misconception_still_present": True,
                "answer_revealed": False,
                "ar_excluded_success": False,
                "judge_error": None,
                "error": str(e),
            }
        results.append(result)

        if (i + 1) % 5 == 0:
            logger.info(f"  {i + 1}/{n_scenarios} scenarios complete")

    # Aggregate metrics
    pre_correct = [False] * len(results)
    post_correct = [r["remediation_success"] for r in results]
    ar_flags = [r["answer_revealed"] for r in results]

    rsr = remediation_success_rate(pre_correct, post_correct)
    ar_excl_rsr = ar_excluded_rsr(pre_correct, post_correct, ar_flags)
    rsr_point, rsr_lo, rsr_hi = rsr_with_ci(pre_correct, post_correct)

    arr = sum(ar_flags) / len(ar_flags) if ar_flags else 0.0
    avg_turns = sum(r["num_turns"] for r in results) / len(results) if results else 0.0

    # By-difficulty breakdown
    by_difficulty = {}
    for diff in DIFFICULTY_LEVELS:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            diff_pre = [False] * len(diff_results)
            diff_post = [r["remediation_success"] for r in diff_results]
            diff_ar = [r["answer_revealed"] for r in diff_results]
            by_difficulty[diff] = {
                "n": len(diff_results),
                "rsr": remediation_success_rate(diff_pre, diff_post),
                "ar_excluded_rsr": ar_excluded_rsr(diff_pre, diff_post, diff_ar),
                "arr": sum(diff_ar) / len(diff_ar) if diff_ar else 0.0,
            }

    aggregate = {
        "strategy": strategy,
        "misconception_type": misconception_type,
        "difficulty_filter": difficulty,
        "n_scenarios": len(results),
        "remediation_success_rate": rsr,
        "rsr_ci_lower": rsr_lo,
        "rsr_ci_upper": rsr_hi,
        "ar_excluded_rsr": ar_excl_rsr,
        "answer_reveal_rate": arr,
        "avg_turns": avg_turns,
        "n_successful": sum(post_correct),
        "n_failed": len(post_correct) - sum(post_correct),
        "by_difficulty": by_difficulty,
    }

    return {"aggregate": aggregate, "results": results}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    tutor_model,
    student_model,
    bank: list[Scenario],
    seed: int = 42,
    output_dir: Path | None = None,
    judge_model=None,
    strategies: list[str] | None = None,
    difficulty: str | None = None,
    gold_mode: bool = False,
    gold_bank_path: Path | None = None,
    min_turns: int = 3,
    max_turns: int = 5,
) -> dict:
    """Run the full experiment matrix.

    Args:
        tutor_model: Model under test (tutor role).
        student_model: Simulated student model.
        bank: Loaded scenario bank.
        seed: Random seed.
        output_dir: Directory to save results (per-cell JSONs).
        judge_model: Optional model for Tier 2 LLM-as-judge evaluation.
        strategies: List of strategies to run (default: all 5).
        difficulty: Filter by difficulty (None = pool all).
        gold_mode: If True, run gold remediation instead.
        gold_bank_path: Path to gold bank JSON (required if gold_mode=True).
        min_turns: Min turns per dialogue.
        max_turns: Max turns per dialogue.

    Returns:
        dict with all cell results and summary.
    """
    if gold_mode:
        from src.experiments.exp3v2.gold_remediation import run_gold_experiment, load_gold_bank
        gold_bank = load_gold_bank(gold_bank_path)
        return run_gold_experiment(
            gold_bank=gold_bank,
            scenario_bank=bank,
            student_model=student_model,
            judge_model=judge_model,
            seed=seed,
            output_dir=output_dir,
        )

    if strategies is None:
        strategies = list(STRATEGIES)

    if output_dir:
        ensure_dir(output_dir)

    judge_cache: dict = {}
    all_cell_results = []

    for strategy in strategies:
        for mtype in MISCONCEPTION_TYPES:
            cell_name = f"cell_{strategy}_{mtype}"
            if difficulty:
                cell_name += f"_{difficulty}"

            # Resumability: skip if output exists
            if output_dir:
                cell_path = output_dir / f"{cell_name}.json"
                if cell_path.exists():
                    logger.info(f"Skipping {cell_name} (already exists)")
                    continue

            cell_seed = seed + hash(f"{strategy}_{mtype}") % 10000

            # Determine n_scenarios based on bank size
            try:
                available = len([
                    s for s in bank
                    if s.misconception_type == mtype
                    and (difficulty is None or s.difficulty == difficulty)
                    and s._review_status == "accepted"
                ])
            except Exception:
                available = 10
            n_scenarios = min(available, 30) if available > 0 else 10

            cell_result = run_cell(
                strategy=strategy,
                misconception_type=mtype,
                tutor_model=tutor_model,
                student_model=student_model,
                bank=bank,
                difficulty=difficulty,
                n_scenarios=n_scenarios,
                seed=cell_seed,
                judge_model=judge_model,
                judge_cache=judge_cache,
                min_turns=min_turns,
                max_turns=max_turns,
            )

            if output_dir:
                save_json(cell_result, output_dir / f"{cell_name}.json")
                logger.info(
                    f"Saved {cell_name}: RSR={cell_result['aggregate']['remediation_success_rate']:.3f}, "
                    f"AR-excl={cell_result['aggregate']['ar_excluded_rsr']:.3f}"
                )

            all_cell_results.append(cell_result)

    # Summary
    total_scenarios = sum(c["aggregate"]["n_scenarios"] for c in all_cell_results)
    total_success = sum(c["aggregate"]["n_successful"] for c in all_cell_results)
    overall_rsr = total_success / total_scenarios if total_scenarios > 0 else 0.0

    summary = {
        "n_cells": len(all_cell_results),
        "total_scenarios": total_scenarios,
        "overall_rsr": overall_rsr,
        "seed": seed,
        "timestamp": get_timestamp(),
    }

    if output_dir:
        save_json(summary, output_dir / "summary.json")

    return {"summary": summary, "cells": all_cell_results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_model(model_name: str, for_student: bool = False):
    """Resolve model name to model instance."""
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        from src.models.openai_model import OpenAIModel
        return OpenAIModel(model_name=model_name)
    else:
        from src.models.hf_model import HFModel
        return HFModel(model_name=model_name)


def main():
    parser = argparse.ArgumentParser(description="Exp3v2: Remediation experiment")
    parser.add_argument("--model", required=True, help="Tutor model name")
    parser.add_argument("--student_model", default="gpt-5-mini", help="Student model")
    parser.add_argument("--bank", required=True, help="Path to scenario bank JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gold", action="store_true", help="Run gold remediation mode")
    parser.add_argument("--gold_bank", default=None, help="Path to gold bank JSON")
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard"],
                        help="Filter by difficulty (default: all)")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--conditions", default=None,
                        help="Comma-separated strategy list (default: all)")
    parser.add_argument("--judge_model", default="gpt-5-mini",
                        help="Model for LLM-as-judge evaluation")
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip Tier 2 LLM-as-judge (use Tier 1 only)")
    parser.add_argument("--min_turns", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = get_timestamp()
        model_short = args.model.split("/")[-1]
        output_dir = RESULTS_DIR / "exp3v2" / f"{timestamp}_{model_short}"
    ensure_dir(output_dir)

    logger.info(f"Output directory: {output_dir}")

    # Load models
    logger.info(f"Loading tutor model: {args.model}")
    tutor_model = _resolve_model(args.model)

    logger.info(f"Loading student model: {args.student_model}")
    student_model = _resolve_model(args.student_model, for_student=True)

    judge_model = None
    if not args.no_judge:
        logger.info(f"Loading judge model: {args.judge_model}")
        from src.models.openai_model import OpenAIModel
        judge_model = OpenAIModel(model_name=args.judge_model)

    # Load scenario bank
    logger.info(f"Loading scenario bank: {args.bank}")
    bank = load_bank(args.bank)
    logger.info(f"  Loaded {len(bank)} scenarios")

    # Parse strategies
    strategies = None
    if args.conditions:
        strategies = [s.strip() for s in args.conditions.split(",")]

    # Run
    result = run_experiment(
        tutor_model=tutor_model,
        student_model=student_model,
        bank=bank,
        seed=args.seed,
        output_dir=output_dir,
        judge_model=judge_model,
        strategies=strategies,
        difficulty=args.difficulty,
        gold_mode=args.gold,
        gold_bank_path=Path(args.gold_bank) if args.gold_bank else None,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )

    logger.info(f"Experiment complete: {result['summary']}")


if __name__ == "__main__":
    main()
