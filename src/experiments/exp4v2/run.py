"""Experiment 4v2: D-G-R Pipeline Progressive Oracle Ablation.

Measures error propagation through the Detect-Diagnose-Remediate pipeline
by progressively removing oracle information across 6 conditions.

Usage:
    python -m src.experiments.exp4v2.run --model qwen-2.5-7b --bank data/exp3v2_scenario_bank.json
    python -m src.experiments.exp4v2.run --model phi-4 --conditions C1,C3,B1 --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
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
from src.experiments.exp3v2.scenario_bank import (
    Scenario,
    load_bank,
    DIFFICULTY_LEVELS,
)
from src.experiments.exp3v2.evaluator import evaluate_remediation
from src.experiments.exp4v2.prompts import (
    CONDITIONS,
    CONDITION_LABELS,
    OPTIMAL_STRATEGY,
    MISCONCEPTION_TYPES,
    build_detection_prompt,
    build_diagnosis_prompt,
    build_tutor_prompt,
    build_tutor_system_prompt,
    build_verification_prompt_v2,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Token adjustment (same as exp3v2)
# ---------------------------------------------------------------------------

_REASONING_PREFIXES = ("gpt-5", "o1", "o3")


def _adjust_tokens(model, base: int) -> int:
    model_name = getattr(model, "model_name", "")
    if any(model_name.startswith(p) for p in _REASONING_PREFIXES):
        return max(base * 4, 4096)
    return base


# ---------------------------------------------------------------------------
# Profile assignment (consistent across conditions)
# ---------------------------------------------------------------------------

def _assign_profiles(
    scenarios: list[Scenario],
    seed: int,
) -> list[StudentProfile]:
    """Tiled-shuffle profile assignment.

    Uses a condition-independent seed so the same scenario gets the same
    profile regardless of which condition is running (fairness).
    """
    rng = random.Random(seed)
    profiles = [
        StudentProfile(k, c, "factual", "", "")  # type set later per scenario
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

    # Set correct misconception_type per scenario
    result = []
    for i, scenario in enumerate(scenarios):
        p = assignments[i]
        result.append(StudentProfile(
            p.knowledge_level, p.cooperation,
            scenario.misconception_type, "", "",
        ))
    return result


# ---------------------------------------------------------------------------
# Answer-revealed detection
# ---------------------------------------------------------------------------

def _check_answer_revealed(tutor_responses: list[str], correct_answer: str) -> bool:
    if not correct_answer:
        return False
    all_tutor_text = " ".join(tutor_responses).lower()
    answer_lower = correct_answer.strip().lower()
    pattern = r'\b' + re.escape(answer_lower) + r'\b'
    return bool(re.search(pattern, all_tutor_text))


# ---------------------------------------------------------------------------
# Pre-step: Detection (C4)
# ---------------------------------------------------------------------------

def run_detection(tutor_model, scenario: Scenario) -> dict:
    """Run misconception detection step. Returns {detected, reasoning}."""
    prompt = build_detection_prompt(
        scenario.problem, scenario.student_answer, scenario.correct_answer,
    )
    try:
        resp = tutor_model.generate(
            prompt=prompt,
            system_prompt="You are an expert mathematics educator analyzing student errors.",
            temperature=0.1,
            max_tokens=_adjust_tokens(tutor_model, 512),
        )
        text = resp.text.strip()
        parsed = _parse_json_response(text)
        if parsed and "detected" in parsed:
            return {
                "detected": bool(parsed["detected"]),
                "reasoning": str(parsed.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning(f"Detection error for {scenario.id}: {e}")

    # Default: assume detected (conservative — avoids false negatives)
    return {"detected": True, "reasoning": "parse_failed_default_true"}


# ---------------------------------------------------------------------------
# Pre-step: Diagnosis (C3, C4)
# ---------------------------------------------------------------------------

def run_diagnosis(tutor_model, scenario: Scenario) -> dict:
    """Run misconception diagnosis step. Returns {misconception, type, reasoning}."""
    prompt = build_diagnosis_prompt(
        scenario.problem, scenario.student_answer, scenario.correct_answer,
    )
    try:
        resp = tutor_model.generate(
            prompt=prompt,
            system_prompt="You are an expert mathematics educator diagnosing student misconceptions.",
            temperature=0.1,
            max_tokens=_adjust_tokens(tutor_model, 512),
        )
        text = resp.text.strip()
        parsed = _parse_json_response(text)
        if parsed and "misconception" in parsed:
            return {
                "misconception": str(parsed["misconception"]),
                "type": str(parsed.get("type", "unknown")),
                "reasoning": str(parsed.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning(f"Diagnosis error for {scenario.id}: {e}")

    return {
        "misconception": "Unable to determine specific misconception",
        "type": "unknown",
        "reasoning": "parse_failed",
    }


# ---------------------------------------------------------------------------
# Strategy parsing (C2)
# ---------------------------------------------------------------------------

_STRATEGY_MAP = {
    "counterexample": "counterexample",
    "analogy": "analogy",
    "cognitive conflict": "cognitive_conflict",
    "cognitive_conflict": "cognitive_conflict",
    "direct instruction": "direct_instruction",
    "direct_instruction": "direct_instruction",
    "socratic": "socratic",
    "socratic questioning": "socratic",
}


def parse_chosen_strategy(tutor_text: str) -> str | None:
    """Extract strategy from [Bracket Notation] in tutor's first response."""
    match = re.search(r'\[([^\]]+)\]', tutor_text)
    if match:
        raw = match.group(1).strip().lower()
        return _STRATEGY_MAP.get(raw)
    # Fallback: look for strategy keywords in first sentence
    first_line = tutor_text.split('\n')[0].lower()
    for keyword, strategy in _STRATEGY_MAP.items():
        if keyword in first_line:
            return strategy
    return None


# ---------------------------------------------------------------------------
# JSON parsing helper
# ---------------------------------------------------------------------------

def _parse_json_response(text: str) -> dict | None:
    """Extract JSON from model response, handling markdown wrapping."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else text[3:]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        if text.startswith("json"):
            text = text[4:].strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Core: single dialogue under one condition
# ---------------------------------------------------------------------------

def run_pipeline_dialogue(
    condition: str,
    scenario: Scenario,
    tutor_model,
    student_model,
    profile: StudentProfile,
    judge_model=None,
    judge_cache: dict | None = None,
    min_turns: int = 3,
    max_turns: int = 5,
) -> dict:
    """Run one full dialogue under a specific pipeline condition."""

    # ----- Pre-steps: detection and diagnosis -----
    detection_result = None
    model_diagnosis = None
    model_diagnosis_type = None
    chosen_strategy = None

    if condition == "C4":
        det = run_detection(tutor_model, scenario)
        detection_result = det
        if not det["detected"]:
            # Model missed the misconception → no remediation
            return _make_skip_result(
                scenario, condition, profile,
                skip_reason="detection_missed",
                detection_result=det,
            )
        diag = run_diagnosis(tutor_model, scenario)
        model_diagnosis = diag["misconception"]
        model_diagnosis_type = diag["type"]

    elif condition == "C3":
        detection_result = {"detected": True, "reasoning": "oracle"}
        diag = run_diagnosis(tutor_model, scenario)
        model_diagnosis = diag["misconception"]
        model_diagnosis_type = diag["type"]

    elif condition == "C3_5":
        detection_result = {"detected": True, "reasoning": "oracle"}

    elif condition in ("C1", "C2"):
        detection_result = {"detected": True, "reasoning": "oracle"}

    # ----- Multi-turn remediation dialogue -----
    student_system = build_student_system_prompt(profile)
    tutor_system = build_tutor_system_prompt()

    conversation_history: list[dict] = []
    tutor_responses: list[str] = []
    student_responses: list[str] = []

    # Initial student message
    initial_msg = (
        f"I solved this problem: {scenario.problem}\n\n"
        f"My answer is: {scenario.student_answer}"
    )
    conversation_history.append({"role": "user", "content": initial_msg})
    student_responses.append(initial_msg)

    for turn in range(max_turns):
        # --- Tutor turn ---
        tutor_prompt = build_tutor_prompt(
            condition=condition,
            problem=scenario.problem,
            student_answer=scenario.student_answer,
            conversation_history=conversation_history,
            misconception_label=scenario.misconception_label,
            misconception_type=scenario.misconception_type,
            model_diagnosis=model_diagnosis,
            model_diagnosis_type=model_diagnosis_type,
            turn=turn,
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

        # Parse strategy from first turn (C2)
        if condition == "C2" and turn == 0:
            chosen_strategy = parse_chosen_strategy(tutor_text)

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

    # ----- Verification -----
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
        verification_answer = verif_resp.text.strip() or ""
    except Exception as e:
        logger.warning(f"Verification error: {e}")
        verification_answer = ""

    # ----- Evaluate -----
    # [FIX] The student solves verification_problem, so use verification_answer as gold.
    # Previously: scenario.correct_answer (the original answer) -> wrong gold in 102/106 scenarios.
    evaluation = evaluate_remediation(
        student_response=verification_answer,
        correct_answer=scenario.verification_answer,
        problem=scenario.verification_problem,
        misconception=scenario.misconception_label,
        misconception_indicators=scenario.indicators,
        judge_model=judge_model,
        cache=judge_cache,
    )

    answer_revealed = _check_answer_revealed(tutor_responses, scenario.correct_answer)

    # For C1, log the assigned strategy
    if condition == "C1":
        chosen_strategy = OPTIMAL_STRATEGY.get(scenario.misconception_type)

    return {
        "scenario_id": scenario.id,
        "condition": condition,
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
        # Pipeline-specific fields
        "detection_result": detection_result,
        "model_diagnosis": model_diagnosis,
        "model_diagnosis_type": model_diagnosis_type,
        "chosen_strategy": chosen_strategy,
        "error": None,
    }


def _make_skip_result(
    scenario: Scenario,
    condition: str,
    profile: StudentProfile,
    skip_reason: str,
    detection_result: dict | None = None,
) -> dict:
    """Create a result dict for skipped scenarios (e.g., detection miss)."""
    return {
        "scenario_id": scenario.id,
        "condition": condition,
        "misconception_type": scenario.misconception_type,
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
        "tier2_correct": False,
        "judge_reasoning": f"Skipped: {skip_reason}",
        "judge_confidence": 1.0,
        "misconception_resolved": False,
        "misconception_still_present": True,
        "answer_revealed": False,
        "ar_excluded_success": False,
        "judge_error": None,
        "detection_result": detection_result,
        "model_diagnosis": None,
        "model_diagnosis_type": None,
        "chosen_strategy": None,
        "error": skip_reason,
    }


# ---------------------------------------------------------------------------
# Condition-level runner
# ---------------------------------------------------------------------------

def run_condition(
    condition: str,
    tutor_model,
    student_model,
    bank: list[Scenario],
    seed: int = 42,
    judge_model=None,
    judge_cache: dict | None = None,
    min_turns: int = 3,
    max_turns: int = 5,
) -> dict:
    """Run all scenarios for one condition. Returns aggregate + per-scenario results."""
    # Use all accepted scenarios
    scenarios = [s for s in bank if s._review_status == "accepted"]
    scenarios.sort(key=lambda s: s.id)

    logger.info(f"Running condition {condition}: {CONDITION_LABELS[condition]}, n={len(scenarios)}")

    # Assign profiles (same seed for all conditions → fair comparison)
    profile_seed = seed + hash("exp4v2_profiles") % 10000
    profiles = _assign_profiles(scenarios, profile_seed)

    results = []
    for i, (scenario, profile) in enumerate(zip(scenarios, profiles)):
        try:
            result = run_pipeline_dialogue(
                condition=condition,
                scenario=scenario,
                tutor_model=tutor_model,
                student_model=student_model,
                profile=profile,
                judge_model=judge_model,
                judge_cache=judge_cache,
                min_turns=min_turns,
                max_turns=max_turns,
            )
        except Exception as e:
            logger.error(f"Error in {condition}/{scenario.id}: {e}")
            result = _make_skip_result(scenario, condition, profile, f"exception: {e}")
        results.append(result)

        if (i + 1) % 10 == 0:
            logger.info(f"  {i + 1}/{len(scenarios)} scenarios complete")

    # ----- Aggregate metrics -----
    pre_correct = [False] * len(results)
    post_correct = [r["remediation_success"] for r in results]
    ar_flags = [r["answer_revealed"] for r in results]

    rsr = remediation_success_rate(pre_correct, post_correct)
    ar_excl = ar_excluded_rsr(pre_correct, post_correct, ar_flags)
    rsr_pt, rsr_lo, rsr_hi = rsr_with_ci(pre_correct, post_correct)
    arr = sum(ar_flags) / len(ar_flags) if ar_flags else 0.0
    avg_turns = sum(r["num_turns"] for r in results) / len(results) if results else 0.0

    # By misconception type
    by_type = {}
    for mtype in MISCONCEPTION_TYPES:
        type_results = [r for r in results if r["misconception_type"] == mtype]
        if type_results:
            t_pre = [False] * len(type_results)
            t_post = [r["remediation_success"] for r in type_results]
            t_ar = [r["answer_revealed"] for r in type_results]
            by_type[mtype] = {
                "n": len(type_results),
                "rsr": remediation_success_rate(t_pre, t_post),
                "ar_excluded_rsr": ar_excluded_rsr(t_pre, t_post, t_ar),
                "arr": sum(t_ar) / len(t_ar) if t_ar else 0.0,
            }

    # By difficulty
    by_difficulty = {}
    for diff in DIFFICULTY_LEVELS:
        diff_results = [r for r in results if r["difficulty"] == diff]
        if diff_results:
            d_pre = [False] * len(diff_results)
            d_post = [r["remediation_success"] for r in diff_results]
            d_ar = [r["answer_revealed"] for r in diff_results]
            by_difficulty[diff] = {
                "n": len(diff_results),
                "rsr": remediation_success_rate(d_pre, d_post),
                "ar_excluded_rsr": ar_excluded_rsr(d_pre, d_post, d_ar),
                "arr": sum(d_ar) / len(d_ar) if d_ar else 0.0,
            }

    # Detection stats (C4 only)
    detection_stats = None
    if condition == "C4":
        detected = sum(1 for r in results if r.get("detection_result", {}).get("detected", False))
        detection_stats = {
            "total": len(results),
            "detected": detected,
            "missed": len(results) - detected,
            "detection_rate": detected / len(results) if results else 0.0,
        }

    # Strategy distribution (C2 only)
    strategy_dist = None
    if condition == "C2":
        from collections import Counter
        strategies = [r.get("chosen_strategy") for r in results]
        strategy_dist = dict(Counter(strategies))

    aggregate = {
        "condition": condition,
        "condition_label": CONDITION_LABELS[condition],
        "n_scenarios": len(results),
        "remediation_success_rate": rsr,
        "rsr_ci_lower": rsr_lo,
        "rsr_ci_upper": rsr_hi,
        "ar_excluded_rsr": ar_excl,
        "answer_reveal_rate": arr,
        "avg_turns": avg_turns,
        "n_successful": sum(post_correct),
        "n_failed": len(post_correct) - sum(post_correct),
        "by_type": by_type,
        "by_difficulty": by_difficulty,
        "detection_stats": detection_stats,
        "strategy_distribution": strategy_dist,
    }

    return {"aggregate": aggregate, "results": results}


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    tutor_model,
    student_model,
    bank: list[Scenario],
    conditions: list[str] | None = None,
    seed: int = 42,
    output_dir: Path | None = None,
    judge_model=None,
    min_turns: int = 3,
    max_turns: int = 5,
) -> dict:
    """Run experiment across selected conditions."""
    if conditions is None:
        conditions = list(CONDITIONS)

    if output_dir:
        ensure_dir(output_dir)

    judge_cache: dict = {}
    all_results = {}

    for condition in conditions:
        cell_name = f"condition_{condition}"
        if output_dir:
            cell_path = output_dir / f"{cell_name}.json"
            if cell_path.exists():
                logger.info(f"Skipping {cell_name} (already exists)")
                continue

        cond_result = run_condition(
            condition=condition,
            tutor_model=tutor_model,
            student_model=student_model,
            bank=bank,
            seed=seed,
            judge_model=judge_model,
            judge_cache=judge_cache,
            min_turns=min_turns,
            max_turns=max_turns,
        )

        if output_dir:
            save_json(cond_result, output_dir / f"{cell_name}.json")
            agg = cond_result["aggregate"]
            logger.info(
                f"Saved {cell_name}: RSR={agg['remediation_success_rate']:.3f}, "
                f"ARR={agg['answer_reveal_rate']:.3f}"
            )

        all_results[condition] = cond_result

    # Summary
    summary = {
        "n_conditions": len(all_results),
        "conditions_run": list(all_results.keys()),
        "seed": seed,
        "timestamp": get_timestamp(),
    }

    if output_dir:
        save_json(summary, output_dir / "summary.json")

    logger.info(f"Experiment complete: {summary}")
    return {"summary": summary, "conditions": all_results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_model(model_name: str):
    if model_name.startswith("gpt-") or model_name.startswith("o1") or model_name.startswith("o3"):
        from src.models.openai_model import OpenAIModel
        return OpenAIModel(model_name=model_name)
    from src.models.hf_model import HFModel
    return HFModel(model_name=model_name)


def main():
    parser = argparse.ArgumentParser(description="Exp4v2: Pipeline ablation experiment")
    parser.add_argument("--model", required=True, help="Tutor model name")
    parser.add_argument("--student_model", default="gpt-5-mini", help="Student model")
    parser.add_argument("--bank", required=True, help="Path to scenario bank JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conditions", default=None,
                        help="Comma-separated condition list (default: all 6)")
    parser.add_argument("--output_dir", default=None, help="Output directory")
    parser.add_argument("--judge_model", default="gpt-5-mini",
                        help="Model for LLM-as-judge evaluation")
    parser.add_argument("--no_judge", action="store_true",
                        help="Skip Tier 2 judge (Tier 1 heuristic only)")
    parser.add_argument("--min_turns", type=int, default=3)
    parser.add_argument("--max_turns", type=int, default=5)
    args = parser.parse_args()

    set_seed(args.seed)

    # Parse conditions
    conditions = None
    if args.conditions:
        conditions = [c.strip() for c in args.conditions.split(",")]
        for c in conditions:
            if c not in CONDITIONS:
                parser.error(f"Unknown condition '{c}'. Valid: {CONDITIONS}")

    # Output dir
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = get_timestamp()
        model_short = args.model.split("/")[-1]
        output_dir = RESULTS_DIR / "exp4v2" / f"{timestamp}_{model_short}"
    ensure_dir(output_dir)
    logger.info(f"Output directory: {output_dir}")

    # Load models
    logger.info(f"Loading tutor model: {args.model}")
    tutor_model = _resolve_model(args.model)

    logger.info(f"Loading student model: {args.student_model}")
    student_model = _resolve_model(args.student_model)

    judge_model = None
    if not args.no_judge:
        logger.info(f"Loading judge model: {args.judge_model}")
        judge_model = _resolve_model(args.judge_model)

    # Load bank
    logger.info(f"Loading scenario bank: {args.bank}")
    bank = load_bank(Path(args.bank))
    accepted = [s for s in bank if s._review_status == "accepted"]
    logger.info(f"Loaded {len(accepted)} accepted scenarios")

    # Run
    run_experiment(
        tutor_model=tutor_model,
        student_model=student_model,
        bank=bank,
        conditions=conditions,
        seed=args.seed,
        output_dir=output_dir,
        judge_model=judge_model,
        min_turns=args.min_turns,
        max_turns=args.max_turns,
    )


if __name__ == "__main__":
    main()
