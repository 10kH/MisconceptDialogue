"""Scenario generation pipeline for exp3v2.

Pipeline: MathDial/BEA extraction → GPT-5 classification → difficulty tagging
          → variation generation → candidate bank assembly → human review.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Optional

from src.experiments.exp3v2.scenario_bank import (
    Scenario,
    MISCONCEPTION_TYPES,
    DIFFICULTY_LEVELS,
    save_bank,
)

logger = logging.getLogger(__name__)


def classify_misconception_type(
    problem: str,
    student_answer: str,
    misconception_description: str,
    model,
) -> str:
    """Classify misconception as factual, procedural, or conceptual using LLM.

    Args:
        problem: The math problem.
        student_answer: The student's incorrect answer.
        misconception_description: Description of the misconception (from teacher_described_confusion or error_description).
        model: OpenAIModel instance (GPT-5-mini recommended).

    Returns:
        One of: "factual", "procedural", "conceptual"
    """
    prompt = f"""Classify this math misconception into exactly one category.

CATEGORIES:
- factual: Student believes an incorrect fact (wrong definition, wrong property, wrong value).
  Example: "All square roots are rational" or "Triangle angles sum to 360 degrees"
- procedural: Student follows an incorrect procedure or algorithm.
  Example: "Divides before subtracting when solving 2x+6=14" or "Multiplies denominators but not numerators when adding fractions"
- conceptual: Student has a fundamental misunderstanding of an underlying concept.
  Example: "Believes multiplication always makes numbers bigger" or "Confuses area with perimeter"

PROBLEM: {problem}
STUDENT'S ANSWER: {student_answer}
MISCONCEPTION: {misconception_description}

Respond with EXACTLY one word: factual, procedural, or conceptual"""

    response = model.generate(
        prompt=prompt,
        system_prompt="You are a mathematics education expert. Classify misconceptions precisely.",
        temperature=0.0,
        max_tokens=10,
    )

    result = response.text.strip().lower()
    # Extract first valid word
    for word in result.split():
        word = word.strip(".,;:\"'")
        if word in MISCONCEPTION_TYPES:
            return word

    logger.warning(f"Could not parse misconception type from: {result!r}. Defaulting to 'conceptual'.")
    return "conceptual"


def tag_difficulty(
    problem: str,
    misconception: str,
    model,
) -> str:
    """Tag difficulty level (easy/medium/hard) using LLM.

    Criteria:
    - easy: Single-step, elementary-level, directly observable error
    - medium: Multi-step, middle-school level, requires some reasoning to identify
    - hard: Multi-concept, high-school level, subtle or deeply embedded misconception

    Returns:
        One of: "easy", "medium", "hard"
    """
    prompt = f"""Rate the difficulty of remediating this math misconception.

CRITERIA:
- easy: Single-step problem, elementary-level math, the misconception is directly observable and can be corrected with one clear explanation.
- medium: Multi-step problem, middle-school level, requires some reasoning to identify and address the misconception.
- hard: Multi-concept problem, high-school level or above, the misconception is subtle, deeply embedded, or interacts with multiple concepts.

PROBLEM: {problem}
MISCONCEPTION: {misconception}

Respond with EXACTLY one word: easy, medium, or hard"""

    response = model.generate(
        prompt=prompt,
        system_prompt="You are a mathematics education expert. Rate remediation difficulty precisely.",
        temperature=0.0,
        max_tokens=10,
    )

    result = response.text.strip().lower()
    for word in result.split():
        word = word.strip(".,;:\"'")
        if word in DIFFICULTY_LEVELS:
            return word

    logger.warning(f"Could not parse difficulty from: {result!r}. Defaulting to 'medium'.")
    return "medium"


def extract_from_mathdial(
    instances: list,
    model,
    max_instances: int | None = None,
) -> list[Scenario]:
    """Extract scenarios from MathDial dataset instances.

    Args:
        instances: List of TutoringInstance from load_mathdial().
        model: OpenAIModel for classification and difficulty tagging.
        max_instances: Maximum number to process (None = all).

    Returns:
        List of Scenario objects with LLM-classified types and difficulties.
    """
    scenarios = []
    to_process = instances[:max_instances] if max_instances else instances

    for i, inst in enumerate(to_process):
        if inst.is_correct:
            continue  # Skip correct answers — we need misconceptions

        # Get misconception description from available fields
        misconception_desc = (
            inst.misconception_label
            or inst.error_description
            or inst.metadata.get("teacher_described_confusion", "")
            or f"Student answered '{inst.student_answer}' instead of '{inst.correct_answer}'"
        )

        # Classify type
        mtype = classify_misconception_type(
            inst.problem, inst.student_answer, misconception_desc, model
        )

        # Tag difficulty
        difficulty = tag_difficulty(inst.problem, misconception_desc, model)

        scenario = Scenario(
            id=f"mathdial_{i:04d}",
            problem=inst.problem,
            student_answer=inst.student_answer,
            correct_answer=inst.correct_answer,
            misconception_label=misconception_desc,
            misconception_type=mtype,
            difficulty=difficulty,
            indicators=[],
            topic=inst.metadata.get("topic", ""),
            source="mathdial",
            source_id=inst.id,
            _review_status="pending",
        )
        scenarios.append(scenario)

        if (i + 1) % 20 == 0:
            logger.info(f"  Extracted {i + 1}/{len(to_process)} from MathDial")

    return scenarios


def extract_from_bea2025(
    instances: list,
    model,
    max_instances: int | None = None,
) -> list[Scenario]:
    """Extract scenarios from BEA 2025 dataset instances.

    Similar to extract_from_mathdial but adapted for BEA data format.
    """
    scenarios = []
    to_process = instances[:max_instances] if max_instances else instances

    for i, inst in enumerate(to_process):
        if inst.is_correct:
            continue

        misconception_desc = (
            inst.misconception_label
            or inst.error_description
            or f"Student answered '{inst.student_answer}' instead of '{inst.correct_answer}'"
        )

        mtype = classify_misconception_type(
            inst.problem, inst.student_answer, misconception_desc, model
        )
        difficulty = tag_difficulty(inst.problem, misconception_desc, model)

        scenario = Scenario(
            id=f"bea_{i:04d}",
            problem=inst.problem,
            student_answer=inst.student_answer,
            correct_answer=inst.correct_answer,
            misconception_label=misconception_desc,
            misconception_type=mtype,
            difficulty=difficulty,
            indicators=[],
            topic=inst.metadata.get("topic", ""),
            source="bea2025",
            source_id=inst.id,
            _review_status="pending",
        )
        scenarios.append(scenario)

        if (i + 1) % 20 == 0:
            logger.info(f"  Extracted {i + 1}/{len(to_process)} from BEA")

    return scenarios


def generate_variations(
    seed_scenario: Scenario,
    n_variations: int,
    model,
) -> list[Scenario]:
    """Generate variations of a seed scenario using LLM.

    Same misconception, different math problem. Preserves type and difficulty.

    Args:
        seed_scenario: The original scenario to create variations of.
        n_variations: Number of variations to generate.
        model: OpenAIModel instance (GPT-5 recommended).

    Returns:
        List of new Scenario objects.
    """
    prompt = f"""Generate {n_variations} variation(s) of this math misconception scenario.

ORIGINAL SCENARIO:
- Problem: {seed_scenario.problem}
- Student's wrong answer: {seed_scenario.student_answer}
- Correct answer: {seed_scenario.correct_answer}
- Misconception: {seed_scenario.misconception_label}
- Type: {seed_scenario.misconception_type}
- Difficulty: {seed_scenario.difficulty}

REQUIREMENTS:
1. Each variation must test the SAME misconception but with a DIFFERENT math problem
2. Maintain the same difficulty level ({seed_scenario.difficulty})
3. The student answer must be wrong in a way that reflects the same misconception
4. The correct answer must be clearly correct

Respond with a JSON array of objects, each with:
{{"problem": "...", "student_answer": "...", "correct_answer": "...", "misconception_label": "...", "topic": "..."}}"""

    response = model.generate(
        prompt=prompt,
        system_prompt="You are a mathematics education expert creating assessment scenarios.",
        temperature=0.7,
        max_tokens=2048,
    )

    text = response.text.strip()
    # Handle markdown code blocks
    if "```" in text:
        # Extract content between first ``` and last ```
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

    try:
        variations_data = json.loads(text)
        if not isinstance(variations_data, list):
            variations_data = [variations_data]
    except json.JSONDecodeError:
        logger.warning(f"Failed to parse variations JSON for {seed_scenario.id}")
        return []

    variations = []
    for j, var in enumerate(variations_data[:n_variations]):
        scenario = Scenario(
            id=f"{seed_scenario.id}_var{j:02d}",
            problem=var.get("problem", ""),
            student_answer=var.get("student_answer", ""),
            correct_answer=var.get("correct_answer", ""),
            misconception_label=var.get("misconception_label", seed_scenario.misconception_label),
            misconception_type=seed_scenario.misconception_type,
            difficulty=seed_scenario.difficulty,
            indicators=[],
            topic=var.get("topic", seed_scenario.topic),
            source="gpt5_variation",
            source_id="",
            variation_of=seed_scenario.id,
            _review_status="pending",
        )
        if scenario.problem and scenario.student_answer and scenario.correct_answer:
            variations.append(scenario)

    return variations


def validate_type_classification(
    scenarios: list[Scenario],
    human_labels: dict[str, str],
    threshold_kappa: float = 0.6,
) -> dict:
    """Validate misconception type classification against human annotations.

    Args:
        scenarios: List of scenarios with LLM-assigned types.
        human_labels: Dict mapping scenario.id -> human-assigned type (factual/procedural/conceptual).
        threshold_kappa: Minimum acceptable Cohen's kappa (default 0.6).

    Returns:
        dict with: kappa, agreement_rate, per_type_agreement, confusion_matrix,
                   validation_passed, n_validated

    Raises:
        ValueError: If fewer than 100 overlapping scenarios.
    """
    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    # Find overlapping scenarios
    llm_labels = []
    h_labels = []
    for s in scenarios:
        if s.id in human_labels:
            llm_labels.append(s.misconception_type)
            h_labels.append(human_labels[s.id])

    if len(llm_labels) < 100:
        raise ValueError(
            f"Need >= 100 overlapping scenarios for validation, "
            f"got {len(llm_labels)}. Annotate more scenarios in human_labels."
        )

    kappa = cohen_kappa_score(h_labels, llm_labels)
    agreement = sum(1 for h, l in zip(h_labels, llm_labels) if h == l) / len(h_labels)

    # Per-type agreement
    per_type = {}
    for t in MISCONCEPTION_TYPES:
        pairs = [(h, l) for h, l in zip(h_labels, llm_labels) if h == t]
        if pairs:
            per_type[t] = sum(1 for h, l in pairs if h == l) / len(pairs)
        else:
            per_type[t] = None

    cm = confusion_matrix(h_labels, llm_labels, labels=MISCONCEPTION_TYPES)

    passed = kappa >= threshold_kappa

    if not passed:
        logger.warning(
            f"Type classification validation FAILED: kappa={kappa:.3f} "
            f"(threshold={threshold_kappa}). Consider re-classifying with "
            f"improved prompts or manual override."
        )
    else:
        logger.info(
            f"Type classification validation PASSED: kappa={kappa:.3f}, "
            f"agreement={agreement:.3f}"
        )

    return {
        "kappa": kappa,
        "agreement_rate": agreement,
        "per_type_agreement": per_type,
        "confusion_matrix": cm.tolist(),
        "validation_passed": passed,
        "n_validated": len(llm_labels),
        "threshold_kappa": threshold_kappa,
    }


def build_candidate_bank(
    mathdial_instances: list | None = None,
    bea_instances: list | None = None,
    n_target_per_cell: int = 15,
    model=None,
    seed: int = 42,
    output_path: Path | str | None = None,
) -> list[Scenario]:
    """Full pipeline: extract -> classify -> tag -> generate variations -> merge.

    Args:
        mathdial_instances: From load_mathdial(). None to skip.
        bea_instances: From load_bea2025(). None to skip.
        n_target_per_cell: Target scenarios per (type, difficulty) cell.
        model: OpenAIModel for LLM calls.
        seed: Random seed.
        output_path: If provided, save candidate bank to this JSON file.

    Returns:
        list[Scenario] — candidate bank for human review.
    """
    rng = random.Random(seed)
    all_scenarios: list[Scenario] = []

    # Step 1: Extract from real data
    if mathdial_instances:
        logger.info("Extracting from MathDial...")
        mathdial_scenarios = extract_from_mathdial(mathdial_instances, model)
        all_scenarios.extend(mathdial_scenarios)
        logger.info(f"  Got {len(mathdial_scenarios)} scenarios from MathDial")

    if bea_instances:
        logger.info("Extracting from BEA...")
        bea_scenarios = extract_from_bea2025(bea_instances, model)
        all_scenarios.extend(bea_scenarios)
        logger.info(f"  Got {len(bea_scenarios)} scenarios from BEA")

    logger.info(f"Total seed scenarios: {len(all_scenarios)}")

    # Step 2: Check coverage and generate variations where needed
    from collections import Counter
    type_diff_counts = Counter(
        (s.misconception_type, s.difficulty) for s in all_scenarios
    )

    variations: list[Scenario] = []
    for mtype in MISCONCEPTION_TYPES:
        for diff in DIFFICULTY_LEVELS:
            current = type_diff_counts.get((mtype, diff), 0)
            needed = max(0, n_target_per_cell - current)

            if needed > 0 and model is not None:
                # Find seed scenarios of this type (any difficulty) to generate variations
                seeds = [s for s in all_scenarios if s.misconception_type == mtype]
                if seeds:
                    logger.info(
                        f"  Generating {needed} variations for ({mtype}, {diff})"
                    )
                    for _ in range(needed):
                        seed_scenario = rng.choice(seeds)
                        # Override difficulty for the variation
                        new_vars = generate_variations(seed_scenario, 1, model)
                        for v in new_vars:
                            v.difficulty = diff
                        variations.extend(new_vars)

    all_scenarios.extend(variations)
    logger.info(f"Total candidate scenarios (with variations): {len(all_scenarios)}")

    # Step 3: Save if output path provided
    if output_path:
        save_bank(all_scenarios, output_path)
        logger.info(f"Saved candidate bank to {output_path}")

    return all_scenarios
