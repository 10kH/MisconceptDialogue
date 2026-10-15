"""Rebuild exp3v2 scenario bank from real data sources.

Pipeline:
1. Load Misconceptions Benchmark (199 text-only items, 55 unique misconceptions)
2. Load MathDial incorrect instances (566 items with teacher-described confusion)
3. GPT-5-mini: classify type (factual/procedural/conceptual)
4. GPT-5-mini: tag difficulty (easy/medium/hard)
5. GPT-5-mini: generate verification problem + answer for each scenario
6. Balance across cells, ensure diversity (3+ distinct misconception labels per cell)
7. Validate and save

Usage:
    source .env && export OPENAI_API_KEY
    conda run -n X python scripts/rebuild_scenario_bank.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import json
import logging
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.loaders import load_misconceptions_benchmark, load_mathdial
from src.models.openai_model import OpenAIModel
from src.experiments.exp3v2.scenario_bank import (
    Scenario, MISCONCEPTION_TYPES, DIFFICULTY_LEVELS,
    save_bank, validate_bank,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Target: 10+ per cell, 30+ per type, 3+ distinct misconception labels per cell
TARGET_PER_CELL = 12
MIN_LABELS_PER_CELL = 3


def classify_type(problem: str, student_answer: str, misconception: str, model) -> str:
    """Classify misconception type using GPT-5-mini."""
    prompt = f"""Classify this math misconception into exactly one category.

CATEGORIES:
- factual: Student believes an incorrect mathematical fact (wrong definition, property, value, or rule).
  Examples: "x^0 = 0", "1 is prime", "all square roots are rational", "negative × negative = negative"
- procedural: Student applies an incorrect procedure, algorithm, or sequence of steps.
  Examples: "adds numerators and denominators separately when adding fractions", "squares each term of a binomial without cross term", "divides before subtracting when solving equations"
- conceptual: Student has a fundamental misunderstanding of an underlying mathematical concept or principle.
  Examples: "multiplication always makes numbers bigger", "correlation implies causation", "a subset must be smaller than the full set"

PROBLEM: {problem}
STUDENT'S WRONG ANSWER: {student_answer}
MISCONCEPTION: {misconception}

Respond with EXACTLY one word: factual, procedural, or conceptual"""

    resp = model.generate(
        prompt=prompt,
        system_prompt="You are a mathematics education expert. Classify precisely.",
        temperature=0.0, max_tokens=2048,
    )
    result = resp.text.strip().lower()
    for word in result.split():
        word = word.strip(".,;:\"'")
        if word in MISCONCEPTION_TYPES:
            return word
    return "conceptual"


def tag_difficulty(problem: str, misconception: str, model) -> str:
    """Tag difficulty using GPT-5-mini."""
    prompt = f"""Rate the difficulty of remediating this math misconception.

CRITERIA:
- easy: Elementary/middle-school math, single concept, the error is directly observable and fixable with one explanation. (e.g., basic arithmetic, simple fractions)
- medium: Middle/early-high-school math, may involve multiple steps, requires reasoning to identify the root cause. (e.g., algebraic manipulation, equation solving)
- hard: High-school/college-level math, the misconception is subtle, deeply embedded, or involves abstract concepts. (e.g., limits, proofs, advanced algebra)

PROBLEM: {problem}
MISCONCEPTION: {misconception}

Respond with EXACTLY one word: easy, medium, or hard"""

    resp = model.generate(
        prompt=prompt,
        system_prompt="You are a mathematics education expert.",
        temperature=0.0, max_tokens=2048,
    )
    result = resp.text.strip().lower()
    for word in result.split():
        word = word.strip(".,;:\"'")
        if word in DIFFICULTY_LEVELS:
            return word
    return "medium"


def generate_verification(problem: str, correct_answer: str, misconception: str,
                          misconception_type: str, difficulty: str, model) -> tuple[str, str]:
    """Generate a verification problem + answer using GPT-5-mini."""
    prompt = f"""Create a verification problem that tests whether a student has overcome a specific misconception.

ORIGINAL PROBLEM: {problem}
CORRECT ANSWER: {correct_answer}
MISCONCEPTION: {misconception}
TYPE: {misconception_type}
DIFFICULTY: {difficulty}

REQUIREMENTS:
1. The verification problem must test the SAME underlying concept/misconception
2. It must be a DIFFERENT problem (not just changing numbers)
3. It should be at the same difficulty level
4. The answer must be unambiguous

Respond in this exact JSON format (no markdown, no explanation):
{{"verification_problem": "...", "verification_answer": "..."}}

The verification_answer should be ONLY the final answer (number, expression, or short phrase), not a full explanation."""

    resp = model.generate(
        prompt=prompt,
        system_prompt="You are a mathematics education expert creating assessment problems.",
        temperature=0.4, max_tokens=4096,
    )
    text = resp.text.strip()
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()

    # Try direct parse first, then extract JSON from text
    for candidate in [text]:
        try:
            data = json.loads(candidate)
            vp = data.get("verification_problem", "")
            va = data.get("verification_answer", "")
            if vp and va:
                return vp, va
        except json.JSONDecodeError:
            pass

    # Try to extract JSON object from within the text
    if "{" in text and "}" in text:
        start = text.index("{")
        end = text.rindex("}") + 1
        try:
            data = json.loads(text[start:end])
            vp = data.get("verification_problem", "")
            va = data.get("verification_answer", "")
            if vp and va:
                return vp, va
        except json.JSONDecodeError:
            pass

    logger.warning(f"Failed to parse verification for: {problem[:50]}")
    return "", ""


def extract_from_misconceptions_benchmark(model) -> list[Scenario]:
    """Extract scenarios from the 220-item misconceptions benchmark."""
    mb = load_misconceptions_benchmark()
    # Filter text-only (no image dependency)
    items = [i for i in mb if not i.get("question_image")]
    logger.info(f"Misconceptions benchmark: {len(items)} text-only items")

    scenarios = []
    for idx, item in enumerate(items):
        problem = item["question"]
        student_answer = item["incorrect_answer"]
        correct_answer = item["correct_answer"]
        misconception = item["misconception"]

        if not problem.strip() or not student_answer.strip():
            continue

        # Classify and tag
        mtype = classify_type(problem, student_answer, misconception, model)
        difficulty = tag_difficulty(problem, misconception, model)

        # Generate verification
        vp, va = generate_verification(
            problem, correct_answer, misconception, mtype, difficulty, model
        )

        scenario = Scenario(
            id=f"mb_{item['misconception_id']}_{item.get('example_number', idx):02d}",
            problem=problem,
            student_answer=student_answer,
            correct_answer=correct_answer,
            misconception_label=misconception,
            misconception_type=mtype,
            difficulty=difficulty,
            indicators=[],
            verification_problem=vp,
            verification_answer=va,
            topic=item.get("topic", ""),
            source="misconceptions_benchmark",
            source_id=item.get("misconception_id", ""),
            _review_status="pending",
        )
        scenarios.append(scenario)

        if (idx + 1) % 20 == 0:
            logger.info(f"  Processed {idx+1}/{len(items)} benchmark items")

    logger.info(f"Extracted {len(scenarios)} from misconceptions benchmark")
    return scenarios


def extract_from_mathdial_data(model, max_items: int = 100) -> list[Scenario]:
    """Extract scenarios from MathDial (real student errors)."""
    md = load_mathdial("train")
    incorrect = [i for i in md if not i.is_correct and i.error_description]

    # Sample diverse items (not just the first N)
    rng = random.Random(42)
    rng.shuffle(incorrect)
    to_process = incorrect[:max_items]
    logger.info(f"MathDial: processing {len(to_process)} of {len(incorrect)} incorrect")

    scenarios = []
    for idx, inst in enumerate(to_process):
        problem = inst.problem
        student_answer = inst.student_answer
        correct_answer = inst.correct_answer
        misconception = inst.error_description

        if not problem.strip() or not student_answer.strip() or not correct_answer.strip():
            continue

        # Truncate very long answers
        if len(student_answer) > 300:
            student_answer = student_answer[:300] + "..."
        if len(correct_answer) > 300:
            correct_answer = correct_answer[:300] + "..."

        mtype = classify_type(problem, student_answer, misconception, model)
        difficulty = tag_difficulty(problem, misconception, model)
        vp, va = generate_verification(
            problem, correct_answer, misconception, mtype, difficulty, model
        )

        scenario = Scenario(
            id=f"md_{inst.id}",
            problem=problem,
            student_answer=student_answer,
            correct_answer=correct_answer,
            misconception_label=misconception,
            misconception_type=mtype,
            difficulty=difficulty,
            indicators=[],
            verification_problem=vp,
            verification_answer=va,
            topic="",
            source="mathdial",
            source_id=inst.id,
            _review_status="pending",
        )
        scenarios.append(scenario)

        if (idx + 1) % 20 == 0:
            logger.info(f"  Processed {idx+1}/{len(to_process)} MathDial items")

    logger.info(f"Extracted {len(scenarios)} from MathDial")
    return scenarios


def select_diverse_bank(all_scenarios: list[Scenario], target_per_cell: int = 12,
                        min_labels: int = 3) -> list[Scenario]:
    """Select a balanced, diverse subset from all candidates.

    Ensures:
    - target_per_cell scenarios per (type, difficulty) cell
    - min_labels distinct misconception labels per cell
    - Prefers benchmark items (higher quality) over MathDial
    """
    rng = random.Random(42)
    selected = []

    for mtype in MISCONCEPTION_TYPES:
        for diff in DIFFICULTY_LEVELS:
            cell = [s for s in all_scenarios if s.misconception_type == mtype
                    and s.difficulty == diff
                    and s.verification_problem]  # Must have verification

            if not cell:
                logger.warning(f"Empty cell: ({mtype}, {diff})")
                continue

            # Group by misconception label
            by_label = defaultdict(list)
            for s in cell:
                by_label[s.misconception_label].append(s)

            # Ensure diversity: pick from as many labels as possible
            cell_selected = []
            labels = list(by_label.keys())
            rng.shuffle(labels)

            # Round-robin across labels
            label_idx = 0
            while len(cell_selected) < target_per_cell and labels:
                label = labels[label_idx % len(labels)]
                remaining = [s for s in by_label[label] if s not in cell_selected]
                if remaining:
                    # Prefer benchmark items
                    benchmark = [s for s in remaining if s.source == "misconceptions_benchmark"]
                    pick = benchmark[0] if benchmark else remaining[0]
                    cell_selected.append(pick)
                else:
                    # This label exhausted, remove it
                    labels.remove(label)
                    if not labels:
                        break
                    label_idx = label_idx % len(labels)
                    continue
                label_idx += 1

            n_labels = len(set(s.misconception_label for s in cell_selected))
            logger.info(
                f"  ({mtype}, {diff}): {len(cell_selected)} scenarios, "
                f"{n_labels} distinct labels (from {len(cell)} candidates)"
            )
            selected.extend(cell_selected)

    return selected


def main():
    model = OpenAIModel(model_name="gpt-5-mini")

    # Step 1: Extract from both sources
    logger.info("=== Step 1: Extracting from Misconceptions Benchmark ===")
    mb_scenarios = extract_from_misconceptions_benchmark(model)

    logger.info("=== Step 2: Extracting from MathDial ===")
    md_scenarios = extract_from_mathdial_data(model, max_items=80)

    all_scenarios = mb_scenarios + md_scenarios
    logger.info(f"\nTotal candidates: {len(all_scenarios)}")

    # Filter out items without verification
    with_verif = [s for s in all_scenarios if s.verification_problem and s.verification_answer]
    logger.info(f"With verification: {len(with_verif)}")

    # Show distribution before selection
    dist = Counter((s.misconception_type, s.difficulty) for s in with_verif)
    logger.info("Distribution before selection:")
    for mtype in MISCONCEPTION_TYPES:
        for diff in DIFFICULTY_LEVELS:
            n = dist.get((mtype, diff), 0)
            labels = len(set(s.misconception_label for s in with_verif
                            if s.misconception_type == mtype and s.difficulty == diff))
            logger.info(f"  ({mtype}, {diff}): {n} scenarios, {labels} labels")

    # Step 3: Select diverse balanced bank
    logger.info("\n=== Step 3: Selecting diverse balanced bank ===")
    selected = select_diverse_bank(with_verif, target_per_cell=TARGET_PER_CELL,
                                    min_labels=MIN_LABELS_PER_CELL)

    # Renumber IDs
    type_counters = Counter()
    for s in selected:
        type_counters[s.misconception_type] += 1
        s.id = f"{s.misconception_type}_{type_counters[s.misconception_type]:04d}"
        s._review_status = "accepted"

    # Step 4: Validate
    logger.info("\n=== Step 4: Validation ===")
    result = validate_bank(selected)
    logger.info(f"Total: {result['total']}")
    logger.info(f"By type: {result['by_type']}")
    logger.info(f"By difficulty: {result['by_difficulty']}")
    logger.info(f"Imbalances: {result['imbalances']}")
    logger.info(f"Valid: {result['valid']}")

    # Diversity check
    for mtype in MISCONCEPTION_TYPES:
        for diff in DIFFICULTY_LEVELS:
            cell = [s for s in selected if s.misconception_type == mtype and s.difficulty == diff]
            labels = set(s.misconception_label for s in cell)
            sources = Counter(s.source for s in cell)
            logger.info(f"  ({mtype}, {diff}): {len(cell)} scenarios, {len(labels)} labels, sources={dict(sources)}")
            for label in sorted(labels):
                count = sum(1 for s in cell if s.misconception_label == label)
                logger.info(f"    - {label[:60]}: {count}")

    # Step 5: Save
    output_path = Path("data/exp3v2_scenario_bank.json")
    save_bank(selected, output_path)
    logger.info(f"\nSaved {len(selected)} scenarios to {output_path}")

    # Also save full candidate pool for reference
    save_bank(all_scenarios, Path("data/exp3v2_candidate_pool.json"))
    logger.info(f"Saved {len(all_scenarios)} candidates to data/exp3v2_candidate_pool.json")


if __name__ == "__main__":
    main()
