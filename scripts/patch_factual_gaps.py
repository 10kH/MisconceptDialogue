"""Patch factual easy/medium gaps in the hybrid scenario bank.

Real data only had 9 factual_easy and 1 factual_medium.
Generate GPT-5.4 scenarios to fill to 12 each.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.openai_model import OpenAIModel
from src.experiments.exp3v2.scenario_bank import (
    Scenario, MISCONCEPTION_TYPES, DIFFICULTY_LEVELS,
    save_bank, validate_bank,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

FACTUAL_EASY_SPECS = [
    {
        "topic": "arithmetic",
        "misconception": "Negative times negative is negative",
        "description": "Generate a scenario where a student believes multiplying two negative numbers gives a negative result."
    },
    {
        "topic": "fractions",
        "misconception": "A fraction with a larger denominator is always larger",
        "description": "Generate a scenario where a student thinks 1/8 > 1/5 because 8 > 5."
    },
    {
        "topic": "exponents",
        "misconception": "x^0 = 0 for any x",
        "description": "Generate a scenario where a student believes any number raised to the power of 0 equals 0 instead of 1."
    },
]

FACTUAL_MEDIUM_SPECS = [
    {
        "topic": "algebra",
        "misconception": "Square root of a sum equals the sum of square roots",
        "description": "Generate a scenario where a student believes sqrt(a+b) = sqrt(a) + sqrt(b)."
    },
    {
        "topic": "geometry",
        "misconception": "All quadrilaterals with equal sides are squares",
        "description": "Generate a scenario where a student confuses rhombus properties with square properties."
    },
    {
        "topic": "algebra",
        "misconception": "(a+b)^2 = a^2 + b^2 (missing cross term)",
        "description": "Generate a scenario where a student squares a binomial without the 2ab cross term."
    },
    {
        "topic": "probability",
        "misconception": "Probability of A or B is P(A) * P(B) (confusing OR with AND)",
        "description": "Generate a scenario where a student multiplies probabilities when they should add them."
    },
    {
        "topic": "functions",
        "misconception": "f(a+b) = f(a) + f(b) for any function",
        "description": "Generate a scenario where a student assumes all functions are linear/additive."
    },
    {
        "topic": "geometry",
        "misconception": "Doubling the radius doubles the area of a circle",
        "description": "Generate a scenario where a student doesn't understand the quadratic relationship between radius and area."
    },
    {
        "topic": "algebra",
        "misconception": "Dividing both sides of an inequality by a negative number doesn't flip the sign",
        "description": "Generate a scenario where a student forgets to flip the inequality sign when dividing by negative."
    },
    {
        "topic": "trigonometry",
        "misconception": "sin(90°) = 0 (confusing sine and cosine values)",
        "description": "Generate a scenario where a student confuses the values of sine and cosine at key angles."
    },
    {
        "topic": "statistics",
        "misconception": "Mean and median are always equal",
        "description": "Generate a scenario where a student assumes mean equals median for any distribution."
    },
    {
        "topic": "number_theory",
        "misconception": "1 is a prime number",
        "description": "Generate a scenario where a student includes 1 in the list of prime numbers."
    },
    {
        "topic": "algebra",
        "misconception": "The absolute value of a negative number is negative",
        "description": "Generate a scenario where a student thinks |x| preserves the sign of x."
    },
]


def generate_scenarios(model, specs: list[dict], mtype: str, difficulty: str) -> list[Scenario]:
    scenarios = []
    for idx, spec in enumerate(specs):
        prompt = f"""{spec['description']}

Create a complete math misconception scenario with these fields:
1. problem: A specific math problem (give actual numbers/expressions)
2. student_answer: A specific WRONG answer reflecting this misconception
3. correct_answer: The correct answer
4. misconception_label: One-sentence label
5. verification_problem: A DIFFERENT problem testing the same concept
6. verification_answer: The correct answer to the verification problem (ONLY the final answer)

Topic: {spec['topic']}
Difficulty: {difficulty} ({'elementary/middle-school' if difficulty == 'easy' else 'middle/high-school'})
Misconception type: {mtype}

Respond in this exact JSON format (no markdown):
{{"problem": "...", "student_answer": "...", "correct_answer": "...", "misconception_label": "...", "verification_problem": "...", "verification_answer": "..."}}"""

        resp = model.generate(
            prompt=prompt,
            system_prompt="You are a mathematics education expert creating assessment scenarios.",
            temperature=0.5,
            max_tokens=4096,
        )

        text = resp.text.strip()
        if "```" in text:
            parts = text.split("```")
            if len(parts) >= 3:
                text = parts[1]
                if text.startswith("json"):
                    text = text[4:]
            text = text.strip()

        data = None
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            if "{" in text and "}" in text:
                try:
                    data = json.loads(text[text.index("{"):text.rindex("}") + 1])
                except json.JSONDecodeError:
                    pass

        if data and data.get("problem") and data.get("verification_problem"):
            scenario = Scenario(
                id=f"patch_{mtype}_{difficulty}_{idx:04d}",
                problem=data["problem"],
                student_answer=data.get("student_answer", ""),
                correct_answer=data.get("correct_answer", ""),
                misconception_label=data.get("misconception_label", spec["misconception"]),
                misconception_type=mtype,
                difficulty=difficulty,
                indicators=[],
                verification_problem=data.get("verification_problem", ""),
                verification_answer=data.get("verification_answer", ""),
                topic=spec["topic"],
                source="gpt5.4_generated",
                _review_status="accepted",
            )
            scenarios.append(scenario)
            logger.info(f"  [{mtype}/{difficulty}] {idx+1}/{len(specs)}: {scenario.misconception_label[:60]}")
        else:
            logger.warning(f"  [{mtype}/{difficulty}] {idx+1}/{len(specs)}: FAILED")

    return scenarios


def main():
    model = OpenAIModel(model_name="gpt-5.4")

    # Load current bank
    bank_path = Path("data/exp3v2_scenario_bank.json")
    with open(bank_path) as f:
        raw = json.load(f)

    existing = []
    for s in raw:
        existing.append(Scenario(**{k: v for k, v in s.items() if k in Scenario.__dataclass_fields__}))

    logger.info(f"Loaded {len(existing)} existing scenarios")

    # Generate factual_easy (3 more)
    logger.info("=== Generating factual_easy supplements ===")
    fe = generate_scenarios(model, FACTUAL_EASY_SPECS, "factual", "easy")

    # Generate factual_medium (11 more)
    logger.info("=== Generating factual_medium supplements ===")
    fm = generate_scenarios(model, FACTUAL_MEDIUM_SPECS, "factual", "medium")

    # Merge
    all_scenarios = existing + fe + fm
    logger.info(f"Total after patch: {len(all_scenarios)}")

    # Renumber IDs
    type_counters = Counter()
    for s in all_scenarios:
        type_counters[s.misconception_type] += 1
        s.id = f"{s.misconception_type}_{type_counters[s.misconception_type]:04d}"

    # Validate
    result = validate_bank(all_scenarios)
    logger.info(f"By type: {result['by_type']}")
    logger.info(f"By difficulty: {result['by_difficulty']}")
    for msg in result["imbalances"]:
        logger.info(f"  Imbalance: {msg}")
    logger.info(f"Valid: {result['valid']}")

    # Diversity report
    for mtype in MISCONCEPTION_TYPES:
        for diff in DIFFICULTY_LEVELS:
            cell = [s for s in all_scenarios if s.misconception_type == mtype and s.difficulty == diff]
            labels = set(s.misconception_label for s in cell)
            sources = Counter(s.source for s in cell)
            logger.info(f"  ({mtype}, {diff}): {len(cell)}, {len(labels)} labels, {dict(sources)}")

    save_bank(all_scenarios, bank_path)
    logger.info(f"Saved {len(all_scenarios)} scenarios to {bank_path}")


if __name__ == "__main__":
    main()
