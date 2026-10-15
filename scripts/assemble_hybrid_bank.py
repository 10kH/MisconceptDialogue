"""Assemble hybrid scenario bank: real data (easy/medium) + GPT-5.4 generated (hard).

Step 1: Load 279 real-data candidates from candidate pool
Step 2: Select diverse easy/medium scenarios from candidates
Step 3: Generate hard scenarios with GPT-5.4 (calculus, set theory, proofs, etc.)
Step 4: Merge, validate, save

Usage:
    source .env && export OPENAI_API_KEY
    conda run -n X python scripts/assemble_hybrid_bank.py
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.openai_model import OpenAIModel
from src.experiments.exp3v2.scenario_bank import (
    Scenario, MISCONCEPTION_TYPES, DIFFICULTY_LEVELS,
    save_bank, validate_bank,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TARGET_PER_CELL = 12


def select_diverse_from_candidates(candidates: list[dict], mtype: str, diff: str,
                                    n: int = 12) -> list[dict]:
    """Select n diverse scenarios for a (type, difficulty) cell."""
    cell = [s for s in candidates
            if s["misconception_type"] == mtype
            and s["difficulty"] == diff
            and s.get("verification_problem")]

    if not cell:
        return []

    # Group by label, round-robin for max diversity
    by_label = defaultdict(list)
    for s in cell:
        by_label[s["misconception_label"]].append(s)

    rng = random.Random(42)
    labels = list(by_label.keys())
    rng.shuffle(labels)

    selected = []
    label_idx = 0
    while len(selected) < n and labels:
        label = labels[label_idx % len(labels)]
        remaining = [s for s in by_label[label] if s not in selected]
        if remaining:
            # Prefer misconceptions_benchmark source
            benchmark = [s for s in remaining if s.get("source") == "misconceptions_benchmark"]
            pick = benchmark[0] if benchmark else remaining[0]
            selected.append(pick)
        else:
            labels.remove(label)
            if not labels:
                break
            label_idx = label_idx % len(labels)
            continue
        label_idx += 1

    return selected


HARD_SCENARIO_SPECS = {
    "factual": [
        {
            "topic": "limits",
            "misconception": "A limit that equals the function value is always continuous",
            "description": "Generate a scenario about limits and continuity where a student confuses existence of a limit with the function being continuous at that point."
        },
        {
            "topic": "number_theory",
            "misconception": "If a|bc then a|b or a|c (prime factorization misunderstanding)",
            "description": "Generate a scenario where a student assumes divisibility distributes over products without requiring a to be prime."
        },
        {
            "topic": "linear_algebra",
            "misconception": "A matrix with all positive entries has a positive determinant",
            "description": "Generate a scenario about determinants where a student assumes positivity of entries implies positive determinant."
        },
        {
            "topic": "calculus",
            "misconception": "Differentiable implies continuously differentiable",
            "description": "Generate a scenario where a student assumes that having a derivative everywhere means the derivative is continuous."
        },
        {
            "topic": "set_theory",
            "misconception": "The power set of a countably infinite set is countably infinite",
            "description": "Generate a scenario where a student thinks P(N) is countably infinite, confusing cardinality of power sets."
        },
        {
            "topic": "sequences",
            "misconception": "A bounded monotone sequence in R^n always converges",
            "description": "Generate a scenario about convergence in higher dimensions where bounded+monotone is confused with the 1D theorem."
        },
        {
            "topic": "complex_analysis",
            "misconception": "sqrt(-1) * sqrt(-1) = sqrt(1) = 1",
            "description": "Generate a scenario where a student incorrectly applies the product rule for square roots to negative numbers."
        },
        {
            "topic": "probability",
            "misconception": "P(A∪B) = P(A) + P(B) always, even for non-disjoint events",
            "description": "Generate a scenario where a student forgets the inclusion-exclusion correction for overlapping events."
        },
        {
            "topic": "abstract_algebra",
            "misconception": "Every group with prime order p is isomorphic to Z_p (but misapplied to composite orders)",
            "description": "Generate a scenario where a student assumes all groups of a given order are cyclic, even for composite orders."
        },
        {
            "topic": "topology",
            "misconception": "An open set in R^2 is always unbounded",
            "description": "Generate a scenario where a student confuses openness with unboundedness, not understanding the topological definition."
        },
        {
            "topic": "real_analysis",
            "misconception": "A uniformly continuous function on R must be bounded",
            "description": "Generate a scenario where a student confuses uniform continuity with boundedness."
        },
        {
            "topic": "calculus",
            "misconception": "If the integral of f from a to b is 0, then f(x)=0 on [a,b]",
            "description": "Generate a scenario where a student assumes zero integral implies identically zero."
        },
    ],
    "procedural": [
        {
            "topic": "integration",
            "misconception": "Integration by parts: choosing u and dv incorrectly",
            "description": "Generate a scenario where a student applies integration by parts but makes a poor choice of u and dv, leading to a more complex integral."
        },
        {
            "topic": "differential_equations",
            "misconception": "Separating variables when the ODE is not separable",
            "description": "Generate a scenario where a student tries to separate variables on a non-separable first-order ODE."
        },
        {
            "topic": "matrix_operations",
            "misconception": "Matrix multiplication is commutative (AB = BA)",
            "description": "Generate a scenario where a student swaps the order of matrix multiplication and gets the wrong result."
        },
        {
            "topic": "partial_fractions",
            "misconception": "Skipping the long division step when degree of numerator >= denominator",
            "description": "Generate a scenario where a student applies partial fractions directly without performing polynomial long division first."
        },
        {
            "topic": "series",
            "misconception": "Using the ratio test when it is inconclusive",
            "description": "Generate a scenario where a student applies the ratio test to a series where the limit equals 1, then incorrectly concludes convergence."
        },
        {
            "topic": "implicit_differentiation",
            "misconception": "Forgetting dy/dx when differentiating y terms implicitly",
            "description": "Generate a scenario where a student differentiates both sides of an implicit equation but forgets to multiply by dy/dx for y terms."
        },
        {
            "topic": "induction",
            "misconception": "Mathematical induction: assuming P(k+1) instead of proving it",
            "description": "Generate a scenario where a student's induction proof assumes what it needs to prove in the inductive step."
        },
        {
            "topic": "limits",
            "misconception": "Applying L'Hôpital's rule when the form is not 0/0 or ∞/∞",
            "description": "Generate a scenario where a student uses L'Hôpital's rule on a limit that is not in indeterminate form."
        },
        {
            "topic": "trigonometry",
            "misconception": "sin(A+B) = sin(A) + sin(B)",
            "description": "Generate a scenario where a student distributes sine over addition instead of using the angle addition formula."
        },
        {
            "topic": "logarithms",
            "misconception": "log(A+B) = log(A) + log(B)",
            "description": "Generate a scenario where a student incorrectly distributes logarithm over addition."
        },
        {
            "topic": "vector_calculus",
            "misconception": "Cross product is commutative (a×b = b×a)",
            "description": "Generate a scenario where a student treats the cross product as commutative, missing the sign change."
        },
        {
            "topic": "Taylor_series",
            "misconception": "Truncating Taylor series without considering remainder or convergence radius",
            "description": "Generate a scenario where a student uses a Taylor polynomial outside its radius of convergence."
        },
    ],
    "conceptual": [
        {
            "topic": "infinity",
            "misconception": "All infinities are the same size",
            "description": "Generate a scenario where a student claims the reals and naturals have the same cardinality, not understanding Cantor's diagonal argument."
        },
        {
            "topic": "continuity",
            "misconception": "Continuous on (a,b) implies bounded on (a,b)",
            "description": "Generate a scenario where a student assumes continuity on an open interval implies the function is bounded there."
        },
        {
            "topic": "convergence",
            "misconception": "If a_n → 0 then Σa_n converges",
            "description": "Generate a scenario where a student thinks terms going to zero is sufficient for series convergence (harmonic series counterexample)."
        },
        {
            "topic": "functions",
            "misconception": "Injective + surjective proof: confusing the two directions",
            "description": "Generate a scenario where a student's proof of bijectivity confuses the injective and surjective arguments."
        },
        {
            "topic": "probability",
            "misconception": "Independence implies mutual exclusivity (or vice versa)",
            "description": "Generate a scenario where a student conflates statistical independence with events being mutually exclusive."
        },
        {
            "topic": "calculus",
            "misconception": "Integral of 1/x from -1 to 1 is 0 by symmetry",
            "description": "Generate a scenario where a student evaluates an improper integral through a singularity by appealing to symmetry."
        },
        {
            "topic": "logic",
            "misconception": "The converse of a true statement is always true",
            "description": "Generate a scenario where a student confuses a theorem with its converse, assuming both hold."
        },
        {
            "topic": "vector_spaces",
            "misconception": "Linear independence of vectors depends on the coordinate system",
            "description": "Generate a scenario where a student thinks changing basis can make linearly dependent vectors independent."
        },
        {
            "topic": "measure_theory",
            "misconception": "A set with measure zero must be finite or countable",
            "description": "Generate a scenario where a student assumes measure zero means countable, not knowing about the Cantor set."
        },
        {
            "topic": "statistics",
            "misconception": "A larger sample always gives a narrower confidence interval",
            "description": "Generate a scenario where a student ignores the effect of population variance and confidence level on interval width."
        },
        {
            "topic": "differential_equations",
            "misconception": "A linear combination of solutions to a nonlinear ODE is also a solution",
            "description": "Generate a scenario about superposition principle being misapplied to nonlinear differential equations."
        },
        {
            "topic": "real_analysis",
            "misconception": "Pointwise convergence implies uniform convergence",
            "description": "Generate a scenario where a student claims a sequence of functions converges uniformly based only on pointwise convergence."
        },
    ],
}


def generate_hard_scenarios(model) -> list[Scenario]:
    """Generate hard scenarios using GPT-5.4."""
    all_hard = []

    for mtype, specs in HARD_SCENARIO_SPECS.items():
        logger.info(f"Generating {len(specs)} hard {mtype} scenarios...")

        for idx, spec in enumerate(specs):
            prompt = f"""{spec['description']}

Create a complete math misconception scenario with these fields:
1. problem: A specific math problem (not just a concept question - give actual numbers/expressions to work with)
2. student_answer: A specific WRONG answer that reflects this misconception
3. correct_answer: The correct answer with brief explanation
4. misconception_label: One-sentence label of the misconception
5. verification_problem: A DIFFERENT problem testing the same concept (not just changing numbers)
6. verification_answer: The correct answer to the verification problem (ONLY the final answer, no explanation)
7. indicators: 2-3 keywords that would appear in a wrong answer reflecting this misconception

Topic: {spec['topic']}
Difficulty: hard (college-level mathematics)
Misconception type: {mtype}

Respond in this exact JSON format (no markdown, no explanation):
{{"problem": "...", "student_answer": "...", "correct_answer": "...", "misconception_label": "...", "verification_problem": "...", "verification_answer": "...", "indicators": ["...", "..."]}}"""

            resp = model.generate(
                prompt=prompt,
                system_prompt="You are a university mathematics professor creating assessment scenarios for a remediation experiment. Generate realistic student misconceptions at college level.",
                temperature=0.5,
                max_tokens=4096,
            )

            text = resp.text.strip()
            # Extract JSON
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
                    id=f"{mtype}_hard_{idx:04d}",
                    problem=data["problem"],
                    student_answer=data.get("student_answer", ""),
                    correct_answer=data.get("correct_answer", ""),
                    misconception_label=data.get("misconception_label", spec["misconception"]),
                    misconception_type=mtype,
                    difficulty="hard",
                    indicators=data.get("indicators", []),
                    verification_problem=data.get("verification_problem", ""),
                    verification_answer=data.get("verification_answer", ""),
                    topic=spec["topic"],
                    source="gpt5.4_generated",
                    _review_status="accepted",
                )
                all_hard.append(scenario)
                logger.info(f"  [{mtype}] {idx+1}/{len(specs)}: {scenario.misconception_label[:50]}")
            else:
                logger.warning(f"  [{mtype}] {idx+1}/{len(specs)}: FAILED to parse")

    logger.info(f"Generated {len(all_hard)} hard scenarios total")
    return all_hard


def main():
    # Step 1: Load real-data candidates
    logger.info("=== Step 1: Loading candidate pool ===")
    with open("data/exp3v2_candidate_pool.json") as f:
        raw_candidates = json.load(f)
    logger.info(f"Loaded {len(raw_candidates)} candidates")

    # Step 2: Select easy/medium from candidates
    logger.info("\n=== Step 2: Selecting easy/medium from real data ===")
    easy_medium = []
    for mtype in MISCONCEPTION_TYPES:
        for diff in ["easy", "medium"]:
            selected = select_diverse_from_candidates(
                raw_candidates, mtype, diff, n=TARGET_PER_CELL
            )
            n_labels = len(set(s["misconception_label"] for s in selected))
            n_sources = Counter(s.get("source", "") for s in selected)
            logger.info(
                f"  ({mtype}, {diff}): {len(selected)} scenarios, "
                f"{n_labels} labels, sources={dict(n_sources)}"
            )
            easy_medium.extend(selected)

    # Convert dicts to Scenario objects
    em_scenarios = []
    type_counters = Counter()
    for s in easy_medium:
        type_counters[(s["misconception_type"], s["difficulty"])] += 1
        idx = type_counters[(s["misconception_type"], s["difficulty"])]
        scenario = Scenario(
            id=f"{s['misconception_type']}_{s['difficulty']}_{idx:04d}",
            problem=s["problem"],
            student_answer=s["student_answer"],
            correct_answer=s["correct_answer"],
            misconception_label=s["misconception_label"],
            misconception_type=s["misconception_type"],
            difficulty=s["difficulty"],
            indicators=s.get("indicators", []),
            verification_problem=s.get("verification_problem", ""),
            verification_answer=s.get("verification_answer", ""),
            topic=s.get("topic", ""),
            source=s.get("source", ""),
            source_id=s.get("source_id", ""),
            _review_status="accepted",
        )
        em_scenarios.append(scenario)

    logger.info(f"Total easy/medium: {len(em_scenarios)}")

    # Step 3: Generate hard scenarios with GPT-5.4
    logger.info("\n=== Step 3: Generating hard scenarios with GPT-5.4 ===")
    model = OpenAIModel(model_name="gpt-5.4")
    hard_scenarios = generate_hard_scenarios(model)

    # Step 4: Merge
    logger.info("\n=== Step 4: Merging and validating ===")
    all_scenarios = em_scenarios + hard_scenarios

    # Renumber all IDs cleanly
    type_counters2 = Counter()
    for s in all_scenarios:
        type_counters2[s.misconception_type] += 1
        s.id = f"{s.misconception_type}_{type_counters2[s.misconception_type]:04d}"

    # Validate
    result = validate_bank(all_scenarios)
    logger.info(f"Total: {result['total']}")
    logger.info(f"By type: {result['by_type']}")
    logger.info(f"By difficulty: {result['by_difficulty']}")
    for msg in result["imbalances"]:
        logger.info(f"  Imbalance: {msg}")
    logger.info(f"Valid: {result['valid']}")

    # Detailed diversity report
    logger.info("\n=== Diversity Report ===")
    for mtype in MISCONCEPTION_TYPES:
        for diff in DIFFICULTY_LEVELS:
            cell = [s for s in all_scenarios
                    if s.misconception_type == mtype and s.difficulty == diff]
            labels = set(s.misconception_label for s in cell)
            sources = Counter(s.source for s in cell)
            logger.info(
                f"  ({mtype}, {diff}): {len(cell)} scenarios, "
                f"{len(labels)} distinct labels, sources={dict(sources)}"
            )

    # Save
    output_path = Path("data/exp3v2_scenario_bank.json")
    save_bank(all_scenarios, output_path)
    logger.info(f"\nSaved {len(all_scenarios)} scenarios to {output_path}")


if __name__ == "__main__":
    main()
