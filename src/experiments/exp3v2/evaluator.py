"""Two-tier evaluation system for exp3v2 remediation assessment.

Tier 1 (heuristic): Fast word-boundary regex matching for progress monitoring.
Tier 2 (LLM-as-judge): GPT-5-mini judges whether student demonstrates correct
    understanding, not just string containment. All official RSR numbers use Tier 2.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Bump this when the judge prompt changes to invalidate cached evaluations
JUDGE_PROMPT_VERSION = "v1.0"


# ---------------------------------------------------------------------------
# Tier 1: Heuristic evaluation (fast, for progress monitoring only)
# ---------------------------------------------------------------------------

def evaluate_tier1_heuristic(
    student_response: str,
    correct_answer: str,
    misconception_indicators: list[str] | None = None,
) -> dict:
    """Word-boundary regex matching — Tier 1 heuristic only.

    Improved over old substring matching by using word boundaries to avoid
    false positives like "3" matching "13".

    Returns:
        dict with: correct (bool), misconception_present (bool), confidence (float)
    """
    response_lower = student_response.lower().strip()
    answer_lower = correct_answer.lower().strip()

    # Word-boundary match to avoid "3" matching "13"
    pattern = r'\b' + re.escape(answer_lower) + r'\b'
    correct = bool(re.search(pattern, response_lower))

    # Check misconception indicators
    misconception_present = False
    if misconception_indicators:
        for indicator in misconception_indicators:
            ind_pattern = r'\b' + re.escape(indicator.lower()) + r'\b'
            if re.search(ind_pattern, response_lower):
                misconception_present = True
                break

    confidence = 0.7
    if correct and not misconception_present:
        confidence = 0.85
    elif not correct and misconception_present:
        confidence = 0.85

    return {
        "correct": correct,
        "misconception_present": misconception_present,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Tier 2: LLM-as-judge evaluation (official)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM_PROMPT = """You are an expert mathematics education evaluator. Your job is to assess whether a student has demonstrated CORRECT UNDERSTANDING of a mathematical concept after a remediation session.

IMPORTANT DISTINCTION:
- "Correct understanding" means the student grasps the underlying concept/procedure/fact.
- "Correct answer" means the student arrived at the right numerical/symbolic answer.
- Understanding matters MORE than the final answer. A student who shows correct reasoning but makes an arithmetic slip has demonstrated understanding. A student who states the right answer but shows wrong reasoning has NOT.

EVALUATION RUBRIC:
1. correct (bool): Does the student demonstrate correct understanding of the concept?
   - true: Student shows they understand the concept, even with minor arithmetic errors
   - false: Student does NOT demonstrate understanding, even if they happen to state the right answer
2. reasoning (str): Brief explanation of your judgment (1-2 sentences)
3. confidence (float): Your confidence in this assessment (0.0-1.0)
4. misconception_resolved (bool): Is the original misconception no longer present in the student's reasoning?

EDGE CASE EXAMPLES:

Example 1: Correct answer but wrong reasoning → correct: false
Student response: "The answer is 180 degrees because my teacher told me so"
Correct answer: "180 degrees"
Misconception: "Triangle angles sum to 360"
→ {"correct": false, "reasoning": "Student states correct answer but provides no evidence of understanding why triangle angles sum to 180. Reliance on authority rather than conceptual grasp.", "confidence": 0.85, "misconception_resolved": false}

Example 2: Correct reasoning but arithmetic slip → correct: true
Student response: "Since we need to subtract 6 from both sides of 2x+6=14, we get 2x=8. Then dividing by 2 gives x=5"
Correct answer: "x = 4"
Misconception: "Incorrect order of operations in linear equations"
→ {"correct": true, "reasoning": "Student demonstrates correct procedure (subtract first, then divide) which addresses the misconception. The final answer x=5 is an arithmetic slip (8/2=4, not 5), but the conceptual understanding is correct.", "confidence": 0.80, "misconception_resolved": true}

Example 3: Partial understanding → correct: false
Student response: "I think you can't always square both sides... but I'm not sure when it works and when it doesn't"
Correct answer: "Squaring both sides is valid but may introduce extraneous solutions"
Misconception: "Squaring both sides is always valid"
→ {"correct": false, "reasoning": "Student shows awareness that squaring is not always straightforward, but cannot articulate when or why. This is partial understanding, not sufficient for correct classification.", "confidence": 0.75, "misconception_resolved": false}

Example 4: Refusal or 'I don't know' → correct: false
Student response: "I don't know how to solve this"
→ {"correct": false, "reasoning": "Student explicitly states they cannot solve the problem. No understanding demonstrated.", "confidence": 0.95, "misconception_resolved": false}

Example 5: Correct answer + correct reasoning → correct: true
Student response: "Since 0 divided by any non-zero number is 0, we have 0÷5 = 0. This is different from dividing by zero (like 5÷0) which is undefined."
Correct answer: "0"
Misconception: "Confuses 0/n with n/0"
→ {"correct": true, "reasoning": "Student clearly distinguishes 0/n from n/0, directly addressing the misconception. Correct answer with correct reasoning.", "confidence": 0.95, "misconception_resolved": true}

Respond ONLY with a JSON object. No markdown, no explanation outside the JSON."""


def _build_judge_prompt(
    student_response: str,
    correct_answer: str,
    problem: str,
    misconception: str,
) -> str:
    """Build the user prompt for the LLM judge."""
    return f"""Evaluate this student's response after a remediation session:

ORIGINAL PROBLEM: {problem}
CORRECT ANSWER: {correct_answer}
MISCONCEPTION BEING ADDRESSED: {misconception}
STUDENT'S RESPONSE: {student_response}

Assess whether the student demonstrates correct understanding. Respond with JSON:
{{"correct": bool, "reasoning": "...", "confidence": float, "misconception_resolved": bool}}"""


def _cache_key(
    student_response: str,
    correct_answer: str,
    problem: str,
    misconception: str,
) -> str:
    """Compute cache key for judge evaluation."""
    content = f"{student_response}|{correct_answer}|{problem}|{misconception}|{JUDGE_PROMPT_VERSION}"
    return hashlib.sha256(content.encode()).hexdigest()


def evaluate_with_llm_judge(
    student_response: str,
    correct_answer: str,
    problem: str,
    misconception: str,
    judge_model,
    cache: dict | None = None,
) -> dict:
    """LLM-as-judge evaluation of remediation success (Tier 2).

    Args:
        student_response: The student's verification answer after remediation.
        correct_answer: The gold correct answer.
        problem: The original math problem.
        misconception: The misconception being addressed.
        judge_model: OpenAIModel instance (GPT-5-mini recommended).
        cache: Optional dict for caching results. Key = hash, value = result dict.

    Returns:
        dict with: correct (bool), reasoning (str), confidence (float),
                   misconception_resolved (bool), judge_error (str|None)
    """
    # Check cache
    key = _cache_key(student_response, correct_answer, problem, misconception)
    if cache is not None and key in cache:
        return cache[key]

    prompt = _build_judge_prompt(student_response, correct_answer, problem, misconception)

    # Retry logic: max 3 retries with exponential backoff
    max_retries = 3
    backoff_times = [1, 2, 4]  # seconds

    for attempt in range(max_retries):
        try:
            response = judge_model.generate(
                prompt=prompt,
                system_prompt=_JUDGE_SYSTEM_PROMPT,
                temperature=0.0,
                max_tokens=4096,
            )

            # Parse JSON from response
            text = response.text.strip()
            # Handle potential markdown wrapping
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)

            # Validate required fields
            parsed = {
                "correct": bool(result.get("correct", False)),
                "reasoning": str(result.get("reasoning", "")),
                "confidence": float(result.get("confidence", 0.5)),
                "misconception_resolved": bool(result.get("misconception_resolved", False)),
                "judge_error": None,
            }

            # Cache result
            if cache is not None:
                cache[key] = parsed

            return parsed

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Judge parse error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {backoff_times[attempt]}s..."
                )
                time.sleep(backoff_times[attempt])
            else:
                logger.error(f"Judge failed after {max_retries} attempts: {e}")
                # Fall back to Tier 1
                tier1 = evaluate_tier1_heuristic(student_response, correct_answer)
                return {
                    "correct": tier1["correct"],
                    "reasoning": f"LLM judge failed ({e}); fell back to Tier 1 heuristic",
                    "confidence": tier1["confidence"] * 0.5,  # reduced confidence
                    "misconception_resolved": not tier1["misconception_present"],
                    "judge_error": str(e),
                }
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"Judge API error (attempt {attempt + 1}/{max_retries}): {e}. "
                    f"Retrying in {backoff_times[attempt]}s..."
                )
                time.sleep(backoff_times[attempt])
            else:
                logger.error(f"Judge API failed after {max_retries} attempts: {e}")
                tier1 = evaluate_tier1_heuristic(student_response, correct_answer)
                return {
                    "correct": tier1["correct"],
                    "reasoning": f"LLM judge API failed ({e}); fell back to Tier 1 heuristic",
                    "confidence": tier1["confidence"] * 0.5,
                    "misconception_resolved": not tier1["misconception_present"],
                    "judge_error": str(e),
                }

    # Should not reach here, but safety fallback
    tier1 = evaluate_tier1_heuristic(student_response, correct_answer)
    return {
        "correct": tier1["correct"],
        "reasoning": "Unexpected fallback to Tier 1",
        "confidence": tier1["confidence"] * 0.5,
        "misconception_resolved": not tier1["misconception_present"],
        "judge_error": "unexpected_fallback",
    }


# ---------------------------------------------------------------------------
# Combined evaluator
# ---------------------------------------------------------------------------

def evaluate_remediation(
    student_response: str,
    correct_answer: str,
    problem: str,
    misconception: str,
    misconception_indicators: list[str] | None = None,
    judge_model=None,
    cache: dict | None = None,
) -> dict:
    """Combined two-tier evaluation.

    Always runs Tier 1 (heuristic). If judge_model is provided, also runs Tier 2.
    Official `correct` field uses Tier 2 when available, Tier 1 otherwise.

    Returns:
        dict with:
            tier1_correct (bool), tier2_correct (bool|None),
            official_correct (bool) — Tier 2 if available, else Tier 1,
            reasoning (str), confidence (float),
            misconception_resolved (bool|None),
            misconception_present (bool) — from Tier 1,
            judge_error (str|None)
    """
    # Always run Tier 1
    tier1 = evaluate_tier1_heuristic(
        student_response, correct_answer, misconception_indicators
    )

    result = {
        "tier1_correct": tier1["correct"],
        "tier2_correct": None,
        "official_correct": tier1["correct"],
        "reasoning": "Tier 1 heuristic only (no judge model)",
        "confidence": tier1["confidence"],
        "misconception_resolved": None,
        "misconception_present": tier1["misconception_present"],
        "judge_error": None,
    }

    # Run Tier 2 if judge model available
    if judge_model is not None:
        tier2 = evaluate_with_llm_judge(
            student_response=student_response,
            correct_answer=correct_answer,
            problem=problem,
            misconception=misconception,
            judge_model=judge_model,
            cache=cache,
        )
        result["tier2_correct"] = tier2["correct"]
        result["official_correct"] = tier2["correct"]
        result["reasoning"] = tier2["reasoning"]
        result["confidence"] = tier2["confidence"]
        result["misconception_resolved"] = tier2["misconception_resolved"]
        result["judge_error"] = tier2.get("judge_error")

    return result


# ---------------------------------------------------------------------------
# Judge validation
# ---------------------------------------------------------------------------

def validate_llm_judge(
    test_cases: list[dict],
    judge_model,
    threshold_agreement: float = 0.90,
    threshold_kappa: float = 0.70,
) -> dict:
    """Validate LLM-as-judge against human annotations.

    Args:
        test_cases: List of dicts with keys:
            response, correct_answer, problem, misconception, human_label (bool)
        judge_model: OpenAIModel instance.
        threshold_agreement: Minimum agreement rate (default 0.90).
        threshold_kappa: Minimum Cohen's kappa (default 0.70).

    Returns:
        dict with: agreement_rate, cohen_kappa, confusion_matrix,
                   validation_passed, n_cases, details

    Raises:
        ValueError: If fewer than 50 test cases provided.
    """
    if len(test_cases) < 50:
        raise ValueError(
            f"Need >= 50 test cases for validation, got {len(test_cases)}"
        )

    from sklearn.metrics import cohen_kappa_score, confusion_matrix

    human_labels = []
    judge_labels = []
    details = []
    cache: dict = {}

    for i, case in enumerate(test_cases):
        human = bool(case["human_label"])
        human_labels.append(human)

        result = evaluate_with_llm_judge(
            student_response=case["response"],
            correct_answer=case["correct_answer"],
            problem=case["problem"],
            misconception=case["misconception"],
            judge_model=judge_model,
            cache=cache,
        )

        judge = result["correct"]
        judge_labels.append(judge)

        details.append({
            "case_idx": i,
            "human": human,
            "judge": judge,
            "agree": human == judge,
            "reasoning": result["reasoning"],
            "confidence": result["confidence"],
        })

        if (i + 1) % 10 == 0:
            logger.info(f"  Validated {i + 1}/{len(test_cases)} cases")

    # Compute metrics
    agreement = sum(1 for h, j in zip(human_labels, judge_labels) if h == j) / len(human_labels)
    kappa = cohen_kappa_score(human_labels, judge_labels)
    cm = confusion_matrix(human_labels, judge_labels, labels=[False, True])

    passed = agreement >= threshold_agreement and kappa >= threshold_kappa

    if not passed:
        logger.warning(
            f"Judge validation FAILED: agreement={agreement:.3f} "
            f"(threshold={threshold_agreement}), kappa={kappa:.3f} "
            f"(threshold={threshold_kappa})"
        )
    else:
        logger.info(
            f"Judge validation PASSED: agreement={agreement:.3f}, kappa={kappa:.3f}"
        )

    return {
        "agreement_rate": agreement,
        "cohen_kappa": kappa,
        "confusion_matrix": cm.tolist(),
        "validation_passed": passed,
        "n_cases": len(test_cases),
        "threshold_agreement": threshold_agreement,
        "threshold_kappa": threshold_kappa,
        "details": details,
    }
