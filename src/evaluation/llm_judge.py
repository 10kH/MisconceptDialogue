"""LLM-as-Judge for pedagogical quality evaluation.

Uses GPT-4o to evaluate tutor responses on MRBench-style rubric dimensions.
"""

from typing import Optional

from ..models.base_model import ModelResponse

# 6-dimension rubric aligned with experiment protocol
RUBRIC_DIMENSIONS = {
    "diagnosis_accuracy": {
        "description": "How accurately does the tutor identify the student's specific misconception?",
        "scale": "1=completely wrong, 2=vaguely related, 3=partially correct, 4=mostly correct, 5=precisely identified",
    },
    "target_relevance": {
        "description": "How well does the explanation target the specific misconception identified?",
        "scale": "1=off-topic, 2=tangentially related, 3=somewhat relevant, 4=mostly targeted, 5=precisely targeted",
    },
    "strategy_appropriateness": {
        "description": "Is the teaching strategy appropriate for this type of misconception?",
        "scale": "1=harmful/confusing, 2=ineffective, 3=neutral, 4=appropriate, 5=optimal for this misconception type",
    },
    "scaffolding_quality": {
        "description": "Does the tutor scaffold learning without revealing the answer?",
        "scale": "1=gives answer directly, 2=heavy hints, 3=moderate scaffolding, 4=good scaffolding, 5=excellent guided discovery",
    },
    "actionability": {
        "description": "How actionable and specific is the feedback for the student?",
        "scale": "1=vague/generic, 2=somewhat specific, 3=moderately actionable, 4=specific and actionable, 5=highly specific with clear next steps",
    },
    "safety": {
        "description": "Is the response safe and pedagogically appropriate?",
        "scale": "1=harmful/reinforces misconception, 2=potentially confusing, 3=neutral, 4=safe, 5=safe and encouraging",
    },
}


def build_judge_prompt(
    problem: str,
    student_response: str,
    tutor_response: str,
    misconception_label: str = "",
    strategy_used: str = "",
) -> str:
    """Build evaluation prompt for LLM judge."""
    rubric_text = "\n".join(
        f"- **{dim}**: {info['description']}\n  Scale: {info['scale']}"
        for dim, info in RUBRIC_DIMENSIONS.items()
    )

    context = f"Misconception: {misconception_label}\n" if misconception_label else ""
    context += f"Strategy used: {strategy_used}\n" if strategy_used else ""

    return f"""You are an expert mathematics education evaluator. Rate the following tutor response on each dimension.

## Context
Problem: {problem}
Student's response: {student_response}
{context}
## Tutor's response:
{tutor_response}

## Evaluation Rubric
{rubric_text}

## Instructions
Rate each dimension from 1-5. Provide a brief justification for each rating.
Output in this exact JSON format:
{{
    "diagnosis_accuracy": {{"score": <int>, "justification": "<str>"}},
    "target_relevance": {{"score": <int>, "justification": "<str>"}},
    "strategy_appropriateness": {{"score": <int>, "justification": "<str>"}},
    "scaffolding_quality": {{"score": <int>, "justification": "<str>"}},
    "actionability": {{"score": <int>, "justification": "<str>"}},
    "safety": {{"score": <int>, "justification": "<str>"}}
}}"""


JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of mathematics tutoring quality. "
    "You evaluate tutoring responses on pedagogical dimensions. "
    "Be strict but fair. Always output valid JSON."
)


def parse_judge_response(response_text: str) -> dict:
    """Parse LLM judge response into scores dict."""
    import json
    import re

    # Try to extract JSON from response
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        return {"error": "No JSON found in judge response"}

    try:
        parsed = json.loads(json_match.group())
        scores = {}
        for dim in RUBRIC_DIMENSIONS:
            if dim in parsed:
                if isinstance(parsed[dim], dict):
                    scores[dim] = parsed[dim].get("score", 0)
                elif isinstance(parsed[dim], (int, float)):
                    scores[dim] = int(parsed[dim])
        scores["avg"] = sum(scores.values()) / len(scores) if scores else 0
        return scores
    except json.JSONDecodeError:
        return {"error": "Invalid JSON in judge response"}
