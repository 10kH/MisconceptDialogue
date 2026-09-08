"""Simulated student for remediation evaluation.

Based on Scarlatos et al. (2026) framework. Simulates students with specific
misconception profiles to evaluate whether tutor remediation is effective.
"""

from dataclasses import dataclass
from typing import Optional

from ..models.base_model import ModelResponse


@dataclass
class StudentProfile:
    """A simulated student profile."""
    knowledge_level: str  # "low", "medium", "high"
    cooperation: str  # "cooperative", "resistant"
    misconception_type: str  # "factual", "procedural", "conceptual"
    misconception_description: str  # e.g., "believes multiplication always makes bigger"
    topic: str  # e.g., "fractions"


# 6 student profiles: {low, medium, high} × {cooperative, resistant}
DEFAULT_PROFILES = [
    StudentProfile("low", "cooperative", "factual", "", ""),
    StudentProfile("low", "resistant", "procedural", "", ""),
    StudentProfile("medium", "cooperative", "procedural", "", ""),
    StudentProfile("medium", "resistant", "conceptual", "", ""),
    StudentProfile("high", "cooperative", "conceptual", "", ""),
    StudentProfile("high", "resistant", "factual", "", ""),
]


def build_student_system_prompt(profile: StudentProfile) -> str:
    """Build system prompt for simulated student."""
    knowledge_desc = {
        "low": "You have very limited understanding of the topic. You struggle with basic concepts and often get confused.",
        "medium": "You have moderate understanding. You know some basics but have gaps in your knowledge.",
        "high": "You have good foundational understanding but have a specific misconception.",
    }

    cooperation_desc = {
        "cooperative": "You are willing to engage, try to answer questions, and follow the tutor's guidance.",
        "resistant": "You are somewhat reluctant to engage. You sometimes give short answers, express frustration, or resist changing your thinking.",
    }

    misconception_desc = {
        "factual": "You have a factual misconception: you believe an incorrect fact to be true.",
        "procedural": "You have a procedural misconception: you follow an incorrect procedure or algorithm.",
        "conceptual": "You have a conceptual misconception: you have a fundamental misunderstanding of an underlying concept.",
    }

    prompt = f"""You are a student in a math tutoring session. Behave realistically according to this profile:

KNOWLEDGE LEVEL: {profile.knowledge_level}
{knowledge_desc[profile.knowledge_level]}

BEHAVIOR: {profile.cooperation}
{cooperation_desc[profile.cooperation]}

MISCONCEPTION TYPE: {profile.misconception_type}
{misconception_desc[profile.misconception_type]}
{f'Specifically: {profile.misconception_description}' if profile.misconception_description else ''}

IMPORTANT RULES:
1. Stay in character as this student throughout the conversation.
2. If the tutor successfully addresses your misconception, gradually show understanding.
3. If the tutor's explanation doesn't address your specific misconception, remain confused.
4. Do NOT reveal your misconception directly - the tutor must discover it.
5. Give realistic student responses (including mistakes, partial understanding, etc.).
6. If remediation is successful, you should be able to solve a similar problem correctly.
"""
    return prompt


def build_verification_prompt(problem: str, student_answer: str) -> str:
    """Build prompt to verify if student has been remediated."""
    return f"""Now solve this similar problem to show what you've learned:

Problem: {problem}

Show your work and give your answer. If you still have the misconception, you will get it wrong.
If the tutor helped you understand, you should get it right.

Your solution:"""


def evaluate_remediation_success(
    student_response: str,
    correct_answer: str,
    misconception_indicators: list[str],
) -> dict:
    """Evaluate whether remediation was successful.

    Returns dict with:
        - correct: bool - did student get the right answer?
        - misconception_present: bool - does response still show the misconception?
        - confidence: float - how confident are we in this assessment?
    """
    response_lower = student_response.lower().strip()
    answer_lower = correct_answer.lower().strip()

    # Check if answer is correct
    correct = answer_lower in response_lower

    # Check if misconception indicators are still present
    misconception_present = any(
        indicator.lower() in response_lower
        for indicator in misconception_indicators
    )

    # Confidence heuristic
    confidence = 0.7  # base confidence
    if correct and not misconception_present:
        confidence = 0.9  # high confidence in success
    elif not correct and misconception_present:
        confidence = 0.9  # high confidence in failure
    # else: ambiguous case, keep at 0.7

    return {
        "correct": correct,
        "misconception_present": misconception_present,
        "confidence": confidence,
    }
