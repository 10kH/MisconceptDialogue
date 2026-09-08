"""Prompts for Experiment 3v2.

Strategy/tutor prompts are re-exported from the shared remediation prompt module.
Verification prompt is NEW — fixes the persona-collapse bug in exp3's version.
"""

from src.experiments.remediation_prompts import (
    STRATEGIES,
    MISCONCEPTION_TYPES,
    build_remediation_prompt,
    build_free_strategy_prompt,
    build_tutor_system_prompt,
    # NOTE: build_verification_problem is intentionally NOT imported.
    # The legacy prompt asked "solve a similar problem" without providing one,
    # causing the student model to self-generate problems and break persona.
)


def build_verification_prompt_v2(
    verification_problem: str,
    misconception_type: str,
) -> str:
    """Build verification prompt with a specific pre-authored problem.

    Key differences from exp3's build_verification_problem:
    1. Provides the actual problem to solve (no self-generation)
    2. Uses student-facing language (no evaluator-mode "I want to see what you learned")
    3. Keeps the student in character throughout

    Args:
        verification_problem: The specific problem the student must solve.
        misconception_type: One of "factual", "procedural", "conceptual".

    Returns:
        Prompt string for the student to solve the verification problem.
    """
    if misconception_type not in MISCONCEPTION_TYPES:
        raise ValueError(
            f"Unknown misconception_type '{misconception_type}'. "
            f"Must be one of: {MISCONCEPTION_TYPES}"
        )

    return f"""Your tutor has one more practice problem for you. Try your best to solve it.

Problem: {verification_problem}

Show your work and give your final answer."""


__all__ = [
    "STRATEGIES",
    "MISCONCEPTION_TYPES",
    "build_remediation_prompt",
    "build_free_strategy_prompt",
    "build_tutor_system_prompt",
    "build_verification_prompt_v2",
]
