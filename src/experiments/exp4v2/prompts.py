"""Prompts for Experiment 4v2: Pipeline Progressive Oracle Ablation.

6 conditions with progressive oracle removal:
  C1:   Oracle All      — gold detect + gold diagnose + optimal strategy
  C2:   Oracle D+G Free — gold detect + gold diagnose + free strategy (logged)
  C3:   Oracle Det      — gold detect + model diagnose
  C3_5: Det Only        — gold detect + no diagnosis info
  C4:   Full Pipeline   — model detect + model diagnose
  B1:   Direct          — no pipeline, minimal info
"""

from __future__ import annotations

from typing import Optional

from src.experiments.remediation_prompts import (
    STRATEGIES,
    MISCONCEPTION_TYPES,
    build_remediation_prompt,
    build_free_strategy_prompt,
    build_tutor_system_prompt,
)
from src.experiments.exp3v2.prompts import build_verification_prompt_v2

CONDITIONS = ["C1", "C2", "C3", "C3_5", "C4", "B1"]

CONDITION_LABELS = {
    "C1": "Oracle All (gold detect + gold diagnose + optimal strategy)",
    "C2": "Oracle D+G Free (gold detect + gold diagnose + free strategy)",
    "C3": "Oracle Det (gold detect + model diagnose)",
    "C3_5": "Det Only (gold detect + no diagnosis)",
    "C4": "Full Pipeline (model detect + model diagnose)",
    "B1": "Direct (no pipeline)",
}

# Best strategy per misconception type, averaged from Exp3v2 across 5 models.
OPTIMAL_STRATEGY = {
    "factual": "direct_instruction",
    "procedural": "socratic",
    "conceptual": "direct_instruction",
}


# ---------------------------------------------------------------------------
# Detection prompt (C4 only)
# ---------------------------------------------------------------------------

def build_detection_prompt(
    problem: str,
    student_answer: str,
    correct_answer: str,
) -> str:
    """Ask the model whether the student's error reflects a misconception."""
    return f"""You are examining a student's response to a math problem.

PROBLEM: {problem}
CORRECT ANSWER: {correct_answer}
STUDENT'S ANSWER: {student_answer}

The student's answer is incorrect. Does this error reflect a specific mathematical misconception (a systematic misunderstanding), or is it just a careless mistake?

Respond with ONLY a JSON object:
{{"detected": true, "reasoning": "brief explanation"}}
or
{{"detected": false, "reasoning": "brief explanation"}}"""


# ---------------------------------------------------------------------------
# Diagnosis prompt (C3, C4)
# ---------------------------------------------------------------------------

def build_diagnosis_prompt(
    problem: str,
    student_answer: str,
    correct_answer: str,
) -> str:
    """Ask the model to identify the specific misconception."""
    return f"""You are examining a student's incorrect answer to identify their mathematical misconception.

PROBLEM: {problem}
CORRECT ANSWER: {correct_answer}
STUDENT'S ANSWER: {student_answer}

Identify the specific mathematical misconception the student holds. Classify it as factual (wrong fact/definition), procedural (wrong algorithm/steps), or conceptual (fundamental misunderstanding).

Respond with ONLY a JSON object:
{{"misconception": "one-sentence description", "type": "factual|procedural|conceptual", "reasoning": "brief explanation"}}"""


# ---------------------------------------------------------------------------
# Condition-specific remediation prompts
# ---------------------------------------------------------------------------

def build_free_strategy_with_logging(
    problem: str,
    student_answer: str,
    misconception_label: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """C2: Free strategy choice with strategy logging via bracket notation."""
    history_text = _format_history(conversation_history)

    return f"""You are tutoring a student who made an error on a math problem.

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}

DIAGNOSED MISCONCEPTION: {misconception_label}
{history_text}

Begin your response by stating your chosen teaching approach in [brackets], choosing from: [Counterexample], [Analogy], [Cognitive Conflict], [Direct Instruction], or [Socratic Questioning].

Then respond to the student using that approach. Your goals:
- Help the student understand and correct their misconception.
- Do NOT reveal the correct answer directly — guide the student to discover it.
- Be encouraging and patient.
- End with a question or a clear next step.

Your tutoring response:"""


def build_model_diagnosis_remediation(
    problem: str,
    student_answer: str,
    model_diagnosis: str,
    model_diagnosis_type: Optional[str] = None,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """C3/C4: Remediation using model's own diagnosis."""
    history_text = _format_history(conversation_history)
    type_info = f"\nMISCONCEPTION TYPE: {model_diagnosis_type}" if model_diagnosis_type else ""

    return f"""You are tutoring a student who made an error on a math problem.

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}

DIAGNOSED MISCONCEPTION: {model_diagnosis}{type_info}
{history_text}

Respond to the student using whatever pedagogical approach you think is most effective.
Your goals:
- Help the student understand and correct their misconception.
- Do NOT reveal the correct answer directly — guide the student to discover it.
- Be encouraging and patient.
- End with a question or a clear next step.

Your tutoring response:"""


def build_no_diagnosis_remediation(
    problem: str,
    student_answer: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """C3.5: Remediation without any diagnosis information."""
    history_text = _format_history(conversation_history)

    return f"""You are tutoring a student who made an error on a math problem.

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}

The student's answer is incorrect. You need to help them understand their mistake, but you don't know exactly what misconception they hold. Try to identify the issue from their answer and guide them toward the correct understanding.
{history_text}

Your goals:
- Help the student understand and correct their error.
- Do NOT reveal the correct answer directly — guide the student to discover it.
- Be encouraging and patient.
- End with a question or a clear next step.

Your tutoring response:"""


def build_direct_remediation(
    problem: str,
    student_answer: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """B1: Direct remediation with minimal info (no pipeline)."""
    history_text = _format_history(conversation_history)

    return f"""You are a math tutor. A student is working on a problem and needs help.

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}
{history_text}

Help the student. Do NOT reveal the correct answer directly — guide them to discover it.
Be encouraging and patient. End with a question or a clear next step.

Your tutoring response:"""


# ---------------------------------------------------------------------------
# Unified prompt builder for multi-turn dialogue
# ---------------------------------------------------------------------------

def build_tutor_prompt(
    condition: str,
    problem: str,
    student_answer: str,
    conversation_history: list[dict],
    misconception_label: str | None = None,
    misconception_type: str | None = None,
    model_diagnosis: str | None = None,
    model_diagnosis_type: str | None = None,
    turn: int = 0,
) -> str:
    """Build the tutor prompt for a given condition and turn.

    Dispatches to the appropriate prompt builder based on condition.
    For C2, only the first turn includes the strategy-logging bracket instruction.
    """
    if condition == "C1":
        strategy = OPTIMAL_STRATEGY.get(misconception_type, "direct_instruction")
        return build_remediation_prompt(
            problem=problem,
            student_answer=student_answer,
            misconception_label=misconception_label or "",
            misconception_type=misconception_type or "factual",
            strategy=strategy,
            conversation_history=conversation_history or None,
        )
    elif condition == "C2":
        if turn == 0:
            return build_free_strategy_with_logging(
                problem=problem,
                student_answer=student_answer,
                misconception_label=misconception_label or "",
                conversation_history=conversation_history or None,
            )
        else:
            return build_free_strategy_prompt(
                problem=problem,
                student_answer=student_answer,
                misconception_label=misconception_label or "",
                conversation_history=conversation_history or None,
            )
    elif condition in ("C3", "C4"):
        return build_model_diagnosis_remediation(
            problem=problem,
            student_answer=student_answer,
            model_diagnosis=model_diagnosis or "Unknown",
            model_diagnosis_type=model_diagnosis_type,
            conversation_history=conversation_history or None,
        )
    elif condition == "C3_5":
        return build_no_diagnosis_remediation(
            problem=problem,
            student_answer=student_answer,
            conversation_history=conversation_history or None,
        )
    elif condition == "B1":
        return build_direct_remediation(
            problem=problem,
            student_answer=student_answer,
            conversation_history=conversation_history or None,
        )
    else:
        raise ValueError(f"Unknown condition: {condition}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_history(conversation_history: Optional[list[dict]]) -> str:
    """Format conversation history into text block."""
    if not conversation_history:
        return ""
    turns = []
    for turn in conversation_history:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        turns.append(f"{role}: {content}")
    return "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(turns)


__all__ = [
    "CONDITIONS",
    "CONDITION_LABELS",
    "OPTIMAL_STRATEGY",
    "STRATEGIES",
    "MISCONCEPTION_TYPES",
    "build_detection_prompt",
    "build_diagnosis_prompt",
    "build_tutor_prompt",
    "build_tutor_system_prompt",
    "build_verification_prompt_v2",
]
