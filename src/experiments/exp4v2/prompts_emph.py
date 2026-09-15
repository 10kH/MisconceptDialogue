"""Diagnosis-emphasis prompt variant for the camera-ready robustness check.

Rationale (AACL-IJCNLP rebuttal commitment):
    The wrong-diagnosis control (condition_C_wrong) found RSR 80.8% (wrong) vs
    76.6% (gold), i.e. little evidence that tutors condition on the diagnosis
    label. This module tests whether that NULL is sensitive to a STRONGER
    diagnosis emphasis in the tutor prompt.

Only two things change relative to the C2 free-strategy prompt used by
scripts/run_wrong_diagnosis_control.py:
    1. Diagnosis-FIRST ordering — the DIAGNOSED MISCONCEPTION is stated at the
       top of the prompt, before the problem and the student's answer.
    2. Explicit directness instruction — the tutor is told its primary
       objective is to directly address / correct THIS specific misconception.

Everything else (bracket strategy-logging on turn 0, "do not reveal answer",
encouragement, end-with-question, temperatures, turn count, judge protocol,
seed, profiles) is held identical to the original control so the wrong-vs-gold
contrast is measured under emphasis and nothing else.

These builders are drop-in replacements for build_tutor_prompt(condition="C2")
as used in scripts/run_wrong_diagnosis_control.py.
"""

from __future__ import annotations

from typing import Optional


def _format_history(conversation_history: Optional[list[dict]]) -> str:
    """Format conversation history into text block (mirrors prompts._format_history)."""
    if not conversation_history:
        return ""
    turns = []
    for turn in conversation_history:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "")
        turns.append(f"{role}: {content}")
    return "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(turns)


# Header block: diagnosis stated FIRST, with an explicit directness mandate.
_EMPHASIS_HEADER = (
    "DIAGNOSED MISCONCEPTION (the specific misconception you must address): "
    "{misconception_label}\n\n"
    "Your primary objective in this tutoring session is to directly address "
    "and correct THIS specific misconception. Keep every turn focused on it: "
    "target the diagnosed misconception directly rather than teaching the "
    "topic in general."
)


def build_free_strategy_with_logging_emph(
    problem: str,
    student_answer: str,
    misconception_label: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """Turn-0 emphasis prompt (mirror of build_free_strategy_with_logging).

    Diagnosis-first + explicit directness; keeps the C2 bracket strategy log.
    """
    history_text = _format_history(conversation_history)
    header = _EMPHASIS_HEADER.format(misconception_label=misconception_label)

    return f"""You are tutoring a student who made an error on a math problem.

{header}

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}
{history_text}

Begin your response by stating your chosen teaching approach in [brackets], choosing from: [Counterexample], [Analogy], [Cognitive Conflict], [Direct Instruction], or [Socratic Questioning].

Then respond to the student using that approach, directly targeting the diagnosed misconception above. Your goals:
- Directly address and correct the diagnosed misconception stated above.
- Do NOT reveal the correct answer directly — guide the student to discover it.
- Be encouraging and patient.
- End with a question or a clear next step.

Your tutoring response:"""


def build_free_strategy_prompt_emph(
    problem: str,
    student_answer: str,
    misconception_label: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """Turn >0 emphasis prompt (mirror of build_free_strategy_prompt)."""
    history_text = _format_history(conversation_history)
    header = _EMPHASIS_HEADER.format(misconception_label=misconception_label)

    return f"""You are tutoring a student who made an error on a math problem.

{header}

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}
{history_text}

Respond to the student, directly targeting the diagnosed misconception above. Your goals:
- Directly address and correct the diagnosed misconception stated above.
- Do NOT reveal the correct answer directly — guide the student to discover it.
- Be encouraging and patient.
- End with a question or a clear next step.

Your tutoring response:"""


def build_tutor_prompt_emph(
    problem: str,
    student_answer: str,
    conversation_history: list[dict],
    misconception_label: str,
    turn: int = 0,
) -> str:
    """Emphasis-variant tutor prompt dispatcher.

    Mirrors build_tutor_prompt(condition="C2", ...): turn 0 carries the
    bracket strategy-logging instruction, later turns drop it. The ONLY
    difference from the C2 path is diagnosis-first ordering + the explicit
    directness mandate injected by the two builders above.
    """
    if turn == 0:
        return build_free_strategy_with_logging_emph(
            problem=problem,
            student_answer=student_answer,
            misconception_label=misconception_label,
            conversation_history=conversation_history or None,
        )
    return build_free_strategy_prompt_emph(
        problem=problem,
        student_answer=student_answer,
        misconception_label=misconception_label,
        conversation_history=conversation_history or None,
    )


__all__ = [
    "build_free_strategy_with_logging_emph",
    "build_free_strategy_prompt_emph",
    "build_tutor_prompt_emph",
]
