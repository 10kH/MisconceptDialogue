"""Detection prompts for Experiment 1: Misconception Detection (BEA 2025 style).

Five tracks mirroring the BEA 2025 shared task:
  Track 1 — Mistake Identification: correct / incorrect / ambiguous
  Track 2 — Mistake Location: where in the student's work is the error
  Track 3 — Guidance Quality: rate the tutor's response quality
  Track 4 — Actionability: is the tutor's feedback actionable
  Track 5 — Misconception Classification (NEW): which misconception applies

The context_turns parameter controls single-turn vs multi-turn context (RQ1):
  context_turns=None or 0 — single-turn (only the final student response)
  context_turns=N          — include the N most recent prior dialogue turns
  context_turns=-1         — include full dialogue history
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt shared by all detection tracks
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_DETECTION = """You are an expert mathematics tutor evaluator. Your role is to analyze student responses in mathematics tutoring dialogues and make precise, evidence-based assessments.

Guidelines:
- Focus on the mathematical content, not writing style or grammar.
- Base your assessment only on what the student has explicitly written.
- When dialogue history is provided, use it to understand the full context of the exchange.
- Be concise and structured in your responses.
- Follow the exact output format specified in each task."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _format_context(context_turns: list[dict] | None) -> str:
    """Format prior dialogue turns into a readable block.

    Args:
        context_turns: List of dicts with keys "role" and "content".
            Roles are typically "tutor" and "student".

    Returns:
        Formatted string, or empty string if no turns provided.
    """
    if not context_turns:
        return ""
    lines = []
    for turn in context_turns:
        role = turn.get("role", "unknown").capitalize()
        content = turn.get("content", "").strip()
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _build_context_block(context_turns: list[dict] | None) -> str:
    """Build the full context section for insertion into a prompt.

    Returns empty string if no context, otherwise returns a formatted
    section header + the turns.
    """
    if not context_turns:
        return ""
    formatted = _format_context(context_turns)
    return f"\n## Prior Dialogue Context\n{formatted}\n"


def _slice_context(
    dialogue_history: list[dict] | None,
    context_turns: int | None,
) -> list[dict] | None:
    """Slice the dialogue history to the requested number of prior turns.

    Args:
        dialogue_history: Full conversation history.
        context_turns: Number of prior turns to include.
            None or 0 — single-turn (return None / empty).
            Positive N — last N turns before the current student response.
            -1         — full history.

    Returns:
        Sliced list of turn dicts, or None for single-turn mode.
    """
    if not dialogue_history or context_turns is None or context_turns == 0:
        return None
    if context_turns == -1:
        return dialogue_history
    return dialogue_history[-context_turns:]


# ---------------------------------------------------------------------------
# Track 1: Mistake Identification
# ---------------------------------------------------------------------------

def build_mistake_identification_prompt(
    problem: str,
    student_response: str,
    context_turns: list[dict] | None = None,
) -> str:
    """Track 1: Classify the student's answer as correct, incorrect, or ambiguous.

    BEA 2025 Track 1 asks models to determine whether a student's response
    contains a mistake. This prompt returns a classification over three labels:
      - correct    — the student's answer is mathematically correct
      - incorrect  — the student's answer contains a clear mathematical error
      - ambiguous  — the response is incomplete, unclear, or cannot be
                     determined without more information

    Args:
        problem: The math problem presented to the student.
        student_response: The student's answer or reply.
        context_turns: Prior dialogue turns (None for single-turn mode).
            Each dict has keys "role" and "content".

    Returns:
        Prompt string ready to pass to model.generate().
    """
    context_block = _build_context_block(context_turns)

    return f"""## Task: Mistake Identification (Track 1)
{context_block}
## Problem
{problem}

## Student Response
{student_response}

## Instructions
Classify the student's response as exactly one of:
  - correct    — the answer is mathematically correct
  - incorrect  — the answer contains a clear mathematical error
  - ambiguous  — the response is incomplete, unclear, or cannot be assessed

Respond in the following format:
CLASSIFICATION: <correct|incorrect|ambiguous>
REASONING: <one sentence justifying your classification>"""


# ---------------------------------------------------------------------------
# Track 2: Mistake Location
# ---------------------------------------------------------------------------

def build_mistake_location_prompt(
    problem: str,
    student_response: str,
    context_turns: list[dict] | None = None,
) -> str:
    """Track 2: Identify where exactly the mistake occurs in the student's work.

    Given an incorrect student response, locate the specific step or element
    where the error first appears. If the student response is correct, state
    that no mistake was found.

    Args:
        problem: The math problem presented to the student.
        student_response: The student's answer or reply (may contain multiple steps).
        context_turns: Prior dialogue turns (None for single-turn mode).

    Returns:
        Prompt string ready to pass to model.generate().
    """
    context_block = _build_context_block(context_turns)

    return f"""## Task: Mistake Location (Track 2)
{context_block}
## Problem
{problem}

## Student Response
{student_response}

## Instructions
Identify the exact location in the student's work where the first mathematical error occurs.

If the student's response is correct, output:
MISTAKE_LOCATION: none
EXPLANATION: The student's response is correct.

If the student's response contains an error, output:
MISTAKE_LOCATION: <quote the specific step, expression, or phrase where the error occurs>
EXPLANATION: <one or two sentences explaining what went wrong at that location and what the correct approach should be>"""


# ---------------------------------------------------------------------------
# Track 3: Guidance Quality
# ---------------------------------------------------------------------------

def build_guidance_quality_prompt(
    problem: str,
    student_response: str,
    tutor_response: str,
) -> str:
    """Track 3: Rate the quality of the tutor's guidance response.

    Evaluates the tutor's pedagogical response on a 3-point scale:
      1 — Poor: unhelpful, incorrect, reveals the answer directly, or ignores the error
      2 — Adequate: partially helpful but could be clearer or more targeted
      3 — Good: clearly identifies the error and guides the student toward correction

    Args:
        problem: The math problem presented to the student.
        student_response: The student's answer (presumably incorrect).
        tutor_response: The tutor's follow-up guidance to the student.

    Returns:
        Prompt string ready to pass to model.generate().
    """
    return f"""## Task: Guidance Quality Rating (Track 3)

## Problem
{problem}

## Student Response
{student_response}

## Tutor Response
{tutor_response}

## Instructions
Rate the quality of the tutor's response on a 3-point scale:
  1 — Poor: the response is unhelpful, incorrect, gives away the answer directly,
            or fails to address the student's actual error
  2 — Adequate: the response is partially helpful but lacks clarity, precision,
                or appropriate scaffolding
  3 — Good: the response clearly identifies the error, provides helpful guidance
            toward the correct approach, and does not simply reveal the answer

Respond in the following format:
QUALITY_RATING: <1|2|3>
REASONING: <one or two sentences justifying your rating>"""


# ---------------------------------------------------------------------------
# Track 4: Actionability
# ---------------------------------------------------------------------------

def build_actionability_prompt(
    problem: str,
    student_response: str,
    tutor_response: str,
) -> str:
    """Track 4: Determine whether the tutor's feedback is actionable.

    Actionable feedback gives the student a concrete next step they can take
    to correct their understanding or approach. Non-actionable feedback is
    vague, generic, or does not tell the student what to do differently.

    Args:
        problem: The math problem presented to the student.
        student_response: The student's answer (presumably incorrect).
        tutor_response: The tutor's follow-up response.

    Returns:
        Prompt string ready to pass to model.generate().
    """
    return f"""## Task: Feedback Actionability (Track 4)

## Problem
{problem}

## Student Response
{student_response}

## Tutor Response
{tutor_response}

## Instructions
Determine whether the tutor's response provides actionable feedback.

Actionable feedback:
  - Gives the student a specific, concrete next step to take
  - Points to a particular concept, procedure, or fact to review or apply
  - Is targeted to the student's specific error, not generic advice

Non-actionable feedback:
  - Is vague ("think more carefully", "try again")
  - Simply confirms the answer is wrong without guidance
  - Is so generic it applies to any math error

Respond in the following format:
ACTIONABLE: <yes|no>
REASONING: <one sentence justifying your judgment>"""


# ---------------------------------------------------------------------------
# Track 5 (NEW): Misconception Classification
# ---------------------------------------------------------------------------

def build_misconception_classification_prompt(
    problem: str,
    student_response: str,
    misconception_list: list[str | dict],
    context_turns: list[dict] | None = None,
) -> str:
    """Track 5 (NEW): Classify which specific misconception the student has.

    Given a curated list of possible misconceptions, identify which one best
    explains the student's error. This extends BEA 2025 beyond binary
    correct/incorrect detection to fine-grained misconception classification.

    The misconception_list can contain either:
      - Plain strings (misconception names/descriptions)
      - Dicts with keys: "id", "name", "description" (and optionally "type", "domain")

    Args:
        problem: The math problem presented to the student.
        student_response: The student's answer or reply.
        misconception_list: Possible misconceptions to classify among.
            Strings are used directly; dicts are formatted as "[id] name: description".
        context_turns: Prior dialogue turns (None for single-turn mode).
            Each dict has keys "role" and "content".

    Returns:
        Prompt string ready to pass to model.generate().
    """
    context_block = _build_context_block(context_turns)

    # Format misconception list
    formatted_options: list[str] = []
    for i, m in enumerate(misconception_list):
        if isinstance(m, dict):
            mid = m.get("id", str(i + 1))
            name = m.get("name", "")
            desc = m.get("description", "")
            type_str = m.get("type", "")
            domain_str = m.get("domain", "")
            qualifier = ""
            if type_str or domain_str:
                qualifier = f" ({', '.join(x for x in (type_str, domain_str) if x)})"
            if desc:
                formatted_options.append(f"[{mid}] {name}{qualifier}: {desc}")
            else:
                formatted_options.append(f"[{mid}] {name}{qualifier}")
        else:
            formatted_options.append(f"[{i + 1}] {m}")

    misconceptions_block = "\n".join(formatted_options)

    return f"""## Task: Misconception Classification (Track 5)
{context_block}
## Problem
{problem}

## Student Response
{student_response}

## Possible Misconceptions
{misconceptions_block}

## Instructions
Identify which misconception from the list above best explains the student's error.

If the student's response is correct (no error), output:
MISCONCEPTION_ID: none
REASONING: The student's response does not contain a detectable error.

Otherwise, output the ID of the best-matching misconception:
MISCONCEPTION_ID: <id>
REASONING: <one or two sentences explaining how this misconception explains the observed error>

If multiple misconceptions could apply, choose the single most likely one. If none of the listed misconceptions fit, output:
MISCONCEPTION_ID: other
REASONING: <describe the apparent misconception in one sentence>"""
