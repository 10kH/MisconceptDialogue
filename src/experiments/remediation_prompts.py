"""Prompts for Experiment 3: Remediation Strategy Effectiveness."""

from typing import Optional

STRATEGIES = ["counterexample", "analogy", "cognitive_conflict", "direct_instruction", "socratic"]
MISCONCEPTION_TYPES = ["factual", "procedural", "conceptual"]

# Strategy-specific instructions for the tutor
_STRATEGY_INSTRUCTIONS = {
    "counterexample": (
        "Use the COUNTEREXAMPLE strategy. Present a specific concrete case or example "
        "where the student's belief leads to a clearly wrong result. Show them a situation "
        "where applying their misconception fails or produces a contradiction. "
        "Do NOT simply state the rule—demonstrate the failure case vividly."
    ),
    "analogy": (
        "Use the ANALOGY strategy. Explain the correct concept by drawing a parallel "
        "to something familiar from everyday life that the student already understands. "
        "Make the analogy concrete and explicit. Connect the familiar situation back to "
        "the math concept clearly."
    ),
    "cognitive_conflict": (
        "Use the COGNITIVE CONFLICT strategy. Ask a targeted sequence of questions that "
        "expose an internal contradiction or tension in the student's reasoning. "
        "Do not directly state the correct answer. Instead, guide the student to see "
        "that their belief leads to two incompatible conclusions."
    ),
    "direct_instruction": (
        "Use the DIRECT INSTRUCTION strategy. Clearly and explicitly state the correct "
        "fact, rule, or procedure. Explain exactly what the student got wrong and why. "
        "Walk through the correct procedure step by step. Be precise and didactic."
    ),
    "socratic": (
        "Use the SOCRATIC QUESTIONING strategy. Guide the student toward self-discovery "
        "through a carefully scaffolded sequence of questions. Each question should build "
        "on the student's last response and move them incrementally toward the correct "
        "understanding. Never state the answer directly—let the student arrive at it."
    ),
}

_MISCONCEPTION_TYPE_CONTEXT = {
    "factual": "The student holds an incorrect factual belief (e.g., a wrong definition or property).",
    "procedural": "The student follows an incorrect procedure or algorithm when solving problems.",
    "conceptual": "The student has a fundamental misunderstanding of an underlying mathematical concept.",
}

_TUTOR_SYSTEM_PROMPT = (
    "You are an expert mathematics tutor. Your goal is to help the student overcome "
    "their misconception without directly revealing the answer. Be patient, encouraging, "
    "and pedagogically effective. Keep responses concise (2-4 sentences unless more is "
    "needed). Always end your turn with either a question or a clear next step for the student."
)


def build_remediation_prompt(
    problem: str,
    student_answer: str,
    misconception_label: str,
    misconception_type: str,
    strategy: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """Build prompt for tutor to remediate using a specific strategy.

    Each strategy has specific instructions:
    - counterexample: Present a case where the misconception fails
    - analogy: Use a familiar analogy to explain the correct concept
    - cognitive_conflict: Ask questions that expose contradictions in student's reasoning
    - direct_instruction: Explicitly teach the correct procedure/fact
    - socratic: Guide through step-by-step questions toward self-discovery

    Args:
        problem: The original math problem.
        student_answer: The student's incorrect answer.
        misconception_label: Short label for the misconception (e.g. "multiplication always increases").
        misconception_type: One of MISCONCEPTION_TYPES ("factual", "procedural", "conceptual").
        strategy: One of STRATEGIES.
        conversation_history: Prior turns as list of {"role": ..., "content": ...} dicts.

    Returns:
        Full prompt string for the tutor model.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}'. Must be one of: {STRATEGIES}")
    if misconception_type not in MISCONCEPTION_TYPES:
        raise ValueError(
            f"Unknown misconception_type '{misconception_type}'. Must be one of: {MISCONCEPTION_TYPES}"
        )

    strategy_instruction = _STRATEGY_INSTRUCTIONS[strategy]
    type_context = _MISCONCEPTION_TYPE_CONTEXT[misconception_type]

    history_text = ""
    if conversation_history:
        turns = []
        for turn in conversation_history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            turns.append(f"{role}: {content}")
        history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(turns)

    prompt = f"""You are tutoring a student who made an error on a math problem.

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}

DIAGNOSED MISCONCEPTION: {misconception_label}
MISCONCEPTION TYPE: {misconception_type} — {type_context}
{history_text}

TEACHING STRATEGY TO USE:
{strategy_instruction}

Now respond to the student using this strategy. Remember:
- Do NOT reveal the correct answer directly.
- Stay focused on the specific misconception above.
- Keep your response concise and engaging.
- End with a question or a clear next step.

Your tutoring response:"""

    return prompt


def build_free_strategy_prompt(
    problem: str,
    student_answer: str,
    misconception_label: str,
    conversation_history: Optional[list[dict]] = None,
) -> str:
    """LLM Default: Let model freely choose its remediation approach.

    This is the baseline condition where the tutor is given full context
    but no strategy constraints, allowing it to choose its own approach.

    Args:
        problem: The original math problem.
        student_answer: The student's incorrect answer.
        misconception_label: Short label for the misconception.
        conversation_history: Prior turns as list of {"role": ..., "content": ...} dicts.

    Returns:
        Full prompt string for the tutor model.
    """
    history_text = ""
    if conversation_history:
        turns = []
        for turn in conversation_history:
            role = turn.get("role", "unknown").capitalize()
            content = turn.get("content", "")
            turns.append(f"{role}: {content}")
        history_text = "\n\nPREVIOUS CONVERSATION:\n" + "\n".join(turns)

    prompt = f"""You are tutoring a student who made an error on a math problem.

PROBLEM: {problem}

STUDENT'S ANSWER: {student_answer}

DIAGNOSED MISCONCEPTION: {misconception_label}
{history_text}

Respond to the student using whatever pedagogical approach you think is most effective.
Your goals:
- Help the student understand and correct their misconception.
- Do NOT reveal the correct answer directly — guide the student to discover it.
- Be encouraging and patient.
- End with a question or a clear next step.

Your tutoring response:"""

    return prompt


def build_tutor_system_prompt() -> str:
    """Return the system prompt for the tutor model."""
    return _TUTOR_SYSTEM_PROMPT


def build_verification_problem(original_problem: str, misconception_type: str) -> str:
    """Generate a prompt asking a similar problem to test if remediation was successful.

    The verification problem is presented to the simulated student after the
    remediation dialogue to assess whether their misconception was corrected.

    Args:
        original_problem: The problem used during the tutoring session.
        misconception_type: One of MISCONCEPTION_TYPES.

    Returns:
        Prompt string asking the student to solve a transfer/verification problem.
    """
    if misconception_type not in MISCONCEPTION_TYPES:
        raise ValueError(
            f"Unknown misconception_type '{misconception_type}'. Must be one of: {MISCONCEPTION_TYPES}"
        )

    type_emphasis = {
        "factual": "This tests whether you now recall the correct fact.",
        "procedural": "This tests whether you can now apply the correct procedure.",
        "conceptual": "This tests whether you now understand the underlying concept.",
    }

    return f"""Our tutoring session is complete. Now I want to see what you've learned.

Original problem context: {original_problem}

Please solve a similar problem that tests the same concept. {type_emphasis[misconception_type]}

Solve it carefully, showing your reasoning step by step. Your answer will tell us whether the tutoring was effective.

Your solution:"""
