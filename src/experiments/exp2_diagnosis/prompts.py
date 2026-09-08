"""Prompts for Experiment 2: Misconception Diagnosis.

Five conditions:
  A. Zero-shot
  B. Few-shot (n examples)
  C. Chain-of-thought + rubric
  D. RAG (retrieved misconceptions)

  E. Generate-retrieve-rerank (two-step: generate then rerank)

Each function returns a plain string prompt ready to be passed to model.generate().
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Taxonomy helpers
# ---------------------------------------------------------------------------

def _format_taxonomy(misconception_taxonomy: list[dict]) -> str:
    """Render a list of misconception entries as a numbered taxonomy block.

    Each entry must have keys: id, name, description, type, domain.
    """
    lines = []
    for entry in misconception_taxonomy:
        lines.append(
            f"[{entry['id']}] {entry['name']} ({entry['type']}, {entry['domain']})\n"
            f"    {entry['description']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Condition A: Zero-shot
# ---------------------------------------------------------------------------

def build_zero_shot_prompt(
    problem: str,
    student_answer: str,
    misconception_taxonomy: list[dict],
) -> str:
    """Condition A: Zero-shot — ask model to identify the misconception without examples.

    Args:
        problem: The math problem text shown to the student.
        student_answer: The student's (incorrect) answer.
        misconception_taxonomy: List of dicts with keys: id, name, description,
            type (factual/procedural/conceptual), domain
            (fractions/algebra/geometry/statistics/functions).

    Returns:
        Prompt string.
    """
    taxonomy_block = _format_taxonomy(misconception_taxonomy)
    return f"""You are an expert math education diagnostician. Your task is to identify the mathematical misconception behind a student's incorrect answer.

## Problem
{problem}

## Student's Answer
{student_answer}

## Misconception Taxonomy
{taxonomy_block}

## Task
Analyze the student's error and identify which misconception(s) from the taxonomy best explain it.

Return a ranked list of misconception IDs, from most likely to least likely, with one ID per line. Start with the single most probable misconception.

Format your response exactly as:
RANKED_MISCONCEPTIONS:
<id_1>
<id_2>
...

Then briefly explain why the top misconception fits (2-3 sentences).
"""


# ---------------------------------------------------------------------------
# Condition B: Few-shot
# ---------------------------------------------------------------------------

def build_few_shot_prompt(
    problem: str,
    student_answer: str,
    misconception_taxonomy: list[dict],
    examples: list[dict],
    n_shots: int = 5,
) -> str:
    """Condition B: Few-shot with n examples of misconception identification.

    Args:
        problem: The math problem text shown to the student.
        student_answer: The student's (incorrect) answer.
        misconception_taxonomy: List of misconception entries (id, name, description,
            type, domain).
        examples: List of example dicts with keys: problem, student_answer,
            correct_misconception_id, explanation.  At least n_shots items expected.
        n_shots: Number of examples to include (default 5).

    Returns:
        Prompt string.
    """
    taxonomy_block = _format_taxonomy(misconception_taxonomy)
    selected = examples[:n_shots]

    example_block_lines = []
    for i, ex in enumerate(selected, 1):
        example_block_lines.append(
            f"### Example {i}\n"
            f"Problem: {ex['problem']}\n"
            f"Student Answer: {ex['student_answer']}\n"
            f"Correct Misconception ID: {ex['correct_misconception_id']}\n"
            f"Explanation: {ex.get('explanation', '')}"
        )
    example_block = "\n\n".join(example_block_lines)

    return f"""You are an expert math education diagnostician. Your task is to identify the mathematical misconception behind a student's incorrect answer.

## Misconception Taxonomy
{taxonomy_block}

## Demonstration Examples
{example_block}

---

## New Case to Diagnose

### Problem
{problem}

### Student's Answer
{student_answer}

## Task
Following the same reasoning style as the examples above, analyze the student's error and return a ranked list of misconception IDs (most likely first), one per line.

Format your response exactly as:
RANKED_MISCONCEPTIONS:
<id_1>
<id_2>
...

Then briefly explain why the top misconception fits (2-3 sentences).
"""


# ---------------------------------------------------------------------------
# Condition C: Chain-of-thought + rubric
# ---------------------------------------------------------------------------

def build_cot_rubric_prompt(
    problem: str,
    student_answer: str,
    misconception_taxonomy: list[dict],
) -> str:
    """Condition C: Chain-of-thought + rubric prompting.

    Instructs the model to:
      1) Analyze the error step by step.
      2) Identify error type (factual / procedural / conceptual).
      3) Match to taxonomy entries.
      4) Return a ranked list of likely misconceptions.

    Args:
        problem: The math problem text shown to the student.
        student_answer: The student's (incorrect) answer.
        misconception_taxonomy: List of misconception entries (id, name, description,
            type, domain).

    Returns:
        Prompt string with explicit CoT scaffolding.
    """
    taxonomy_block = _format_taxonomy(misconception_taxonomy)
    return f"""You are an expert math education diagnostician. Use careful step-by-step reasoning to diagnose the student's misconception.

## Problem
{problem}

## Student's Answer
{student_answer}

## Misconception Taxonomy
{taxonomy_block}

## Diagnostic Rubric — follow ALL steps in order

STEP 1 — ERROR ANALYSIS
Describe exactly what the student did wrong. What operation or concept did they apply incorrectly? What would the correct procedure look like?

STEP 2 — ERROR TYPE CLASSIFICATION
Classify the error as one of:
  - FACTUAL: incorrect recall of a mathematical fact or definition
  - PROCEDURAL: incorrect application of a procedure or algorithm
  - CONCEPTUAL: flawed understanding of an underlying mathematical concept

Justify your classification in one sentence.

STEP 3 — TAXONOMY MATCHING
List the 5 most plausible misconception IDs from the taxonomy. For each, give one sentence explaining why it could explain the observed error. Focus on entries whose type matches the error type identified in Step 2.

STEP 4 — FINAL RANKING
Based on your analysis, produce a ranked list of misconception IDs from most to least probable.

Format your final ranking exactly as:
RANKED_MISCONCEPTIONS:
<id_1>
<id_2>
<id_3>
<id_4>
<id_5>
"""


# ---------------------------------------------------------------------------
# Condition D: RAG
# ---------------------------------------------------------------------------

def build_rag_prompt(
    problem: str,
    student_answer: str,
    retrieved_misconceptions: list[dict],
) -> str:
    """Condition D: RAG — given retrieved similar misconceptions, classify.

    The retrieval step is performed externally (e.g., via sentence-transformers).
    This prompt receives the top-k retrieved misconception entries and asks the
    model to rank and select among them.

    Args:
        problem: The math problem text shown to the student.
        student_answer: The student's (incorrect) answer.
        retrieved_misconceptions: Top-k retrieved misconception dicts (id, name,
            description, type, domain, similarity_score).

    Returns:
        Prompt string.
    """
    retrieved_lines = []
    for entry in retrieved_misconceptions:
        score_str = (
            f"  similarity={entry['similarity_score']:.3f}"
            if "similarity_score" in entry
            else ""
        )
        retrieved_lines.append(
            f"[{entry['id']}] {entry['name']} ({entry['type']}, {entry['domain']}){score_str}\n"
            f"    {entry['description']}"
        )
    retrieved_block = "\n".join(retrieved_lines)

    return f"""You are an expert math education diagnostician. You have been given a set of candidate misconceptions retrieved by semantic similarity to help you diagnose a student's error.

## Problem
{problem}

## Student's Answer
{student_answer}

## Retrieved Candidate Misconceptions (ranked by similarity)
{retrieved_block}

## Task
Examine each candidate misconception and determine which best explain the student's error. Re-rank the candidates from most likely to least likely based on how well each explains the observed error — do NOT simply preserve the retrieval order.

Format your response exactly as:
RANKED_MISCONCEPTIONS:
<id_1>
<id_2>
...

Then explain in 2-3 sentences why the top-ranked misconception is the best fit.
"""


# ---------------------------------------------------------------------------
# Condition E: Generate-retrieve-rerank (two-step)
# ---------------------------------------------------------------------------

def build_gen_retrieve_rerank_prompt(
    problem: str,
    student_answer: str,
) -> str:
    """Condition E — Step 1: Generate a candidate diagnosis description.

    The generated free-text diagnosis is then used as a query to retrieve
    similar misconceptions from the taxonomy (retrieval step happens externally).
    The retrieved candidates are then passed to build_rerank_prompt().

    Args:
        problem: The math problem text shown to the student.
        student_answer: The student's (incorrect) answer.

    Returns:
        Prompt string for Step 1 (diagnosis generation).
    """
    return f"""You are an expert math education diagnostician. A student has made an error on a math problem. Your task is to describe, in precise mathematical terms, the misconception or error pattern the student appears to hold.

## Problem
{problem}

## Student's Answer
{student_answer}

## Task
Write a clear, concise description (2-4 sentences) of:
1. What the student did wrong.
2. What underlying misunderstanding or incorrect rule the student seems to be applying.
3. The type of error (factual, procedural, or conceptual).

Do NOT reference any external taxonomy. Write only the diagnosis description.

DIAGNOSIS:
"""


def build_rerank_prompt(
    problem: str,
    student_answer: str,
    candidate_diagnosis: str,
    retrieved_misconceptions: list[dict],
) -> str:
    """Condition E — Step 3: Rerank retrieved misconceptions given the generated diagnosis.

    Combines the model-generated diagnosis (from build_gen_retrieve_rerank_prompt)
    with retrieved taxonomy entries to produce a final ranked list.

    Args:
        problem: The math problem text shown to the student.
        student_answer: The student's (incorrect) answer.
        candidate_diagnosis: Free-text diagnosis generated in Step 1.
        retrieved_misconceptions: Top-k misconception dicts retrieved using the
            candidate diagnosis as a query (id, name, description, type, domain,
            similarity_score).

    Returns:
        Prompt string for Step 3 (reranking).
    """
    retrieved_lines = []
    for entry in retrieved_misconceptions:
        score_str = (
            f"  similarity={entry['similarity_score']:.3f}"
            if "similarity_score" in entry
            else ""
        )
        retrieved_lines.append(
            f"[{entry['id']}] {entry['name']} ({entry['type']}, {entry['domain']}){score_str}\n"
            f"    {entry['description']}"
        )
    retrieved_block = "\n".join(retrieved_lines)

    return f"""You are an expert math education diagnostician. You previously generated a diagnosis of a student error. Now you must match that diagnosis to the most relevant entries in a curated misconception taxonomy.

## Problem
{problem}

## Student's Answer
{student_answer}

## Your Previously Generated Diagnosis
{candidate_diagnosis}

## Retrieved Candidate Misconceptions
{retrieved_block}

## Task
Using your diagnosis as a guide, re-rank the retrieved misconceptions from most to least likely to explain the student's error. Consider both how well the description matches and whether the error type aligns.

Format your response exactly as:
RANKED_MISCONCEPTIONS:
<id_1>
<id_2>
...

Then explain in 2-3 sentences why the top-ranked misconception is the best fit.
"""
