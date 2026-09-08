#!/usr/bin/env python3
"""Comprehensive quality screening for ALL 2,250 exp3v2 dialogues.

Screens every dialogue for 11 quality issues and classifies each as
clean / minor_issues / dirty.  Outputs a JSON report and prints a
summary table to stdout.

Usage:
    python scripts/screen_exp3v2_for_human_eval.py
"""

import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "results" / "exp3v2"
OUTPUT_PATH = RESULTS_DIR / "quality_screen_results.json"

MODEL_DIRS = {
    "qwen-2.5-7b": "20260325_103441_qwen-2.5-7b",
    "mistral-7b": "20260325_103441_mistral-7b",
    "llama-3.1-8b": "20260325_103441_llama-3.1-8b",
    "phi-4": "20260326_010332_phi-4",
    "qwen3.5-9b": "20260326_010332_qwen3.5-9b",
}

# ── Role-confusion patterns (from semantic_quality_screen.py) ─────────

TUTOR_PHRASES = [
    # Praise patterns
    r"(?:you'?re|you are) (?:on the right track|doing (?:great|well|a great job|an excellent job|fantastic))",
    r"(?:great|good|excellent|fantastic|wonderful|amazing) (?:job|work|observation|question|thinking|progress|insight|point)",
    r"(?:i'?m )?(?:proud of you|glad you)",
    r"that'?s (?:a )?(?:great|good|excellent|fantastic|wonderful) (?:question|observation|point|start|answer|insight)",
    r"well done",
    r"keep (?:up the|it up|going)",
    r"(?:exactly|absolutely|precisely)[!.]?\s*(?:you'?ve|you have|you got|you correctly|you found|that'?s|since)",
    # Direct affirmation of other's correctness
    r"(?:you'?re|you are) (?:absolutely |completely |totally )?(?:right|correct)",
    r"(?:you'?ve|you have|you) got (?:it|this|the (?:right|correct|hang))",
    # Pedagogical prompts
    r"does (?:this|that) (?:\w+ )?(?:help|make sense|clarify)",
    r"(?:can|could) you (?:think of|try|explain)",
    r"(?:let'?s|shall we) (?:try|explore|think about|consider|look at|move on|apply|practice|solve|work through|write)",
    r"would you like to try",
    r"(?:here'?s|let me give you) (?:a|an|another) (?:example|hint|clue)",
    # Comfort/confidence checks
    r"(?:do|how do) you feel (?:more )?(?:comfortable|confident|about)",
    r"(?:does|did) (?:this|that|this process) (?:\w+ )?(?:click|resonate|land|feel comfortable|feel right|make sense)",
    r"(?:are|were) you (?:comfortable|confident) (?:with|using|applying)",
    r"(?:is|was) (?:this|that) (?:clear|clearer) (?:now|to you)",
    # Closing/summarizing patterns
    r"(?:if you have|feel free to) (?:any )?(?:more )?questions",
    r"don'?t hesitate to ask",
    r"(?:i hope|hopefully) (?:this|that) (?:helps|clarifies|makes sense)",
    r"does (?:this|that) (?:help )?(?:solidify|reinforce|clarify|deepen) your understanding",
    r"to (?:summarize|recap|sum up)",
    # Encouragement/validation
    r"(?:this|that) (?:is|was) (?:a )?(?:really |very )?(?:important|key|crucial|valuable) (?:step|insight|observation|realization)",
    r"(?:i'?m |i am )?glad (?:we|you) (?:could|can|were able to) (?:clear|work|figure)",
    r"(?:you'?ve|you have) (?:correctly|successfully|already) (?:identified|found|shown|demonstrated|worked)",
    # Pedagogical offers
    r"(?:would|do) you (?:like|want) (?:to|me to) (?:try|practice|solve|see|work|do)",
    r"(?:shall|should) we (?:try|move|continue|go|practice)",
    r"(?:it'?s|it is) (?:great|good|wonderful|fantastic) that you",
]

TUTOR_PHRASE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in TUTOR_PHRASES]

# ── N-gram helpers ────────────────────────────────────────────────────


def _ngrams(text: str, n: int = 3) -> list[tuple[str, ...]]:
    """Extract character-cleaned word n-grams from text."""
    words = re.findall(r"\w+", text.lower())
    if len(words) < n:
        return []
    return [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]


def ngram_overlap(text_a: str, text_b: str, n: int = 3) -> float:
    """Fraction of n-grams in text_a that also appear in text_b.

    Returns 0.0 if either text produces no n-grams.
    """
    grams_a = _ngrams(text_a, n)
    grams_b = _ngrams(text_b, n)
    if not grams_a or not grams_b:
        return 0.0
    set_b = set(grams_b)
    matches = sum(1 for g in grams_a if g in set_b)
    return matches / len(grams_a)


# ── Individual quality checks ────────────────────────────────────────


def check_error_field(result: dict) -> bool:
    """Flag 1: error field is not None."""
    return result.get("error") is not None


def check_empty_verification(result: dict) -> bool:
    """Flag 2: verification_answer is empty or very short (<20 chars)."""
    va = result.get("verification_answer") or ""
    return len(va.strip()) < 20


def check_truncated_verification(result: dict) -> bool:
    """Flag 3: verification_answer ends mid-sentence."""
    va = (result.get("verification_answer") or "").strip()
    if not va:
        return False  # already caught by empty_verification
    # Ends with "..."
    if va.endswith("..."):
        return True
    # No terminal punctuation at end
    if va and va[-1] not in ".!?":
        return True
    return False


def check_role_confusion(result: dict) -> tuple[str | None, int]:
    """Flag 4: Student turns contain tutor-like phrases.

    Returns (flag_name | None, match_count).
    flag_name is 'severe_role_confusion' (>=3), 'moderate_role_confusion' (1-2), or None.
    """
    student_turns = result.get("student_responses") or []
    total_matches = 0
    for turn in student_turns:
        for pattern in TUTOR_PHRASE_PATTERNS:
            total_matches += len(pattern.findall(turn))
    if total_matches >= 3:
        return "severe_role_confusion", total_matches
    if total_matches >= 1:
        return "moderate_role_confusion", total_matches
    return None, 0


def check_answer_revealed(result: dict) -> bool:
    """Flag 5: answer_revealed is True."""
    return bool(result.get("answer_revealed"))


def check_student_echo(result: dict) -> bool:
    """Flag 6: Any student turn has >60% 3-gram overlap with preceding tutor turn."""
    tutor_turns = result.get("tutor_responses") or []
    student_turns = result.get("student_responses") or []
    for i, sturn in enumerate(student_turns):
        if i < len(tutor_turns):
            if ngram_overlap(sturn, tutor_turns[i], n=3) > 0.60:
                return True
    return False


def check_tutor_repetition(result: dict) -> bool:
    """Flag 7: Any two consecutive tutor turns have >70% 3-gram overlap."""
    tutor_turns = result.get("tutor_responses") or []
    for i in range(1, len(tutor_turns)):
        if ngram_overlap(tutor_turns[i], tutor_turns[i - 1], n=3) > 0.70:
            return True
    return False


def check_very_short_dialogue(result: dict) -> bool:
    """Flag 8: num_turns < 3."""
    return (result.get("num_turns") or 0) < 3


def check_very_long_dialogue(result: dict) -> bool:
    """Flag 9: num_turns > 8."""
    return (result.get("num_turns") or 0) > 8


def check_empty_turns(result: dict) -> bool:
    """Flag 10: Any tutor or student turn is empty or whitespace-only."""
    for turn in (result.get("tutor_responses") or []):
        if not turn or not turn.strip():
            return True
    for turn in (result.get("student_responses") or []):
        if not turn or not turn.strip():
            return True
    return False


def check_duplicated_student_turns(result: dict) -> bool:
    """Flag 11: Any two student turns are >90% similar (3-gram overlap)."""
    student_turns = result.get("student_responses") or []
    for i in range(len(student_turns)):
        for j in range(i + 1, len(student_turns)):
            if ngram_overlap(student_turns[i], student_turns[j], n=3) > 0.90:
                return True
    return False


# ── Classification ────────────────────────────────────────────────────

MINOR_ONLY_FLAGS = {"moderate_role_confusion", "very_long_dialogue"}


def classify(flags: list[str]) -> str:
    """Classify a dialogue based on its flags."""
    if not flags:
        return "clean"
    if all(f in MINOR_ONLY_FLAGS for f in flags):
        return "minor_issues"
    return "dirty"


# ── Screen one dialogue ──────────────────────────────────────────────


def screen_dialogue(result: dict) -> tuple[list[str], dict]:
    """Run all quality checks on one dialogue result.

    Returns (flags, flag_details).
    """
    flags: list[str] = []
    details: dict = {}

    # 1. error_field
    if check_error_field(result):
        flags.append("error_field")

    # 2. empty_verification
    if check_empty_verification(result):
        flags.append("empty_verification")

    # 3. truncated_verification
    if check_truncated_verification(result):
        flags.append("truncated_verification")

    # 4. role_confusion
    rc_flag, rc_count = check_role_confusion(result)
    if rc_flag:
        flags.append(rc_flag)
        details["role_confusion_count"] = rc_count

    # 5. answer_revealed
    if check_answer_revealed(result):
        flags.append("answer_revealed")

    # 6. student_echo
    if check_student_echo(result):
        flags.append("student_echo")

    # 7. tutor_repetition
    if check_tutor_repetition(result):
        flags.append("tutor_repetition")

    # 8. very_short_dialogue
    if check_very_short_dialogue(result):
        flags.append("very_short_dialogue")

    # 9. very_long_dialogue
    if check_very_long_dialogue(result):
        flags.append("very_long_dialogue")

    # 10. empty_turns
    if check_empty_turns(result):
        flags.append("empty_turns")

    # 11. duplicated_student_turns
    if check_duplicated_student_turns(result):
        flags.append("duplicated_student_turns")

    return flags, details


# ── Main ──────────────────────────────────────────────────────────────


def main() -> None:
    all_results: list[dict] = []
    flag_counter: Counter = Counter()
    model_summary: dict[str, dict] = {}

    for model_name, dir_name in MODEL_DIRS.items():
        model_dir = RESULTS_DIR / dir_name
        if not model_dir.is_dir():
            print(f"WARNING: missing directory {model_dir}", file=sys.stderr)
            continue

        m_stats = {"total": 0, "clean": 0, "minor_issues": 0, "dirty": 0}

        for cell_path in sorted(model_dir.glob("cell_*.json")):
            with open(cell_path) as f:
                cell_data = json.load(f)

            results_list = cell_data.get("results", [])
            for idx, result in enumerate(results_list):
                flags, details = screen_dialogue(result)
                classification = classify(flags)

                for fl in flags:
                    flag_counter[fl] += 1

                m_stats["total"] += 1
                m_stats[classification] += 1

                all_results.append(
                    {
                        "model": model_name,
                        "strategy": result.get("strategy", ""),
                        "misconception_type": result.get("misconception_type", ""),
                        "scenario_id": result.get("scenario_id", ""),
                        "cell_file": cell_path.name,
                        "idx_in_cell": idx,
                        "tier1_correct": result.get("tier1_correct", False),
                        "tier2_correct": result.get("tier2_correct", False),
                        "flags": flags,
                        "classification": classification,
                        "flag_details": details,
                    }
                )

        model_summary[model_name] = m_stats

    # ── Aggregate counts ──────────────────────────────────────────────
    total = len(all_results)
    n_clean = sum(1 for r in all_results if r["classification"] == "clean")
    n_minor = sum(1 for r in all_results if r["classification"] == "minor_issues")
    n_dirty = sum(1 for r in all_results if r["classification"] == "dirty")

    output = {
        "metadata": {
            "total": total,
            "clean": n_clean,
            "minor_issues": n_minor,
            "dirty": n_dirty,
            "timestamp": datetime.now().isoformat(),
        },
        "summary_by_model": model_summary,
        "summary_by_flag": dict(flag_counter.most_common()),
        "results": all_results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(output, f, indent=2)

    # ── Print summary table ───────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  Exp3v2 Quality Screening — {total} dialogues")
    print(f"{'='*72}\n")

    # Per-model table
    header = f"{'Model':<18} {'Total':>6} {'Clean':>6} {'Minor':>6} {'Dirty':>6} {'Clean%':>7}"
    print(header)
    print("-" * len(header))
    for model_name in MODEL_DIRS:
        ms = model_summary.get(model_name, {})
        t = ms.get("total", 0)
        c = ms.get("clean", 0)
        mi = ms.get("minor_issues", 0)
        d = ms.get("dirty", 0)
        pct = f"{100*c/t:.1f}%" if t else "N/A"
        print(f"{model_name:<18} {t:>6} {c:>6} {mi:>6} {d:>6} {pct:>7}")
    print("-" * len(header))
    pct_all = f"{100*n_clean/total:.1f}%" if total else "N/A"
    print(f"{'TOTAL':<18} {total:>6} {n_clean:>6} {n_minor:>6} {n_dirty:>6} {pct_all:>7}")

    # Flag frequency table
    print(f"\n{'Flag':<30} {'Count':>6} {'%':>7}")
    print("-" * 45)
    for flag, count in flag_counter.most_common():
        print(f"{flag:<30} {count:>6} {100*count/total:>6.1f}%")

    # Classification breakdown
    print(f"\nClassification: {n_clean} clean, {n_minor} minor, {n_dirty} dirty")
    print(f"Saved to: {OUTPUT_PATH}\n")


if __name__ == "__main__":
    main()
