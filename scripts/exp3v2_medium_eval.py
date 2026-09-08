#!/usr/bin/env python3
"""
Medium-tier numerical evaluation for exp3v2 remediation dialogues.

Three-tier spectrum:
  Tier 1 (Strict):  word-boundary heuristic match (~5% RSR)
  Medium (this):    numerical/keyword extraction + comparison
  Tier 2 (Lenient): GPT-5-mini LLM-as-judge (~55% RSR)

Reads cell_*.json results from each model directory, loads gold answers
from the scenario bank, extracts answers from student free-form text,
and compares using type-appropriate logic (numeric tolerance, keyword
match, expression normalization).

Usage:
    python scripts/exp3v2_medium_eval.py
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "exp3v2"
SCENARIO_BANK = PROJECT_ROOT / "data" / "exp3v2_scenario_bank.json"
OUTPUT_JSON = RESULTS_DIR / "three_tier_analysis.json"

MODEL_DIRS: dict[str, str] = {
    "20260325_103441_qwen-2.5-7b": "Qwen-2.5-7B",
    "20260325_103441_llama-3.1-8b": "Llama-3.1-8B",
    "20260325_103441_mistral-7b": "Mistral-7B",
    "20260326_010332_phi-4": "Phi-4",
    "20260326_010332_qwen3.5-9b": "Qwen3.5-9B",
}

# Numeric comparison tolerances
REL_TOL = 0.01   # 1% relative
ABS_TOL = 0.01   # absolute

# ---------------------------------------------------------------------------
# Gold answer parsing
# ---------------------------------------------------------------------------

# Keywords that should be matched as text (case-insensitive)
TEXT_KEYWORDS = [
    "true", "false", "yes", "no",
    "converges", "diverges",
    "neither", "sometimes true",
    "uncountable", "countable",
    "independent", "dependent",
    "injective", "surjective",
    "continuous", "not continuous",
    "differentiable", "not differentiable",
    "uniform", "not uniform",
    "unbounded", "bounded",
]


def _clean_numeric_str(s: str) -> str:
    """Remove $, commas, leading/trailing whitespace from a numeric string."""
    s = s.strip()
    s = s.replace("$", "").replace(",", "").replace(" ", "")
    return s


def _try_parse_number(s: str) -> Optional[float]:
    """Try to parse a string as a number (int, float, fraction, percentage)."""
    s = _clean_numeric_str(s)
    if not s:
        return None

    # Percentage: "40%" -> 0.40? No -- keep as 40 for comparison since
    # student might write "40%" or "40 percent" or just "40".
    # We store the raw number (40) for percentages.
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except ValueError:
            pass

    # Mixed number: "1 2/3" -> 1.6667
    mixed_match = re.match(r'^(-?\d+)\s+(\d+)/(\d+)$', s)
    if mixed_match:
        whole = int(mixed_match.group(1))
        num = int(mixed_match.group(2))
        den = int(mixed_match.group(3))
        if den != 0:
            sign = -1 if whole < 0 else 1
            return whole + sign * num / den
        return None

    # Fraction: "9/4" -> 2.25
    frac_match = re.match(r'^(-?\d+)/(\d+)$', s)
    if frac_match:
        num = int(frac_match.group(1))
        den = int(frac_match.group(2))
        if den != 0:
            return num / den
        return None

    # Plain int or float
    try:
        return float(s)
    except ValueError:
        pass

    return None


def _normalize_expr(s: str) -> str:
    """Normalize a math expression string for comparison.

    Strips whitespace, lowercases, removes $ and certain LaTeX commands,
    normalizes common variations like ^ vs **, etc.
    """
    s = s.strip().lower()
    # Remove LaTeX wrappers
    s = s.replace("\\(", "").replace("\\)", "")
    s = s.replace("\\[", "").replace("\\]", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = s.replace("\\cdot", "*").replace("\\times", "*")
    s = s.replace("\\div", "/")
    s = s.replace("\\pi", "pi")
    s = s.replace("\\infty", "infty")
    s = s.replace("+\\", "+").replace("-\\", "-")
    # Remove spaces
    s = re.sub(r'\s+', '', s)
    # Normalize ** and ^
    s = s.replace("**", "^")
    # Remove $ signs
    s = s.replace("$", "")
    return s


class GoldAnswer:
    """Parsed representation of a gold answer."""

    def __init__(self, raw: str, scenario_id: str):
        self.raw = raw
        self.scenario_id = scenario_id
        self.answer_type = "expression"  # default fallback
        self.numeric_value: Optional[float] = None
        self.keywords: list[str] = []
        self.is_percentage = False
        self.is_multi_part = False
        self.parts: list[dict[str, Any]] = []  # for multi-part answers
        self.normalized_expr = ""

        self._parse()

    def _parse(self) -> None:
        raw = self.raw.strip()

        # ---- Multi-part answers ----
        # Detect multi-part: semicolons, or "True, False, True, False" pattern,
        # or "Child tickets = 159, Student tickets = 204, Adult tickets = 237"
        if self._try_parse_multi_part(raw):
            return

        # ---- Pure keyword answers ----
        if self._try_parse_keyword(raw):
            return

        # ---- Numeric answers ----
        if self._try_parse_numeric(raw):
            return

        # ---- Expression fallback ----
        self.answer_type = "expression"
        self.normalized_expr = _normalize_expr(raw)

    def _try_parse_keyword(self, raw: str) -> bool:
        """Check if answer is a simple keyword like True, No, Converges."""
        raw_lower = raw.lower().strip().rstrip(".")
        for kw in TEXT_KEYWORDS:
            if raw_lower == kw:
                self.answer_type = "keyword"
                self.keywords = [kw]
                return True
        return False

    def _try_parse_numeric(self, raw: str) -> bool:
        """Check if the answer is a single number."""
        val = _try_parse_number(raw)
        if val is not None:
            self.answer_type = "numeric"
            self.numeric_value = val
            self.is_percentage = raw.strip().endswith("%")
            return True
        return False

    def _try_parse_multi_part(self, raw: str) -> bool:
        """Detect and parse multi-part gold answers."""
        # Pattern: "True, False, True, False"
        tf_match = re.match(
            r'^(True|False)(?:\s*,\s*(True|False))+$', raw, re.IGNORECASE
        )
        if tf_match:
            parts_str = [p.strip() for p in raw.split(",")]
            self.answer_type = "multi_keyword"
            self.is_multi_part = True
            self.keywords = [p.strip().lower() for p in parts_str]
            return True

        # Pattern: "Child tickets = 159, Student tickets = 204, ..."
        # or "Mean = 7.5 and median = 5.5"
        # or "Researcher 1: 4.935; Researcher 2: 7.728; ..."
        kv_parts = re.split(r'[;,]\s*', raw)
        if len(kv_parts) >= 2:
            parsed_parts = []
            for part in kv_parts:
                part = part.strip()
                if not part:
                    continue
                # Try "label = value" or "label: value"
                kv_match = re.match(r'^(.+?)\s*[=:]\s*(.+)$', part)
                if kv_match:
                    label = kv_match.group(1).strip().lower()
                    val_str = kv_match.group(2).strip()
                    num = _try_parse_number(val_str)
                    parsed_parts.append({
                        "label": label,
                        "raw": val_str,
                        "numeric": num,
                    })
                else:
                    # Not key=value; might still be multi-part text
                    parsed_parts.append({
                        "label": "",
                        "raw": part,
                        "numeric": _try_parse_number(part),
                    })

            # Only treat as multi-part if we got at least 2 meaningful parts
            # AND at least one has a numeric value or recognizable label
            has_content = sum(
                1 for p in parsed_parts
                if p["numeric"] is not None or p["label"]
            )
            if has_content >= 2:
                self.answer_type = "multi_part"
                self.is_multi_part = True
                self.parts = parsed_parts
                return True

        # "X and Y" pattern: "Mean = 7.5 and median = 5.5"
        and_parts = re.split(r'\s+and\s+', raw, flags=re.IGNORECASE)
        if len(and_parts) >= 2:
            parsed_parts = []
            for part in and_parts:
                part = part.strip()
                kv_match = re.match(r'^(.+?)\s*[=:]\s*(.+)$', part)
                if kv_match:
                    label = kv_match.group(1).strip().lower()
                    val_str = kv_match.group(2).strip()
                    num = _try_parse_number(val_str)
                    parsed_parts.append({
                        "label": label,
                        "raw": val_str,
                        "numeric": num,
                    })
            if len(parsed_parts) >= 2:
                self.answer_type = "multi_part"
                self.is_multi_part = True
                self.parts = parsed_parts
                return True

        return False


# ---------------------------------------------------------------------------
# Student answer extraction
# ---------------------------------------------------------------------------

# Patterns for explicit "final answer" markers (ordered by priority)
FINAL_ANSWER_PATTERNS = [
    # "Final answer: X" or "Final answer is X" or "My final answer: X"
    re.compile(
        r'(?:my\s+)?final\s+answers?\s*(?:is|are|:)\s*(.+)',
        re.IGNORECASE,
    ),
    # "The answer is X"
    re.compile(
        r'(?:the|my)\s+answer\s+(?:is|are|:)\s*(.+)',
        re.IGNORECASE,
    ),
    # "Answer: X"
    re.compile(
        r'^answer\s*:\s*(.+)',
        re.IGNORECASE | re.MULTILINE,
    ),
    # "So X is the answer" or "Therefore, X"
    re.compile(
        r'(?:so|therefore|thus|hence)[,]?\s+(?:the\s+)?(?:answer\s+is\s+)?(.+?)(?:\.|$)',
        re.IGNORECASE,
    ),
]

# Patterns for "= X" at end of a line (captures the RHS)
EQUALS_PATTERN = re.compile(r'=\s*(.+?)(?:\s*$)', re.MULTILINE)


def _extract_last_number(text: str) -> Optional[tuple[str, float]]:
    """Extract the last number from free-form text.

    Returns (raw_match, float_value) or None.
    """
    # Match integers, decimals, negatives, fractions
    # Also handle things like "$7", "50%"
    candidates = list(re.finditer(
        r'(?<![a-zA-Z])(-?\d+(?:\.\d+)?(?:/\d+)?%?)\s*(?:\.|,|;|\s|$)',
        text,
    ))
    if not candidates:
        # Try simpler: any number
        candidates = list(re.finditer(r'(-?\d+(?:\.\d+)?(?:/\d+)?%?)', text))

    for match in reversed(candidates):
        raw = match.group(1).strip()
        val = _try_parse_number(raw)
        if val is not None:
            return (raw, val)
    return None


def _extract_keyword_from_text(
    text: str, keywords: list[str]
) -> Optional[tuple[str, str]]:
    """Check if any of the keywords appear in the student text.

    Returns (extraction_method, matched_keyword) or None.
    """
    text_lower = text.lower()
    for kw in keywords:
        # Use word boundary to avoid partial matches
        if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
            return ("keyword_match", kw)
    return None


def extract_student_answer(
    student_text: str, gold: GoldAnswer
) -> tuple[Optional[str], str, str]:
    """Extract the student's answer from free-form verification text.

    Returns:
        (extracted_value, extraction_method, raw_extracted_text)

    extracted_value: the cleaned/parsed answer string, or None if extraction failed
    extraction_method: how we extracted it
    raw_extracted_text: the raw text we pulled from the student response
    """
    if not student_text or not student_text.strip():
        return (None, "empty_response", "")

    text = student_text.strip()

    # ---- Strategy depends on gold answer type ----

    if gold.answer_type == "keyword":
        # For simple keywords (True/False/Yes/No/Converges/etc.)
        return _extract_keyword_answer(text, gold)

    if gold.answer_type == "multi_keyword":
        # For "True, False, True, False" style
        return _extract_multi_keyword_answer(text, gold)

    if gold.answer_type == "multi_part":
        # For multi-part numeric answers
        return _extract_multi_part_answer(text, gold)

    if gold.answer_type == "numeric":
        return _extract_numeric_answer(text, gold)

    if gold.answer_type == "expression":
        return _extract_expression_answer(text, gold)

    return (None, "unknown_gold_type", "")


def _extract_keyword_answer(
    text: str, gold: GoldAnswer
) -> tuple[Optional[str], str, str]:
    """Extract keyword answer from student text."""
    kw = gold.keywords[0]

    # Check "final answer" patterns first
    for pat in FINAL_ANSWER_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            answer_region = matches[-1].group(1).strip()
            if re.search(r'\b' + re.escape(kw) + r'\b', answer_region.lower()):
                return (kw, "final_answer_keyword", answer_region)

    # Check full text for keyword presence
    result = _extract_keyword_from_text(text, gold.keywords)
    if result:
        return (result[1], result[0], text[:200])

    return (None, "keyword_not_found", "")


def _extract_multi_keyword_answer(
    text: str, gold: GoldAnswer
) -> tuple[Optional[str], str, str]:
    """Extract multi-keyword answer like 'True, False, True, False'."""
    expected = gold.keywords  # e.g., ["true", "false", "true", "false"]

    # Look for explicit "final answers:" line
    for pat in FINAL_ANSWER_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            answer_region = matches[-1].group(1).strip()
            found = _extract_tf_sequence(answer_region, len(expected))
            if found is not None:
                return (
                    ", ".join(found),
                    "final_answer_multi_keyword",
                    answer_region,
                )

    # Scan the full text for a sequence of True/False
    found = _extract_tf_sequence(text, len(expected))
    if found is not None:
        return (", ".join(found), "text_scan_multi_keyword", text[:300])

    # Fallback: look for numbered items "(1) True (2) False ..."
    numbered = re.findall(
        r'\(\d+\)\s*(true|false)', text, re.IGNORECASE
    )
    if len(numbered) == len(expected):
        found = [x.lower() for x in numbered]
        return (", ".join(found), "numbered_multi_keyword", text[:300])

    return (None, "multi_keyword_not_found", "")


def _extract_tf_sequence(text: str, expected_count: int) -> Optional[list[str]]:
    """Extract a comma/space separated sequence of True/False values."""
    # Try comma-separated first
    tf_matches = re.findall(r'\b(true|false)\b', text, re.IGNORECASE)
    if len(tf_matches) >= expected_count:
        # Take the last `expected_count` occurrences (likely the final answer)
        result = [x.lower() for x in tf_matches[-expected_count:]]
        return result
    return None


def _extract_numeric_answer(
    text: str, gold: GoldAnswer
) -> tuple[Optional[str], str, str]:
    """Extract a numeric answer from student text."""

    # 1. Try explicit "final answer" markers
    for pat in FINAL_ANSWER_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            answer_region = matches[-1].group(1).strip()
            # Clean up trailing punctuation and words
            answer_region = re.sub(
                r'[\.\!\?]+$', '', answer_region
            ).strip()
            # Try to extract a number from the answer region
            val = _try_parse_number(answer_region)
            if val is not None:
                return (str(val), "final_answer_numeric", answer_region)
            # The answer region might have extra text: "50 apples"
            num_match = re.match(
                r'(-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?)', answer_region
            )
            if num_match:
                raw = num_match.group(1)
                val = _try_parse_number(raw)
                if val is not None:
                    return (str(val), "final_answer_numeric_prefix", raw)
            # Try extracting last number from the answer region
            result = _extract_last_number(answer_region)
            if result:
                return (str(result[1]), "final_answer_last_number", result[0])

    # 2. Try "= X" pattern (last occurrence, common in math work)
    eq_matches = list(EQUALS_PATTERN.finditer(text))
    if eq_matches:
        # Take the last "= X" that yields a number
        for match in reversed(eq_matches):
            rhs = match.group(1).strip()
            rhs = re.sub(r'[\.\!\?]+$', '', rhs).strip()
            val = _try_parse_number(rhs)
            if val is not None:
                return (str(val), "equals_rhs", rhs)
            # "= 50 apples" or "= $7"
            num_match = re.match(
                r'(-?\$?\d[\d,]*(?:\.\d+)?(?:/\d+)?%?)', rhs
            )
            if num_match:
                raw = num_match.group(1)
                val = _try_parse_number(raw)
                if val is not None:
                    return (str(val), "equals_rhs_prefix", raw)

    # 3. Fallback: last number in the text
    result = _extract_last_number(text)
    if result:
        return (str(result[1]), "last_number", result[0])

    return (None, "numeric_not_found", "")


def _extract_multi_part_answer(
    text: str, gold: GoldAnswer
) -> tuple[Optional[str], str, str]:
    """Extract a multi-part answer from student text.

    For multi-part answers, we try to find each part's value in the
    student text and return a combined representation.
    """
    # Strategy: for each gold part, look for its label or numeric value
    found_parts = []
    for part in gold.parts:
        label = part.get("label", "")
        expected_num = part.get("numeric")
        expected_raw = part.get("raw", "")

        # Search for "label = value" or "label: value" or "label is value"
        if label:
            # Escape label for regex, allow flexible matching
            label_words = re.escape(label)
            pat = re.compile(
                label_words + r'\s*[=:]\s*([^\n,;]+)',
                re.IGNORECASE,
            )
            match = pat.search(text)
            if match:
                val_str = match.group(1).strip()
                num = _try_parse_number(val_str)
                found_parts.append({
                    "label": label,
                    "raw": val_str,
                    "numeric": num,
                })
                continue

        # If we have a numeric expectation, look for that number in text
        if expected_num is not None:
            # Check if the number appears in the text
            nums_in_text = re.findall(
                r'(-?\d+(?:\.\d+)?(?:/\d+)?)', text
            )
            for n_str in nums_in_text:
                val = _try_parse_number(n_str)
                if val is not None and _numbers_close(val, expected_num):
                    found_parts.append({
                        "label": label,
                        "raw": n_str,
                        "numeric": val,
                    })
                    break
            else:
                found_parts.append({
                    "label": label,
                    "raw": "",
                    "numeric": None,
                })
        else:
            # Text part -- check if key terms appear
            if expected_raw and expected_raw.lower() in text.lower():
                found_parts.append({
                    "label": label,
                    "raw": expected_raw,
                    "numeric": None,
                })
            else:
                found_parts.append({
                    "label": label,
                    "raw": "",
                    "numeric": None,
                })

    # Build combined extracted answer
    extracted_strs = []
    for fp in found_parts:
        if fp["raw"]:
            s = f"{fp['label']}={fp['raw']}" if fp["label"] else fp["raw"]
            extracted_strs.append(s)
    if extracted_strs:
        combined = "; ".join(extracted_strs)
        return (combined, "multi_part_extraction", combined)

    return (None, "multi_part_not_found", "")


def _extract_expression_answer(
    text: str, gold: GoldAnswer
) -> tuple[Optional[str], str, str]:
    """Extract an expression answer from student text.

    For complex expressions (like "4y^2 - 28y + 49"), we look for
    explicit answer markers or the last substantial expression.
    """
    gold_normalized = gold.normalized_expr

    # 1. Check explicit answer markers
    for pat in FINAL_ANSWER_PATTERNS:
        matches = list(pat.finditer(text))
        if matches:
            answer_region = matches[-1].group(1).strip()
            # Clean trailing punctuation
            answer_region = re.sub(r'[\.\!\?]+$', '', answer_region).strip()
            normalized = _normalize_expr(answer_region)
            if normalized:
                return (normalized, "final_answer_expression", answer_region)

    # 2. Try "= X" (last occurrence with substantial content)
    eq_matches = list(EQUALS_PATTERN.finditer(text))
    if eq_matches:
        for match in reversed(eq_matches):
            rhs = match.group(1).strip()
            rhs = re.sub(r'[\.\!\?]+$', '', rhs).strip()
            normalized = _normalize_expr(rhs)
            # Only accept if it looks like a math expression (has alphanumeric)
            if normalized and re.search(r'[a-z0-9]', normalized):
                return (normalized, "equals_rhs_expression", rhs)

    # 3. Fallback: look for the gold expression (normalized) in the text
    text_normalized = _normalize_expr(text)
    if gold_normalized and gold_normalized in text_normalized:
        return (gold_normalized, "expression_substring_match", gold.raw)

    # 4. For keyword-like expressions ("converges", "diverges", etc.)
    # that didn't match the keyword type (because they have extra text
    # in the gold, e.g. "The integral diverges.")
    for kw in TEXT_KEYWORDS:
        if kw in gold.raw.lower():
            result = _extract_keyword_from_text(text, [kw])
            if result:
                return (result[1], "expression_keyword_fallback", text[:200])

    return (None, "expression_not_found", "")


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def _numbers_close(a: float, b: float) -> bool:
    """Compare two numbers with relative and absolute tolerance."""
    if math.isnan(a) or math.isnan(b):
        return False
    if math.isinf(a) or math.isinf(b):
        return a == b
    if abs(b) > ABS_TOL:
        # Use relative tolerance
        return abs(a - b) / abs(b) <= REL_TOL
    else:
        # Use absolute tolerance for values near zero
        return abs(a - b) <= ABS_TOL


def compare_answers(
    extracted: Optional[str],
    extraction_method: str,
    gold: GoldAnswer,
) -> bool:
    """Compare extracted student answer against gold answer.

    Returns True if the medium-tier evaluation considers the answer correct.
    """
    if extracted is None:
        return False

    # ---- Keyword comparison ----
    if gold.answer_type == "keyword":
        return extracted.lower().strip() == gold.keywords[0]

    # ---- Multi-keyword comparison ----
    if gold.answer_type == "multi_keyword":
        extracted_parts = [
            p.strip().lower() for p in extracted.split(",")
        ]
        if len(extracted_parts) != len(gold.keywords):
            return False
        return all(e == g for e, g in zip(extracted_parts, gold.keywords))

    # ---- Numeric comparison ----
    if gold.answer_type == "numeric":
        ext_val = _try_parse_number(extracted)
        if ext_val is not None and gold.numeric_value is not None:
            return _numbers_close(ext_val, gold.numeric_value)
        return False

    # ---- Multi-part comparison ----
    if gold.answer_type == "multi_part":
        return _compare_multi_part(extracted, gold)

    # ---- Expression comparison ----
    if gold.answer_type == "expression":
        return _compare_expression(extracted, extraction_method, gold)

    return False


def _compare_multi_part(extracted: str, gold: GoldAnswer) -> bool:
    """Compare multi-part answers.

    We require that at least half of the parts with numeric values match,
    AND all numeric parts match (if any).
    """
    numeric_parts = [p for p in gold.parts if p.get("numeric") is not None]
    if not numeric_parts:
        # All text parts -- fall back to keyword matching
        gold_text = " ".join(p.get("raw", "") for p in gold.parts).lower()
        return gold_text in extracted.lower() or extracted.lower() in gold_text

    # Check how many numeric parts appear in the extracted answer
    matches = 0
    for part in numeric_parts:
        expected = part["numeric"]
        # Look for numbers in the extracted string
        nums = re.findall(r'(-?\d+(?:\.\d+)?(?:/\d+)?)', extracted)
        for n_str in nums:
            val = _try_parse_number(n_str)
            if val is not None and _numbers_close(val, expected):
                matches += 1
                break

    # Require ALL numeric parts to match
    return matches == len(numeric_parts)


def _compare_expression(
    extracted: str, extraction_method: str, gold: GoldAnswer
) -> bool:
    """Compare expression answers."""
    # If extraction was via keyword fallback, just check keyword match
    if extraction_method == "expression_keyword_fallback":
        for kw in TEXT_KEYWORDS:
            if kw in gold.raw.lower() and kw == extracted.lower():
                return True
        return False

    # If extraction was via substring match, it's correct by definition
    if extraction_method == "expression_substring_match":
        return True

    # Normalize both and compare
    ext_normalized = _normalize_expr(extracted)
    gold_normalized = gold.normalized_expr

    if not ext_normalized or not gold_normalized:
        return False

    # Direct string match after normalization
    if ext_normalized == gold_normalized:
        return True

    # Try numeric comparison as fallback (in case both are actually numbers)
    ext_val = _try_parse_number(extracted)
    gold_val = _try_parse_number(gold.raw)
    if ext_val is not None and gold_val is not None:
        return _numbers_close(ext_val, gold_val)

    # Check if one contains the other (for partial matches like
    # student writes "4x^2-12x+9" and gold is "4x^2-12x+9")
    if gold_normalized in ext_normalized or ext_normalized in gold_normalized:
        # Only accept if the shorter one is at least 60% of the longer
        shorter = min(len(ext_normalized), len(gold_normalized))
        longer = max(len(ext_normalized), len(gold_normalized))
        if longer > 0 and shorter / longer >= 0.6:
            return True

    return False


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def load_scenario_bank() -> dict[str, dict]:
    """Load scenario bank and return dict keyed by scenario_id."""
    with open(SCENARIO_BANK) as f:
        scenarios = json.load(f)
    return {s["id"]: s for s in scenarios}


def load_model_results(
    model_dir: str,
) -> list[dict]:
    """Load all cell results for one model directory."""
    dirpath = RESULTS_DIR / model_dir
    results = []
    for fn in sorted(os.listdir(dirpath)):
        if not fn.endswith(".json") or fn == "summary.json":
            continue
        with open(dirpath / fn) as f:
            data = json.load(f)
        if "results" in data:
            for r in data["results"]:
                r["_source_file"] = fn
                results.append(r)
    return results


def compute_rsr(items: list[dict], key: str) -> float:
    """Compute remediation success rate for a given boolean key."""
    if not items:
        return 0.0
    return sum(1 for r in items if r.get(key)) / len(items)


def build_breakdown(
    items: list[dict], group_key: str
) -> dict[str, dict]:
    """Group items by a key and compute 3-tier RSR for each group."""
    groups: dict[str, list] = defaultdict(list)
    for r in items:
        groups[r[group_key]].append(r)

    breakdown = {}
    for name, group in sorted(groups.items()):
        breakdown[name] = {
            "tier1_rsr": round(compute_rsr(group, "tier1_correct"), 4),
            "medium_rsr": round(compute_rsr(group, "medium_correct"), 4),
            "tier2_rsr": round(compute_rsr(group, "tier2_correct"), 4),
            "n": len(group),
        }
    return breakdown


def main() -> None:
    scenario_bank = load_scenario_bank()
    print(f"Loaded {len(scenario_bank)} scenarios from bank")
    print(f"Results dir: {RESULTS_DIR}")
    print()

    # Precompute GoldAnswer objects
    gold_answers: dict[str, GoldAnswer] = {}
    for sid, sc in scenario_bank.items():
        gold_answers[sid] = GoldAnswer(sc["verification_answer"], sid)

    # Log answer type distribution
    type_dist: dict[str, int] = defaultdict(int)
    for ga in gold_answers.values():
        type_dist[ga.answer_type] += 1
    print("Gold answer type distribution:")
    for t, c in sorted(type_dist.items()):
        print(f"  {t}: {c}")
    print()

    all_results: dict[str, list[dict]] = {}  # model_name -> results
    per_scenario_records: list[dict] = []
    extraction_failures = 0
    extraction_total = 0

    for dir_name, model_name in MODEL_DIRS.items():
        print(f"Processing {model_name} ({dir_name})...")
        results = load_model_results(dir_name)
        print(f"  Loaded {len(results)} results")

        model_failures = 0
        for r in results:
            sid = r["scenario_id"]
            extraction_total += 1

            if sid not in gold_answers:
                print(f"  WARNING: scenario {sid} not in bank, skipping")
                r["medium_correct"] = False
                r["extracted_answer"] = None
                r["extraction_method"] = "scenario_not_found"
                extraction_failures += 1
                continue

            gold = gold_answers[sid]
            student_text = r["verification_answer"]

            extracted, method, raw_text = extract_student_answer(
                student_text, gold
            )
            correct = compare_answers(extracted, method, gold)

            # Enforce monotonicity: if tier1 (strict heuristic) says
            # correct, medium must also say correct.  Tier1 is a subset
            # of medium by design.
            if r.get("tier1_correct") and not correct:
                correct = True
                method = "tier1_promotion"

            r["medium_correct"] = correct
            r["extracted_answer"] = extracted
            r["extraction_method"] = method

            if extracted is None:
                model_failures += 1
                extraction_failures += 1

            per_scenario_records.append({
                "scenario_id": sid,
                "model": model_name,
                "tier1": r.get("tier1_correct", False),
                "medium": correct,
                "tier2": r.get("tier2_correct", False),
                "gold_answer": gold.raw,
                "gold_type": gold.answer_type,
                "extracted_answer": extracted,
                "extraction_method": method,
                "difficulty": r.get("difficulty", ""),
                "misconception_type": r.get("misconception_type", ""),
                "strategy": r.get("strategy", ""),
            })

        print(f"  Extraction failures: {model_failures}/{len(results)}")
        all_results[model_name] = results

    print()
    print(f"Total extraction failures: {extraction_failures}/{extraction_total}")
    print()

    # ---------------------------------------------------------------------------
    # Build output structure
    # ---------------------------------------------------------------------------

    output: dict[str, Any] = {"models": {}, "aggregate": {}, "per_scenario": []}

    # Per-model summaries
    for model_name in MODEL_DIRS.values():
        items = all_results[model_name]
        model_data = {
            "overall": {
                "tier1_rsr": round(compute_rsr(items, "tier1_correct"), 4),
                "medium_rsr": round(compute_rsr(items, "medium_correct"), 4),
                "tier2_rsr": round(compute_rsr(items, "tier2_correct"), 4),
                "n": len(items),
            },
            "by_type": build_breakdown(items, "misconception_type"),
            "by_difficulty": build_breakdown(items, "difficulty"),
            "by_strategy": build_breakdown(items, "strategy"),
        }
        output["models"][model_name] = model_data

    # Aggregate across all models
    all_items = []
    for items in all_results.values():
        all_items.extend(items)

    output["aggregate"] = {
        "overall": {
            "tier1_rsr": round(compute_rsr(all_items, "tier1_correct"), 4),
            "medium_rsr": round(compute_rsr(all_items, "medium_correct"), 4),
            "tier2_rsr": round(compute_rsr(all_items, "tier2_correct"), 4),
            "n": len(all_items),
        },
        "by_type": build_breakdown(all_items, "misconception_type"),
        "by_difficulty": build_breakdown(all_items, "difficulty"),
    }

    output["per_scenario"] = per_scenario_records

    # ---------------------------------------------------------------------------
    # Print tables
    # ---------------------------------------------------------------------------

    print("=" * 80)
    print("THREE-TIER REMEDIATION SUCCESS RATE (RSR)")
    print("=" * 80)
    print()

    # --- Table 1: Per-model summary ---
    print("Table 1: Per-Model Overall RSR")
    print("-" * 70)
    print(f"{'Model':<16} {'n':>5}  {'Tier1 (Strict)':>14}  {'Medium':>10}  {'Tier2 (Lenient)':>15}")
    print("-" * 70)
    for model_name in MODEL_DIRS.values():
        d = output["models"][model_name]["overall"]
        print(
            f"{model_name:<16} {d['n']:>5}  "
            f"{d['tier1_rsr']:>13.1%}  "
            f"{d['medium_rsr']:>9.1%}  "
            f"{d['tier2_rsr']:>14.1%}"
        )
    d = output["aggregate"]["overall"]
    print("-" * 70)
    print(
        f"{'Aggregate':<16} {d['n']:>5}  "
        f"{d['tier1_rsr']:>13.1%}  "
        f"{d['medium_rsr']:>9.1%}  "
        f"{d['tier2_rsr']:>14.1%}"
    )
    print()

    # --- Table 2: Per-model x per-type ---
    print("Table 2: RSR by Misconception Type")
    print("-" * 80)
    print(
        f"{'Model':<16} {'Type':<13} {'n':>4}  "
        f"{'Tier1':>7}  {'Medium':>7}  {'Tier2':>7}"
    )
    print("-" * 80)
    for model_name in MODEL_DIRS.values():
        by_type = output["models"][model_name]["by_type"]
        for t in ["factual", "procedural", "conceptual"]:
            if t in by_type:
                d = by_type[t]
                print(
                    f"{model_name:<16} {t:<13} {d['n']:>4}  "
                    f"{d['tier1_rsr']:>6.1%}  "
                    f"{d['medium_rsr']:>6.1%}  "
                    f"{d['tier2_rsr']:>6.1%}"
                )
        print()
    # Aggregate
    by_type = output["aggregate"]["by_type"]
    for t in ["factual", "procedural", "conceptual"]:
        if t in by_type:
            d = by_type[t]
            print(
                f"{'Aggregate':<16} {t:<13} {d['n']:>4}  "
                f"{d['tier1_rsr']:>6.1%}  "
                f"{d['medium_rsr']:>6.1%}  "
                f"{d['tier2_rsr']:>6.1%}"
            )
    print()

    # --- Table 3: Per-model x per-difficulty ---
    print("Table 3: RSR by Difficulty")
    print("-" * 80)
    print(
        f"{'Model':<16} {'Difficulty':<10} {'n':>4}  "
        f"{'Tier1':>7}  {'Medium':>7}  {'Tier2':>7}"
    )
    print("-" * 80)
    for model_name in MODEL_DIRS.values():
        by_diff = output["models"][model_name]["by_difficulty"]
        for d_name in ["easy", "medium", "hard"]:
            if d_name in by_diff:
                d = by_diff[d_name]
                print(
                    f"{model_name:<16} {d_name:<10} {d['n']:>4}  "
                    f"{d['tier1_rsr']:>6.1%}  "
                    f"{d['medium_rsr']:>6.1%}  "
                    f"{d['tier2_rsr']:>6.1%}"
                )
        print()
    # Aggregate
    by_diff = output["aggregate"]["by_difficulty"]
    for d_name in ["easy", "medium", "hard"]:
        if d_name in by_diff:
            d = by_diff[d_name]
            print(
                f"{'Aggregate':<16} {d_name:<10} {d['n']:>4}  "
                f"{d['tier1_rsr']:>6.1%}  "
                f"{d['medium_rsr']:>6.1%}  "
                f"{d['tier2_rsr']:>6.1%}"
            )
    print()

    # --- Extraction method summary ---
    print("Extraction Method Distribution:")
    print("-" * 50)
    method_counts: dict[str, int] = defaultdict(int)
    for rec in per_scenario_records:
        method_counts[rec["extraction_method"]] += 1
    for method, count in sorted(method_counts.items(), key=lambda x: -x[1]):
        pct = count / len(per_scenario_records) * 100
        print(f"  {method:<35} {count:>5}  ({pct:>5.1f}%)")
    print()

    # --- Tier agreement analysis ---
    print("Tier Agreement Analysis:")
    print("-" * 50)
    n_total = len(per_scenario_records)
    t1_only = sum(
        1 for r in per_scenario_records
        if r["tier1"] and not r["medium"] and not r["tier2"]
    )
    med_only = sum(
        1 for r in per_scenario_records
        if not r["tier1"] and r["medium"] and not r["tier2"]
    )
    t2_only = sum(
        1 for r in per_scenario_records
        if not r["tier1"] and not r["medium"] and r["tier2"]
    )
    all_agree_yes = sum(
        1 for r in per_scenario_records
        if r["tier1"] and r["medium"] and r["tier2"]
    )
    all_agree_no = sum(
        1 for r in per_scenario_records
        if not r["tier1"] and not r["medium"] and not r["tier2"]
    )
    t1_and_med = sum(
        1 for r in per_scenario_records
        if r["tier1"] and r["medium"]
    )
    med_and_t2 = sum(
        1 for r in per_scenario_records
        if r["medium"] and r["tier2"]
    )
    t1_subset_med = sum(
        1 for r in per_scenario_records
        if r["tier1"] and not r["medium"]
    )
    med_subset_t2 = sum(
        1 for r in per_scenario_records
        if r["medium"] and not r["tier2"]
    )
    print(f"  All three agree (correct):     {all_agree_yes:>5}  ({all_agree_yes/n_total*100:>5.1f}%)")
    print(f"  All three agree (incorrect):   {all_agree_no:>5}  ({all_agree_no/n_total*100:>5.1f}%)")
    print(f"  Tier1 correct, Medium not:     {t1_subset_med:>5}  ({t1_subset_med/n_total*100:>5.1f}%)")
    print(f"  Medium correct, Tier2 not:     {med_subset_t2:>5}  ({med_subset_t2/n_total*100:>5.1f}%)")
    print(f"  Tier2 only (not Tier1/Medium): {t2_only:>5}  ({t2_only/n_total*100:>5.1f}%)")
    print()

    # --- Save JSON ---
    with open(OUTPUT_JSON, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"Saved comprehensive JSON to: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
