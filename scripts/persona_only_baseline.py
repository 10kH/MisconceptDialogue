#!/usr/bin/env python3
"""
Persona-only baseline: GPT-5-mini answers verification problems WITH misconception
persona but WITHOUT any tutoring dialogue.

This creates the missing control condition for the 3-way comparison:
  A. No persona, no tutoring  → results/no_tutoring_baseline/  (Tier2=98.1%)
  B. Persona, no tutoring     → results/persona_only_baseline/  (THIS SCRIPT)
  C. Persona + tutoring       → Exp.3 corrected results        (Tier2=81.3%)

The A→B gap measures the role-play effect (how much the misconception persona
alone reduces verification accuracy).  The B→C gap measures the net tutoring
effect (whether tutoring dialogue helps the student overcome the persona).

Uses the same three-tier evaluation as the tutored experiments:
  Tier 1:  word-boundary heuristic
  Medium:  numerical extraction + tolerance comparison
  Tier 2:  LLM-as-judge (GPT-5-mini)

Output:
  results/persona_only_baseline/results.json
  results/persona_only_baseline/summary.txt

Usage:
  set -a && source .env && set +a && conda run -n X --no-capture-output python scripts/persona_only_baseline.py
  ... --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.experiments.exp3v2.evaluator import (
    evaluate_tier1_heuristic,
    evaluate_with_llm_judge,
)
from src.models.openai_model import OpenAIModel

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Medium-tier: numerical extraction & comparison
# (identical to no_tutoring_baseline.py for consistency)
# ---------------------------------------------------------------------------

def extract_student_answer(text: str) -> str | None:
    """Extract the final numerical answer from a student free-form response."""
    if not text or not text.strip():
        return None

    patterns = [
        r'(?:final\s+answer|the\s+answer)\s*(?:is|=|:)\s*\$?\s*([-\d.,/]+)',
        r'(?:therefore|thus|so)\s*,?\s*(?:the\s+answer\s+is\s*)?\$?\s*([-\d.,/]+)',
        r'=\s*\$?\s*([-\d.,/]+)\s*$',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1)

    numbers = re.findall(r'\$?\s*([-]?\d+(?:[.,]\d+)?(?:/\d+)?)', text)
    if numbers:
        return numbers[-1]
    return None


def _try_parse_number(s: str) -> float | None:
    """Parse a string to float; handles $, commas, fractions, percentages."""
    s = s.strip().replace("$", "").replace(",", "").replace(" ", "")
    if not s:
        return None
    if s.endswith("%"):
        try:
            return float(s[:-1])
        except ValueError:
            pass
    frac = re.match(r'^(-?\d+)/(\d+)$', s)
    if frac:
        num, den = int(frac.group(1)), int(frac.group(2))
        if den != 0:
            return num / den
        return None
    try:
        return float(s)
    except ValueError:
        return None


def medium_eval(student_response: str, gold_answer_str: str) -> bool:
    """Numerical medium-tier evaluation with rel_tol=0.01, abs_tol=0.01."""
    extracted = extract_student_answer(student_response)
    if extracted is None:
        return False
    student_num = _try_parse_number(extracted)
    gold_num = _try_parse_number(gold_answer_str)
    if student_num is None or gold_num is None:
        return extracted.strip().lower() == gold_answer_str.strip().lower()
    if math.isnan(student_num) or math.isnan(gold_num):
        return False
    if gold_num == 0:
        return abs(student_num) <= 0.01
    rel_close = abs(student_num - gold_num) / max(abs(gold_num), 1e-9) <= 0.01
    abs_close = abs(student_num - gold_num) <= 0.01
    return rel_close or abs_close


# ---------------------------------------------------------------------------
# Student system prompt WITH misconception persona
# ---------------------------------------------------------------------------

def build_persona_prompt(scenario: dict) -> str:
    """Build a student system prompt that includes the misconception persona.

    Matches the persona structure used in Exp.3 (simulated_student.py) but
    WITHOUT the tutoring dialogue context.  The student is told to role-play
    as holding a specific misconception and to answer the problem accordingly.
    """
    mtype = scenario["misconception_type"]
    mlabel = scenario.get("misconception_label", "")

    type_desc = {
        "factual": "You have a factual misconception: you believe an incorrect mathematical fact to be true.",
        "procedural": "You have a procedural misconception: you follow an incorrect procedure or algorithm.",
        "conceptual": "You have a conceptual misconception: you have a fundamental misunderstanding of an underlying concept.",
    }

    prompt = (
        "You are a student answering a math problem. "
        "You have the following misconception and should answer consistently with it.\n\n"
        f"MISCONCEPTION TYPE: {mtype}\n"
        f"{type_desc.get(mtype, '')}\n"
        f"SPECIFIC MISCONCEPTION: {mlabel}\n\n"
        "IMPORTANT RULES:\n"
        "1. Stay in character as a student who holds this misconception.\n"
        "2. Show your work and give your final answer.\n"
        "3. Your answer should reflect this misconception — apply the incorrect "
        "reasoning naturally, as a real student would.\n"
        "4. Do NOT mention that you have a misconception or that your approach is wrong.\n"
    )
    return prompt


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def load_scenarios(bank_path: Path) -> list[dict]:
    with open(bank_path) as f:
        return json.load(f)


def compute_rsr(items: list[dict], key: str) -> float:
    if not items:
        return 0.0
    return sum(1 for r in items if r.get(key)) / len(items)


def build_by_type(items: list[dict]) -> dict:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in items:
        groups[r["misconception_type"]].append(r)
    result = {}
    for mtype, group in sorted(groups.items()):
        result[mtype] = {
            "tier1_rsr": round(compute_rsr(group, "tier1_correct"), 4),
            "medium_rsr": round(compute_rsr(group, "medium_correct"), 4),
            "tier2_rsr": round(compute_rsr(group, "tier2_correct"), 4),
            "n": len(group),
        }
    return result


def run(args: argparse.Namespace) -> None:
    bank_path = Path(args.bank)
    output_dir = Path(__file__).resolve().parent.parent / "results" / "persona_only_baseline"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.json"

    scenarios = load_scenarios(bank_path)
    logger.info("Loaded %d scenarios from %s", len(scenarios), bank_path)

    # Resume: load partial results if present
    already_done: dict[str, dict] = {}
    if args.resume and results_path.exists():
        with open(results_path) as f:
            partial = json.load(f)
        for rec in partial.get("per_scenario", []):
            already_done[rec["scenario_id"]] = rec
        logger.info("Resuming: %d scenarios already completed", len(already_done))

    # Initialise models
    student_model = OpenAIModel(model_name=args.model)
    judge_model = OpenAIModel(model_name=args.model)
    judge_cache: dict = {}

    per_scenario: list[dict] = list(already_done.values())
    todo = [s for s in scenarios if s["id"] not in already_done]
    logger.info("%d scenarios remaining to process", len(todo))

    for idx, scenario in enumerate(todo, start=1):
        sid = scenario["id"]
        mtype = scenario["misconception_type"]
        vprob = scenario["verification_problem"]
        vanswer = scenario["verification_answer"]
        misconception = scenario.get("misconception_label", "")

        # --- Build persona-aware student prompt ---
        student_system = build_persona_prompt(scenario)

        # --- Student generates answer (with misconception persona) ---
        response = student_model.generate(
            prompt=vprob,
            system_prompt=student_system,
            temperature=0.3,
            max_tokens=2048,
        )
        student_response = response.text

        # --- Tier 1: heuristic ---
        t1_result = evaluate_tier1_heuristic(
            student_response=student_response,
            correct_answer=vanswer,
            misconception_indicators=scenario.get("indicators") or None,
        )
        tier1_correct = t1_result["correct"]

        # --- Medium: numerical ---
        medium_correct = medium_eval(student_response, vanswer)
        if tier1_correct and not medium_correct:
            medium_correct = True

        # --- Tier 2: LLM judge ---
        t2_result = evaluate_with_llm_judge(
            student_response=student_response,
            correct_answer=vanswer,
            problem=vprob,
            misconception=misconception,
            judge_model=judge_model,
            cache=judge_cache,
        )
        tier2_correct = t2_result["correct"]

        rec = {
            "scenario_id": sid,
            "misconception_type": mtype,
            "misconception_label": misconception,
            "verification_problem": vprob,
            "verification_answer": vanswer,
            "student_system_prompt": student_system,
            "student_response": student_response,
            "tier1_correct": tier1_correct,
            "medium_correct": medium_correct,
            "tier2_correct": tier2_correct,
            "judge_reasoning": t2_result.get("reasoning", ""),
            "judge_error": t2_result.get("judge_error"),
        }
        per_scenario.append(rec)

        if idx % 10 == 0 or idx == len(todo):
            done_total = len(already_done) + idx
            logger.info(
                "Progress: %d/%d  |  Tier1=%.1f%%  Med=%.1f%%  Tier2=%.1f%%",
                done_total,
                len(scenarios),
                compute_rsr(per_scenario, "tier1_correct") * 100,
                compute_rsr(per_scenario, "medium_correct") * 100,
                compute_rsr(per_scenario, "tier2_correct") * 100,
            )
            _save_results(results_path, per_scenario, args)

    # Final save
    _save_results(results_path, per_scenario, args)

    # Print & write summary
    summary = _build_summary(per_scenario, args)
    summary_path = output_dir / "summary.txt"
    summary_path.write_text(summary)
    print(summary)
    logger.info("Results saved to %s", results_path)
    logger.info("Summary saved to %s", summary_path)


def _build_output(per_scenario: list[dict], args: argparse.Namespace) -> dict:
    n = len(per_scenario)
    by_type = build_by_type(per_scenario)
    return {
        "metadata": {
            "n_scenarios": n,
            "student_model": args.model,
            "judge_model": args.model,
            "condition": "persona_only (misconception persona, no tutoring)",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "overall": {
            "tier1_rsr": round(compute_rsr(per_scenario, "tier1_correct"), 4),
            "medium_rsr": round(compute_rsr(per_scenario, "medium_correct"), 4),
            "tier2_rsr": round(compute_rsr(per_scenario, "tier2_correct"), 4),
            "n": n,
        },
        "by_type": by_type,
        "per_scenario": per_scenario,
    }


def _save_results(path: Path, per_scenario: list[dict], args: argparse.Namespace) -> None:
    output = _build_output(per_scenario, args)
    with open(path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def _build_summary(per_scenario: list[dict], args: argparse.Namespace) -> str:
    output = _build_output(per_scenario, args)
    lines = [
        "=" * 60,
        "PERSONA-ONLY BASELINE — Remediation Success Rate",
        "(Misconception persona, NO tutoring dialogue)",
        "=" * 60,
        f"Student model : {args.model}",
        f"Judge model   : {args.model}",
        f"N scenarios   : {output['overall']['n']}",
        f"Timestamp     : {output['metadata']['timestamp']}",
        "",
        "Overall RSR:",
        f"  Tier 1  (heuristic) : {output['overall']['tier1_rsr']:.1%}",
        f"  Medium  (numerical) : {output['overall']['medium_rsr']:.1%}",
        f"  Tier 2  (LLM judge) : {output['overall']['tier2_rsr']:.1%}",
        "",
        "By misconception type:",
    ]
    for mtype in ("factual", "procedural", "conceptual"):
        if mtype in output["by_type"]:
            d = output["by_type"][mtype]
            lines.append(
                f"  {mtype:<12}  n={d['n']:>3}  "
                f"T1={d['tier1_rsr']:.1%}  Med={d['medium_rsr']:.1%}  "
                f"T2={d['tier2_rsr']:.1%}"
            )
    lines += [
        "",
        "3-way comparison (Tier 2):",
        f"  A. No persona, no tutoring : 98.1%",
        f"  B. Persona only (this run) : {output['overall']['tier2_rsr']:.1%}",
        f"  C. Persona + tutoring      : 81.3%",
        "",
        "=" * 60,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure persona-only baseline RSR (misconception persona, no tutoring)."
    )
    parser.add_argument(
        "--bank",
        default="data/exp3v2_scenario_bank.json",
        help="Path to scenario bank JSON (default: data/exp3v2_scenario_bank.json)",
    )
    parser.add_argument(
        "--model",
        default="gpt-5-mini",
        help="OpenAI model name for both student and judge (default: gpt-5-mini)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42; reserved for future use)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip scenarios already present in a partial results file",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bank_path = Path(args.bank)
    if not bank_path.is_absolute():
        bank_path = Path(__file__).resolve().parent.parent / bank_path
    args.bank = str(bank_path)
    run(args)
