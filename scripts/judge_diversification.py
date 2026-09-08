#!/usr/bin/env python3
"""Judge diversification: re-score Exp3v2 verification responses with GPT-5.4.

Compares GPT-5.4 judge with original GPT-5-mini judge (Tier 2) to measure
inter-judge agreement (Cohen's kappa, % agreement).

Usage:
  set -a && source .env && set +a && conda run -n X --no-capture-output \
    python scripts/judge_diversification.py [--dry-run] [--max-per-model N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results" / "exp3v2"
SCENARIO_BANK = PROJECT_ROOT / "data" / "exp3v2_scenario_bank.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "judge_diversification"

# Exp3v2 result directories (latest per model)
MODEL_DIRS = {
    "qwen-2.5-7b": "20260325_103441_qwen-2.5-7b",
    "mistral-7b": "20260325_103441_mistral-7b",
    "llama-3.1-8b": "20260325_103441_llama-3.1-8b",
    "phi-4": "20260326_010332_phi-4",
    "qwen3.5-9b": "20260326_010332_qwen3.5-9b",
}


def load_scenarios() -> dict[str, dict]:
    """Load scenario bank indexed by ID."""
    bank = json.loads(SCENARIO_BANK.read_text())
    return {s["id"]: s for s in bank}


def load_exp3v2_results() -> list[dict]:
    """Load all Exp3v2 results with verification responses."""
    scenarios = load_scenarios()
    items = []

    for model, dirname in MODEL_DIRS.items():
        model_dir = RESULTS_DIR / dirname
        if not model_dir.exists():
            logger.warning("Missing: %s", model_dir)
            continue

        for cell_file in sorted(model_dir.glob("cell_*.json")):
            try:
                cell = json.loads(cell_file.read_text())
                for r in cell.get("results", []):
                    sid = r.get("scenario_id", "")
                    scenario = scenarios.get(sid, {})
                    verification = r.get("verification_answer", "")
                    if not verification:
                        continue

                    items.append({
                        "tutor_model": model,
                        "scenario_id": sid,
                        "strategy": r.get("strategy", ""),
                        "misconception_type": r.get("misconception_type", ""),
                        # [FIX] 학생은 verification_problem을 풀므로 verification 필드 사용.
                        # 이전: scenario.get("correct_answer") → 원문 정답으로 judge에 전달.
                        "problem": scenario.get("verification_problem", ""),
                        "correct_answer": scenario.get("verification_answer", ""),
                        "misconception": scenario.get("misconception_label", ""),
                        "verification_answer": verification,
                        # Original GPT-5-mini judge results
                        "gpt5mini_tier2": r.get("tier2_correct"),
                        "gpt5mini_reasoning": r.get("judge_reasoning", ""),
                        "gpt5mini_confidence": r.get("judge_confidence"),
                        "gpt5mini_misconception_resolved": r.get("misconception_resolved"),
                        "tier1_correct": r.get("tier1_correct"),
                        "answer_revealed": r.get("answer_revealed", False),
                    })
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Error reading %s: %s", cell_file, e)

    return items


def rescore_with_gpt54(
    items: list[dict],
    max_items: int | None = None,
    dry_run: bool = False,
) -> list[dict]:
    """Re-score verification responses using GPT-5.4 as judge."""
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.experiments.exp3v2.evaluator import (
        evaluate_with_llm_judge,
    )
    from src.models.openai_model import OpenAIModel

    if dry_run:
        logger.info("DRY RUN: would re-score %d items with GPT-5.4", len(items))
        return items

    judge = OpenAIModel("gpt-5.4")
    cache: dict = {}

    if max_items:
        items = items[:max_items]

    total = len(items)
    logger.info("Re-scoring %d items with GPT-5.4 judge...", total)

    for i, item in enumerate(items):
        try:
            result = evaluate_with_llm_judge(
                student_response=item["verification_answer"],
                correct_answer=item["correct_answer"],
                problem=item["problem"],
                misconception=item["misconception"],
                judge_model=judge,
                cache=cache,
            )
            item["gpt54_tier2"] = result["correct"]
            item["gpt54_reasoning"] = result["reasoning"]
            item["gpt54_confidence"] = result["confidence"]
            item["gpt54_misconception_resolved"] = result["misconception_resolved"]
            item["gpt54_error"] = result.get("judge_error")

        except Exception as e:
            logger.error("Error on item %d (%s): %s", i, item["scenario_id"], e)
            item["gpt54_tier2"] = None
            item["gpt54_error"] = str(e)

        if (i + 1) % 50 == 0:
            logger.info("  %d/%d done", i + 1, total)

    return items


def compute_agreement(items: list[dict]) -> dict:
    """Compute inter-judge agreement between GPT-5-mini and GPT-5.4."""
    # Filter to items where both judges produced results
    valid = [
        it for it in items
        if it.get("gpt5mini_tier2") is not None and it.get("gpt54_tier2") is not None
    ]

    if len(valid) < 10:
        return {"error": f"Only {len(valid)} valid items"}

    gpt5mini = [it["gpt5mini_tier2"] for it in valid]
    gpt54 = [it["gpt54_tier2"] for it in valid]

    # Agreement rate
    agree = sum(1 for a, b in zip(gpt5mini, gpt54) if a == b)
    agreement_rate = agree / len(valid)

    # Cohen's kappa
    try:
        from sklearn.metrics import cohen_kappa_score, confusion_matrix
        kappa = cohen_kappa_score(gpt5mini, gpt54)
        cm = confusion_matrix(gpt5mini, gpt54, labels=[False, True]).tolist()
    except ImportError:
        # Manual Cohen's kappa
        n = len(valid)
        p_o = agreement_rate
        p_yes_mini = sum(gpt5mini) / n
        p_yes_54 = sum(gpt54) / n
        p_e = p_yes_mini * p_yes_54 + (1 - p_yes_mini) * (1 - p_yes_54)
        kappa = (p_o - p_e) / (1 - p_e) if p_e < 1 else 0
        cm = None

    # Per-model breakdown
    per_model = defaultdict(lambda: {"n": 0, "agree": 0, "mini_pos": 0, "gpt54_pos": 0})
    for it in valid:
        m = it["tutor_model"]
        per_model[m]["n"] += 1
        if it["gpt5mini_tier2"] == it["gpt54_tier2"]:
            per_model[m]["agree"] += 1
        if it["gpt5mini_tier2"]:
            per_model[m]["mini_pos"] += 1
        if it["gpt54_tier2"]:
            per_model[m]["gpt54_pos"] += 1

    model_breakdown = {}
    for m, stats in sorted(per_model.items()):
        model_breakdown[m] = {
            "n": stats["n"],
            "agreement": stats["agree"] / stats["n"],
            "gpt5mini_rsr": stats["mini_pos"] / stats["n"],
            "gpt54_rsr": stats["gpt54_pos"] / stats["n"],
        }

    # Per-type breakdown
    per_type = defaultdict(lambda: {"n": 0, "agree": 0, "mini_pos": 0, "gpt54_pos": 0})
    for it in valid:
        t = it.get("misconception_type", "unknown")
        per_type[t]["n"] += 1
        if it["gpt5mini_tier2"] == it["gpt54_tier2"]:
            per_type[t]["agree"] += 1
        if it["gpt5mini_tier2"]:
            per_type[t]["mini_pos"] += 1
        if it["gpt54_tier2"]:
            per_type[t]["gpt54_pos"] += 1

    type_breakdown = {}
    for t, stats in sorted(per_type.items()):
        type_breakdown[t] = {
            "n": stats["n"],
            "agreement": stats["agree"] / stats["n"],
            "gpt5mini_rsr": stats["mini_pos"] / stats["n"],
            "gpt54_rsr": stats["gpt54_pos"] / stats["n"],
        }

    return {
        "n_total": len(valid),
        "agreement_rate": agreement_rate,
        "cohen_kappa": kappa,
        "confusion_matrix": cm,
        "gpt5mini_positive_rate": sum(gpt5mini) / len(gpt5mini),
        "gpt54_positive_rate": sum(gpt54) / len(gpt54),
        "per_model": model_breakdown,
        "per_type": type_breakdown,
    }


def format_report(agreement: dict) -> str:
    """Format human-readable report."""
    lines = [
        "=" * 70,
        "JUDGE DIVERSIFICATION — GPT-5-mini vs GPT-5.4",
        "=" * 70,
        "",
        f"Total evaluated: {agreement['n_total']}",
        f"Agreement rate: {agreement['agreement_rate']:.3f}",
        f"Cohen's kappa: {agreement['cohen_kappa']:.3f}",
        f"GPT-5-mini positive rate: {agreement['gpt5mini_positive_rate']:.3f}",
        f"GPT-5.4 positive rate: {agreement['gpt54_positive_rate']:.3f}",
        "",
    ]

    if agreement.get("confusion_matrix"):
        cm = agreement["confusion_matrix"]
        lines.append("Confusion Matrix (GPT-5-mini rows, GPT-5.4 cols):")
        lines.append(f"              GPT54-F  GPT54-T")
        lines.append(f"  Mini-F      {cm[0][0]:>6}   {cm[0][1]:>6}")
        lines.append(f"  Mini-T      {cm[1][0]:>6}   {cm[1][1]:>6}")
        lines.append("")

    lines.append("Per-Model Breakdown:")
    lines.append(f"  {'Model':<16} {'n':>5} {'Agree':>7} {'Mini RSR':>9} {'GPT54 RSR':>10}")
    lines.append("  " + "-" * 50)
    for model, stats in sorted(agreement.get("per_model", {}).items()):
        lines.append(
            f"  {model:<16} {stats['n']:>5} {stats['agreement']:>7.3f} "
            f"{stats['gpt5mini_rsr']:>9.3f} {stats['gpt54_rsr']:>10.3f}"
        )

    lines.append("")
    lines.append("Per-Type Breakdown:")
    lines.append(f"  {'Type':<16} {'n':>5} {'Agree':>7} {'Mini RSR':>9} {'GPT54 RSR':>10}")
    lines.append("  " + "-" * 50)
    for mtype, stats in sorted(agreement.get("per_type", {}).items()):
        lines.append(
            f"  {mtype:<16} {stats['n']:>5} {stats['agreement']:>7.3f} "
            f"{stats['gpt5mini_rsr']:>9.3f} {stats['gpt54_rsr']:>10.3f}"
        )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Judge diversification: GPT-5.4 re-scoring of Exp3v2"
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--max-per-model", type=int, default=None,
        help="Max items to re-score (for testing)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load data
    items = load_exp3v2_results()
    logger.info("Loaded %d verification responses from Exp3v2", len(items))

    if args.dry_run:
        logger.info("DRY RUN — would re-score %d items", len(items))
        return

    # Re-score
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    items = rescore_with_gpt54(items, max_items=args.max_per_model)

    # Save raw results
    raw_path = OUTPUT_DIR / "raw_results.json"
    with open(raw_path, "w") as f:
        json.dump(items, f, indent=2)
    logger.info("Raw results saved to %s", raw_path)

    # Compute agreement
    agreement = compute_agreement(items)

    # Save and display
    report = format_report(agreement)
    print(report)

    (OUTPUT_DIR / "agreement_report.txt").write_text(report)
    with open(OUTPUT_DIR / "agreement_stats.json", "w") as f:
        json.dump(agreement, f, indent=2)
    logger.info("Results saved to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()
