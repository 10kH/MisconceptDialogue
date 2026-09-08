"""Analysis module for Exp4v2: Pipeline Progressive Oracle Ablation.

Loads condition results, computes ablation tables, type breakdowns,
pairwise deltas, and detection/strategy analysis.

Usage:
    python -m src.experiments.exp4v2.analysis --results_dir results/exp4v2/
    python -m src.experiments.exp4v2.analysis --results_dir results/exp4v2/ --format latex
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

from src.experiments.exp4v2.prompts import CONDITIONS, CONDITION_LABELS, MISCONCEPTION_TYPES

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_model_results(model_dir: Path) -> dict[str, dict]:
    """Load all condition JSONs from a single model's result directory."""
    results = {}
    for cond in CONDITIONS:
        path = model_dir / f"condition_{cond}.json"
        if path.exists():
            with open(path) as f:
                results[cond] = json.load(f)
    return results


def discover_models(results_dir: Path) -> dict[str, Path]:
    """Discover all model result directories under results_dir."""
    models = {}
    for d in sorted(results_dir.iterdir()):
        if d.is_dir() and not d.name.startswith("dryrun"):
            # Extract model name from directory name: {timestamp}_{model}
            parts = d.name.split("_", 2)
            if len(parts) >= 3:
                model_name = parts[2]
                models[model_name] = d
    return models


# ---------------------------------------------------------------------------
# Ablation table: condition × model RSR
# ---------------------------------------------------------------------------

def compute_ablation_table(
    all_results: dict[str, dict[str, dict]],
) -> dict:
    """Compute RSR for each (model, condition) pair.

    Args:
        all_results: {model_name: {condition: condition_result_dict}}

    Returns:
        {model: {condition: {rsr, arr, n, ci_lo, ci_hi}}}
    """
    table = {}
    for model, cond_results in all_results.items():
        table[model] = {}
        for cond, data in cond_results.items():
            agg = data["aggregate"]
            table[model][cond] = {
                "rsr": agg["remediation_success_rate"],
                "arr": agg["answer_reveal_rate"],
                "n": agg["n_scenarios"],
                "ci_lo": agg.get("rsr_ci_lower", 0),
                "ci_hi": agg.get("rsr_ci_upper", 0),
            }
    return table


# ---------------------------------------------------------------------------
# Type breakdown: condition × type RSR
# ---------------------------------------------------------------------------

def compute_type_breakdown(
    all_results: dict[str, dict[str, dict]],
) -> dict:
    """Compute RSR by misconception type for each (model, condition).

    Returns:
        {model: {condition: {type: {rsr, n, arr}}}}
    """
    breakdown = {}
    for model, cond_results in all_results.items():
        breakdown[model] = {}
        for cond, data in cond_results.items():
            by_type = data["aggregate"].get("by_type", {})
            breakdown[model][cond] = {}
            for mtype in MISCONCEPTION_TYPES:
                if mtype in by_type:
                    breakdown[model][cond][mtype] = {
                        "rsr": by_type[mtype]["rsr"],
                        "n": by_type[mtype]["n"],
                        "arr": by_type[mtype].get("arr", 0),
                    }
    return breakdown


# ---------------------------------------------------------------------------
# Pairwise deltas
# ---------------------------------------------------------------------------

_PAIRWISE_COMPARISONS = [
    ("C1", "C2", "Strategy assignment effect"),
    ("C2", "C3", "Diagnosis error cost"),
    ("C3", "C3_5", "Diagnosis info value"),
    ("C3", "C4", "Detection error cost"),
    ("C3_5", "B1", "Detection-only value"),
    ("C4", "B1", "Pipeline value-add"),
    ("C1", "C4", "Total pipeline degradation"),
    ("C1", "B1", "Oracle vs no-pipeline"),
]


def compute_pairwise_deltas(
    ablation_table: dict[str, dict[str, dict]],
) -> list[dict]:
    """Compute RSR deltas between paired conditions for each model."""
    deltas = []
    for upper, lower, label in _PAIRWISE_COMPARISONS:
        row = {"comparison": f"{upper} vs {lower}", "label": label}
        for model, conds in ablation_table.items():
            if upper in conds and lower in conds:
                delta = conds[upper]["rsr"] - conds[lower]["rsr"]
                row[model] = round(delta, 3)
            else:
                row[model] = None
        deltas.append(row)
    return deltas


# ---------------------------------------------------------------------------
# Detection analysis (C4)
# ---------------------------------------------------------------------------

def compute_detection_analysis(
    all_results: dict[str, dict[str, dict]],
) -> dict:
    """Analyze detection rates in C4 condition."""
    analysis = {}
    for model, cond_results in all_results.items():
        if "C4" in cond_results:
            stats = cond_results["C4"]["aggregate"].get("detection_stats")
            if stats:
                analysis[model] = stats
    return analysis


# ---------------------------------------------------------------------------
# Strategy analysis (C2)
# ---------------------------------------------------------------------------

def compute_strategy_analysis(
    all_results: dict[str, dict[str, dict]],
) -> dict:
    """Analyze freely chosen strategies in C2 condition."""
    analysis = {}
    for model, cond_results in all_results.items():
        if "C2" in cond_results:
            dist = cond_results["C2"]["aggregate"].get("strategy_distribution")
            if dist:
                analysis[model] = dist
    return analysis


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def print_ablation_table(table: dict, fmt: str = "text"):
    """Print condition × model RSR table."""
    models = sorted(table.keys())
    conditions = [c for c in CONDITIONS if any(c in table[m] for m in models)]

    if fmt == "latex":
        _print_ablation_latex(table, models, conditions)
    else:
        _print_ablation_text(table, models, conditions)


def _print_ablation_text(table, models, conditions):
    header = f"{'Condition':12s}" + "".join(f"{m:>15s}" for m in models)
    print(header)
    print("-" * len(header))
    for cond in conditions:
        row = f"{cond:12s}"
        for model in models:
            if cond in table[model]:
                rsr = table[model][cond]["rsr"]
                row += f"{rsr:>14.1%} "
            else:
                row += f"{'—':>15s}"
        print(row)


def _print_ablation_latex(table, models, conditions):
    n_models = len(models)
    print(f"\\begin{{tabular}}{{l{'c' * n_models}}}")
    print("\\toprule")
    print("Condition & " + " & ".join(m.replace("-", " ") for m in models) + " \\\\")
    print("\\midrule")
    for cond in conditions:
        label = CONDITION_LABELS.get(cond, cond)
        row = f"{label}"
        for model in models:
            if cond in table[model]:
                rsr = table[model][cond]["rsr"]
                row += f" & {rsr:.1%}"
            else:
                row += " & —"
        print(row + " \\\\")
    print("\\bottomrule")
    print("\\end{tabular}")


def print_pairwise_deltas(deltas: list[dict], models: list[str]):
    header = f"{'Comparison':20s} {'Effect':30s}" + "".join(f"{m:>12s}" for m in models)
    print(header)
    print("-" * len(header))
    for row in deltas:
        line = f"{row['comparison']:20s} {row['label']:30s}"
        for m in models:
            val = row.get(m)
            if val is not None:
                sign = "+" if val > 0 else ""
                line += f"{sign}{val:>10.1%}  "
            else:
                line += f"{'—':>12s}"
        print(line)


def print_type_breakdown(breakdown: dict, fmt: str = "text"):
    """Print condition × type RSR for each model."""
    for model, cond_data in sorted(breakdown.items()):
        print(f"\n=== {model} ===")
        conditions = [c for c in CONDITIONS if c in cond_data]
        header = f"{'Condition':12s}" + "".join(f"{t:>12s}" for t in MISCONCEPTION_TYPES)
        print(header)
        print("-" * len(header))
        for cond in conditions:
            row = f"{cond:12s}"
            for mtype in MISCONCEPTION_TYPES:
                if mtype in cond_data[cond]:
                    rsr = cond_data[cond][mtype]["rsr"]
                    row += f"{rsr:>11.1%} "
                else:
                    row += f"{'—':>12s}"
            print(row)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exp4v2 analysis")
    parser.add_argument("--results_dir", required=True, help="Path to results/exp4v2/")
    parser.add_argument("--format", default="text", choices=["text", "latex"])
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    models = discover_models(results_dir)

    if not models:
        logger.error(f"No model directories found in {results_dir}")
        sys.exit(1)

    logger.info(f"Found {len(models)} models: {list(models.keys())}")

    # Load all results
    all_results = {}
    for model_name, model_dir in models.items():
        all_results[model_name] = load_model_results(model_dir)
        logger.info(f"  {model_name}: {len(all_results[model_name])} conditions")

    # Ablation table
    print("\n" + "=" * 60)
    print("ABLATION TABLE: Condition × Model RSR")
    print("=" * 60)
    table = compute_ablation_table(all_results)
    print_ablation_table(table, fmt=args.format)

    # Pairwise deltas
    print("\n" + "=" * 60)
    print("PAIRWISE DELTAS (upper - lower)")
    print("=" * 60)
    deltas = compute_pairwise_deltas(table)
    print_pairwise_deltas(deltas, sorted(table.keys()))

    # Type breakdown
    print("\n" + "=" * 60)
    print("TYPE BREAKDOWN: Condition × Type RSR")
    print("=" * 60)
    breakdown = compute_type_breakdown(all_results)
    print_type_breakdown(breakdown, fmt=args.format)

    # Detection analysis
    det = compute_detection_analysis(all_results)
    if det:
        print("\n" + "=" * 60)
        print("DETECTION ANALYSIS (C4)")
        print("=" * 60)
        for model, stats in sorted(det.items()):
            print(f"  {model}: {stats['detected']}/{stats['total']} detected "
                  f"({stats['detection_rate']:.1%}), {stats['missed']} missed")

    # Strategy analysis
    strat = compute_strategy_analysis(all_results)
    if strat:
        print("\n" + "=" * 60)
        print("STRATEGY DISTRIBUTION (C2 - Free Choice)")
        print("=" * 60)
        for model, dist in sorted(strat.items()):
            print(f"  {model}: {dist}")


if __name__ == "__main__":
    main()
