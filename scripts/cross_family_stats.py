#!/usr/bin/env python3
"""
Statistical analysis for cross-family validation results.

Reads from results/cross_family_v2/ and performs:
  1. Kruskal-Wallis test across evaluators (non-parametric ANOVA)
  2. Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction
  3. ICC (Intraclass Correlation Coefficient) for evaluator agreement
  4. Aggregated table: evaluator × mean RSR (±SD) across tutors

Usage:
  conda run -n X --no-capture-output python scripts/cross_family_stats.py
  # Optional: override the default result directory or directory glob.
  python scripts/cross_family_stats.py --results-dir results/cross_family_v2 --pattern "*__*"
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

# Try scipy imports
try:
    from scipy import stats as sp_stats
except ImportError:
    print("ERROR: scipy required. Install: pip install scipy", file=sys.stderr)
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "cross_family_v2"


VALID_MODELS = {"qwen-2.5-7b", "mistral-7b", "llama-3.1-8b", "phi-4", "qwen3.5-9b", "gpt-5.4"}


def load_results(results_dir: Path, pattern: str = "*__*") -> dict:
    """Load all (tutor, evaluator) pair results.

    Returns: {evaluator: {tutor: {"p1_rsr": float, "b1_rsr": float, "arr": float}}}
    """
    data: dict[str, dict[str, dict]] = defaultdict(dict)

    for d in sorted(results_dir.glob(pattern)):
        if not d.is_dir():
            continue
        summary_path = d / "summary.json"
        if not summary_path.exists():
            continue

        try:
            summary = json.loads(summary_path.read_text())
            metrics = summary.get("condition_metrics", {})

            # Parse directory name for tutor/evaluator
            name = d.name
            if "__" in name:
                tutor, evaluator = name.split("__", 1)
            elif name.startswith("cross_family_"):
                # Legacy v1 format
                evaluator = name.replace("cross_family_", "").replace("_student", "")
                tutor = summary.get("model", "unknown")
            else:
                continue

            # Filter to valid models only
            if tutor not in VALID_MODELS or evaluator not in VALID_MODELS:
                continue

            p1 = metrics.get("P1", {})
            b1 = metrics.get("B1", {})
            p1_rsr = p1.get("remediation_success_rate")
            b1_rsr = b1.get("remediation_success_rate")
            p1_arr = p1.get("answer_reveal_rate", 0)
            b1_arr = b1.get("answer_reveal_rate", 0)

            if p1_rsr is not None and b1_rsr is not None:
                data[evaluator][tutor] = {
                    "p1_rsr": p1_rsr,
                    "b1_rsr": b1_rsr,
                    "arr": (p1_arr + b1_arr) / 2,
                    "n": p1.get("n", 0),
                }
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Warning: could not read {summary_path}: {e}", file=sys.stderr)

    return dict(data)


def compute_mean_rsr(data: dict) -> dict:
    """Compute mean RSR (±SD) across tutors for each evaluator."""
    table = {}
    for evaluator, tutor_data in sorted(data.items()):
        p1_vals = [v["p1_rsr"] for v in tutor_data.values()]
        b1_vals = [v["b1_rsr"] for v in tutor_data.values()]
        arr_vals = [v["arr"] for v in tutor_data.values()]

        table[evaluator] = {
            "n_tutors": len(tutor_data),
            "tutors": list(tutor_data.keys()),
            "p1_mean": np.mean(p1_vals),
            "p1_sd": np.std(p1_vals, ddof=1) if len(p1_vals) > 1 else 0,
            "b1_mean": np.mean(b1_vals),
            "b1_sd": np.std(b1_vals, ddof=1) if len(b1_vals) > 1 else 0,
            "arr_mean": np.mean(arr_vals),
            "arr_sd": np.std(arr_vals, ddof=1) if len(arr_vals) > 1 else 0,
        }
    return table


def kruskal_wallis_test(data: dict, metric: str = "p1_rsr") -> dict:
    """Kruskal-Wallis H-test across evaluators.

    Uses mean RSR per evaluator (averaged across tutors) pooled together.
    Each evaluator contributes one value per tutor.
    """
    groups = []
    labels = []
    for evaluator, tutor_data in sorted(data.items()):
        vals = [v[metric] for v in tutor_data.values()]
        if vals:
            groups.append(vals)
            labels.append(evaluator)

    if len(groups) < 2:
        return {"error": "Need at least 2 evaluators for Kruskal-Wallis test"}

    H, p = sp_stats.kruskal(*groups)
    return {
        "test": "Kruskal-Wallis",
        "metric": metric,
        "H": float(H),
        "p": float(p),
        "n_groups": len(groups),
        "group_sizes": [len(g) for g in groups],
        "labels": labels,
    }


def pairwise_wilcoxon(data: dict, metric: str = "p1_rsr") -> list[dict]:
    """Pairwise Wilcoxon signed-rank tests with Holm-Bonferroni correction.

    For each evaluator pair, compares their RSR values across the same tutors.
    """
    evaluators = sorted(data.keys())
    results = []

    for e1, e2 in combinations(evaluators, 2):
        # Find shared tutors
        shared = set(data[e1].keys()) & set(data[e2].keys())
        if len(shared) < 3:
            results.append({
                "evaluator_1": e1,
                "evaluator_2": e2,
                "error": f"Only {len(shared)} shared tutors (need >=3)",
            })
            continue

        vals1 = [data[e1][t][metric] for t in sorted(shared)]
        vals2 = [data[e2][t][metric] for t in sorted(shared)]

        try:
            stat, p = sp_stats.wilcoxon(vals1, vals2, alternative="two-sided")
            diff = np.mean(vals1) - np.mean(vals2)
            results.append({
                "evaluator_1": e1,
                "evaluator_2": e2,
                "mean_diff": float(diff),
                "W": float(stat),
                "p_raw": float(p),
                "n_shared": len(shared),
            })
        except ValueError as e:
            results.append({
                "evaluator_1": e1,
                "evaluator_2": e2,
                "error": str(e),
            })

    # Holm-Bonferroni correction
    valid = [r for r in results if "p_raw" in r]
    valid.sort(key=lambda r: r["p_raw"])
    m = len(valid)
    for i, r in enumerate(valid):
        r["p_holm"] = min(1.0, r["p_raw"] * (m - i))
        r["significant"] = r["p_holm"] < 0.05

    return results


def icc_two_way(data: dict, metric: str = "p1_rsr") -> dict:
    """ICC(2,1) - Two-way random, single measures.

    Rows = tutors, Columns = evaluators.
    """
    evaluators = sorted(data.keys())
    # Find tutors present in ALL evaluators
    all_tutors = None
    for e in evaluators:
        tutors = set(data[e].keys())
        if all_tutors is None:
            all_tutors = tutors
        else:
            all_tutors &= tutors

    if not all_tutors or len(all_tutors) < 2 or len(evaluators) < 2:
        return {"error": "Need >=2 tutors and >=2 evaluators for ICC"}

    tutors = sorted(all_tutors)
    n = len(tutors)  # rows (subjects)
    k = len(evaluators)  # columns (raters)

    # Build matrix
    matrix = np.zeros((n, k))
    for j, e in enumerate(evaluators):
        for i, t in enumerate(tutors):
            matrix[i, j] = data[e][t][metric]

    # ANOVA components
    grand_mean = matrix.mean()
    ss_total = np.sum((matrix - grand_mean) ** 2)
    ss_rows = k * np.sum((matrix.mean(axis=1) - grand_mean) ** 2)  # Between subjects
    ss_cols = n * np.sum((matrix.mean(axis=0) - grand_mean) ** 2)  # Between raters
    ss_error = ss_total - ss_rows - ss_cols

    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))

    # ICC(2,1) formula
    icc = (ms_rows - ms_error) / (ms_rows + (k - 1) * ms_error + k * (ms_cols - ms_error) / n)

    return {
        "icc_type": "ICC(2,1)",
        "icc": float(icc),
        "n_tutors": n,
        "n_evaluators": k,
        "metric": metric,
        "interpretation": (
            "poor" if icc < 0.5 else
            "moderate" if icc < 0.75 else
            "good" if icc < 0.9 else
            "excellent"
        ),
    }


def format_report(
    data: dict,
    mean_table: dict,
    kw_p1: dict,
    kw_b1: dict,
    pairwise: list[dict],
    icc_result: dict,
) -> str:
    """Format the full statistical report."""
    lines = [
        "=" * 70,
        "CROSS-FAMILY VALIDATION v2 — Statistical Analysis",
        "=" * 70,
        "",
    ]

    # Aggregated table
    lines.append("1. AGGREGATED TABLE (mean ± SD across tutors)")
    lines.append("-" * 70)
    lines.append(f"{'Evaluator':<20} {'n':>3} {'P1 RSR':>14} {'B1 RSR':>14} {'ARR':>14}")
    lines.append("-" * 70)
    for evaluator, stats in sorted(mean_table.items(), key=lambda x: -x[1]["p1_mean"]):
        lines.append(
            f"{evaluator:<20} {stats['n_tutors']:>3} "
            f"{stats['p1_mean']:.2f}±{stats['p1_sd']:.2f}  "
            f"{stats['b1_mean']:.2f}±{stats['b1_sd']:.2f}  "
            f"{stats['arr_mean']:.2f}±{stats['arr_sd']:.2f}"
        )
    lines.append("")

    # Kruskal-Wallis
    lines.append("2. KRUSKAL-WALLIS TEST (evaluator effect)")
    lines.append("-" * 70)
    for label, kw in [("Pipeline RSR (P1)", kw_p1), ("Direct RSR (B1)", kw_b1)]:
        if "error" in kw:
            lines.append(f"  {label}: {kw['error']}")
        else:
            sig = "***" if kw["p"] < 0.001 else "**" if kw["p"] < 0.01 else "*" if kw["p"] < 0.05 else "n.s."
            lines.append(f"  {label}: H={kw['H']:.2f}, p={kw['p']:.4f} {sig}")
    lines.append("")

    # Pairwise
    lines.append("3. PAIRWISE COMPARISONS (Wilcoxon signed-rank, Holm-corrected)")
    lines.append("-" * 70)
    sig_pairs = [r for r in pairwise if r.get("significant")]
    lines.append(f"  Significant pairs (p_holm < 0.05): {len(sig_pairs)}/{len(pairwise)}")
    for r in sorted(pairwise, key=lambda x: x.get("p_holm", 1)):
        if "error" in r:
            lines.append(f"  {r['evaluator_1']} vs {r['evaluator_2']}: {r['error']}")
        elif r.get("significant"):
            lines.append(
                f"  {r['evaluator_1']} vs {r['evaluator_2']}: "
                f"Δ={r['mean_diff']:+.3f}, W={r['W']:.0f}, "
                f"p_raw={r['p_raw']:.4f}, p_holm={r['p_holm']:.4f} *"
            )
    lines.append("")

    # ICC
    lines.append("4. INTER-EVALUATOR AGREEMENT (ICC)")
    lines.append("-" * 70)
    if "error" in icc_result:
        lines.append(f"  {icc_result['error']}")
    else:
        lines.append(
            f"  {icc_result['icc_type']}: {icc_result['icc']:.3f} "
            f"({icc_result['interpretation']})"
        )
        lines.append(
            f"  Based on {icc_result['n_tutors']} tutors × "
            f"{icc_result['n_evaluators']} evaluators"
        )

    lines.append("")
    lines.append("=" * 70)
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical analysis for cross-family validation"
    )
    parser.add_argument(
        "--results-dir", type=str, default=str(DEFAULT_RESULTS),
        help="Results directory (default: results/cross_family_v2/)",
    )
    parser.add_argument(
        "--pattern", type=str, default="*__*",
        help="Glob pattern for result directories (default: *__*)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output file (default: <results-dir>/statistical_analysis.txt)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"ERROR: {results_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # Load data
    data = load_results(results_dir, args.pattern)
    if not data:
        print("ERROR: No results found", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {sum(len(v) for v in data.values())} (tutor, evaluator) pairs")
    print(f"Evaluators: {sorted(data.keys())}")

    # Compute statistics
    mean_table = compute_mean_rsr(data)
    kw_p1 = kruskal_wallis_test(data, "p1_rsr")
    kw_b1 = kruskal_wallis_test(data, "b1_rsr")
    pairwise = pairwise_wilcoxon(data, "p1_rsr")
    icc_result = icc_two_way(data, "p1_rsr")

    # Format report
    report = format_report(data, mean_table, kw_p1, kw_b1, pairwise, icc_result)
    print(report)

    # Save
    out_path = Path(args.output) if args.output else results_dir / "statistical_analysis.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"\nSaved to {out_path}")

    # Save JSON
    json_out = {
        "mean_table": mean_table,
        "kruskal_wallis_p1": kw_p1,
        "kruskal_wallis_b1": kw_b1,
        "pairwise_wilcoxon": pairwise,
        "icc": icc_result,
    }
    json_path = out_path.with_suffix(".json")
    json_path.write_text(json.dumps(json_out, indent=2))
    print(f"JSON saved to {json_path}")


if __name__ == "__main__":
    main()
