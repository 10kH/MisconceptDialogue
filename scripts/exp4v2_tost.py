"""
TOST (Two One-Sided Tests) Equivalence Analysis for Exp.4v2

Tests whether pairs of pipeline conditions produce statistically equivalent
remediation success rates (RSR), pooled across all 5 models.

Usage: conda run -n X python scripts/exp4v2_tost.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import json
import glob
import os
from itertools import combinations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "results", "exp4v2"
)
OUTPUT_FILE = os.path.join(RESULTS_DIR, "tost_analysis.txt")

CONDITIONS = ["C1", "C2", "C3", "C3_5", "C4", "B1"]
CONDITION_LABELS = {
    "C1":    "C1 Oracle-All",
    "C2":    "C2 Oracle-Det",
    "C3":    "C3 Oracle-Diag",
    "C3_5":  "C3.5 Oracle-Strat",
    "C4":    "C4 Full-Pipeline",
    "B1":    "B1 Direct",
}

# Priority comparisons shown first
PRIORITY_PAIRS = [
    ("C4", "B1"),   # Full pipeline vs direct — most important
    ("C1", "B1"),   # Oracle all vs direct
    ("C1", "C4"),   # Oracle all vs full pipeline
    ("C2", "C4"),   # Oracle detection vs full pipeline
    ("C3", "C4"),   # Oracle diagnosis vs full pipeline
    ("C3_5", "C4"), # Oracle strategy vs full pipeline
]

MARGINS = [0.05, 0.10]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_model_dirs():
    """Return {model_name: path} for all 5 standardised models."""
    dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, "2026*")))
    mapping = {}
    for d in dirs:
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        # Strip timestamp prefix  YYYYMMDD_HHMMSS_
        parts = base.split("_", 2)
        if len(parts) == 3:
            model = parts[2]
        else:
            model = base
        mapping[model] = d
    return mapping


def load_successes(model_dir: str, condition: str) -> np.ndarray:
    """Return binary array (1=success, 0=failure) for one model/condition."""
    path = os.path.join(model_dir, f"condition_{condition}.json")
    with open(path) as f:
        data = json.load(f)
    return np.array(
        [1 if r["remediation_success"] else 0 for r in data["results"]],
        dtype=int,
    )


def load_all_data(model_dirs: dict) -> dict:
    """
    Returns {condition: np.ndarray of shape (n_models * 106,)}
    Pooled across all models, in model-alphabetical order.
    """
    pooled = {c: [] for c in CONDITIONS}
    for model in sorted(model_dirs.keys()):
        path = model_dirs[model]
        for cond in CONDITIONS:
            arr = load_successes(path, cond)
            pooled[cond].append(arr)
    return {c: np.concatenate(pooled[c]) for c in CONDITIONS}

# ---------------------------------------------------------------------------
# TOST for difference of two proportions
# ---------------------------------------------------------------------------

def tost_proportions(x1: np.ndarray, x2: np.ndarray, delta: float, alpha: float = 0.05):
    """
    Two One-Sided Tests for equivalence of two proportions.

    H0_lower:  p1 - p2 <= -delta   (test with H1: p1 - p2 > -delta)
    H0_upper:  p1 - p2 >=  delta   (test with H1: p1 - p2 <  delta)

    Equivalence is declared when BOTH one-sided null hypotheses are rejected,
    i.e., p_TOST = max(p_lower, p_upper) < alpha.

    Returns a dict with full results.
    """
    n1, n2 = len(x1), len(x2)
    p1_hat = x1.mean()
    p2_hat = x2.mean()
    diff = p1_hat - p2_hat

    # Pooled SE for each one-sided test (under H0 boundary, using observed proportions)
    # Standard approach: SE = sqrt(p1(1-p1)/n1 + p2(1-p2)/n2)
    se = np.sqrt(p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2)

    # z-statistics
    # Lower test: H0: diff <= -delta  →  z = (diff - (-delta)) / se
    z_lower = (diff + delta) / se
    # Upper test: H0: diff >= +delta  →  z = (diff - delta) / se
    z_upper = (diff - delta) / se

    # One-sided p-values
    p_lower = 1 - stats.norm.cdf(z_lower)   # P(Z > z_lower)  ... want small
    p_upper = stats.norm.cdf(z_upper)        # P(Z < z_upper)  ... want small

    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < alpha

    # 90% CI for the difference (corresponds to alpha=0.05 two-sided, i.e., 1-2*alpha)
    z_90 = stats.norm.ppf(1 - alpha)        # 1.645 for alpha=0.05
    ci_lower = diff - z_90 * se
    ci_upper = diff + z_90 * se

    return {
        "p1": p1_hat,
        "p2": p2_hat,
        "n1": n1,
        "n2": n2,
        "diff": diff,
        "se": se,
        "z_lower": z_lower,
        "z_upper": z_upper,
        "p_lower": p_lower,
        "p_upper": p_upper,
        "p_tost": p_tost,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "delta": delta,
        "alpha": alpha,
        "equivalent": equivalent,
    }

# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_pval(p: float) -> str:
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}"


def make_table(pairs, data, delta, alpha=0.05) -> str:
    header = (
        f"{'Comparison':<18}  {'RSR_A':>6}  {'RSR_B':>6}  {'Diff':>7}  "
        f"{'90% CI':^15}  {'z_lo':>7}  {'z_up':>7}  {'p_TOST':>7}  {'Equiv?':>7}"
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for cA, cB in pairs:
        r = tost_proportions(data[cA], data[cB], delta=delta, alpha=alpha)
        comp = f"{cA} vs {cB}"
        ci_str = f"[{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}]"
        equiv_str = "YES ***" if r["equivalent"] else "no"
        row = (
            f"{comp:<18}  {r['p1']:>6.3f}  {r['p2']:>6.3f}  {r['diff']:>+7.3f}  "
            f"{ci_str:^15}  {r['z_lower']:>7.3f}  {r['z_upper']:>7.3f}  "
            f"{fmt_pval(r['p_tost']):>7}  {equiv_str:>7}"
        )
        rows.append(row)

    return "\n".join(rows)


def interpretation(pairs, data, delta, alpha=0.05) -> str:
    lines = []
    equiv_pairs = []
    non_equiv = []
    for cA, cB in pairs:
        r = tost_proportions(data[cA], data[cB], delta=delta, alpha=alpha)
        if r["equivalent"]:
            equiv_pairs.append((cA, cB, r))
        else:
            non_equiv.append((cA, cB, r))

    if equiv_pairs:
        lines.append(f"  Equivalent pairs (p_TOST < {alpha}, |diff| within ±{delta:.0%}):")
        for cA, cB, r in equiv_pairs:
            lines.append(
                f"    {cA} vs {cB}: diff={r['diff']:+.3f}, "
                f"90% CI [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}], "
                f"p_TOST={fmt_pval(r['p_tost'])}"
            )
    else:
        lines.append("  No pairs declared equivalent.")

    if non_equiv:
        lines.append(f"  Non-equivalent pairs (p_TOST >= {alpha}):")
        for cA, cB, r in non_equiv:
            lines.append(
                f"    {cA} vs {cB}: diff={r['diff']:+.3f}, "
                f"90% CI [{r['ci_lower']:+.3f}, {r['ci_upper']:+.3f}], "
                f"p_TOST={fmt_pval(r['p_tost'])}"
            )

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # Load data
    model_dirs = find_model_dirs()
    assert len(model_dirs) == 5, f"Expected 5 models, found {len(model_dirs)}: {list(model_dirs)}"
    data = load_all_data(model_dirs)

    n_per_condition = len(data["C1"])
    assert n_per_condition == 530, f"Expected 530 obs per condition, got {n_per_condition}"

    # All 15 pairwise comparisons
    all_pairs = list(combinations(CONDITIONS, 2))

    # Build output lines
    out_lines = []

    out_lines.append("=" * 80)
    out_lines.append("TOST Equivalence Analysis — Exp.4v2")
    out_lines.append("=" * 80)
    out_lines.append(f"Models: {', '.join(sorted(model_dirs.keys()))}")
    out_lines.append(f"N per condition: {n_per_condition} (5 models × 106 scenarios)")
    out_lines.append(f"Conditions: {', '.join(CONDITIONS)}")
    out_lines.append(f"Significance level: α = 0.05")
    out_lines.append("")

    # Observed RSR per condition
    out_lines.append("Pooled RSR per condition:")
    out_lines.append(f"  {'Condition':<10}  {'RSR':>6}  {'n_success':>10}  {'n_total':>8}")
    out_lines.append("  " + "-" * 42)
    for c in CONDITIONS:
        arr = data[c]
        lbl = CONDITION_LABELS[c]
        out_lines.append(
            f"  {lbl:<20}  {arr.mean():.4f}  {arr.sum():>10}  {len(arr):>8}"
        )
    out_lines.append("")

    for delta in MARGINS:
        alpha = 0.05
        out_lines.append("=" * 80)
        out_lines.append(f"Margin: δ = {delta:.2f} ({delta*100:.0f}pp)")
        out_lines.append("=" * 80)
        out_lines.append("")

        # --- Priority comparisons ---
        out_lines.append("Priority Comparisons")
        out_lines.append("-" * 80)
        out_lines.append(make_table(PRIORITY_PAIRS, data, delta=delta, alpha=alpha))
        out_lines.append("")
        out_lines.append("Interpretation:")
        out_lines.append(interpretation(PRIORITY_PAIRS, data, delta=delta, alpha=alpha))
        out_lines.append("")

        # --- All 15 pairwise ---
        out_lines.append("All 15 Pairwise Comparisons")
        out_lines.append("-" * 80)
        out_lines.append(make_table(all_pairs, data, delta=delta, alpha=alpha))
        out_lines.append("")
        out_lines.append("Interpretation:")
        out_lines.append(interpretation(all_pairs, data, delta=delta, alpha=alpha))
        out_lines.append("")

    # Per-model RSR breakdown (informational)
    out_lines.append("=" * 80)
    out_lines.append("Per-Model RSR Breakdown (informational)")
    out_lines.append("=" * 80)
    header = f"  {'Model':<20}" + "".join(f"  {c:>7}" for c in CONDITIONS)
    out_lines.append(header)
    out_lines.append("  " + "-" * (20 + 9 * len(CONDITIONS)))
    for model in sorted(model_dirs.keys()):
        path = model_dirs[model]
        row = f"  {model:<20}"
        for c in CONDITIONS:
            arr = load_successes(path, c)
            row += f"  {arr.mean():.4f}"
        out_lines.append(row)
    out_lines.append("")

    # TOST method note
    out_lines.append("=" * 80)
    out_lines.append("Method Notes")
    out_lines.append("=" * 80)
    out_lines.append(
        "TOST procedure for two independent proportions (Schuirmann 1987).\n"
        "Two one-sided z-tests:\n"
        "  H0_lower: p_A - p_B <= -δ   (lower equivalence boundary)\n"
        "  H0_upper: p_A - p_B >=  δ   (upper equivalence boundary)\n"
        "Equivalence declared when BOTH nulls rejected at α=0.05,\n"
        "equivalently when the 90% CI for (p_A - p_B) lies entirely within [-δ, +δ].\n"
        "SE computed as sqrt(p1(1-p1)/n1 + p2(1-p2)/n2) using observed proportions.\n"
        "p_TOST = max(p_lower, p_upper)."
    )
    out_lines.append("")

    output = "\n".join(out_lines)

    # Write file
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)

    # Print to stdout
    print(output)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
