"""
TOST (Two One-Sided Tests) Equivalence Analysis for Exp.4v2 — CORRECTED.

`scripts/exp4v2_tost.py` reads the `remediation_success` field straight out of each
model's raw `condition_*.json` files. That field is identical to those files'
`tier2_correct` field (verified: matches for all 3180 records) — i.e. it is the
PRE-FIX, stale judge verdict from before the verification-gold-mismatch correction.

This script runs the exact same TOST procedure (see Method Notes at the bottom of the
output, unchanged from the original) but pools the CORRECTED verdict instead: the
`correct` field of `results/exp4v2/tier2_corrected/tier2_recomputed.json`, joined on
(scenario_id, model, condition) — a unique key (3180 records, 0 duplicates, 0 missing).
That file's own `original_tier2` field is confirmed to reproduce the raw
`remediation_success`/`tier2_correct` values exactly, corroborating the join is correct.

Usage: conda run -n X python scripts/exp4v2_tost_corrected.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import json
import os
from collections import defaultdict
from itertools import combinations

import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(ROOT, "results", "exp4v2")
TIER2_CORRECTED_FILE = os.path.join(RESULTS_DIR, "tier2_corrected", "tier2_recomputed.json")
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

EXPECTED_MODELS = {"qwen-2.5-7b", "mistral-7b", "llama-3.1-8b", "phi-4", "qwen3.5-9b"}

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
# Data loading — from the corrected tier2 artifact, not the raw condition files
# ---------------------------------------------------------------------------

def load_all_data():
    """
    Returns {condition: np.ndarray of shape (n_models * 106,)} using the CORRECTED
    `correct` verdict from tier2_recomputed.json, pooled across all 5 models.
    """
    with open(TIER2_CORRECTED_FILE) as f:
        records = json.load(f)

    by_model_cond = defaultdict(list)
    seen_keys = set()
    for r in records:
        key = (r["scenario_id"], r["model"], r["condition"])
        assert key not in seen_keys, f"Duplicate key: {key}"
        seen_keys.add(key)
        by_model_cond[(r["model"], r["condition"])].append(1 if r["correct"] else 0)

    models = sorted({m for (m, _c) in by_model_cond.keys()})
    assert set(models) == EXPECTED_MODELS, f"Unexpected model set: {models}"

    pooled = {}
    for cond in CONDITIONS:
        vals = []
        for model in models:
            vals.extend(by_model_cond[(model, cond)])
        pooled[cond] = np.array(vals, dtype=int)

    return pooled, models, len(records)

# ---------------------------------------------------------------------------
# TOST for difference of two proportions (identical to exp4v2_tost.py)
# ---------------------------------------------------------------------------

def tost_proportions(x1: np.ndarray, x2: np.ndarray, delta: float, alpha: float = 0.05):
    n1, n2 = len(x1), len(x2)
    p1_hat = x1.mean()
    p2_hat = x2.mean()
    diff = p1_hat - p2_hat

    se = np.sqrt(p1_hat * (1 - p1_hat) / n1 + p2_hat * (1 - p2_hat) / n2)

    z_lower = (diff + delta) / se
    z_upper = (diff - delta) / se

    p_lower = 1 - stats.norm.cdf(z_lower)
    p_upper = stats.norm.cdf(z_upper)

    p_tost = max(p_lower, p_upper)
    equivalent = p_tost < alpha

    z_90 = stats.norm.ppf(1 - alpha)
    ci_lower = diff - z_90 * se
    ci_upper = diff + z_90 * se

    return {
        "p1": p1_hat, "p2": p2_hat, "n1": n1, "n2": n2, "diff": diff, "se": se,
        "z_lower": z_lower, "z_upper": z_upper, "p_lower": p_lower, "p_upper": p_upper,
        "p_tost": p_tost, "ci_lower": ci_lower, "ci_upper": ci_upper,
        "delta": delta, "alpha": alpha, "equivalent": equivalent,
    }

# ---------------------------------------------------------------------------
# Formatting helpers (identical to exp4v2_tost.py)
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
    data, models, n_records = load_all_data()

    n_per_condition = len(data["C1"])
    assert n_per_condition == 530, f"Expected 530 obs per condition, got {n_per_condition}"

    all_pairs = list(combinations(CONDITIONS, 2))

    out_lines = []
    out_lines.append("=" * 80)
    out_lines.append("TOST Equivalence Analysis — Exp.4v2 (CORRECTED, verification-gold-mismatch fix)")
    out_lines.append("=" * 80)
    out_lines.append(
        "Source: results/exp4v2/tier2_corrected/tier2_recomputed.json ('correct' field), "
        "joined on (scenario_id, model, condition). This supersedes the pre-fix pooled RSRs "
        "that scripts/exp4v2_tost.py computed from the raw condition_*.json 'remediation_success' "
        "field (== that file's own stale 'tier2_correct'; confirmed identical to "
        "tier2_recomputed.json's 'original_tier2' for all 3180 records)."
    )
    out_lines.append(f"Models: {', '.join(models)}")
    out_lines.append(f"N per condition: {n_per_condition} (5 models × 106 scenarios)")
    out_lines.append(f"Conditions: {', '.join(CONDITIONS)}")
    out_lines.append(f"Significance level: α = 0.05")
    out_lines.append("")

    out_lines.append("Pooled RSR per condition (corrected):")
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

        out_lines.append("Priority Comparisons")
        out_lines.append("-" * 80)
        out_lines.append(make_table(PRIORITY_PAIRS, data, delta=delta, alpha=alpha))
        out_lines.append("")
        out_lines.append("Interpretation:")
        out_lines.append(interpretation(PRIORITY_PAIRS, data, delta=delta, alpha=alpha))
        out_lines.append("")

        out_lines.append("All 15 Pairwise Comparisons")
        out_lines.append("-" * 80)
        out_lines.append(make_table(all_pairs, data, delta=delta, alpha=alpha))
        out_lines.append("")
        out_lines.append("Interpretation:")
        out_lines.append(interpretation(all_pairs, data, delta=delta, alpha=alpha))
        out_lines.append("")

    # Per-model RSR breakdown (informational) — recomputed from corrected data
    with open(TIER2_CORRECTED_FILE) as f:
        records = json.load(f)
    per_model_cond = defaultdict(list)
    for r in records:
        per_model_cond[(r["model"], r["condition"])].append(1 if r["correct"] else 0)

    out_lines.append("=" * 80)
    out_lines.append("Per-Model RSR Breakdown (informational, corrected)")
    out_lines.append("=" * 80)
    header = f"  {'Model':<20}" + "".join(f"  {c:>7}" for c in CONDITIONS)
    out_lines.append(header)
    out_lines.append("  " + "-" * (20 + 9 * len(CONDITIONS)))
    for model in models:
        row = f"  {model:<20}"
        for c in CONDITIONS:
            arr = np.array(per_model_cond[(model, c)], dtype=int)
            row += f"  {arr.mean():.4f}"
        out_lines.append(row)
    out_lines.append("")

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

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)

    print(output)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
