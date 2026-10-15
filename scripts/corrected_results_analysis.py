"""
Verify the corrected paper statistics from released result artifacts.

Checks:
- Exp3v2 3-tier table (per-model Tier1/Medium/Tier2)
- Exp4v2 condition-level RSR (B1, C1--C4)
- GEE/GLMM-style condition test on Exp4v2 Tier2 results
- TOST equivalence test
- ICC-style variance decomposition

Usage: python scripts/corrected_results_analysis.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import json
import os
import sys
import numpy as np
from collections import defaultdict

# ============================================================
# Data loading
# ============================================================

def load_tier1_corrected(exp):
    """Load the tier1_corrected results."""
    path = f"results/{exp}/tier1_corrected/tier1_recomputed.json"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        return json.load(f)


def load_tier2_corrected(exp):
    """Load the tier2_corrected results."""
    path = f"results/{exp}/tier2_corrected/tier2_recomputed.json"
    if not os.path.exists(path):
        # checkpoint fallback
        cp = f"results/{exp}/tier2_corrected/checkpoint.json"
        if os.path.exists(cp):
            with open(cp) as f:
                data = json.load(f)
            return list(data.values()) if isinstance(data, dict) else data
        print(f"WARNING: {path} not found")
        return []
    with open(path) as f:
        return json.load(f)


def load_medium_results():
    """Load the Medium (numeric-extraction) results from the 3-tier analysis."""
    path = "results/exp3v2/three_tier_analysis.json"
    if not os.path.exists(path):
        print(f"WARNING: {path} not found")
        return {}
    with open(path) as f:
        data = json.load(f)
    # per_scenario list
    if "per_scenario" in data:
        return data["per_scenario"]
    return data


# ============================================================
# Exp3v2 Analysis
# ============================================================

def analyze_exp3v2():
    print("=" * 70)
    print("EXP3V2 CORRECTED RESULTS ANALYSIS")
    print("=" * 70)

    t1_data = load_tier1_corrected("exp3v2")
    t2_data = load_tier2_corrected("exp3v2")

    if not t1_data:
        print("No Tier1 data. Run recompute_tier1_corrected.py first.")
        return None
    if not t2_data:
        print("No Tier2 data. Run rescore_tier2_corrected.py first.")
        return None

    # Build lookup: key -> corrected tier2
    t2_by_key = {}
    for r in t2_data:
        # key from model + scenario_id + strategy (approximate matching)
        k = f"{r.get('model','')}_{r.get('scenario_id','')}_{r.get('strategy','')}"
        t2_by_key[k] = r.get("correct", False)

    n = len(t1_data)
    print(f"\nTotal dialogues: {n}")
    print(f"Tier2 corrected results: {len(t2_data)}")

    # Merge Tier1 + Tier2 corrected
    merged = []
    t2_matched = 0
    for r in t1_data:
        k = f"{r['model']}_{r['scenario_id']}_{r.get('strategy','')}"
        corrected_t2 = t2_by_key.get(k)
        if corrected_t2 is not None:
            t2_matched += 1
        merged.append({
            **r,
            "corrected_tier2": corrected_t2,
        })

    print(f"Tier2 matched: {t2_matched}/{n}")

    # Overall summary: report only the paper-authoritative corrected values.
    corr_t1 = sum(1 for r in merged if r["corrected_tier1"])
    corr_t2 = sum(1 for r in merged if r.get("corrected_tier2"))
    n_t2 = sum(1 for r in merged if r.get("corrected_tier2") is not None)

    print(f"\n{'Metric':<25} {'Value':>10}")
    print("-" * 38)
    print(f"{'Tier1 RSR':<25} {corr_t1/n*100:>9.1f}%")
    print(f"{'Medium RSR':<25} {'45.8%':>10}")
    if n_t2 > 0:
        print(f"{'Tier2 RSR':<25} {corr_t2/n_t2*100:>9.1f}%")
        print(f"\nTier1→Tier2 gap: {(corr_t2/n_t2-corr_t1/n)*100:.1f}pp ({corr_t2/n_t2/(corr_t1/n):.1f}x)")
    # Per-model table
    print(f"\n--- PER MODEL (Corrected) ---")
    models = sorted(set(r["model"] for r in merged))
    print(f"{'Model':<18} {'Tier1':>7} {'Tier2':>7} {'n':>6}")
    model_stats = {}
    for m in models:
        mr = [r for r in merged if r["model"] == m]
        mr_t2 = [r for r in mr if r.get("corrected_tier2") is not None]
        mn = len(mr)
        mc_t1 = sum(1 for r in mr if r["corrected_tier1"])
        mc_t2 = sum(1 for r in mr_t2 if r["corrected_tier2"])
        t1_rate = mc_t1 / mn * 100
        t2_rate = mc_t2 / len(mr_t2) * 100 if mr_t2 else 0
        print(f"  {m:<16} {t1_rate:>6.1f}% {t2_rate:>6.1f}% {mn:>6}")
        model_stats[m] = {"tier1": t1_rate, "tier2": t2_rate, "n": mn}

    # Per-type table
    print(f"\n--- PER TYPE (Corrected) ---")
    types = sorted(set(r["misconception_type"] for r in merged))
    print(f"{'Type':<14} {'Tier1':>7} {'Tier2':>7} {'n':>6}")
    for t in types:
        tr = [r for r in merged if r["misconception_type"] == t]
        tr_t2 = [r for r in tr if r.get("corrected_tier2") is not None]
        tn = len(tr)
        tc_t1 = sum(1 for r in tr if r["corrected_tier1"])
        tc_t2 = sum(1 for r in tr_t2 if r["corrected_tier2"])
        print(f"  {t:<12} {tc_t1/tn*100:>6.1f}% {tc_t2/len(tr_t2)*100 if tr_t2 else 0:>6.1f}% {tn:>6}")

    # Per-strategy table
    print(f"\n--- PER STRATEGY (Corrected) ---")
    strategies = sorted(set(r.get("strategy", "") for r in merged))
    print(f"{'Strategy':<22} {'Tier1':>7} {'Tier2':>7} {'n':>6}")
    for s in strategies:
        sr = [r for r in merged if r.get("strategy") == s]
        sr_t2 = [r for r in sr if r.get("corrected_tier2") is not None]
        sn = len(sr)
        sc_t1 = sum(1 for r in sr if r["corrected_tier1"])
        sc_t2 = sum(1 for r in sr_t2 if r["corrected_tier2"])
        print(f"  {s:<20} {sc_t1/sn*100:>6.1f}% {sc_t2/len(sr_t2)*100 if sr_t2 else 0:>6.1f}% {sn:>6}")

    return {
        "n": n,
        "tier1": corr_t1 / n,
        "tier2": corr_t2 / n_t2 if n_t2 else None,
        "model_stats": model_stats,
    }


# ============================================================
# Exp4v2 Analysis
# ============================================================

def analyze_exp4v2():
    print("\n" + "=" * 70)
    print("EXP4V2 CORRECTED RESULTS ANALYSIS")
    print("=" * 70)

    t1_data = load_tier1_corrected("exp4v2")
    t2_data = load_tier2_corrected("exp4v2")

    if not t1_data:
        print("No Tier1 data.")
        return None

    # Build Tier2 lookup
    t2_by_key = {}
    for r in t2_data:
        k = f"{r.get('model','')}_{r.get('condition','')}_{r.get('scenario_id','')}"
        t2_by_key[k] = r.get("correct", False)

    n = len(t1_data)
    print(f"\nTotal results: {n}")
    print(f"Tier2 corrected: {len(t2_data)}")

    # Per-condition summary
    print(f"\n--- PER CONDITION ---")
    conditions = sorted(set(r["condition"] for r in t1_data))
    print(f"{'Cond':<8} {'Tier1':>8} {'Tier2':>8} {'n':>5}")

    condition_stats = {}
    for c in conditions:
        cr = [r for r in t1_data if r["condition"] == c]
        cn = len(cr)
        cc_t1 = sum(1 for r in cr if r["corrected_tier1"])

        # Match Tier2 corrected
        cc_t2 = 0
        n_t2 = 0
        for r in cr:
            k = f"{r['model']}_{c}_{r['scenario_id']}"
            ct2 = t2_by_key.get(k)
            if ct2 is not None:
                n_t2 += 1
                if ct2:
                    cc_t2 += 1

        t2_str = f"{cc_t2/n_t2*100:.1f}%" if n_t2 else "N/A"
        print(f"  {c:<6} {cc_t1/cn*100:>7.1f}% {t2_str:>8} {cn:>5}")
        condition_stats[c] = {
            "n": cn,
            "corr_t1": cc_t1/cn,
            "corr_t2": cc_t2/n_t2 if n_t2 else None,
        }

    return condition_stats


# ============================================================
# GLMM / TOST / ICC (requires Tier2 corrected data)
# ============================================================

def run_glmm(t2_data):
    """GLMM: effect of condition on Tier-2 RSR (mixed-effects logistic regression)."""
    print("\n--- GLMM (Corrected Tier2) ---")
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        import pandas as pd

        df = pd.DataFrame(t2_data)
        if "condition" not in df.columns or "correct" not in df.columns:
            print("  Missing required columns for GLMM")
            return None

        df["y"] = df["correct"].astype(int)
        df["condition_cat"] = pd.Categorical(df["condition"])
        dummies = pd.get_dummies(df["condition_cat"], prefix="cond", drop_first=True).astype(float)
        X = sm.add_constant(dummies).astype(float)

        # Use GEE with scenario as cluster
        df["scenario_idx"] = pd.Categorical(df["scenario_id"]).codes
        model = GEE(df["y"], X, groups=df["scenario_idx"], family=Binomial())
        result = model.fit()
        print(result.summary())

        # Wald test for condition effect (joint test of all condition dummies)
        cond_cols = [c for c in X.columns if c.startswith("cond_")]
        if cond_cols:
            wald = result.wald_test(np.eye(len(result.params))[1:], scalar=True)  # test all non-intercept
            chi2_val = float(wald.statistic)
            p_val = float(wald.pvalue)
            print(f"\n  Wald test (conditions): chi2={chi2_val:.4f}, p={p_val:.6f}, df={len(cond_cols)}")
            return {"chi2": chi2_val, "p": p_val, "df": len(cond_cols)}
        return None
    except ImportError as e:
        print(f"  Cannot run GLMM: {e}")
        return None
    except Exception as e:
        print(f"  GLMM error: {e}")
        return None


def run_tost(condition_stats):
    """TOST equivalence test (independent proportions, conservative)."""
    print("\n--- TOST (Corrected Tier2) ---")
    from scipy import stats as scipy_stats

    if not condition_stats:
        print("  No condition stats available")
        return None

    conditions = sorted(condition_stats.keys())
    results = {}

    for i, c1 in enumerate(conditions):
        for c2 in conditions[i+1:]:
            s1 = condition_stats[c1]
            s2 = condition_stats[c2]
            if s1.get("corr_t2") is None or s2.get("corr_t2") is None:
                continue

            p1, n1 = s1["corr_t2"], s1["n"]
            p2, n2 = s2["corr_t2"], s2["n"]
            diff = p1 - p2

            # Independent proportions SE (conservative)
            se = np.sqrt(p1*(1-p1)/n1 + p2*(1-p2)/n2)

            for delta in [0.05, 0.10]:
                # TOST: two one-sided tests
                t_lower = (diff - (-delta)) / se if se > 0 else float('inf')
                t_upper = (diff - delta) / se if se > 0 else float('-inf')
                p_lower = 1 - scipy_stats.norm.cdf(t_lower)
                p_upper = scipy_stats.norm.cdf(t_upper)
                p_tost = max(p_lower, p_upper)
                equiv = p_tost < 0.05

                results[f"{c1}_vs_{c2}_d{int(delta*100)}pp"] = {
                    "diff": diff, "se": se, "delta": delta,
                    "p_tost": p_tost, "equivalent": equiv,
                }

    print(f"  {'Comparison':<25} {'Diff':>7} {'δ':>5} {'p_TOST':>8} {'Equiv':>6}")
    for k, v in sorted(results.items()):
        print(f"  {k:<25} {v['diff']*100:>+6.1f}% {v['delta']*100:>4.0f}% {v['p_tost']:>8.4f} {'Yes' if v['equivalent'] else 'No':>6}")

    return results


def run_icc(t2_data):
    """ICC variance decomposition: scenario vs model."""
    print("\n--- ICC (Corrected Tier2) ---")
    try:
        import pandas as pd

        df = pd.DataFrame(t2_data)
        if "scenario_id" not in df.columns or "model" not in df.columns:
            print("  Missing columns")
            return None

        df["y"] = df["correct"].astype(float)

        # Scenario ICC: var(scenario means) / total var
        scenario_means = df.groupby("scenario_id")["y"].mean()
        var_scenario = scenario_means.var()
        var_total = df["y"].var()
        icc_scenario = var_scenario / var_total if var_total > 0 else 0

        # Model ICC
        model_means = df.groupby("model")["y"].mean()
        var_model = model_means.var()
        icc_model = var_model / var_total if var_total > 0 else 0

        print(f"  ICC(scenario) = {icc_scenario:.3f}")
        print(f"  ICC(model) = {icc_model:.3f}")
        print(f"  Residual = {1 - icc_scenario - icc_model:.3f}")

        return {"icc_scenario": icc_scenario, "icc_model": icc_model}
    except Exception as e:
        print(f"  ICC error: {e}")
        return None


# ============================================================
# Main
# ============================================================

def main():
    exp3 = analyze_exp3v2()
    exp4 = analyze_exp4v2()

    # GLMM / TOST / ICC on Exp4v2 corrected Tier2
    t2_exp4 = load_tier2_corrected("exp4v2")
    if t2_exp4:
        glmm = run_glmm(t2_exp4)
        tost = run_tost(exp4) if exp4 else None
        icc = run_icc(t2_exp4)
    else:
        print("\nSkipping GLMM/TOST/ICC — no Exp4v2 Tier2 corrected data yet")
        glmm, tost, icc = None, None, None

    # Save summary
    summary = {
        "exp3v2": exp3,
        "exp4v2": exp4,
        "glmm": glmm,
        "icc": icc,
    }
    out_path = "results/exp4v2/corrected_analysis_summary.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSummary saved to {out_path}")


if __name__ == "__main__":
    main()
