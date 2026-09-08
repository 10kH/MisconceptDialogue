#!/usr/bin/env python3
"""
Comprehensive statistical analysis of exp3v2 three-tier evaluation results.

Tests performed:
  1. McNemar tests for pairwise tier comparisons (Tier1 vs Medium, etc.)
  2. Cochran's Q test (omnibus 3-way comparison)
  3. GLMM / GEE for misconception-type effect on Tier2 RSR
  4. Strategy effect on Tier2 RSR
  5. Type x Strategy interaction
  6. Bootstrap 95% CIs for all tier RSR estimates
  7. Tier gap analysis with bootstrap CIs
  8. Answer-revealing analysis

Usage:
    conda run -n X python scripts/exp3v2_statistical_analysis.py
"""

import json
import os
import sys
import warnings
from pathlib import Path
from io import StringIO

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE = PROJECT_ROOT / "results" / "exp3v2"
THREE_TIER_JSON = BASE / "three_tier_analysis.json"
OUTPUT_TXT = BASE / "statistical_analysis_3tier.txt"

MODEL_DIRS = {
    "Qwen-2.5-7B": "20260325_103441_qwen-2.5-7b",
    "Llama-3.1-8B": "20260325_103441_llama-3.1-8b",
    "Mistral-7B": "20260325_103441_mistral-7b",
    "Phi-4": "20260326_010332_phi-4",
    "Qwen3.5-9B": "20260326_010332_qwen3.5-9b",
}

SEED = 42
N_BOOTSTRAP = 10_000

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
output_buf = StringIO()


def pr(text=""):
    """Print to both stdout and buffer."""
    print(text)
    output_buf.write(text + "\n")


def section(title):
    pr()
    pr("=" * 72)
    pr(f"  {title}")
    pr("=" * 72)


def subsection(title):
    pr()
    pr(f"--- {title} ---")


def bootstrap_ci(data, stat_fn=np.mean, n_boot=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI. Returns (estimate, lo, hi)."""
    rng = np.random.RandomState(seed)
    n = len(data)
    estimate = stat_fn(data)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = stat_fn(data[idx])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return estimate, lo, hi


def bootstrap_paired_diff_ci(a, b, n_boot=N_BOOTSTRAP, seed=SEED, alpha=0.05):
    """Bootstrap CI for the mean difference (a - b) on paired binary arrays."""
    rng = np.random.RandomState(seed)
    n = len(a)
    diff = a - b
    est = np.mean(diff)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boots[i] = np.mean(diff[idx])
    lo = np.percentile(boots, 100 * alpha / 2)
    hi = np.percentile(boots, 100 * (1 - alpha / 2))
    return est, lo, hi


def mcnemar_test(a, b, label):
    """McNemar test on paired binary arrays a, b. Print results."""
    # Contingency: a=1,b=1 | a=1,b=0
    #              a=0,b=1 | a=0,b=0
    n11 = int(np.sum((a == 1) & (b == 1)))
    n10 = int(np.sum((a == 1) & (b == 0)))
    n01 = int(np.sum((a == 0) & (b == 1)))
    n00 = int(np.sum((a == 0) & (b == 0)))
    pr(f"  Contingency table ({label}):")
    pr(f"    +/+ = {n11}  +/- = {n10}")
    pr(f"    -/+ = {n01}  -/- = {n00}")
    discordant = n10 + n01
    if discordant == 0:
        pr(f"  No discordant pairs; McNemar test not applicable.")
        return
    # Use exact (binomial) if discordant < 25, else chi-squared
    if discordant < 25:
        method = "exact (binomial)"
        result = stats.binomtest(n10, n10 + n01, 0.5)
        p_val = result.pvalue
        chi2 = None
    else:
        method = "chi-squared (continuity-corrected)"
        chi2 = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
        p_val = stats.chi2.sf(chi2, df=1)
    # Odds ratio (n10/n01) with Wald 95% CI on log scale
    # Use 0.5 continuity correction for zero cells
    n10c = max(n10, 0.5)
    n01c = max(n01, 0.5)
    odds_ratio = n10c / n01c
    log_or = np.log(odds_ratio)
    se_log_or = np.sqrt(1.0 / n10c + 1.0 / n01c)
    or_lo = np.exp(log_or - 1.96 * se_log_or)
    or_hi = np.exp(log_or + 1.96 * se_log_or)
    pr(f"  Method: {method}")
    if chi2 is not None:
        pr(f"  chi2 = {chi2:.4f}")
    pr(f"  p-value = {p_val:.6g}")
    if n10 == 0 or n01 == 0:
        pr(f"  Odds Ratio = {n10}/{n01} (zero cell; OR with 0.5 correction = {odds_ratio:.4f})")
        pr(f"  95% CI (0.5-corrected): [{or_lo:.4f}, {or_hi:.4f}]")
    else:
        pr(f"  Odds Ratio = {odds_ratio:.4f}  95% CI [{or_lo:.4f}, {or_hi:.4f}]")


def cochrans_q(binary_matrix):
    """Cochran's Q test. binary_matrix: (n_subjects, k_treatments)."""
    n, k = binary_matrix.shape
    row_sums = binary_matrix.sum(axis=1)  # L_i
    col_sums = binary_matrix.sum(axis=0)  # T_j
    T = col_sums
    L = row_sums
    grand_total = binary_matrix.sum()
    numerator = (k - 1) * (k * np.sum(T ** 2) - grand_total ** 2)
    denominator = k * grand_total - np.sum(L ** 2)
    if denominator == 0:
        return float("nan"), k - 1, float("nan")
    Q = numerator / denominator
    df = k - 1
    p_val = stats.chi2.sf(Q, df)
    return Q, df, p_val


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
def load_data():
    with open(THREE_TIER_JSON) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw["per_scenario"])
    # Convert booleans to int
    for col in ["tier1", "medium", "tier2"]:
        df[col] = df[col].astype(int)

    # Load answer_revealed from cell files
    ar_map = {}  # (model, scenario_id) -> bool
    for model_name, dir_name in MODEL_DIRS.items():
        model_dir = os.path.join(BASE, dir_name)
        for fname in os.listdir(model_dir):
            if fname.startswith("cell_") and fname.endswith(".json"):
                with open(os.path.join(model_dir, fname)) as f:
                    cell = json.load(f)
                for r in cell["results"]:
                    key = (model_name, r["scenario_id"])
                    ar_map[key] = bool(r.get("answer_revealed", False))
    df["answer_revealed"] = df.apply(
        lambda row: ar_map.get((row["model"], row["scenario_id"]), False), axis=1
    )
    return df


# ---------------------------------------------------------------------------
# 1. McNemar Tests
# ---------------------------------------------------------------------------
def test_mcnemar(df):
    section("1. McNemar Tests for Tier Comparisons (Paired, n=2250)")

    t1 = df["tier1"].values
    med = df["medium"].values
    t2 = df["tier2"].values

    subsection("Tier1 vs Medium")
    mcnemar_test(t1, med, "Tier1 vs Medium")

    subsection("Tier1 vs Tier2")
    mcnemar_test(t1, t2, "Tier1 vs Tier2")

    subsection("Medium vs Tier2")
    mcnemar_test(med, t2, "Medium vs Tier2")


# ---------------------------------------------------------------------------
# 2. Cochran's Q
# ---------------------------------------------------------------------------
def test_cochran(df):
    section("2. Cochran's Q Test (Omnibus: Tier1 vs Medium vs Tier2)")

    mat = df[["tier1", "medium", "tier2"]].values
    Q, dof, p = cochrans_q(mat)
    pr(f"  Q = {Q:.4f}")
    pr(f"  df = {dof}")
    pr(f"  p-value = {p:.6g}")


# ---------------------------------------------------------------------------
# 3. GLMM / GEE for Type Effect on Tier2
# ---------------------------------------------------------------------------
def test_type_effect(df):
    section("3. Type Effect on Tier2 RSR (GLMM / GEE)")

    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        import statsmodels.api as sm

        # Create dummy variables for type (reference = conceptual)
        df_model = df.copy()
        df_model["type_factual"] = (df_model["misconception_type"] == "factual").astype(int)
        df_model["type_procedural"] = (df_model["misconception_type"] == "procedural").astype(int)

        # Encode model as numeric group for GEE
        model_codes = {m: i for i, m in enumerate(sorted(df_model["model"].unique()))}
        df_model["model_code"] = df_model["model"].map(model_codes)

        # GEE with exchangeable correlation, grouped by model
        endog = df_model["tier2"]
        exog = sm.add_constant(df_model[["type_factual", "type_procedural"]])
        groups = df_model["model_code"]

        gee_model = GEE(endog, exog, groups=groups, family=Binomial(), cov_struct=Exchangeable())
        gee_result = gee_model.fit()

        pr("  GEE (logistic, exchangeable correlation, grouped by model)")
        pr(f"  {'Parameter':<20} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>10} {'OR':>8} {'OR 95% CI':>18}")
        pr(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8} {'-'*18}")
        params = gee_result.params
        bse = gee_result.bse
        pvalues = gee_result.pvalues
        conf = gee_result.conf_int()
        for name in params.index:
            coef = params[name]
            se = bse[name]
            z = coef / se if se > 0 else float("nan")
            p = pvalues[name]
            or_val = np.exp(coef)
            or_lo = np.exp(conf.loc[name, 0])
            or_hi = np.exp(conf.loc[name, 1])
            pr(f"  {name:<20} {coef:>8.4f} {se:>8.4f} {z:>8.3f} {p:>10.6f} {or_val:>8.3f} [{or_lo:.3f}, {or_hi:.3f}]")

        pr()
        pr(f"  Reference category: conceptual")
        pr(f"  Scale: {gee_result.scale:.4f}")

    except Exception as e:
        pr(f"  GEE failed: {e}")
        pr("  Falling back to chi-squared test for type effect on Tier2")
        ct = pd.crosstab(df["misconception_type"], df["tier2"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        pr(f"  Chi-squared = {chi2:.4f}, df = {dof}, p = {p:.6g}")


# ---------------------------------------------------------------------------
# 4. Strategy Effect on Tier2
# ---------------------------------------------------------------------------
def test_strategy_effect(df):
    section("4. Strategy Effect on Tier2 RSR")

    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        import statsmodels.api as sm

        df_model = df.copy()
        strategies = sorted(df_model["strategy"].unique())
        ref = strategies[0]  # analogy as reference
        for s in strategies[1:]:
            df_model[f"strat_{s}"] = (df_model["strategy"] == s).astype(int)

        model_codes = {m: i for i, m in enumerate(sorted(df_model["model"].unique()))}
        df_model["model_code"] = df_model["model"].map(model_codes)

        strat_cols = [f"strat_{s}" for s in strategies[1:]]
        endog = df_model["tier2"]
        exog = sm.add_constant(df_model[strat_cols])
        groups = df_model["model_code"]

        gee_model = GEE(endog, exog, groups=groups, family=Binomial(), cov_struct=Exchangeable())
        gee_result = gee_model.fit()

        pr("  GEE (logistic, exchangeable correlation, grouped by model)")
        pr(f"  Reference strategy: {ref}")
        pr(f"  {'Parameter':<30} {'Coef':>8} {'SE':>8} {'z':>8} {'p':>10} {'OR':>8}")
        pr(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
        params = gee_result.params
        bse = gee_result.bse
        pvalues = gee_result.pvalues
        for name in params.index:
            coef = params[name]
            se = bse[name]
            z = coef / se if se > 0 else float("nan")
            p = pvalues[name]
            or_val = np.exp(coef)
            pr(f"  {name:<30} {coef:>8.4f} {se:>8.4f} {z:>8.3f} {p:>10.6f} {or_val:>8.3f}")

        # Joint Wald test for all strategy dummies
        try:
            # Test all strategy coefficients = 0
            r_matrix = np.zeros((len(strat_cols), len(params)))
            for i, col in enumerate(strat_cols):
                j = list(params.index).index(col)
                r_matrix[i, j] = 1
            wald_stat = gee_result.wald_test(r_matrix)
            pr(f"\n  Joint Wald test (all strategy effects = 0):")
            pr(f"  Statistic = {float(wald_stat.statistic):.4f}, "
               f"df = {len(strat_cols)}, p = {float(wald_stat.pvalue):.6g}")
        except Exception as e:
            pr(f"  Joint Wald test failed: {e}")

    except Exception as e:
        pr(f"  GEE failed: {e}")
        pr("  Falling back to chi-squared test for strategy effect on Tier2")
        ct = pd.crosstab(df["strategy"], df["tier2"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        pr(f"  Chi-squared = {chi2:.4f}, df = {dof}, p = {p:.6g}")


# ---------------------------------------------------------------------------
# 5. Type x Strategy Interaction
# ---------------------------------------------------------------------------
def test_interaction(df):
    section("5. Type x Strategy Interaction on Tier2 RSR")

    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        import statsmodels.api as sm

        df_model = df.copy()
        types = sorted(df_model["misconception_type"].unique())
        strategies = sorted(df_model["strategy"].unique())
        type_ref = types[0]  # conceptual
        strat_ref = strategies[0]  # analogy

        for t in types[1:]:
            df_model[f"type_{t}"] = (df_model["misconception_type"] == t).astype(int)
        for s in strategies[1:]:
            df_model[f"strat_{s}"] = (df_model["strategy"] == s).astype(int)
        interaction_cols = []
        for t in types[1:]:
            for s in strategies[1:]:
                col = f"type_{t}_x_strat_{s}"
                df_model[col] = df_model[f"type_{t}"] * df_model[f"strat_{s}"]
                interaction_cols.append(col)

        type_cols = [f"type_{t}" for t in types[1:]]
        strat_cols = [f"strat_{s}" for s in strategies[1:]]
        all_cols = type_cols + strat_cols + interaction_cols

        model_codes = {m: i for i, m in enumerate(sorted(df_model["model"].unique()))}
        df_model["model_code"] = df_model["model"].map(model_codes)

        endog = df_model["tier2"]
        groups = df_model["model_code"]

        # Fit FULL model (type + strategy + interaction)
        exog_full = sm.add_constant(df_model[all_cols])
        gee_full = GEE(endog, exog_full, groups=groups, family=Binomial(),
                        cov_struct=Exchangeable())
        result_full = gee_full.fit()

        # Fit REDUCED model (type + strategy, NO interaction)
        reduced_cols = type_cols + strat_cols
        exog_reduced = sm.add_constant(df_model[reduced_cols])
        gee_reduced = GEE(endog, exog_reduced, groups=groups, family=Binomial(),
                           cov_struct=Exchangeable())
        result_reduced = gee_reduced.fit()

        pr("  GEE with Type x Strategy interaction (full model)")
        pr(f"  Reference: type={type_ref}, strategy={strat_ref}")
        pr(f"  {'Parameter':<40} {'Coef':>8} {'SE':>8} {'p':>10}")
        pr(f"  {'-'*40} {'-'*8} {'-'*8} {'-'*10}")
        params = result_full.params
        bse = result_full.bse
        pvalues = result_full.pvalues
        for name in params.index:
            coef = params[name]
            se = bse[name]
            p = pvalues[name]
            pr(f"  {name:<40} {coef:>8.4f} {se:>8.4f} {p:>10.6f}")

        # QIC comparison: full vs reduced
        pr()
        try:
            qic_full = result_full.qic()
            qic_reduced = result_reduced.qic()
            pr(f"  QIC comparison (lower = better):")
            pr(f"    Full model  (type+strat+interaction): QIC={qic_full[0]:.2f}, QICu={qic_full[1]:.2f}")
            pr(f"    Reduced     (type+strat only):        QIC={qic_reduced[0]:.2f}, QICu={qic_reduced[1]:.2f}")
            delta_qic = qic_full[0] - qic_reduced[0]
            pr(f"    Delta QIC (full - reduced) = {delta_qic:.2f}")
            if delta_qic < 0:
                pr(f"    Interpretation: Full model preferred (interaction improves fit)")
            else:
                pr(f"    Interpretation: Reduced model preferred (interaction does not improve fit)")
        except Exception as e:
            pr(f"    QIC comparison failed: {e}")

        # Individual interaction term p-values summary
        pr()
        sig_interactions = [(n, pvalues[n]) for n in interaction_cols if pvalues[n] < 0.05]
        pr(f"  Significant interaction terms (p<0.05): {len(sig_interactions)} of {len(interaction_cols)}")
        for n, p in sig_interactions:
            pr(f"    {n}: p={p:.6f}")

    except Exception as e:
        pr(f"  GEE with interaction failed: {e}")

    # Always report stratified chi-squared as a complementary nonparametric test
    subsection("Stratified chi-squared: strategy effect within each type")
    for t in sorted(df["misconception_type"].unique()):
        sub = df[df["misconception_type"] == t]
        ct = pd.crosstab(sub["strategy"], sub["tier2"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        pr(f"  {t}: chi2={chi2:.3f}, df={dof}, p={p:.4g}")

    subsection("Stratified chi-squared: type effect within each strategy")
    for s in sorted(df["strategy"].unique()):
        sub = df[df["strategy"] == s]
        ct = pd.crosstab(sub["misconception_type"], sub["tier2"])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        pr(f"  {s}: chi2={chi2:.3f}, df={dof}, p={p:.4g}")


# ---------------------------------------------------------------------------
# 6. Bootstrap CIs
# ---------------------------------------------------------------------------
def test_bootstrap(df):
    section("6. Bootstrap 95% CIs for Tier RSR Estimates (n_boot=10,000)")

    for tier_name, tier_col in [("Tier1", "tier1"), ("Medium", "medium"), ("Tier2", "tier2")]:
        subsection(f"{tier_name} RSR")

        vals = df[tier_col].values
        est, lo, hi = bootstrap_ci(vals, seed=SEED)
        pr(f"  Overall: {est:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(vals)})")

        pr(f"\n  Per model:")
        for model in sorted(df["model"].unique()):
            sub = df[df["model"] == model][tier_col].values
            est, lo, hi = bootstrap_ci(sub, seed=SEED)
            pr(f"    {model:<18} {est:.4f}  [{lo:.4f}, {hi:.4f}]  (n={len(sub)})")

        pr(f"\n  Per misconception type:")
        for t in sorted(df["misconception_type"].unique()):
            sub = df[df["misconception_type"] == t][tier_col].values
            est, lo, hi = bootstrap_ci(sub, seed=SEED)
            pr(f"    {t:<18} {est:.4f}  [{lo:.4f}, {hi:.4f}]  (n={len(sub)})")


# ---------------------------------------------------------------------------
# 7. Tier Gap Analysis
# ---------------------------------------------------------------------------
def test_tier_gaps(df):
    section("7. Tier Gap Analysis (Paired Differences with Bootstrap 95% CIs)")

    t1 = df["tier1"].values
    med = df["medium"].values
    t2 = df["tier2"].values

    gaps = [
        ("Medium - Tier1", med, t1),
        ("Tier2 - Medium", t2, med),
        ("Tier2 - Tier1", t2, t1),
    ]
    for label, a, b in gaps:
        est, lo, hi = bootstrap_paired_diff_ci(a, b, seed=SEED)
        pr(f"  {label:<20}: {est:>+.4f}  95% CI [{lo:>+.4f}, {hi:>+.4f}]  ({est*100:>+.1f} pp)")

    subsection("Per-model tier gaps (Tier2 - Tier1)")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        a = sub["tier2"].values
        b = sub["tier1"].values
        est, lo, hi = bootstrap_paired_diff_ci(a, b, seed=SEED)
        pr(f"  {model:<18}: {est:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}]  ({est*100:>+.1f} pp)")

    subsection("Per-model tier gaps (Medium - Tier1)")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        a = sub["medium"].values
        b = sub["tier1"].values
        est, lo, hi = bootstrap_paired_diff_ci(a, b, seed=SEED)
        pr(f"  {model:<18}: {est:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}]  ({est*100:>+.1f} pp)")

    subsection("Per-model tier gaps (Tier2 - Medium)")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        a = sub["tier2"].values
        b = sub["medium"].values
        est, lo, hi = bootstrap_paired_diff_ci(a, b, seed=SEED)
        pr(f"  {model:<18}: {est:>+.4f}  [{lo:>+.4f}, {hi:>+.4f}]  ({est*100:>+.1f} pp)")


# ---------------------------------------------------------------------------
# 8. Answer Revealing Analysis
# ---------------------------------------------------------------------------
def test_answer_revealed(df):
    section("8. Answer Revealing Analysis")

    subsection("Answer-Revealed Rate (ARR) per model")
    pr(f"  {'Model':<18} {'ARR':>8} {'AR_n':>6} {'Total':>6}")
    pr(f"  {'-'*18} {'-'*8} {'-'*6} {'-'*6}")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        ar_count = sub["answer_revealed"].sum()
        n = len(sub)
        rate = ar_count / n if n > 0 else 0
        pr(f"  {model:<18} {rate:>8.4f} {ar_count:>6} {n:>6}")
    total_ar = df["answer_revealed"].sum()
    total_n = len(df)
    pr(f"  {'Overall':<18} {total_ar/total_n:>8.4f} {total_ar:>6} {total_n:>6}")

    subsection("Tier2 RSR conditioned on answer_revealed")
    for ar_val, label in [(True, "AR=True"), (False, "AR=False")]:
        sub = df[df["answer_revealed"] == ar_val]
        if len(sub) == 0:
            pr(f"  {label}: no data")
            continue
        rsr = sub["tier2"].mean()
        est, lo, hi = bootstrap_ci(sub["tier2"].values, seed=SEED)
        pr(f"  {label}: RSR={rsr:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  (n={len(sub)})")

    subsection("Tier2 RSR: AR=True vs AR=False (chi-squared test)")
    ct = pd.crosstab(df["answer_revealed"], df["tier2"])
    if ct.shape == (2, 2):
        chi2, p, dof, _ = stats.chi2_contingency(ct, correction=True)
        pr(f"  Chi-squared = {chi2:.4f}, df = {dof}, p = {p:.6g}")
        # Also Fisher exact if cells are small
        if ct.values.min() < 5:
            try:
                oddsratio, p_fisher = stats.fisher_exact(ct)
                pr(f"  Fisher exact: OR = {oddsratio:.4f}, p = {p_fisher:.6g}")
            except Exception:
                pass
    else:
        pr("  Cannot compute (not a 2x2 table)")

    subsection("Conditional Tier2 RSR per model (AR=True vs AR=False)")
    pr(f"  {'Model':<18} {'RSR|AR=T':>10} {'n(AR=T)':>8} {'RSR|AR=F':>10} {'n(AR=F)':>8}")
    pr(f"  {'-'*18} {'-'*10} {'-'*8} {'-'*10} {'-'*8}")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        ar_t = sub[sub["answer_revealed"] == True]
        ar_f = sub[sub["answer_revealed"] == False]
        rsr_t = ar_t["tier2"].mean() if len(ar_t) > 0 else float("nan")
        rsr_f = ar_f["tier2"].mean() if len(ar_f) > 0 else float("nan")
        pr(f"  {model:<18} {rsr_t:>10.4f} {len(ar_t):>8} {rsr_f:>10.4f} {len(ar_f):>8}")

    subsection("Medium and Tier1 RSR conditioned on answer_revealed")
    for tier_name, tier_col in [("Tier1", "tier1"), ("Medium", "medium")]:
        for ar_val, label in [(True, "AR=True"), (False, "AR=False")]:
            sub = df[df["answer_revealed"] == ar_val]
            if len(sub) == 0:
                continue
            rsr = sub[tier_col].mean()
            pr(f"  {tier_name} {label}: RSR={rsr:.4f}  (n={len(sub)})")


# ---------------------------------------------------------------------------
# Descriptive summary
# ---------------------------------------------------------------------------
def descriptive_summary(df):
    section("0. Descriptive Summary")

    pr(f"  Total scenarios: {len(df)}")
    pr(f"  Models: {sorted(df['model'].unique())}")
    pr(f"  Strategies: {sorted(df['strategy'].unique())}")
    pr(f"  Types: {sorted(df['misconception_type'].unique())}")
    pr(f"  Difficulties: {sorted(df['difficulty'].unique())}")

    subsection("Overall tier rates")
    for col, label in [("tier1", "Tier1"), ("medium", "Medium"), ("tier2", "Tier2")]:
        rate = df[col].mean()
        pr(f"  {label}: {rate:.4f} ({df[col].sum()}/{len(df)})")

    subsection("Per-model tier rates")
    pr(f"  {'Model':<18} {'n':>5} {'Tier1':>8} {'Medium':>8} {'Tier2':>8}")
    pr(f"  {'-'*18} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for model in sorted(df["model"].unique()):
        sub = df[df["model"] == model]
        pr(f"  {model:<18} {len(sub):>5} {sub['tier1'].mean():>8.4f} "
           f"{sub['medium'].mean():>8.4f} {sub['tier2'].mean():>8.4f}")

    subsection("Per-type tier rates")
    pr(f"  {'Type':<18} {'n':>5} {'Tier1':>8} {'Medium':>8} {'Tier2':>8}")
    pr(f"  {'-'*18} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for t in sorted(df["misconception_type"].unique()):
        sub = df[df["misconception_type"] == t]
        pr(f"  {t:<18} {len(sub):>5} {sub['tier1'].mean():>8.4f} "
           f"{sub['medium'].mean():>8.4f} {sub['tier2'].mean():>8.4f}")

    subsection("Per-strategy tier rates")
    pr(f"  {'Strategy':<22} {'n':>5} {'Tier1':>8} {'Medium':>8} {'Tier2':>8}")
    pr(f"  {'-'*22} {'-'*5} {'-'*8} {'-'*8} {'-'*8}")
    for s in sorted(df["strategy"].unique()):
        sub = df[df["strategy"] == s]
        pr(f"  {s:<22} {len(sub):>5} {sub['tier1'].mean():>8.4f} "
           f"{sub['medium'].mean():>8.4f} {sub['tier2'].mean():>8.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pr("Exp3v2 Three-Tier Statistical Analysis")
    pr(f"Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}")
    pr(f"Seed: {SEED}, Bootstrap resamples: {N_BOOTSTRAP}")

    df = load_data()

    descriptive_summary(df)
    test_mcnemar(df)
    test_cochran(df)
    test_type_effect(df)
    test_strategy_effect(df)
    test_interaction(df)
    test_bootstrap(df)
    test_tier_gaps(df)
    test_answer_revealed(df)

    section("END OF ANALYSIS")

    # Save to file
    with open(OUTPUT_TXT, "w") as f:
        f.write(output_buf.getvalue())
    pr(f"\nResults saved to: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
