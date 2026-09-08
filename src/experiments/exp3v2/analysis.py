"""Hypothesis-driven analysis module for exp3v2.

Tests 5 hypotheses about why LLMs fail at conversational remediation:
- H1: Tutor explanation ability (gold vs LLM RSR)
- H2: Misconception type effect (factual vs procedural vs conceptual)
- H3: Difficulty effect (easy vs medium vs hard)
- H4: Evaluation/simulation rigidity (gold RSR near 0?)
- H5: Answer reveal as only path (AR-excluded RSR near 0?)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

MISCONCEPTION_TYPES = ["factual", "procedural", "conceptual"]
DIFFICULTY_LEVELS = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_experiment_results(results_dir: Path) -> pd.DataFrame:
    """Load all cell JSONs from an experiment run into a single DataFrame.

    Expects cell_*.json files with 'results' key containing per-scenario dicts.
    """
    rows = []
    for cell_file in sorted(results_dir.glob("cell_*.json")):
        with open(cell_file) as f:
            cell_data = json.load(f)
        for r in cell_data.get("results", []):
            rows.append(r)

    if not rows:
        raise ValueError(f"No results found in {results_dir}")

    return pd.DataFrame(rows)


def load_gold_results(gold_path: Path) -> pd.DataFrame:
    """Load gold remediation results."""
    with open(gold_path) as f:
        data = json.load(f)
    return pd.DataFrame(data.get("results", []))


# ---------------------------------------------------------------------------
# Hypothesis tests
# ---------------------------------------------------------------------------

def test_h1_tutor_ability(
    llm_df: pd.DataFrame,
    gold_df: pd.DataFrame,
) -> dict:
    """H1: Is failure due to tutor explanation quality?

    Compares gold RSR vs LLM RSR using Fisher's exact test.
    If gold RSR >> LLM RSR → LLM tutoring is the bottleneck.
    If gold RSR ≈ LLM RSR → evaluation/student simulation is the bottleneck.
    """
    llm_success = llm_df["remediation_success"].sum()
    llm_fail = len(llm_df) - llm_success
    gold_success = gold_df["remediation_success"].sum()
    gold_fail = len(gold_df) - gold_success

    # Fisher's exact test
    table = [[llm_success, llm_fail], [gold_success, gold_fail]]
    odds_ratio, p_value = stats.fisher_exact(table)

    llm_rsr = llm_success / len(llm_df) if len(llm_df) > 0 else 0
    gold_rsr = gold_success / len(gold_df) if len(gold_df) > 0 else 0
    effect_size = gold_rsr - llm_rsr

    if p_value < 0.05 and gold_rsr > llm_rsr:
        interpretation = (
            f"H1 SUPPORTED: Gold RSR ({gold_rsr:.3f}) significantly higher than "
            f"LLM RSR ({llm_rsr:.3f}), p={p_value:.4f}. "
            f"LLM tutoring quality is a contributing factor."
        )
    elif gold_rsr <= llm_rsr + 0.05:
        interpretation = (
            f"H1 REFUTED: Gold RSR ({gold_rsr:.3f}) not meaningfully higher than "
            f"LLM RSR ({llm_rsr:.3f}). Evaluation/simulation is the bottleneck, "
            f"not tutor quality."
        )
    else:
        interpretation = (
            f"H1 INCONCLUSIVE: Gold RSR ({gold_rsr:.3f}) vs LLM RSR ({llm_rsr:.3f}), "
            f"p={p_value:.4f}. Difference exists but not statistically significant."
        )

    return {
        "hypothesis": "H1: Tutor explanation ability",
        "test": "Fisher's exact test",
        "llm_rsr": llm_rsr,
        "gold_rsr": gold_rsr,
        "odds_ratio": odds_ratio,
        "p_value": p_value,
        "effect_size": effect_size,
        "n_llm": len(llm_df),
        "n_gold": len(gold_df),
        "interpretation": interpretation,
    }


def test_h2_type_effect(df: pd.DataFrame) -> dict:
    """H2: Is failure type-dependent?

    Chi-squared test on RSR across factual/procedural/conceptual.
    If significant → failure depends on misconception type.
    """
    type_groups = {}
    for mtype in MISCONCEPTION_TYPES:
        group = df[df["misconception_type"] == mtype]
        if len(group) > 0:
            type_groups[mtype] = {
                "n": len(group),
                "success": int(group["remediation_success"].sum()),
                "fail": int(len(group) - group["remediation_success"].sum()),
                "rsr": float(group["remediation_success"].mean()),
            }

    # Build contingency table
    observed = np.array([
        [type_groups[t]["success"], type_groups[t]["fail"]]
        for t in MISCONCEPTION_TYPES if t in type_groups
    ])

    if observed.shape[0] < 2:
        return {
            "hypothesis": "H2: Misconception type effect",
            "test": "Chi-squared",
            "p_value": None,
            "interpretation": "Insufficient data for test (need >= 2 types)",
            "type_rsrs": type_groups,
        }

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Cramér's V as effect size
    n_total = observed.sum()
    k = min(observed.shape)
    cramers_v = np.sqrt(chi2 / (n_total * (k - 1))) if n_total > 0 and k > 1 else 0

    if p_value < 0.05:
        interpretation = (
            f"H2 SUPPORTED: RSR differs significantly by misconception type "
            f"(χ²={chi2:.2f}, p={p_value:.4f}, V={cramers_v:.3f}). "
            f"Failure is type-dependent."
        )
    else:
        interpretation = (
            f"H2 REFUTED: No significant RSR difference by type "
            f"(χ²={chi2:.2f}, p={p_value:.4f}). Failure is type-independent."
        )

    return {
        "hypothesis": "H2: Misconception type effect",
        "test": "Chi-squared",
        "test_statistic": float(chi2),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "effect_size": float(cramers_v),
        "type_rsrs": type_groups,
        "interpretation": interpretation,
    }


def test_h3_difficulty_effect(df: pd.DataFrame) -> dict:
    """H3: Is failure difficulty-dependent?

    Chi-squared test + Cochran-Armitage trend test on RSR by difficulty.
    If easy RSR > hard RSR → difficulty-dependent limitation.
    If easy RSR ≈ 0 → fundamental limitation regardless of difficulty.
    """
    diff_groups = {}
    for diff in DIFFICULTY_LEVELS:
        group = df[df["difficulty"] == diff]
        if len(group) > 0:
            diff_groups[diff] = {
                "n": len(group),
                "success": int(group["remediation_success"].sum()),
                "fail": int(len(group) - group["remediation_success"].sum()),
                "rsr": float(group["remediation_success"].mean()),
            }

    # Chi-squared test
    observed = np.array([
        [diff_groups[d]["success"], diff_groups[d]["fail"]]
        for d in DIFFICULTY_LEVELS if d in diff_groups
    ])

    if observed.shape[0] < 2:
        return {
            "hypothesis": "H3: Difficulty effect",
            "test": "Chi-squared",
            "p_value": None,
            "interpretation": "Insufficient data",
            "difficulty_rsrs": diff_groups,
        }

    chi2, p_value, dof, expected = stats.chi2_contingency(observed)

    # Ordinal trend: Spearman correlation between difficulty rank and success
    difficulty_rank = {"easy": 1, "medium": 2, "hard": 3}
    ranks = [difficulty_rank.get(d, 2) for d in df["difficulty"]]
    successes = df["remediation_success"].astype(int).tolist()
    if len(set(ranks)) > 1 and len(set(successes)) > 1:
        rho, rho_p = stats.spearmanr(ranks, successes)
    else:
        rho, rho_p = 0.0, 1.0

    easy_rsr = diff_groups.get("easy", {}).get("rsr", 0)
    hard_rsr = diff_groups.get("hard", {}).get("rsr", 0)

    if easy_rsr < 0.05:
        interpretation = (
            f"H3 OBSERVATION: Even easy problems have RSR≈{easy_rsr:.3f}, "
            f"suggesting fundamental limitation regardless of difficulty."
        )
    elif p_value < 0.05:
        interpretation = (
            f"H3 SUPPORTED: RSR differs by difficulty (χ²={chi2:.2f}, p={p_value:.4f}). "
            f"Easy={easy_rsr:.3f}, Hard={hard_rsr:.3f}. "
            f"Trend: ρ={rho:.3f} (p={rho_p:.4f})."
        )
    else:
        interpretation = (
            f"H3 REFUTED: No significant difficulty effect "
            f"(χ²={chi2:.2f}, p={p_value:.4f})."
        )

    return {
        "hypothesis": "H3: Difficulty effect",
        "test": "Chi-squared + Spearman trend",
        "test_statistic": float(chi2),
        "chi2": float(chi2),
        "p_value": float(p_value),
        "dof": int(dof),
        "spearman_rho": float(rho),
        "spearman_p": float(rho_p),
        "effect_size": float(rho),
        "difficulty_rsrs": diff_groups,
        "interpretation": interpretation,
    }


def test_h4_evaluation_rigidity(gold_df: pd.DataFrame) -> dict:
    """H4: Is the simulated student/evaluation too rigid?

    If gold RSR ≈ 0% → the evaluation framework itself is the bottleneck,
    not the LLM's tutoring ability. Even perfect tutoring can't "succeed"
    under the current evaluation.
    """
    gold_rsr = gold_df["remediation_success"].mean() if len(gold_df) > 0 else 0
    n = len(gold_df)

    # Bootstrap CI for gold RSR
    if n > 0:
        rng = np.random.RandomState(42)
        boot_rsrs = []
        for _ in range(10000):
            idx = rng.choice(n, n, replace=True)
            boot_rsrs.append(gold_df.iloc[idx]["remediation_success"].mean())
        ci_lower = float(np.percentile(boot_rsrs, 2.5))
        ci_upper = float(np.percentile(boot_rsrs, 97.5))
    else:
        ci_lower, ci_upper = 0.0, 0.0

    if gold_rsr < 0.10:
        interpretation = (
            f"H4 SUPPORTED: Gold RSR is only {gold_rsr:.3f} "
            f"[{ci_lower:.3f}, {ci_upper:.3f}]. Even expert-authored remediation "
            f"rarely succeeds, indicating the evaluation framework "
            f"(simulated student + verification) is too rigid."
        )
    elif gold_rsr > 0.50:
        interpretation = (
            f"H4 REFUTED: Gold RSR is {gold_rsr:.3f} "
            f"[{ci_lower:.3f}, {ci_upper:.3f}]. Expert remediation succeeds "
            f"substantially, so the evaluation is sensitive enough. "
            f"LLM tutoring quality is the limiting factor."
        )
    else:
        interpretation = (
            f"H4 PARTIAL: Gold RSR is {gold_rsr:.3f} "
            f"[{ci_lower:.3f}, {ci_upper:.3f}]. Evaluation shows some sensitivity "
            f"but may still underestimate remediation success."
        )

    return {
        "hypothesis": "H4: Evaluation/simulation rigidity",
        "test": "Bootstrap CI on gold RSR",
        "gold_rsr": float(gold_rsr),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_gold": n,
        "effect_size": float(gold_rsr),
        "p_value": None,  # Not applicable — descriptive
        "interpretation": interpretation,
    }


def test_h5_answer_reveal(df: pd.DataFrame) -> dict:
    """H5: Is answer revealing the only path to 'success'?

    Compares RSR vs AR-excluded RSR.
    If AR-excluded RSR ≈ 0 across all conditions → H5 confirmed.
    """
    total_rsr = df["remediation_success"].mean() if len(df) > 0 else 0

    # AR-excluded: success AND NOT answer_revealed
    if "ar_excluded_success" in df.columns:
        ar_excl_rsr = df["ar_excluded_success"].mean() if len(df) > 0 else 0
    elif "answer_revealed" in df.columns:
        genuine = df["remediation_success"] & ~df["answer_revealed"]
        ar_excl_rsr = genuine.mean() if len(df) > 0 else 0
    else:
        ar_excl_rsr = total_rsr  # Can't compute without AR data

    arr = df["answer_revealed"].mean() if "answer_revealed" in df.columns and len(df) > 0 else 0

    if ar_excl_rsr < 0.02:
        interpretation = (
            f"H5 SUPPORTED: AR-excluded RSR is {ar_excl_rsr:.3f} vs total RSR "
            f"{total_rsr:.3f}. Virtually all 'successful' remediations involved "
            f"answer revelation (AR rate={arr:.3f}). Genuine remediation ≈ 0%."
        )
    elif ar_excl_rsr > 0.10:
        interpretation = (
            f"H5 REFUTED: AR-excluded RSR is {ar_excl_rsr:.3f}, showing "
            f"meaningful genuine remediation without answer reveals."
        )
    else:
        interpretation = (
            f"H5 PARTIAL: AR-excluded RSR is {ar_excl_rsr:.3f} vs total RSR "
            f"{total_rsr:.3f}. Some genuine remediation exists but AR "
            f"remains a major factor (AR rate={arr:.3f})."
        )

    return {
        "hypothesis": "H5: Answer reveal as only path",
        "test": "RSR vs AR-excluded RSR comparison",
        "total_rsr": float(total_rsr),
        "ar_excluded_rsr": float(ar_excl_rsr),
        "answer_reveal_rate": float(arr),
        "effect_size": float(total_rsr - ar_excl_rsr),
        "p_value": None,  # Descriptive comparison
        "n": len(df),
        "interpretation": interpretation,
    }


# ---------------------------------------------------------------------------
# Power analysis
# ---------------------------------------------------------------------------

def power_analysis(
    baseline_rsr: float = 0.05,
    target_rsr: float = 0.20,
    n_scenarios: int = 30,
    n_profiles: int = 6,
    alpha: float = 0.05,
) -> dict:
    """Power analysis for RSR comparisons.

    Reports both scenario-level (n=n_scenarios, conservative) and
    dialogue-level (n=n_scenarios*n_profiles, optimistic) power.

    Uses statsmodels for proportion comparison power.
    """
    try:
        from statsmodels.stats.power import NormalIndPower
        from statsmodels.stats.proportion import proportion_effectsize
    except ImportError:
        logger.warning("statsmodels not installed; skipping power analysis")
        return {
            "error": "statsmodels not installed",
            "scenario_power": None,
            "dialogue_power": None,
        }

    power_solver = NormalIndPower()
    effect_size = proportion_effectsize(baseline_rsr, target_rsr)

    # Scenario-level (conservative: each scenario is one independent unit)
    scenario_power = power_solver.solve_power(
        effect_size=effect_size,
        nobs1=n_scenarios,
        alpha=alpha,
        alternative="two-sided",
    )

    # Dialogue-level (optimistic: treating each dialogue as independent)
    n_dialogues = n_scenarios * n_profiles
    dialogue_power = power_solver.solve_power(
        effect_size=effect_size,
        nobs1=n_dialogues,
        alpha=alpha,
        alternative="two-sided",
    )

    # Minimum detectable effect at 0.80 power
    mde_h = power_solver.solve_power(
        nobs1=n_scenarios,
        alpha=alpha,
        power=0.80,
        alternative="two-sided",
    )

    return {
        "baseline_rsr": baseline_rsr,
        "target_rsr": target_rsr,
        "effect_size_h": float(effect_size),
        "alpha": alpha,
        "n_scenarios": n_scenarios,
        "n_dialogues": n_dialogues,
        "scenario_power": float(scenario_power),
        "dialogue_power": float(dialogue_power),
        "mde_cohens_h": float(mde_h),
        "note": (
            f"With n={n_scenarios} scenarios, power={scenario_power:.3f} for "
            f"detecting {baseline_rsr:.0%}→{target_rsr:.0%} difference. "
            f"MDE at 0.80 power requires Cohen's h={mde_h:.3f}."
        ),
    }


# ---------------------------------------------------------------------------
# Tables and figures
# ---------------------------------------------------------------------------

def generate_summary_tables(
    df: pd.DataFrame,
    gold_df: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Generate paper-ready summary tables.

    Returns dict of DataFrames, each exportable to LaTeX via .to_latex().
    """
    tables = {}

    # Table 1: RSR by model x type (if model column exists)
    if "model" in df.columns:
        table1 = df.pivot_table(
            values="remediation_success",
            index="model",
            columns="misconception_type",
            aggfunc="mean",
        ).reindex(columns=MISCONCEPTION_TYPES)
        tables["rsr_by_model_type"] = table1

    # Table 2: RSR by strategy x type
    if "strategy" in df.columns:
        table2 = df.pivot_table(
            values="remediation_success",
            index="strategy",
            columns="misconception_type",
            aggfunc="mean",
        ).reindex(columns=MISCONCEPTION_TYPES)
        tables["rsr_by_strategy_type"] = table2

    # Table 3: RSR by difficulty
    if "difficulty" in df.columns:
        table3 = df.pivot_table(
            values="remediation_success",
            index="strategy" if "strategy" in df.columns else df.index,
            columns="difficulty",
            aggfunc="mean",
        ).reindex(columns=DIFFICULTY_LEVELS)
        tables["rsr_by_difficulty"] = table3

    # Table 4: RSR vs AR-excluded RSR
    rsr_col = df["remediation_success"].mean()
    if "ar_excluded_success" in df.columns:
        ar_excl = df["ar_excluded_success"].mean()
    else:
        ar_excl = (df["remediation_success"] & ~df["answer_revealed"]).mean()
    arr = df["answer_revealed"].mean() if "answer_revealed" in df.columns else 0

    table4 = pd.DataFrame({
        "RSR": [rsr_col],
        "AR-excluded RSR": [ar_excl],
        "AR Rate": [arr],
    }, index=["Overall"])
    tables["rsr_vs_ar_excluded"] = table4

    # Table 5: Gold vs LLM comparison
    if gold_df is not None and len(gold_df) > 0:
        gold_rsr = gold_df["remediation_success"].mean()
        table5 = pd.DataFrame({
            "RSR": [rsr_col, gold_rsr],
            "N": [len(df), len(gold_df)],
        }, index=["LLM Tutor", "Gold (Expert)"])
        tables["gold_vs_llm"] = table5

    return tables


def generate_figures(
    df: pd.DataFrame,
    gold_df: pd.DataFrame | None = None,
    output_dir: Path | str = "paper/figures",
) -> list[str]:
    """Generate paper-ready figures as PDFs.

    Returns list of generated file paths.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # Fig 1: RSR heatmap (strategy x type)
    try:
        pivot = df.pivot_table(
            values="remediation_success",
            index="strategy",
            columns="misconception_type",
            aggfunc="mean",
        ).reindex(columns=MISCONCEPTION_TYPES)

        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.5)
        ax.set_xticks(range(len(MISCONCEPTION_TYPES)))
        ax.set_xticklabels(MISCONCEPTION_TYPES, rotation=45, ha="right")
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        # Add text annotations
        for i in range(len(pivot.index)):
            for j in range(len(MISCONCEPTION_TYPES)):
                val = pivot.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9)
        plt.colorbar(im, label="RSR")
        ax.set_title("Remediation Success Rate by Strategy × Type")
        plt.tight_layout()
        path = output_dir / "fig_rsr_heatmap.pdf"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        generated.append(str(path))
    except Exception as e:
        logger.warning(f"Failed to generate heatmap: {e}")

    # Fig 2: Difficulty gradient bar chart
    if "difficulty" in df.columns:
        try:
            diff_rsrs = []
            for d in DIFFICULTY_LEVELS:
                group = df[df["difficulty"] == d]
                diff_rsrs.append(group["remediation_success"].mean() if len(group) > 0 else 0)

            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ["#2ecc71", "#f39c12", "#e74c3c"]
            bars = ax.bar(DIFFICULTY_LEVELS, diff_rsrs, color=colors, edgecolor="black")
            for bar, val in zip(bars, diff_rsrs):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            ax.set_ylabel("RSR")
            ax.set_title("RSR by Difficulty Level")
            ax.set_ylim(0, max(diff_rsrs) * 1.3 + 0.01)
            plt.tight_layout()
            path = output_dir / "fig_difficulty_gradient.pdf"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(path))
        except Exception as e:
            logger.warning(f"Failed to generate difficulty chart: {e}")

    # Fig 3: Gold vs LLM comparison
    if gold_df is not None and len(gold_df) > 0:
        try:
            llm_rsr = df["remediation_success"].mean()
            gold_rsr = gold_df["remediation_success"].mean()

            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.bar(["LLM Tutor", "Gold (Expert)"], [llm_rsr, gold_rsr],
                         color=["#3498db", "#f1c40f"], edgecolor="black")
            for bar, val in zip(bars, [llm_rsr, gold_rsr]):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=10)
            ax.set_ylabel("RSR")
            ax.set_title("Gold vs LLM Remediation Success")
            ax.set_ylim(0, max(llm_rsr, gold_rsr) * 1.3 + 0.01)
            plt.tight_layout()
            path = output_dir / "fig_gold_vs_llm.pdf"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            generated.append(str(path))
        except Exception as e:
            logger.warning(f"Failed to generate gold comparison: {e}")

    return generated


# ---------------------------------------------------------------------------
# Full analysis orchestrator
# ---------------------------------------------------------------------------

def run_full_analysis(
    results_dir: Path,
    gold_path: Path | None = None,
    output_dir: Path | None = None,
) -> dict:
    """Run all hypothesis tests and generate tables/figures.

    Args:
        results_dir: Directory with cell_*.json files from experiment run.
        gold_path: Path to gold_results.json (optional).
        output_dir: Directory to save analysis outputs.

    Returns:
        dict with all hypothesis test results, power analysis, and file paths.
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    df = load_experiment_results(results_dir)
    logger.info(f"Loaded {len(df)} dialogue results from {results_dir}")

    gold_df = None
    if gold_path and Path(gold_path).exists():
        gold_df = load_gold_results(gold_path)
        logger.info(f"Loaded {len(gold_df)} gold results")

    # Run hypothesis tests
    results = {}

    results["h2_type_effect"] = test_h2_type_effect(df)
    logger.info(f"H2: {results['h2_type_effect']['interpretation'][:80]}...")

    results["h3_difficulty_effect"] = test_h3_difficulty_effect(df)
    logger.info(f"H3: {results['h3_difficulty_effect']['interpretation'][:80]}...")

    results["h5_answer_reveal"] = test_h5_answer_reveal(df)
    logger.info(f"H5: {results['h5_answer_reveal']['interpretation'][:80]}...")

    if gold_df is not None and len(gold_df) > 0:
        results["h1_tutor_ability"] = test_h1_tutor_ability(df, gold_df)
        logger.info(f"H1: {results['h1_tutor_ability']['interpretation'][:80]}...")

        results["h4_evaluation_rigidity"] = test_h4_evaluation_rigidity(gold_df)
        logger.info(f"H4: {results['h4_evaluation_rigidity']['interpretation'][:80]}...")

    # Power analysis
    results["power_analysis"] = power_analysis()

    # Tables
    tables = generate_summary_tables(df, gold_df)
    results["tables"] = {name: tbl.to_dict() for name, tbl in tables.items()}

    # LaTeX export
    if output_dir:
        for name, tbl in tables.items():
            latex_path = output_dir / f"table_{name}.tex"
            tbl.to_latex(latex_path, float_format="%.3f")
            logger.info(f"Saved LaTeX table: {latex_path}")

    # Figures
    fig_dir = output_dir if output_dir else Path("paper/figures")
    results["figures"] = generate_figures(df, gold_df, fig_dir)

    # Save report
    if output_dir:
        # Convert non-serializable items
        report = {}
        for k, v in results.items():
            if k == "tables":
                report[k] = v
            elif k == "figures":
                report[k] = v
            else:
                report[k] = v

        report_path = output_dir / "analysis_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str, ensure_ascii=False)
        logger.info(f"Saved analysis report: {report_path}")

    return results
