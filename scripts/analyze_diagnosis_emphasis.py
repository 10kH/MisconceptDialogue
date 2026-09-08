"""Analyze the diagnosis-emphasis robustness variant.

Recomputes the wrong-vs-gold RSR delta UNDER emphasis, with McNemar (paired)
and a bootstrap 95% CI on the paired difference — mirroring the original
control's results/exp4v2/condition_C_wrong/summary.json format.

Primary contrast (robustness claim):
    condition_C_wrong_emph  vs  condition_C2_emph
    (both under the emphasis prompt; judge always against original misconception)

Context contrasts (did emphasis move each arm vs the non-emphasis originals?):
    wrong_emph vs original condition_C_wrong  (checkpoint.jsonl)
    gold_emph  vs original condition_C2        (exp4v2 per-model dirs)

Usage:
  python scripts/analyze_diagnosis_emphasis.py
  python scripts/analyze_diagnosis_emphasis.py --out results/exp4v2/diagnosis_emphasis_summary.json
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RESULTS_BASE = "results/exp4v2"
GOLD_EMPH_DIR = "results/exp4v2/condition_C2_emph"
WRONG_EMPH_DIR = "results/exp4v2/condition_C_wrong_emph"
ORIG_WRONG_DIR = "results/exp4v2/condition_C_wrong"
MODEL_NAMES = ["qwen-2.5-7b", "mistral-7b", "llama-3.1-8b", "phi-4", "qwen3.5-9b"]


def load_arm_checkpoint(output_dir):
    """Return {(model, scenario_id): tier2_bool} for valid rows."""
    cp = f"{output_dir}/checkpoint.jsonl"
    out = {}
    if not os.path.exists(cp):
        return out
    with open(cp) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            out[(r["model"], r["scenario_id"])] = bool(r.get("tier2_correct"))
    return out


def load_orig_wrong():
    return load_arm_checkpoint(ORIG_WRONG_DIR)


def find_model_dir(model_name):
    for d in sorted(glob.glob(f"{RESULTS_BASE}/2026*")):
        if d.endswith(f"_{model_name}"):
            return d
    return None


TIER2_CORRECTED = "results/exp4v2/tier2_corrected/tier2_recomputed.json"


def load_orig_c2():
    """Original (non-emphasis) gold condition C2, CORRECTED tier2.

    NOTE (Q1 fix): the raw results/exp4v2/*/condition_C2.json tier2_correct
    field holds the OLD buggy verification-gold score (~0.52). The corrected
    C2 baseline (0.766) that the original wrong-diagnosis control compared
    against lives in tier2_corrected/tier2_recomputed.json, keyed by
    (model, scenario_id, condition). We read that here so the context
    contrast uses the same corrected judge as everything else.
    """
    out = {}
    if not os.path.exists(TIER2_CORRECTED):
        return out
    rows = json.load(open(TIER2_CORRECTED))
    for r in rows:
        if r.get("condition") == "C2":
            out[(r["model"], r["scenario_id"])] = bool(r.get("correct"))
    return out


def mcnemar(a, b):
    """Paired McNemar on binary arrays a (test) vs b (baseline)."""
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    n10 = int(np.sum((a == 1) & (b == 0)))  # test succeeds, baseline fails
    n01 = int(np.sum((a == 0) & (b == 1)))  # test fails, baseline succeeds
    discordant = n10 + n01
    if discordant == 0:
        return {"n10": n10, "n01": n01, "p": 1.0, "method": "no_discordant"}
    if discordant < 25:
        p = stats.binomtest(n10, discordant, 0.5).pvalue
        method = "exact_binomial"
    else:
        chi2 = (abs(n10 - n01) - 1) ** 2 / discordant
        p = float(stats.chi2.sf(chi2, df=1))
        method = "chi2_continuity_corrected"
    return {"n10": n10, "n01": n01, "p": float(p), "method": method}


def bootstrap_diff(a, b, n_boot=10000, seed=42):
    """Bootstrap 95% CI on paired diff (mean(a) - mean(b)) in percentage points."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append((a[idx].mean() - b[idx].mean()) * 100)
    return [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]


def paired(test_map, base_map, label):
    """Align two {(model,sid): bool} maps on shared keys; return paired stats."""
    keys = sorted(set(test_map) & set(base_map))
    a = [1 if test_map[k] else 0 for k in keys]
    b = [1 if base_map[k] else 0 for k in keys]
    n = len(keys)
    if n == 0:
        return {"label": label, "n": 0, "note": "no shared keys"}
    mc = mcnemar(a, b)
    return {
        "label": label,
        "n": n,
        "test_rsr": sum(a) / n,
        "baseline_rsr": sum(b) / n,
        "diff_pp": (sum(a) - sum(b)) / n * 100,
        "mcnemar_p": mc["p"],
        "mcnemar_method": mc["method"],
        "discordant_test_success_base_fail": mc["n10"],
        "discordant_test_fail_base_success": mc["n01"],
        "bootstrap95_diff_pp": bootstrap_diff(a, b),
    }


def per_model(mapping):
    out = {}
    for m in MODEL_NAMES:
        vals = [v for (mm, _), v in mapping.items() if mm == m]
        if vals:
            out[m] = {"n": len(vals), "tier2_success": sum(vals),
                      "tier2": sum(vals) / len(vals)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/exp4v2/diagnosis_emphasis_summary.json")
    args = ap.parse_args()

    gold_emph = load_arm_checkpoint(GOLD_EMPH_DIR)
    wrong_emph = load_arm_checkpoint(WRONG_EMPH_DIR)
    orig_wrong = load_orig_wrong()
    orig_c2 = load_orig_c2()

    def rsr(mp):
        return (sum(mp.values()) / len(mp)) if mp else None

    summary = {
        "n_gold_emph": len(gold_emph),
        "n_wrong_emph": len(wrong_emph),
        "rsr_gold_emph": rsr(gold_emph),
        "rsr_wrong_emph": rsr(wrong_emph),
        "primary_contrast_wrong_vs_gold_under_emphasis":
            paired(wrong_emph, gold_emph, "wrong_emph vs gold_emph"),
        "context_wrong_emph_vs_orig_wrong":
            paired(wrong_emph, orig_wrong, "wrong_emph vs orig condition_C_wrong")
            if orig_wrong else {"note": "orig condition_C_wrong not found"},
        "context_gold_emph_vs_orig_c2":
            paired(gold_emph, orig_c2, "gold_emph vs orig condition_C2")
            if orig_c2 else {"note": "orig condition_C2 not found"},
        "per_model_gold_emph": per_model(gold_emph),
        "per_model_wrong_emph": per_model(wrong_emph),
        "reference_original_control": {
            "C_wrong": 0.8075, "C2_gold": 0.7660, "B1_direct": 0.7849,
            "wrong_vs_C2_diff_pp": 4.15, "wrong_vs_C2_mcnemar_p": 0.0777,
        },
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSaved: {args.out}")

    pc = summary["primary_contrast_wrong_vs_gold_under_emphasis"]
    if pc.get("n"):
        print(f"\n=== PRIMARY: wrong_emph vs gold_emph (n={pc['n']}) ===")
        print(f"  wrong={pc['test_rsr']*100:.1f}%  gold={pc['baseline_rsr']*100:.1f}%  "
              f"diff={pc['diff_pp']:+.2f}pp  McNemar p={pc['mcnemar_p']:.4f}")
        print(f"  bootstrap95 diff: [{pc['bootstrap95_diff_pp'][0]:.2f}, "
              f"{pc['bootstrap95_diff_pp'][1]:.2f}] pp")
        print("  Original control (no emphasis): wrong-vs-gold = +4.15pp, p=0.078")


if __name__ == "__main__":
    main()
