"""
Generate every number reported in the paper from the corrected scoring.
Reads the Tier-1 corrected and Tier-2 corrected results and prints the before/after
comparison together with all numbers needed for the paper.

Usage: python scripts/generate_paper_numbers.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)
import json, os
from collections import defaultdict


def load_data(exp, tier):
    path = f"results/{exp}/tier{tier}_corrected/{'tier1_recomputed' if tier==1 else 'tier2_recomputed'}.json"
    if not os.path.exists(path):
        # Try checkpoint for tier2
        cp = f"results/{exp}/tier2_corrected/checkpoint.json"
        if tier == 2 and os.path.exists(cp):
            with open(cp) as f:
                d = json.load(f)
            return list(d.values()) if isinstance(d, dict) else d
        return []
    with open(path) as f:
        return json.load(f)


def main():
    print("=" * 70)
    print("PAPER NUMBER GENERATOR — Corrected Values")
    print("=" * 70)

    # === EXP3V2 ===
    t1 = load_data("exp3v2", 1)
    t2 = load_data("exp3v2", 2)

    if not t1 or not t2:
        print(f"WARNING: Exp3v2 data incomplete (Tier1={len(t1)}, Tier2={len(t2)})")

    # Build Tier2 lookup
    t2_map = {}
    for r in t2:
        k = f"{r.get('model','')}_{r.get('scenario_id','')}_{r.get('strategy','')}"
        t2_map[k] = r.get("correct", False)

    # Per-model Tier2 corrected
    model_t2 = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in t2:
        m = r.get("model", "")
        model_t2[m]["n"] += 1
        if r.get("correct"):
            model_t2[m]["correct"] += 1

    # Per-model Tier1 corrected
    model_t1 = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in t1:
        m = r.get("model", "")
        model_t1[m]["n"] += 1
        if r.get("corrected_tier1"):
            model_t1[m]["correct"] += 1

    print("\n### TABLE 3 (tab:exp3) — Per-model corrected values ###")
    print(f"{'Model':<18} {'Tier1':>7} {'Medium':>7} {'Tier2':>7}")
    mediums = {"qwen-2.5-7b": 46.4, "phi-4": 47.1, "qwen3.5-9b": 46.4,
               "llama-3.1-8b": 44.4, "mistral-7b": 44.4}

    all_t1_corr = 0
    all_t2_corr = 0
    n_t1 = 0
    n_t2 = 0
    for m in ["qwen-2.5-7b", "phi-4", "qwen3.5-9b", "llama-3.1-8b", "mistral-7b"]:
        t1_n = model_t1[m]["n"]
        t1_c = model_t1[m]["correct"]
        t2_n = model_t2[m]["n"]
        t2_c = model_t2[m]["correct"]
        t1_pct = t1_c / t1_n * 100 if t1_n else 0
        t2_pct = t2_c / t2_n * 100 if t2_n else 0
        med = mediums.get(m, 0)
        status = "✓" if t2_n >= 450 else f"({t2_n}/450)"
        print(f"  {m:<16} {t1_pct:>6.1f}% {med:>6.1f}% {t2_pct:>6.1f}% {status}")
        all_t1_corr += t1_c
        all_t2_corr += t2_c
        n_t1 += t1_n
        n_t2 += t2_n

    mean_t1 = all_t1_corr / n_t1 * 100 if n_t1 else 0
    mean_t2 = all_t2_corr / n_t2 * 100 if n_t2 else 0
    print(f"  {'MEAN':<16} {mean_t1:>6.1f}% {'45.8':>6}% {mean_t2:>6.1f}%")

    # Per-type
    print("\n### PER TYPE — Corrected Tier2 ###")
    type_t2 = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in t2:
        t = r.get("misconception_type", "")
        type_t2[t]["n"] += 1
        if r.get("correct"):
            type_t2[t]["correct"] += 1

    for t in ["conceptual", "factual", "procedural"]:
        tn = type_t2[t]["n"]
        tc = type_t2[t]["correct"]
        pct = tc / tn * 100 if tn else 0
        print(f"  {t:<14} {pct:.1f}% (n={tn})")

    # Key derived values
    print(f"\n### KEY PAPER VALUES ###")
    print(f"Tier1 mean (corrected): {mean_t1:.1f}%")
    print(f"Medium mean: 45.8%")
    print(f"Tier2 mean (corrected): {mean_t2:.1f}%")
    print(f"Tier1→Tier2 gap: {mean_t2 - mean_t1:.1f}pp")
    print(f"Tier1→Tier2 ratio: {mean_t2/mean_t1:.1f}x" if mean_t1 > 0 else "N/A")
    print(f"Human RSR: 88%")
    print(f"Human-Tier2 gap: {88 - mean_t2:.1f}pp")
    print(f"Persona B→C recovery: {mean_t2 - 6.6:.1f}pp")
    print(f"Tier1 range: {min(model_t1[m]['correct']/model_t1[m]['n']*100 for m in model_t1 if model_t1[m]['n']>0):.1f}--{max(model_t1[m]['correct']/model_t1[m]['n']*100 for m in model_t1 if model_t1[m]['n']>0):.1f}%")
    if n_t2 >= 2250:
        print(f"Tier2 range: {min(model_t2[m]['correct']/model_t2[m]['n']*100 for m in model_t2 if model_t2[m]['n']>0):.1f}--{max(model_t2[m]['correct']/model_t2[m]['n']*100 for m in model_t2 if model_t2[m]['n']>0):.1f}%")

    # === EXP4V2 ===
    print(f"\n### TABLE 4 (tab:exp4) — Per-condition corrected ###")
    t2_exp4 = load_data("exp4v2", 2)
    if t2_exp4:
        cond_stats = defaultdict(lambda: {"n": 0, "correct": 0})
        for r in t2_exp4:
            c = r.get("condition", "")
            cond_stats[c]["n"] += 1
            if r.get("correct"):
                cond_stats[c]["correct"] += 1

        print(f"{'Condition':<12} {'Corr T2':>8} {'n':>6}")
        for c in ["C1", "C2", "C3", "C3_5", "C4", "B1"]:
            cn = cond_stats[c]["n"]
            cc = cond_stats[c]["correct"]
            pct = cc / cn * 100 if cn else 0
            status = "✓" if cn >= 530 else f"({cn}/530)"
            print(f"  {c:<10} {pct:>7.1f}% {cn:>6} {status}")
    else:
        print("  No Exp4v2 data yet")

    print(f"\n{'='*70}")
    print(f"Status: Exp3v2 Tier2 {'COMPLETE' if n_t2 >= 2250 else f'{n_t2}/2250'}")
    print(f"Status: Exp4v2 Tier2 {len(t2_exp4) if t2_exp4 else 0}/3180")


if __name__ == "__main__":
    main()
