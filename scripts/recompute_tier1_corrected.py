"""
Tier-1 recomputation: using verification_answer as the correct gold, recompute the Tier-1
(regex) results for the 2,250 Exp3v2 dialogues and every Exp4v2 condition.

Background: the earlier run.py passed scenario.correct_answer (the original problem's answer) as gold;
102/106 scenarios used the original-problem answer instead of the verification gold in the legacy Tier1 run.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import json
import glob
import os
import re
from collections import defaultdict


def evaluate_tier1_heuristic(student_response: str, correct_answer: str) -> bool:
    """Word-boundary regex match, the same logic as Tier-1 in evaluator.py."""
    response_lower = student_response.lower().strip()
    answer_lower = correct_answer.lower().strip()
    if not answer_lower:
        return False
    pattern = r'\b' + re.escape(answer_lower) + r'\b'
    return bool(re.search(pattern, response_lower))


def load_scenario_bank():
    """Load the scenario bank: id -> {verification_answer, correct_answer, ...}"""
    with open("data/exp3v2_scenario_bank.json") as f:
        return {s["id"]: s for s in json.load(f)}


def recompute_exp3v2(bank):
    """Recompute all 2,250 Exp3v2 dialogues."""
    print("=" * 60)
    print("EXP3V2 TIER1 RECOMPUTATION")
    print("=" * 60)

    all_results = []
    for d in sorted(glob.glob("results/exp3v2/2026*")):
        if not os.path.isdir(d):
            continue
        model = d.split("_")[-1]
        for cf in sorted(glob.glob(f"{d}/cell_*.json")):
            with open(cf) as f:
                cell = json.load(f)
            for r in cell["results"]:
                sid = r["scenario_id"]
                sc = bank.get(sid, {})
                student = r.get("verification_answer", "")

                # Original gold (bug): correct_answer
                original_gold = str(sc.get("correct_answer", ""))
                # Corrected gold: verification_answer
                correct_gold = str(sc.get("verification_answer", ""))

                original_t1 = evaluate_tier1_heuristic(student, original_gold)
                corrected_t1 = evaluate_tier1_heuristic(student, correct_gold)

                all_results.append({
                    "scenario_id": sid,
                    "model": model,
                    "strategy": r.get("strategy", ""),
                    "misconception_type": r.get("misconception_type", ""),
                    "original_tier1": original_t1,
                    "corrected_tier1": corrected_t1,
                    "tier2_correct": r.get("tier2_correct", r.get("remediation_success")),
                    "gold_match": original_gold.strip().lower() == correct_gold.strip().lower(),
                })

    n = len(all_results)
    print(f"\nTotal dialogues: {n}")
    if n == 0:
        raise SystemExit(
            "No Exp3v2 dialogues found under results/exp3v2/2026*/cell_*.json.\n"
            "This script recomputes Tier-1 from the raw per-dialogue dumps, so the\n"
            "summary-only tier1_corrected/ and tier2_corrected/ directories are not\n"
            "sufficient. Regenerate the raw run directories (see REPRODUCE.md Track B)\n"
            "or place existing ones under results/exp3v2/ before rerunning."
        )

    # Overall before/after
    orig_count = sum(1 for r in all_results if r["original_tier1"])
    corr_count = sum(1 for r in all_results if r["corrected_tier1"])
    t2_count = sum(1 for r in all_results if r["tier2_correct"])
    print(f"\n--- OVERALL ---")
    print(f"  Original Tier1 (wrong gold):  {orig_count}/{n} = {orig_count/n*100:.1f}%")
    print(f"  Corrected Tier1 (right gold): {corr_count}/{n} = {corr_count/n*100:.1f}%")
    print(f"  Tier2 (stored, unchanged):    {t2_count}/{n} = {t2_count/n*100:.1f}%")
    print(f"  Medium (reference):           ~45.8%")

    # MATCH group invariance check
    match_results = [r for r in all_results if r["gold_match"]]
    mismatch_results = [r for r in all_results if not r["gold_match"]]
    print(f"\n--- MATCH GROUP (correct_answer == verification_answer) ---")
    print(f"  n = {len(match_results)}")
    if match_results:
        m_orig = sum(1 for r in match_results if r["original_tier1"])
        m_corr = sum(1 for r in match_results if r["corrected_tier1"])
        print(f"  Original Tier1: {m_orig}/{len(match_results)} = {m_orig/len(match_results)*100:.1f}%")
        print(f"  Corrected Tier1: {m_corr}/{len(match_results)} = {m_corr/len(match_results)*100:.1f}%")
        print(f"  INVARIANCE CHECK: {'PASS ✓' if m_orig == m_corr else 'FAIL ✗ — values differ!'}")

    # Per-model
    print(f"\n--- PER MODEL ---")
    models = sorted(set(r["model"] for r in all_results))
    print(f"{'Model':<18} {'Orig T1':>8} {'Corr T1':>8} {'Tier2':>8} {'n':>6}")
    for m in models:
        mr = [r for r in all_results if r["model"] == m]
        mn = len(mr)
        mo = sum(1 for r in mr if r["original_tier1"])
        mc = sum(1 for r in mr if r["corrected_tier1"])
        mt = sum(1 for r in mr if r["tier2_correct"])
        print(f"  {m:<16} {mo/mn*100:>7.1f}% {mc/mn*100:>7.1f}% {mt/mn*100:>7.1f}% {mn:>6}")

    # Per-type
    print(f"\n--- PER MISCONCEPTION TYPE ---")
    types = sorted(set(r["misconception_type"] for r in all_results))
    print(f"{'Type':<14} {'Orig T1':>8} {'Corr T1':>8} {'Tier2':>8} {'n':>6}")
    for t in types:
        tr = [r for r in all_results if r["misconception_type"] == t]
        tn = len(tr)
        to = sum(1 for r in tr if r["original_tier1"])
        tc = sum(1 for r in tr if r["corrected_tier1"])
        tt = sum(1 for r in tr if r["tier2_correct"])
        print(f"  {t:<12} {to/tn*100:>7.1f}% {tc/tn*100:>7.1f}% {tt/tn*100:>7.1f}% {tn:>6}")

    # Per-strategy
    print(f"\n--- PER STRATEGY ---")
    strategies = sorted(set(r["strategy"] for r in all_results))
    print(f"{'Strategy':<22} {'Orig T1':>8} {'Corr T1':>8} {'Tier2':>8} {'n':>6}")
    for s in strategies:
        sr = [r for r in all_results if r["strategy"] == s]
        sn = len(sr)
        so = sum(1 for r in sr if r["original_tier1"])
        sc_ = sum(1 for r in sr if r["corrected_tier1"])
        st = sum(1 for r in sr if r["tier2_correct"])
        print(f"  {s:<20} {so/sn*100:>7.1f}% {sc_/sn*100:>7.1f}% {st/sn*100:>7.1f}% {sn:>6}")

    # Save results
    out_dir = "results/exp3v2/tier1_corrected"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tier1_recomputed.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} results to {out_path}")

    return all_results


def recompute_exp4v2(bank):
    """Recompute every Exp4v2 condition."""
    print("\n" + "=" * 60)
    print("EXP4V2 TIER1 RECOMPUTATION")
    print("=" * 60)

    all_results = []
    for d in sorted(glob.glob("results/exp4v2/2026*")):
        if not os.path.isdir(d):
            continue
        model = d.split("_")[-1]
        for cf in sorted(glob.glob(f"{d}/condition_*.json")):
            condition = os.path.basename(cf).replace(".json", "").replace("condition_", "")
            with open(cf) as f:
                data = json.load(f)
            results = data if isinstance(data, list) else data.get("results", [])
            for r in results:
                sid = r.get("scenario_id", "")
                sc = bank.get(sid, {})
                student = r.get("verification_answer", "")

                original_gold = str(sc.get("correct_answer", ""))
                correct_gold = str(sc.get("verification_answer", ""))

                original_t1 = evaluate_tier1_heuristic(student, original_gold)
                corrected_t1 = evaluate_tier1_heuristic(student, correct_gold)

                all_results.append({
                    "scenario_id": sid,
                    "model": model,
                    "condition": condition,
                    "original_tier1": original_t1,
                    "corrected_tier1": corrected_t1,
                    "tier2_correct": r.get("tier2_correct", r.get("remediation_success")),
                })

    n = len(all_results)
    print(f"\nTotal results: {n}")

    if n == 0:
        print("No Exp4v2 results found.")
        return all_results

    # Per-condition
    print(f"\n--- PER CONDITION ---")
    conditions = sorted(set(r["condition"] for r in all_results))
    print(f"{'Condition':<12} {'Orig T1':>8} {'Corr T1':>8} {'Tier2':>8} {'n':>6}")
    for c in conditions:
        cr = [r for r in all_results if r["condition"] == c]
        cn = len(cr)
        co = sum(1 for r in cr if r["original_tier1"])
        cc = sum(1 for r in cr if r["corrected_tier1"])
        ct = sum(1 for r in cr if r["tier2_correct"])
        print(f"  {c:<10} {co/cn*100:>7.1f}% {cc/cn*100:>7.1f}% {ct/cn*100:>7.1f}% {cn:>6}")

    # Per-model
    print(f"\n--- PER MODEL ---")
    models = sorted(set(r["model"] for r in all_results))
    print(f"{'Model':<18} {'Orig T1':>8} {'Corr T1':>8} {'n':>6}")
    for m in models:
        mr = [r for r in all_results if r["model"] == m]
        mn = len(mr)
        mo = sum(1 for r in mr if r["original_tier1"])
        mc = sum(1 for r in mr if r["corrected_tier1"])
        print(f"  {m:<16} {mo/mn*100:>7.1f}% {mc/mn*100:>7.1f}% {mn:>6}")

    # Save results
    out_dir = "results/exp4v2/tier1_corrected"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "tier1_recomputed.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(all_results)} results to {out_path}")

    return all_results


if __name__ == "__main__":
    bank = load_scenario_bank()
    print(f"Loaded {len(bank)} scenarios from bank")

    exp3_results = recompute_exp3v2(bank)
    exp4_results = recompute_exp4v2(bank)

    print("\n" + "=" * 60)
    print("DONE — Tier1 recomputation complete")
    print("=" * 60)
