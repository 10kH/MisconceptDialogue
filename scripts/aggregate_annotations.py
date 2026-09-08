#!/usr/bin/env python3
"""Aggregate human evaluation annotations and compute agreement metrics.

Usage:
  python scripts/aggregate_annotations.py \
      --annotations paper/human_eval/filled/ann1.tsv paper/human_eval/filled/ann2.tsv \
      --key paper/human_eval/answer_key.tsv \
      --out paper/human_eval/results/

Outputs:
  - agreement_report.txt   Inter-annotator agreement (Cohen's κ, Fleiss' κ)
  - human_vs_auto.tsv      Comparison of human RSR vs automated evaluator RSR
  - summary_stats.tsv      Aggregated statistics by category/strategy/tutor
"""
import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np


# ── I/O ──────────────────────────────────────────────────────────────────

def load_annotations(path: str) -> list[dict]:
    """Load a filled annotation TSV."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            # Normalize fields
            r["Q1_remediation_success"] = r.get("Q1_remediation_success", "").strip()
            r["Q2_answer_revealed"] = r.get("Q2_answer_revealed", "").strip()
            q3 = r.get("Q3_tutoring_quality", "").strip()
            r["Q3_tutoring_quality"] = int(q3) if q3.isdigit() else None
            rows.append(r)
    return rows


def load_answer_key(path: str) -> dict[tuple[str, str], dict]:
    """Load answer key, indexed by (sample_id, dialogue)."""
    key = {}
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for r in reader:
            key[(r["sample_id"], r["dialogue"])] = r
    return key


# ── Agreement metrics ────────────────────────────────────────────────────

def cohens_kappa(labels1: list, labels2: list) -> float:
    """Compute Cohen's kappa for two annotators."""
    assert len(labels1) == len(labels2)
    n = len(labels1)
    if n == 0:
        return float("nan")

    categories = sorted(set(labels1) | set(labels2))
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)

    # Confusion matrix
    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(labels1, labels2):
        matrix[cat_idx[a], cat_idx[b]] += 1

    po = np.trace(matrix) / n  # observed agreement
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = np.sum(row_sums * col_sums)  # expected agreement

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def fleiss_kappa(ratings_matrix: np.ndarray) -> float:
    """Compute Fleiss' kappa for multiple annotators.

    ratings_matrix: (n_items, n_categories) — counts of annotators per category.
    """
    n_items, n_cats = ratings_matrix.shape
    n_raters = int(ratings_matrix[0].sum())

    if n_raters <= 1:
        return float("nan")

    # Proportion per category
    p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)

    # Per-item agreement
    P_i = (np.sum(ratings_matrix ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    P_bar = np.mean(P_i)
    P_e = np.sum(p_j ** 2)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) interpretation."""
    if k < 0:
        return "poor"
    elif k < 0.21:
        return "slight"
    elif k < 0.41:
        return "fair"
    elif k < 0.61:
        return "moderate"
    elif k < 0.81:
        return "substantial"
    else:
        return "almost perfect"


# ── Analysis ─────────────────────────────────────────────────────────────

def compute_pairwise_agreement(all_annotations: dict[str, list[dict]], question: str):
    """Compute pairwise Cohen's kappa for a question across annotators."""
    annotators = sorted(all_annotations.keys())
    results = []

    for a1, a2 in combinations(annotators, 2):
        rows1 = {(r["sample_id"], r["dialogue"]): r[question] for r in all_annotations[a1]}
        rows2 = {(r["sample_id"], r["dialogue"]): r[question] for r in all_annotations[a2]}
        common = sorted(set(rows1.keys()) & set(rows2.keys()))

        labels1 = [rows1[k] for k in common if rows1[k] and rows2[k]]
        labels2 = [rows2[k] for k in common if rows1[k] and rows2[k]]

        if len(labels1) < 5:
            continue

        k = cohens_kappa(labels1, labels2)
        po = sum(1 for a, b in zip(labels1, labels2) if a == b) / len(labels1)
        results.append({
            "annotator_1": a1,
            "annotator_2": a2,
            "n": len(labels1),
            "observed_agreement": po,
            "cohens_kappa": k,
            "interpretation": interpret_kappa(k),
        })

    return results


def compute_fleiss(all_annotations: dict[str, list[dict]], question: str, categories: list[str]):
    """Compute Fleiss' kappa across all annotators for a question."""
    annotators = sorted(all_annotations.keys())

    # Index all annotations by (sample_id, dialogue)
    by_key = defaultdict(list)
    for ann_id in annotators:
        for r in all_annotations[ann_id]:
            val = r[question]
            if val in categories:
                by_key[(r["sample_id"], r["dialogue"])].append(val)

    # Only keep items rated by all annotators
    n_ann = len(annotators)
    items = [k for k, v in by_key.items() if len(v) == n_ann]

    if not items:
        return float("nan"), 0

    cat_idx = {c: i for i, c in enumerate(categories)}
    matrix = np.zeros((len(items), len(categories)), dtype=int)
    for i, k in enumerate(items):
        for val in by_key[k]:
            matrix[i, cat_idx[val]] += 1

    return fleiss_kappa(matrix), len(items)


def majority_vote(values: list[str]) -> str:
    """Return majority label (ties broken alphabetically)."""
    if not values:
        return ""
    c = Counter(values)
    max_count = max(c.values())
    winners = sorted(k for k, v in c.items() if v == max_count)
    return winners[0]


def compare_human_vs_auto(all_annotations: dict[str, list[dict]], answer_key: dict):
    """Compare human majority RSR vs automated evaluator RSR."""
    annotators = sorted(all_annotations.keys())

    # Collect per-item human Q1 votes
    q1_votes = defaultdict(list)
    for ann_id in annotators:
        for r in all_annotations[ann_id]:
            val = r["Q1_remediation_success"]
            if val in ("Yes", "No", "Unclear"):
                q1_votes[(r["sample_id"], r["dialogue"])].append(val)

    results = []
    for (sid, dial), votes in sorted(q1_votes.items()):
        human_rsr = majority_vote(votes)
        key_entry = answer_key.get((sid, dial), {})
        auto_rsr = key_entry.get("auto_rsr", "")
        evaluator = key_entry.get("evaluator", "")
        category = key_entry.get("category", "")
        tutor = key_entry.get("tutor_model", "")
        strategy = key_entry.get("strategy", "")
        mtype = key_entry.get("misconception_type", "")

        results.append({
            "sample_id": sid,
            "dialogue": dial,
            "evaluator": evaluator,
            "human_rsr": human_rsr,
            "auto_rsr": auto_rsr,
            "agree": "Yes" if (human_rsr == auto_rsr) else "No",
            "category": category,
            "tutor_model": tutor,
            "strategy": strategy,
            "misconception_type": mtype,
            "n_votes": len(votes),
            "vote_distribution": dict(Counter(votes)),
        })

    return results


# ── Output ───────────────────────────────────────────────────────────────

def write_agreement_report(pairwise_q1, pairwise_q2, pairwise_q3,
                           fleiss_q1, fleiss_q2, fleiss_q3,
                           n_annotators: int, n_items: int,
                           out_path: Path):
    """Write agreement report."""
    lines = []
    lines.append("=" * 70)
    lines.append("INTER-ANNOTATOR AGREEMENT REPORT")
    lines.append("=" * 70)
    lines.append(f"\nAnnotators: {n_annotators}")
    lines.append(f"Items: {n_items} (100 samples × 2 dialogues)")
    lines.append("")

    # Fleiss' kappa
    lines.append("-" * 70)
    lines.append("Fleiss' Kappa (all annotators)")
    lines.append("-" * 70)
    for name, (kappa, n) in [
        ("Q1 (RSR)", fleiss_q1),
        ("Q2 (ARR)", fleiss_q2),
        ("Q3 (Quality, binned 1-2/3/4-5)", fleiss_q3),
    ]:
        if np.isnan(kappa):
            lines.append(f"  {name}: N/A (insufficient data)")
        else:
            lines.append(f"  {name}: κ = {kappa:.3f} ({interpret_kappa(kappa)}, n={n})")
    lines.append("")

    # Pairwise Cohen's kappa
    for qname, pairs in [("Q1 (RSR)", pairwise_q1),
                          ("Q2 (ARR)", pairwise_q2),
                          ("Q3 (Quality)", pairwise_q3)]:
        lines.append("-" * 70)
        lines.append(f"Pairwise Cohen's Kappa — {qname}")
        lines.append("-" * 70)
        if not pairs:
            lines.append("  No valid pairs (need ≥5 common items)")
        for p in pairs:
            lines.append(
                f"  {p['annotator_1']} vs {p['annotator_2']}: "
                f"κ = {p['cohens_kappa']:.3f} ({p['interpretation']}), "
                f"Po = {p['observed_agreement']:.3f}, n = {p['n']}"
            )
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


def write_human_vs_auto(comparisons: list[dict], out_path: Path):
    """Write human vs automated evaluator comparison TSV."""
    fieldnames = [
        "sample_id", "dialogue", "evaluator", "human_rsr", "auto_rsr",
        "agree", "category", "tutor_model", "strategy", "misconception_type",
        "n_votes", "vote_distribution",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in comparisons:
            row_copy = dict(row)
            row_copy["vote_distribution"] = json.dumps(row["vote_distribution"])
            writer.writerow(row_copy)
    print(f"Wrote: {out_path}")


def write_summary_stats(comparisons: list[dict], out_path: Path):
    """Write summary statistics by various dimensions."""
    lines = []
    lines.append("=" * 70)
    lines.append("HUMAN vs AUTOMATED EVALUATOR SUMMARY")
    lines.append("=" * 70)

    # Overall agreement
    n_total = len(comparisons)
    n_agree = sum(1 for c in comparisons if c["agree"] == "Yes")
    lines.append(f"\nOverall agreement: {n_agree}/{n_total} ({100*n_agree/n_total:.1f}%)")

    # By evaluator
    lines.append("\n--- By Evaluator ---")
    for ev in ["same_scale", "gpt5_mini"]:
        subset = [c for c in comparisons if c["evaluator"] == ev]
        if not subset:
            continue
        agree = sum(1 for c in subset if c["agree"] == "Yes")
        lines.append(f"  {ev}: {agree}/{len(subset)} ({100*agree/len(subset):.1f}%)")

        # Breakdown: auto=Yes/human=Yes, auto=Yes/human=No, etc.
        for auto_val in ["Yes", "No"]:
            for human_val in ["Yes", "No", "Unclear"]:
                n = sum(1 for c in subset
                        if c["auto_rsr"] == auto_val and c["human_rsr"] == human_val)
                if n > 0:
                    lines.append(f"    auto={auto_val}, human={human_val}: {n}")

    # By category
    lines.append("\n--- By Category ---")
    for cat in sorted(set(c["category"] for c in comparisons)):
        subset = [c for c in comparisons if c["category"] == cat]
        agree = sum(1 for c in subset if c["agree"] == "Yes")
        lines.append(f"  {cat}: {agree}/{len(subset)} ({100*agree/len(subset):.1f}%)")

    # By tutor
    lines.append("\n--- By Tutor ---")
    for tutor in sorted(set(c["tutor_model"] for c in comparisons)):
        subset = [c for c in comparisons if c["tutor_model"] == tutor]
        agree = sum(1 for c in subset if c["agree"] == "Yes")
        human_yes = sum(1 for c in subset if c["human_rsr"] == "Yes")
        auto_yes = sum(1 for c in subset if c["auto_rsr"] == "Yes")
        lines.append(
            f"  {tutor}: agree={agree}/{len(subset)} ({100*agree/len(subset):.1f}%), "
            f"human_RSR={human_yes}/{len(subset)}, auto_RSR={auto_yes}/{len(subset)}"
        )

    # By strategy
    lines.append("\n--- By Strategy ---")
    for strat in sorted(set(c["strategy"] for c in comparisons)):
        subset = [c for c in comparisons if c["strategy"] == strat]
        agree = sum(1 for c in subset if c["agree"] == "Yes")
        lines.append(f"  {strat}: {agree}/{len(subset)} ({100*agree/len(subset):.1f}%)")

    # By misconception type
    lines.append("\n--- By Misconception Type ---")
    for mt in sorted(set(c["misconception_type"] for c in comparisons)):
        subset = [c for c in comparisons if c["misconception_type"] == mt]
        agree = sum(1 for c in subset if c["agree"] == "Yes")
        lines.append(f"  {mt}: {agree}/{len(subset)} ({100*agree/len(subset):.1f}%)")

    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote: {out_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Aggregate human evaluation annotations")
    parser.add_argument("--annotations", nargs="+", required=True,
                        help="Filled annotation TSV files (one per annotator)")
    parser.add_argument("--key", default="paper/human_eval/answer_key.tsv",
                        help="Answer key TSV")
    parser.add_argument("--out", default="paper/human_eval/results/",
                        help="Output directory")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    answer_key = load_answer_key(args.key)
    all_annotations = {}
    for path in args.annotations:
        rows = load_annotations(path)
        # Extract annotator ID from first non-empty row
        ann_id = None
        for r in rows:
            if r.get("annotator_id", "").strip():
                ann_id = r["annotator_id"].strip()
                break
        if ann_id is None:
            ann_id = Path(path).stem  # fallback to filename
        all_annotations[ann_id] = rows
        print(f"Loaded {len(rows)} rows from {path} (annotator: {ann_id})")

    n_annotators = len(all_annotations)
    n_items = 200  # 100 samples × 2 dialogues
    print(f"\nAnnotators: {n_annotators}, Expected items: {n_items}")

    # ── 1. Inter-annotator agreement ──────────────────────────────────
    print("\n=== Computing agreement ===")

    # Q1: RSR (Yes/No/Unclear)
    pairwise_q1 = compute_pairwise_agreement(all_annotations, "Q1_remediation_success")
    fleiss_q1 = compute_fleiss(all_annotations, "Q1_remediation_success",
                               ["Yes", "No", "Unclear"])

    # Q2: ARR (Yes/No)
    pairwise_q2 = compute_pairwise_agreement(all_annotations, "Q2_answer_revealed")
    fleiss_q2 = compute_fleiss(all_annotations, "Q2_answer_revealed", ["Yes", "No"])

    # Q3: Quality (bin into low/mid/high for kappa)
    # Map 1-2 → "low", 3 → "mid", 4-5 → "high"
    for ann_id in all_annotations:
        for r in all_annotations[ann_id]:
            q3 = r["Q3_tutoring_quality"]
            if q3 is not None:
                if q3 <= 2:
                    r["Q3_binned"] = "low"
                elif q3 == 3:
                    r["Q3_binned"] = "mid"
                else:
                    r["Q3_binned"] = "high"
            else:
                r["Q3_binned"] = ""

    pairwise_q3 = compute_pairwise_agreement(all_annotations, "Q3_binned")
    fleiss_q3 = compute_fleiss(all_annotations, "Q3_binned", ["low", "mid", "high"])

    write_agreement_report(
        pairwise_q1, pairwise_q2, pairwise_q3,
        fleiss_q1, fleiss_q2, fleiss_q3,
        n_annotators, n_items,
        out_dir / "agreement_report.txt",
    )

    # ── 2. Human vs automated comparison ──────────────────────────────
    print("\n=== Comparing human vs automated ===")
    comparisons = compare_human_vs_auto(all_annotations, answer_key)
    write_human_vs_auto(comparisons, out_dir / "human_vs_auto.tsv")
    write_summary_stats(comparisons, out_dir / "summary_stats.txt")

    # ── 3. Quick console summary ──────────────────────────────────────
    print(f"\n{'='*70}")
    print("QUICK SUMMARY")
    print(f"{'='*70}")

    if pairwise_q1:
        avg_k = np.mean([p["cohens_kappa"] for p in pairwise_q1])
        print(f"Q1 (RSR) avg Cohen's κ: {avg_k:.3f} ({interpret_kappa(avg_k)})")
    if pairwise_q2:
        avg_k = np.mean([p["cohens_kappa"] for p in pairwise_q2])
        print(f"Q2 (ARR) avg Cohen's κ: {avg_k:.3f} ({interpret_kappa(avg_k)})")
    if pairwise_q3:
        avg_k = np.mean([p["cohens_kappa"] for p in pairwise_q3])
        print(f"Q3 (Quality) avg Cohen's κ: {avg_k:.3f} ({interpret_kappa(avg_k)})")

    n_total = len(comparisons)
    n_agree = sum(1 for c in comparisons if c["agree"] == "Yes")
    print(f"\nHuman-Auto agreement: {n_agree}/{n_total} ({100*n_agree/n_total:.1f}%)")

    for ev in ["same_scale", "gpt5_mini"]:
        subset = [c for c in comparisons if c["evaluator"] == ev]
        if subset:
            agree = sum(1 for c in subset if c["agree"] == "Yes")
            print(f"  {ev}: {agree}/{len(subset)} ({100*agree/len(subset):.1f}%)")


if __name__ == "__main__":
    main()
