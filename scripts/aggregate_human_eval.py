"""Aggregate human evaluation results from 3 or 4 annotators.

Usage:
    python scripts/aggregate_human_eval.py \
        --annotators file1.json file2.json file3.json [file4.json] \
        --samples paper/human_eval_v2/annotation_samples.json \
        --output paper/human_eval_v2/results/aggregated.json

Each annotator JSON is the export from annotate_standalone.html:
    { "HE_001": {"q1": "Yes", "q2": 4, "q3": "No", "q4": 3}, ... }
  or wrapped format:
    { "annotator_id": "annotator_A", "annotations": {"HE_001": {"q1_understanding": ...}} }
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np


# ── Cohen's kappa ──

def cohens_kappa(labels1, labels2):
    """Compute Cohen's kappa for two lists of labels."""
    assert len(labels1) == len(labels2)
    n = len(labels1)
    cats = sorted(set(labels1) | set(labels2))
    agree = sum(a == b for a, b in zip(labels1, labels2))
    p_o = agree / n
    p_e = sum(
        (sum(a == c for a in labels1) / n) * (sum(b == c for b in labels2) / n)
        for c in cats
    )
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1 - p_e)


def fleiss_kappa(ratings_matrix):
    """Compute Fleiss' kappa for multiple raters.

    ratings_matrix: list of dicts, each {category: count} for one item.
    """
    n_items = len(ratings_matrix)
    cats = sorted(set().union(*[r.keys() for r in ratings_matrix]))
    n_raters = sum(ratings_matrix[0].values())
    n_cats = len(cats)

    # P_i for each item
    P = []
    for row in ratings_matrix:
        counts = [row.get(c, 0) for c in cats]
        P_i = (sum(c * (c - 1) for c in counts)) / (n_raters * (n_raters - 1))
        P.append(P_i)
    P_bar = np.mean(P)

    # P_j for each category
    p_j = []
    for c in cats:
        total = sum(row.get(c, 0) for row in ratings_matrix)
        p_j.append(total / (n_items * n_raters))
    P_e = sum(p ** 2 for p in p_j)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ── Consensus logic ──

def consensus_q1(votes):
    """Determine consensus for Q1 (Yes/Partial/No) from 3-4 annotators.

    Rules for 3 raters:
    - Majority (2+/3) wins
    - 1-1-1 three-way split → Partial (middle ground)

    Rules for 4 raters:
    - Majority (3+/4) wins
    - 2-2 tie: Yes vs No → Partial; Yes vs Partial → Partial;
      Partial vs No → No (conservative)
    - 2-1-1 split: go with the mode
    """
    c = Counter(votes)
    n = len(votes)
    majority = (n // 2) + 1  # 2 for 3 raters, 3 for 4 raters

    # Check for majority
    for label in ["Yes", "Partial", "No"]:
        if c.get(label, 0) >= majority:
            return label

    if n == 4:
        # 2-2 ties
        top2 = c.most_common(2)
        if len(top2) == 2 and top2[0][1] == 2 and top2[1][1] == 2:
            pair = {top2[0][0], top2[1][0]}
            if pair == {"Yes", "No"}:
                return "Partial"
            if pair == {"Yes", "Partial"}:
                return "Partial"
            if pair == {"Partial", "No"}:
                return "No"
        # 2-1-1 split: go with the mode
        return c.most_common(1)[0][0]

    # 3 raters: 1-1-1 three-way split → Partial
    return "Partial"


def consensus_binary(votes):
    """Majority vote for Yes/No. Tie (2-2) → Yes (conservative: assume revealed)."""
    c = Counter(votes)
    if c.get("Yes", 0) > c.get("No", 0):
        return "Yes"
    if c.get("No", 0) > c.get("Yes", 0):
        return "No"
    return "Yes"  # Tie → conservative


def consensus_score(scores):
    """Median for 1-5 Likert scales."""
    return float(np.median(scores))


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Aggregate human eval results (3-4 annotators)")
    parser.add_argument("--annotators", nargs="+", required=True,
                        help="Annotator JSON files (3 or 4)")
    # Legacy support for --ann1/--ann2/--ann3/--ann4
    parser.add_argument("--ann1", help="(legacy) annotator 1 JSON")
    parser.add_argument("--ann2", help="(legacy) annotator 2 JSON")
    parser.add_argument("--ann3", help="(legacy) annotator 3 JSON")
    parser.add_argument("--ann4", help="(legacy) annotator 4 JSON")
    parser.add_argument("--samples", default="paper/human_eval_v2/annotation_samples.json")
    parser.add_argument("--output", default="paper/human_eval_v2/results/aggregated.json")
    args = parser.parse_args()

    # Resolve annotator files: prefer --annotators, fallback to --ann1..4
    if args.annotators:
        ann_files = args.annotators
    else:
        ann_files = [f for f in [args.ann1, args.ann2, args.ann3, args.ann4] if f]

    if len(ann_files) < 3:
        print("ERROR: Need at least 3 annotator files")
        sys.exit(1)
    if len(ann_files) > 4:
        print("ERROR: Maximum 4 annotator files supported")
        sys.exit(1)

    n_annotators = len(ann_files)
    print(f"Running with {n_annotators} annotators")

    annotators = {}
    annotator_names = []
    for i, path in enumerate(ann_files):
        with open(path) as f:
            raw = json.load(f)
        # Handle both export formats:
        # Format A (wrapped): {"annotations": {"HE_001": {"q1_understanding": ...}}}
        # Format B (flat):    {"HE_001": {"q1": ...}}
        if "annotations" in raw:
            ann_data = raw["annotations"]
            name = raw.get("annotator_id", f"ann{i+1}")
            n_complete = raw.get("n_complete", "?")
            print(f"Loaded {name}: {n_complete}/{len(ann_data)} samples")
        else:
            ann_data = raw
            name = Path(path).stem.split("_")[0] if "_" in Path(path).stem else f"ann{i+1}"
            print(f"Loaded {name}: {len(ann_data)} samples")
        # Normalize field names: q1_understanding -> q1, etc.
        normalized = {}
        for sid, entry in ann_data.items():
            normalized[sid] = {
                "q1": entry.get("q1_understanding") or entry.get("q1"),
                "q2": entry.get("q2_reasoning_quality") or entry.get("q2"),
                "q3": entry.get("q3_answer_revealed") or entry.get("q3"),
                "q4": entry.get("q4_naturalness") or entry.get("q4"),
                "notes": entry.get("notes", ""),
            }
        annotators[name] = normalized
        annotator_names.append(name)

    with open(args.samples) as f:
        samples_data = json.load(f)
    samples = {s["sample_id"]: s for s in samples_data["samples"]}
    sample_ids = sorted(samples.keys())
    print(f"Total samples: {len(sample_ids)}")

    # Check completeness
    for name, ann in annotators.items():
        missing = [sid for sid in sample_ids if sid not in ann]
        if missing:
            print(f"WARNING: {name} missing {len(missing)} samples: {missing[:5]}...")

    # ── 1. Inter-annotator agreement ──
    print("\n" + "=" * 60)
    print("INTER-ANNOTATOR AGREEMENT")
    print("=" * 60)

    # Q1: Fleiss' kappa (4 raters, 3 categories)
    q1_matrix = []
    for sid in sample_ids:
        votes = {}
        for name in annotator_names:
            v = annotators[name].get(sid, {}).get("q1")
            if v:
                votes[v] = votes.get(v, 0) + 1
        q1_matrix.append(votes)
    fk_q1 = fleiss_kappa(q1_matrix)
    print(f"Q1 (understanding) Fleiss' kappa: {fk_q1:.3f}")

    # Q1 pairwise Cohen's kappa
    print("\nQ1 pairwise Cohen's kappa:")
    q1_kappas = []
    for (n1, n2) in combinations(annotator_names, 2):
        l1 = [annotators[n1].get(sid, {}).get("q1", "?") for sid in sample_ids]
        l2 = [annotators[n2].get(sid, {}).get("q1", "?") for sid in sample_ids]
        k = cohens_kappa(l1, l2)
        q1_kappas.append(k)
        print(f"  {n1} vs {n2}: {k:.3f}")
    print(f"  Mean pairwise: {np.mean(q1_kappas):.3f}")

    # Q3: Fleiss' kappa (4 raters, 2 categories)
    q3_matrix = []
    for sid in sample_ids:
        votes = {}
        for name in annotator_names:
            v = annotators[name].get(sid, {}).get("q3")
            if v:
                votes[v] = votes.get(v, 0) + 1
        q3_matrix.append(votes)
    fk_q3 = fleiss_kappa(q3_matrix)
    print(f"\nQ3 (answer revealed) Fleiss' kappa: {fk_q3:.3f}")

    # Q2, Q4: ICC or mean pairwise correlation
    for q_name, q_key in [("Q2 (reasoning)", "q2"), ("Q4 (naturalness)", "q4")]:
        scores = []
        for name in annotator_names:
            s = [annotators[name].get(sid, {}).get(q_key) for sid in sample_ids]
            scores.append(s)
        # Pairwise Spearman
        from scipy.stats import spearmanr
        corrs = []
        for (i, j) in combinations(range(n_annotators), 2):
            valid = [(scores[i][k], scores[j][k]) for k in range(len(sample_ids))
                     if scores[i][k] is not None and scores[j][k] is not None]
            if valid:
                s1, s2 = zip(*valid)
                r, _ = spearmanr(s1, s2)
                corrs.append(r)
        print(f"\n{q_name} mean pairwise Spearman: {np.mean(corrs):.3f}")

    # ── 2. Consensus labels ──
    print("\n" + "=" * 60)
    print("CONSENSUS RESULTS")
    print("=" * 60)

    consensus = {}
    for sid in sample_ids:
        q1_votes = [annotators[n].get(sid, {}).get("q1") for n in annotator_names]
        q1_votes = [v for v in q1_votes if v]
        q3_votes = [annotators[n].get(sid, {}).get("q3") for n in annotator_names]
        q3_votes = [v for v in q3_votes if v]
        q2_scores = [annotators[n].get(sid, {}).get("q2") for n in annotator_names]
        q2_scores = [v for v in q2_scores if v is not None]
        q4_scores = [annotators[n].get(sid, {}).get("q4") for n in annotator_names]
        q4_scores = [v for v in q4_scores if v is not None]

        consensus[sid] = {
            "q1": consensus_q1(q1_votes) if q1_votes else None,
            "q2": consensus_score(q2_scores) if q2_scores else None,
            "q3": consensus_binary(q3_votes) if q3_votes else None,
            "q4": consensus_score(q4_scores) if q4_scores else None,
        }

    # Q1 distribution
    q1_dist = Counter(c["q1"] for c in consensus.values())
    print(f"Q1 distribution: {dict(q1_dist)}")

    # Human RSR (Yes → correct, Partial/No → incorrect)
    n_correct = sum(1 for c in consensus.values() if c["q1"] == "Yes")
    n_total = len(consensus)
    human_rsr = n_correct / n_total
    print(f"Human RSR (Yes only): {n_correct}/{n_total} = {human_rsr:.1%}")

    # Q2 mean
    q2_vals = [c["q2"] for c in consensus.values() if c["q2"] is not None]
    print(f"Q2 mean reasoning quality: {np.mean(q2_vals):.2f}/5")

    # Q3 answer revealed
    n_revealed = sum(1 for c in consensus.values() if c["q3"] == "Yes")
    print(f"Q3 answer revealed: {n_revealed}/{n_total} = {n_revealed/n_total:.1%}")

    # Q4 naturalness
    q4_vals = [c["q4"] for c in consensus.values() if c["q4"] is not None]
    print(f"Q4 mean naturalness: {np.mean(q4_vals):.2f}/5")

    # ── 3. Comparison with automated tiers ──
    print("\n" + "=" * 60)
    print("HUMAN vs AUTOMATED EVALUATION")
    print("=" * 60)

    human_labels = []  # 1 = correct, 0 = incorrect
    tier1_labels = []
    tier2_labels = []
    gpt54_labels = []

    for sid in sample_ids:
        s = samples[sid]
        h = 1 if consensus[sid]["q1"] == "Yes" else 0
        t1 = 1 if s.get("tier1_correct") else 0
        t2 = 1 if s.get("tier2_correct") else 0
        g54 = 1 if s.get("gpt54_correct") else 0
        human_labels.append(h)
        tier1_labels.append(t1)
        tier2_labels.append(t2)
        gpt54_labels.append(g54)

    tier1_rsr = sum(tier1_labels) / len(tier1_labels)
    tier2_rsr = sum(tier2_labels) / len(tier2_labels)
    gpt54_rsr = sum(gpt54_labels) / len(gpt54_labels)

    print(f"Human RSR:    {human_rsr:.1%}")
    print(f"Tier 1 RSR:   {tier1_rsr:.1%}")
    print(f"GPT-5.4 RSR:  {gpt54_rsr:.1%}")
    print(f"Tier 2 RSR:   {tier2_rsr:.1%}")

    # Kappa vs human
    k_tier1 = cohens_kappa(human_labels, tier1_labels)
    k_tier2 = cohens_kappa(human_labels, tier2_labels)
    k_gpt54 = cohens_kappa(human_labels, gpt54_labels)
    agree_tier1 = sum(a == b for a, b in zip(human_labels, tier1_labels)) / n_total
    agree_tier2 = sum(a == b for a, b in zip(human_labels, tier2_labels)) / n_total
    agree_gpt54 = sum(a == b for a, b in zip(human_labels, gpt54_labels)) / n_total

    print(f"\nTier 1 vs Human:  kappa={k_tier1:.3f}, agreement={agree_tier1:.1%}")
    print(f"GPT-5.4 vs Human: kappa={k_gpt54:.3f}, agreement={agree_gpt54:.1%}")
    print(f"Tier 2 vs Human:  kappa={k_tier2:.3f}, agreement={agree_tier2:.1%}")

    # ── 4. By misconception type ──
    print("\n" + "=" * 60)
    print("BY MISCONCEPTION TYPE")
    print("=" * 60)

    for mtype in ["factual", "procedural", "conceptual"]:
        type_sids = [sid for sid in sample_ids if samples[sid]["misconception_type"] == mtype]
        n_t = len(type_sids)
        h_rsr = sum(1 for sid in type_sids if consensus[sid]["q1"] == "Yes") / n_t if n_t else 0
        t2_rsr_t = sum(1 for sid in type_sids if samples[sid].get("tier2_correct")) / n_t if n_t else 0
        print(f"{mtype:12s} (n={n_t}): human={h_rsr:.1%}, tier2={t2_rsr_t:.1%}")

    # ── 5. By model ──
    print("\n" + "=" * 60)
    print("BY MODEL")
    print("=" * 60)

    for model in sorted(set(samples[sid]["model"] for sid in sample_ids)):
        model_sids = [sid for sid in sample_ids if samples[sid]["model"] == model]
        n_m = len(model_sids)
        h_rsr_m = sum(1 for sid in model_sids if consensus[sid]["q1"] == "Yes") / n_m if n_m else 0
        print(f"{model:20s} (n={n_m}): human RSR={h_rsr_m:.1%}")

    # ── 6. Disagreement analysis ──
    print("\n" + "=" * 60)
    print("DISAGREEMENT ANALYSIS (Human vs Tier 2)")
    print("=" * 60)

    # Cases where human and tier2 disagree
    h_yes_t2_no = []
    h_no_t2_yes = []
    for sid in sample_ids:
        h = consensus[sid]["q1"] == "Yes"
        t2 = bool(samples[sid].get("tier2_correct"))
        if h and not t2:
            h_yes_t2_no.append(sid)
        elif not h and t2:
            h_no_t2_yes.append(sid)

    print(f"Human=Yes, Tier2=No (human more lenient): {len(h_yes_t2_no)}")
    print(f"Human=No,  Tier2=Yes (Tier2 more lenient): {len(h_no_t2_yes)}")

    # Type breakdown of Tier2-lenient cases
    if h_no_t2_yes:
        type_dist = Counter(samples[sid]["misconception_type"] for sid in h_no_t2_yes)
        print(f"  Tier2-lenient by type: {dict(type_dist)}")

    # ── 7. Save output ──
    output = {
        "agreement": {
            "q1_fleiss_kappa": round(fk_q1, 3),
            "q1_mean_pairwise_kappa": round(np.mean(q1_kappas), 3),
            "q3_fleiss_kappa": round(fk_q3, 3),
        },
        "consensus": consensus,
        "summary": {
            "human_rsr": round(human_rsr, 3),
            "tier1_rsr": round(tier1_rsr, 3),
            "tier2_rsr": round(tier2_rsr, 3),
            "kappa_tier1_vs_human": round(k_tier1, 3),
            "kappa_gpt54_vs_human": round(k_gpt54, 3),
            "kappa_tier2_vs_human": round(k_tier2, 3),
            "agree_tier1": round(agree_tier1, 3),
            "agree_gpt54": round(agree_gpt54, 3),
            "agree_tier2": round(agree_tier2, 3),
            "gpt54_rsr": round(gpt54_rsr, 3),
            "q1_distribution": dict(q1_dist),
            "q2_mean": round(np.mean(q2_vals), 2),
            "q3_revealed_rate": round(n_revealed / n_total, 3),
            "q4_mean": round(np.mean(q4_vals), 2),
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {args.output}")

    # ── 8. Paper-ready values ──
    print("\n" + "=" * 60)
    print("PAPER VALUES")
    print("=" * 60)
    print(f"Inter-annotator: Fleiss' kappa={fk_q1:.2f}")
    print(f"Table 9:")
    print(f"  Human (consensus): {human_rsr:.0%}")
    print(f"  GPT-5.4 judge:     RSR={gpt54_rsr:.0%}, kappa={k_gpt54:.2f}, agree={agree_gpt54:.0%}")
    print(f"  Tier 2 (GPT-5-mini): RSR={tier2_rsr:.0%}, kappa={k_tier2:.2f}, agree={agree_tier2:.0%}")
    print(f"  Tier 1 (heuristic):  RSR={tier1_rsr:.0%}, kappa={k_tier1:.2f}, agree={agree_tier1:.0%}")
    print(f"Naturalness: {np.mean(q4_vals):.1f}/5")
    print(f"Answer revealed (human): {n_revealed/n_total:.0%}")


if __name__ == "__main__":
    main()
