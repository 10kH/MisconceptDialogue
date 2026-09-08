"""Regenerate human-eval aggregate files against the CORRECTED per-sample fields.

Context: `scripts/aggregate_human_eval.py` reads `tier1_correct`/`tier2_correct`/
`gpt54_correct` from the samples file. After the verification-gold-mismatch fix, the
samples schema was renamed: the corrected verdicts now live under
`tier1_corrected`/`tier2_corrected`/`gpt54_corrected`, and the old pre-fix values were
preserved for audit under `tier1_legacy_buggy`/`tier2_legacy_buggy`/`gpt54_legacy_buggy`.
Because the old field names no longer exist, `aggregate_human_eval.py` silently reads
`None` for every sample when re-run today — this script fixes that by reading the
`_corrected` fields directly, while keeping every other part of the methodology
(Cohen's/Fleiss' kappa, Q1/Q3 consensus rules, Q2/Q4 median) byte-identical to the
original script.

Conservative-by-design: the HUMAN vs AUTOMATED EVALUATION comparison table (RSR + kappa
vs. tier1/tier2/gpt54) is only computed when `tier1_corrected`, `tier2_corrected`, AND
`gpt54_corrected` are ALL present (non-null) for every sample — matching the design
already used for `human_eval_dirty100/results/aggregated.json`, whose `meta.omitted_section`
explains that a partial (2-of-3) table was deliberately not fabricated because
`gpt54_corrected` is absent from that dataset's schema. When the table is skipped, a
`meta.omitted_section` note is written explaining why.

Usage:
    conda run -n X python scripts/regenerate_human_eval_aggregates.py --dataset human_eval_v2
    conda run -n X python scripts/regenerate_human_eval_aggregates.py --dataset human_eval_dirty100
    conda run -n X python scripts/regenerate_human_eval_aggregates.py --all
"""

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent

# Dataset presets: (samples path, annotator file glob-list, output path, label, sample-id prefix)
PRESETS = {
    "human_eval_v2": dict(
        samples=ROOT / "paper/human_eval_v2/annotation_samples.json",
        annotators=[
            ROOT / "paper/human_eval_v2/results/annotations_v2_annotator_A.json",
            ROOT / "paper/human_eval_v2/results/annotations_v2_annotator_B.json",
            ROOT / "paper/human_eval_v2/results/annotations_v2_annotator_C.json",
        ],
        output=ROOT / "paper/human_eval_v2/results/aggregated.json",
        label="human_eval_v2 (clean+borderline, exp3v2 100-sample set)",
    ),
    "human_eval_dirty100": dict(
        samples=ROOT / "paper/human_eval_dirty100/annotation_samples.json",
        annotators=[
            ROOT / "paper/human_eval_dirty100/results/annotations_dirty100_annotator_A.json",
            ROOT / "paper/human_eval_dirty100/results/annotations_dirty100_annotator_B.json",
            ROOT / "paper/human_eval_dirty100/results/annotations_dirty100_annotator_C.json",
        ],
        output=ROOT / "paper/human_eval_dirty100/results/aggregated.json",
        label="human_eval_dirty100 (borderline-quality, exp3v2 dirty100)",
    ),
}


# ── Cohen's / Fleiss' kappa (identical to aggregate_human_eval.py) ──

def cohens_kappa(labels1, labels2):
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
    n_items = len(ratings_matrix)
    cats = sorted(set().union(*[r.keys() for r in ratings_matrix]))
    n_raters = sum(ratings_matrix[0].values())

    P = []
    for row in ratings_matrix:
        counts = [row.get(c, 0) for c in cats]
        P_i = (sum(c * (c - 1) for c in counts)) / (n_raters * (n_raters - 1))
        P.append(P_i)
    P_bar = np.mean(P)

    p_j = []
    for c in cats:
        total = sum(row.get(c, 0) for row in ratings_matrix)
        p_j.append(total / (n_items * n_raters))
    P_e = sum(p ** 2 for p in p_j)

    if P_e == 1.0:
        return 1.0
    return (P_bar - P_e) / (1 - P_e)


# ── Consensus logic (identical to aggregate_human_eval.py) ──

def consensus_q1(votes):
    c = Counter(votes)
    n = len(votes)
    majority = (n // 2) + 1

    for label in ["Yes", "Partial", "No"]:
        if c.get(label, 0) >= majority:
            return label

    if n == 4:
        top2 = c.most_common(2)
        if len(top2) == 2 and top2[0][1] == 2 and top2[1][1] == 2:
            pair = {top2[0][0], top2[1][0]}
            if pair == {"Yes", "No"}:
                return "Partial"
            if pair == {"Yes", "Partial"}:
                return "Partial"
            if pair == {"Partial", "No"}:
                return "No"
        return c.most_common(1)[0][0]

    return "Partial"


def consensus_binary(votes):
    c = Counter(votes)
    if c.get("Yes", 0) > c.get("No", 0):
        return "Yes"
    if c.get("No", 0) > c.get("Yes", 0):
        return "No"
    return "Yes"


def consensus_score(scores):
    return float(np.median(scores))


# ── Core aggregation for one dataset ──

def load_annotators(ann_files):
    annotators = {}
    annotator_names = []
    for i, path in enumerate(ann_files):
        with open(path) as f:
            raw = json.load(f)
        if "annotations" in raw:
            ann_data = raw["annotations"]
            name = raw.get("annotator_id", f"ann{i+1}")
            n_complete = raw.get("n_complete", "?")
            print(f"  Loaded {name}: {n_complete}/{len(ann_data)} samples")
        else:
            ann_data = raw
            name = Path(path).stem.split("_")[0] if "_" in Path(path).stem else f"ann{i+1}"
            print(f"  Loaded {name}: {len(ann_data)} samples")
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
    return annotators, annotator_names


def regenerate(dataset_key, samples_path, ann_files, output_path, label):
    print("\n" + "=" * 70)
    print(f"DATASET: {dataset_key}  ({label})")
    print("=" * 70)

    annotators, annotator_names = load_annotators(ann_files)
    n_annotators = len(annotator_names)

    with open(samples_path) as f:
        samples_data = json.load(f)
    samples = {s["sample_id"]: s for s in samples_data["samples"]}
    sample_ids = sorted(samples.keys())
    n_total = len(sample_ids)
    print(f"  Total samples: {n_total}")

    for name, ann in annotators.items():
        missing = [sid for sid in sample_ids if sid not in ann]
        if missing:
            print(f"  WARNING: {name} missing {len(missing)} samples: {missing[:5]}...")

    # ── Inter-annotator agreement ──
    q1_matrix = []
    for sid in sample_ids:
        votes = {}
        for name in annotator_names:
            v = annotators[name].get(sid, {}).get("q1")
            if v:
                votes[v] = votes.get(v, 0) + 1
        q1_matrix.append(votes)
    fk_q1 = fleiss_kappa(q1_matrix)

    q1_kappas = []
    for (n1, n2) in combinations(annotator_names, 2):
        l1 = [annotators[n1].get(sid, {}).get("q1", "?") for sid in sample_ids]
        l2 = [annotators[n2].get(sid, {}).get("q1", "?") for sid in sample_ids]
        q1_kappas.append(cohens_kappa(l1, l2))

    q3_matrix = []
    for sid in sample_ids:
        votes = {}
        for name in annotator_names:
            v = annotators[name].get(sid, {}).get("q3")
            if v:
                votes[v] = votes.get(v, 0) + 1
        q3_matrix.append(votes)
    fk_q3 = fleiss_kappa(q3_matrix)

    for q_name, q_key in [("Q2 (reasoning)", "q2"), ("Q4 (naturalness)", "q4")]:
        scores = [[annotators[n].get(sid, {}).get(q_key) for sid in sample_ids] for n in annotator_names]
        corrs = []
        for (i, j) in combinations(range(n_annotators), 2):
            valid = [(scores[i][k], scores[j][k]) for k in range(len(sample_ids))
                     if scores[i][k] is not None and scores[j][k] is not None]
            if valid:
                s1, s2 = zip(*valid)
                r, _ = spearmanr(s1, s2)
                corrs.append(r)
        print(f"  {q_name} mean pairwise Spearman: {np.mean(corrs):.3f}")

    print(f"  Q1 Fleiss kappa: {fk_q1:.3f}  |  Q1 mean pairwise Cohen's kappa: {np.mean(q1_kappas):.3f}")
    print(f"  Q3 Fleiss kappa: {fk_q3:.3f}")

    # ── Consensus labels ──
    consensus = {}
    for sid in sample_ids:
        q1_votes = [v for v in (annotators[n].get(sid, {}).get("q1") for n in annotator_names) if v]
        q3_votes = [v for v in (annotators[n].get(sid, {}).get("q3") for n in annotator_names) if v]
        q2_scores = [v for v in (annotators[n].get(sid, {}).get("q2") for n in annotator_names) if v is not None]
        q4_scores = [v for v in (annotators[n].get(sid, {}).get("q4") for n in annotator_names) if v is not None]
        consensus[sid] = {
            "q1": consensus_q1(q1_votes) if q1_votes else None,
            "q2": consensus_score(q2_scores) if q2_scores else None,
            "q3": consensus_binary(q3_votes) if q3_votes else None,
            "q4": consensus_score(q4_scores) if q4_scores else None,
        }

    q1_dist = Counter(c["q1"] for c in consensus.values())
    n_correct = sum(1 for c in consensus.values() if c["q1"] == "Yes")
    human_rsr = n_correct / n_total
    q2_vals = [c["q2"] for c in consensus.values() if c["q2"] is not None]
    n_revealed = sum(1 for c in consensus.values() if c["q3"] == "Yes")
    q4_vals = [c["q4"] for c in consensus.values() if c["q4"] is not None]

    print(f"  Q1 distribution: {dict(q1_dist)}")
    print(f"  Human RSR (Yes only): {n_correct}/{n_total} = {human_rsr:.1%}")
    print(f"  Q2 mean: {np.mean(q2_vals):.2f}  Q4 mean: {np.mean(q4_vals):.2f} (SD {np.std(q4_vals, ddof=1):.3f})")
    print(f"  Q3 answer revealed: {n_revealed}/{n_total} = {n_revealed/n_total:.1%}")

    summary = {
        "human_rsr": round(human_rsr, 3),
        "q1_distribution": dict(q1_dist),
        "q2_mean": round(float(np.mean(q2_vals)), 2),
        "q2_sd": round(float(np.std(q2_vals, ddof=1)), 2),
        "q3_revealed_rate": round(n_revealed / n_total, 3),
        "q4_mean": round(float(np.mean(q4_vals)), 2),
        "q4_sd": round(float(np.std(q4_vals, ddof=1)), 2),
    }

    output = {
        "agreement": {
            "q1_fleiss_kappa": round(fk_q1, 3),
            "q1_mean_pairwise_kappa": round(float(np.mean(q1_kappas)), 3),
            "q3_fleiss_kappa": round(fk_q3, 3),
        },
        "consensus": consensus,
    }
    meta = {
        "dataset": label,
        "n_samples": n_total,
        "annotators": annotator_names,
        "method": (
            "replicates scripts/aggregate_human_eval.py consensus/agreement methodology "
            "(median for Q2/Q4, majority-vote+1-1-1->Partial for Q1, majority-vote+tie->Yes "
            "for Q3, Cohen's/Fleiss' kappa as defined there), reading the CORRECTED per-sample "
            "fields (tier1_corrected/tier2_corrected/gpt54_corrected) instead of the legacy "
            "tier1_correct/tier2_correct/gpt54_correct field names."
        ),
        "regenerated_by": "scripts/regenerate_human_eval_aggregates.py",
    }

    # ── HUMAN vs AUTOMATED EVALUATION — only when all three corrected fields present ──
    required = ["tier1_corrected", "tier2_corrected", "gpt54_corrected"]
    present = {
        f: all(f in samples[sid] and samples[sid][f] is not None for sid in sample_ids)
        for f in required
    }
    missing_fields = [f for f, ok in present.items() if not ok]

    if not missing_fields:
        human_labels, tier1_labels, tier2_labels, gpt54_labels = [], [], [], []
        for sid in sample_ids:
            s = samples[sid]
            human_labels.append(1 if consensus[sid]["q1"] == "Yes" else 0)
            tier1_labels.append(1 if s["tier1_corrected"] else 0)
            tier2_labels.append(1 if s["tier2_corrected"] else 0)
            gpt54_labels.append(1 if s["gpt54_corrected"] else 0)

        tier1_rsr = sum(tier1_labels) / n_total
        tier2_rsr = sum(tier2_labels) / n_total
        gpt54_rsr = sum(gpt54_labels) / n_total

        k_tier1 = cohens_kappa(human_labels, tier1_labels)
        k_tier2 = cohens_kappa(human_labels, tier2_labels)
        k_gpt54 = cohens_kappa(human_labels, gpt54_labels)
        agree_tier1 = sum(a == b for a, b in zip(human_labels, tier1_labels)) / n_total
        agree_tier2 = sum(a == b for a, b in zip(human_labels, tier2_labels)) / n_total
        agree_gpt54 = sum(a == b for a, b in zip(human_labels, gpt54_labels)) / n_total

        print("  --- HUMAN vs AUTOMATED (corrected fields) ---")
        print(f"  Tier1 RSR={tier1_rsr:.1%} kappa={k_tier1:.3f} agree={agree_tier1:.1%}")
        print(f"  Tier2 RSR={tier2_rsr:.1%} kappa={k_tier2:.3f} agree={agree_tier2:.1%}")
        print(f"  GPT-5.4 RSR={gpt54_rsr:.1%} kappa={k_gpt54:.3f} agree={agree_gpt54:.1%}")

        summary.update({
            "tier1_rsr": round(tier1_rsr, 3),
            "tier2_rsr": round(tier2_rsr, 3),
            "gpt54_rsr": round(gpt54_rsr, 3),
            "kappa_tier1_vs_human": round(k_tier1, 3),
            "kappa_tier2_vs_human": round(k_tier2, 3),
            "kappa_gpt54_vs_human": round(k_gpt54, 3),
            "agree_tier1": round(agree_tier1, 3),
            "agree_tier2": round(agree_tier2, 3),
            "agree_gpt54": round(agree_gpt54, 3),
        })
    else:
        print(f"  --- HUMAN vs AUTOMATED skipped: missing corrected field(s) {missing_fields} ---")
        meta["omitted_section"] = (
            "HUMAN vs AUTOMATED EVALUATION (tier1/tier2/gpt54 RSR + kappa) omitted: "
            f"corrected field(s) {missing_fields} absent (or null for some sample) in this "
            "samples schema. Not computed here to avoid saving a silently-wrong or fabricated "
            "partial (2-of-3) table."
        )

    output["summary"] = summary
    output["meta"] = meta

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        backup_path = output_path.with_name(output_path.name + ".legacy_prefix_bug")
        output_path.replace(backup_path)
        print(f"  Backed up previous file to: {backup_path}")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  Saved corrected aggregate to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Regenerate human-eval aggregates with corrected fields")
    parser.add_argument("--dataset", choices=sorted(PRESETS.keys()), help="Preset dataset to regenerate")
    parser.add_argument("--all", action="store_true", help="Regenerate all presets")
    parser.add_argument("--samples", help="Custom samples JSON path (overrides preset)")
    parser.add_argument("--annotators", nargs="+", help="Custom annotator JSON paths (overrides preset)")
    parser.add_argument("--output", help="Custom output path (overrides preset)")
    parser.add_argument("--label", default="custom", help="Label for meta.dataset when using custom paths")
    args = parser.parse_args()

    if args.samples or args.annotators or args.output:
        if not (args.samples and args.annotators and args.output):
            print("ERROR: --samples/--annotators/--output must all be given together for custom mode")
            sys.exit(1)
        regenerate("custom", args.samples, args.annotators, args.output, args.label)
        return

    keys = list(PRESETS.keys()) if args.all else ([args.dataset] if args.dataset else [])
    if not keys:
        print("ERROR: pass --dataset <name>, --all, or the custom --samples/--annotators/--output trio")
        sys.exit(1)

    for key in keys:
        p = PRESETS[key]
        regenerate(key, p["samples"], p["annotators"], p["output"], p["label"])


if __name__ == "__main__":
    main()
