#!/usr/bin/env python3
"""Select 50 semantically clean samples from the screened pool.

Strategy:
  1. Take ALL rare categories (concordant_success, discordant) — census
  2. Fill remaining with concordant_failure, balanced by tutor/strategy/type
  3. Quality <= 0.25, pair_clean=True
  4. Tutor balance: proportional to available clean pairs (Mistral gets all 4)
"""
import json
import random
from pathlib import Path
from collections import Counter

SEED = 42
TARGET = 50
INPUT = Path("paper/human_eval/semantic_screen_results.json")
OUTPUT_DIR = Path("paper/human_eval")


def main():
    random.seed(SEED)
    data = json.load(open(INPUT))

    # Filter: pair_clean AND quality<=0.25
    eligible = [r for r in data if r["pair_clean"] and r["composite_score"] <= 0.25]
    print(f"Eligible pool: {len(eligible)}")

    selected = []

    # ── Tier 1: Census of rare categories ──────────────────────────────
    for cat in ["concordant_success", "discordant_same_yes_gpt_no", "discordant_same_no_gpt_yes"]:
        items = [r for r in eligible if r["category"] == cat]
        items.sort(key=lambda r: r["composite_score"])
        selected.extend(items)
        print(f"  Tier 1 ({cat}): {len(items)}")

    tier1_n = len(selected)
    tier1_keys = {(r["tutor"], r["strategy"], r["misconception_type"], r["instance_id"]) for r in selected}
    print(f"  Tier 1 total: {tier1_n}")

    # ── Tier 2: Stratified concordant_failure ───────────────────────────
    n_remaining = TARGET - tier1_n
    failures = [r for r in eligible
                if r["category"] == "concordant_failure"
                and (r["tutor"], r["strategy"], r["misconception_type"], r["instance_id"]) not in tier1_keys]
    failures.sort(key=lambda r: r["composite_score"])

    print(f"\n  Tier 2 pool: {len(failures)} concordant_failure pairs")
    print(f"  Need: {n_remaining} more")

    # Tutor-balanced quotas
    tier1_by_tutor = Counter(r["tutor"] for r in selected)
    tutors = sorted(set(r["tutor"] for r in failures) | set(tier1_by_tutor.keys()))

    # Available per tutor
    avail_by_tutor = Counter(r["tutor"] for r in failures)
    print(f"\n  Available by tutor: {dict(avail_by_tutor)}")
    print(f"  Tier 1 by tutor: {dict(tier1_by_tutor)}")

    # Compute quotas: try to equalize total per tutor
    target_per_tutor = TARGET // len(tutors)  # 10 each for 5 tutors
    tier2_quota = {}
    for t in tutors:
        desired = max(target_per_tutor - tier1_by_tutor.get(t, 0), 0)
        available = avail_by_tutor.get(t, 0)
        tier2_quota[t] = min(desired, available)

    # Distribute remainder
    allocated = sum(tier2_quota.values())
    remainder = n_remaining - allocated
    if remainder > 0:
        # Give extra to tutors with most available
        for t in sorted(tutors, key=lambda t: avail_by_tutor.get(t, 0) - tier2_quota[t], reverse=True):
            if remainder <= 0:
                break
            extra = min(remainder, avail_by_tutor.get(t, 0) - tier2_quota[t])
            if extra > 0:
                tier2_quota[t] += extra
                remainder -= extra

    print(f"  Tier 2 quotas: {dict(tier2_quota)}")

    # Within each tutor, stratify by (strategy, type)
    tier2 = []
    for tutor in tutors:
        quota = tier2_quota[tutor]
        if quota == 0:
            continue

        tutor_pool = [r for r in failures if r["tutor"] == tutor]

        # Group by (strategy, type)
        strata = {}
        for r in tutor_pool:
            key = (r["strategy"], r["misconception_type"])
            strata.setdefault(key, []).append(r)

        # First pass: 1 best per stratum
        tutor_selected = []
        for key in sorted(strata.keys()):
            items = sorted(strata[key], key=lambda r: r["composite_score"])
            if items:
                tutor_selected.append(items[0])

        if len(tutor_selected) >= quota:
            # Shuffle for randomness, then take quota
            rng = random.Random(SEED + hash(tutor))
            rng.shuffle(tutor_selected)
            tutor_selected = tutor_selected[:quota]
        else:
            # Fill from remaining
            used = {id(r) for r in tutor_selected}
            pool = [r for r in tutor_pool if id(r) not in used]
            pool.sort(key=lambda r: r["composite_score"])
            tutor_selected.extend(pool[:quota - len(tutor_selected)])

        tier2.extend(tutor_selected)

    selected.extend(tier2)
    print(f"  Tier 2 selected: {len(tier2)}")
    print(f"\n  Total selected: {len(selected)}")

    # ── Diversity report ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("DIVERSITY REPORT")
    print(f"{'='*60}")
    for dim in ["tutor", "category", "strategy", "misconception_type"]:
        dist = Counter(r[dim] for r in selected)
        items_str = ", ".join(f"{k}={v}" for k, v in sorted(dist.items()))
        print(f"  {dim}: {items_str}")

    # Quality stats
    scores = [r["composite_score"] for r in selected]
    print(f"\n  Quality: min={min(scores):.4f}, max={max(scores):.4f}, "
          f"mean={sum(scores)/len(scores):.4f}")

    # ── Save selection ─────────────────────────────────────────────────
    # Shuffle for randomness
    random.shuffle(selected)

    # Assign sample IDs
    for i, r in enumerate(selected):
        r["sample_id"] = f"S{i+1:03d}"

    out_path = OUTPUT_DIR / "clean50_selection.json"
    with open(out_path, "w") as f:
        json.dump({
            "metadata": {
                "n_samples": len(selected),
                "pool_size": len(eligible),
                "quality_threshold": 0.25,
                "seed": SEED,
                "selection_method": "semantic_screen + stratified",
            },
            "selected": selected,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved selection to {out_path}")

    # Print selected keys for reference
    print(f"\n--- Selected sample keys ---")
    for r in sorted(selected, key=lambda r: r["sample_id"]):
        print(f"  {r['sample_id']}: {r['tutor']} / {r['strategy']} / "
              f"{r['misconception_type']} / {r['instance_id']} "
              f"[{r['category']}, q={r['composite_score']:.3f}]")


if __name__ == "__main__":
    main()
