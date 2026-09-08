"""
exp2_verify_corrections.py
==========================
Audits Exp2 result JSON files for the parse bug where model responses like
  "1. MaE02 (conceptual, fractions) - The student..."
were stored verbatim as predicted_ranked[0] instead of the bare ID "MaE02".

For each (model, condition) cell we:
  1. Read stored acc@1 from the metrics block.
  2. Re-compute corrected acc@1 by extracting the first Ma-style misconception-ID pattern from
     predicted_ranked[0] before comparing to true_misconception_id.
  3. Print a comparison table and flag disputed cells.

Usage:
    conda run -n X python scripts/exp2_verify_corrections.py
"""

import json
import re
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "exp2"

MODELS = [
    "qwen-2_5-7b",
    "qwen3_5-9b",
    "llama-3_1-8b",
    "mistral-7b",
    "phi-4",
]

CONDITIONS = [
    ("zero_shot",          "ZS"),
    ("few_shot_5",         "FS"),
    ("cot_rubric",         "CoT"),
    ("rag",                "RAG"),
    ("gen_retrieve_rerank","GRR"),
]

# Disputed cells to highlight explicitly
DISPUTED = {
    ("qwen3_5-9b",  "zero_shot"):           {"stored_expected": 0.523, "audit_expected": 0.545},
    ("mistral-7b",  "cot_rubric"):          {"stored_expected": 0.000, "audit_expected": 0.214},
    ("mistral-7b",  "gen_retrieve_rerank"): {"stored_expected": 0.000, "audit_expected": 0.255},
}

# ---------------------------------------------------------------------------
# ID extraction
# ---------------------------------------------------------------------------
# Primary pattern: Ma<Letter><Digit><Digit>  e.g. MaE01, MaA12
# Also handles less-common prefixes if they ever appear.
_ID_RE = re.compile(r'\b(Ma[A-Z]\d{2,3})\b')


def extract_id(raw: str) -> str | None:
    """Return the first misconception ID found in *raw*, or None."""
    m = _ID_RE.search(raw)
    return m.group(1) if m else None


def is_bare_id(s: str) -> bool:
    """True when the string IS a bare misconception ID (nothing else)."""
    return bool(re.fullmatch(r'Ma[A-Z]\d{2,3}', s.strip()))


# ---------------------------------------------------------------------------
# Per-file analysis
# ---------------------------------------------------------------------------
def analyse_file(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)

    stored_acc1 = data["metrics"]["overall"]["acc_at_1"]
    predictions = data["predictions"]
    n = len(predictions)

    corrected_hits = 0
    stored_hits    = 0
    parse_bugs     = []   # indices where top-1 was dirty

    for i, pred in enumerate(predictions):
        true_id = pred.get("true_misconception_id") or pred.get("true_misconception")
        ranked  = pred.get("predicted_ranked", [])

        if not ranked:
            continue

        raw_top1 = ranked[0]

        # -- stored hit (exact match as-is) --
        if raw_top1.strip() == true_id:
            stored_hits += 1

        # -- corrected hit --
        if is_bare_id(raw_top1):
            extracted = raw_top1.strip()
        else:
            extracted = extract_id(raw_top1)
            if extracted is not None:
                parse_bugs.append({
                    "idx":    i,
                    "qid":    pred.get("question_id", "?"),
                    "true":   true_id,
                    "raw":    raw_top1[:120],
                    "parsed": extracted,
                })

        if extracted and extracted == true_id:
            corrected_hits += 1

    corrected_acc1 = corrected_hits / n if n else 0.0
    recomputed_stored = stored_hits / n if n else 0.0

    return {
        "n":                n,
        "stored_acc1":      stored_acc1,
        "recomputed_stored": recomputed_stored,  # sanity-check against stored
        "corrected_acc1":   corrected_acc1,
        "delta":            corrected_acc1 - stored_acc1,
        "n_bugs":           len(parse_bugs),
        "bugs":             parse_bugs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 90)
    print("Exp2 Parse-Bug Correction Audit")
    print("=" * 90)
    print()

    # Column headers
    hdr = f"{'Model':<18} {'Cond':<6} {'n':>4}  {'StoredAcc@1':>11}  {'RecompStored':>12}  "
    hdr += f"{'CorrAcc@1':>10}  {'Delta':>7}  {'nBugs':>6}  {'Flag'}"
    print(hdr)
    print("-" * 90)

    results_table = []

    for model in MODELS:
        for cond_key, cond_label in CONDITIONS:
            fname = f"{model}_{cond_key}.json"
            fpath = RESULTS_DIR / fname
            if not fpath.exists():
                print(f"  !! MISSING: {fname}")
                continue

            r = analyse_file(fpath)

            flag = ""
            key  = (model, cond_key)
            if key in DISPUTED:
                exp = DISPUTED[key]
                flag = "DISPUTED"
                # Annotate with what audit expected
                flag += f"  audit_expected={exp['audit_expected']:.3f}"

            delta_str = f"{r['delta']:+.4f}" if abs(r['delta']) > 1e-6 else "    —  "
            row = (
                f"{model:<18} {cond_label:<6} {r['n']:>4}  "
                f"{r['stored_acc1']:>11.4f}  "
                f"{r['recomputed_stored']:>12.4f}  "
                f"{r['corrected_acc1']:>10.4f}  "
                f"{delta_str:>7}  "
                f"{r['n_bugs']:>6}  "
                f"{flag}"
            )
            print(row)
            results_table.append({
                "model": model, "condition": cond_key, "label": cond_label, **r,
                "disputed": key in DISPUTED,
            })

        print()   # blank line between models

    # ------------------------------------------------------------------
    # Disputed-cell deep-dive
    # ------------------------------------------------------------------
    print("=" * 90)
    print("DISPUTED CELLS – detailed breakdown")
    print("=" * 90)

    for model in MODELS:
        for cond_key, cond_label in CONDITIONS:
            key = (model, cond_key)
            if key not in DISPUTED:
                continue

            fname = f"{model}_{cond_key}.json"
            fpath = RESULTS_DIR / fname
            if not fpath.exists():
                continue

            r = analyse_file(fpath)
            exp = DISPUTED[key]

            print(f"\n--- {model} / {cond_label} ---")
            print(f"  stored acc@1         : {r['stored_acc1']:.4f}  (expected in paper: {exp['stored_expected']:.3f})")
            print(f"  recomputed stored    : {r['recomputed_stored']:.4f}  (sanity check)")
            print(f"  corrected acc@1      : {r['corrected_acc1']:.4f}  (audit expected:   {exp['audit_expected']:.3f})")
            print(f"  delta                : {r['delta']:+.4f}")
            print(f"  n predictions        : {r['n']}")
            print(f"  n parse-bug entries  : {r['n_bugs']}")

            if r['bugs']:
                print(f"\n  First 5 parse-bug examples:")
                for b in r['bugs'][:5]:
                    match_marker = "HIT" if b['parsed'] == b['true'] else "miss"
                    print(f"    [{match_marker}] qid={b['qid']}  true={b['true']}  parsed={b['parsed']}")
                    print(f"          raw: {b['raw']}")

    # ------------------------------------------------------------------
    # Summary: models where any correction changes the value
    # ------------------------------------------------------------------
    print()
    print("=" * 90)
    print("SUMMARY – cells with non-zero delta (corrected ≠ stored)")
    print("=" * 90)
    any_changed = False
    for r in results_table:
        if abs(r['delta']) > 1e-6:
            any_changed = True
            print(f"  {r['model']:<18} {r['label']:<6}  "
                  f"stored={r['stored_acc1']:.4f}  corrected={r['corrected_acc1']:.4f}  "
                  f"delta={r['delta']:+.4f}  n_bugs={r['n_bugs']}")
    if not any_changed:
        print("  (none – all stored values match corrected values)")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
