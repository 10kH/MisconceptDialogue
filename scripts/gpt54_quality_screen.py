#!/usr/bin/env python3
"""GPT-5.4 qualitative screening of clean exp3v2 dialogues for human evaluation.

Second-pass screening: takes the 1,199 heuristically-clean dialogues and has
GPT-5.4 evaluate each for subtle quality issues that regex/n-gram checks miss.

Usage:
  set -a && source .env && set +a && conda run -n X --no-capture-output \
    python scripts/gpt54_quality_screen.py [--dry-run] [--resume]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.utils.common import latest_run_dirs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCREEN_RESULTS = PROJECT_ROOT / "results" / "exp3v2" / "quality_screen_results.json"
SCENARIO_BANK = PROJECT_ROOT / "data" / "exp3v2_scenario_bank.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / "exp3v2" / "gpt54_screen"

# {model_short: run_dir_name}, newest run per model under results/exp3v2/
MODEL_DIRS = latest_run_dirs()

QUALITY_PROMPT = """You are an expert evaluator of AI tutoring dialogue quality. Evaluate this math tutoring dialogue for use in a human evaluation study.

## Scenario
- Problem: {problem}
- Correct answer: {correct_answer}
- Student's misconception: {misconception}
- Tutoring strategy: {strategy}

## Dialogue
{dialogue_text}

## Student's Verification Response (after tutoring)
{verification_answer}

## Evaluation Criteria
Rate each dimension 1-5:

1. **dialogue_coherence** (1-5): Do tutor and student turns flow naturally? No non-sequiturs, no turns that ignore previous context?
2. **tutor_quality** (1-5): Does the tutor actually address the misconception? Uses appropriate pedagogy? Doesn't just give the answer?
3. **student_authenticity** (1-5): Does the student behave like a real student with this misconception? Not too compliant, not robotic?
4. **role_integrity** (1-5): Does each party stay in role? Student doesn't act like a teacher, tutor doesn't act confused?
5. **verification_quality** (1-5): Is the verification response substantive and interpretable? Not truncated, not evasive?
6. **overall_suitability** (1-5): Overall, is this dialogue suitable for expert human evaluation of tutoring effectiveness?

Also flag any specific issues:
- answer_leaked: Did the tutor explicitly reveal the correct answer?
- student_parroting: Does the student just repeat what the tutor said without showing understanding?
- nonsensical: Any mathematically nonsensical statements?
- meta_talk: Does either party break character or reference being an AI?

Respond ONLY with JSON:
{{"dialogue_coherence": int, "tutor_quality": int, "student_authenticity": int, "role_integrity": int, "verification_quality": int, "overall_suitability": int, "issues": ["issue1", ...], "brief_note": "one-sentence quality summary"}}"""


def load_clean_dialogues() -> list[dict]:
    """Load clean dialogues from quality screen results + original cell files."""
    screen = json.loads(SCREEN_RESULTS.read_text())
    scenarios = {s["id"]: s for s in json.loads(SCENARIO_BANK.read_text())}

    clean_items = [r for r in screen["results"] if r["classification"] == "clean"]
    logger.info("Found %d clean dialogues from heuristic screen", len(clean_items))

    # Load actual dialogue content from cell files
    cell_cache: dict[str, list] = {}
    dialogues = []

    for item in clean_items:
        model = item["model"]
        cell_file = item["cell_file"]
        idx = item["idx_in_cell"]

        cache_key = f"{model}/{cell_file}"
        if cache_key not in cell_cache:
            model_dir = MODEL_DIRS[model]
            path = PROJECT_ROOT / "results" / "exp3v2" / model_dir / cell_file
            if path.exists():
                cell_cache[cache_key] = json.loads(path.read_text()).get("results", [])
            else:
                cell_cache[cache_key] = []

        results = cell_cache[cache_key]
        if idx < len(results):
            r = results[idx]
            scenario = scenarios.get(r.get("scenario_id", ""), {})

            # Build dialogue text
            tutor_turns = r.get("tutor_responses", [])
            student_turns = r.get("student_responses", [])
            dialogue_lines = []
            for i in range(max(len(tutor_turns), len(student_turns))):
                if i < len(tutor_turns):
                    dialogue_lines.append(f"Tutor: {tutor_turns[i]}")
                if i < len(student_turns):
                    dialogue_lines.append(f"Student: {student_turns[i]}")

            dialogues.append({
                "model": model,
                "strategy": item["strategy"],
                "misconception_type": item["misconception_type"],
                "scenario_id": item["scenario_id"],
                "cell_file": cell_file,
                "idx_in_cell": idx,
                "tier1_correct": item["tier1_correct"],
                "tier2_correct": item["tier2_correct"],
                # [FIX] The student solves verification_problem, so use the verification fields.
                "problem": scenario.get("verification_problem", ""),
                "correct_answer": scenario.get("verification_answer", ""),
                "misconception": scenario.get("misconception_label", ""),
                "verification_answer": r.get("verification_answer", ""),
                "dialogue_text": "\n\n".join(dialogue_lines),
                "tutor_responses": tutor_turns,
                "student_responses": student_turns,
                "num_turns": r.get("num_turns", 0),
            })

    return dialogues


def screen_with_gpt54(
    dialogues: list[dict],
    resume_from: dict | None = None,
) -> list[dict]:
    """Screen dialogues using GPT-5.4."""
    from openai import OpenAI
    client = OpenAI()

    results = list(resume_from.values()) if resume_from else []
    seen = set(resume_from.keys()) if resume_from else set()

    total = len(dialogues)
    to_process = [d for d in dialogues if f"{d['model']}__{d['scenario_id']}__{d['strategy']}" not in seen]
    logger.info("Screening %d dialogues (%d already done)", len(to_process), len(seen))

    for i, d in enumerate(to_process):
        key = f"{d['model']}__{d['scenario_id']}__{d['strategy']}"

        prompt = QUALITY_PROMPT.format(
            problem=d["problem"],
            correct_answer=d["correct_answer"],
            misconception=d["misconception"],
            strategy=d["strategy"],
            dialogue_text=d["dialogue_text"][:3000],  # truncate very long dialogues
            verification_answer=d["verification_answer"][:1000],
        )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model="gpt-5.4",
                    messages=[
                        {"role": "system", "content": "You are an expert evaluator of tutoring dialogue quality. Always respond with valid JSON only."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.0,
                    max_completion_tokens=512,
                )
                text = response.choices[0].message.content.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                scores = json.loads(text)
                result = {
                    **{k: d[k] for k in ["model", "strategy", "misconception_type", "scenario_id",
                                          "cell_file", "idx_in_cell", "tier1_correct", "tier2_correct"]},
                    "scores": scores,
                    "min_score": min(scores.get(dim, 0) for dim in
                                     ["dialogue_coherence", "tutor_quality", "student_authenticity",
                                      "role_integrity", "verification_quality", "overall_suitability"]),
                    "mean_score": sum(scores.get(dim, 0) for dim in
                                      ["dialogue_coherence", "tutor_quality", "student_authenticity",
                                       "role_integrity", "verification_quality", "overall_suitability"]) / 6,
                    "issues": scores.get("issues", []),
                    "brief_note": scores.get("brief_note", ""),
                }
                results.append(result)
                break

            except (json.JSONDecodeError, KeyError) as e:
                if attempt < max_retries - 1:
                    logger.warning("Parse error on %s (attempt %d): %s", key, attempt + 1, e)
                    time.sleep(1)
                else:
                    logger.error("Failed after %d attempts: %s", max_retries, key)
                    results.append({
                        **{k: d[k] for k in ["model", "strategy", "misconception_type", "scenario_id",
                                              "cell_file", "idx_in_cell", "tier1_correct", "tier2_correct"]},
                        "scores": {},
                        "min_score": 0,
                        "mean_score": 0,
                        "issues": ["parse_error"],
                        "brief_note": f"GPT-5.4 parse error: {e}",
                    })
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning("API error on %s (attempt %d): %s", key, attempt + 1, e)
                    time.sleep(2 ** attempt)
                else:
                    logger.error("API failed after %d attempts: %s - %s", max_retries, key, e)
                    results.append({
                        **{k: d[k] for k in ["model", "strategy", "misconception_type", "scenario_id",
                                              "cell_file", "idx_in_cell", "tier1_correct", "tier2_correct"]},
                        "scores": {},
                        "min_score": 0,
                        "mean_score": 0,
                        "issues": ["api_error"],
                        "brief_note": f"GPT-5.4 API error: {e}",
                    })

        if (i + 1) % 50 == 0:
            logger.info("  %d/%d done (total results: %d)", i + 1, len(to_process), len(results))
            # Checkpoint save
            _save_checkpoint(results)

    return results


def _save_checkpoint(results: list[dict]):
    """Save intermediate results."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = OUTPUT_DIR / "checkpoint.json"
    checkpoint.write_text(json.dumps(results, indent=2))


def select_samples(results: list[dict], min_samples: int = 100) -> list[dict]:
    """Select high-quality samples with balanced distribution.

    Strategy:
    1. Filter to min_score >= 4 (all dimensions at least 4/5)
    2. No flagged issues (empty issues list)
    3. Stratified by (model, misconception_type) for balance
    4. Within each stratum, pick highest mean_score
    5. Include both tier2 success and failure cases
    """
    # Blocking issues — real quality problems that invalidate the dialogue
    blocking_issues = {"answer_leaked", "role_integrity", "verification_mismatch"}

    # Primary filter: mean_score >= 3.0, no blocking issues
    # "nonsensical" and "student_parroting" are over-flagged by GPT-5.4; allow them
    high_quality = [
        r for r in results
        if r["mean_score"] >= 3.0
        and not (set(r.get("issues", [])) & blocking_issues)
    ]
    logger.info("High quality (mean>=3.0, no blocking issues): %d / %d", len(high_quality), len(results))

    if len(high_quality) < min_samples:
        # Relax to mean_score >= 2.5
        high_quality = [
            r for r in results
            if r["mean_score"] >= 2.5
            and not (set(r.get("issues", [])) & blocking_issues)
        ]
        logger.info("Relaxed (mean>=2.5, no blocking issues): %d", len(high_quality))

    # Group by (model, type) for stratified selection
    from collections import defaultdict
    strata = defaultdict(list)
    for r in high_quality:
        key = (r["model"], r["misconception_type"])
        strata[key].append(r)

    # Target: balanced across 5 models × 3 types = 15 strata
    target_per_stratum = max(min_samples // 15 + 1, 4)  # at least 4 per stratum

    selected = []
    for key, items in sorted(strata.items()):
        # Sort by mean_score descending, then diversify by strategy
        items.sort(key=lambda x: (-x["mean_score"], x["strategy"]))

        # Try to include both tier2 success and failure
        successes = [r for r in items if r["tier2_correct"]]
        failures = [r for r in items if not r["tier2_correct"]]

        n_success = min(len(successes), target_per_stratum // 2 + 1)
        n_failure = min(len(failures), target_per_stratum - n_success)
        n_success = min(len(successes), target_per_stratum - n_failure)

        picked = successes[:n_success] + failures[:n_failure]

        # Diversify strategies within the stratum
        seen_strategies = set()
        final = []
        for r in picked:
            if r["strategy"] not in seen_strategies or len(final) < target_per_stratum:
                final.append(r)
                seen_strategies.add(r["strategy"])

        selected.extend(final[:target_per_stratum])

    logger.info("Selected %d samples across %d strata", len(selected), len(strata))
    return selected


def generate_annotation_package(selected: list[dict], dialogues: list[dict]):
    """Generate annotation files for human evaluators."""
    # Build lookup for full dialogue data
    dlg_lookup = {}
    for d in dialogues:
        key = f"{d['model']}__{d['scenario_id']}__{d['strategy']}"
        dlg_lookup[key] = d

    # Build annotation samples
    import random
    random.seed(42)
    random.shuffle(selected)

    annotation_samples = {
        "metadata": {
            "experiment": "exp3v2",
            "description": "Human evaluation - GPT-5.4 quality-screened dialogues from exp3v2",
            "n_samples": len(selected),
            "selection_method": "heuristic_screen + gpt54_quality (mean>=3.0, no blocking issues) + stratified",
            "models": sorted(set(r["model"] for r in selected)),
            "strategies": sorted(set(r["strategy"] for r in selected)),
            "types": sorted(set(r["misconception_type"] for r in selected)),
        },
        "samples": [],
    }

    for i, r in enumerate(selected):
        key = f"{r['model']}__{r['scenario_id']}__{r['strategy']}"
        dlg = dlg_lookup.get(key, {})

        annotation_samples["samples"].append({
            "sample_id": f"HE_{i+1:03d}",
            "model": r["model"],
            "strategy": r["strategy"],
            "misconception_type": r["misconception_type"],
            "scenario_id": r["scenario_id"],
            "problem": dlg.get("problem", ""),
            "correct_answer": dlg.get("correct_answer", ""),
            "misconception": dlg.get("misconception", ""),
            "tutor_responses": dlg.get("tutor_responses", []),
            "student_responses": dlg.get("student_responses", []),
            "verification_answer": dlg.get("verification_answer", ""),
            "tier1_correct": r["tier1_correct"],
            "tier2_correct": r["tier2_correct"],
            "gpt54_quality_score": r["mean_score"],
            "gpt54_note": r["brief_note"],
        })

    # Save
    out_dir = PROJECT_ROOT / "paper" / "human_eval_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / "annotation_samples.json"
    samples_path.write_text(json.dumps(annotation_samples, indent=2, ensure_ascii=False))
    logger.info("Annotation samples saved to %s", samples_path)

    # Generate TSV annotation form
    tsv_lines = ["sample_id\tmodel\tstrategy\ttype\tproblem_preview\tcorrect_understanding\treasoning_quality\tconfidence\tnotes"]
    for s in annotation_samples["samples"]:
        problem_preview = s["problem"][:80].replace("\t", " ").replace("\n", " ")
        tsv_lines.append(f"{s['sample_id']}\t{s['model']}\t{s['strategy']}\t{s['misconception_type']}\t{problem_preview}\t\t\t\t")

    tsv_path = out_dir / "annotation_form.tsv"
    tsv_path.write_text("\n".join(tsv_lines))
    logger.info("Annotation form saved to %s", tsv_path)

    # Summary
    print(f"\n{'='*60}")
    print(f"ANNOTATION PACKAGE: {len(selected)} samples")
    print(f"{'='*60}")

    from collections import Counter
    model_counts = Counter(r["model"] for r in selected)
    type_counts = Counter(r["misconception_type"] for r in selected)
    strat_counts = Counter(r["strategy"] for r in selected)
    tier2_counts = Counter(r["tier2_correct"] for r in selected)

    print(f"\nBy Model:")
    for m, c in sorted(model_counts.items()):
        print(f"  {m:<16} {c}")
    print(f"\nBy Type:")
    for t, c in sorted(type_counts.items()):
        print(f"  {t:<16} {c}")
    print(f"\nBy Strategy:")
    for s, c in sorted(strat_counts.items()):
        print(f"  {s:<20} {c}")
    print(f"\nTier2 Success/Failure: {tier2_counts[True]}/{tier2_counts[False]}")
    print(f"Mean GPT-5.4 quality: {sum(r['mean_score'] for r in selected)/len(selected):.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--select-only", action="store_true", help="Skip screening, just select from existing results")
    parser.add_argument("--min-samples", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dialogues = load_clean_dialogues()

    if args.dry_run:
        logger.info("DRY RUN: would screen %d clean dialogues with GPT-5.4", len(dialogues))
        return

    if args.select_only:
        # Load existing screening results
        results_path = OUTPUT_DIR / "full_results.json"
        if not results_path.exists():
            results_path = OUTPUT_DIR / "checkpoint.json"
        results = json.loads(results_path.read_text())
        logger.info("Loaded %d existing results", len(results))
    else:
        # Screen with GPT-5.4
        resume_data = None
        if args.resume:
            checkpoint = OUTPUT_DIR / "checkpoint.json"
            if checkpoint.exists():
                existing = json.loads(checkpoint.read_text())
                resume_data = {
                    f"{r['model']}__{r['scenario_id']}__{r['strategy']}": r
                    for r in existing
                }
                logger.info("Resuming from checkpoint: %d results", len(resume_data))

        results = screen_with_gpt54(dialogues, resume_from=resume_data)

        # Save full results
        full_path = OUTPUT_DIR / "full_results.json"
        full_path.write_text(json.dumps(results, indent=2))
        logger.info("Full results saved to %s (%d items)", full_path, len(results))

    # Select and generate package
    selected = select_samples(results, min_samples=args.min_samples)
    generate_annotation_package(selected, dialogues)


if __name__ == "__main__":
    main()
