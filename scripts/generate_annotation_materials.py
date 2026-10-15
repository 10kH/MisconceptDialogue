#!/usr/bin/env python3
"""Generate annotation materials from the extracted human-evaluation samples.

Input: the samples file written by scripts/gpt54_quality_screen.py
(default paper/human_eval_v2/annotation_samples.json).

Outputs (under the output directory, default paper/human_eval_v2/):
  1. annotation_form.tsv — response form (one row per dialogue)
  2. samples/ — individual sample files for annotators to read
  3. annotation_form_template.tsv — empty form per annotator

Usage:
  python scripts/generate_annotation_materials.py [--input PATH] [--output-dir DIR]
"""

import json
import re
import csv
from pathlib import Path

import argparse

_ap = argparse.ArgumentParser(description="Generate human-evaluation annotation materials.")
_ap.add_argument("--input", default="paper/human_eval_v2/annotation_samples.json",
                 help="samples file written by scripts/gpt54_quality_screen.py")
_ap.add_argument("--output-dir", default="paper/human_eval_v2", help="where to write the materials")
_args = _ap.parse_args()
INPUT = Path(_args.input)
OUTPUT_DIR = Path(_args.output_dir)
SAMPLES_DIR = OUTPUT_DIR / "samples"


def strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning traces from model output."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def format_dialogue(turns: list[dict]) -> str:
    """Format dialogue turns into readable text."""
    lines = []
    for t in turns:
        role = "TUTOR" if t["role"] == "tutor" else "STUDENT"
        content = strip_think_blocks(t["content"])
        # Truncate only extremely long turns (>3000 chars)
        if len(content) > 3000:
            content = content[:3000] + "\n[... truncated ...]"
        lines.append(f"[{role}]: {content}")
    return "\n\n".join(lines)


def format_verification(answer: str) -> str:
    """Clean up verification answer."""
    answer = strip_think_blocks(answer)
    if len(answer) > 3000:
        answer = answer[:3000] + "\n[... truncated ...]"
    return answer


def generate_sample_file(sample: dict, idx: int) -> str:
    """Generate a readable text file for one sample."""
    lines = []
    lines.append(f"{'='*70}")
    lines.append(f"SAMPLE {sample['sample_id']}")
    lines.append(f"{'='*70}")
    lines.append("")
    lines.append(f"Strategy: {sample['strategy']}")
    lines.append(f"Misconception Type: {sample['misconception_type']}")
    lines.append("")

    # Dialogue A
    lines.append(f"{'-'*70}")
    lines.append("DIALOGUE A")
    lines.append(f"{'-'*70}")
    lines.append("")
    lines.append(format_dialogue(sample["dialogue_A"]))
    lines.append("")
    lines.append("--- VERIFICATION ANSWER (A) ---")
    lines.append(format_verification(sample["verification_answer_A"]))
    lines.append("")

    # Dialogue B
    lines.append(f"{'-'*70}")
    lines.append("DIALOGUE B")
    lines.append(f"{'-'*70}")
    lines.append("")
    lines.append(format_dialogue(sample["dialogue_B"]))
    lines.append("")
    lines.append("--- VERIFICATION ANSWER (B) ---")
    lines.append(format_verification(sample["verification_answer_B"]))
    lines.append("")

    return "\n".join(lines)


def main():
    data = json.load(open(INPUT))
    samples = data["samples"]
    n = len(samples)
    print(f"Loaded {n} samples")

    # ── 1. Generate individual sample files ──────────────────────────
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(samples):
        content = generate_sample_file(s, i)
        path = SAMPLES_DIR / f"{s['sample_id']}.txt"
        path.write_text(content, encoding="utf-8")
    print(f"Generated {n} sample files in {SAMPLES_DIR}/")

    # ── 2. Generate combined samples document ────────────────────────
    all_samples = []
    for i, s in enumerate(samples):
        all_samples.append(generate_sample_file(s, i))
    combined_path = OUTPUT_DIR / "all_samples.txt"
    combined_path.write_text("\n\n".join(all_samples), encoding="utf-8")
    print(f"Generated combined document: {combined_path}")

    # ── 3. Generate annotation form (TSV) ────────────────────────────
    # Two rows per sample: one for Dialogue A, one for Dialogue B
    form_path = OUTPUT_DIR / "annotation_form.tsv"
    with open(form_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "annotator_id",
            "sample_id",
            "dialogue",
            "strategy",
            "misconception_type",
            "Q1_remediation_success",
            "Q2_answer_revealed",
            "Q3_tutoring_quality",
            "notes",
        ])
        for s in samples:
            for dial in ["A", "B"]:
                writer.writerow([
                    "",  # annotator fills
                    s["sample_id"],
                    dial,
                    s["strategy"],
                    s["misconception_type"],
                    "",  # Q1: Yes/No/Unclear
                    "",  # Q2: Yes/No
                    "",  # Q3: 1-5
                    "",  # notes
                ])
    print(f"Generated annotation form: {form_path} ({n * 2} rows)")

    # ── 4. Generate answer key (hidden from annotators) ──────────────
    key_path = OUTPUT_DIR / "answer_key.tsv"
    with open(key_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow([
            "sample_id",
            "dialogue",
            "evaluator",
            "auto_rsr",
            "category",
            "tutor_model",
            "misconception_type",
            "strategy",
            "instance_id",
        ])
        for s in samples:
            for dial in ["A", "B"]:
                evaluator = s["_mapping"][dial]
                if evaluator == "same_scale":
                    auto_rsr = s["_same_scale_success"]
                else:
                    auto_rsr = s["_gpt5_mini_success"]
                writer.writerow([
                    s["sample_id"],
                    dial,
                    evaluator,
                    "Yes" if auto_rsr else "No",
                    s["_category"],
                    s["tutor_model"],
                    s["misconception_type"],
                    s["strategy"],
                    s["instance_id"],
                ])
    print(f"Generated answer key: {key_path}")

    # ── 5. Summary ───────────────────────────────────────────────────
    print(f"\n=== Annotation Package Complete ===")
    print(f"  Samples: {n}")
    print(f"  Judgments needed per annotator: {n * 2} (A + B for each sample)")
    print(f"  Files:")
    print(f"    Guidelines:    {OUTPUT_DIR}/annotation_guidelines.md")
    print(f"    Sample files:  {SAMPLES_DIR}/S001.txt ... S{n:03d}.txt")
    print(f"    Combined doc:  {OUTPUT_DIR}/all_samples.txt")
    print(f"    Response form: {OUTPUT_DIR}/annotation_form.tsv")
    print(f"    Answer key:    {OUTPUT_DIR}/answer_key.tsv (DO NOT share)")


if __name__ == "__main__":
    main()
