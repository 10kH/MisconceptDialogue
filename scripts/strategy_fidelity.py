"""
strategy_fidelity.py

Analyzes linguistic patterns in Exp.3v2 tutor responses to verify strategy fidelity.

For each (strategy, type) cell across all model directories, concatenates all
tutor_responses per dialogue, counts marker occurrences, computes distinctiveness
ratios, and runs a chi-squared test to assess whether strategies are
linguistically distinguishable.
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import json
import re
import glob
import os
from collections import defaultdict

import numpy as np
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "exp3v2"
)
OUTPUT_JSON = os.path.join(RESULTS_DIR, "strategy_fidelity.json")
OUTPUT_TXT = os.path.join(RESULTS_DIR, "strategy_fidelity.txt")

STRATEGIES = [
    "counterexample",
    "analogy",
    "cognitive_conflict",
    "direct_instruction",
    "socratic",
]

TYPES = ["factual", "procedural", "conceptual"]

# Markers are matched as whole-phrase substrings (case-insensitive).
# Phrases with "?" are matched literally as the character.
STRATEGY_MARKERS: dict[str, list[str]] = {
    "counterexample": [
        "what if",
        "consider the case",
        "suppose",
        "for example",
        "let's try",
        "imagine",
        "what happens when",
        "plug in",
        "substitute",
    ],
    "analogy": [
        "like",
        "similar to",
        "think of it as",
        "just as",
        "compare",
        "imagine",
        "real world",
        "everyday",
        "relate",
    ],
    "cognitive_conflict": [
        "?",
        "what do you think",
        "does that make sense",
        "how would you explain",
        "why do you think",
        "can you see",
        "notice that",
        "but then",
        "contradiction",
    ],
    "direct_instruction": [
        "the correct",
        "actually",
        "the rule is",
        "remember that",
        "the answer is",
        "you should",
        "the formula",
        "in fact",
        "let me explain",
    ],
    "socratic": [
        "?",
        "what",
        "why",
        "how",
        "can you",
        "what if",
        "do you think",
        "tell me",
        "explain",
    ],
}


# ---------------------------------------------------------------------------
# Helper: count marker occurrences in text
# ---------------------------------------------------------------------------

def _count_marker(text: str, marker: str) -> int:
    """Count non-overlapping occurrences of marker in text (case-insensitive)."""
    if marker == "?":
        return text.count("?")
    # Word-boundary aware for single-word markers to avoid partial matches.
    # For multi-word phrases, plain substring search is appropriate.
    words_in_marker = marker.split()
    if len(words_in_marker) == 1:
        pattern = r"\b" + re.escape(marker) + r"\b"
        return len(re.findall(pattern, text, flags=re.IGNORECASE))
    else:
        return len(re.findall(re.escape(marker), text, flags=re.IGNORECASE))


def count_markers_for_strategy(text: str, strategy: str) -> int:
    """Sum all marker counts for a given strategy."""
    total = 0
    for marker in STRATEGY_MARKERS[strategy]:
        total += _count_marker(text, marker)
    return total


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_model_dirs() -> list[str]:
    """Return all 2026* model directories under RESULTS_DIR."""
    pattern = os.path.join(RESULTS_DIR, "2026*")
    dirs = sorted(glob.glob(pattern))
    return [d for d in dirs if os.path.isdir(d)]


def load_all_dialogues() -> list[dict]:
    """
    Load every cell file from every model directory.

    Returns a list of dicts with keys:
        model, strategy, misconception_type, tutor_text
    where tutor_text is the concatenated tutor_responses for one scenario.
    """
    records = []
    model_dirs = find_model_dirs()
    if not model_dirs:
        raise FileNotFoundError(
            f"No model directories found under {RESULTS_DIR}"
        )

    for model_dir in model_dirs:
        model_name = re.sub(r"^\d{8}_\d{6}_", "", os.path.basename(model_dir))
        for strategy in STRATEGIES:
            for mtype in TYPES:
                cell_path = os.path.join(
                    model_dir, f"cell_{strategy}_{mtype}.json"
                )
                if not os.path.exists(cell_path):
                    continue
                with open(cell_path, encoding="utf-8") as f:
                    cell = json.load(f)
                for scenario in cell.get("results", []):
                    tutor_responses = scenario.get("tutor_responses", [])
                    if not tutor_responses:
                        continue
                    tutor_text = " ".join(tutor_responses).lower()
                    records.append(
                        {
                            "model": model_name,
                            "strategy": strategy,
                            "misconception_type": mtype,
                            "tutor_text": tutor_text,
                        }
                    )
    return records


# ---------------------------------------------------------------------------
# Compute unique / shared markers
# ---------------------------------------------------------------------------

def marker_overlap_report() -> dict:
    """
    For each strategy, report which markers are unique to it vs. shared.
    """
    marker_to_strategies: dict[str, list[str]] = defaultdict(list)
    for strategy, markers in STRATEGY_MARKERS.items():
        for m in markers:
            marker_to_strategies[m].append(strategy)

    unique: dict[str, list[str]] = {s: [] for s in STRATEGIES}
    shared: dict[str, list[str]] = {s: [] for s in STRATEGIES}

    for marker, strats in marker_to_strategies.items():
        if len(strats) == 1:
            unique[strats[0]].append(marker)
        else:
            for s in strats:
                shared[s].append(marker)

    return {"unique": unique, "shared": shared}


# ---------------------------------------------------------------------------
# Per-strategy analysis
# ---------------------------------------------------------------------------

def analyze(records: list[dict]) -> dict:
    """
    For each (strategy, dialogue), compute self and other marker counts.

    Returns a nested dict:
        per_strategy[strategy] = {
            "self_counts": [...],   # one value per dialogue
            "other_counts": [...],
            "n_dialogues": int,
        }
    and a contingency table for chi-squared testing.
    """
    per_strategy: dict[str, dict] = {
        s: {"self_counts": [], "other_counts": []} for s in STRATEGIES
    }

    for rec in records:
        text = rec["tutor_text"]
        strategy = rec["strategy"]
        self_count = count_markers_for_strategy(text, strategy)
        other_count = sum(
            count_markers_for_strategy(text, s)
            for s in STRATEGIES
            if s != strategy
        )
        per_strategy[strategy]["self_counts"].append(self_count)
        per_strategy[strategy]["other_counts"].append(other_count)

    # Summarise
    summary: dict[str, dict] = {}
    for s in STRATEGIES:
        sc = np.array(per_strategy[s]["self_counts"], dtype=float)
        oc = np.array(per_strategy[s]["other_counts"], dtype=float)
        n = len(sc)
        total = sc + oc
        ratio = np.where(total > 0, sc / total, 0.0)
        summary[s] = {
            "n_dialogues": n,
            "mean_self": float(np.mean(sc)) if n > 0 else 0.0,
            "mean_other": float(np.mean(oc)) if n > 0 else 0.0,
            "mean_ratio": float(np.mean(ratio)) if n > 0 else 0.0,
            "self_counts": per_strategy[s]["self_counts"],
            "other_counts": per_strategy[s]["other_counts"],
        }
    return summary


def build_contingency_table(summary: dict) -> np.ndarray:
    """
    Build a (5 x 2) contingency table: rows = strategies,
    cols = [total_self_markers, total_other_markers].
    """
    table = []
    for s in STRATEGIES:
        total_self = int(sum(summary[s]["self_counts"]))
        total_other = int(sum(summary[s]["other_counts"]))
        table.append([total_self, total_other])
    return np.array(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading dialogues...")
    records = load_all_dialogues()
    print(f"  Loaded {len(records)} dialogue-scenarios across all models/cells.")

    summary = analyze(records)
    overlap = marker_overlap_report()

    table = build_contingency_table(summary)
    # chi2_contingency expects at least a 2x2 or larger table.
    chi2, p, dof, expected = chi2_contingency(table)

    # ------------------------------------------------------------------
    # Build JSON output
    # ------------------------------------------------------------------
    per_strategy_out = {}
    for s in STRATEGIES:
        per_strategy_out[s] = {
            "n_dialogues": summary[s]["n_dialogues"],
            "mean_self_markers": round(summary[s]["mean_self"], 4),
            "mean_other_markers": round(summary[s]["mean_other"], 4),
            "distinctiveness_ratio": round(summary[s]["mean_ratio"], 4),
            "unique_markers": overlap["unique"][s],
            "shared_markers": overlap["shared"][s],
        }

    json_output = {
        "n_total_dialogues": len(records),
        "per_strategy": per_strategy_out,
        "chi2_test": {
            "chi2": round(float(chi2), 4),
            "df": int(dof),
            "p_value": float(p),
            "significant_at_0.05": bool(p < 0.05),
        },
        "contingency_table": {
            "rows": STRATEGIES,
            "cols": ["total_self_markers", "total_other_markers"],
            "data": table.tolist(),
        },
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved JSON -> {OUTPUT_JSON}")

    # ------------------------------------------------------------------
    # Build text output
    # ------------------------------------------------------------------
    lines = []
    lines.append("Strategy Fidelity Analysis")
    lines.append("==========================")
    lines.append("")
    lines.append(f"Total dialogue-scenarios analyzed: {len(records)}")
    lines.append("")
    lines.append("Per-Strategy Marker Frequencies (mean per dialogue):")
    lines.append(
        f"{'Strategy':<22} {'Self_Markers':>12} {'Other_Markers':>14} "
        f"{'Ratio':>7} {'n_dialogues':>12}"
    )
    lines.append("-" * 72)
    for s in STRATEGIES:
        d = per_strategy_out[s]
        lines.append(
            f"{s:<22} {d['mean_self_markers']:>12.3f} "
            f"{d['mean_other_markers']:>14.3f} "
            f"{d['distinctiveness_ratio']:>7.3f} "
            f"{d['n_dialogues']:>12}"
        )

    lines.append("")
    lines.append("Marker overlap between strategies:")
    for s in STRATEGIES:
        uniq = ", ".join(overlap["unique"][s]) if overlap["unique"][s] else "(none)"
        shrd = ", ".join(overlap["shared"][s]) if overlap["shared"][s] else "(none)"
        lines.append(f"  {s}:")
        lines.append(f"    unique : {uniq}")
        lines.append(f"    shared : {shrd}")

    lines.append("")
    lines.append("Contingency table (total marker counts):")
    lines.append(
        f"  {'Strategy':<22} {'Self':>8} {'Other':>8}"
    )
    for i, s in enumerate(STRATEGIES):
        lines.append(
            f"  {s:<22} {table[i, 0]:>8} {table[i, 1]:>8}"
        )

    lines.append("")
    lines.append("Chi-squared test for strategy distinctiveness:")
    lines.append(
        f"  chi2={chi2:.4f}, df={dof}, p={p:.6f}"
    )
    lines.append("")
    if p < 0.05:
        conclusion = (
            "Conclusion: strategies ARE linguistically distinguishable "
            f"(chi2={chi2:.2f}, p={p:.4f} < 0.05)."
        )
    else:
        conclusion = (
            "Conclusion: strategies are NOT significantly distinguishable "
            f"(chi2={chi2:.2f}, p={p:.4f} >= 0.05)."
        )
    lines.append(conclusion)
    lines.append("")

    txt_content = "\n".join(lines)
    with open(OUTPUT_TXT, "w", encoding="utf-8") as f:
        f.write(txt_content)
    print(f"  Saved TXT  -> {OUTPUT_TXT}")

    # Print to stdout as well
    print()
    print(txt_content)


if __name__ == "__main__":
    main()
