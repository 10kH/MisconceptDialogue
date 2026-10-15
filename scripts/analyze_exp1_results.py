"""Quick analysis of Exp1 Detection results across all models and datasets.

Usage:
    python scripts/analyze_exp1_results.py
"""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)
import json
from pathlib import Path

RESULTS_DIR = Path("results/exp1")
CONDITIONS = ["single_turn", "multi_turn_2", "multi_turn_full"]

def main():
    # Find all result files
    files = sorted(RESULTS_DIR.glob("*.json"))

    # Group by model and dataset
    results = {}
    for f in files:
        data = json.loads(f.read_text())
        model = data.get("model", "unknown")
        dataset = f.stem.split("_mathdial_")[0].split("_bea2025_")[0].split("_bridge_")[0] if "_mathdial_" in f.stem or "_bea2025_" in f.stem or "_bridge_" in f.stem else "unknown"

        # Determine dataset
        for ds in ["mathdial", "bea2025", "bridge"]:
            if f"_{ds}_" in f.stem:
                dataset_name = ds
                break
        else:
            dataset_name = "unknown"

        # Determine condition
        for cond in CONDITIONS:
            if f"_{cond}_" in f.stem:
                condition = cond
                break
        else:
            condition = "unknown"

        # Extract metrics
        metrics = data.get("metrics", {})
        mi = metrics.get("mistake_identification", metrics)
        f1 = mi.get("macro_f1", -1)
        acc = mi.get("accuracy", -1)

        key = (model, dataset_name)
        if key not in results:
            results[key] = {}
        results[key][condition] = {"f1": f1, "acc": acc}

    # Print table
    print(f"{'Model':<25} {'Dataset':<10} {'Single F1':>10} {'Multi2 F1':>10} {'Full F1':>10} {'Single Acc':>10} {'Multi2 Acc':>10} {'Full Acc':>10}")
    print("-" * 105)

    for (model, dataset), conds in sorted(results.items()):
        row = f"{model:<25} {dataset:<10}"
        for metric_type in ["f1", "acc"]:
            for cond in CONDITIONS:
                val = conds.get(cond, {}).get(metric_type, -1)
                if val >= 0:
                    row += f" {val:>10.3f}"
                else:
                    row += f" {'---':>10}"
        print(row)


if __name__ == "__main__":
    main()
