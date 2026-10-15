"""Common utilities for MisconceptDialogue experiments."""

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = PROJECT_ROOT / "datasets"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
CONFIGS_DIR = PROJECT_ROOT / "src" / "config"
LOGS_DIR = PROJECT_ROOT / "logs"


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def load_config(config_path: str | Path) -> dict:
    """Load YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def save_json(data: Any, path: str | Path):
    """Save data as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    """Load JSON file."""
    with open(path) as f:
        return json.load(f)


def save_jsonl(data: list[dict], path: str | Path):
    """Save data as JSONL."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_jsonl(path: str | Path) -> list[dict]:
    """Load JSONL file."""
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_timestamp() -> str:
    """Get current timestamp string."""
    return time.strftime("%Y%m%d_%H%M%S")


def exit_on_help(doc: str | None) -> None:
    """Print a script's module docstring and exit if `-h`/`--help` is given.

    For scripts without an argparse interface: call this before any API client is
    created or any result file is read, so `--help` works with no API key and no
    generated results.
    """
    import sys
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print((doc or "").strip() or "No help available.")
        sys.exit(0)


# Exp. 3 tutor models, keyed by the short name each run directory ends with
# (`results/exp3v2/<YYYYMMDD_HHMMSS>_<short>/`, see src/experiments/exp3v2/run.py).
EXP3V2_MODELS = {
    "qwen-2.5-7b": "Qwen-2.5-7B",
    "mistral-7b": "Mistral-7B",
    "llama-3.1-8b": "Llama-3.1-8B",
    "phi-4": "Phi-4",
    "qwen3.5-9b": "Qwen3.5-9B",
}


def latest_run_dirs(experiment: str = "exp3v2", models=None) -> dict[str, str]:
    """Return {model_short: run_dir_name} for the newest run directory of each model.

    Run directories are named `<YYYYMMDD_HHMMSS>_<model_short>` by the experiment
    runners, so the lexicographically largest match is the most recent run. Models
    with no run directory yet are omitted. This replaces hard-coded timestamped
    directory names, so analyses work on any machine's own reruns.
    """
    base = RESULTS_DIR / experiment
    out: dict[str, str] = {}
    for short in (models or EXP3V2_MODELS):
        runs = sorted(p.name for p in base.glob(f"2*_{short}") if p.is_dir())
        if runs:
            out[short] = runs[-1]
    return out


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists and return Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
