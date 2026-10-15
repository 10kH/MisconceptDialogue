#!/usr/bin/env python3
"""Download all datasets for MisconceptDialogue experiments."""

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from src.utils.common import exit_on_help  # noqa: E402
exit_on_help(__doc__)

import os
import subprocess
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "datasets"


def download_mathdial():
    """Download MathDial from HuggingFace."""
    dest = BASE_DIR / "mathdial"
    if (dest / "done.marker").exists():
        print("[MathDial] Already downloaded, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("[MathDial] Downloading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("eth-nlped/mathdial")
        ds.save_to_disk(str(dest / "hf_cache"))
        (dest / "done.marker").touch()
        print(f"[MathDial] Done. Splits: {list(ds.keys())}, sizes: {[len(ds[s]) for s in ds.keys()]}")
    except Exception as e:
        print(f"[MathDial] Error: {e}")


def download_bridge():
    """Download Bridge dataset from GitHub."""
    dest = BASE_DIR / "bridge"
    if (dest / "done.marker").exists():
        print("[Bridge] Already downloaded, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("[Bridge] Cloning from GitHub...")
    repo_dir = dest / "repo"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1", "https://github.com/rosewang2008/bridge.git", str(repo_dir)],
            check=True, capture_output=True, text=True
        )
    (dest / "done.marker").touch()
    print("[Bridge] Done.")


def download_bea2025():
    """Download BEA 2025 Shared Task data from GitHub."""
    dest = BASE_DIR / "bea2025"
    if (dest / "done.marker").exists():
        print("[BEA2025] Already downloaded, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("[BEA2025] Cloning from GitHub...")
    repo_dir = dest / "repo"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/kaushal0494/UnifyingAITutorEvaluation.git", str(repo_dir)],
            check=True, capture_output=True, text=True
        )
    (dest / "done.marker").touch()
    print("[BEA2025] Done.")


def download_eedi():
    """Prepare the optional EEDI misconception data (manual download).

    EEDI is optional: only the Exp. 2 misconception taxonomy can use it, and it falls back
    to the Math Misconceptions Benchmark when EEDI is absent. The data is distributed only
    through Kaggle under the competition's terms, so it cannot be fetched automatically.
    """
    dest = BASE_DIR / "eedi"
    if (dest / "misconception_mapping.csv").exists():
        print("[EEDI] Found misconception_mapping.csv, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    note = (
        "EEDI is optional (used only by the Exp. 2 taxonomy; it falls back to the Math\n"
        "Misconceptions Benchmark when absent).\n\n"
        "To use it, accept the competition rules and download the data from Kaggle:\n"
        "  https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data\n"
        "then place train.csv, test.csv and misconception_mapping.csv in this directory\n"
        f"({dest}).\n"
    )
    (dest / "MANUAL_DOWNLOAD_NEEDED.txt").write_text(note)
    print("[EEDI] Optional, manual download only (Kaggle). Instructions written to "
          f"{dest / 'MANUAL_DOWNLOAD_NEEDED.txt'}")


def download_mrbench():
    """Download MRBench from GitHub."""
    dest = BASE_DIR / "mrbench"
    if (dest / "done.marker").exists():
        print("[MRBench] Already downloaded, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("[MRBench] Cloning from GitHub (UnifyingAITutorEvaluation)...")
    # MRBench is part of the same repo as BEA 2025
    bea_repo = BASE_DIR / "bea2025" / "repo"
    if bea_repo.exists():
        # Symlink to avoid double download
        repo_link = dest / "repo_link"
        if not repo_link.exists() and not repo_link.is_symlink():
            os.symlink(str(bea_repo), str(repo_link))
        (dest / "done.marker").touch()
        print("[MRBench] Linked from BEA2025 repo.")
    else:
        repo_dir = dest / "repo"
        if not repo_dir.exists():
            subprocess.run(
                ["git", "clone", "--depth", "1",
                 "https://github.com/kaushal0494/UnifyingAITutorEvaluation.git", str(repo_dir)],
                check=True, capture_output=True, text=True
            )
        (dest / "done.marker").touch()
        print("[MRBench] Done.")


def download_misconceptions_benchmark():
    """Download Math Misconceptions Benchmark (55 misconceptions)."""
    dest = BASE_DIR / "misconceptions_benchmark"
    if (dest / "done.marker").exists():
        print("[MisconceptionsBench] Already downloaded, skipping.")
        return
    dest.mkdir(parents=True, exist_ok=True)
    print("[MisconceptionsBench] Cloning from GitHub...")
    repo_dir = dest / "repo"
    if not repo_dir.exists():
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/nancyotero-projects/math-misconceptions.git", str(repo_dir)],
            check=True, capture_output=True, text=True
        )
    (dest / "done.marker").touch()
    print("[MisconceptionsBench] Done.")


if __name__ == "__main__":
    print("=" * 60)
    print("MisconceptDialogue Dataset Downloader")
    print("=" * 60)

    download_mathdial()
    download_bridge()
    download_bea2025()
    download_eedi()
    download_mrbench()
    download_misconceptions_benchmark()

    print("\n" + "=" * 60)
    print("Dataset download complete!")
    print("=" * 60)
