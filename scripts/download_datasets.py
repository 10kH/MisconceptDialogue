#!/usr/bin/env python3
"""Download all datasets for MisconceptDialogue experiments."""

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
    print("[MathDial] Downloading from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset("eth-nlped/mathdial", trust_remote_code=True)
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
    """Download EEDI misconception data."""
    dest = BASE_DIR / "eedi"
    if (dest / "done.marker").exists():
        print("[EEDI] Already downloaded, skipping.")
        return
    print("[EEDI] Note: Full EEDI Kaggle dataset requires Kaggle credentials.")
    print("[EEDI] Downloading publicly available misconception taxonomy...")
    # Download the NeurIPS 2020 EEDI dataset (public)
    try:
        from datasets import load_dataset
        ds = load_dataset("marksverdhei/eedi-mining-misconceptions-in-mathematics",
                          trust_remote_code=True)
        ds.save_to_disk(str(dest / "hf_cache"))
        (dest / "done.marker").touch()
        print(f"[EEDI] Done. Splits: {list(ds.keys())}")
    except Exception as e:
        print(f"[EEDI] HuggingFace download failed: {e}")
        print("[EEDI] Trying alternative: EEDI NeurIPS 2020 public data...")
        try:
            repo_dir = dest / "neurips2020"
            if not repo_dir.exists():
                subprocess.run(
                    ["git", "clone", "--depth", "1",
                     "https://github.com/haradai/eedi-dataset.git", str(repo_dir)],
                    check=True, capture_output=True, text=True
                )
            (dest / "done.marker").touch()
            print("[EEDI] NeurIPS 2020 data downloaded.")
        except Exception as e2:
            print(f"[EEDI] Alternative also failed: {e2}")
            print("[EEDI] Will need manual download. Creating placeholder.")
            (dest / "MANUAL_DOWNLOAD_NEEDED.txt").write_text(
                "Download EEDI data from:\n"
                "- Kaggle: https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics\n"
                "- NeurIPS 2020: https://arxiv.org/abs/2007.12061\n"
            )


def download_mrbench():
    """Download MRBench from GitHub."""
    dest = BASE_DIR / "mrbench"
    if (dest / "done.marker").exists():
        print("[MRBench] Already downloaded, skipping.")
        return
    print("[MRBench] Cloning from GitHub (UnifyingAITutorEvaluation)...")
    # MRBench is part of the same repo as BEA 2025
    bea_repo = BASE_DIR / "bea2025" / "repo"
    if bea_repo.exists():
        # Symlink to avoid double download
        os.symlink(str(bea_repo), str(dest / "repo_link"))
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
