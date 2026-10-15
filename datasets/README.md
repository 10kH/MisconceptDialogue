# Datasets

Dataset payloads are not committed to this repository. Prepare them locally with:

```bash
python scripts/download_datasets.py
```

| Directory | Source | How it is obtained | Used by |
|-----------|--------|--------------------|---------|
| `mathdial/` | MathDial (Macina et al., 2023) | Hugging Face `eth-nlped/mathdial`, automatic | Exp. 1 |
| `bridge/` | Bridge (Wang et al., 2024) | GitHub, automatic | Exp. 1 |
| `bea2025/` | BEA 2025 Shared Task | GitHub, automatic | Exp. 1 |
| `misconceptions_benchmark/` | Math Misconceptions Benchmark (Otero et al., 2025) | GitHub, automatic | Exp. 2, scenario bank |
| `mrbench/` | MRBench (part of the BEA 2025 repository) | GitHub, automatic (linked to `bea2025/` when present) | optional |
| `eedi/` | EEDI Kaggle 2024 competition | **manual**: accept the rules and download `train.csv`, `test.csv`, `misconception_mapping.csv` from [Kaggle](https://www.kaggle.com/competitions/eedi-mining-misconceptions-in-mathematics/data) | optional (Exp. 2 taxonomy; falls back to the Misconceptions Benchmark) |

Each dataset keeps its own license; see [`DATA_LICENSE.md`](../DATA_LICENSE.md).
