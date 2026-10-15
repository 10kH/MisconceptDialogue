# Data licenses

The MIT License in [`LICENSE`](LICENSE) covers the software in this repository (`src/`, `scripts/`)
and its documentation. It does **not** relicense third-party data.

## Third-party and derived assets

Bundled scenario/config assets under `data/` are derived in part from third-party datasets that
carry their own terms:

| Dataset | Reference | Terms |
|---|---|---|
| MathDial | Macina et al., 2023 | CC BY-SA 4.0 |
| Bridge | Wang et al., 2024 | CC BY 4.0 |
| BEA 2025 Shared Task data | Kochmar et al., 2025 | see the original release |
| Math Misconceptions Benchmark | Otero et al., 2025 | see the original release |
| EEDI (optional, not bundled) | Kaggle 2024 competition | Kaggle competition rules |

Portions of `data/exp3v2_scenario_bank.json` are adapted from these sources or generated as
problem-context variants. Users redistributing or modifying the bundled assets must comply with the
terms of the corresponding upstream releases, including the ShareAlike condition of CC BY-SA 4.0
where it applies.

Dataset payloads themselves are not redistributed here; see [`datasets/README.md`](datasets/README.md)
for download instructions.

## Result artifacts

The result artifacts under `results/` (scores, judge outputs, and aggregate statistics) are released
with the software under the MIT License. They contain no student or annotator personal data;
human annotators are identified only by anonymous codes (annotator A/B/C).
