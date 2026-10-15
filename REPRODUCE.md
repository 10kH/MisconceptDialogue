# Reproducing Paper Results

This guide explains how to inspect the MisconceptDialogue code and reproduce the experiments reported in the paper.

## What this repository tracks

This repository holds the code, scripts, documentation, the scenario/config JSON under `data/`, and the final scoring artifacts behind the paper's numbers under `results/` (listed in `results/README.md`). It does not track:

- downloaded dataset payloads under `datasets/`
- raw per-dialogue run directories and the outputs of other experiments under `results/`
- paper sources and figures
- caches, secrets, logs, and model weights

So the paper's numbers can be verified from the shipped artifacts in minutes on a CPU (Track A), and every experiment can also be rerun from scratch (Track B), which requires GPUs and API access.

## Prerequisites

```bash
conda create -n mcd python=3.11 -y
conda activate mcd
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY if running API-based student/judge experiments
```

Only a blank `.env.example` is tracked. Keep real API keys in a local `.env`
file; `.env` is git-ignored.

## Smoke checks

These checks verify that the package is syntactically valid and the main entry points are visible. They do not run models or APIs.

```bash
python -m compileall -q src scripts
python -m src.experiments.exp1_detection.run --help
python -m src.experiments.exp2_diagnosis.run --help
python -m src.experiments.exp3v2.run --help
python -m src.experiments.exp4v2.run --help
```

## Prepare datasets

Dataset payloads are not tracked. Prepare them locally with:

```bash
python scripts/download_datasets.py
```

Some sources may require manual access or separate credentials. See `datasets/README.md`.

## Track A: Verify paper numbers from the shipped artifacts

No datasets, GPUs, or API key are needed: the final scoring artifacts are in `results/`. Run:

```bash
# 1. Table 3 / Table 4 numbers
python scripts/generate_paper_numbers.py

# 2. GLMM / ICC statistics
python scripts/corrected_results_analysis.py

# 3. Equivalence testing (TOST)
python scripts/exp4v2_tost_corrected.py
```

Expected output:

- `generate_paper_numbers.py`: Table 3 Mean = 42.1 / 45.8 / 81.3 (strict / numeric / judge), a 39.2 pp gap; Table 4 condition range 76.6--81.3%
- `corrected_results_analysis.py`: GLMM condition effect χ² = 6.47, p = 0.263; ICC(scenario) = 0.206
- `exp4v2_tost_corrected.py`: at δ = 10 pp all 15 condition pairs are equivalent; at δ = 5 pp only 3 are

The wrong-diagnosis control, the diagnosis-emphasis variant, and the human calibration are summarized in `results/exp4v2/condition_C_wrong/summary.json`, `results/exp4v2/diagnosis_emphasis_summary.json`, and `results/human_eval/{clean,borderline}/aggregated.json`.

`scripts/recompute_tier1_corrected.py` recomputes Tier-1 from the raw per-dialogue dumps, which are not shipped; it runs after a Track B rerun and should report Tier-1 RSR = 42.1% with MATCH group invariance PASS.

## Track B: Full rerun (requires GPUs + API, several hours)

### Exp. 1: Detection

```bash
# Example: one model × one dataset × default track/condition
python src/experiments/exp1_detection/run.py --model qwen-2.5-7b --dataset mathdial --max_samples 10

# Full run: repeat for all model × dataset × context combinations
python src/experiments/exp1_detection/run.py --model qwen-2.5-7b --dataset mathdial --all_conditions
```

### Exp. 2: Diagnosis

```bash
# Example smoke-sized run
python src/experiments/exp2_diagnosis/run.py --model qwen-2.5-7b --condition zero_shot --max_samples 10

# Full run: repeat for all model × condition combinations
python src/experiments/exp2_diagnosis/run.py --model qwen-2.5-7b --all_conditions
```

### Exp. 3: Remediation (v2)

```bash
# 5 models × 5 strategies × 3 types × 30 scenarios = 2,250 dialogues
python src/experiments/exp3v2/run.py \
  --model qwen-2.5-7b \
  --bank data/exp3v2_scenario_bank.json

# Repeat for all 5 models.
```

### Exp. 4: Pipeline Ablation (v2)

```bash
# 5 models × 6 conditions × 106 scenarios = 3,180 dialogues
python src/experiments/exp4v2/run.py \
  --model qwen-2.5-7b \
  --bank data/exp3v2_scenario_bank.json

# Repeat for all 5 models.
```

### Post-experiment evaluation

```bash
# Tier-1 final-score audit (local, no API)
python scripts/recompute_tier1_corrected.py

# Tier-2 matched-context rescoring (requires OpenAI API)
python scripts/rescore_tier2_corrected.py          # Exp3v2
python scripts/rescore_tier2_corrected.py --exp4    # Exp4v2

# Statistical analysis
python scripts/corrected_results_analysis.py

# Generate paper numbers for verification
python scripts/generate_paper_numbers.py
```

## Models

| Model | HuggingFace ID | Size |
|-------|---------------|------|
| Qwen-2.5-7B | Qwen/Qwen2.5-7B-Instruct | 7B |
| Mistral-7B | mistralai/Mistral-7B-Instruct-v0.3 | 7B |
| Llama-3.1-8B | meta-llama/Llama-3.1-8B-Instruct | 8B |
| Phi-4 | microsoft/phi-4 | 14B |
| Qwen3.5-9B | Qwen/Qwen3.5-9B | 9B |

The local models were run in bfloat16 on H100 GPUs. GPT-5-mini was used as the simulated student and default LLM judge through the OpenAI API.

## Scenario bank

The 106-scenario bank is tracked at `data/exp3v2_scenario_bank.json`, covering factual, procedural, and conceptual misconception types. It is tracked because Exp. 3/4 and the wrong-diagnosis control expect these files.

## Notes

- Tier-2 rescoring uses GPT-5-mini with `max_completion_tokens=8192`.
- The paper-authoritative `tier1_corrected/` and `tier2_corrected/` artifacts are shipped under `results/`; rerunning the scoring steps above regenerates them in place.
- Results may vary slightly due to API model updates and sampling stochasticity.
