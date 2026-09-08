# Reproducing Paper Results

This guide explains how to inspect the MisconceptDialogue code and reproduce the experiments reported in the paper.

## What this repository tracks

This repository holds the code, scripts, documentation, and the lightweight scenario/config JSON under `data/`. It intentionally does not track:

- downloaded dataset payloads under `datasets/`
- generated experiment outputs under `results/`
- paper sources, figures, and review/human-evaluation materials
- internal notes, archives, caches, secrets, logs, and model weights

Therefore, commands that verify paper numbers from precomputed outputs require result artifacts you generate yourself. Full reruns require GPUs and API access.

## Prerequisites

```bash
conda create -n mcd python=3.11 -y
conda activate mcd
pip install -r requirements.txt
cp .env.example .env  # fill in OPENAI_API_KEY if running API-based student/judge experiments
```

The release package includes only a blank `.env.example`. Keep real API keys in
a local `.env` file; `.env` is git-ignored and is not included in `MCD.zip`.

## Smoke checks for the software package

These checks verify that the package is syntactically valid and the main entry points are visible. They do not run models or APIs.

```bash
python -m compileall -q src scripts
python -m src.experiments.exp1_detection.run --help
python -m src.experiments.exp2_diagnosis.run --help
python -m src.experiments.exp3v2.run --help
python -m src.experiments.exp4v2.run --help
```

## Prepare datasets

Datasets are not bundled in `MCD.zip`. Prepare them locally with:

```bash
python scripts/download_datasets.py
```

Some sources may require manual access or separate credentials. See `datasets/README.md`.

## Track A: Verify paper numbers from result artifacts

This track requires precomputed outputs under `results/`. The software ZIP does not bundle those outputs by default. If you have generated or separately obtained the result artifacts, place them under the paths documented in `results/README.md`, then run:

```bash
# 1. Verify Table 3 / Table 4 numbers
python scripts/generate_paper_numbers.py

# 2. Verify GLMM / ICC statistics
python scripts/corrected_results_analysis.py

# 3. Verify Tier-1 final-score audit
python scripts/recompute_tier1_corrected.py
```

Expected outputs with the paper-authoritative result artifacts:

- `generate_paper_numbers.py`: Table 3 Mean = 42.1 / 45.8 / 81.3; Table 4 condition range 76.6--81.3%
- `corrected_results_analysis.py`: GLMM p=0.263, ICC(scenario)=0.206
- `recompute_tier1_corrected.py`: Tier-1 final RSR = 42.1%, MATCH group invariance PASS

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
- The paper-authoritative `tier1_corrected/` and `tier2_corrected/` result directories are not tracked here; regenerate them by rerunning the scoring steps above.
- Results may vary slightly due to API model updates and sampling stochasticity.
