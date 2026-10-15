# MisconceptDialogue: Evaluation Criteria Shape Measured Success in LLM Math Tutoring

> **AACL-IJCNLP 2026** | Official Repository

**Authors:** Woojin Lee<sup>1\*</sup>, Yoonkyu Woo<sup>1,2\*</sup>, Jinah Kim<sup>1,2</sup>, Hyeok-Min Gwon<sup>1</sup>, Jin-Xia Huang<sup>1,2†</sup>  
\* *Co-first authors* | † *Corresponding author* (`hgh@etri.re.kr`)

<sup>1</sup> Electronics and Telecommunications Research Institute (ETRI), Republic of Korea  
<sup>2</sup> Department of Artificial Intelligence, University of Science and Technology (UST), Republic of Korea

MisconceptDialogue is an automated evaluation framework for studying whether LLM tutors can
**detect** mathematical mistakes, **diagnose** the underlying misconception, and **remediate**
that misconception through multi-turn tutoring dialogue.

The paper evaluates five open-weight 7B–14B tutor models across detection, diagnosis,
remediation, and end-to-end pipeline ablations. A central result is that remediation success
depends strongly on the evaluation lens: strict answer matching, numeric extraction, and
LLM-as-judge evaluation operationalize different notions of success on the same dialogues.

## Scope of this repository

This is a **public** repository holding the runnable code and the lightweight scenario/config
assets. The published paper makes no artifact-availability claim and cites no code URL.

Deliberately not tracked here:

- downloaded dataset payloads under `datasets/` (fetch them with the script below)
- raw per-dialogue run directories under `results/` (regenerate by rerunning experiments);
  the final scoring artifacts behind the paper's numbers **are** included (see `results/README.md`)
- paper sources and figures
- caches, secrets, logs, and model weights

## Layout

```text
MisconceptDialogue/
├── data/                  # Scenario/config assets for Exp. 3/4
│   ├── exp3v2_scenario_bank.json     # 106 evaluation scenarios
│   ├── exp3v2_candidate_pool.json
│   ├── exp3v2_gold_bank.json
│   └── wrong_diagnosis_mapping.json  # same-type swaps for the wrong-diagnosis control
├── datasets/              # Dataset download location (payloads untracked)
│   └── README.md
├── results/               # Final scoring artifacts behind the paper (see results/README.md)
│   └── README.md
├── scripts/               # Reproduction and analysis scripts
├── src/                   # Experiment, model, data, and evaluation code
├── REPRODUCE.md           # Reproduction guide
├── LICENSE                # MIT (software and result artifacts)
├── DATA_LICENSE.md        # Third-party data terms for data/
├── CITATION.cff
├── requirements.txt
├── .gitignore
└── .env.example
```

## Setup

```bash
conda create -n mcd python=3.11 -y
conda activate mcd
pip install -r requirements.txt
cp .env.example .env
# Fill OPENAI_API_KEY in .env if running API-based student/judge experiments.
```

Only the blank `.env.example` template is tracked. Never commit a populated `.env`; it is
git-ignored.

Datasets are not bundled. Prepare them locally with:

```bash
python scripts/download_datasets.py
```

Some datasets require manual access or separate licensing; see `datasets/README.md` and
`REPRODUCE.md`.

## Quick checks

These require no datasets, GPUs, or API calls:

```bash
python -m compileall -q src scripts
python -m src.experiments.exp1_detection.run --help
python -m src.experiments.exp2_diagnosis.run --help
python -m src.experiments.exp3v2.run --help
python -m src.experiments.exp4v2.run --help
```

## Main experiment entry points

- Exp. 1 detection: `src/experiments/exp1_detection/run.py`
- Exp. 2 diagnosis: `src/experiments/exp2_diagnosis/run.py`
- Exp. 3 remediation: `src/experiments/exp3v2/run.py`
- Exp. 4 pipeline ablation: `src/experiments/exp4v2/run.py`
- Wrong-diagnosis control: `scripts/run_wrong_diagnosis_control.py`
- Diagnosis-emphasis variant: `scripts/run_diagnosis_emphasis_variant.py`

## Audit and scoring utilities

These read the result artifacts under `results/`. The paper's final scoring artifacts are
included, so the first three run as-is; the Tier-1 audit also needs the raw run directories from a
rerun (see `results/README.md`).

- Paper-number audit: `scripts/generate_paper_numbers.py`
- Statistical audit: `scripts/corrected_results_analysis.py`
- Equivalence testing: `scripts/exp4v2_tost_corrected.py`
- Tier-1 final-score audit: `scripts/recompute_tier1_corrected.py`
- Tier-2 matched-context rescoring: `scripts/rescore_tier2_corrected.py` (requires OpenAI API)

## Reproduction

See [`REPRODUCE.md`](REPRODUCE.md) for two tracks:

- **Track A:** verify the paper's numbers from the shipped result artifacts. No datasets, GPUs,
  or API key needed; it takes minutes on a CPU.
- **Track B:** if you want to, rerun every experiment from scratch, including local tutor inference
  and API-based simulated-student/judge evaluation, and verify the regenerated outputs the same way.

## License and citation

The MIT License in [`LICENSE`](LICENSE) governs the software and the result artifacts. The bundled
scenario/config assets under `data/` are derived from third-party datasets and remain subject to the
upstream terms recorded in [`DATA_LICENSE.md`](DATA_LICENSE.md) (notably the ShareAlike condition of
MathDial's CC BY-SA 4.0).

Citation metadata is in [`CITATION.cff`](CITATION.cff):

```bibtex
@inproceedings{lee2026misconceptdialogue,
  title={{MisconceptDialogue}: Evaluation Criteria Shape Measured Success in {LLM} Math Tutoring},
  author={Lee, Woojin and Woo, Yoonkyu and Kim, Jinah and Gwon, Hyeok-Min and Huang, Jin-Xia},
  booktitle={Proceedings of the 5th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 15th International Joint Conference on Natural Language Processing (AACL-IJCNLP 2026)},
  year={2026}
}
```
