# Results Directory

This directory is the expected location for experiment outputs produced by the MisconceptDialogue scripts.

## Tracking policy

Only this README is tracked; **generated result outputs are not**. This keeps the repository small and avoids mixing runnable code with bulky experiment artifacts.

Place generated result artifacts under the paths below so the audit scripts can find them. To regenerate them, follow `REPRODUCE.md`.

## Authoritative result layout used in the paper

| Directory | Experiment | Notes |
|-----------|-----------|-------|
| `exp1/` | Detection (Exp. 1) | 5 models × 3 datasets × 3 conditions |
| `exp2/` | Diagnosis (Exp. 2) | 5 models × 5 conditions (ZS, FS, CoT, RAG, GRR) |
| `exp3v2/` | Remediation (Exp. 3) | 5 models × 5 strategies × 3 types |
| `exp3v2/tier1_corrected/` | Final strict-match scoring | Paper-authoritative strict-match RSR values |
| `exp3v2/tier2_corrected/` | Final LLM-judge scoring | Paper-authoritative LLM-judge RSR values |
| `exp4v2/` | Pipeline ablation (Exp. 4) | 5 models × 6 main conditions plus wrong-diagnosis control |
| `exp4v2/tier1_corrected/` | Final strict-match scoring | Paper-authoritative strict-match RSR values |
| `exp4v2/tier2_corrected/` | Final LLM-judge scoring | Paper-authoritative LLM-judge RSR values |
| `exp4v2/condition_C_wrong/` | Wrong-diagnosis control | Same-type swapped diagnosis, detection held fixed |
| `exp4v2/condition_C2_emph/` | Diagnosis-emphasis variant (gold arm) | Emphasized diagnosis presentation, 530 dialogues |
| `exp4v2/condition_C_wrong_emph/` | Diagnosis-emphasis variant (wrong arm) | Emphasized presentation with swapped diagnosis, 530 dialogues |
| `judge_diversification/` | Judge diversification | GPT-5.4 re-evaluation with matched verification gold |
| `no_tutoring_baseline/` | Persona decomposition | Verification-only baseline |
| `persona_only_baseline/` | Persona decomposition | Misconception persona without tutoring |
| `cross_family_v2/` | Cross-family validation | Full tutor--evaluator grid |

## Important

Raw dialogue outputs in `exp3v2/2026*/` and `exp4v2/2026*/` are not tracked here. The paper-authoritative numbers come from final scoring artifacts in `tier1_corrected/` and `tier2_corrected/`, which score each verification response against the verification problem's gold answer.

The raw dumps are not merely supplementary: `scripts/recompute_tier1_corrected.py` recomputes Tier-1 from them and exits with a clear error if they are absent. `generate_paper_numbers.py` and `corrected_results_analysis.py` need only the `*_corrected/` summaries.
