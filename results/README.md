# Results

This directory ships the **final scoring artifacts behind the paper's numbers**, so the reported
results can be checked without rerunning any experiment, GPU job, or API call. Rerunning the
experiments (`REPRODUCE.md`, Track B) writes new outputs to the same layout, and the audit scripts
then verify those instead.

## Included artifacts

| Path | Backs | Contents |
|------|-------|----------|
| `exp3v2/tier1_corrected/tier1_recomputed.json` | Table 3 (strict match) | Per-dialogue strict-match scores, 2,250 Exp. 3 dialogues |
| `exp3v2/tier2_corrected/tier2_recomputed.json` | Table 3 (LLM judge) | Per-dialogue LLM-judge scores, 2,250 Exp. 3 dialogues |
| `exp3v2/tier2_corrected/checkpoint.json` | Table 3 | Per-dialogue judge verdicts with the judge's rationale |
| `exp3v2/three_tier_analysis.json` | Table 3 (numeric extraction) | Three-tier (strict / numeric / judge) analysis |
| `exp4v2/tier1_corrected/tier1_recomputed.json` | Table 4 (strict match) | Per-dialogue strict-match scores, 3,180 Exp. 4 dialogues |
| `exp4v2/tier2_corrected/tier2_recomputed.json` | Table 4, GLMM, TOST | Per-dialogue LLM-judge scores, 3,180 Exp. 4 dialogues |
| `exp4v2/tier2_corrected/checkpoint.json` | Table 4 | Per-dialogue judge verdicts with the judge's rationale |
| `exp4v2/corrected_analysis_summary.json` | GLMM, ICC | Condition effect and variance decomposition |
| `exp4v2/tost_analysis.txt` | Equivalence testing (Appendix K) | TOST at δ = 5 pp and δ = 10 pp |
| `exp4v2/condition_C_wrong/summary.json` | Wrong-diagnosis control | Wrong vs. correct diagnosis RSR |
| `exp4v2/diagnosis_emphasis_summary.json` | Diagnosis-emphasis variant | Wrong vs. correct diagnosis under emphasis |
| `human_eval/clean/aggregated.json` | Human calibration (clean set) | Consensus and agreement, 100 dialogues |
| `human_eval/borderline/aggregated.json` | Human calibration (borderline set) | Consensus and agreement, 100 dialogues |

`exp3v2/tier1_corrected/README_FIELD_WARNING.md` documents the field semantics of the Tier-1 files.
Human annotators appear only as anonymous codes (annotator A/B/C).

## Not included

Raw per-dialogue run directories (`exp3v2/<timestamp>_<model>/`, `exp4v2/<timestamp>_<model>/`) and
the outputs of the other experiments are not shipped; rerun the experiments to produce them. They
are needed only by `scripts/recompute_tier1_corrected.py`, which recomputes Tier-1 from the raw
dumps and exits with a clear error if they are absent.

## Layout written by the experiments

| Directory | Experiment | Notes |
|-----------|-----------|-------|
| `exp1/` | Detection (Exp. 1) | 5 models × 3 datasets × 3 conditions |
| `exp2/` | Diagnosis (Exp. 2) | 5 models × 5 conditions (ZS, FS, CoT, RAG, GRR) |
| `exp3v2/` | Remediation (Exp. 3) | 5 models × 5 strategies × 3 types |
| `exp4v2/` | Pipeline ablation (Exp. 4) | 5 models × 6 main conditions plus wrong-diagnosis control |
| `exp4v2/condition_C2_emph/` | Diagnosis-emphasis variant (gold arm) | Emphasized diagnosis presentation, 530 dialogues |
| `exp4v2/condition_C_wrong_emph/` | Diagnosis-emphasis variant (wrong arm) | Emphasized presentation with swapped diagnosis, 530 dialogues |
| `judge_diversification/` | Judge diversification | GPT-5.4 re-evaluation with matched verification gold |
| `no_tutoring_baseline/` | Persona decomposition | Verification-only baseline |
| `persona_only_baseline/` | Persona decomposition | Misconception persona without tutoring |
| `cross_family_v2/` | Cross-family validation | Full tutor-evaluator grid |
