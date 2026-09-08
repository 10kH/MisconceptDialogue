#!/bin/bash
# Run exp3v2 for all 5 models on GPUs 4, 5, 6
# Round 1: 3 models in parallel
# Round 2: 2 models in parallel

set -e
source .env && export OPENAI_API_KEY
cd "$(dirname "$0")/.."

BANK="data/exp3v2_scenario_bank.json"
STUDENT="gpt-5-mini"
JUDGE="gpt-5-mini"
SEED=42

run_model() {
    local GPU=$1
    local MODEL=$2
    local MODEL_SHORT=$3
    local TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    local OUTDIR="results/exp3v2/${TIMESTAMP}_${MODEL_SHORT}"

    echo "[$(date)] Starting ${MODEL_SHORT} on GPU ${GPU} -> ${OUTDIR}"
    CUDA_VISIBLE_DEVICES=$GPU conda run -n X python -u -m src.experiments.exp3v2.run \
        --model "$MODEL" \
        --student_model "$STUDENT" \
        --judge_model "$JUDGE" \
        --bank "$BANK" \
        --seed $SEED \
        --output_dir "$OUTDIR" \
        --max_turns 5 \
        2>&1 | tee "results/exp3v2/log_${MODEL_SHORT}.txt"
    echo "[$(date)] Finished ${MODEL_SHORT}"
}

mkdir -p results/exp3v2

echo "=========================================="
echo "Exp3v2: Round 1 (3 models in parallel)"
echo "=========================================="

run_model 4 "qwen-2.5-7b" "qwen-2.5-7b" &
PID1=$!
run_model 5 "mistral-7b" "mistral-7b" &
PID2=$!
run_model 6 "llama-3.1-8b" "llama-3.1-8b" &
PID3=$!

echo "Waiting for Round 1: PIDs $PID1 $PID2 $PID3"
wait $PID1 $PID2 $PID3
echo "[$(date)] Round 1 complete!"

echo ""
echo "=========================================="
echo "Exp3v2: Round 2 (2 models in parallel)"
echo "=========================================="

run_model 4 "phi-4" "phi-4" &
PID4=$!
run_model 5 "qwen3.5-9b" "qwen3.5-9b" &
PID5=$!

echo "Waiting for Round 2: PIDs $PID4 $PID5"
wait $PID4 $PID5
echo "[$(date)] Round 2 complete!"

echo ""
echo "=========================================="
echo "All 5 models complete!"
echo "=========================================="
ls -la results/exp3v2/
