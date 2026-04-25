#!/bin/bash
# Evaluate qwen3vl-8b-online results

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$PROJ_ROOT/results-qwen3vl-8b-online}"

cd "$PROJ_ROOT/baselines"

for bench in ap si sg md pa; do
    python ../evaluations/evaluate.py \
        --model_name qwen3vl-8b-online \
        --benchmark_name "$bench" \
        --pred_path "$OUTPUT_DIR/${bench}_qwen3vl-8b-online.jsonl" \
        --output_dir "$OUTPUT_DIR" \
        --num_tasks 8 &
done
wait
