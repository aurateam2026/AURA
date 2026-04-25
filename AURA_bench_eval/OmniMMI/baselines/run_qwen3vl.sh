#!/bin/bash
# Run qwen3vl-8b-online inference on all benchmarks

PROJ_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$PROJ_ROOT/results-qwen3vl-8b-online}"
CKPT_PATH="aurateam/AURA"

source "$PROJ_ROOT/../.venv/bin/activate"

export PYTHONPATH="$PROJ_ROOT:$PYTHONPATH"
cd "$PROJ_ROOT/baselines"

for task_info in \
    "ap:action_prediction.json" \
    "si:speaker_identification.json" \
    "md:multiturn_dependency_reasoning.json" \
    "sg:dynamic_state_grounding.json" \
    "pa:proactive_alerting.json"; do

    bench="${task_info%%:*}"
    qfile="${task_info#*:}"

    python ../evaluations/inference.py \
        --model_name qwen3vl-8b-online \
        --benchmark_name "$bench" \
        --cache_dir ./cache_dir \
        --video_dir /scratch/dyvm6xra/dyvm6xrauser36/stream_benchs_datasets/OmniMMI \
        --questions_file "/scratch/dyvm6xra/dyvm6xrauser36/stream_benchs_datasets/OmniMMI/$qfile" \
        --output_dir "$OUTPUT_DIR" \
        --ckpt_path "$CKPT_PATH" \
        --num_chunks 1 \
        --chunk_idx 0

    cp "$OUTPUT_DIR/${bench}_qwen3vl-8b-online_0.json" \
       "$OUTPUT_DIR/${bench}_qwen3vl-8b-online.jsonl"
done
