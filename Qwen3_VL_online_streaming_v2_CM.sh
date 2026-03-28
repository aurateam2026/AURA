#!/bin/bash

CASE_NAME=qwen3vl-8b_20260322_02

# Qwen3 Omni 启动示例:
MODEL_PATH=/home/dyvm6xra/dyvm6xrauser36/Projects/streaming_video_understanding/${CASE_NAME}/
echo "MODEL_PATH: $MODEL_PATH"

# vLLM 主推理 (TTS 已拆分为独立服务 tts_service.py)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TP=${TP_SIZE:-1}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, tensor-parallel-size=$TP"

python -u Qwen3_VL_online_streaming_v2_ContextManaged.py \
    --listen-port 12345 \
    --model $MODEL_PATH \
    --tensor-parallel-size $TP \
    --max-model-len 262144 \
    --max-seq-len 262144 \
    --gpu-memory-utilization 0.9 \
    --asr-url http://localhost:8001/asr \
    --kv-offloading-size 10 \
    --disable-hybrid-kv-cache-manager \
    --block-size 16 \
    --prefix-caching-hash-algo xxhash \
    --mm-encoder-attn-backend FLASH_ATTN \
    --mm-encoder-tp-mode data \
    --max-num-batched-tokens 15360 \
    --temperature 0.5 \
    --max-tokens 128 \
    --enable-tts \
    --tts-service-url http://localhost:8002 \
    --tts-output-dir tts_results \
    --cross-turn-penalty 0.5 \
    --cross-turn-lookback 10 \
    --cross-turn-ngram-sizes \
    --enable-pruning \
    --max-rounds 45 \
    --num-rounds-keep 30 \
    --max-context-qas 10 \
    --debug-context-file debug_context.jsonl \
    --debug-context
