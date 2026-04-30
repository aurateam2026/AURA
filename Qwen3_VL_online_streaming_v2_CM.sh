#!/bin/bash

MODEL_PATH="${AURA_MODEL_PATH:?Please set AURA_MODEL_PATH to your Qwen3-VL/Omni model directory (see .env.example)}"
echo "MODEL_PATH: $MODEL_PATH"

# vLLM 主推理 (TTS 已拆分为独立服务 tts_service.py)
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
TP=${TP_SIZE:-1}
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, tensor-parallel-size=$TP"

python -u Qwen3_VL_online_streaming_v2_ContextManaged.py \
    --listen-port "${AURA_INFER_PORT:-12345}" \
    --model $MODEL_PATH \
    --tensor-parallel-size $TP \
    --max-model-len 262144 \
    --max-seq-len 262144 \
    --gpu-memory-utilization 0.9 \
    --asr-url "http://localhost:${AURA_ASR_PORT:-8001}/asr" \
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
    --tts-service-url "http://localhost:${AURA_TTS_PORT:-8002}" \
    --tts-output-dir tts_results \
    --cross-turn-penalty 1 \
    --cross-turn-lookback 10 \
    --cross-turn-ngram-sizes \
    --enable-pruning \
    --max-rounds 45 \
    --num-rounds-keep 30 \
    --max-context-qas 10 \
    --debug-context-file debug_context.jsonl \
    --debug-context
