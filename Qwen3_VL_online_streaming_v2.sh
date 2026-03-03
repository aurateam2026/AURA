#!/bin/bash

# Qwen3 Omni 启动示例:
MODEL_PATH=/home/dyvm6xra/dyvm6xrauser36/Projects/streaming_video_understanding/qwen3vl-8b_20260225_02/
echo "MODEL_PATH: $MODEL_PATH"

CUDA_VISIBLE_DEVICES=1,2,3 numactl --cpunodebind=0 --membind=0 python -u Qwen3_VL_online_streaming_v2.py \
    --listen-port 12345 \
    --model $MODEL_PATH \
    --tensor-parallel-size 1 \
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
    --enable-tts \
    --tts-gpu 3 \
    --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --tts-language Chinese \
    --tts-ref-audio test_query.mp3 \
    --tts-ref-text "仔细观察当前你看到的画面，并且结合之前你看到的画面，仔细描述你看到了什么" \
    --tts-output-dir tts_results \
    --enable-pruning \
    --max-rounds 120 \
    --num-rounds-keep 20
