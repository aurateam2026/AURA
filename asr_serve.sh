#!/bin/bash
# ASR 服务启动脚本
# 与 TTS 共享同一张 GPU，需降低 gpu-memory-utilization 为 TTS 留出显存

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
python -u Qwen3_asr_serve.py \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization 0.3 \
    --no-forced-aligner
