#!/bin/bash
# ASR 服务启动脚本
# gpu-memory-utilization 由上游 start_all.sh 根据是否与 TTS 共卡自动设定

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
GPU_UTIL=${ASR_GPU_UTIL:-0.3}

python -u Qwen3_asr_serve.py \
    --host 0.0.0.0 \
    --port 8001 \
    --gpu-memory-utilization $GPU_UTIL \
    --no-forced-aligner
