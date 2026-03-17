#!/bin/bash

# source /home/dyvm6xra/dyvm6xrauser36/Projects/vllm-omni/.venv_streaming_new/bin/activate

## 比较ovobench刷榜数据的影响 
# CASE_NAME=qwen3vl-8b_20260225_02  ## no ovobench
# CASE_NAME=qwen3vl-8b_20260310_01

## ovobench切分chunk
# CASE_NAME=qwen3vl-8b_20260310_01  ## no chunkwise 
# CASE_NAME=qwen3vl-8b_20260311_01  ## chunkwise

# ## 比较质检后的1qna数据的影响
# CASE_NAME=qwen3vl-8b_20260225_02  ## no 1qna v2，baseline
# CASE_NAME=qwen3vl-8b_20260311_02  ## 使用了1qna v2，抽取了35k条数据

# ### 比较4.5分，1w+条数据
# CASE_NAME=qwen3vl-8b_20260225_02  ## no 1qna v2，baseline
# CASE_NAME=qwen3vl-8b_20260311_02  ## 使用了1qna v2，抽取了35k条数据
# CASE_NAME=qwen3vl-8b_20260311_03  ## 4.5分，1w+条数据

# ## 比较全部的 刷榜 数据
# CASE_NAME=qwen3vl-8b_20260225_02   ## baseline
# CASE_NAME=qwen3vl-8b_20260314_01   ## +刷榜数据，ovobench+streamingbench+omnimmi

## 比较全部的 非刷榜 数据 
# CASE_NAME=
# CASE_NAME=qwen3vl-8b_20260314_02

## +25%盘古sft数据
# CASE_NAME=qwen3vl-8b_20260316_01

## 
CASE_NAME=qwen3vl-8b_20260317_01


# Qwen3 Omni 启动示例:
MODEL_PATH=/home/dyvm6xra/dyvm6xrauser36/Projects/streaming_video_understanding/${CASE_NAME}/
echo "MODEL_PATH: $MODEL_PATH"

# CUDA_VISIBLE_DEVICES=1,2,3 numactl --cpunodebind=0 --membind=0 python -u Qwen3_VL_online_streaming_v2_ContextManaged.py \
CUDA_VISIBLE_DEVICES=1,2,3 python -u Qwen3_VL_online_streaming_v2_ContextManaged.py \
    --listen-port 12345 \
    --model $MODEL_PATH \
    --tensor-parallel-size 2 \
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
    --tts-gpu 2 \
    --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --tts-language Chinese \
    --tts-ref-audio test_query.mp3 \
    --tts-ref-text "仔细观察当前你看到的画面，并且结合之前你看到的画面，仔细描述你看到了什么" \
    --tts-output-dir tts_results \
    --enable-pruning \
    --max-rounds 30 \
    --num-rounds-keep 10 \
    --max-context-qas 10 \
    --debug-context-file debug_context.jsonl \
    --debug-context 
