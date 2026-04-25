# Starts a long-running local vLLM server for AURA.
# Default config: GPU 0, port 8028.
CUDA_VISIBLE_DEVICES=0
DATA_PARALLEL_SIZE="$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")"
PORT="8028"

vllm serve aurateam/AURA \
  --data-parallel-size "$DATA_PARALLEL_SIZE" \
  --tensor-parallel-size 1 \
  --max-model-len 32768 \
  --media-io-kwargs '{"video": {"num_frames": -1}}' \
  --limit-mm-per-prompt.video 50 \
  --gpu-memory-utilization 0.9 \
  --port "$PORT"
