#!/bin/bash
# 一键启动所有服务: ASR + TTS + vLLM 主推理
#
# 日志分别输出到 logs/ 目录下
# Ctrl+C 会自动终止所有后台服务

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── GPU 分配（按需修改）──
GPU_ASR=${GPU_ASR:-0}            # ASR 使用此 GPU
GPU_TTS=${GPU_TTS:-0}            # TTS 使用此 GPU（与 ASR 相同则共卡）
GPU_INFERENCE=${GPU_INFERENCE:-1} # vLLM 主推理，多卡用逗号分隔（如 3,4）

# ── 服务端口（可通过环境变量覆盖；见 .env.example）──
export AURA_FLASK_PORT="${AURA_FLASK_PORT:-5003}"
export AURA_INFER_PORT="${AURA_INFER_PORT:-12345}"
export AURA_ASR_PORT="${AURA_ASR_PORT:-8001}"
export AURA_TTS_PORT="${AURA_TTS_PORT:-8002}"

# ── 自动计算 Tensor Parallel 大小 ──
IFS=',' read -ra _GPU_LIST <<< "$GPU_INFERENCE"
TP_SIZE=${TP_SIZE:-${#_GPU_LIST[@]}}

# ── ASR 显存比例：共卡时留空间给 TTS，独占时可用更多 ──
if [ "$GPU_ASR" = "$GPU_TTS" ]; then
    ASR_GPU_UTIL=${ASR_GPU_UTIL:-0.3}
    echo "ℹ️  ASR & TTS share GPU $GPU_ASR → ASR gpu-memory-utilization=$ASR_GPU_UTIL"
else
    ASR_GPU_UTIL=${ASR_GPU_UTIL:-0.6}
    echo "ℹ️  ASR on GPU $GPU_ASR, TTS on GPU $GPU_TTS (separate) → ASR gpu-memory-utilization=$ASR_GPU_UTIL"
fi

LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# ── 启动前清理残留进程 ──
kill_port() {
    local port=$1
    local pids
    pids=$(ss -tlnp "sport = :$port" 2>/dev/null | awk 'NR>1{match($0,/pid=([0-9]+)/,a); if(a[1]) print a[1]}' | sort -u)
    if [ -n "$pids" ]; then
        echo "⚠️  Port $port is occupied by PID(s): $pids — killing..."
        echo "$pids" | xargs kill 2>/dev/null
        sleep 2
        echo "$pids" | xargs kill -9 2>/dev/null
        sleep 1
    fi
}

echo "🧹 Checking for leftover processes on ports $AURA_ASR_PORT, $AURA_TTS_PORT, $AURA_INFER_PORT..."
kill_port "$AURA_ASR_PORT"
kill_port "$AURA_TTS_PORT"
kill_port "$AURA_INFER_PORT"

PIDS=()

cleanup() {
    echo ""
    echo "🛑 Shutting down all services..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null
            echo "  Stopped PID $pid"
        fi
    done
    wait 2>/dev/null
    echo "👋 All services stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# ── 1. ASR 服务 ──
echo "🎙  Starting ASR service (GPU $GPU_ASR, port $AURA_ASR_PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_ASR ASR_GPU_UTIL=$ASR_GPU_UTIL bash asr_serve.sh > "$LOG_DIR/asr.log" 2>&1 &
PIDS+=($!)
echo "    PID=${PIDS[-1]}, log: logs/asr.log"

# 等待 ASR 就绪
echo "    Waiting for ASR to be ready..."
for i in $(seq 1 120); do
    if curl -s "http://localhost:$AURA_ASR_PORT/docs" > /dev/null 2>&1; then
        echo "    ✓ ASR service ready"
        break
    fi
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
        echo "    ✗ ASR process exited unexpectedly, check logs/asr.log"
        cleanup
    fi
    sleep 2
done

# ── 2. TTS 服务 ──
echo "🔊 Starting TTS service (GPU $GPU_TTS, port $AURA_TTS_PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_TTS bash tts_service.sh > "$LOG_DIR/tts.log" 2>&1 &
PIDS+=($!)
echo "    PID=${PIDS[-1]}, log: logs/tts.log"

# 等待 TTS 就绪
echo "    Waiting for TTS to be ready..."
for i in $(seq 1 180); do
    if curl -s "http://localhost:$AURA_TTS_PORT/v1/tts/health" 2>/dev/null | grep -q '"status":"ok"'; then
        echo "    ✓ TTS service ready"
        break
    fi
    if ! kill -0 "${PIDS[-1]}" 2>/dev/null; then
        echo "    ✗ TTS process exited unexpectedly, check logs/tts.log"
        cleanup
    fi
    sleep 2
done

# ── 3. 主推理服务 ──
echo "🚀 Starting vLLM inference server (GPU $GPU_INFERENCE, TP=$TP_SIZE, port $AURA_INFER_PORT)..."
CUDA_VISIBLE_DEVICES=$GPU_INFERENCE TP_SIZE=$TP_SIZE bash Qwen3_VL_online_streaming_v2_CM.sh > "$LOG_DIR/vllm.log" 2>&1 &
PIDS+=($!)
echo "    PID=${PIDS[-1]}, log: logs/vllm.log"

echo ""
echo "============================================"
echo "  All services launched!"
echo "  ASR:  http://localhost:$AURA_ASR_PORT  (GPU $GPU_ASR)"
echo "  TTS:  http://localhost:$AURA_TTS_PORT  (GPU $GPU_TTS)"
echo "  vLLM: port $AURA_INFER_PORT            (GPU $GPU_INFERENCE, TP=$TP_SIZE)"
echo ""
echo "  Logs: $LOG_DIR/"
echo "  Press Ctrl+C to stop all services"
echo "============================================"

# 前台等待，Ctrl+C 触发 cleanup
wait
