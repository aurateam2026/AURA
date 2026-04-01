<div align="center">

<img src="mascot2.png" width="200" alt="AURA Mascot">

# AURA

### Always-On Understanding and Real-Time Assistance via Video Streams

<img src="aura_cha.jpg" width="600" alt="AURA">

**A real-time multimodal streaming system powered by AURA-8B, supporting continuous video understanding with speech interaction.**

[中文](#中文说明) | **English**

<a href="https://huggingface.co/"><img src="hf-logo.pdf" height="20" alt="Hugging Face"></a>&nbsp;&nbsp;
[Model on Hugging Face](https://huggingface.co/) • [Paper](#) • [Demo Video](#)

</div>

---

## Highlights

- **Real-Time Streaming**: Continuously processes live video at 2 FPS with sub-second response latency
- **Full Pipeline**: Integrated ASR → Vision-Language Model → Streaming TTS, all running locally
- **Context Management**: Sliding-window history with automatic pruning and prefix KV cache reuse for bounded latency
- **Cross-Turn Anti-Repetition**: `logit_bias` soft penalty + optional `bad_words` hard blocking to prevent repetitive responses
- **Voice Clone TTS**: Sentence-level streaming synthesis with custom voice cloning support
- **One-Click Launch**: Single script (`start_all.sh`) to start all services with automatic GPU allocation

## Requirements

| Category | Requirement |
|----------|-------------|
| Python | 3.12 |
| PyTorch | 2.10+ with CUDA 12.8 |
| vLLM | >= 0.17.1 (V1 engine with Automatic Prefix Caching) |
| GPU | 2+ (minimum: 1× for ASR+TTS, 1× for AURA-8B inference) |
| System | `ffmpeg`, `numactl` |
| OS | Linux (tested on Ubuntu 22.04) |

## Installation

### Option A: Use Pre-built Environment (Recommended)

The repository ships with a ready-to-use `.venv/` that contains all 230 pre-installed packages (Python 3.12, PyTorch 2.10, vLLM 0.17.1, flash-attn 2.8.3, etc.). Just activate it:

```bash
git clone <repo-url> && cd streaming_demo_modified
source .venv/bin/activate

# Verify
python --version   # Python 3.12.12
python -c "import vllm; print(vllm.__version__)"   # 0.17.1
python -c "import torch; print(torch.__version__)"  # 2.10.0
```

### Option B: Create Environment from Scratch

If the pre-built `.venv/` is unavailable or incompatible with your platform:

```bash
git clone <repo-url> && cd streaming_demo_modified

# 1. Create and activate venv
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install system dependencies
sudo apt install -y ffmpeg numactl

# 3. Install all Python packages (228 packages, pinned versions)
pip install -r requirements.txt

# 4. Install flash-attn (requires manual .whl matching your CUDA/PyTorch/arch)
#    Download the correct wheel from https://github.com/Dao-AILab/flash-attention/releases
#    Example for CUDA 12 + PyTorch 2.10 + x86_64:
pip install flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

> **Note:** `flash-attn` is **not** included in `requirements.txt` because it requires a platform-specific `.whl` file. You must download the correct wheel that matches your CUDA version, PyTorch version, and CPU architecture, then install manually.

> **Note:** The `Qwen3-TTS-streaming/` subdirectory is a local library loaded at runtime via `sys.path` — it does **not** need separate `pip install`. The `vllm-omni/` directory is for reference only; the system uses the pip-installed `vllm==0.17.1`.

### Verify Installation

```bash
source .venv/bin/activate
python -c "
import torch, vllm, flask, qwen_omni_utils
print(f'PyTorch:  {torch.__version__}')
print(f'CUDA:     {torch.version.cuda}')
print(f'vLLM:     {vllm.__version__}')
print(f'GPUs:     {torch.cuda.device_count()}')
"
```

Expected output:
```
PyTorch:  2.10.0
CUDA:     12.8
vLLM:     0.17.1
GPUs:     2  (or more)
```

## Quick Start

### 1. Download Models (下载模型)

Download the following models from [Hugging Face](https://huggingface.co/):

| Model | Purpose | Size |
|-------|---------|------|
| AURA-8B | Main vision-language model | ~16 GB |
| [Qwen3-ASR-1.7B](https://huggingface.co/) | Automatic Speech Recognition | ~3 GB |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/) | Text-to-Speech synthesis | ~4 GB |

### 2. One-Click Launch (Recommended)

```bash
# Default: GPU 0 for ASR+TTS, GPU 1 for AURA inference
bash start_all.sh
```

The script automatically:
- Cleans up any leftover processes on ports 8001, 8002, 12345
- Starts ASR → TTS → vLLM inference server in order
- Waits for each service to be healthy before proceeding
- Logs to `logs/asr.log`, `logs/tts.log`, `logs/vllm.log`
- `Ctrl+C` cleanly shuts down all services

**Custom GPU allocation:**

```bash
GPU_ASR=0 GPU_TTS=0 GPU_INFERENCE=1 bash start_all.sh

# Multi-GPU inference (tensor parallel)
GPU_ASR=0 GPU_TTS=0 GPU_INFERENCE=2,3 bash start_all.sh
```

### 3. Launch Web Frontend

In a separate terminal:

```bash
source .venv/bin/activate
python realtime_capture_video_audio_streaming.py
```

Open browser at `http://localhost:5003`.

| Mode | Command |
|------|---------|
| HTTP (default) | `python realtime_capture_video_audio_streaming.py` |
| HTTPS | `python realtime_capture_video_audio_streaming.py --https` |
| Cloudflare Tunnel | `python realtime_capture_video_audio_streaming.py --tunnel` |

### 4. Use the Demo

1. Click **"开启摄像头"** to start video capture
2. Hold the **microphone button** to record speech, release to send
3. Watch streaming text responses appear in real-time
4. Hear TTS audio playback automatically

## Manual Service Launch

If you prefer to start services individually:

<details>
<summary><b>Step 1: ASR Service (Port 8001)</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 python Qwen3_asr_serve.py \
    --host 0.0.0.0 --port 8001 \
    --model Qwen/Qwen3-ASR-1.7B \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B \
    --gpu-memory-utilization 0.3
```

Verify: `curl -X POST http://localhost:8001/asr -F "file=@test_query.mp3" -F "run_vllm=false"`

</details>

<details>
<summary><b>Step 2: TTS Service (Port 8002)</b></summary>

```bash
CUDA_VISIBLE_DEVICES=0 bash tts_service.sh
```

Verify: `curl http://localhost:8002/v1/tts/health`

</details>

<details>
<summary><b>Step 3: Main Inference Server (Port 12345)</b></summary>

```bash
CUDA_VISIBLE_DEVICES=1 bash Qwen3_VL_online_streaming_v2_CM.sh
```

Wait for: `🌐 Server listening on port 12345`

</details>

<details>
<summary><b>Step 4: Web Frontend (Port 5003)</b></summary>

```bash
python realtime_capture_video_audio_streaming.py
```

Open: `http://localhost:5003`

</details>

## GPU Allocation Reference

| GPU | Service | VRAM |
|-----|---------|------|
| GPU 0 | ASR (Qwen3-ASR-1.7B) + TTS (Qwen3-TTS-1.7B) | ~7 GB |
| GPU 1 | AURA-8B inference (vLLM, TP=1) | ~16 GB |

## Key Configuration

Main inference parameters in `Qwen3_VL_online_streaming_v2_CM.sh`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-model-len` | 262144 | Maximum context length (256K tokens) |
| `--temperature` | 0.5 | Sampling temperature |
| `--max-tokens` | 128 | Max tokens per response |
| `--cross-turn-penalty` | 1 | Cross-turn repetition penalty strength |
| `--cross-turn-lookback` | 10 | Number of recent turns to penalize |
| `--enable-pruning` | — | Enable sliding-window context pruning |
| `--max-rounds` | 45 | Trigger pruning when rounds exceed this |
| `--num-rounds-keep` | 30 | Rounds to keep after pruning |
| `--kv-offloading-size` | 10 | KV cache CPU offload size (GB) |

## Project Structure

```
├── start_all.sh                              # One-click launch script
├── Qwen3_VL_online_streaming_v2_CM.sh        # Main inference launch script
├── Qwen3_VL_online_streaming_v2_ContextManaged.py  # Core: vLLM engine + context management + TCP server
├── Qwen3_asr_serve.py                        # ASR service (FastAPI + Qwen3-ASR)
├── tts_service.py / tts_service.sh           # TTS service (streaming synthesis)
├── realtime_capture_video_audio_streaming.py  # Web frontend middleware (Flask)
├── templates/index_streaming.html            # Browser UI template
├── benchmark_latency.py                      # Latency benchmark tool
├── context_manage.py                         # Context management utilities
├── requirements.txt                          # Python dependencies
├── .venv/                                    # Pre-built virtual environment
├── Qwen3-TTS-streaming/                      # TTS model inference library
└── vllm-omni/                                # vLLM omni-modal fork
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `sched_setaffinity: Invalid argument` | Remove `numactl` from launch script |
| ASR returns empty text | Ensure ASR service is running on port 8001 before main server |
| TTS voice clone fails | Verify reference audio file exists in working directory |
| OOM on main GPU | Reduce `--gpu-memory-utilization` or `--max-model-len` |
| vLLM version error | Requires vLLM >= 0.17.1 with V1 engine support |

---

<a name="中文说明"></a>

## 中文说明

AURA（Always-On Understanding and Real-Time Assistance via Video Streams）是一个基于 AURA-8B 视觉语言模型的实时流式视频理解系统，支持视频输入、语音识别（ASR）、大模型推理、语音合成（TTS）全链路流式处理。

### 环境安装

**方式一：使用预构建环境（推荐）**

仓库自带 `.venv/` 目录，包含全部 230 个已安装 Python 包（Python 3.12、PyTorch 2.10、vLLM 0.17.1、flash-attn 2.8.3 等），直接激活即可：

```bash
source .venv/bin/activate
```

**方式二：从零创建环境**

```bash
# 创建 Python 3.12 虚拟环境
python3.12 -m venv .venv && source .venv/bin/activate

# 安装系统依赖
sudo apt install -y ffmpeg numactl

# 安装 Python 依赖（228 个包，锁定版本）
pip install -r requirements.txt

# 手动安装 flash-attn（需下载与 CUDA/PyTorch/架构匹配的 .whl）
pip install flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

> `Qwen3-TTS-streaming/` 是运行时通过 `sys.path` 加载的本地库，无需单独安装。

### 快速启动

```bash
# 1. 激活环境
source .venv/bin/activate

# 2. 一键启动所有服务（ASR → TTS → 推理引擎）
bash start_all.sh

# 3. 在另一个终端启动 Web 前端
source .venv/bin/activate
python realtime_capture_video_audio_streaming.py

# 4. 浏览器访问
# http://localhost:5003
```

详细配置和参数说明请参考上方英文文档。

## License

This project is released under the [Apache-2.0 License](LICENSE).

## Citation

```bibtex
@article{aura2026,
  title={AURA: Always-On Understanding and Real-Time Assistance via Video Streams},
  year={2026}
}
```
