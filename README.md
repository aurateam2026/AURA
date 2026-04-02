<div align="center">

<img src="mascot2.png" width="200" alt="AURA Mascot">

# AURA

### Always-On Understanding and Real-Time Assistance via Video Streams

<img src="aura_cha.jpg" width="600" alt="AURA">

**A real-time multimodal streaming system powered by AURA-8B, supporting continuous video understanding with speech interaction.**

<a href="https://huggingface.co/"><img src="hf-logo.pdf" height="20" alt="Hugging Face"></a>&nbsp;&nbsp;
[Model on Hugging Face](https://huggingface.co/) • [Paper](#) • [Demo Video](#)

</div>

---

## Highlights

- **Real-Time Streaming**: Continuously processes live video at 2 FPS with sub-second response latency
- **Full Pipeline**: Integrated ASR, Vision-Language Model, and Streaming TTS, all running locally
- **Context Management**: Sliding-window history with automatic pruning and prefix KV cache reuse for bounded latency
- **Cross-Turn Anti-Repetition**: `logit_bias` soft penalty and optional `bad_words` hard blocking to prevent repetitive responses
- **Voice Clone TTS**: Sentence-level streaming synthesis with custom voice cloning support
- **One-Click Launch**: Single script (`start_all.sh`) to start all services with automatic GPU allocation

## Requirements

| Category | Requirement |
|----------|-------------|
| Python | 3.12 |
| PyTorch | 2.10+ with CUDA 12.8 |
| vLLM | >= 0.17.1 (V1 engine with Automatic Prefix Caching) |
| GPU | 2+ (minimum: 1 for ASR+TTS, 1 for AURA-8B inference) |
| System | `ffmpeg`, `numactl` |
| OS | Linux (tested on Ubuntu 22.04) |

## Installation

```bash
git clone https://github.com/aurateam2026/AURA.git && cd AURA

# 1. Create and activate a Python 3.12 virtual environment
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install system dependencies
sudo apt install -y ffmpeg numactl

# 3. Install all Python packages (228 packages, pinned versions)
pip install -r requirements.txt

# 4. Install flash-attn (requires a platform-specific .whl matching your CUDA/PyTorch/arch)
#    Download the correct wheel from https://github.com/Dao-AILab/flash-attention/releases
#    Example for CUDA 12 + PyTorch 2.10 + x86_64:
pip install flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```

> **Note:** `flash-attn` is **not** included in `requirements.txt` because it requires a platform-specific `.whl` file. You must download the correct wheel that matches your CUDA version, PyTorch version, and CPU architecture, then install it manually.

> **Note:** The `Qwen3-TTS-streaming/` subdirectory is a local library loaded at runtime via `sys.path`. It does **not** need a separate `pip install`.

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

### 1. Download Models

Download the following models from [Hugging Face](https://huggingface.co/):

| Model | Purpose | Size |
|-------|---------|------|
| AURA-8B | Main vision-language model | ~16 GB |
| [Qwen3-ASR-1.7B](https://huggingface.co/) | Automatic Speech Recognition | ~3 GB |
| [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/) | Text-to-Speech synthesis | ~4 GB |

### 2. One-Click Launch

```bash
# Default: GPU 0 for ASR+TTS, GPU 1 for AURA inference
bash start_all.sh
```

The script automatically:
- Cleans up any leftover processes on ports 8001, 8002, 12345
- Starts ASR, TTS, and vLLM inference server in order
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

The web frontend connects to the backend inference server via a TCP socket. By default, the backend hostname is configured in `realtime_capture_video_audio_streaming.py`:

```python
SERVER_HOST = 'hk01dgx030'   # Change this to match your setup
SERVER_PORT = 12345
```

- **If the frontend and backend run on the same machine**, change `SERVER_HOST` to `'localhost'`.
- **If they run on different machines**, set `SERVER_HOST` to the hostname or IP address of the machine running the backend services.

Then start the frontend in a separate terminal:

```bash
source .venv/bin/activate
python realtime_capture_video_audio_streaming.py
```

| Mode | Command |
|------|---------|
| HTTP (default) | `python realtime_capture_video_audio_streaming.py` |
| HTTPS | `python realtime_capture_video_audio_streaming.py --https` |
| Cloudflare Tunnel | `python realtime_capture_video_audio_streaming.py --tunnel` |

### 4. Access from a Browser

**Local access (desktop):**

Open `http://localhost:5003` in your browser.

**Remote access from a phone:**

To use AURA from a phone browser (e.g., Safari on iPhone or Chrome on Android), the phone must be able to reach the frontend server. There are several ways:

1. **Same LAN**: If the phone and the server are on the same network, open `http://<server-ip>:5003` on the phone. Note that most browsers **require HTTPS** to access the camera and microphone from a non-localhost address.

2. **HTTPS mode** (recommended for LAN access from phone):
   ```bash
   # Generate a self-signed certificate (one-time setup)
   openssl req -x509 -newkey rsa:2048 -keyout key.pem -out cert.pem -days 365 -nodes

   # Start the frontend with HTTPS
   python realtime_capture_video_audio_streaming.py --https
   ```
   Then open `https://<server-ip>:5003` on your phone. You will need to accept the self-signed certificate warning in the browser.

3. **Cloudflare Tunnel** (recommended for public/cross-network access):
   ```bash
   python realtime_capture_video_audio_streaming.py --tunnel
   ```
   This creates a public HTTPS URL that you can open on any device without network restrictions.

### 5. Using the Demo

The interface has three buttons at the bottom of the screen:

| Button | Icon | Action |
|--------|------|--------|
| **Start** | Camera | Tap to start/stop the video stream. The camera feed is sent to the backend for real-time understanding. |
| **Record** | Microphone | **Press and hold** to record your voice. **Release** to stop recording and send the audio to the server for ASR. Do not tap -- you must hold the button down while speaking. |
| **Flip** | Camera Rotate | Tap to switch between the front and rear cameras (useful on phones). |

**Typical workflow:**

1. Tap **Start** to activate the camera. Grant camera permission when prompted.
2. Point the camera at something you want AURA to understand.
3. **Press and hold** the **Record** button while asking your question out loud. Release when done.
4. Watch the streaming text response appear on screen in real-time.
5. The TTS audio response will play automatically through your speaker.
6. Tap **Flip** to switch cameras if needed.
7. Tap **Start** again to stop the video stream.

> **Tip:** On mobile devices, make sure to grant both camera and microphone permissions when prompted by the browser.

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

Wait for: `Server listening on port 12345`

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
├── context_manage.py                         # Context management utilities
├── realtime_capture_video_audio_streaming.py  # Web frontend middleware (Flask)
├── templates/index_streaming.html            # Browser UI (main interface)
├── templates/video-call.html                 # Browser UI (video call style)
├── requirements.txt                          # Python dependencies
├── shuhan.mp3                                # TTS reference audio for voice cloning
└── Qwen3-TTS-streaming/                      # TTS model inference library
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `sched_setaffinity: Invalid argument` | Remove `numactl` from the launch script |
| ASR returns empty text | Ensure the ASR service is running on port 8001 before starting the main server |
| TTS voice clone fails | Verify the reference audio file exists in the working directory |
| OOM on main GPU | Reduce `--gpu-memory-utilization` or `--max-model-len` |
| vLLM version error | Requires vLLM >= 0.17.1 with V1 engine support |
| Phone cannot access camera/mic | Use HTTPS mode or Cloudflare Tunnel (browsers require HTTPS for media on non-localhost) |
| `SERVER_HOST` connection refused | Verify `SERVER_HOST` in `realtime_capture_video_audio_streaming.py` matches your backend host |

## License

This project is released under the [Apache-2.0 License](LICENSE).

## Citation

```bibtex
@article{aura2026,
  title={AURA: Always-On Understanding and Real-Time Assistance via Video Streams},
  year={2026}
}
```
