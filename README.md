# Qwen3 实时流式视频理解系统

基于 Qwen3-VL 模型的实时视频流理解系统，支持视频输入、语音识别（ASR）、大模型推理、语音合成（TTS）全链路流式处理。

## 系统架构

```
┌──────────────┐  HTTP :5003   ┌─────────────────────┐  TCP :12345   ┌──────────────────────────────┐
│  浏览器客户端  │ ◄──────────► │  Flask 中间层          │ ◄──────────► │  主服务 (vLLM + TTS)          │
│  (Web UI)    │              │  (capture_streaming)  │              │  (Qwen3_VL_online_streaming) │
└──────────────┘              └─────────────────────┘              └──────────────┬───────────────┘
                                                                                  │ HTTP :8001
                                                                                  ▼
                                                                   ┌──────────────────────────┐
                                                                   │  ASR 服务                  │
                                                                   │  (Qwen3_asr_serve.py)     │
                                                                   └──────────────────────────┘
```

**数据流：**

1. 浏览器采集视频/音频 → Flask 中间层 → 通过 TCP Socket 发送给主服务
2. 主服务收到音频后 → 调用 ASR HTTP 接口转文字
3. 主服务收到视频 + 文字后 → vLLM 推理生成回复
4. 回复文本 → 内嵌 TTS 模块生成语音 → 流式返回给浏览器播放

## 环境依赖

- Python 3.12
- vLLM >= 0.14.0rc2（需要 V1 引擎 StreamingInput 支持）
- PyTorch 2.9+ with CUDA 12.8
- 4 张 GPU（推荐配置，可根据实际情况调整）

安装依赖：

```bash
pip install -r requirements.txt
```

## 启动流程

系统由 3 个进程组成，需要按顺序在 3 个终端中启动。

### 第一步：启动 ASR 服务（终端 1）

ASR 服务基于 `Qwen3-ASR-1.7B` 模型，通过 FastAPI 提供 HTTP 接口，监听端口 `8001`。

```bash
CUDA_VISIBLE_DEVICES=0 python Qwen3_asr_serve.py \
    --host 0.0.0.0 \
    --port 8001 \
    --model Qwen/Qwen3-ASR-1.7B \
    --forced-aligner Qwen/Qwen3-ForcedAligner-0.6B \
    --gpu-memory-utilization 0.8 \
    --device cuda:0
```

**参数说明：**

| 参数 | 说明 |
|---|---|
| `CUDA_VISIBLE_DEVICES=0` | 分配 GPU 0 给 ASR |
| `--port 8001` | HTTP 端口，主服务通过 `--asr-url http://localhost:8001/asr` 调用 |
| `--no-forced-aligner` | 可选，禁用时间戳对齐（节省显存） |

**验证启动成功：** 日志出现 `Qwen3 ASR Model ... loaded successfully`，可用以下命令测试：

```bash
curl -X POST http://localhost:8001/asr \
  -F "file=@test_audio.mp3" \
  -F "run_vllm=false"
```

### 第二步：启动主服务 — vLLM 引擎 + TTS（终端 2）

核心服务，通过入口脚本启动，内部会依次：

1. 初始化 TTS 模型（加载到指定 GPU）
2. 启动 TTS worker 线程
3. 初始化 vLLM AsyncLLM 引擎（加载主模型）
4. 开启 TCP Socket 监听（端口 12345）

```bash
bash Qwen3_VL_online_streaming_v2.sh
```

启动脚本实际执行：

```bash
MODEL_PATH=/home/dyvm6xra/dyvm6xrauser36/Projects/streaming_video_understanding/qwen3vl-8b_20260225_02/

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
```

**参数说明：**

| 参数 | 说明 |
|---|---|
| `CUDA_VISIBLE_DEVICES=1,2,3` | 主服务可见 GPU 1/2/3（GPU 0 已留给 ASR） |
| `--listen-port 12345` | TCP Socket 监听端口，Flask 中间层连接此端口 |
| `--model` | Qwen3 VL 8B 模型路径 |
| `--tensor-parallel-size 1` | 张量并行数（当前仅用 1 张卡跑主模型） |
| `--max-model-len 262144` | 最大上下文长度 256K tokens |
| `--gpu-memory-utilization 0.9` | vLLM 占用 90% 显存 |
| `--asr-url` | 指向第一步启动的 ASR 服务地址 |
| `--kv-offloading-size 10` | KV Cache offload 到 CPU 的大小（GB） |
| `--enable-tts` | 启用内嵌 TTS |
| `--tts-gpu 3` | TTS 模型加载到 GPU 3（CUDA_VISIBLE_DEVICES 映射后的索引） |
| `--tts-model` | TTS 模型路径（Base 模型支持 Voice Clone 流式输出） |
| `--tts-ref-audio` / `--tts-ref-text` | Voice Clone 参考音频和对应文本 |
| `--enable-pruning` | 启用历史轮次裁剪，防止 context 超长 |
| `--max-rounds 120` | 触发裁剪的最大轮次数 |
| `--num-rounds-keep 20` | 裁剪时保留最近的轮次数 |

**验证启动成功：** 日志依次出现：

1. `✓ TTS model initialized on cuda:X` — TTS 就绪
2. `✅ Qwen3 VL AsyncLLM engine initialized successfully` — vLLM 引擎就绪
3. `🌐 Server listening on port 12345` — TCP 监听就绪

### 第三步：启动 Web 前端中间层（终端 3）

Flask 应用作为浏览器和主服务之间的桥梁，监听端口 `5003`。

```bash
python realtime_capture_video_audio_streaming.py
```

> **注意：** 启动前需确认 `realtime_capture_video_audio_streaming.py` 中的 `SERVER_HOST` 和 `SERVER_PORT` 与主服务一致。当前默认值：
>
> ```python
> SERVER_HOST = 'hk01dgx050'
> SERVER_PORT = 12345
> ```
>
> 如果主服务在本机运行，需要将 `SERVER_HOST` 改为 `localhost`。

**可选启动方式：**

| 模式 | 命令 |
|---|---|
| HTTP（默认） | `python realtime_capture_video_audio_streaming.py` |
| HTTPS（需要 cert.pem / key.pem） | `python realtime_capture_video_audio_streaming.py --https` |
| Cloudflare Tunnel（公网访问） | `python realtime_capture_video_audio_streaming.py --tunnel` |

**验证启动成功：**

- 日志出现 `✓ 已连接到服务端 ...:12345`
- 浏览器访问 `http://localhost:5003` 看到实时视频音频捕获界面

如果主服务未启动，会提示 `⚠ 服务端 ... 不可用`，Flask 会每 5 秒自动重试连接。

### 浏览器访问

打开浏览器访问 `http://localhost:5003`（或对应的 HTTPS / Tunnel 地址）。

页面功能：
- 点击「开启摄像头」获取会话并开始视频采集
- 按住麦克风按钮录制语音，松开后自动发送
- 实时显示流式生成的文本回复
- 自动播放 TTS 语音回复

## GPU 分配参考

| GPU | 用途 | 显存需求（约） |
|---|---|---|
| GPU 0 | ASR 服务（Qwen3-ASR-1.7B + ForcedAligner-0.6B） | ~3 GB |
| GPU 1 | vLLM 主模型（Qwen3-VL-8B） | ~16 GB |
| GPU 2 | 对 vLLM 可见但 TP=1 未使用 | — |
| GPU 3 | TTS 模型（Qwen3-TTS-12Hz-1.7B-Base） | ~4 GB |

> GPU 编号基于 `CUDA_VISIBLE_DEVICES` 映射。主服务内部的 `--tts-gpu 3` 对应的是映射后的第 3 个设备（即物理 GPU 3）。

## 启动顺序速查

```
终端 1 (GPU 0):  ASR 服务
    CUDA_VISIBLE_DEVICES=0 python Qwen3_asr_serve.py --port 8001
    等待: "Qwen3 ASR Model loaded successfully"

终端 2 (GPU 1,2,3):  主服务 (vLLM + TTS)
    bash Qwen3_VL_online_streaming_v2.sh
    等待: "Server listening on port 12345"

终端 3:  Web 前端
    python realtime_capture_video_audio_streaming.py
    等待: "已连接到服务端"

浏览器:  访问 http://localhost:5003
```

## 通信协议

Flask 中间层与主服务之间通过自定义 TCP 二进制协议通信。

消息格式：`[Type: 1 byte] [Length: 8 bytes big-endian] [Payload: variable]`

| Type | 方向 | 含义 |
|---|---|---|
| 1 | Client → Server | 视频帧（WebM） |
| 2 | Client → Server | 音频录制（MP3） |
| 3 | Server → Client | 完整文本响应 |
| 4 | Client → Server | 清空上下文 |
| 5 | Server → Client | TTS 音频（完整 WAV，句子级） |
| 6 | Client → Server | 开启摄像头（重置状态） |
| 7 | Server → Client | 错误消息 |
| 8 | Server → Client | 流式 Token（JSON） |
| 9 | Server → Client | TTS 音频 Chunk（PCM int16 流式） |

## 注意事项

1. **启动顺序**：ASR 服务必须先于主服务启动（或至少在主服务收到音频前就绪），否则语音识别会失败返回空文本
2. **GPU 显存**：ASR 占 ~3GB，主模型 8B 占 ~16GB，TTS 占 ~4GB，确保各 GPU 有足够显存
3. **参考音频**：`test_query.mp3` 必须存在于工作目录中，否则 TTS Voice Clone 初始化会失败
4. **numactl**：启动脚本使用了 NUMA 绑定（`--cpunodebind=0 --membind=0`），需确保服务器安装了 `numactl`
5. **vLLM 版本**：需要 vLLM >= 0.14.0rc2 以支持 V1 引擎的 StreamingInput API

## 文件说明

| 文件 | 说明 |
|---|---|
| `Qwen3_VL_online_streaming_v2.sh` | 主服务启动脚本（入口） |
| `Qwen3_VL_online_streaming_v2.py` | 主服务实现（vLLM 引擎 + TTS + TCP Socket） |
| `Qwen3_asr_serve.py` | ASR 服务（FastAPI + Qwen3-ASR） |
| `realtime_capture_video_audio_streaming.py` | Web 前端中间层（Flask + TCP 客户端） |
| `context_manage.py` | 上下文管理工具（历史裁剪、相似度过滤） |
| `templates/index_streaming.html` | 前端页面模板 |
| `Qwen3-TTS-streaming/` | TTS 模型推理库（本地包） |
