# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Qwen3 Omni Streaming Input Example

本文件实现了基于 vLLM V1 流式输入 API 的 Qwen3 Omni 模型实时视频流处理。

基于 Qwen3_VL_online_streaming.py 修改，适配 Qwen3 Omni 模型。

1. 输入模态: whisper 转化后的 text + video
2. 输出模态: 仅文字 (之后利用外部 TTS 将文字回复变成语音)
3. 输出模态: 仅文字 (之后利用外部 TTS 将文字回复变成语音)

关键配置差异 (Qwen3 Omni vs Qwen3 VL):
- Silent Token ID: 151676 (Omni) vs 151669 (VL)
- 架构: Qwen3OmniMoeForConditionalGeneration (需要 trust_remote_code=True)
- <|audio_start|> = 151669 (在 Omni 中，VL 中的 151669 是 silent token)
- <|silent|> = 151676 (在 Omni 中)

架构:
┌─────────────────┐      ┌────────────────────┐      ┌──────────────────┐
│   Web Client    │ ◄──► │  AsyncLLM Engine   │ ◄──► │  GPU Workers     │
│   (Browser)     │      │  (Streaming Input) │      │  (Model)         │
└─────────────────┘      └────────────────────┘      └──────────────────┘

启动方式:
    python Qwen3_omni_online_streaming.py --listen-port 12345

依赖:
    - vllm >= 0.14.0rc2 (支持 StreamingInput)
    - 需要 vLLM V1 引擎
"""

import argparse
import asyncio
import base64
import json
import os
import signal
import socket
import struct
import threading
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import aiohttp
import requests
import re  # Added for TTS sentence splitting
import sys

from datetime import datetime

# Add Qwen3-TTS-streaming to path
sys.path.append(os.path.join(os.path.dirname(__file__), "Qwen3-TTS-streaming"))

# vLLM V1 imports for streaming input
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput

# Context management (reuse from original)
import context_manage

# Global configuration
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class VideoFrame:
    """A single video frame with timestamp."""
    data: bytes  # JPEG or raw frame data
    timestamp: float
    frame_id: int


# ============================================================================
# Session History Management
# ============================================================================

# Special token IDs
# Qwen3 Omni 和 Qwen3 VL 的 silent token ID 不同:
#   Qwen3 VL:   SILENT_TOKEN_ID = 151669  (<|silent|>)
#   Qwen3 Omni: SILENT_TOKEN_ID = 151676  (<|silent|>), 151669 在 Omni 中是 <|audio_start|>
# SILENT_TOKEN_ID 在 main() 中根据 --model 路径动态设置
SILENT_TOKEN_ID = None  # 由 detect_model_type() 设置
IM_END_TOKEN_ID = 151645   # <|im_end|> token id
SILENT_TEXT = "<|silent|>"

VISION_START_TOKEN_ID = 151652 # <|vision_start|>
VISION_END_TOKEN_ID = 151653   # <|vision_end|>
VIDEO_PAD_TOKEN_ID = 151656    # <|video_pad|>
IMAGE_PAD_TOKEN_ID = 151655    # <|image_pad|>


def detect_model_type(model_path: str) -> str:
    """根据模型路径判断模型类型: 'omni' 或 'vl'"""
    name = os.path.basename(model_path.rstrip("/")).lower()
    if "omni" in name:
        return "omni"
    return "vl"


def setup_silent_token_id(model_path: str):
    """根据模型路径设置全局 SILENT_TOKEN_ID"""
    global SILENT_TOKEN_ID
    model_type = detect_model_type(model_path)
    if model_type == "omni":
        SILENT_TOKEN_ID = 151676
    else:
        SILENT_TOKEN_ID = 151669
    print(f"🔧 Model type detected: {model_type} → SILENT_TOKEN_ID = {SILENT_TOKEN_ID}")


class SessionHistory:
    """
    Manages conversation history for a session to enable KV Cache Reuse (Prefix Caching).
    Strategy: Sliding window - keep most recent N rounds when limit is reached.
    """
    def __init__(self, max_rounds: int = 20, num_rounds_keep: int = 15, pruning_enabled: bool = False):
        self.history = []  # List of {"role": str, "content": str/list}
        # max_rounds determines when to prune history
        self.max_rounds = max_rounds
        # num_rounds_keep is used whiling pruning history
        self.num_rounds_keep = num_rounds_keep
        self.pruning_enabled = pruning_enabled
        self.current_rounds = 0

        # Initial system message
        # 注意: Qwen3 Omni 输入是 text + video, 输出仅 text
        self.system_prompt = "You are receiving a live video stream where the final frame is the present moment. Respond only when a response is needed based on the user's message or the visual context. Otherwise, output '<|silent|>' to signify silence. Respond in Chinese."
        self._reset()

    def _reset(self):
        """Reset history to initial state (complete reset)."""
        self.history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        # self.history = []
        self.current_rounds = 0
        print(f"🔄 Session history reset")

    def _extract_user_text(self, content) -> str:
        """
        从 user message 的 content 中提取纯文字。

        Args:
            content: 可能是 str 或 list[dict]

        Returns:
            字符串（可能为空字符串 ""）
        """
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            texts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    text = item.get("text", "")
                    if text:
                        texts.append(text)
            return " ".join(texts)  # 可能为空字符串 ""

        return ""

    def _is_silent_response(self, content) -> bool:
        """检查 assistant 回复是否为 <|silent|>"""
        if isinstance(content, str):
            return content.strip() == SILENT_TEXT
        return False

    def _prune_history(self):
        """
        智能裁剪历史记录，分为三个区域：

        1. 原始区（最新的 num_rounds_keep 轮）：保持原样，包含视频数据
        2. 精简区（num_rounds_keep 到 2*num_rounds_keep 之间）：
           - silent 回复的轮次：删除
           - 非 silent 回复的轮次：转为纯文本保留
        3. 超出区（超过 2*num_rounds_keep）：强制删除，只保留 num_rounds_keep 轮

        History 结构:
        [system] + [精简区: 纯文本轮次] + [原始区: 带视频轮次]
                   ↑ 旧                    ↑ 新
        """
        if len(self.history) <= 1:  # Only system message, nothing to prune
            return

        # Step 1: 将消息配对为 (user, assistant) 轮次
        messages_after_system = self.history[1:]
        rounds = []  # List of (user_msg, assistant_msg) tuples
        i = 0

        while i < len(messages_after_system):
            msg = messages_after_system[i]
            if msg["role"] == "user":
                user_msg = msg
                assistant_msg = None
                if i + 1 < len(messages_after_system) and messages_after_system[i + 1]["role"] == "assistant":
                    assistant_msg = messages_after_system[i + 1]
                    i += 2
                else:
                    i += 1
                rounds.append((user_msg, assistant_msg))
            else:
                # 孤立的 assistant（不应该发生）
                i += 1

        num_rounds = len(rounds)
        max_total = 2 * self.num_rounds_keep  # 绝对上限

        # 如果没超过 max_rounds，不需要裁剪
        if num_rounds <= self.max_rounds:
            return

        # Step 2: 根据年龄处理每个轮次
        # rounds[0] 是最旧的，rounds[-1] 是最新的
        new_rounds = []
        silent_removed = 0
        force_removed = 0
        pruned_count = 0

        for idx, (user_msg, assistant_msg) in enumerate(rounds):
            # 年龄：从最新的开始计数，最新的 age=1
            age = num_rounds - idx

            if age <= self.num_rounds_keep:
                # 原始区：最新的 num_rounds_keep 轮，保持原样
                new_rounds.append((user_msg, assistant_msg, False))  # False = 不精简
            elif age <= max_total:
                # 精简区：num_rounds_keep < age <= 2*num_rounds_keep
                if assistant_msg and self._is_silent_response(assistant_msg["content"]):
                    # Silent 回复：删除整个轮次
                    silent_removed += 1
                    continue
                else:
                    # 非 Silent：转为纯文本保留
                    new_rounds.append((user_msg, assistant_msg, True))  # True = 需要精简
                    pruned_count += 1
            else:
                # 超出区：age > 2*num_rounds_keep，强制删除
                force_removed += 1
                continue

        # Step 3: 重建 history
        new_history = [self.history[0]]  # system message

        for user_msg, assistant_msg, should_prune in new_rounds:
            if should_prune:
                # 精简：user content 转为纯字符串
                user_text = self._extract_user_text(user_msg["content"])
                new_history.append({"role": "user", "content": user_text})
            else:
                # 保持原样（带视频）
                new_history.append(user_msg)

            if assistant_msg:
                new_history.append(assistant_msg)

        old_count = len(self.history)
        self.history = new_history

        # Recalculate current_rounds
        self.current_rounds = sum(1 for msg in self.history if msg["role"] == "user")

        print(f"✂️ History pruned: {old_count} → {len(self.history)} messages | "
              f"silent_removed={silent_removed}, force_removed={force_removed}, "
              f"pruned_to_text={pruned_count}, keeping {self.current_rounds} rounds")

    def add_user_message(self, text: str, images: list = None, video_tuple: tuple = None):
        """
        Add user message with optional images or video.

        Args:
            text: User text message
            images: List of PIL images (for image mode)
            video_tuple: Tuple of (numpy_array, metadata_dict) for video mode
                        numpy_array shape: (num_frames, height, width, 3)
                        metadata_dict: {"fps": float, "duration": float, "total_num_frames": int, ...}
        """
        # Check limit and prune old context when exceeding max_rounds
        if self.pruning_enabled and self.current_rounds >= self.max_rounds:
            print(f"⚠️ Max rounds ({self.max_rounds}) reached, pruning old context...")
            self._prune_history()  # 智能裁剪：原始区 + 精简区，上限 2*max_rounds

        content = []

        # Add video if provided (mutually exclusive with images)
        # Qwen3-VL expects video as (numpy_array, metadata_dict) tuple
        if video_tuple:
            content.append({"type": "video", "video": video_tuple})
        # Add images if provided
        elif images:
            content.extend([{"type": "image", "image": img} for img in images])

        # Add text
        if text:
            content.append({"type": "text", "text": text})
        elif not images and not video_tuple:
            # If no text AND no media, skip adding empty turn
            return

        self.history.append({"role": "user", "content": content})
        self.current_rounds += 1

    def add_assistant_message(self, text: str):
        """Add assistant response."""
        self.history.append({"role": "assistant", "content": text})

    def get_vllm_inputs(self):
        """
        Construct prompt and multi_modal_data for vLLM.
        MUST include ALL history media to match the text prompt for Prefix Caching.
        Supports both images and videos.
        """
        full_prompt = ""
        all_images = []
        all_videos = []

        for msg in self.history:
            role = msg["role"]
            content = msg["content"]

            full_prompt += f"<|im_start|>{role}"

            if isinstance(content, str):
                full_prompt += content
            elif isinstance(content, list):
                for item in content:
                    if item.get("type") == "text":
                        full_prompt += item.get("text", "")
                    elif item.get("type") == "image":
                        # Qwen3-VL image tokens
                        full_prompt += "<|vision_start|><|image_pad|><|vision_end|>"
                        all_images.append(item.get("image"))
                    elif item.get("type") == "video":
                        # Qwen3-VL video tokens
                        full_prompt += "<|vision_start|><|video_pad|><|vision_end|>"
                        all_videos.append(item.get("video"))

            full_prompt += "<|im_end|>"

        # Add generation prompt
        full_prompt += "<|im_start|>assistant"

        # Build multi_modal_data
        multi_modal_data = {}
        if all_images:
            multi_modal_data["image"] = all_images
        if all_videos:
            multi_modal_data["video"] = all_videos

        return {
            "prompt": full_prompt,
            "multi_modal_data": multi_modal_data
        }

@dataclass
class StreamingSession:
    """A streaming session for client."""
    session_id: str
    history: SessionHistory
    input_queue: asyncio.Queue
    output_queue: asyncio.Queue
    is_generating: bool = False  # Flag to prevent overlapping generation tasks
    is_auto_generating: bool = False  # True if generating without user prompt (background)
    current_task: Optional[asyncio.Task] = None  # Reference to the current generation task

# ============================================================================
# Global State
# ============================================================================

# Server state
active_connection = None
connection_lock = threading.Lock()

# TTS related globals
tts_omni = None
tts_enabled = False
tts_streaming = False
TTS_OUTPUT_DIR = "tts_results"
tts_lock = threading.Lock()
tts_event_loop = None
voice_clone_prompt_cache = None  # Cache for Base model voice clone prompt
tts_ref_audio_path = None  # Reference audio path for Base model
tts_ref_text = None  # Reference text for Base model

# TTS pending task management
pending_tts_task = None
pending_tts_lock = threading.RLock()
tts_worker_running = False
current_tts_response_id = None
tts_cancel_flag = False

# TTS sentence queue for streaming TTS (new)
import queue
tts_sentence_queue = queue.Queue()
tts_sentence_queue_lock = threading.Lock()

# TTS latency logging
TTS_LATENCY_LOG_PATH = "tts_latency.log"
tts_latency_log_lock = threading.Lock()

def log_tts_latency(
    response_id: str,
    sentence_idx: int,
    text: str,
    first_chunk_latency: float,
    total_latency: float,
    audio_duration: float,
    num_chunks: int,
    model_type: str = "unknown"
):
    """
    Log TTS latency metrics to file.

    Args:
        response_id: Unique response ID
        sentence_idx: Index of the sentence in the response
        text: The text that was synthesized
        first_chunk_latency: Time to first audio chunk (seconds)
        total_latency: Total processing time (seconds)
        audio_duration: Duration of generated audio (seconds)
        num_chunks: Number of audio chunks generated
        model_type: TTS model type (base/custom_voice)
    """
    import datetime

    # Calculate RTF (Real-Time Factor) - lower is better
    rtf = total_latency / audio_duration if audio_duration > 0 else float('inf')

    # Format log entry
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    text_preview = text[:50].replace('\n', ' ') + ('...' if len(text) > 50 else '')

    log_entry = (
        f"[{timestamp}] "
        f"response_id={response_id} | "
        f"sentence={sentence_idx} | "
        f"model={model_type} | "
        f"first_chunk={first_chunk_latency*1000:.1f}ms | "
        f"total={total_latency*1000:.1f}ms | "
        f"audio={audio_duration:.2f}s | "
        f"RTF={rtf:.3f} | "
        f"chunks={num_chunks} | "
        f"text=\"{text_preview}\"\n"
    )

    # Write to log file (thread-safe)
    with tts_latency_log_lock:
        try:
            with open(TTS_LATENCY_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)
        except Exception as e:
            print(f"⚠️ Failed to write TTS latency log: {e}")

    # Also print to console for immediate feedback
    print(f"📊 [TTS Latency] first_chunk={first_chunk_latency*1000:.1f}ms, total={total_latency*1000:.1f}ms, "
          f"audio={audio_duration:.2f}s, RTF={rtf:.3f}")

# Streaming session state
streaming_sessions: dict[str, StreamingSession] = {}
session_lock = asyncio.Lock()

# Directories
VIDEO_DIR = "real_time_captured_video"
AUDIO_DIR = "real_time_captured_audio"
TTS_OUTPUT_DIR = "tts_results"

# Engine instance
async_engine: Optional[AsyncLLM] = None

# Response ID counter
response_id_counter = 0
response_id_lock = threading.Lock()


def generate_response_id() -> str:
    """Generate a unique response ID."""
    global response_id_counter
    with response_id_lock:
        response_id_counter += 1
        return f"resp_{int(time.time() * 1000)}_{response_id_counter}"


# ============================================================================
# Video Frame Processing
# ============================================================================



def _decode_video_sync(
    file_data: bytes,
    input_path: str,
    target_fps: float,
    resize: bool,
) -> tuple:
    """Sync helper for run_in_executor: write file, downsample, remove. Returns (video_array, metadata) or (None, None)."""
    with open(input_path, "wb") as f:
        f.write(file_data)
    try:
        return downsample_video_to_numpy(input_path, target_fps=target_fps, resize=resize)
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


def downsample_video_to_numpy(
    input_path: str,
    target_fps: float = 2.0,
    resize: bool = False,
) -> tuple:
    """
    Downsample a video to target FPS and return as (numpy_array, metadata_dict) tuple.
    This format is compatible with Qwen3-VL's video input in vLLM.

    Args:
        input_path: Path to input video file
        target_fps: Target frame rate (default: 2 fps)
        resize: If True, resize frames to 1/8 resolution to reduce tokens and TTFT (default: True)

    Returns:
        Tuple of (numpy_array, metadata_dict) or (None, None) if failed
        numpy_array shape: (num_frames, height, width, 3), dtype=uint8
        metadata_dict: {"fps": float, "duration": float, "total_num_frames": int, ...}
    """
    import numpy as np

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            # Try to get more info about the file
            import os
            file_size = os.path.getsize(input_path) if os.path.exists(input_path) else 0
            print(f"❌ Cannot open video: {input_path} (file size: {file_size} bytes)")
            print(f"   This may be due to unsupported codec (iOS Chrome often uses different codecs)")
            return None, None

        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Debug: print video properties for diagnosis
        need_sequential_read = False
        if original_fps <= 0 or total_frames <= 0:
            print(f"⚠️ Video metadata issue: fps={original_fps}, frames={total_frames}, size={width}x{height}")
            need_sequential_read = True
            # Try to read frames manually if metadata is invalid (iOS Chrome compatibility)
            if original_fps <= 0:
                original_fps = 15.0  # Assume 15fps as fallback
                print(f"⚠️ Using fallback fps: {original_fps}")

        # Calculate frame step
        step = max(1, int(original_fps / target_fps))

        frames = []
        frame_indices = []

        if need_sequential_read:
            # iOS Chrome compatibility: read ALL frames sequentially, then subsample
            # Some codecs don't support random frame access (cap.set doesn't work)
            cap.release()
            cap = cv2.VideoCapture(input_path)  # Reopen to reset position

            all_frames = []
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append((frame_idx, frame))
                frame_idx += 1

            print(f"⚠️ Sequential read: got {len(all_frames)} frames, subsampling with step={step}")

            # Subsample frames
            for idx, frame in all_frames[::step]:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(idx)
        else:
            # Normal mode: use frame seeking (works for most codecs)
            duration = total_frames / original_fps
            frame_idx = 0

            while frame_idx < total_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
                    frame_indices.append(frame_idx)
                frame_idx += step

        cap.release()

        if not frames:
            print(f"❌ No frames extracted from video")
            return None, None

        # Stack frames into numpy array: (num_frames, height, width, 3)
        video_array = np.stack(frames, axis=0)

        # Resize to 1/8 resolution to reduce token count and TTFT (unless --no-video-resize)
        if resize:
            h, w = video_array.shape[1], video_array.shape[2]
            video_array = np.stack([
                cv2.resize(video_array[i], (w / 8, h / 8), interpolation=cv2.INTER_AREA)
                for i in range(video_array.shape[0])
            ], axis=0)

        # Calculate original duration based on extracted frame indices
        if frame_indices:
            original_frame_count = frame_indices[-1] + 1 if need_sequential_read else total_frames
            duration = original_frame_count / original_fps
        else:
            duration = len(frames) / target_fps

        # Create metadata dict required by Qwen3-VL
        metadata = {
            "fps": target_fps,
            "duration": len(frames) / target_fps,
            "total_num_frames": len(frames),
            "frames_indices": frame_indices,
            "video_backend": "opencv",
            "do_sample_frames": False,  # Already sampled
        }

        print(f"📹 Video downsampled: {duration:.1f}s @ {original_fps:.0f}fps → {len(frames)} frames @ {target_fps}fps")

        return video_array, metadata

    except Exception as e:
        print(f"❌ Error downsampling video: {e}")
        return None, None


# ============================================================================
# ASR (Automatic Speech Recognition)
# ============================================================================

def get_audio_prompt(audio_path: str, asr_url: str) -> str:
    """
    Transcribe audio file to text using ASR service (synchronous version).

    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.)
        asr_url: URL of the ASR service

    Returns:
        Transcribed text, or empty string if failed
    """
    print(f"🎤 Transcribing audio from {audio_path}...", flush=True)
    try:
        with open(audio_path, 'rb') as f:
            files = {'file': f}
            # Request only ASR, do not trigger vLLM in ASR service
            response = requests.post(asr_url, files=files, params={"run_vllm": "false"}, timeout=30)

        if response.status_code == 200:
            data = response.json()
            text = data.get("text", "")
            print(f"✅ Transcribed: {text!r}", flush=True)
            return text
        else:
            print(f"❌ ASR failed with status {response.status_code}: {response.text}")
            return ""
    except requests.exceptions.Timeout:
        print("❌ ASR request timeout")
        return ""
    except Exception as e:
        print(f"❌ ASR error: {e}")
        return ""


async def transcribe_audio_async(audio_path: str, asr_url: str) -> str:
    """
    Transcribe audio file to text using ASR service (asynchronous version).

    Args:
        audio_path: Path to the audio file (MP3, WAV, etc.)
        asr_url: URL of the ASR service

    Returns:
        Transcribed text, or empty string if failed
    """
    print(f"🎤 [Async] Transcribing audio from {audio_path}...", flush=True)
    try:
        async with aiohttp.ClientSession() as session:
            with open(audio_path, 'rb') as f:
                data = aiohttp.FormData()
                data.add_field('file', f, filename=os.path.basename(audio_path))

                async with session.post(
                    asr_url,
                    data=data,
                    params={"run_vllm": "false"},
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        text = result.get("text", "")
                        print(f"✅ [Async] Transcribed: {text!r}", flush=True)
                        return text
                    else:
                        error_text = await response.text()
                        print(f"❌ [Async] ASR failed with status {response.status}: {error_text}")
                        return ""
    except asyncio.TimeoutError:
        print("❌ [Async] ASR request timeout")
        return ""
    except Exception as e:
        print(f"❌ [Async] ASR error: {e}")
        return ""


# ============================================================================
# TTS (Text-to-Speech)
# ============================================================================

def split_text_to_sentences(text: str) -> list:
    """Split text into sentences, preserving punctuation."""
    if not text or not text.strip():
        return []
    
    # Split by sentence terminators
    sentence_pattern = r'([^。！？；.!?;]+[。！？；.!?;]?)'
    raw_sentences = re.findall(sentence_pattern, text)
    
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if not s:
            continue
        
        # Split long sentences at commas
        if len(s) > 50:
            sub_pattern = r'([^，,]+[，,]?)'
            sub_sentences = re.findall(sub_pattern, s)
            for sub in sub_sentences:
                sub = sub.strip()
                if sub:
                    sentences.append(sub)
        else:
            sentences.append(s)
    
    if not sentences and text.strip():
        sentences = [text.strip()]
    
    return sentences

def set_pending_tts_task(task: dict):
    """Set pending TTS task, overwriting old one and cancelling current."""
    global pending_tts_task, tts_cancel_flag, current_tts_response_id
    with pending_tts_lock:
        old_task = pending_tts_task
        pending_tts_task = task
        
        if current_tts_response_id is not None:
            tts_cancel_flag = True
            print(f"⏭ Cancelling current TTS (id={current_tts_response_id})")
        
        if old_task is not None:
            print(f"⏭ Dropping pending TTS task (id={old_task.get('response_id', 'unknown')})")

def get_pending_tts_task() -> dict:
    global pending_tts_task
    with pending_tts_lock:
        task = pending_tts_task
        pending_tts_task = None
        return task


# ============================================================================
# TTS Sentence Queue Functions (for streaming TTS)
# ============================================================================

# Sentence terminators for Chinese and English
SENTENCE_TERMINATORS = "。！？；.!?;，,"
SENTENCE_TERMINATORS_SET = set(SENTENCE_TERMINATORS)

def enqueue_tts_sentence(sentence: str, response_id: str, sentence_idx: int, args):
    """
    Add a sentence to the TTS queue for processing.
    This enables pipeline parallelism: model generates next sentence while TTS processes current.
    """
    if not sentence or not sentence.strip():
        return

    clean_text = context_manage.remove_markdown(sentence)
    if not clean_text.strip():
        return

    task = {
        "response_id": response_id,
        "sentence_idx": sentence_idx,
        "text": clean_text,
        "language": args.tts_language if args else "Chinese",
        "speaker": args.tts_speaker if args else "Vivian",
        "instruct": args.tts_instruct if args else "",
        "output_dir": args.tts_output_dir if args else "tts_results"
    }

    tts_sentence_queue.put(task)
    print(f"🎤 [Queue] Enqueued sentence {sentence_idx}: {clean_text[:30]}... (queue size: {tts_sentence_queue.qsize()})")


def clear_tts_sentence_queue(new_response_id: str = None):
    """
    Clear the TTS sentence queue (pending sentences only).
    Called when a new response starts or when interrupted.

    Note: This does NOT cancel the currently processing TTS task.
    The current TTS will complete normally, only queued sentences are cleared.
    """
    with tts_sentence_queue_lock:
        # Clear queue (pending sentences only, don't cancel current TTS)
        cleared = 0
        while not tts_sentence_queue.empty():
            try:
                tts_sentence_queue.get_nowait()
                cleared += 1
            except queue.Empty:
                break

        if cleared > 0:
            print(f"🗑 Cleared {cleared} pending TTS sentences (current TTS continues)")


def get_tts_sentence_task(timeout: float = 0.1):
    """
    Get a sentence task from the queue.
    Returns None if queue is empty after timeout.
    """
    try:
        return tts_sentence_queue.get(timeout=timeout)
    except queue.Empty:
        return None

def should_cancel_tts() -> bool:
    global tts_cancel_flag
    with pending_tts_lock:
        return tts_cancel_flag

def clear_cancel_flag():
    global tts_cancel_flag
    with pending_tts_lock:
        tts_cancel_flag = False

def init_tts_model(args) -> bool:
    """Initialize the TTS model using Qwen3TTSModel (Direct integration)."""
    global tts_omni, tts_enabled, tts_streaming, tts_event_loop, voice_clone_prompt_cache
    global tts_ref_audio_path, tts_ref_text

    # Store ref audio/text settings for Base model
    tts_ref_audio_path = getattr(args, 'tts_ref_audio', None)
    tts_ref_text = getattr(args, 'tts_ref_text', None)

    try:
        # Import directly from local path
        try:
            from qwen_tts import Qwen3TTSModel
        except ImportError:
            # Fallback to local import if sys.path update didn't propagate well
            sys.path.append(os.path.join(os.path.dirname(__file__), "Qwen3-TTS-streaming"))
            from qwen_tts import Qwen3TTSModel

        import torch

        # Load ref audio once
        voice_clone_prompt_cache = None

        print(f"🔊 Initializing TTS model: {args.tts_model}")

        # Configure device map to use the specific GPU
        # Check available devices
        num_devices = torch.cuda.device_count()
        print(f"ℹ️  Available CUDA devices: {num_devices}")

        target_gpu_idx = int(args.tts_gpu.split(',')[0]) if ',' in args.tts_gpu else int(args.tts_gpu)

        # If target index is out of bounds (e.g. user passed physical ID 5, but we only have 5 devices 0-4),
        # try to map it or fallback to the last device.
        if target_gpu_idx >= num_devices:
            print(f"⚠️  Target GPU index {target_gpu_idx} is out of bounds (0-{num_devices-1}).")
            # Heuristic: If vLLM uses TP=4, it likely uses 0,1,2,3. We should use 4.
            # Let's assume the user wants the last available GPU if the specified one is invalid.
            fallback_idx = num_devices - 1
            print(f"🔄 Fallback: Using last available GPU: cuda:{fallback_idx}")
            target_gpu_idx = fallback_idx

        device = f"cuda:{target_gpu_idx}"

        # Load model directly
        tts_omni = Qwen3TTSModel.from_pretrained(
            args.tts_model,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )

        # Enable optimizations
        print("🚀 Enabling streaming optimizations for TTS...")
        tts_omni.enable_streaming_optimizations(
            decode_window_frames=80,
            use_compile=True,
            compile_mode="reduce-overhead",
        )

        tts_streaming = True # Always force streaming with this backend
        tts_enabled = True
        print(f"✓ TTS model initialized on {device}")

        return True
    except Exception as e:
        print(f"Error initializing TTS model: {e}")
        import traceback
        traceback.print_exc()
        tts_enabled = False
        return False

def _text_to_speech_generator(text: str,
                         language: str = "Chinese",
                         speaker: str = "Vivian",
                         instruct: str = ""):
    """Generator implementation of text_to_speech using Qwen3TTSModel.

    For CustomVoice models: Uses non-streaming fast generation (more stable)
    For Base models: Uses true streaming with voice clone
    """
    global tts_omni

    # Check model type
    model_type = getattr(tts_omni.model, "tts_model_type", "base")

    import numpy as np

    if model_type == "custom_voice":
        # CustomVoice path - Use non-streaming fast generation
        # This is actually faster and more stable than trying to use stream_generate_pcm
        # because CustomVoice doesn't have full streaming support
        print(f"🎤 [Fast] Converting to speech (CustomVoice - {speaker}): {text[:40]}...")

        try:
            import time
            start_time = time.time()

            # Use the high-level generate_custom_voice which is optimized
            wavs, sr = tts_omni.generate_custom_voice(
                text=text,
                speaker=speaker,
                language=language if language else "Auto",
                instruct=instruct if instruct else None,
                # Faster generation params
                do_sample=True,
                temperature=0.85,
                top_k=30,
            )

            gen_time = time.time() - start_time
            audio_duration = len(wavs[0]) / sr
            rtf = gen_time / audio_duration if audio_duration > 0 else 0
            print(f"🎤 [Fast] Generated {audio_duration:.2f}s audio in {gen_time:.2f}s (RTF: {rtf:.2f})")

            # Yield the entire audio as one chunk (non-streaming)
            audio_int16 = (wavs[0] * 32767).clip(-32768, 32767).astype(np.int16)
            yield audio_int16.tobytes(), sr

        except Exception as e:
            print(f"TTS generation error (CustomVoice): {e}")
            import traceback
            traceback.print_exc()

    else:
        # Base (Voice Clone) path - True streaming with optimizations
        # Use global ref_audio/ref_text settings (set during init or from args)
        global voice_clone_prompt_cache, tts_ref_audio_path, tts_ref_text

        # Create prompt if not cached
        if voice_clone_prompt_cache is None:
            ref_audio_path = tts_ref_audio_path if tts_ref_audio_path else "kuklina-1.wav"
            ref_text = tts_ref_text if tts_ref_text else (
                "这是凯蒂的弟弟，我的同学。你的手怎么了？你为什么不穿衣服？他有很多武术奖项。"
                "凯蒂告诉过我，对吗，莱奥？你知道你打败了谁吗，莱娅？"
                "摸摸这些肌肉，我不知道你有一只这么棒的猫。生于月亮之下。"
                "莱娅总是能挖出一些奇特的东西。是的，只是可惜它占据了她几乎所有的时间。"
                "我不明白这破烂为什么不能等你和妹妹玩完再等。"
            )

            if os.path.exists(ref_audio_path):
                print(f"🎤 Creating voice clone prompt from: {ref_audio_path}")
                voice_clone_prompt_cache = tts_omni.create_voice_clone_prompt(
                    ref_audio=ref_audio_path,
                    ref_text=ref_text,
                )
            else:
                print(f"⚠️ Reference audio not found: {ref_audio_path}")
                print("⚠️ Base model needs --tts-ref-audio and --tts-ref-text!")
                voice_clone_prompt_cache = None

        print(f"🎤 [Stream] Converting to speech (VoiceClone): {text[:40]}...")

        try:
            # True streaming generation with low-latency params
            for chunk, sr in tts_omni.stream_generate_voice_clone(
                text=text,
                language=language if language in ["Chinese", "English", "Russian", "Japanese", "Korean"] else "Chinese",
                voice_clone_prompt=voice_clone_prompt_cache,
                emit_every_frames=2,  # Lower = faster first chunk (was 4)
                decode_window_frames=60,  # Smaller = faster (was 80)
                overlap_samples=256,  # Smaller overlap for speed
            ):
                # Convert to int16 PCM
                audio_int16 = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
                yield audio_int16.tobytes(), sr

        except Exception as e:
            print(f"TTS generation error: {e}")
            import traceback
            traceback.print_exc()

def merge_short_sentences(sentences: list, min_chars: int = 15) -> list:
    """Merge short sentences to reduce TTS calls and improve naturalness."""
    if not sentences:
        return sentences
    
    merged = []
    buffer = ""
    
    for s in sentences:
        if len(buffer) + len(s) < min_chars * 3:  # Allow merging up to ~45 chars
            buffer = (buffer + s) if buffer else s
        else:
            if buffer:
                merged.append(buffer)
            buffer = s
    
    if buffer:
        merged.append(buffer)
    
    return merged


def tts_worker_loop(args):
    """
    TTS worker thread loop - now uses sentence queue for streaming TTS.

    New behavior (Step 1 + Step 2):
    - Pulls individual sentences from tts_sentence_queue
    - Each sentence is processed immediately when enqueued by model generation
    - Pipeline parallelism: model generates next sentence while TTS processes current
    - TRUE STREAMING: Each audio chunk is sent immediately via Type 9 protocol
      (No more collecting all chunks before sending)

    Protocol modes:
    - Base model: Type 9 (TTS Audio Chunk) - true streaming PCM chunks
    - CustomVoice model: Type 5 (TTS Audio) - complete WAV (no true streaming available)
    """
    global current_tts_response_id

    # Check model type for optimization strategy
    model_type = "unknown"
    if tts_omni is not None:
        model_type = getattr(tts_omni.model, "tts_model_type", "base")

    # Determine if we can use chunk streaming (only Base model supports true streaming)
    use_chunk_streaming = (model_type == "base")

    print(f"🔊 TTS Worker started (Model: {model_type}, Chunk Streaming: {use_chunk_streaming})")

    import numpy as np
    import soundfile as sf
    import io
    import time as _time

    while True:
        # Get sentence task from queue (blocking with timeout)
        task = get_tts_sentence_task(timeout=0.1)

        if task is None:
            # No task available, continue waiting
            continue

        try:
            sentence_start = _time.time()
            first_chunk_sent = False

            response_id = task.get("response_id", "")
            sentence_idx = task.get("sentence_idx", 0)
            text = task.get("text", "")
            language = task.get("language", "Chinese")
            speaker = task.get("speaker", "Vivian")
            instruct = task.get("instruct", "")

            if not text.strip():
                continue

            with pending_tts_lock:
                current_tts_response_id = response_id
                clear_cancel_flag()

            print(f"🎤 [TTS] Processing sentence {sentence_idx}: {text[:40]}...")

            chunk_idx = 0
            sr = 24000
            total_samples = 0
            first_chunk_latency = 0.0

            if use_chunk_streaming:
                # ========== TRUE STREAMING MODE (Base model) ==========
                # Send each chunk immediately via Type 9 protocol
                for audio_bytes, sample_rate in _text_to_speech_generator(
                    text, language, speaker, instruct
                ):
                    if should_cancel_tts():
                        print(f"⏹ TTS cancelled for sentence {sentence_idx}")
                        break

                    sr = sample_rate
                    total_samples += len(audio_bytes) // 2  # int16 = 2 bytes per sample

                    # Send chunk immediately (not collecting!)
                    send_audio_chunk_to_client(
                        pcm_bytes=audio_bytes,
                        response_id=response_id,
                        sentence_idx=sentence_idx,
                        chunk_idx=chunk_idx,
                        sample_rate=sr,
                        is_final=False
                    )

                    if not first_chunk_sent:
                        first_chunk_latency = _time.time() - sentence_start
                        print(f"🚀 [TTS] First chunk sent in {first_chunk_latency:.3f}s")
                        first_chunk_sent = True

                    chunk_idx += 1

                # Log TTS latency metrics - always log if any chunks were processed
                # (moved outside of cancel check to avoid race condition)
                if chunk_idx > 0:
                    sentence_time = _time.time() - sentence_start
                    audio_duration = total_samples / sr if sr > 0 else 0

                    # Log latency regardless of cancel status
                    log_tts_latency(
                        response_id=response_id,
                        sentence_idx=sentence_idx,
                        text=text,
                        first_chunk_latency=first_chunk_latency,
                        total_latency=sentence_time,
                        audio_duration=audio_duration,
                        num_chunks=chunk_idx,
                        model_type=model_type
                    )

                    # Send final marker only if not cancelled
                    if not should_cancel_tts():
                        send_audio_chunk_to_client(
                            pcm_bytes=b'',  # Empty data for final marker
                            response_id=response_id,
                            sentence_idx=sentence_idx,
                            chunk_idx=chunk_idx,
                            sample_rate=sr,
                            is_final=True
                        )
                        print(f"🔊 [TTS] Sentence {sentence_idx} complete: {chunk_idx} chunks, "
                              f"{audio_duration:.1f}s audio in {sentence_time:.2f}s")
                    else:
                        print(f"⏹ [TTS] Sentence {sentence_idx} cancelled after {chunk_idx} chunks")

            else:
                # ========== LEGACY MODE (CustomVoice model) ==========
                # Collect all chunks, send as complete WAV via Type 5 protocol
                audio_parts = []

                for audio_bytes, sample_rate in _text_to_speech_generator(
                    text, language, speaker, instruct
                ):
                    if should_cancel_tts():
                        print(f"⏹ TTS cancelled for sentence {sentence_idx}")
                        break

                    # Record first chunk time for CustomVoice too
                    if not first_chunk_sent:
                        first_chunk_latency = _time.time() - sentence_start
                        first_chunk_sent = True

                    audio_parts.append(np.frombuffer(audio_bytes, dtype=np.int16))
                    sr = sample_rate

                # Log TTS latency metrics - always log if any chunks were processed
                # (moved outside of cancel check to avoid race condition)
                if audio_parts:
                    full_audio = np.concatenate(audio_parts)
                    sentence_time = _time.time() - sentence_start
                    audio_duration = len(full_audio) / sr

                    # Log latency regardless of cancel status
                    log_tts_latency(
                        response_id=response_id,
                        sentence_idx=sentence_idx,
                        text=text,
                        first_chunk_latency=first_chunk_latency,
                        total_latency=sentence_time,
                        audio_duration=audio_duration,
                        num_chunks=len(audio_parts),
                        model_type=model_type
                    )

                    # Send complete WAV only if not cancelled
                    if not should_cancel_tts():
                        # Write to WAV buffer
                        wav_buffer = io.BytesIO()
                        sf.write(wav_buffer, full_audio, sr, format='WAV')
                        wav_bytes = wav_buffer.getvalue()

                        # Send complete WAV
                        estimated_total = sentence_idx + 1
                        send_audio_to_client(wav_bytes, response_id, sentence_idx, estimated_total)
                        print(f"🔊 [TTS] Sent sentence {sentence_idx} ({audio_duration:.1f}s WAV in {sentence_time:.2f}s)")
                    else:
                        print(f"⏹ [TTS] Sentence {sentence_idx} cancelled after {len(audio_parts)} chunks")

            with pending_tts_lock:
                current_tts_response_id = None

        except Exception as e:
            print(f"TTS Worker error: {e}")
            import traceback
            traceback.print_exc()
            import traceback
            traceback.print_exc()

def start_tts_worker(args):
    """Start the TTS worker thread."""
    global tts_worker_running
    if not tts_worker_running:
        tts_worker_running = True
        t = threading.Thread(target=tts_worker_loop, args=(args,), daemon=True)
        t.start()

# ============================================================================
# AsyncLLM Engine Management
# ============================================================================

async def init_async_engine(args) -> AsyncLLM:
    """Initialize the AsyncLLM engine with streaming support."""
    global async_engine

    # Build engine args dict, only include non-None values
    engine_kwargs = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "max_model_len": args.max_model_len,
        "trust_remote_code": detect_model_type(args.model) == "omni",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": args.enforce_eager,
        "limit_mm_per_prompt": {"image": args.max_images_per_prompt},
        "enable_expert_parallel": args.enable_expert_parallel
    }

    # Add optional advanced parameters if specified
    if args.kv_offloading_size is not None:
        engine_kwargs["kv_offloading_size"] = args.kv_offloading_size
    if args.mm_encoder_attn_backend is not None:
        engine_kwargs["mm_encoder_attn_backend"] = args.mm_encoder_attn_backend
    if args.mm_encoder_tp_mode is not None:
        engine_kwargs["mm_encoder_tp_mode"] = args.mm_encoder_tp_mode
    if args.disable_hybrid_kv_cache_manager:
        engine_kwargs["disable_hybrid_kv_cache_manager"] = True
    if args.block_size is not None:
        engine_kwargs["block_size"] = args.block_size
    if args.cache_dtype is not None:
        engine_kwargs["kv_cache_dtype"] = args.cache_dtype
    if args.prefix_caching_hash_algo is not None:
        engine_kwargs["prefix_caching_hash_algo"] = args.prefix_caching_hash_algo
    if args.max_num_batched_tokens is not None:
        engine_kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens

    engine_kwargs["cudagraph_capture_sizes"] = [1,2,4]

    engine_args = AsyncEngineArgs(**engine_kwargs)

    model_label = "Qwen3 Omni" if detect_model_type(args.model) == "omni" else "Qwen3 VL"
    print(f"🚀 Initializing {model_label} AsyncLLM engine with model: {args.model}")
    print(f"   trust_remote_code={engine_kwargs['trust_remote_code']}, SILENT_TOKEN_ID={SILENT_TOKEN_ID}")
    async_engine = AsyncLLM.from_engine_args(engine_args)
    print(f"✅ {model_label} AsyncLLM engine initialized successfully")

    # ========== DEBUG: 保存词表到日志文件 ==========
    try:
        tokenizer = async_engine.get_tokenizer()
        vocab = tokenizer.get_vocab()  # Dict[str, int]: token_str -> token_id

        vocab_log_path = "vocab_debug.log"
        with open(vocab_log_path, "w", encoding="utf-8") as f:
            f.write(f"# Vocabulary Debug Log\n")
            f.write(f"# Model: {args.model}\n")
            f.write(f"# Vocab Size: {len(vocab)}\n")
            f.write(f"# Format: token_id | token_str | repr(token_str)\n")
            f.write("=" * 80 + "\n\n")

            # 按 token_id 排序输出
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            for token_str, token_id in sorted_vocab:
                # 保持格式：ID | 原始字符串 | repr表示（显示转义字符）
                f.write(f"{token_id:>8} | {token_str:<40} | {repr(token_str)}\n")

        print(f"📝 Vocabulary saved to {vocab_log_path} ({len(vocab)} tokens)")
    except Exception as e:
        print(f"⚠️ Failed to save vocabulary: {e}")
    # ========== END DEBUG ==========

    # Note: No need to load transformers processor!
    # vLLM handles multimodal processing internally.
    # We use the pattern from test_qwen2_5_vl.py:
    # - Build prompt string with placeholders
    # - Pass images via multi_modal_data

    return async_engine


def summarize_vllm_inputs(inputs: dict) -> dict:
    """
    Summarize vLLM inputs for logging, replacing video arrays and images with metadata.
    This avoids printing huge pixel arrays in logs.
    """
    summary = {"prompt": inputs.get("prompt", "")}
    mm_data = inputs.get("multi_modal_data", {})
    if mm_data:
        summary["multi_modal_data"] = {}
        for key, value in mm_data.items():
            if key == "video" and isinstance(value, list):
                # Replace video tuples with just metadata
                summary["multi_modal_data"]["video"] = [
                    {"shape": v[0].shape, "dtype": str(v[0].dtype), "metadata": v[1]}
                    if isinstance(v, tuple) else f"<video {type(v)}>"
                    for v in value
                ]
            elif key == "image" and isinstance(value, list):
                summary["multi_modal_data"]["image"] = [
                    f"<{type(v).__name__} {v.size if hasattr(v, 'size') else ''}>"
                    for v in value
                ]
            else:
                summary["multi_modal_data"][key] = value
    return summary


async def generate_response_with_video(
    session: StreamingSession,
    video_tuple: tuple,
    prompt: str,
    sampling_params: SamplingParams,
    args = None,
):
    """
    Generate response for a single turn using video input.
    Qwen3-VL expects video as (numpy_array, metadata_dict) tuple.

    Args:
        video_tuple: Tuple of (numpy_array, metadata_dict)
                    numpy_array shape: (num_frames, height, width, 3)
                    metadata_dict: {"fps": float, "duration": float, ...}
    """
    print(f"==== Calling generate_response_with_video() ====")
    global async_engine

    if async_engine is None:
        raise RuntimeError("AsyncLLM engine not initialized")

    if video_tuple is None or video_tuple[0] is None:
        print("⚠️ No valid video for generation")
        session.is_generating = False
        return

    # Qwen3-VL requires at least 2 frames (temporal_factor=2)
    # Defensive check: if only 1 frame, duplicate it
    video_array_check = video_tuple[0]
    if video_array_check.shape[0] < 2:
        import numpy as np
        print(f"⚠️ Video has only {video_array_check.shape[0]} frame(s), duplicating to meet Qwen3-VL minimum (2 frames)")
        duplicated_array = np.concatenate([video_array_check] * 2, axis=0)[:2]
        video_metadata = video_tuple[1].copy() if video_tuple[1] else {}
        video_metadata["total_num_frames"] = 2
        video_metadata["duration"] = 2 / video_metadata.get("fps", 2.0)
        video_tuple = (duplicated_array, video_metadata)

    try:
        # Add current turn to history with video tuple
        session.history.add_user_message(prompt, video_tuple=video_tuple)

        # Get vLLM inputs (with history for Prefix Caching)
        # t_get_inputs_start = time.time()
        vllm_inputs = session.history.get_vllm_inputs()
        # t_get_inputs_end = time.time()
        # print(f"⏱️ [TIMING] get_vllm_inputs() took {(t_get_inputs_end - t_get_inputs_start)*1000:.1f}ms")

        # summarized_vllm_inputs = summarize_vllm_inputs(vllm_inputs)
        # print(f"summarized_vllm_inputs.keys(): {summarized_vllm_inputs.keys()}")
        # print(f"len(summarized_vllm_inputs['multi_modal_data']['video']): {len(summarized_vllm_inputs['multi_modal_data']['video'])}")

        # print(f"🎬 [Session {session.session_id}] vLLM inputs: {summarize_vllm_inputs(vllm_inputs)}, ================== ")

        # Generate unique request ID for this turn
        request_id = generate_response_id()

        video_array, video_metadata = video_tuple
        print(f"🎬 [Session {session.session_id}] Starting generation (request_id={request_id})")
        print(f"📥 Input: {video_array.shape[0]} video frames ({video_array.shape}), prompt='{prompt}'")

        full_response = ""
        previous_text = ""
        is_silent_response = False  # Flag to detect <|silent|> response
        is_first_token = True  # Flag to send query with first token

        # Streaming TTS: sentence buffer and counter
        tts_enabled_for_this_response = args and args.enable_tts and tts_enabled
        sentence_buffer = ""
        sentence_idx = 0

        # Only clear TTS queue for user-initiated responses (with a real prompt).
        # Background auto-generations (empty prompt) should NOT clear the queue,
        # otherwise pending sentences from the previous response get dropped
        # before the TTS worker can process them.
        if tts_enabled_for_this_response and prompt:
            clear_tts_sentence_queue(new_response_id=request_id)

        # Timing measurements
        generation_start_time = time.time()
        first_token_time = None
        token_count = 0
        print(f"[TTFT_DEBUG] stream generate_start request_id={request_id} t={generation_start_time:.6f}")

        # Generate with video
        async for response in async_engine.generate(
            prompt=vllm_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            # Record time to first token
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - generation_start_time
                print(f"[TTFT_DEBUG] stream first_token request_id={request_id} t={first_token_time:.6f} ttft_ms={ttft*1000:.1f}")

            token_count += 1
            print(f"🔍 [DEBUG-VIDEO] Got response, outputs count: {len(response.outputs) if response.outputs else 0}")
            if response.outputs:
                output = response.outputs[0]
                # 查看 token IDs（解码前的数字 ID 列表）
                print(f"🔢 [VIDEO] Token IDs: {output.token_ids}")
                print(f"🔢 [VIDEO] Token IDs length: {len(output.token_ids)}")

                # Check for silent/empty response: only look at the FIRST token.
                # In vLLM streaming, token_ids is cumulative and ordered,
                # so token_ids[0] is always the model's first generated token.
                # If the first token is <|silent|> or <|im_end|>, the model
                # chose not to respond — treat as silent and don't send to user.
                if len(output.token_ids) > 0 and output.token_ids[0] in (SILENT_TOKEN_ID, IM_END_TOKEN_ID):
                    is_silent_response = True
                    first_tid = output.token_ids[0]
                    tag = "SILENT" if first_tid == SILENT_TOKEN_ID else "IM_END"
                    print(f"🔇 [Session {session.session_id}] {tag} as first token → silent response "
                          f"(first_token_id={first_tid}, token_ids={list(output.token_ids[:5])}...)")
                    break  # Stop generation early

                if hasattr(output, 'text') and output.text:
                    current_text = output.text
                    if len(current_text) > len(previous_text):
                        delta = current_text[len(previous_text):]
                        previous_text = current_text
                        print(f"🤖 [Session {session.session_id}] Delta: {delta!r}")

                        # Send first token with query (ASR result)
                        if is_first_token:
                            send_streaming_token_to_client(delta, request_id, query=prompt, is_start=True)
                            is_first_token = False
                        else:
                            send_streaming_token_to_client(delta, request_id)
                        full_response += delta

                        # ========== Streaming TTS: Sentence boundary detection ==========
                        if tts_enabled_for_this_response:
                            sentence_buffer += delta

                            # Check if delta contains sentence terminator
                            for terminator in SENTENCE_TERMINATORS_SET:
                                if terminator in delta:
                                    # Find the last terminator position
                                    last_term_pos = -1
                                    for t in SENTENCE_TERMINATORS_SET:
                                        pos = sentence_buffer.rfind(t)
                                        if pos > last_term_pos:
                                            last_term_pos = pos

                                    if last_term_pos >= 0:
                                        # Extract complete sentence(s) up to and including the terminator
                                        complete_part = sentence_buffer[:last_term_pos + 1]
                                        remaining_part = sentence_buffer[last_term_pos + 1:]

                                        # Enqueue the complete sentence for TTS
                                        if complete_part.strip():
                                            enqueue_tts_sentence(complete_part, request_id, sentence_idx, args)
                                            sentence_idx += 1

                                        # Keep the remaining part for next sentence
                                        sentence_buffer = remaining_part
                                    break  # Only need to check once per delta
                        # ========== End Streaming TTS ==========

        # Finished
        # print timing information
        generation_end_time = time.time()
        total_time = generation_end_time - generation_start_time

        print(f"✅ [Session {session.session_id}] Generation finished")
        print(f"⏱️ [TIMING] Time to first token (TTFT): {ttft*1000:.1f}ms, timestamp: {time.time()}")
        print(f"⏱️ [TIMING] TTFT avg. by {video_array.shape[0]} frames: {(ttft*1000/video_array.shape[0]):.1f}ms")
        print(f"⏱️ [TIMING] Total generation time: {total_time*1000:.1f}ms")
        print(f"⏱️ [TIMING] Tokens generated: {token_count}")

        if is_silent_response:
            # Silent mode: add to context AND send silent marker to client
            print(f"🔇 [Session {session.session_id}] Silent mode - adding to history and notifying client")
            session.history.add_assistant_message(SILENT_TEXT)
            # Send silent marker to client with is_final=True so client knows response is complete
            # Use is_silent=True flag to tell client this is a silent response (not empty string)
            send_streaming_token_to_client(SILENT_TEXT, request_id, is_final=True, is_silent=True)
            # Don't trigger TTS for silent responses
        else:
            # Normal mode: send final token marker (response already streamed)
            send_streaming_token_to_client("", request_id, is_final=True)
            # NOTE: Removed send_response_to_client() - response already sent via streaming tokens

            # Update History with Assistant Response
            session.history.add_assistant_message(full_response)

            # ========== Streaming TTS: Handle remaining sentence buffer ==========
            if tts_enabled_for_this_response and sentence_buffer.strip():
                # Enqueue the final incomplete sentence
                enqueue_tts_sentence(sentence_buffer, request_id, sentence_idx, args)
                print(f"🎤 [Queue] Final sentence enqueued, total: {sentence_idx + 1} sentences")
            # ========== End Streaming TTS ==========

    except asyncio.CancelledError:
        print(f"⏹ [Session {session.session_id}] Generation cancelled (context reset)")
    except Exception as e:
        print(f"❌ [Session {session.session_id}] Generation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # CRITICAL: Always reset the generating flags
        session.is_generating = False
        session.is_auto_generating = False
        session.current_task = None
        print(f"🔓 [Session {session.session_id}] Released generation lock")


# ============================================================================
# Socket Server for Client Communication
# ============================================================================

def send_streaming_token_to_client(token: str, response_id: str, is_final: bool = False,
                                     query: str = None, is_start: bool = False,
                                     is_silent: bool = False):
    """Send a streaming token to the client.

    Protocol: Type 8 (STREAMING_TOKEN)
    - Type 7 is reserved for ERROR messages in the client

    Args:
        token: The token text to send
        response_id: Unique response ID
        is_final: Whether this is the final token
        query: User query (ASR result) - only sent at start
        is_start: Whether this is the start of a new response
        is_silent: Whether this is a silent response (model chose not to respond)
    """
    with connection_lock:
        if active_connection:
            try:
                response_data = {
                    "response_id": response_id,
                    "token": token,
                    "is_final": is_final,
                    "type": "streaming_token"
                }
                # Include query at start of response
                if is_start and query:
                    response_data["query"] = query
                    response_data["is_start"] = True

                # Mark silent responses so client can handle them appropriately
                if is_silent:
                    response_data["is_silent"] = True

                payload = json.dumps(response_data, ensure_ascii=False).encode('utf-8')
                # Use type 8 for streaming tokens (type 7 is ERROR in client)
                header = struct.pack(">BQ", 8, len(payload))
                active_connection.sendall(header + payload)
            except Exception as e:
                print(f"Error sending streaming token: {e}")


def send_audio_to_client(audio_bytes: bytes, response_id: str = None,
                         sentence_idx: int = 0, total_sentences: int = 1):
    """Send audio data to the connected client.

    Protocol: Type 5 (TTS Audio) - Complete WAV file per sentence
    """
    with connection_lock:
        if active_connection:
            try:
                response_id_bytes = (response_id or "").encode('utf-8')
                response_id_len = len(response_id_bytes)

                # Protocol: Type 5 | length | response_id_len | response_id | sentence_idx | total_sentences | audio_data
                payload = (
                    struct.pack(">B", response_id_len) +
                    response_id_bytes +
                    struct.pack(">HH", sentence_idx, total_sentences) +
                    audio_bytes
                )
                header = struct.pack(">BQ", 5, len(payload))
                active_connection.sendall(header + payload)
                print(f"🔊 Sent TTS sentence {sentence_idx + 1}/{total_sentences} ({len(audio_bytes)} bytes)")
            except Exception as e:
                print(f"Error sending audio: {e}")


def send_audio_chunk_to_client(pcm_bytes: bytes, response_id: str,
                                sentence_idx: int, chunk_idx: int,
                                sample_rate: int, is_final: bool = False):
    """Send a streaming audio chunk to the client (Raw PCM int16).

    Protocol: Type 9 (TTS Audio Chunk)

    This enables true streaming TTS - each chunk is sent as soon as it's generated,
    allowing the client to start playback before the entire sentence is synthesized.

    Payload format:
    - response_id_len (1 byte)
    - response_id (variable)
    - sentence_idx (2 bytes, big-endian)
    - chunk_idx (2 bytes, big-endian)
    - sample_rate (4 bytes, big-endian)
    - is_final (1 byte: 0 or 1)
    - pcm_data (Raw int16 PCM)
    """
    with connection_lock:
        if active_connection:
            try:
                response_id_bytes = (response_id or "").encode('utf-8')
                response_id_len = len(response_id_bytes)

                payload = (
                    struct.pack(">B", response_id_len) +
                    response_id_bytes +
                    struct.pack(">HHIB", sentence_idx, chunk_idx, sample_rate, 1 if is_final else 0) +
                    pcm_bytes
                )
                header = struct.pack(">BQ", 9, len(payload))  # Type 9 for audio chunks
                active_connection.sendall(header + payload)

                if is_final:
                    print(f"🔊 [Chunk] Sent final chunk for sentence {sentence_idx} (chunk {chunk_idx})")
            except Exception as e:
                print(f"Error sending audio chunk: {e}")



def recv_exactly(conn, n: int, timeout: float = 30.0) -> bytes:
    """Receive exactly n bytes from socket, blocking."""
    conn.settimeout(timeout)
    data = b""
    while len(data) < n:
        try:
            chunk = conn.recv(n - len(data))
            if not chunk:
                raise ConnectionError("Connection closed")
            data += chunk
        except socket.timeout:
            raise TimeoutError(f"Timeout waiting for {n} bytes")
    return data


async def handle_client_connection_async(conn, addr, args):
    """Handle client connection with async support."""
    global active_connection

    print(f"================================================")
    print(f"✅ Connected by {addr} with SUYI")

    with connection_lock:
        active_connection = conn

    # Set socket to blocking mode with timeout
    conn.setblocking(True)
    conn.settimeout(1.0)  # 1 second timeout for initial reads

    # Create a streaming session for this client
    session_id = f"client-{addr[0]}-{addr[1]}-{int(time.time())}"

    # Session state with History
    session = StreamingSession(
        session_id=session_id,
        history=SessionHistory(
            max_rounds=args.max_rounds,
            num_rounds_keep=args.num_rounds_keep,
            pruning_enabled=args.enable_pruning
        ),
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        is_generating=False
    )

    async with session_lock:
        streaming_sessions[session_id] = session

    # Generation task tracking (not used for long-running loop anymore)
    generation_task = None
    accumulated_video_frames: list = []  # List of numpy arrays (each shape: num_frames, H, W, 3)
    last_prompt = ""

    try:
        while True:
            # Read header: [Type 1 byte] [Length 8 bytes] = 9 bytes total
            try:
                header = await asyncio.get_event_loop().run_in_executor(
                    None, recv_exactly, conn, 9, 5.0
                )
            except TimeoutError:
                # No data received, continue waiting
                continue
            except ConnectionError:
                print("🔌 Client disconnected")
                break
            except Exception as e:
                print(f"❌ Header read error: {e}")
                break

            file_type, file_len = struct.unpack(">BQ", header)
            print(f"📩 Received: type={file_type}, length={file_len}, time={datetime.now().strftime('%H:%M:%S.%f')}")

            # Sanity check for length (prevent memory issues)
            if file_len > 100 * 1024 * 1024:  # 100MB max
                print(f"⚠ Invalid length {file_len}, skipping message")
                continue

            # Read file content
            try:
                file_data = await asyncio.get_event_loop().run_in_executor(
                    None, recv_exactly, conn, file_len, 30.0
                )
            except TimeoutError:
                print(f"⚠ Timeout reading {file_len} bytes, skipping")
                continue
            except ConnectionError:
                print("🔌 Client disconnected during data read")
                break
            except Exception as e:
                print(f"❌ Data read error: {e}")
                break

            # Yield immediately so other tasks (e.g. generate() waiting for first token) can run.
            # Avoids event loop starvation that causes ~800ms TTFT delay.
            await asyncio.sleep(0)

            if file_type == 1:  # Video (WebM)
                # Sanity check: WebM files need a minimum size for valid EBML header
                # A valid WebM file is typically at least 1KB even for short clips
                MIN_WEBM_SIZE = 1000  # 1KB minimum
                if len(file_data) < MIN_WEBM_SIZE:
                    print(f"⚠️ Video data too small ({len(file_data)} bytes < {MIN_WEBM_SIZE}), skipping corrupted/incomplete data")
                    continue

                # Downsample video to target FPS and get as numpy array with metadata
                print("🎥 Processing video data...")
                timestamp = int(time.time() * 1000)
                input_path = f"/tmp/video_{timestamp}_input.webm"
                
                with open(input_path, "wb") as f:
                    f.write(file_data)
                
                # Downsample video and get (numpy_array, metadata) tuple
                video_array, metadata = downsample_video_to_numpy(input_path, target_fps=args.target_fps)
                
                # Clean up input file
                try:
                    os.remove(input_path)
                except:
                    pass

                # # Run video decode in executor to avoid blocking the event loop (prevents TTFT starvation).
                # print("🎥 Processing video data...")
                # timestamp = int(time.time() * 1000)
                # input_path = f"/tmp/video_{timestamp}_input.webm"
                # print("New......")
                # loop = asyncio.get_event_loop()
                # video_array, metadata = await loop.run_in_executor(
                #     None,
                #     _decode_video_sync,
                #     file_data,
                #     input_path,
                #     args.target_fps,
                #     not args.no_video_resize,
                # )

                if video_array is not None:
                    # Accumulate video frames (as numpy arrays)
                    accumulated_video_frames.append(video_array)
                    total_frames = sum(arr.shape[0] for arr in accumulated_video_frames)
                    print(f"📹 Got {video_array.shape[0]} frames, total accumulated: {total_frames}")

                    # Process when we have frames OR if we have a pending prompt
                    should_process = False

                    # Priority trigger: Pending user prompt
                    if last_prompt and total_frames > 0:
                        print(f"⚡ Triggering immediate generation for user prompt (frames={total_frames})")
                        should_process = True
                    # Background trigger: Have enough frames AND idle
                    elif total_frames >= 2 and not session.is_generating:
                        print(f"⚡ Triggering background generation (frames={total_frames})")
                        should_process = True

                    if should_process:
                        if session.is_generating and last_prompt:
                             print("⏳ Waiting for previous generation to finish before processing prompt...")
                             pass

                        if not session.is_generating:
                            import numpy as np

                            # Concatenate all accumulated frames
                            all_frames = np.concatenate(accumulated_video_frames, axis=0)

                            # Qwen3-VL requires at least 2 frames (temporal_factor=2)
                            # If we only have 1 frame, duplicate it to meet the minimum requirement
                            if all_frames.shape[0] == 1:
                                print(f"⚠️ Only 1 frame, duplicating to meet Qwen3-VL minimum requirement (2 frames)")
                                all_frames = np.concatenate([all_frames, all_frames], axis=0)

                            # Limit to max 16 frames to avoid OOM
                            if all_frames.shape[0] > 16:
                                all_frames = all_frames[-16:]

                            # Create metadata for the combined video
                            video_metadata = {
                                "fps": args.target_fps,
                                "duration": all_frames.shape[0] / args.target_fps,
                                "total_num_frames": all_frames.shape[0],
                                "frames_indices": list(range(all_frames.shape[0])),
                                "video_backend": "opencv",
                                "do_sample_frames": False,
                            }
                            video_tuple = (all_frames, video_metadata)

                            accumulated_video_frames = []

                            # Mark as generating IMMEDIATELY to prevent double trigger
                            session.is_generating = True

                            # Default prompt if none
                            current_prompt = last_prompt if last_prompt else ""
                            last_prompt = ""  # Clear prompt after using

                            # Track if this is an auto-generation (no user prompt)
                            session.is_auto_generating = (current_prompt == "")

                            sampling_params = SamplingParams(
                                temperature=0.9,
                                max_tokens=512,
                            )

                            # Launch background task with video tuple
                            session.current_task = asyncio.create_task(generate_response_with_video(
                                session,
                                video_tuple,
                                current_prompt,
                                sampling_params,
                                args
                            ))
                else:
                    print("❌ Video processing failed - no frames extracted (possible codec incompatibility with iOS Chrome)")

            elif file_type == 2:  # Audio
                # Save audio for ASR
                audio_path = os.path.join(AUDIO_DIR, "latest.mp3")
                os.makedirs(AUDIO_DIR, exist_ok=True)
                with open(audio_path, "wb") as f:
                    f.write(file_data)
                print(f"🎤 Saved audio to {audio_path}")

                # Call ASR service to transcribe audio
                if args.asr_sync:
                    # Synchronous version
                    loop = asyncio.get_event_loop()
                    transcribed_text = await loop.run_in_executor(
                        None, get_audio_prompt, audio_path, args.asr_url
                    )
                else:
                    # Asynchronous version (default)
                    transcribed_text = await transcribe_audio_async(audio_path, args.asr_url)

                if transcribed_text:
                    last_prompt = transcribed_text
                    print(f"📝 Set prompt from ASR: {last_prompt[:50]}...")

                    # NOTE: ASR result (query) is now sent with the first streaming token
                    # No need to send separately via send_response_to_client()

                    # Optimization: Try to trigger immediately if we have ANY video frames
                    if accumulated_video_frames:
                        print("🚀 Audio arrived, attempting immediate trigger...")
                        if session.is_generating and session.is_auto_generating:
                            if session.current_task and not session.current_task.done():
                                print("🛑 Interrupting auto-generation for user prompt (from Audio event)!")
                                session.current_task.cancel()
                else:
                    print("⚠ ASR returned empty, will use default prompt")

                # Note: Generation is triggered by Video frame loop when frames arrive
                # If frames are already there, we could trigger here, but to avoid race conditions
                # we let the video loop handle it.

            elif file_type == 4:  # Clear Context
                print("🗑 Clearing context...")
                # Cancel any running generation task first
                if session.current_task and not session.current_task.done():
                    print("⏹ Cancelling running generation task...")
                    session.current_task.cancel()
                    session.is_generating = False
                    session.is_auto_generating = False
                context_manage.clear_global_history()
                session.history._reset() # Reset session history
                accumulated_video_frames = []
                last_prompt = ""

            elif file_type == 6:  # Start Camera
                print("📷 Camera started, resetting state...")
                # Cancel any running generation task first
                if session.current_task and not session.current_task.done():
                    print("⏹ Cancelling running generation task...")
                    session.current_task.cancel()
                    session.is_generating = False
                    session.is_auto_generating = False
                context_manage.clear_global_history()
                session.history._reset()
                accumulated_video_frames = []
                last_prompt = ""

    except Exception as e:
        print(f"❌ Connection error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print(f"👋 Connection closed by {addr}")

        # Cleanup
        # No streaming input to stop

        async with session_lock:
            if session_id in streaming_sessions:
                del streaming_sessions[session_id]

        with connection_lock:
            if active_connection == conn:
                active_connection = None
        conn.close()


async def run_accept_loop(server_sock, args):
    """
    Run accept loop in the same event loop as the engine.
    This ensures handle_client_connection_async and engine.generate() share one loop,
    fixing TTFT delay caused by two separate loops (output put in one loop, generate() awaiting in another).
    """
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(AUDIO_DIR, exist_ok=True)
    loop = asyncio.get_event_loop()
    while True:
        conn, addr = await loop.run_in_executor(None, server_sock.accept)
        asyncio.create_task(handle_client_connection_async(conn, addr, args))


def _create_listen_socket(port: int):
    """Create, bind and listen on a TCP socket. Call from main thread before starting accept loop."""
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind(("0.0.0.0", port))
    server_sock.listen(1)
    return server_sock


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3 omni Streaming Input Server")

    # Server configuration
    parser.add_argument("--host", type=str, default="localhost",
                        help="vLLM API server host (for HTTP mode)")
    parser.add_argument("--port", type=int, default=8000,
                        help="vLLM API server port (for HTTP mode)")
    parser.add_argument("--listen-port", type=int, default=12345,
                        help="Port to listen for client connections")

    # Model configuration
    parser.add_argument("--model", type=str,
                        default="/home/dyvm6xra/dyvm6xrauser36/Projects/streaming_video_understanding/qwen3omni-30b_a3b_20260128_01",
                        help="Model name or path (Qwen3 Omni)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for tensor parallelism")
    parser.add_argument("--pipeline-parallel-size", type=int, default=1,
                        help="Number of GPUs for pipeline parallelism")
    parser.add_argument("--max-model-len", type=int, default=256*1024,
                        help="Maximum model context length")
    parser.add_argument("--max-seq-len", type=int, default=256*1024,
                        help="Maximum sequence length (256k tokens)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                        help="GPU memory utilization")
    parser.add_argument("--enforce-eager", action="store_true",
                        help="Enforce eager execution (disable CUDA graphs)")
    parser.add_argument("--max-images-per-prompt", type=int, default=600,
                        help="Maximum images per prompt (vLLM engine limit)")
    parser.add_argument("--max-streaming-images", type=int, default=600,
                        help="Maximum images for streaming input session")
    parser.add_argument("--max-num-batched-tokens", type=int, default=None,
                        help="Maximum number of batched tokens (SchedulerConfig)")
    parser.add_argument("--enable-expert-parallel", action="store_true", default=False,
                        help="Enable expert parallel (default: False)")

    # Advanced vLLM engine options
    parser.add_argument("--kv-offloading-size", type=int, default=None,
                        help="KV cache offloading size (in GB)")
    parser.add_argument("--mm-encoder-attn-backend", type=str, default=None,
                        choices=["FLASH_ATTN", "XFORMERS", "TORCH_SDPA"],
                        help="Multimodal encoder attention backend")
    parser.add_argument("--mm-encoder-tp-mode", type=str, default=None,
                        choices=["data", "model"],
                        help="Multimodal encoder tensor parallelism mode")
    parser.add_argument("--disable-hybrid-kv-cache-manager", action="store_true",
                        help="Disable hybrid KV cache manager")
    parser.add_argument("--block-size", type=int, default=None,
                        choices=[1, 8, 16, 32, 64, 128, 256],
                        help="KV cache block size in tokens (vLLM CacheConfig)")
    parser.add_argument("--cache-dtype", type=str, default="auto",
                        choices=["auto", "fp8"],
                        help="Cache dtype (vLLM CacheConfig)")
    parser.add_argument("--prefix-caching-hash-algo", type=str, default=None,
                        choices=["sha256", "sha256_cbor", "xxhash", "xxhash_cbor"],
                        help="Hash algorithm for prefix caching (vLLM CacheConfig)")

    # Mode selection
    parser.add_argument("--use-http-api", action="store_true",
                        help="Use HTTP API instead of embedded engine")

    # ASR configuration
    parser.add_argument("--asr-url", type=str, default="http://localhost:8001/asr",
                        help="ASR service URL")
    parser.add_argument("--asr-sync", action="store_true",
                        help="Use synchronous ASR (default: async). Use sync mode if async has issues.")

    # Streaming configuration
    parser.add_argument("--frame-buffer-size", type=int, default=8,
                        help="Number of frames to buffer before processing")
    parser.add_argument("--stream-interval", type=float, default=0.5,
                        help="Interval between streaming inputs (seconds)")
    parser.add_argument("--target-fps", type=float, default=2.0,
                        help="Target FPS for frame extraction (default: 2.0)")
    parser.add_argument("--min-yield-interval", type=float, default=3.0,
                        help="Minimum seconds between yielding StreamingInputs (throttling)")
    parser.add_argument("--video-resize", action="store_true",
                        help="Enable video frame resize (use full resolution, slower TTFT)")
    parser.add_argument("--enable-pruning", action="store_true",
                        help="Enable video frame pruning")
    parser.add_argument("--max-rounds", type=int, default=60,
                        help="Maximum number of rounds to keep in history")
    parser.add_argument("--num-rounds-keep", type=int, default=15,
                        help="Number of rounds to keep in history")

    # TTS configuration
    parser.add_argument("--enable-tts", action="store_true", help="Enable TTS")
    parser.add_argument("--tts-model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                        help="TTS model path")
    parser.add_argument("--tts-speaker", type=str, default="Vivian",
                        help="TTS speaker name")
    parser.add_argument("--tts-language", type=str, default="Chinese", choices=["Chinese", "English"],
                        help="TTS language")
    parser.add_argument("--tts-instruct", type=str, default="",
                        help="TTS instruction")
    parser.add_argument("--tts-output-dir", type=str, default="tts_results",
                        help="TTS output directory")
    parser.add_argument("--tts-stage-configs", type=str, default=None,
                        help="TTS stage configs path")
    parser.add_argument("--tts-stage-timeout", type=int, default=300,
                        help="TTS stage init timeout")
    parser.add_argument("--tts-gpu", type=str, default="1",
                        help="TTS GPU ID")
    parser.add_argument("--tts-streaming", action="store_true",
                        help="Enable TTS streaming mode")
    # Base model voice clone settings
    parser.add_argument("--tts-ref-audio", type=str, default=None,
                        help="Reference audio file for Base model voice cloning (5-15 seconds)")
    parser.add_argument("--tts-ref-text", type=str, default=None,
                        help="Reference text matching the audio content")

    return parser.parse_args()


async def main_async(args):
    """Main async entry point."""
    # 根据 model path 动态设置 SILENT_TOKEN_ID
    setup_silent_token_id(args.model)

    model_type = detect_model_type(args.model)
    print("=" * 60)
    print(f"Qwen3 {'Omni' if model_type == 'omni' else 'VL'} Streaming Input Server")
    print("=" * 60)
    print(f"Mode: {'HTTP API' if args.use_http_api else 'Embedded Engine'}")
    print(f"Model: {args.model} (type: {model_type})")
    print(f"Listen Port: {args.listen_port}")
    print("-" * 60)
    print("Streaming Configuration:")
    print(f"  Target FPS: {args.target_fps} (frames extracted per second)")
    print(f"  Max Streaming Images: {args.max_streaming_images}")
    print(f"  Max Images Per Prompt: {args.max_images_per_prompt}")
    print(f"  Min Yield Interval: {args.min_yield_interval}s (throttling)")
    print(f"  Note: Client records at 15fps, server extracts at {args.target_fps}fps")
    print(f"  Capacity: ~{int(args.max_streaming_images / args.target_fps / 60)} minutes of video")
    print(f"  Video resize: {'ENABLED (1/8 resolution)' if args.video_resize else 'DISABLED (full res)'}")
    print(f"  Enable pruning: {'ENABLED' if args.enable_pruning else 'DISABLED'}")
    print(f"  Num rounds keep: {args.num_rounds_keep}")
    print(f"  Max rounds: {args.max_rounds}")
    print(f"  Enable expert parallel: {'ENABLED' if args.enable_expert_parallel else 'DISABLED'}")
    if args.kv_offloading_size is not None:
        print(f"  KV Offloading Size: {args.kv_offloading_size} GB")
    if args.mm_encoder_attn_backend is not None:
        print(f"  MM Encoder Attention Backend: {args.mm_encoder_attn_backend}")
    if args.mm_encoder_tp_mode is not None:
        print(f"  MM Encoder TP Mode: {args.mm_encoder_tp_mode}")
    if args.disable_hybrid_kv_cache_manager:
        print(f"  Hybrid KV Cache Manager: DISABLED")
    if args.block_size is not None:
        print(f"  Block size: {args.block_size}")
    if args.cache_dtype is not None:
        print(f"  Cache dtype: {args.cache_dtype}")
    if args.prefix_caching_hash_algo is not None:
        print(f"  Prefix caching hash algo: {args.prefix_caching_hash_algo}")
    if args.max_num_batched_tokens is not None:
        print(f"  Max num batched tokens: {args.max_num_batched_tokens}")
    if args.enable_tts:
        print(f"  TTS Enabled: {args.tts_model} (GPU {args.tts_gpu})")
    print("=" * 60)

    # Initialize TTS if enabled
    if args.enable_tts:
        if init_tts_model(args):
            start_tts_worker(args)
        else:
            print("⚠ TTS initialization failed, TTS will be disabled")
    # print("⚠ TTS initialization failed, TTS will be disabled")

    if not args.use_http_api:
        # Warning about GPU conflict between TTS and vLLM
        if args.enable_tts:
            current_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if current_visible:
                # Parse visible devices
                visible_list = [x.strip() for x in current_visible.split(",") if x.strip()]
                tts_gpus = [x.strip() for x in args.tts_gpu.split(",") if x.strip()]

                # Check for potential conflict but DO NOT modify CUDA_VISIBLE_DEVICES
                # as it hides GPUs from the process entirely.
                vllm_gpus = [g for g in visible_list if g not in tts_gpus]

                print(f"ℹ️  GPU Configuration Check:")
                print(f"  CUDA_VISIBLE_DEVICES: {current_visible}")
                print(f"  TTS Requested GPU(s): {tts_gpus}")
                print(f"  vLLM Expected GPU(s): {visible_list[:args.tensor_parallel_size]}") # Assuming vLLM takes first N

                if len(set(visible_list[:args.tensor_parallel_size]) & set(tts_gpus)) > 0:
                    print(f"⚠️  POTENTIAL GPU CONFLICT DETECTED!")
                    print(f"  vLLM (TP={args.tensor_parallel_size}) likely uses {visible_list[:args.tensor_parallel_size]}")
                    print(f"  TTS uses {tts_gpus}")
                    print("  Ensure you have enough VRAM or different devices assigned.")
                else:
                    print(f"✅ GPU assignment looks safe (vLLM: {visible_list[:args.tensor_parallel_size]}, TTS: {tts_gpus})")

        # Initialize embedded engine
        await init_async_engine(args)

    # Initialize context
    context_manage.clear_global_history()
    context_manage._global_context.history.append({
        "role": "system",
        "content": "You are receiving a live video stream where the final frame is the present moment. Respond only when a response is needed based on the user's message or the visual context. Otherwise, output '<|silent|>' to signify silence Respond in Chinese."
    })

    # Run TCP accept loop in the same event loop as the engine (fixes TTFT ~800ms delay
    # caused by engine and connection handler living in different loops).
    server_sock = _create_listen_socket(args.listen_port)
    print(f"🌐 Server listening on port {args.listen_port}")
    asyncio.create_task(run_accept_loop(server_sock, args))

    print("✅ Server started. Press Ctrl+C to exit.")

    # Keep main thread alive
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")


def main():
    args = parse_args()

    # Handle signals
    def signal_handler(sig, frame):
        print("\n👋 Received shutdown signal")
        exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run async main
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=1,2,3,4,5 python Qwen3_VL_online_streaming.py --listen-port 12345 --model /home/dyvm6xra/dyvm6xrauser36/Projects/streaming_video_understanding/qwen3vl-30b_a3b_20260128_01 --tensor-parallel-size 4 --max-model-len 128000 --gpu-memory-utilization 0.85 --asr-url http://localhost:8001/asr --kv-offloading-size 300 --disable-hybrid-kv-cache-manager --mm-encoder-attn-backend FLASH_ATTN --mm-encoder-tp-mode data --enable-tts --tts-gpu 5 --tts-model Qwen/Qwen3-TTS-12Hz-1.7B-Base --tts-language Chinese --tts-ref-audio test_query.mp3 --tts-ref-text "仔细观察当前你看到的画面，并且结合之前你看到的画面，仔细描述你看到了什么" --tts-output-dir tts_results

# ssh -L 5003:hk01dgx027:5003 dyvm6xrauser36@10.248.12.12