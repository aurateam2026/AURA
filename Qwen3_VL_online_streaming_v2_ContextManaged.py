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
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import aiohttp
import requests
import re  # Added for TTS sentence splitting
import sys

from datetime import datetime

# TTS is now a separate service (tts_service.py), no local model import needed

# vLLM V1 imports for streaming input
from vllm import SamplingParams
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM, StreamingInput

# Context management (reuse remove_markdown for TTS)
from context_manage import remove_markdown

# Global configuration
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# ============================================================================
# Data Classes
# ============================================================================

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
    Manages conversation history with two-tier context management (per context.md):
    - Sliding Window: recent rounds with full multimedia (video, images, <|silent|>)
    - Context History: compressed historical QAs (text-only, no video/silent)

    §3.1: When sliding window rounds > max_rounds, keep last num_rounds_keep in sliding window,
          move the rest to context history
    §3.2: Apply rewrite rules A-E when moving
    §3.3: Context history max max_context_qas QAs
    """
    def __init__(self, max_rounds: int = 20, num_rounds_keep: int = 15,
                 pruning_enabled: bool = False, debug_context_file: str = None,
                 max_context_qas: int = 10, max_1qna_rounds: int = 4):
        self.history = []
        self.max_rounds = max_rounds
        self.num_rounds_keep = num_rounds_keep
        self.pruning_enabled = pruning_enabled
        self.current_rounds = 0
        self.debug_context_file = debug_context_file
        self.max_context_qas = max_context_qas
        self.max_1qna_rounds = max_1qna_rounds

        self.system_prompt = "You are receiving a live video stream where the final frame is the present moment. Respond only when a response is needed based on the user's message or the visual context. Otherwise, output '<|silent|>' to signify silence. Respond in Chinese."
        self._system_msg = {"role": "system", "content": self.system_prompt}
        self._context_history = []   # list of QAs; each QA = list of message dicts (text-only)
        self._sliding_window = []    # list of message dicts (may contain multimedia)
        self._reset()

    def _reset(self):
        """Reset history to initial state (complete reset)."""
        self._context_history = []
        self._sliding_window = []
        self._rebuild_history()
        self.current_rounds = 0
        print(f"🔄 Session history reset")

    def _rebuild_history(self):
        """Compose self.history = [system] + context_history msgs + sliding_window msgs."""
        self.history = [self._system_msg]
        for qa in self._context_history:
            self.history.extend(qa)
        self.history.extend(self._sliding_window)

    def _sw_round_count(self):
        """Count user messages (rounds) in the sliding window."""
        return sum(1 for m in self._sliding_window if m["role"] == "user")

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

    def _serialize_history_for_debug(self) -> list:
        """将 history 转换为 JSON 可序列化格式，视频/图片用占位符代替。"""
        serialized = []
        for msg in self.history:
            entry = {"role": msg["role"]}
            content = msg["content"]
            if isinstance(content, str):
                entry["content"] = content
            elif isinstance(content, list):
                serialized_content = []
                for item in content:
                    if not isinstance(item, dict):
                        serialized_content.append(str(item))
                        continue
                    item_type = item.get("type", "")
                    if item_type == "text":
                        serialized_content.append(item)
                    elif item_type == "video":
                        serialized_content.append("<video>")
                    elif item_type == "image":
                        serialized_content.append("<image>")
                    else:
                        serialized_content.append({"type": item_type})
                entry["content"] = serialized_content
            else:
                entry["content"] = str(content)
            serialized.append(entry)
        return serialized

    def save_context_debug(self, request_id: str = ""):
        """将序列化前的结构化消息上下文按 JSONL 逐条写入。"""
        if not self.debug_context_file:
            return

        def _json_default(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            if hasattr(obj, 'tolist'):
                return obj.tolist()
            return str(obj)

        try:
            record = {
                "timestamp": time.time(),
                "datetime": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                "request_id": request_id,
                "current_rounds": self.current_rounds,
                "num_messages": len(self.history),
                # 序列化前的结构化消息（role/content），媒体使用占位符避免落盘大对象
                "history": self._serialize_history_for_debug(),
            }
            with open(self.debug_context_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            print(f"📝 [Debug] Structured context saved to {self.debug_context_file} "
                  f"(request_id={request_id}, round={self.current_rounds})")
        except Exception as e:
            print(f"⚠️ [Debug] Failed to save context: {e}")

    def _has_user_text(self, user_msg) -> bool:
        """检查 user 消息是否包含实际文本内容（非空）。"""
        content = user_msg.get("content", [])
        if isinstance(content, str):
            return bool(content.strip())
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text" and item.get("text", "").strip():
                    return True
        return False

    # ------------------------------------------------------------------
    # Context management helpers (per context.md §2-§3)
    # ------------------------------------------------------------------

    def _parse_sw_rounds(self):
        """Parse sliding window messages into (user_msg, assistant_msg|None) pairs."""
        rounds = []
        i = 0
        while i < len(self._sliding_window):
            msg = self._sliding_window[i]
            if msg["role"] == "user":
                user_msg = msg
                assistant_msg = None
                if (i + 1 < len(self._sliding_window) and
                        self._sliding_window[i + 1]["role"] == "assistant"):
                    assistant_msg = self._sliding_window[i + 1]
                    i += 2
                else:
                    i += 1
                rounds.append((user_msg, assistant_msg))
            else:
                i += 1
        return rounds

    def _group_rounds_into_qas(self, rounds):
        """
        Group rounds into QA units for context management.
        A round whose user message contains text starts a new QA;
        subsequent video-only rounds are continuations of the same QA.
        """
        groups = []
        current_group = []
        for user_msg, assistant_msg in rounds:
            if self._has_user_text(user_msg):
                if current_group:
                    groups.append(current_group)
                current_group = [(user_msg, assistant_msg)]
            else:
                current_group.append((user_msg, assistant_msg))
        if current_group:
            groups.append(current_group)
        return groups

    def _rewrite_qa_for_history(self, qa_rounds):
        """
        Apply rewrite rules A & B to a QA group from the sliding window.
        Rule A: Remove <video>, keep only text in user messages.
        Rule B: Remove entire round if assistant is <|silent|>.
        Returns a list of message dicts (context history format), or None.
        """
        rewritten = []
        for user_msg, assistant_msg in qa_rounds:
            if assistant_msg and self._is_silent_response(assistant_msg["content"]):
                continue
            user_text = self._extract_user_text(user_msg["content"])
            rewritten.append({"role": "user", "content": user_text})
            if assistant_msg:
                rewritten.append({"role": "assistant", "content": assistant_msg["content"]})
        return rewritten if rewritten else None

    def _qa_to_round_pairs(self, qa_messages):
        """Convert flat QA message list → [(user_content, assistant_content), ...]."""
        pairs = []
        i = 0
        while i < len(qa_messages):
            if qa_messages[i]["role"] == "user":
                u = qa_messages[i]["content"]
                a = None
                if i + 1 < len(qa_messages) and qa_messages[i + 1]["role"] == "assistant":
                    a = qa_messages[i + 1]["content"]
                    i += 2
                else:
                    i += 1
                pairs.append((u, a))
            else:
                i += 1
        return pairs

    def _count_qa_rounds(self, qa_messages):
        """Count rounds (user messages) in a QA."""
        return sum(1 for m in qa_messages if m["role"] == "user")

    def _classify_qa(self, qa_messages):
        """
        Classify a context-history QA (§2.2).
        Returns: "basic" | "1q1a" | "1qna" | "truncated" | None
        """
        pairs = self._qa_to_round_pairs(qa_messages)
        n = len(pairs)
        if n == 0:
            return None
        first_has_text = bool(pairs[0][0] and pairs[0][0].strip())
        if n == 1:
            return "basic" if first_has_text else "truncated"
        if n == 2 and first_has_text:
            return "1q1a"
        if n >= 3 and first_has_text:
            return "1qna"
        if not first_has_text:
            return "truncated"
        return None

    def _enforce_1qna_limit(self, qa_messages):
        """
        Rule E: 1QNA total rounds ≤ max_1qna_rounds (default 4).
        Delete earliest "" + assistant round (skipping the first round).
        """
        while self._count_qa_rounds(qa_messages) > self.max_1qna_rounds:
            found = False
            i = 2  # never delete the first round (indices 0, 1)
            while i < len(qa_messages):
                if (qa_messages[i]["role"] == "user"
                        and qa_messages[i]["content"] == ""):
                    del qa_messages[i]
                    if i < len(qa_messages) and qa_messages[i]["role"] == "assistant":
                        del qa_messages[i]
                    found = True
                    break
                i += 1
            if not found:
                break

    def _merge_truncated_qa(self, truncated_messages):
        """
        Rule D: Merge a truncated QA ("" + assistant) with the last QA in
        context history.  After merge, enforce Rule E if it became 1QNA.
        """
        if self._context_history:
            last_qa = self._context_history[-1]
            last_qa.extend(truncated_messages)
            if self._count_qa_rounds(last_qa) > self.max_1qna_rounds:
                self._enforce_1qna_limit(last_qa)
        else:
            self._context_history.append(list(truncated_messages))

    def _prune_history(self):
        """
        Context management per context.md §3:

        §3.1 — Trigger & migration:
            When sliding window rounds > max_rounds, keep the last
            num_rounds_keep rounds in sliding window; move the earlier
            (total - num_rounds_keep) rounds to context history (hard cut).

        §3.2 — Rewrite rules applied to moved rounds:
            A: Remove <video> from user messages
            B: Remove silent rounds
            C: Validate each QA matches §2.2 types
            D: Merge truncated QAs with last context history QA
            E: 1QNA in context history ≤ max_1qna_rounds rounds

        §3.3 — Context history capacity ≤ max_context_qas QAs.
        """
        rounds = self._parse_sw_rounds()
        if len(rounds) <= self.max_rounds:
            return

        num_to_move = len(rounds) - self.num_rounds_keep
        if num_to_move <= 0:
            return
        rounds_to_move = rounds[:num_to_move]
        rounds_remaining = rounds[num_to_move:]

        qa_groups = self._group_rounds_into_qas(rounds_to_move)

        moved_qas = 0
        merged_count = 0

        for qa_rounds in qa_groups:
            head_has_text = self._has_user_text(qa_rounds[0][0])
            rewritten = self._rewrite_qa_for_history(qa_rounds)
            if not rewritten:
                continue

            if head_has_text:
                qa_type = self._classify_qa(rewritten)
                if qa_type == "1qna":
                    self._enforce_1qna_limit(rewritten)
                self._context_history.append(rewritten)
                moved_qas += 1
            else:
                # Orphan continuation rounds → split into individual truncated
                # QAs and merge each via Rule D.
                i = 0
                while i < len(rewritten):
                    if rewritten[i]["role"] == "user":
                        truncated = [rewritten[i]]
                        if (i + 1 < len(rewritten)
                                and rewritten[i + 1]["role"] == "assistant"):
                            truncated.append(rewritten[i + 1])
                            i += 2
                        else:
                            i += 1
                        self._merge_truncated_qa(truncated)
                        merged_count += 1
                    else:
                        i += 1

        # §3.3: enforce capacity limit
        while len(self._context_history) > self.max_context_qas:
            self._context_history.pop(0)

        # Rebuild sliding window from remaining rounds
        self._sliding_window = []
        for user_msg, assistant_msg in rounds_remaining:
            self._sliding_window.append(user_msg)
            if assistant_msg is not None:
                self._sliding_window.append(assistant_msg)

        self.current_rounds = self._sw_round_count()
        self._rebuild_history()

        print(f"✂️ History pruned: moved {moved_qas} QAs, "
              f"merged {merged_count} truncated rounds | "
              f"context_history={len(self._context_history)} QAs, "
              f"sliding_window={self.current_rounds} rounds, "
              f"total_messages={len(self.history)}")

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
        content = []

        if video_tuple:
            content.append({"type": "video", "video": video_tuple})
        elif images:
            content.extend([{"type": "image", "image": img} for img in images])

        if text:
            content.append({"type": "text", "text": text})
        elif not images and not video_tuple:
            return

        msg = {"role": "user", "content": content}
        self._sliding_window.append(msg)
        self.history.append(msg)
        self.current_rounds += 1

        # §3.1: Trigger pruning when sliding window rounds exceed max_rounds
        if self.pruning_enabled and self._sw_round_count() > self.max_rounds:
            print(f"⚠️ Sliding window rounds ({self._sw_round_count()}) "
                  f"> max_rounds ({self.max_rounds}), pruning...")
            self._prune_history()

    def add_assistant_message(self, text: str):
        """Add assistant response to history."""
        msg = {"role": "assistant", "content": text}
        self._sliding_window.append(msg)
        self.history.append(msg)

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


# ============================================================================
# Cross-Turn Repetition Penalty (adapted from streaming_client.py)
# ============================================================================

_PENALTY_PUNCT = frozenset(
    ".,!?;:，。！？；：、'\"()[]{}""''…—–\n\t\r /-_@#$%^&*+=<>~`|\\（）【】《》"
)


class CrossTurnPenalty:
    """Cross-turn repetition penalty for embedded vLLM engine.

    Two complementary mechanisms:
    1. logit_bias  — soft penalty on content tokens from recent responses
    2. bad_words   — hard n-gram blocking via logits processor

    All penalty data is pre-computed before generation to minimize per-token
    overhead.  The logits processor itself only does tensor indexing (logit_bias)
    and a few dict lookups (bad_words) per step.
    """

    def __init__(
        self,
        tokenizer,
        window: int = 2,
        logit_penalty: float = 2.0,
        ngram_sizes: list[int] | None = None,
        max_bad_ngrams: int = 200,
        max_bias_tokens: int = 500,
    ):
        self.tokenizer = tokenizer
        self.window = window
        self.logit_penalty = logit_penalty
        self.ngram_sizes = ngram_sizes if ngram_sizes is not None else [3, 4, 5]
        self.max_bad_ngrams = max_bad_ngrams
        self.max_bias_tokens = max_bias_tokens
        self._history: list[str | None] = []   # None = silent turn, str = spoken turn
        self._special_ids = set(self.tokenizer.all_special_ids)
        self._penalizable_cache: dict[int, bool] = {}

    def _is_penalizable(self, token_id: int) -> bool:
        cached = self._penalizable_cache.get(token_id)
        if cached is not None:
            return cached
        if token_id in self._special_ids:
            self._penalizable_cache[token_id] = False
            return False
        decoded = self.tokenizer.decode([token_id]).strip()
        if not decoded or all(c in _PENALTY_PUNCT for c in decoded) or decoded.isdigit():
            self._penalizable_cache[token_id] = False
            return False
        self._penalizable_cache[token_id] = True
        return True

    def _spoken_history(self) -> list[str]:
        """Return only spoken (non-silent) entries from _history."""
        return [text for text in self._history if text is not None]

    def _build_logit_bias(self) -> dict[int, float]:
        spoken = self._spoken_history()
        if len(spoken) < 2:
            return {}
        n = len(spoken)

        # Phase 1: find tokens that appear in 2+ distinct spoken responses
        token_presence: dict[int, int] = {}
        for text in spoken:
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for tid in set(ids):
                token_presence[tid] = token_presence.get(tid, 0) + 1
        cross_turn_tids = {tid for tid, cnt in token_presence.items() if cnt >= 2}

        if not cross_turn_tids:
            return {}

        # Phase 2: compute penalty only for cross-turn tokens
        bias: dict[int, float] = {}
        for idx, text in enumerate(spoken):
            recency = (idx + 1) / n
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            freq = Counter(ids)
            for tid, cnt in freq.items():
                if tid not in cross_turn_tids:
                    continue
                if not self._is_penalizable(tid):
                    continue
                p = self.logit_penalty * min(cnt, 3) * recency
                bias[tid] = bias.get(tid, 0.0) + p

        penalized_details = ", ".join(
            f"'{self.tokenizer.decode([tid]).strip()}'({-val:.1f})"
            for tid, val in sorted(bias.items(), key=lambda kv: kv[1], reverse=True)[:20]
        )
        total_turns = len(self._history)
        print(f"🔧 [CrossTurnPenalty] window: {total_turns} actual turns "
              f"({len(spoken)} spoken, {total_turns - len(spoken)} silent)")
        print(f"🔧 [CrossTurnPenalty] cross-turn tokens: {len(cross_turn_tids)} "
              f"(penalizable: {len(bias)}) out of {len(token_presence)} total unique tokens")
        print(f"🔧 [CrossTurnPenalty] penalized: [{penalized_details}]")

        if len(bias) > self.max_bias_tokens:
            items = sorted(bias.items(), key=lambda kv: kv[1], reverse=True)
            bias = dict(items[: self.max_bias_tokens])
        return {k: min(v, 100.0) for k, v in bias.items()}

    def _build_bad_ngram_map(self) -> dict[tuple, set]:
        """prefix (n-1 token IDs) → set of blocked completing token IDs."""
        spoken = self._spoken_history()
        if not spoken:
            return {}
        prefix_map: dict[tuple, set] = {}
        seen: set[tuple] = set()
        count = 0
        for text in reversed(spoken):
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            for ng_size in self.ngram_sizes:
                if len(ids) < ng_size:
                    continue
                for i in range(len(ids) - ng_size + 1):
                    ngram = tuple(ids[i : i + ng_size])
                    if ngram in seen:
                        continue
                    phrase = self.tokenizer.decode(list(ngram)).strip()
                    if not phrase or all(c in _PENALTY_PUNCT for c in phrase):
                        continue
                    seen.add(ngram)
                    prefix = ngram[:-1]
                    if prefix not in prefix_map:
                        prefix_map[prefix] = set()
                    prefix_map[prefix].add(ngram[-1])
                    count += 1
                    if count >= self.max_bad_ngrams:
                        return prefix_map
        return prefix_map

    def build_sampling_kwargs(self) -> dict:
        """Return kwargs for SamplingParams: logit_bias and bad_words.

        Uses SamplingParams-native logit_bias (dict[int, float]) to softly
        penalise repeated content tokens, and bad_words (list[str]) to hard-
        block previously seen n-grams.  This replaces the old logits_processors
        approach which is no longer supported by vLLM V1.
        """
        raw_bias = self._build_logit_bias()
        bad_ngram_map = self._build_bad_ngram_map()

        if not raw_bias and not bad_ngram_map:
            return {}

        kwargs: dict = {}

        if raw_bias:
            kwargs["logit_bias"] = {tid: -val for tid, val in raw_bias.items()}

        if bad_ngram_map:
            bad_words: list[str] = []
            seen: set[tuple] = set()
            for prefix, blocked_set in bad_ngram_map.items():
                for last_tok in blocked_set:
                    ngram = prefix + (last_tok,)
                    if ngram in seen:
                        continue
                    seen.add(ngram)
                    phrase = self.tokenizer.decode(list(ngram))
                    if phrase.strip():
                        bad_words.append(phrase)
            if bad_words:
                kwargs["bad_words"] = bad_words

        bias_count = len(raw_bias)
        bw_count = len(kwargs.get("bad_words", []))
        spoken_count = len(self._spoken_history())
        total_turns = len(self._history)
        print(
            f"🔧 [CrossTurnPenalty] logit_bias: {bias_count} tokens | "
            f"bad_words: {bw_count} phrases | "
            f"window: {total_turns} turns ({spoken_count} spoken, "
            f"{total_turns - spoken_count} silent)"
        )

        return kwargs

    def record(self, response_text: str | None = None):
        """Call after every assistant turn (both spoken and silent).

        Args:
            response_text: The response text for spoken turns, or None for silent turns.
        """
        if response_text and response_text.strip():
            self._history.append(response_text)
        else:
            self._history.append(None)
        if len(self._history) > self.window:
            self._history.pop(0)

    def reset(self):
        self._history.clear()


@dataclass
class StreamingSession:
    """A streaming session for client."""
    session_id: str
    history: SessionHistory
    input_queue: asyncio.Queue
    output_queue: asyncio.Queue
    is_generating: bool = False
    is_auto_generating: bool = False
    current_task: Optional[asyncio.Task] = None
    cross_turn_penalty: Optional[CrossTurnPenalty] = None

# ============================================================================
# Global State
# ============================================================================

# Server state
active_connection = None
connection_lock = threading.Lock()

# TTS related globals (TTS model is now a remote service)
tts_enabled = False
tts_streaming = False
TTS_OUTPUT_DIR = "tts_results"
tts_service_url = None  # URL of the standalone TTS service

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
model_tokenizer = None

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

    clean_text = remove_markdown(sentence)
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
    """Verify the remote TTS service is reachable."""
    global tts_enabled, tts_streaming, tts_service_url

    url = getattr(args, "tts_service_url", None)
    if not url:
        print("⚠️ --tts-service-url not specified")
        tts_enabled = False
        return False

    tts_service_url = url.rstrip("/")
    print(f"🔊 Checking TTS service at {tts_service_url} ...")

    try:
        resp = requests.get(f"{tts_service_url}/v1/tts/health", timeout=10)
        resp.raise_for_status()
        info = resp.json()
        print(f"✓ TTS service connected: {info}")
        tts_streaming = True
        tts_enabled = True
        return True
    except Exception as e:
        print(f"⚠️ TTS service unreachable ({tts_service_url}): {e}")
        tts_enabled = False
        return False

def _text_to_speech_generator(text: str,
                         language: str = "Chinese",
                         speaker: str = "Vivian",
                         instruct: str = ""):
    """Stream PCM chunks from the remote TTS service.

    The service returns a binary stream where each chunk is:
        [sample_rate : 4 bytes big-endian uint32]
        [pcm_length  : 4 bytes big-endian uint32]
        [pcm_data    : pcm_length bytes, int16 LE]
    """
    print(f"🎤 [Remote] Requesting TTS: {text[:40]}...")

    try:
        resp = requests.post(
            f"{tts_service_url}/v1/tts/stream",
            json={"text": text, "language": language, "speaker": speaker, "instruct": instruct},
            stream=True,
            timeout=(5, 120),
        )
        resp.raise_for_status()

        buf = b""
        for raw_chunk in resp.iter_content(chunk_size=8192):
            buf += raw_chunk
            while len(buf) >= 8:
                sr, pcm_len = struct.unpack(">II", buf[:8])
                if len(buf) < 8 + pcm_len:
                    break
                pcm_data = buf[8 : 8 + pcm_len]
                buf = buf[8 + pcm_len :]
                yield pcm_data, sr

    except Exception as e:
        print(f"TTS remote call error: {e}")
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

    # Query model type from remote TTS service
    model_type = "base"
    try:
        resp = requests.get(f"{tts_service_url}/v1/tts/health", timeout=5)
        if resp.ok:
            model_type = resp.json().get("model_type", "base")
    except Exception:
        pass

    # Determine if we can use chunk streaming (only Base model supports true streaming)
    use_chunk_streaming = (model_type == "base")

    print(f"🔊 TTS Worker started (Remote service, Model: {model_type}, Chunk Streaming: {use_chunk_streaming})")

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

    # Store tokenizer globally for CrossTurnPenalty
    global model_tokenizer
    model_tokenizer = async_engine.get_tokenizer()

    # ========== DEBUG: 保存词表到日志文件 ==========
    try:
        tokenizer = model_tokenizer
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

    request_id = ""
    streaming_started = False
    try:
        # Add current turn to history with video tuple
        session.history.add_user_message(prompt, video_tuple=video_tuple)

        # Get vLLM inputs (with history for Prefix Caching)
        # t_get_inputs_start = time.time()
        vllm_inputs = session.history.get_vllm_inputs()
        # t_get_inputs_end = time.time()
        # print(f"⏱️ [TIMING] get_vllm_inputs() took {(t_get_inputs_end - t_get_inputs_start)*1000:.1f}ms")

        # Generate unique request ID for this turn
        request_id = generate_response_id()

        session.history.save_context_debug(request_id=request_id)

        video_array, video_metadata = video_tuple
        print(f"🎬 [Session {session.session_id}] Starting generation (request_id={request_id})")
        print(f"📥 Input: {video_array.shape[0]} video frames ({video_array.shape}), prompt='{prompt}'")

        full_response = ""
        previous_text = ""
        is_silent_response = False
        ttft = 0

        tts_enabled_for_this_response = args and args.enable_tts and tts_enabled

        if tts_enabled_for_this_response and prompt:
            clear_tts_sentence_queue(new_response_id=request_id)

        # Timing measurements
        generation_start_time = time.time()
        first_token_time = None
        token_count = 0
        print(f"[TTFT_DEBUG] stream generate_start request_id={request_id} t={generation_start_time:.6f}")
        print(f"⏱️ [TIMING] prefill_submit_time={generation_start_time:.6f} (engine.generate called, prefill starts)")

        # Incremental sentence buffer for streaming TTS
        _tts_sentence_buf = ""
        _tts_sentence_idx = 0
        _SENT_ENDS = frozenset("。！？；.!?;\n")
        _COMMA_ENDS = frozenset("，,")
        _TTS_MIN_CHARS = 10
        streaming_started = False

        # ===== Streaming generation: send tokens to frontend as they arrive =====
        async for response in async_engine.generate(
            prompt=vllm_inputs,
            sampling_params=sampling_params,
            request_id=request_id,
        ):
            if first_token_time is None:
                first_token_time = time.time()
                ttft = first_token_time - generation_start_time
                print(f"[TTFT_DEBUG] stream first_token request_id={request_id} t={first_token_time:.6f} ttft_ms={ttft*1000:.1f}")

            token_count += 1
            if response.outputs:
                output = response.outputs[0]

                if len(output.token_ids) > 0 and output.token_ids[0] in (SILENT_TOKEN_ID, IM_END_TOKEN_ID):
                    is_silent_response = True
                    first_tid = output.token_ids[0]
                    tag = "SILENT" if first_tid == SILENT_TOKEN_ID else "IM_END"
                    print(f"🔇 [Session {session.session_id}] {tag} as first token → silent response "
                          f"(first_token_id={first_tid}, token_ids={list(output.token_ids[:5])}...)")
                    break

                if hasattr(output, 'text') and output.text:
                    current_text = output.text
                    if len(current_text) > len(previous_text):
                        delta = current_text[len(previous_text):]
                        previous_text = current_text
                        full_response += delta

                        # --- Stream delta to frontend ---
                        if not streaming_started:
                            send_streaming_token_to_client(delta, request_id, is_start=True)
                            streaming_started = True
                        else:
                            send_streaming_token_to_client(delta, request_id)

                        # --- Incremental TTS sentence detection ---
                        if tts_enabled_for_this_response:
                            _tts_sentence_buf += delta
                            while _tts_sentence_buf:
                                split_pos = -1
                                for i, ch in enumerate(_tts_sentence_buf):
                                    if ch in _SENT_ENDS:
                                        split_pos = i + 1
                                        break
                                    if ch in _COMMA_ENDS and i + 1 >= _TTS_MIN_CHARS:
                                        split_pos = i + 1
                                        break
                                if split_pos < 0:
                                    break
                                sentence = _tts_sentence_buf[:split_pos]
                                _tts_sentence_buf = _tts_sentence_buf[split_pos:]
                                if sentence.strip():
                                    enqueue_tts_sentence(sentence, request_id, _tts_sentence_idx, args)
                                    _tts_sentence_idx += 1

        # ===== Generation finished — decide: silent / send =====
        generation_end_time = time.time()
        total_time = generation_end_time - generation_start_time

        print(f"✅ [Session {session.session_id}] Generation finished")
        print(f"⏱️ [TIMING] Time to first token (TTFT): {ttft*1000:.1f}ms, timestamp: {time.time()}")
        print(f"⏱️ [TIMING] TTFT avg. by {video_array.shape[0]} frames: {(ttft*1000/video_array.shape[0]):.1f}ms")
        print(f"⏱️ [TIMING] Total generation time: {total_time*1000:.1f}ms")
        print(f"⏱️ [TIMING] Tokens generated: {token_count}")
        if token_count > 1 and first_token_time is not None:
            decode_only_ms = (generation_end_time - first_token_time) * 1000
            avg_decode_per_token = decode_only_ms / (token_count - 1)
            print(f"⏱️ [TIMING] Decode phase: {decode_only_ms:.1f}ms for {token_count-1} tokens, "
                  f"avg={avg_decode_per_token:.1f}ms/token ({1000/avg_decode_per_token:.1f} tokens/s)")

        print(f"📋 [DECISION] request_id={request_id} | is_silent={is_silent_response} | "
              f"full_response({len(full_response)} chars)='{full_response[:80]}'")

        if is_silent_response:
            print(f"🔇 [DECISION] → MODEL_SILENT (first token was silent/im_end)")
            session.history.add_assistant_message(SILENT_TEXT)
            send_streaming_token_to_client(SILENT_TEXT, request_id, is_final=True, is_silent=True)
            if session.cross_turn_penalty is not None:
                session.cross_turn_penalty.record(None)

        else:
            # Send final marker to frontend (content already streamed)
            send_streaming_token_to_client("", request_id, is_final=True)
            session.history.add_assistant_message(full_response)
            if session.cross_turn_penalty is not None:
                session.cross_turn_penalty.record(full_response)

            # Flush remaining TTS sentence buffer
            if tts_enabled_for_this_response and _tts_sentence_buf.strip():
                enqueue_tts_sentence(_tts_sentence_buf, request_id, _tts_sentence_idx, args)
                _tts_sentence_idx += 1

            if tts_enabled_for_this_response:
                print(f"🎤 [Queue] Streamed {_tts_sentence_idx} sentences to TTS")

    except asyncio.CancelledError:
        print(f"⏹ [Session {session.session_id}] Generation cancelled (context reset)")
        if streaming_started and request_id:
            send_streaming_token_to_client("", request_id, is_final=True)
    except Exception as e:
        print(f"❌ [Session {session.session_id}] Generation error: {e}")
        import traceback
        traceback.print_exc()
        if request_id:
            send_streaming_token_to_client("", request_id, is_final=True)
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
                if is_start:
                    response_data["is_start"] = True
                if query:
                    response_data["query"] = query

                # Mark silent responses so client can handle them appropriately
                if is_silent:
                    response_data["is_silent"] = True

                payload = json.dumps(response_data, ensure_ascii=False).encode('utf-8')
                # Use type 8 for streaming tokens (type 7 is ERROR in client)
                header = struct.pack(">BQ", 8, len(payload))
                active_connection.sendall(header + payload)
            except Exception as e:
                print(f"Error sending streaming token: {e}")


def send_asr_query_to_client(query: str):
    """Send ASR-transcribed query text to client immediately (before model inference).

    Protocol: Type 10 (ASR_QUERY_ECHO) - a dedicated message type so that
    clients won't mistake it for a model streaming response (Type 8).
    This allows the client to display the user's query as soon as ASR finishes,
    without waiting for the model to start generating (Plan 2 optimization).
    """
    with connection_lock:
        if active_connection:
            try:
                response_data = {
                    "type": "asr_query",
                    "query": query,
                }
                payload = json.dumps(response_data, ensure_ascii=False).encode('utf-8')
                header = struct.pack(">BQ", 10, len(payload))
                active_connection.sendall(header + payload)
                print(f"📤 [Plan2] Sent ASR query to client immediately: {query[:50]}...")
            except Exception as e:
                print(f"Error sending ASR query to client: {e}")


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

    # Build cross-turn penalty manager (if enabled)
    penalty_mgr = None
    if getattr(args, "cross_turn_penalty", 0) > 0 and model_tokenizer is not None:
        penalty_mgr = CrossTurnPenalty(
            tokenizer=model_tokenizer,
            window=getattr(args, "cross_turn_lookback", 2),
            logit_penalty=args.cross_turn_penalty,
            ngram_sizes=getattr(args, "cross_turn_ngram_sizes", [3, 4, 5]),
        )
        print(f"🔧 [Session {session_id}] CrossTurnPenalty enabled: "
              f"penalty={args.cross_turn_penalty}, window={args.cross_turn_lookback}, "
              f"ngram_sizes={args.cross_turn_ngram_sizes}")

    # Session state with History
    session = StreamingSession(
        session_id=session_id,
        history=SessionHistory(
            max_rounds=args.max_rounds,
            num_rounds_keep=args.num_rounds_keep,
            pruning_enabled=args.enable_pruning,
            debug_context_file=args.debug_context_file if args.debug_context else None,
            max_context_qas=args.max_context_qas,
        ),
        input_queue=asyncio.Queue(),
        output_queue=asyncio.Queue(),
        is_generating=False,
        cross_turn_penalty=penalty_mgr,
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

                            penalty_kwargs = {}
                            if session.cross_turn_penalty is not None:
                                penalty_kwargs = session.cross_turn_penalty.build_sampling_kwargs()

                            sampling_params = SamplingParams(
                                temperature=args.temperature,
                                max_tokens=args.max_tokens,
                                **penalty_kwargs,
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
                _asr_start = time.time()
                if args.asr_sync:
                    # Synchronous version
                    loop = asyncio.get_event_loop()
                    transcribed_text = await loop.run_in_executor(
                        None, get_audio_prompt, audio_path, args.asr_url
                    )
                else:
                    # Asynchronous version (default)
                    transcribed_text = await transcribe_audio_async(audio_path, args.asr_url)
                _asr_end = time.time()
                print(f"⏱️ [TIMING] ASR latency: {(_asr_end - _asr_start)*1000:.1f}ms")

                if transcribed_text:
                    last_prompt = transcribed_text
                    print(f"📝 Set prompt from ASR: {last_prompt[:50]}...")

                    # Plan 2: ASR query is sent to client immediately,
                    # model inference result will be sent separately later.
                    send_asr_query_to_client(transcribed_text)

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
                session.history._reset()
                if session.cross_turn_penalty is not None:
                    session.cross_turn_penalty.reset()
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
                session.history._reset()
                if session.cross_turn_penalty is not None:
                    session.cross_turn_penalty.reset()
                accumulated_video_frames = []
                last_prompt = ""

            elif file_type == 11:  # Text Prompt
                # Direct text-prompt injection. Equivalent to the post-ASR path
                # in the Audio (type 2) handler, but without the round-trip
                # through TTS/ASR. The payload is a UTF-8 string that becomes
                # `last_prompt` and triggers the same generation priority logic
                # the audio path uses.
                try:
                    prompt_text = file_data.decode("utf-8").strip()
                except UnicodeDecodeError:
                    print("⚠ Text prompt payload was not valid UTF-8, ignoring")
                    continue

                if not prompt_text:
                    print("⚠ Text prompt payload was empty, ignoring")
                    continue

                last_prompt = prompt_text
                print(f"📝 Set prompt from text: {last_prompt[:80]}...")

                # Echo the prompt to the client the same way the ASR path does
                # so callers see a consistent acknowledgement.
                send_asr_query_to_client(prompt_text)

                # If a background auto-generation is in flight, interrupt it so
                # the next round uses the user prompt instead of running silent.
                if accumulated_video_frames:
                    print("🚀 Text prompt arrived, attempting immediate trigger...")
                    if session.is_generating and session.is_auto_generating:
                        if session.current_task and not session.current_task.done():
                            print("🛑 Interrupting auto-generation for user prompt (from Text event)!")
                            session.current_task.cancel()

                # Generation is triggered by the Video frame loop when frames
                # arrive, mirroring the Audio handler's pattern.

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

    # Sampling parameters
    parser.add_argument("--temperature", type=float, default=0.9,
                        help="Sampling temperature for generation (default: 0.9)")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Maximum number of tokens to generate (default: 512)")

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
                        help="Number of rounds to keep in sliding window after pruning")
    parser.add_argument("--max-context-qas", type=int, default=10,
                        help="Maximum number of QAs to keep in context history")
    parser.add_argument("--dedup-threshold", type=float, default=0.0,
                        help="(deprecated, no longer used)")
    parser.add_argument("--cross-turn-penalty", type=float, default=0.0,
                        help="Cross-turn repetition penalty strength "
                             "(0=disabled, 2.0~3.0 recommended). "
                             "Combines soft logit_bias + hard n-gram blocking.")
    parser.add_argument("--cross-turn-lookback", type=int, default=2,
                        help="Number of recent assistant responses to penalize (window size)")
    parser.add_argument("--cross-turn-ngram-sizes", type=int, nargs="*", default=[3, 4, 5],
                        help="N-gram sizes for bad_words hard blocking (default: 3 4 5, pass empty to disable)")
    parser.add_argument("--debug-context", action="store_true",
                        help="Enable debug: save inference context to JSONL file")
    parser.add_argument("--debug-context-file", type=str, default="context_debug.jsonl",
                        help="JSONL file path for context debug output")

    # TTS configuration (TTS is now a remote service)
    parser.add_argument("--enable-tts", action="store_true", help="Enable TTS")
    parser.add_argument("--tts-service-url", type=str, default="http://localhost:8002",
                        help="URL of the standalone TTS service")
    parser.add_argument("--tts-speaker", type=str, default="Vivian",
                        help="TTS speaker name")
    parser.add_argument("--tts-language", type=str, default="Chinese", choices=["Chinese", "English"],
                        help="TTS language")
    parser.add_argument("--tts-instruct", type=str, default="",
                        help="TTS instruction")
    parser.add_argument("--tts-output-dir", type=str, default="tts_results",
                        help="TTS output directory")

    return parser.parse_args()


async def main_async(args):
    """Main async entry point."""
    # 服务启动时清空 debug context 文件
    if args.debug_context:
        try:
            with open(args.debug_context_file, "w", encoding="utf-8") as f:
                pass
            print(f"🗑 [Debug] Cleared context file on server start: {args.debug_context_file}")
        except Exception as e:
            print(f"⚠️ [Debug] Failed to clear context file: {e}")

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
    print(f"  Max context QAs: {args.max_context_qas}")
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
        print(f"  TTS Enabled: service at {args.tts_service_url}")
    print("=" * 60)

    # Initialize TTS if enabled
    if args.enable_tts:
        if init_tts_model(args):
            start_tts_worker(args)
        else:
            print("⚠ TTS initialization failed, TTS will be disabled")
    # print("⚠ TTS initialization failed, TTS will be disabled")

    if not args.use_http_api:
        # Initialize embedded engine
        await init_async_engine(args)

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