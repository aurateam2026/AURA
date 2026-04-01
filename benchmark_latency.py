#!/usr/bin/env python3
"""
离线基准测试脚本：独立测量 ASR / Qwen3 VL / TTS 各组件延迟。

使用方法:
    # 确保 ASR (8001)、TTS (8002)、vLLM 主服务 (12345) 已启动
    python benchmark_latency.py \
        --video test_video.webm \
        --audio test_audio.mp3 \
        --tts-text "你好，这是一个延迟测试句子。" \
        --runs 5

    # 只测某个组件
    python benchmark_latency.py --video test_video.webm --audio test_audio.mp3 --only llm --runs 5
    python benchmark_latency.py --audio test_audio.mp3 --only asr --runs 10
    python benchmark_latency.py --tts-text "测试文本" --only tts --runs 10

    # 测试完后解析 vllm.log 获取服务端精确计时
    python benchmark_latency.py --parse-log logs/vllm.log

输出:
    每个组件的 p50/p90/p99/mean/std 延迟（毫秒）

前提:
    需要准备一个固定的测试视频（1s chunk WebM）和一段测试音频（MP3/WAV）。
    从浏览器录制一段 1s 视频保存为 webm，或直接用 ffmpeg 截取:
        ffmpeg -i input.mp4 -t 1 -c:v libvpx -c:a libvorbis test_video.webm
    音频同理:
        ffmpeg -i input.mp4 -t 3 -vn -acodec libmp3lame test_audio.mp3
"""

import argparse
import json
import os
import re
import struct
import socket
import sys
import time
import statistics

import requests


# ============================================================================
# ASR Benchmark
# ============================================================================

def benchmark_asr(audio_path: str, asr_url: str, runs: int) -> list[dict]:
    """
    向 ASR 服务发送音频文件，测量端到端延迟。
    返回每次运行的 {latency_ms, text_length, text} 列表。
    """
    results = []
    file_size = os.path.getsize(audio_path)
    print(f"\n{'='*60}")
    print(f"ASR Benchmark: {audio_path} ({file_size} bytes)")
    print(f"ASR URL: {asr_url}")
    print(f"Runs: {runs}")
    print(f"{'='*60}")

    for i in range(runs):
        with open(audio_path, 'rb') as f:
            files = {'file': ('test_audio', f)}
            t0 = time.perf_counter()
            resp = requests.post(asr_url, files=files, params={"run_vllm": "false"}, timeout=30)
            t1 = time.perf_counter()

        latency_ms = (t1 - t0) * 1000
        text = ""
        if resp.status_code == 200:
            data = resp.json()
            text = data.get("text", "")

        results.append({
            "run": i + 1,
            "latency_ms": latency_ms,
            "text_length": len(text),
            "text": text[:80],
        })
        print(f"  [{i+1}/{runs}] latency={latency_ms:.1f}ms  text({len(text)})=\"{text[:50]}\"")

    return results


# ============================================================================
# LLM (Qwen3 VL) Benchmark via TCP Protocol
# ============================================================================

def _tcp_send_message(sock: socket.socket, msg_type: int, data: bytes):
    """按照项目的 TCP 二进制协议发送消息：type(1B) + length(8B, big-endian) + data"""
    header = struct.pack(">BQ", msg_type, len(data))
    sock.sendall(header + data)


def _tcp_recv_header(sock: socket.socket) -> tuple[int, int]:
    """接收 9 字节 header，返回 (msg_type, payload_length)"""
    buf = b""
    while len(buf) < 9:
        chunk = sock.recv(9 - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf += chunk
    msg_type, length = struct.unpack(">BQ", buf)
    return msg_type, length


def _tcp_recv_payload(sock: socket.socket, length: int) -> bytes:
    """接收指定长度的 payload"""
    buf = b""
    while len(buf) < length:
        chunk = sock.recv(min(length - len(buf), 65536))
        if not chunk:
            raise ConnectionError("Connection closed")
        buf += chunk
    return buf


def benchmark_llm_tcp(
    video_path: str,
    audio_path: str,
    host: str,
    port: int,
    runs: int,
) -> list[dict]:
    """
    通过 TCP 协议发送视频 + 音频，触发完整的 ASR → LLM 推理流程，测量:
      - client_ttft: 客户端视角，从发送视频+音频完毕到收到首个 Type 8 token
      - client_total: 客户端视角，从发送完毕到收到 is_final
      - token_count: 生成的 token 数量

    注意:
      client_ttft 包含: ASR延迟 + video_decode + prefill + 首decode + 网络往返。
      如需精确的服务端 prefill/decode 时间，请配合 --parse-log 解析 vllm.log。
    """
    results = []
    video_data = open(video_path, 'rb').read()
    audio_data = open(audio_path, 'rb').read()
    print(f"\n{'='*60}")
    print(f"LLM (Qwen3 VL) Benchmark via TCP")
    print(f"Video: {video_path} ({len(video_data)} bytes)")
    print(f"Audio: {audio_path} ({len(audio_data)} bytes)")
    print(f"Server: {host}:{port}")
    print(f"Runs: {runs}")
    print(f"{'='*60}")
    print(f"  注意: client_ttft = ASR + video_decode + prefill + 首decode + 网络")
    print(f"  精确的 prefill/decode 时间请用: --parse-log logs/vllm.log\n")

    for i in range(runs):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect((host, port))

        try:
            # Step 1: Type 6 (Start Camera) 初始化会话
            _tcp_send_message(sock, 6, b"start")
            time.sleep(0.5)

            # Step 2: Type 2 (Audio) FIRST — 发送音频让 ASR 设置 prompt
            # 必须先于视频发送，否则视频到达后 prompt 为空，模型会返回 SILENT
            _tcp_send_message(sock, 2, audio_data)

            # Step 3: 等待 ASR echo (Type 10)，确认 prompt 已设置
            asr_wait_start = time.perf_counter()
            asr_echo = ""
            while time.perf_counter() - asr_wait_start < 15:
                try:
                    sock.settimeout(15)
                    msg_type, length = _tcp_recv_header(sock)
                    payload = _tcp_recv_payload(sock, length)
                    if msg_type == 10:  # ASR echo
                        asr_data = json.loads(payload.decode('utf-8'))
                        asr_echo = asr_data.get("query", "")
                        print(f"    ASR echo received: \"{asr_echo[:40]}...\"")
                        break
                    elif msg_type == 8:
                        # Auto-gen might have started; absorb and continue
                        pass
                except socket.timeout:
                    print(f"    ⚠️ ASR echo timeout, proceeding anyway")
                    break

            # Step 4: Type 1 (Video) — 发送视频触发推理（此时 prompt 已就绪）
            t_send_done = time.perf_counter()
            sock.settimeout(120)
            _tcp_send_message(sock, 1, video_data)

            # Step 4: 接收流式回复
            first_token_time = None
            final_time = None
            token_count = 0
            full_text = ""
            asr_echo = ""
            is_silent = False
            got_response = False

            while True:
                try:
                    msg_type, length = _tcp_recv_header(sock)
                    payload = _tcp_recv_payload(sock, length)

                    if msg_type == 8:  # Streaming token (Type 8)
                        token_data = json.loads(payload.decode('utf-8'))
                        token_text = token_data.get("token", "")
                        is_final = token_data.get("is_final", False)
                        is_start = token_data.get("is_start", False)

                        if token_data.get("is_silent", False):
                            is_silent = True
                            final_time = time.perf_counter()
                            got_response = True
                            break

                        if first_token_time is None and token_text:
                            first_token_time = time.perf_counter()

                        if token_text:
                            token_count += 1
                            full_text += token_text

                        if is_final:
                            final_time = time.perf_counter()
                            got_response = True
                            break

                    elif msg_type == 10:  # ASR echo
                        asr_data = json.loads(payload.decode('utf-8'))
                        asr_echo = asr_data.get("query", "")

                    elif msg_type == 9:  # TTS PCM chunk, skip
                        pass

                    elif msg_type == 3:  # Complete text (legacy)
                        final_time = time.perf_counter()
                        full_text = payload.decode('utf-8')
                        got_response = True
                        break

                    elif msg_type == 7:  # Error
                        print(f"    ❌ Server error: {payload.decode('utf-8', errors='replace')}")
                        got_response = True
                        break

                except socket.timeout:
                    print(f"  [{i+1}/{runs}] TIMEOUT waiting for response")
                    break

            if got_response:
                client_ttft_ms = (first_token_time - t_send_done) * 1000 if first_token_time else float('nan')
                client_total_ms = (final_time - t_send_done) * 1000 if final_time else float('nan')

                result = {
                    "run": i + 1,
                    "client_ttft_ms": client_ttft_ms,
                    "client_total_ms": client_total_ms,
                    "token_count": token_count,
                    "is_silent": is_silent,
                    "asr_text": asr_echo[:50],
                    "text": full_text[:80],
                }
                results.append(result)

                status = "SILENT" if is_silent else f"tokens={token_count}"
                print(f"  [{i+1}/{runs}] client_ttft={client_ttft_ms:.1f}ms  "
                      f"client_total={client_total_ms:.1f}ms  {status}  "
                      f"asr=\"{asr_echo[:30]}\"  text=\"{full_text[:40]}\"")
            else:
                results.append({"run": i + 1, "client_ttft_ms": float('nan'),
                                "client_total_ms": float('nan'), "token_count": 0,
                                "is_silent": False, "asr_text": "", "text": "TIMEOUT"})
                print(f"  [{i+1}/{runs}] FAILED (no response)")

        finally:
            try:
                _tcp_send_message(sock, 4, b"clear")
                time.sleep(0.5)
            except Exception:
                pass
            sock.close()

        time.sleep(2)

    return results


# ============================================================================
# LLM (Qwen3 VL) Benchmark — KV Cache HIT (warm session, prefix cached)
# ============================================================================

def _collect_one_response(sock) -> dict:
    """Send video already queued; collect streaming response until final/silent."""
    first_token_time = None
    final_time = None
    token_count = 0
    full_text = ""
    is_silent = False
    got_response = False
    asr_echo = ""

    while True:
        try:
            msg_type, length = _tcp_recv_header(sock)
            payload = _tcp_recv_payload(sock, length)

            if msg_type == 8:
                token_data = json.loads(payload.decode('utf-8'))
                token_text = token_data.get("token", "")
                is_final = token_data.get("is_final", False)

                if token_data.get("is_silent", False):
                    is_silent = True
                    final_time = time.perf_counter()
                    got_response = True
                    break

                if first_token_time is None and token_text:
                    first_token_time = time.perf_counter()

                if token_text:
                    token_count += 1
                    full_text += token_text

                if is_final:
                    final_time = time.perf_counter()
                    got_response = True
                    break

            elif msg_type == 10:
                asr_data = json.loads(payload.decode('utf-8'))
                asr_echo = asr_data.get("query", "")
            elif msg_type in (9,):
                pass
            elif msg_type == 3:
                final_time = time.perf_counter()
                full_text = payload.decode('utf-8')
                got_response = True
                break
            elif msg_type == 7:
                print(f"    Server error: {payload.decode('utf-8', errors='replace')}")
                got_response = True
                break
        except socket.timeout:
            break

    return {
        "first_token_time": first_token_time,
        "final_time": final_time,
        "token_count": token_count,
        "full_text": full_text,
        "is_silent": is_silent,
        "got_response": got_response,
        "asr_echo": asr_echo,
    }


def benchmark_llm_tcp_cache_hit(
    video_path: str,
    audio_path: str,
    host: str,
    port: int,
    runs: int,
    warmup_runs: int = 1,
    measure_audio_path: str = None,
) -> list[dict]:
    """
    KV Cache Hit benchmark — fair comparison with cache-miss.

    For EACH measurement run:
      1. Open a fresh TCP connection (new session)
      2. Send 1 warmup video (builds the prefix in KV cache, ~2 turns of context)
      3. Send the 2nd video and MEASURE it (prefix is now cached)
      4. Close connection

    This ensures every measurement has the same context depth (~2 turns),
    and only 1 extra turn vs cache-miss.  The difference is purely
    whether the system-prompt + 1st-turn prefix is in the KV cache.

    `measure_audio_path` — if provided, use a different audio for the
    measurement round to avoid the model returning SILENT due to
    repeated identical prompts.
    """
    results = []
    video_data = open(video_path, 'rb').read()
    audio_data = open(audio_path, 'rb').read()
    measure_audio_data = audio_data
    if measure_audio_path and os.path.exists(measure_audio_path):
        measure_audio_data = open(measure_audio_path, 'rb').read()

    print(f"\n{'='*60}")
    print(f"LLM (Qwen3 VL) Cache HIT Benchmark via TCP")
    print(f"  (fair: independent session per run, 1 warmup + 1 measurement)")
    print(f"Video: {video_path} ({len(video_data)} bytes)")
    print(f"Audio: {audio_path} ({len(audio_data)} bytes)")
    if measure_audio_path:
        print(f"Measure audio: {measure_audio_path} ({len(measure_audio_data)} bytes)")
    print(f"Server: {host}:{port}")
    print(f"Runs: {runs} (each with {warmup_runs} warmup round)")
    print(f"{'='*60}\n")

    for i in range(runs):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(120)
        sock.connect((host, port))

        try:
            _tcp_send_message(sock, 6, b"start")
            time.sleep(0.3)

            # ---- Warmup: 1st video (populates prefix cache) ----
            for w in range(warmup_runs):
                _tcp_send_message(sock, 2, audio_data)
                asr_wait = time.perf_counter()
                while time.perf_counter() - asr_wait < 15:
                    try:
                        sock.settimeout(15)
                        mt, ln = _tcp_recv_header(sock)
                        pl = _tcp_recv_payload(sock, ln)
                        if mt == 10:
                            break
                        elif mt in (8, 9):
                            pass
                    except socket.timeout:
                        break

                sock.settimeout(120)
                _tcp_send_message(sock, 1, video_data)
                warmup_resp = _collect_one_response(sock)
                w_status = "SILENT" if warmup_resp["is_silent"] else f"tokens={warmup_resp['token_count']}"
                print(f"  [{i+1}/{runs}] warmup: {w_status}")
                time.sleep(0.5)

            # ---- Measurement: 2nd video (prefix should be cached) ----
            _tcp_send_message(sock, 2, measure_audio_data)
            asr_echo = ""
            asr_wait = time.perf_counter()
            while time.perf_counter() - asr_wait < 15:
                try:
                    sock.settimeout(15)
                    mt, ln = _tcp_recv_header(sock)
                    pl = _tcp_recv_payload(sock, ln)
                    if mt == 10:
                        asr_data = json.loads(pl.decode('utf-8'))
                        asr_echo = asr_data.get("query", "")
                        break
                    elif mt in (8, 9):
                        pass
                except socket.timeout:
                    break

            t_send_done = time.perf_counter()
            sock.settimeout(120)
            _tcp_send_message(sock, 1, video_data)
            resp = _collect_one_response(sock)

            if resp["got_response"]:
                client_ttft_ms = ((resp["first_token_time"] - t_send_done) * 1000
                                  if resp["first_token_time"] else float('nan'))
                client_total_ms = ((resp["final_time"] - t_send_done) * 1000
                                   if resp["final_time"] else float('nan'))

                status = "SILENT" if resp["is_silent"] else f"tokens={resp['token_count']}"
                print(f"  [{i+1}/{runs}] MEASURE ttft={client_ttft_ms:.1f}ms  "
                      f"total={client_total_ms:.1f}ms  {status}  "
                      f"text=\"{resp['full_text'][:40]}\"")

                results.append({
                    "run": i + 1,
                    "client_ttft_ms": client_ttft_ms,
                    "client_total_ms": client_total_ms,
                    "token_count": resp["token_count"],
                    "is_silent": resp["is_silent"],
                    "asr_text": asr_echo[:50],
                    "text": resp["full_text"][:80],
                })
            else:
                print(f"  [{i+1}/{runs}] MEASURE FAILED (no response)")
                results.append({"run": i + 1, "client_ttft_ms": float('nan'),
                                "client_total_ms": float('nan'), "token_count": 0,
                                "is_silent": False, "asr_text": "", "text": "TIMEOUT"})

        finally:
            try:
                _tcp_send_message(sock, 4, b"clear")
                time.sleep(0.3)
            except Exception:
                pass
            sock.close()

        time.sleep(2)

    return results


# ============================================================================
# LLM Sustained Streaming Benchmark (AURA paper conditions)
# ============================================================================

def benchmark_llm_sustained_streaming(
    video_path: str,
    host: str,
    port: int,
    duration_sec: float = 300.0,
    send_interval: float = 1.0,
    config: str = "full",
) -> dict:
    """
    AURA 论文条件的持续流式 benchmark。

    模拟真实使用场景：在单个 session 中连续发送视频流，测量每轮 TTFT
    随时间变化的趋势。不发送 audio/prompt，让模型做 auto-generation
    （静默或自主回复），以隔离推理延迟。

    Args:
        video_path: 测试视频路径 (1s WebM chunk)
        host: vLLM 服务地址
        port: vLLM 服务端口
        duration_sec: 持续发送时长 (秒)，默认 300 (5分钟)
        send_interval: 视频发送间隔 (秒)，默认 1.0
        config: 测试配置标签 (仅用于日志标识)

    Returns:
        dict with keys:
          - config: 配置名
          - duration_sec: 实际持续时间
          - total_rounds: 总轮数
          - results: list of per-round dicts
          - summary: aggregated stats
    """
    video_data = open(video_path, 'rb').read()

    print(f"\n{'='*60}")
    print(f"LLM Sustained Streaming Benchmark (AURA-style)")
    print(f"  Config: {config}")
    print(f"  Video: {video_path} ({len(video_data)} bytes)")
    print(f"  Duration: {duration_sec}s ({duration_sec/60:.1f} min)")
    print(f"  Send interval: {send_interval}s")
    print(f"  Server: {host}:{port}")
    print(f"{'='*60}\n")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120)
    sock.connect((host, port))

    per_round = []
    round_idx = 0

    try:
        # Start camera / initialize session
        _tcp_send_message(sock, 6, b"start")
        time.sleep(0.5)

        stream_start = time.time()

        while True:
            elapsed = time.time() - stream_start
            if elapsed >= duration_sec:
                break

            round_idx += 1
            t_send = time.perf_counter()

            # Send video chunk (Type 1) — no audio, triggers auto-generation
            _tcp_send_message(sock, 1, video_data)

            # Collect response (auto-gen: silent or spoken)
            first_token_time = None
            final_time = None
            token_count = 0
            full_text = ""
            is_silent = False
            got_response = False

            sock.settimeout(30)
            while True:
                try:
                    msg_type, length = _tcp_recv_header(sock)
                    payload = _tcp_recv_payload(sock, length)

                    if msg_type == 8:  # Streaming token
                        token_data = json.loads(payload.decode('utf-8'))
                        token_text = token_data.get("token", "")
                        is_final = token_data.get("is_final", False)

                        if token_data.get("is_silent", False):
                            is_silent = True
                            final_time = time.perf_counter()
                            got_response = True
                            break

                        if first_token_time is None and token_text:
                            first_token_time = time.perf_counter()

                        if token_text:
                            token_count += 1
                            full_text += token_text

                        if is_final:
                            final_time = time.perf_counter()
                            got_response = True
                            break

                    elif msg_type in (9, 10):
                        pass  # TTS chunk / ASR echo, skip

                    elif msg_type == 3:  # Legacy complete text
                        final_time = time.perf_counter()
                        full_text = payload.decode('utf-8')
                        got_response = True
                        break

                except socket.timeout:
                    print(f"  [Round {round_idx}] TIMEOUT")
                    break

            if got_response:
                client_ttft_ms = (
                    (first_token_time - t_send) * 1000
                    if first_token_time else float('nan')
                )
                client_total_ms = (
                    (final_time - t_send) * 1000
                    if final_time else float('nan')
                )

                entry = {
                    "round": round_idx,
                    "elapsed_sec": elapsed,
                    "client_ttft_ms": client_ttft_ms,
                    "client_total_ms": client_total_ms,
                    "token_count": token_count,
                    "is_silent": is_silent,
                    "text": full_text[:60],
                }
                per_round.append(entry)

                status = "SILENT" if is_silent else f"tokens={token_count}"
                ttft_str = f"{client_ttft_ms:.1f}ms" if client_ttft_ms == client_ttft_ms else "N/A"
                if round_idx <= 10 or round_idx % 20 == 0:
                    print(f"  [Round {round_idx:>4}  t={elapsed:>6.1f}s]  "
                          f"TTFT={ttft_str:>9}  total={client_total_ms:.1f}ms  {status}")
            else:
                per_round.append({
                    "round": round_idx, "elapsed_sec": elapsed,
                    "client_ttft_ms": float('nan'), "client_total_ms": float('nan'),
                    "token_count": 0, "is_silent": False, "text": "TIMEOUT",
                })

            # Wait for next send interval (subtract time already spent)
            spent = time.perf_counter() - t_send
            wait = max(0, send_interval - spent)
            if wait > 0:
                time.sleep(wait)

    finally:
        try:
            _tcp_send_message(sock, 4, b"clear")
            time.sleep(0.3)
        except Exception:
            pass
        sock.close()

    # --- Compute summary ---
    actual_duration = time.time() - stream_start if per_round else 0

    spoken_rounds = [r for r in per_round if not r.get("is_silent") and r["client_ttft_ms"] == r["client_ttft_ms"]]
    all_valid = [r for r in per_round if r["client_ttft_ms"] == r["client_ttft_ms"]
                 or r.get("is_silent")]

    # TTFT from spoken rounds (non-silent, non-NaN)
    ttft_vals = [r["client_ttft_ms"] for r in spoken_rounds]
    # For silent rounds, TTFT ≈ client_total (time to silent token)
    silent_ttft = [r["client_total_ms"] for r in per_round
                   if r.get("is_silent") and r["client_total_ms"] == r["client_total_ms"]]
    all_ttft = ttft_vals + silent_ttft

    summary = {
        "config": config,
        "actual_duration_sec": actual_duration,
        "total_rounds": len(per_round),
        "spoken_rounds": len(spoken_rounds),
        "silent_rounds": sum(1 for r in per_round if r.get("is_silent")),
    }

    if all_ttft:
        sorted_ttft = sorted(all_ttft)
        n = len(sorted_ttft)
        summary.update({
            "ttft_mean_ms": statistics.mean(sorted_ttft),
            "ttft_std_ms": statistics.stdev(sorted_ttft) if n > 1 else 0,
            "ttft_p50_ms": sorted_ttft[int(n * 0.5)],
            "ttft_p90_ms": sorted_ttft[int(min(n * 0.9, n - 1))],
            "ttft_p99_ms": sorted_ttft[int(min(n * 0.99, n - 1))],
            "ttft_min_ms": min(sorted_ttft),
            "ttft_max_ms": max(sorted_ttft),
        })

    print(f"\n{'='*60}")
    print(f"📊 Sustained Streaming Summary  (config={config})")
    print(f"{'='*60}")
    print(f"  Duration: {actual_duration:.1f}s  ({actual_duration/60:.1f} min)")
    print(f"  Total rounds: {len(per_round)}")
    print(f"  Spoken: {len(spoken_rounds)}, Silent: {summary['silent_rounds']}")
    if all_ttft:
        print(f"  TTFT (all rounds, incl. silent):")
        print(f"    mean={summary['ttft_mean_ms']:.1f}ms  std={summary['ttft_std_ms']:.1f}ms")
        print(f"    p50={summary['ttft_p50_ms']:.1f}ms  p90={summary['ttft_p90_ms']:.1f}ms  "
              f"p99={summary['ttft_p99_ms']:.1f}ms")
        print(f"    min={summary['ttft_min_ms']:.1f}ms  max={summary['ttft_max_ms']:.1f}ms")
    if ttft_vals:
        st = sorted(ttft_vals)
        n = len(st)
        print(f"  TTFT (spoken rounds only, n={n}):")
        print(f"    mean={statistics.mean(st):.1f}ms  "
              f"p50={st[int(n*0.5)]:.1f}ms  p90={st[int(min(n*0.9,n-1))]:.1f}ms")

    return {
        "config": config,
        "duration_sec": actual_duration,
        "total_rounds": len(per_round),
        "results": per_round,
        "summary": summary,
    }


# ============================================================================
# Server-side log helpers
# ============================================================================

def get_log_line_count(log_path: str) -> int:
    if not os.path.exists(log_path):
        return 0
    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        return sum(1 for _ in f)


def parse_timing_from_log_range(
    log_path: str, start_line: int,
    stride: int = 1, offset: int = 0,
) -> dict:
    """Parse TIMING entries from vllm.log starting at `start_line` (0-based).

    When stride > 1, only keep every `stride`-th inference group starting
    from `offset`.  E.g. stride=2, offset=1 keeps the 2nd, 4th, 6th … group
    (useful for extracting measurement rounds from warmup+measurement pairs).
    """
    ttft_values, ttft_per_frame_values = [], []
    total_gen_values, token_counts, decode_per_token_values = [], [], []
    group_idx = -1  # incremented each time we see a TTFT line (= new inference)

    re_ttft = re.compile(r'\[TIMING\] Time to first token \(TTFT\): ([\d.]+)ms')
    re_total = re.compile(r'\[TIMING\] Total generation time: ([\d.]+)ms')
    re_tokens = re.compile(r'\[TIMING\] Tokens generated: (\d+)')
    re_ttft_frame = re.compile(r'\[TIMING\] TTFT avg\. by \d+ frames: ([\d.]+)ms')
    re_decode = re.compile(
        r'\[TIMING\] Decode phase: [\d.]+ms for \d+ tokens, avg=([\d.]+)ms/token'
    )

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for idx, line in enumerate(f):
            if idx < start_line:
                continue

            m = re_ttft.search(line)
            if m:
                group_idx += 1
                if group_idx % stride == offset:
                    ttft_values.append(float(m.group(1)))
                continue
            m = re_ttft_frame.search(line)
            if m:
                if group_idx % stride == offset:
                    ttft_per_frame_values.append(float(m.group(1)))
                continue
            m = re_total.search(line)
            if m:
                if group_idx % stride == offset:
                    total_gen_values.append(float(m.group(1)))
                continue
            m = re_tokens.search(line)
            if m:
                if group_idx % stride == offset:
                    token_counts.append(int(m.group(1)))
                continue
            m = re_decode.search(line)
            if m:
                if group_idx % stride == offset:
                    decode_per_token_values.append(float(m.group(1)))
                continue

    return {
        "ttft": ttft_values,
        "ttft_per_frame": ttft_per_frame_values,
        "total_gen": total_gen_values,
        "token_counts": token_counts,
        "decode_per_token": decode_per_token_values,
    }


# ============================================================================
# Report generation (server-side metrics only)
# ============================================================================

def generate_report(
    report_path: str,
    audio_path: str = None,
    audio_duration: float = None,
    video_path: str = None,
    video_duration: float = None,
    tts_text: str = None,
    asr_results: list = None,
    llm_miss_results: list = None,
    llm_hit_results: list = None,
    tts_results: list = None,
    server_miss_timing: dict = None,
    server_hit_timing: dict = None,
    sustained_results: list = None,
    server_sustained_timing: dict = None,
):
    """Write server-side latency report to file."""
    import datetime

    lines = []
    def w(s=""):
        lines.append(s)

    def fmt_stats(values, unit="ms"):
        clean = sorted([v for v in values if v == v and v != float('inf')])
        if not clean:
            return "    (no valid data)"
        n = len(clean)
        mean = statistics.mean(clean)
        std = statistics.stdev(clean) if n > 1 else 0
        p50 = clean[int(n * 0.5)]
        p90 = clean[int(min(n * 0.9, n - 1))]
        p99 = clean[int(min(n * 0.99, n - 1))]
        s  = f"    mean={mean:.1f}{unit}  std={std:.1f}{unit}  (n={n})\n"
        s += f"    p50={p50:.1f}{unit}  p90={p90:.1f}{unit}  p99={p99:.1f}{unit}\n"
        s += f"    min={min(clean):.1f}{unit}  max={max(clean):.1f}{unit}"
        return s

    w("=" * 70)
    w("Latency Benchmark Report  (Server-Side Metrics)")
    w(f"Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w("=" * 70)

    # ---- ASR ----
    if asr_results is not None:
        w()
        w("-" * 70)
        w("1. ASR Latency")
        w("-" * 70)
        if audio_path:
            size = os.path.getsize(audio_path) if os.path.exists(audio_path) else 0
            w(f"  Audio: {audio_path} ({size} bytes, duration={audio_duration:.2f}s)")
        w(f"  Runs: {len(asr_results)}")
        w(f"  E2E Latency:")
        w(fmt_stats([r["latency_ms"] for r in asr_results]))

    # ---- LLM: per-run breakdown + cache miss/hit split ----
    stiming = server_miss_timing  # use the main LLM timing
    if stiming and stiming.get("ttft"):
        w()
        w("-" * 70)
        w("2. Qwen3 VL Inference Latency  (server-side, per-run breakdown)")
        w("-" * 70)
        if video_path:
            size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
            w(f"  Video: {video_path} ({size} bytes, duration={video_duration:.1f}s)")

        ttft = stiming["ttft"]
        total = stiming.get("total_gen", [])
        tokens = stiming.get("token_counts", [])
        decode = stiming.get("decode_per_token", [])
        n = len(ttft)

        w(f"  Runs: {n}")
        w()
        header = f"  {'Run':>4}  {'TTFT (prefill+1st decode)':>26}  {'Total Gen':>12}  {'Tokens':>7}  {'Decode':>14}"
        w(header)
        w("  " + "-" * len(header.strip()))
        dec_idx = 0
        for i in range(n):
            t = f"{ttft[i]:.1f}ms"
            tg = f"{total[i]:.1f}ms" if i < len(total) else "-"
            tk = f"{tokens[i]}" if i < len(tokens) else "-"
            has_decode = i < len(tokens) and tokens[i] > 1
            if has_decode and dec_idx < len(decode):
                ds = f"~{decode[dec_idx]:.1f}ms/token"
                dec_idx += 1
            else:
                ds = "-"
            w(f"  {i+1:>4}  {t:>26}  {tg:>12}  {tk:>7}  {ds:>14}")

        # Split: Run 1 = cache miss, Runs 2+ = cache hit
        w()
        if n >= 2:
            w(f"  Run 1 (Prefix Cache MISS):  TTFT = {ttft[0]:.1f}ms")
            warm_ttft = ttft[1:]
            warm_mean = statistics.mean(warm_ttft)
            warm_p50 = sorted(warm_ttft)[len(warm_ttft) // 2]
            w(f"  Run 2+ (Prefix Cache HIT):  TTFT mean = {warm_mean:.1f}ms, "
              f"p50 = {warm_p50:.1f}ms  (n={len(warm_ttft)})")
            if stiming.get("ttft_per_frame"):
                pf = stiming["ttft_per_frame"]
                warm_pf = pf[1:] if len(pf) > 1 else pf
                w(f"  TTFT per frame (cache hit): mean = {statistics.mean(warm_pf):.1f}ms")
            if decode:
                w(f"  Decode speed:  ~{statistics.mean(decode):.1f}ms/token "
                  f"({1000/statistics.mean(decode):.1f} tokens/s)")
        else:
            w(f"  TTFT: {ttft[0]:.1f}ms")

    # ---- TTS ----
    if tts_results is not None:
        w()
        w("-" * 70)
        w("3. TTS Latency")
        w("-" * 70)
        if tts_text:
            w(f"  Text: \"{tts_text}\"")
        w(f"  Runs: {len(tts_results)}")
        w(f"  First Chunk Latency:")
        w(fmt_stats([r["first_chunk_ms"] for r in tts_results]))
        w(f"  Total Latency:")
        w(fmt_stats([r["total_ms"] for r in tts_results]))
        w(f"  Audio Duration:")
        w(fmt_stats([r["audio_duration_s"] for r in tts_results], unit="s"))
        w(f"  RTF (Real-Time Factor):")
        w(fmt_stats([r["rtf"] for r in tts_results], unit=""))

    # ---- E2E breakdown ----
    w()
    w("-" * 70)
    w("4. End-to-End Latency Breakdown  (user speaks -> hears first audio)")
    w("-" * 70)

    asr_mean = tts_fc_mean = tts_fc_p50 = None
    if asr_results:
        asr_mean = statistics.mean([r["latency_ms"] for r in asr_results])
    if tts_results:
        fc_vals = sorted([r["first_chunk_ms"] for r in tts_results])
        tts_fc_mean = statistics.mean(fc_vals)
        tts_fc_p50 = fc_vals[len(fc_vals) // 2]

    if stiming and stiming.get("ttft") and len(stiming["ttft"]) >= 2:
        ttft = stiming["ttft"]
        ttft_miss = ttft[0]
        ttft_hit_mean = statistics.mean(ttft[1:])
        decode_mean = statistics.mean(decode) if decode else 7.0

        w("  User finishes speaking")
        w("    |")
        w(f"  ASR:       ~{asr_mean:.0f}ms" if asr_mean else "  ASR:       (not measured)")
        w("    |")
        w(f"  Prefill:   ~{ttft_hit_mean:.0f}ms     (1s chunk, Prefix Cache HIT)")
        w(f"             ~{ttft_miss:.0f}ms    (Prefix Cache MISS, first request)")
        w("    |")
        w(f"  1st decode: ~{decode_mean:.0f}ms")
        w("    |")
        w(f"  TTS 1st chunk: ~{tts_fc_p50:.0f}ms" if tts_fc_p50 else "  TTS:       (not measured)")
        w("    |")
        w("  ====================================")
        if asr_mean and tts_fc_p50:
            e2e_hit = asr_mean + ttft_hit_mean + decode_mean + tts_fc_p50
            e2e_miss = asr_mean + ttft_miss + decode_mean + tts_fc_p50
            w(f"  E2E (Cache HIT):   ~{e2e_hit:.0f}ms")
            w(f"  E2E (Cache MISS):  ~{e2e_miss:.0f}ms")

    # ---- Sustained Streaming (AURA-style) ----
    if sustained_results:
        w()
        w("-" * 70)
        w("5. Sustained Streaming Benchmark  (AURA-style)")
        w("-" * 70)
        for sr in sustained_results:
            s = sr["summary"]
            w(f"  Config: {s['config']}")
            w(f"  Duration: {s['actual_duration_sec']:.1f}s  "
              f"({s['actual_duration_sec']/60:.1f} min)")
            w(f"  Total rounds: {s['total_rounds']}  "
              f"(spoken={s['spoken_rounds']}, silent={s['silent_rounds']})")
            if "ttft_mean_ms" in s:
                w(f"  TTFT (all rounds):")
                w(f"    mean={s['ttft_mean_ms']:.1f}ms  std={s['ttft_std_ms']:.1f}ms")
                w(f"    p50={s['ttft_p50_ms']:.1f}ms  p90={s['ttft_p90_ms']:.1f}ms  "
                  f"p99={s['ttft_p99_ms']:.1f}ms")
                w(f"    min={s['ttft_min_ms']:.1f}ms  max={s['ttft_max_ms']:.1f}ms")
            w()

        # Time-series: show TTFT trend every ~30s
        for sr in sustained_results:
            results_list = sr["results"]
            if len(results_list) > 20:
                w(f"  TTFT trend over time (config={sr['config']}):")
                bucket_sec = 30
                buckets: dict[int, list] = {}
                for r in results_list:
                    b = int(r["elapsed_sec"] // bucket_sec)
                    ttft_val = r.get("client_ttft_ms", float('nan'))
                    if r.get("is_silent"):
                        ttft_val = r.get("client_total_ms", float('nan'))
                    if ttft_val == ttft_val:  # not NaN
                        buckets.setdefault(b, []).append(ttft_val)
                for b in sorted(buckets.keys()):
                    vals = buckets[b]
                    t_start = b * bucket_sec
                    t_end = t_start + bucket_sec
                    avg = statistics.mean(vals)
                    w(f"    {t_start:>4}-{t_end:>4}s: mean={avg:>7.1f}ms  (n={len(vals)})")
                w()

    if server_sustained_timing and server_sustained_timing.get("ttft"):
        st = server_sustained_timing
        w(f"  Server-side TTFT (from vllm.log):")
        w(fmt_stats(st["ttft"]))
        if st.get("ttft_per_frame"):
            w(f"  Server-side TTFT per frame:")
            w(fmt_stats(st["ttft_per_frame"]))

    w()
    w("=" * 70)

    report_text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    print(f"\n📁 Report written to: {report_path}")
    print(report_text)


# ============================================================================
# TTS Benchmark
# ============================================================================

def benchmark_tts(
    text: str,
    tts_url: str,
    runs: int,
    language: str = "Chinese",
    speaker: str = "Vivian",
) -> list[dict]:
    """
    向 TTS 服务发送文本，流式接收 PCM，测量:
      - first_chunk_latency: 收到首个 PCM chunk 的时间
      - total_latency: 全部 PCM 接收完毕的时间
      - audio_duration: 生成音频时长
      - RTF (Real-Time Factor)
    """
    results = []
    print(f"\n{'='*60}")
    print(f"TTS Benchmark")
    print(f"Text: \"{text}\"")
    print(f"TTS URL: {tts_url}")
    print(f"Runs: {runs}")
    print(f"{'='*60}")

    endpoint = f"{tts_url}/v1/tts/stream"

    for i in range(runs):
        payload = {
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": "",
        }

        t0 = time.perf_counter()
        first_chunk_time = None
        total_samples = 0
        chunk_count = 0
        sample_rate = 24000

        resp = requests.post(endpoint, json=payload, stream=True, timeout=60)
        raw = resp.raw

        while True:
            # TTS wire format: [sample_rate: 4B big-endian uint32] [pcm_len: 4B big-endian uint32] [pcm_data]
            header = raw.read(8)
            if not header or len(header) < 8:
                break
            sr, pcm_len = struct.unpack(">II", header)
            if sr > 0:
                sample_rate = sr

            if pcm_len <= 0:
                break

            pcm_data = b""
            while len(pcm_data) < pcm_len:
                chunk = raw.read(pcm_len - len(pcm_data))
                if not chunk:
                    break
                pcm_data += chunk

            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            total_samples += pcm_len // 2  # int16
            chunk_count += 1

        t_end = time.perf_counter()

        first_chunk_ms = (first_chunk_time - t0) * 1000 if first_chunk_time else float('nan')
        total_ms = (t_end - t0) * 1000
        audio_dur = total_samples / sample_rate if sample_rate > 0 else 0
        rtf = (total_ms / 1000) / audio_dur if audio_dur > 0 else float('inf')

        results.append({
            "run": i + 1,
            "first_chunk_ms": first_chunk_ms,
            "total_ms": total_ms,
            "audio_duration_s": audio_dur,
            "rtf": rtf,
            "chunks": chunk_count,
            "sample_rate": sample_rate,
        })
        print(f"  [{i+1}/{runs}] first_chunk={first_chunk_ms:.1f}ms  total={total_ms:.1f}ms  "
              f"audio={audio_dur:.2f}s  RTF={rtf:.3f}  chunks={chunk_count}")

    return results


# ============================================================================
# 服务端日志解析 —— 从 vllm.log 提取精确的 prefill / decode 计时
# ============================================================================

def parse_vllm_log(log_path: str):
    """
    解析 vllm.log，提取服务端视角的精确计时:
      - TTFT (≈ prefill + 首decode)
      - Decode 速度 (ms/token)
      - ASR 延迟
      - 每帧平均 TTFT

    匹配的日志行格式（来自 generate_response_with_video）:
      ⏱️ [TIMING] Time to first token (TTFT): 1234.5ms
      ⏱️ [TIMING] Total generation time: 5678.9ms
      ⏱️ [TIMING] Tokens generated: 42
      ⏱️ [TIMING] Decode phase: 300.0ms for 41 tokens, avg=7.3ms/token (136.8 tokens/s)
      ⏱️ [TIMING] ASR latency: 200.0ms
      ⏱️ [TIMING] TTFT avg. by 2 frames: 617.2ms
    """
    print(f"\n{'='*60}")
    print(f"📋 解析服务端日志: {log_path}")
    print(f"{'='*60}")

    if not os.path.exists(log_path):
        print(f"  ❌ 文件不存在: {log_path}")
        return

    ttft_values = []
    total_gen_values = []
    token_counts = []
    decode_per_token_values = []
    decode_tokens_per_sec = []
    asr_latency_values = []
    ttft_per_frame_values = []

    re_ttft = re.compile(r'\[TIMING\] Time to first token \(TTFT\): ([\d.]+)ms')
    re_total = re.compile(r'\[TIMING\] Total generation time: ([\d.]+)ms')
    re_tokens = re.compile(r'\[TIMING\] Tokens generated: (\d+)')
    re_decode = re.compile(r'\[TIMING\] Decode phase: [\d.]+ms for \d+ tokens, avg=([\d.]+)ms/token \(([\d.]+) tokens/s\)')
    re_asr = re.compile(r'\[TIMING\] ASR latency: ([\d.]+)ms')
    re_ttft_frame = re.compile(r'\[TIMING\] TTFT avg\. by \d+ frames: ([\d.]+)ms')

    with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            m = re_ttft.search(line)
            if m:
                ttft_values.append(float(m.group(1)))
                continue
            m = re_total.search(line)
            if m:
                total_gen_values.append(float(m.group(1)))
                continue
            m = re_tokens.search(line)
            if m:
                token_counts.append(int(m.group(1)))
                continue
            m = re_decode.search(line)
            if m:
                decode_per_token_values.append(float(m.group(1)))
                decode_tokens_per_sec.append(float(m.group(2)))
                continue
            m = re_asr.search(line)
            if m:
                asr_latency_values.append(float(m.group(1)))
                continue
            m = re_ttft_frame.search(line)
            if m:
                ttft_per_frame_values.append(float(m.group(1)))
                continue

    print(f"\n--- 服务端 LLM 推理指标 (从日志中提取) ---")
    print_stats("TTFT (prefill + 首decode)", ttft_values)
    print_stats("TTFT per frame", ttft_per_frame_values)
    print_stats("Total Generation", total_gen_values)
    if token_counts:
        print(f"  Token count: mean={statistics.mean(token_counts):.1f}, "
              f"min={min(token_counts)}, max={max(token_counts)} (n={len(token_counts)})")
    print_stats("Decode speed (ms/token)", decode_per_token_values)
    if decode_tokens_per_sec:
        print_stats("Decode speed (tokens/s)", decode_tokens_per_sec, unit="")

    print(f"\n--- 服务端 ASR 指标 ---")
    print_stats("ASR E2E Latency", asr_latency_values)

    return {
        "ttft": ttft_values,
        "total_gen": total_gen_values,
        "token_counts": token_counts,
        "decode_per_token": decode_per_token_values,
        "asr_latency": asr_latency_values,
    }


# ============================================================================
# Statistics
# ============================================================================

def print_stats(name: str, values: list[float], unit: str = "ms"):
    """打印延迟统计信息"""
    clean = [v for v in values if v == v and v != float('inf')]  # filter NaN/Inf
    if not clean:
        print(f"  {name}: no valid data")
        return

    clean.sort()
    n = len(clean)
    mean = statistics.mean(clean)
    std = statistics.stdev(clean) if n > 1 else 0
    p50 = clean[int(n * 0.5)]
    p90 = clean[int(min(n * 0.9, n - 1))]
    p99 = clean[int(min(n * 0.99, n - 1))]
    mn, mx = min(clean), max(clean)

    print(f"  {name}:")
    print(f"    mean={mean:.1f}{unit}  std={std:.1f}{unit}")
    print(f"    p50={p50:.1f}{unit}  p90={p90:.1f}{unit}  p99={p99:.1f}{unit}")
    print(f"    min={mn:.1f}{unit}  max={mx:.1f}{unit}  (n={n})")


def summarize(component: str, results: list[dict]):
    """汇总并打印某组件的统计"""
    print(f"\n{'='*60}")
    print(f"📊 {component} Summary ({len(results)} runs)")
    print(f"{'='*60}")

    if component == "ASR":
        print_stats("E2E Latency", [r["latency_ms"] for r in results])
    elif component == "LLM":
        non_silent = [r for r in results if not r.get("is_silent", False)]
        silent_count = len(results) - len(non_silent)
        if silent_count:
            print(f"  ⚠️ {silent_count}/{len(results)} runs got silent response (model chose not to reply)")
        if non_silent:
            print_stats("Client TTFT (含ASR+网络+video_decode+prefill+decode)",
                        [r["client_ttft_ms"] for r in non_silent])
            print_stats("Client Total", [r["client_total_ms"] for r in non_silent])
            tokens = [r["token_count"] for r in non_silent]
            if tokens:
                avg_tok = statistics.mean(tokens)
                if avg_tok > 1:
                    ttft_vals = [r["client_ttft_ms"] for r in non_silent
                                 if r["client_ttft_ms"] == r["client_ttft_ms"]]
                    total_vals = [r["client_total_ms"] for r in non_silent
                                  if r["client_total_ms"] == r["client_total_ms"]]
                    if ttft_vals and total_vals:
                        avg_ttft = statistics.mean(ttft_vals)
                        avg_total = statistics.mean(total_vals)
                        decode_per_token = (avg_total - avg_ttft) / (avg_tok - 1)
                        print(f"  Avg tokens: {avg_tok:.1f}")
                        if decode_per_token > 0:
                            print(f"  Avg decode speed (client): {decode_per_token:.1f}ms/token "
                                  f"({1000/decode_per_token:.1f} tokens/s)")
        print(f"\n  💡 提示: 以上是客户端视角，包含网络和ASR延迟。")
        print(f"     服务端精确计时请用: python benchmark_latency.py --parse-log logs/vllm.log")
    elif component == "TTS":
        print_stats("First Chunk Latency", [r["first_chunk_ms"] for r in results])
        print_stats("Total Latency", [r["total_ms"] for r in results])
        print_stats("Audio Duration", [r["audio_duration_s"] for r in results], unit="s")
        print_stats("RTF", [r["rtf"] for r in results], unit="")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="离线基准测试：独立测量 ASR / Qwen3 VL / TTS 延迟",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全部测试（LLM 测试需要同时提供 video + audio）
  python benchmark_latency.py --video test.webm --audio test.mp3 --tts-text "测试文本" --runs 5

  # 只测 ASR
  python benchmark_latency.py --audio test.mp3 --only asr --runs 10

  # 只测 TTS
  python benchmark_latency.py --tts-text "今天天气真好，阳光明媚。" --only tts --runs 10

  # 只测 LLM (通过TCP，需提供 video + audio 来触发推理)
  python benchmark_latency.py --video test.webm --audio test.mp3 --only llm --runs 5

  # AURA 论文条件: 5分钟持续流式视频，测量稳态 TTFT
  python benchmark_latency.py --video test_1s.webm --only llm \\
      --llm-mode sustained-streaming --stream-duration 300 \\
      --report logs/sustained_report.log --output logs/sustained.json

  # 自定义时长和间隔
  python benchmark_latency.py --video test_1s.webm --only llm \\
      --llm-mode sustained-streaming --stream-duration 60 --send-interval 1.0

  # 解析 vllm.log 获取服务端精确的 prefill / decode 计时
  python benchmark_latency.py --parse-log logs/vllm.log
"""
    )

    parser.add_argument("--video", type=str, help="测试视频路径 (WebM, 1s chunk)")
    parser.add_argument("--audio", type=str, help="测试音频路径 (MP3/WAV)")
    parser.add_argument("--tts-text", type=str, default="你好，这是一个延迟测试的句子。",
                        help="TTS 测试文本")
    parser.add_argument("--runs", type=int, default=5, help="每组件测试次数")
    parser.add_argument("--only", type=str, choices=["asr", "llm", "tts"],
                        help="只测某个组件")

    parser.add_argument("--asr-url", type=str, default="http://localhost:8001/asr")
    parser.add_argument("--tts-url", type=str, default="http://localhost:8002")
    parser.add_argument("--llm-host", type=str, default="localhost")
    parser.add_argument("--llm-port", type=int, default=12345)

    parser.add_argument("--parse-log", type=str, metavar="LOG_PATH",
                        help="解析 vllm.log 提取服务端精确计时 (不做在线测试)")
    parser.add_argument("--output", type=str, help="结果输出 JSON 文件路径")
    parser.add_argument("--llm-mode", type=str, default="cache-miss",
                        choices=["cache-miss", "cache-hit", "both", "sustained-streaming"],
                        help="LLM 测试模式: cache-miss / cache-hit / both / sustained-streaming")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Cache-hit 测试的预热轮数 (默认 2)")
    parser.add_argument("--stream-duration", type=float, default=300.0,
                        help="sustained-streaming 模式持续时长 (秒)，默认 300 (5分钟)")
    parser.add_argument("--send-interval", type=float, default=1.0,
                        help="sustained-streaming 模式视频发送间隔 (秒)，默认 1.0")
    parser.add_argument("--report", type=str, metavar="REPORT_PATH",
                        help="生成可读报告并写入文件")
    parser.add_argument("--vllm-log", type=str, default="logs/vllm.log",
                        help="vLLM 日志路径，用于提取服务端精确计时")

    args = parser.parse_args()

    # --- 日志解析模式 ---
    if args.parse_log:
        parse_vllm_log(args.parse_log)
        return

    all_results = {}
    asr_results = None
    llm_miss_results = None
    llm_hit_results = None
    tts_results = None
    sustained_results = []
    server_miss_timing = None
    server_hit_timing = None
    server_sustained_timing = None

    audio_duration = None
    if args.audio and os.path.exists(args.audio):
        try:
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", args.audio],
                capture_output=True, text=True, timeout=10,
            )
            audio_duration = float(r.stdout.strip())
            print(f"📎 Audio duration: {audio_duration:.2f}s ({args.audio})")
        except Exception:
            pass

    video_duration = None
    if args.video and os.path.exists(args.video):
        try:
            import subprocess
            r = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "csv=p=0", args.video],
                capture_output=True, text=True, timeout=10,
            )
            video_duration = float(r.stdout.strip())
            print(f"📎 Video duration: {video_duration:.2f}s ({args.video})")
        except Exception:
            pass

    # --- ASR ---
    if args.only in (None, "asr"):
        if args.audio:
            asr_results = benchmark_asr(args.audio, args.asr_url, args.runs)
            summarize("ASR", asr_results)
            all_results["asr"] = asr_results
        elif args.only == "asr":
            print("ERROR: --audio is required for ASR benchmark")
            sys.exit(1)

    # --- LLM ---
    if args.only in (None, "llm"):
        if args.llm_mode == "sustained-streaming":
            if not args.video:
                print("ERROR: --video is required for sustained-streaming benchmark")
                sys.exit(1)

            vllm_log = args.vllm_log
            log_pos = get_log_line_count(vllm_log)

            sr = benchmark_llm_sustained_streaming(
                video_path=args.video,
                host=args.llm_host,
                port=args.llm_port,
                duration_sec=args.stream_duration,
                send_interval=args.send_interval,
                config="AURA",
            )
            sustained_results.append(sr)
            all_results["sustained_streaming"] = sr

            time.sleep(1)
            server_sustained_timing = parse_timing_from_log_range(vllm_log, log_pos)

        elif args.video and args.audio:
            vllm_log = args.vllm_log

            if args.llm_mode in ("cache-miss", "both"):
                log_pos = get_log_line_count(vllm_log)
                llm_miss_results = benchmark_llm_tcp(
                    args.video, args.audio, args.llm_host, args.llm_port, args.runs)
                summarize("LLM", llm_miss_results)
                all_results["llm_cache_miss"] = llm_miss_results
                time.sleep(1)
                server_miss_timing = parse_timing_from_log_range(vllm_log, log_pos)

            if args.llm_mode in ("cache-hit", "both"):
                log_pos = get_log_line_count(vllm_log)
                llm_hit_results = benchmark_llm_tcp_cache_hit(
                    args.video, args.audio, args.llm_host, args.llm_port,
                    args.runs, warmup_runs=args.warmup)
                summarize("LLM", llm_hit_results)
                all_results["llm_cache_hit"] = llm_hit_results
                time.sleep(1)
                warmup_per_run = args.warmup
                server_hit_timing = parse_timing_from_log_range(
                    vllm_log, log_pos,
                    stride=warmup_per_run + 1, offset=warmup_per_run)

        elif args.only == "llm":
            if not args.video:
                print("ERROR: --video is required for LLM benchmark")
                sys.exit(1)
            if not args.audio:
                print("ERROR: --audio is required for LLM benchmark (用于触发ASR→推理)")
                sys.exit(1)

    # --- TTS ---
    if args.only in (None, "tts"):
        tts_results = benchmark_tts(args.tts_text, args.tts_url, args.runs)
        summarize("TTS", tts_results)
        all_results["tts"] = tts_results

    # --- Save JSON ---
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\n📁 Results saved to {args.output}")

    # --- Generate report ---
    if args.report:
        generate_report(
            report_path=args.report,
            audio_path=args.audio,
            audio_duration=audio_duration,
            video_path=args.video,
            video_duration=video_duration,
            tts_text=args.tts_text,
            asr_results=asr_results,
            llm_miss_results=llm_miss_results,
            llm_hit_results=llm_hit_results,
            tts_results=tts_results,
            server_miss_timing=server_miss_timing,
            server_hit_timing=server_hit_timing,
            sustained_results=sustained_results if sustained_results else None,
            server_sustained_timing=server_sustained_timing,
        )

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
