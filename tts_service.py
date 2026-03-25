# coding=utf-8
# Standalone TTS Service
# Extracted from Qwen3_VL_online_streaming_v2_ContextManaged.py
#
# Provides streaming PCM audio generation via HTTP API.
# Supports both Base (voice clone) and CustomVoice models.

import argparse
import os
import struct
import sys
import threading
import time
from contextlib import asynccontextmanager

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Add Qwen3-TTS-streaming to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "Qwen3-TTS-streaming"))

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
tts_model = None
tts_model_type = "base"
voice_clone_prompt_cache = None
tts_lock = threading.Lock()

_ref_audio_path = None
_ref_text = None
_language = "Chinese"


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------
class TTSRequest(BaseModel):
    text: str
    language: str = "Chinese"
    speaker: str = "Vivian"
    instruct: str = ""


class CloneRequest(BaseModel):
    ref_audio_path: str
    ref_text: str


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_tts_model(args):
    global tts_model, tts_model_type, voice_clone_prompt_cache
    global _ref_audio_path, _ref_text, _language

    _ref_audio_path = getattr(args, "ref_audio", None)
    _ref_text = getattr(args, "ref_text", None)
    _language = getattr(args, "language", "Chinese")

    from qwen_tts import Qwen3TTSModel
    import torch

    gpu_idx = int(args.gpu)
    num_devices = torch.cuda.device_count()
    if gpu_idx >= num_devices:
        print(f"⚠️  GPU {gpu_idx} out of bounds (0-{num_devices-1}), using last device")
        gpu_idx = num_devices - 1
    device = f"cuda:{gpu_idx}"

    print(f"🔊 Loading TTS model: {args.model} on {device}")
    tts_model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    print("🚀 Enabling streaming optimizations...")
    tts_model.enable_streaming_optimizations(
        decode_window_frames=80,
        use_compile=True,
        compile_mode="reduce-overhead",
    )

    tts_model_type = getattr(tts_model.model, "tts_model_type", "base")
    print(f"✓ TTS model loaded (type={tts_model_type})")

    # Pre-build voice clone prompt if ref audio provided
    if _ref_audio_path and tts_model_type != "custom_voice":
        _build_voice_clone_prompt(_ref_audio_path, _ref_text)


def _build_voice_clone_prompt(audio_path: str, ref_text: str | None):
    global voice_clone_prompt_cache
    if not os.path.exists(audio_path):
        print(f"⚠️ Reference audio not found: {audio_path} (cwd={os.getcwd()})")
        return

    text = ref_text or (
        "这是凯蒂的弟弟，我的同学。你的手怎么了？你为什么不穿衣服？他有很多武术奖项。"
        "凯蒂告诉过我，对吗，莱奥？你知道你打败了谁吗，莱娅？"
        "摸摸这些肌肉，我不知道你有一只这么棒的猫。生于月亮之下。"
        "莱娅总是能挖出一些奇特的东西。是的，只是可惜它占据了她几乎所有的时间。"
        "我不明白这破烂为什么不能等你和妹妹玩完再等。"
    )
    print(f"🎤 Creating voice clone prompt from: {audio_path}")
    print(f"🎤 Reference text: {text[:60]}...")
    voice_clone_prompt_cache = tts_model.create_voice_clone_prompt(
        ref_audio=audio_path,
        ref_text=text,
    )
    print(f"✅ Voice clone prompt created: type={type(voice_clone_prompt_cache)}, "
          f"is_none={voice_clone_prompt_cache is None}")


# ---------------------------------------------------------------------------
# Audio generation (generator)
# ---------------------------------------------------------------------------
def _generate_pcm(text: str, language: str, speaker: str, instruct: str):
    """Yield (pcm_int16_bytes, sample_rate) tuples."""
    if tts_model_type == "custom_voice":
        print(f"🎤 [Fast] CustomVoice ({speaker}): {text[:40]}...")
        t0 = time.time()
        wavs, sr = tts_model.generate_custom_voice(
            text=text,
            speaker=speaker,
            language=language or "Auto",
            instruct=instruct or None,
            do_sample=True,
            temperature=0.85,
            top_k=30,
        )
        dt = time.time() - t0
        dur = len(wavs[0]) / sr
        print(f"🎤 [Fast] {dur:.2f}s audio in {dt:.2f}s (RTF={dt/dur:.2f})")
        audio_int16 = (wavs[0] * 32767).clip(-32768, 32767).astype(np.int16)
        yield audio_int16.tobytes(), sr
    else:
        global voice_clone_prompt_cache
        if voice_clone_prompt_cache is None and _ref_audio_path:
            _build_voice_clone_prompt(_ref_audio_path, _ref_text)

        print(f"🎤 [Stream] VoiceClone: {text[:40]}...")
        lang = language if language in ("Chinese", "English", "Russian", "Japanese", "Korean") else "Chinese"
        for chunk, sr in tts_model.stream_generate_voice_clone(
            text=text,
            language=lang,
            voice_clone_prompt=voice_clone_prompt_cache,
            emit_every_frames=2,
            decode_window_frames=60,
            overlap_samples=256,
        ):
            audio_int16 = (chunk * 32767).clip(-32768, 32767).astype(np.int16)
            yield audio_int16.tobytes(), sr


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="Qwen3 TTS Service", lifespan=lifespan)


@app.get("/v1/tts/health")
def health():
    return {
        "status": "ok" if tts_model is not None else "not_loaded",
        "model_type": tts_model_type,
        "voice_clone_ready": voice_clone_prompt_cache is not None,
    }


@app.post("/v1/tts/clone")
def update_clone(req: CloneRequest):
    if tts_model is None:
        raise HTTPException(503, "TTS model not loaded")
    with tts_lock:
        _build_voice_clone_prompt(req.ref_audio_path, req.ref_text)
    return {"status": "ok", "voice_clone_ready": voice_clone_prompt_cache is not None}


@app.post("/v1/tts/stream")
def stream_tts(req: TTSRequest):
    """Stream PCM audio chunks.

    Wire format per chunk:
        [sample_rate : 4 bytes big-endian uint32]
        [pcm_length  : 4 bytes big-endian uint32]
        [pcm_data    : pcm_length bytes, int16 LE]
    """
    if tts_model is None:
        raise HTTPException(503, "TTS model not loaded")

    if not req.text.strip():
        raise HTTPException(400, "Empty text")

    def _iter():
        with tts_lock:
            try:
                for pcm_bytes, sr in _generate_pcm(
                    req.text, req.language, req.speaker, req.instruct
                ):
                    header = struct.pack(">II", sr, len(pcm_bytes))
                    yield header + pcm_bytes
            except Exception as e:
                print(f"TTS stream error: {e}")
                import traceback
                traceback.print_exc()

    return StreamingResponse(_iter(), media_type="application/octet-stream")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Standalone TTS Service")
    p.add_argument("--port", type=int, default=8002)
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--gpu", type=str, default="0")
    p.add_argument("--model", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    p.add_argument("--ref-audio", type=str, default=None,
                   help="Reference audio for voice cloning (5-15s)")
    p.add_argument("--ref-text", type=str, default=None,
                   help="Transcript of the reference audio")
    p.add_argument("--language", type=str, default="Chinese",
                   choices=["Chinese", "English", "Russian", "Japanese", "Korean"])
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    load_tts_model(args)
    print(f"🌐 TTS service starting on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
