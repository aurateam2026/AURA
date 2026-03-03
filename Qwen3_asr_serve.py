# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

# ============================================================
# 必须在导入任何 torch/CUDA 相关库之前设置 multiprocessing 启动方法
# vLLM v1 使用多进程，CUDA 不支持 fork 模式，必须使用 spawn
# ============================================================
import os
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import multiprocessing
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    # 如果已经设置过启动方法，忽略错误
    pass

import io
import base64
import argparse
import uvicorn
import aiofiles
import requests
import numpy as np
import soundfile as sf
import torch
import traceback
from typing import Optional, Tuple, List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

from qwen_asr import Qwen3ASRModel

# 尝试导入 librosa 用于处理 MP3 等格式
try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False
    print("⚠️ librosa not installed. MP3 support may be limited.")


# 初始化 FastAPI 应用
app = FastAPI(title="Qwen3 ASR Service")

# 全局模型变量
asr_model = None

# 默认模型路径
ASR_MODEL_PATH = "Qwen/Qwen3-ASR-1.7B"
FORCED_ALIGNER_PATH = "Qwen/Qwen3-ForcedAligner-0.6B"


class TimeStampInfo(BaseModel):
    text: str
    start_time: float
    end_time: float


class ASRResponse(BaseModel):
    text: str
    language: str
    time_stamps: Optional[List[TimeStampInfo]] = None
    vllm_response: Optional[str] = None


def _read_audio_from_file(file_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    """
    从文件读取音频，支持多种格式（WAV, MP3, FLAC, OGG 等）
    
    优先使用 librosa（支持更多格式），回退到 soundfile
    自动重采样到目标采样率（默认 16000Hz，ASR 模型标准）
    
    Args:
        file_path: 音频文件路径
        target_sr: 目标采样率，默认 16000Hz（ASR 标准）
    
    Returns:
        (audio_data, sample_rate) 元组
    """
    try:
        if HAS_LIBROSA:
            # librosa 支持 MP3, WAV, FLAC, OGG 等多种格式
            # sr=target_sr 会自动重采样到目标采样率
            wav, sr = librosa.load(file_path, sr=target_sr, mono=True)
            print(f"📊 Audio resampled to {sr}Hz (librosa)")
            return np.asarray(wav, dtype=np.float32), int(sr)
        else:
            # 回退到 soundfile（仅支持 WAV, FLAC 等，不支持 MP3）
            wav, sr = sf.read(file_path, dtype="float32", always_2d=False)
            # 手动重采样（如果需要）
            if sr != target_sr:
                print(f"⚠️ Manual resampling from {sr}Hz to {target_sr}Hz (no librosa)")
                # 简单的线性重采样（不如 librosa 精确，但可用）
                duration = len(wav) / sr
                target_length = int(duration * target_sr)
                indices = np.linspace(0, len(wav) - 1, target_length)
                wav = np.interp(indices, np.arange(len(wav)), wav).astype(np.float32)
                sr = target_sr
            return np.asarray(wav, dtype=np.float32), int(sr)
    except Exception as e:
        print(f"❌ Error reading audio file {file_path}: {e}")
        raise


def _read_wav_from_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """从字节数据读取音频（仅支持 WAV 格式）"""
    with io.BytesIO(audio_bytes) as f:
        wav, sr = sf.read(f, dtype="float32", always_2d=False)
    return np.asarray(wav, dtype=np.float32), int(sr)


def _to_data_url_base64(audio_bytes: bytes, mime: str = "audio/wav") -> str:
    """将音频字节转换为 base64 data URL"""
    b64 = base64.b64encode(audio_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def load_qwen3_asr_model(
    model_path: str = ASR_MODEL_PATH,
    forced_aligner_path: str = FORCED_ALIGNER_PATH,
    gpu_memory_utilization: float = 0.8,
    device: str = "cuda:0",
    use_forced_aligner: bool = True,
):
    """加载 Qwen3 ASR 模型"""
    global asr_model
    print(f"Loading Qwen3 ASR model: {model_path}...")
    try:
        forced_aligner = forced_aligner_path if use_forced_aligner else None
        forced_aligner_kwargs = None
        
        if use_forced_aligner:
            forced_aligner_kwargs = dict(
                dtype=torch.bfloat16,
                device_map=device,
            )
        
        asr_model = Qwen3ASRModel.LLM(
            model=model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            forced_aligner=forced_aligner,
            forced_aligner_kwargs=forced_aligner_kwargs,
            max_inference_batch_size=32,
            max_new_tokens=1024,
        )
        print(f"Qwen3 ASR Model {model_path} loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Failed to load model: {e}")


def call_vllm_service(prompt: str, api_url: str):
    """调用 vLLM 服务"""
    headers = {"User-Agent": "ASR Client"}
    
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    pload = {
        "model": "Qwen/Qwen3-VL-30B-A3B-Instruct",
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 2048,
        "stream": False,
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=pload)
        if response.status_code == 200:
            data = response.json()
            if "choices" in data:
                return data["choices"][0]["message"]["content"]
            else:
                return f"Error: Missing 'choices' in response: {data}"
        else:
            return f"Error: vLLM API Request failed with status code {response.status_code}: {response.text}"
    except Exception as e:
        return f"Error calling vLLM service: {str(e)}"


@app.on_event("startup")
async def startup_event():
    """服务启动时加载模型"""
    model_path = os.getenv("QWEN3_ASR_MODEL", ASR_MODEL_PATH)
    forced_aligner_path = os.getenv("QWEN3_FORCED_ALIGNER", FORCED_ALIGNER_PATH)
    gpu_memory_utilization = float(os.getenv("GPU_MEMORY_UTILIZATION", "0.8"))
    device = os.getenv("ASR_DEVICE", "cuda:0")
    use_forced_aligner = os.getenv("USE_FORCED_ALIGNER", "true").lower() == "true"
    
    load_qwen3_asr_model(
        model_path=model_path,
        forced_aligner_path=forced_aligner_path,
        gpu_memory_utilization=gpu_memory_utilization,
        device=device,
        use_forced_aligner=use_forced_aligner,
    )


@app.post("/asr", response_model=ASRResponse)
async def transcribe_audio(
    file: UploadFile = File(...),
    run_vllm: bool = False,
    language: Optional[str] = None,
    context: Optional[str] = None,
    return_time_stamps: bool = False,
):
    """
    转录音频文件
    
    Args:
        file: 上传的音频文件
        run_vllm: 是否调用 vLLM 服务进行后处理
        language: 强制指定语言 (如 "Chinese", "English")，None 表示自动检测
        context: 上下文提示文本，用于提高特定词汇的识别准确率
        return_time_stamps: 是否返回时间戳信息
    """
    if not asr_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 保存上传的文件到临时路径
    temp_filename = f"temp_{file.filename}"
    try:
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        print(f"🎤 Received audio file: {file.filename} ({len(content)} bytes)")
        
        # 读取音频数据（支持 MP3, WAV, FLAC 等多种格式）
        try:
            audio_data, sample_rate = _read_audio_from_file(temp_filename)
            print(f"✅ Audio loaded: {len(audio_data)} samples @ {sample_rate}Hz")
        except Exception as e:
            print(f"❌ Failed to read audio: {e}")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Failed to read audio file: {e}")
        
        # 使用 Qwen3 ASR 进行转录
        print(f"🔄 Transcribing with Qwen3 ASR...")
        results = asr_model.transcribe(
            audio=(audio_data, sample_rate),
            language=language,
            context=context if context else "",
            return_time_stamps=return_time_stamps,
        )
        
        # 获取转录结果
        result = results[0]
        transcribed_text = result.text
        detected_lang = result.language if result.language else "unknown"
        
        # 打印转录结果
        print(f"✅ Transcribed text: {transcribed_text!r}")
        print(f"📝 Detected language: {detected_lang}")
        
        # 处理时间戳
        time_stamps_list = None
        if return_time_stamps and result.time_stamps:
            time_stamps_list = [
                TimeStampInfo(
                    text=ts.text,
                    start_time=ts.start_time,
                    end_time=ts.end_time
                )
                for ts in result.time_stamps
            ]
        
        # 调用 vLLM 服务 (Optional)
        vllm_resp = None
        if run_vllm:
            vllm_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
            vllm_resp = call_vllm_service(transcribed_text, vllm_url)
        
        return ASRResponse(
            text=transcribed_text,
            language=detected_lang,
            time_stamps=time_stamps_list,
            vllm_response=vllm_resp
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        if os.path.exists(temp_filename):
            os.remove(temp_filename)


@app.post("/asr/batch")
async def transcribe_audio_batch(
    files: List[UploadFile] = File(...),
    run_vllm: bool = False,
    languages: Optional[str] = None,
    contexts: Optional[str] = None,
    return_time_stamps: bool = False,
):
    """
    批量转录多个音频文件
    
    Args:
        files: 上传的音频文件列表
        run_vllm: 是否调用 vLLM 服务进行后处理
        languages: 语言列表，用逗号分隔 (如 "Chinese,English,None")
        contexts: 上下文列表，用 ||| 分隔
        return_time_stamps: 是否返回时间戳信息
    """
    if not asr_model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # 解析语言和上下文参数
    lang_list = None
    if languages:
        lang_list = [l.strip() if l.strip().lower() != "none" else None for l in languages.split(",")]
    
    ctx_list = None
    if contexts:
        ctx_list = [c.strip() for c in contexts.split("|||")]
    
    temp_files = []
    audio_data_list = []
    
    try:
        # 读取所有音频文件
        for i, file in enumerate(files):
            temp_filename = f"temp_batch_{i}_{file.filename}"
            temp_files.append(temp_filename)
            
            async with aiofiles.open(temp_filename, 'wb') as out_file:
                content = await file.read()
                await out_file.write(content)
            
            print(f"🎤 Batch file {i}: {file.filename} ({len(content)} bytes)")
            
            # 读取音频数据（支持 MP3, WAV, FLAC 等多种格式）
            audio_data, sample_rate = _read_audio_from_file(temp_filename)
            audio_data_list.append((audio_data, sample_rate))
        
        # 批量转录
        results = asr_model.transcribe(
            audio=audio_data_list,
            language=lang_list if lang_list else [None] * len(files),
            context=ctx_list if ctx_list else [""] * len(files),
            return_time_stamps=return_time_stamps,
        )
        
        # 处理结果
        responses = []
        for i, result in enumerate(results):
            time_stamps_list = None
            if return_time_stamps and result.time_stamps:
                time_stamps_list = [
                    TimeStampInfo(
                        text=ts.text,
                        start_time=ts.start_time,
                        end_time=ts.end_time
                    )
                    for ts in result.time_stamps
                ]
            
            vllm_resp = None
            if run_vllm:
                vllm_url = os.getenv("VLLM_API_URL", "http://localhost:8000/v1/chat/completions")
                vllm_resp = call_vllm_service(result.text, vllm_url)
            
            responses.append(ASRResponse(
                text=result.text,
                language=result.language if result.language else "unknown",
                time_stamps=time_stamps_list,
                vllm_response=vllm_resp
            ))
        
        return responses

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # 清理临时文件
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)


def main():
    parser = argparse.ArgumentParser(description="Start Qwen3 ASR Service")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind")
    parser.add_argument("--vllm-url", type=str, default="http://localhost:8000/v1/chat/completions", help="vLLM API URL")
    parser.add_argument("--model", type=str, default=ASR_MODEL_PATH, help="Qwen3 ASR model path")
    parser.add_argument("--forced-aligner", type=str, default=FORCED_ALIGNER_PATH, help="Forced aligner model path")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device for forced aligner")
    parser.add_argument("--no-forced-aligner", action="store_true", help="Disable forced aligner (no timestamps)")
    
    args = parser.parse_args()
    
    # 设置环境变量供 startup 使用
    os.environ["QWEN3_ASR_MODEL"] = args.model
    os.environ["QWEN3_FORCED_ALIGNER"] = args.forced_aligner
    os.environ["GPU_MEMORY_UTILIZATION"] = str(args.gpu_memory_utilization)
    os.environ["ASR_DEVICE"] = args.device
    os.environ["USE_FORCED_ALIGNER"] = "false" if args.no_forced_aligner else "true"
    os.environ["VLLM_API_URL"] = args.vllm_url
    
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

    # 使用示例:
    # CUDA_VISIBLE_DEVICES=0 python Qwen3_asr_serve.py --host 0.0.0.0 --port 8001
    # 
    # 不使用 forced aligner (无时间戳功能):
    # CUDA_VISIBLE_DEVICES=0 python Qwen3_asr_serve.py --host 0.0.0.0 --port 8001 --no-forced-aligner

    # ./cloudflared tunnel --url http://localhost:5003   
