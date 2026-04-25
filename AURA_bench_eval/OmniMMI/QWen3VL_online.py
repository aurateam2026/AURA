import base64
import math
import os
import subprocess

from openai import OpenAI

SYSTEM_PROMPT = "You are receiving a live video stream where the final frame is the present moment. Respond only when a response is needed based on the user's message or the visual context. Otherwise, output `<|silent|>` to signify silence."


def encode_video_base64(video_path: str) -> str:
    with open(video_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


CHUNK_VIDEO_DIR = "/scratch/dyvm6xra/dyvm6xrauser36/stream_benchs_datasets/OmniMMI/data/chunkwise_videos"

MIN_VALID_SEGMENT_SIZE = 500  # bytes; an empty mp4 container header is ~261 bytes


def split_video_from_end(video_path: str, max_segments: int = 30, end_time: float = None) -> list:
    """Split video into 1-second segments from the tail end.

    Returns a chronologically ordered list of segment file paths.
    Segments are cached on disk — existing files are not re-cut.
    Naming convention: ``{stem}_{t_start}_{t_end}.mp4``
    where t_start / t_end are kept to two decimal places.
    """
    real_duration = get_video_duration(video_path)
    duration = min(real_duration, end_time) if end_time is not None else real_duration
    if duration < 0.01:
        return [video_path]

    n_segments = min(max_segments, math.floor(duration))
    if n_segments < 1:
        return [video_path]

    t_end = round(duration, 2)
    t_start = round(t_end - n_segments, 2)
    if t_start < 0:
        t_start = 0.0
        n_segments = math.floor(t_end)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(CHUNK_VIDEO_DIR, exist_ok=True)

    segments = []
    for i in range(n_segments):
        seg_start = round(t_start + i, 2)
        seg_end = round(t_start + i + 1, 2)
        seg_name = f"{stem}_{seg_start:.2f}_{seg_end:.2f}.mp4"
        seg_path = os.path.join(CHUNK_VIDEO_DIR, seg_name)
        if not os.path.exists(seg_path):
            subprocess.run(
                [
                    "ffmpeg", "-y",
                    "-ss", f"{seg_start:.2f}",
                    "-i", video_path,
                    "-t", "1",
                    "-c:v", "libx264",
                    "-c:a", "aac",
                    seg_path,
                ],
                capture_output=True,
                check=True,
            )
        if os.path.getsize(seg_path) >= MIN_VALID_SEGMENT_SIZE:
            segments.append(seg_path)

    if not segments:
        return [video_path]
    return segments


class EvalQWen3VL_online:
    def __init__(self, args) -> None:
        self.args = args
        self.frame_fps = 1.0
        self.frame_resolution = None

        if isinstance(args, dict):
            base_url = args.get("base_url", "http://localhost:8028/v1")
            self.model_name = args.get("model_path", None) or "aurateam/AURA"
        else:
            base_url = getattr(args, "base_url", "http://localhost:8028/v1")
            self.model_name = getattr(args, "model_path", None) or "aurateam/AURA"

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
            timeout=3600,
        )

    def load_video(self, video_path):
        self.video_path = video_path
        self.video_duration = get_video_duration(video_path)

    def inference_PA(self, video_file_name, prompt, question_time=None):
        segments = split_video_from_end(video_file_name, max_segments=30, end_time=question_time)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        })
        messages.append({
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "OK, I will let you know then.",
                },
            ],
        })

        for seg_path in segments[:-1]:
            video_b64 = encode_video_base64(seg_path)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    },
                ],
            })
            messages.append({
                "role": "assistant",
                "content": "<|silent|>",
            })

        last_b64 = encode_video_base64(segments[-1])
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{last_b64}"},
                },
            ],
        })

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=128,
            temperature=0,
            extra_body={
                "mm_processor_kwargs": {
                    "fps": 1.0,
                    "size": {
                        "shortest_edge": 4096,
                        "longest_edge": 602112,
                    },
                }
            },
        )

        output_text = response.choices[0].message.content
        return output_text

    def inference(self, video_file_name, prompt, question_time=None):
        segments = split_video_from_end(video_file_name, max_segments=30, end_time=question_time)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        for seg_path in segments[:-1]:
            video_b64 = encode_video_base64(seg_path)
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": f"data:video/mp4;base64,{video_b64}"},
                    },
                ],
            })
            messages.append({
                "role": "assistant",
                "content": "<|silent|>",
            })

        last_b64 = encode_video_base64(segments[-1])
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"data:video/mp4;base64,{last_b64}"},
                },
                {
                    "type": "text",
                    "text": prompt,
                },
            ],
        })

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=128,
                temperature=0,
                extra_body={
                    "mm_processor_kwargs": {
                        "fps": 1.0,
                        "size": {
                            "shortest_edge": 4096,
                            "longest_edge": 602112,
                        },
                    }
                },
            )
        except Exception as e:
            print(f"[ERROR] video={video_file_name}, question_time={question_time}, num_segments={len(segments)}")
            for i, seg in enumerate(segments):
                sz = os.path.getsize(seg) if os.path.exists(seg) else -1
                print(f"[ERROR]   seg[{i}]: {seg}  size={sz}")
            raise

        output_text = response.choices[0].message.content
        return output_text
