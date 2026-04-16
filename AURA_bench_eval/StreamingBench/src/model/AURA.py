import base64
import math
import os
import subprocess

from openai import OpenAI
from model.modelclass import Model


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


def split_video_from_end(video_path: str, chunked_1s_dir: str, max_segments: int = 30) -> list:
    """Split video into 1-second segments from the tail end.

    Returns a chronologically ordered list of segment file paths.
    Segments are cached on disk — existing files are not re-cut.
    """
    duration = get_video_duration(video_path)
    if duration < 0.01:
        return [video_path]

    n_segments = min(max_segments, math.floor(duration))
    if n_segments < 1:
        return [video_path]

    t_end = round(duration, 2)
    t_start = round(t_end - n_segments, 2)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(chunked_1s_dir, exist_ok=True)

    segments = []
    for i in range(n_segments):
        seg_start = round(t_start + i, 2)
        seg_end = round(t_start + i + 1, 2)
        seg_name = f"{stem}_{seg_start:.2f}_{seg_end:.2f}.mp4"
        seg_path = os.path.join(chunked_1s_dir, seg_name)
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
        segments.append(seg_path)

    return segments


class AURA(Model):
    def __init__(self, base_url="http://localhost:8028/v1",
                 model_path="aurateam/AURA",
                 chunked_1s_dir="src/data/chunked_1s_videos"):
        self.base_url = base_url
        self.model_name = model_path
        self.chunked_1s_dir = chunked_1s_dir

        self.client = OpenAI(
            api_key="EMPTY",
            base_url=self.base_url,
            timeout=3600,
        )

    def Run(self, file, inp, **kwargs):
        return self._run(file, inp)

    def name(self) -> str:
        return "AURA"

    def _run(self, file, inp):
        segments = split_video_from_end(
            file,
            chunked_1s_dir=self.chunked_1s_dir,
            max_segments=30,
        )

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
                    "text": inp,
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

        return response.choices[0].message.content
