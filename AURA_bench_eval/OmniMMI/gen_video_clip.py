import argparse
import math
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

CHUNK_VIDEO_DIR = "chunkwise_videos"

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".wmv", ".webm"}

DEFAULT_WORKERS = 96


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _cut_one_segment(video_path: str, seg_start: float, seg_path: str):
    """切出单个 1 秒 clip，已存在则跳过。返回 seg_path。"""
    if os.path.exists(seg_path):
        return seg_path
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{seg_start:.2f}",
            "-i",
            video_path,
            "-t",
            "1",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            seg_path,
        ],
        capture_output=True,
        check=True,
    )
    return seg_path


def split_video(video_path: str, executor: ThreadPoolExecutor) -> list:
    """将视频切分为 1 秒的 clip，片段级别并行。"""
    duration = get_video_duration(video_path)
    if duration < 0.01:
        return []

    n_segments = math.floor(duration)
    if n_segments < 1:
        return []

    stem = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(CHUNK_VIDEO_DIR, exist_ok=True)

    futures = {}
    for i in range(n_segments):
        seg_start = round(i, 2)
        seg_end = round(i + 1, 2)
        seg_name = f"{stem}_{seg_start:.2f}_{seg_end:.2f}.mp4"
        seg_path = os.path.join(CHUNK_VIDEO_DIR, seg_name)
        fut = executor.submit(_cut_one_segment, video_path, seg_start, seg_path)
        futures[fut] = i

    segments = [None] * n_segments
    for fut in as_completed(futures):
        idx = futures[fut]
        segments[idx] = fut.result()

    return segments


def main():
    global CHUNK_VIDEO_DIR

    parser = argparse.ArgumentParser(description="将目录中的所有视频切分为 1 秒的 clip")
    parser.add_argument("path", type=str, help="包含视频文件的目录路径")
    parser.add_argument(
        "--chunk_dir", type=str, default=CHUNK_VIDEO_DIR, help="切分后 clip 的存储目录"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"并行 ffmpeg 进程数 (默认 {DEFAULT_WORKERS})",
    )
    args = parser.parse_args()

    CHUNK_VIDEO_DIR = args.chunk_dir

    video_dir = args.path
    if not os.path.isdir(video_dir):
        raise FileNotFoundError(f"目录不存在: {video_dir}")

    video_files = sorted(
        f
        for f in os.listdir(video_dir)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        print(f"在 {video_dir} 中未找到视频文件")
        return

    print(f"共找到 {len(video_files)} 个视频，clip 将存入 {CHUNK_VIDEO_DIR}")
    print(f"并行 worker 数: {args.workers}")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for idx, vf in enumerate(video_files, 1):
            video_path = os.path.join(video_dir, vf)
            print(f"[{idx}/{len(video_files)}] 正在处理: {vf}")
            segments = split_video(video_path, executor)
            print(f"  -> 生成 {len(segments)} 个 clip")

    print("全部完成")


if __name__ == "__main__":
    main()
