"""
Pre-split all chunk videos into 1-second segments (multi-threaded).

Run this BEFORE inference to avoid on-the-fly ffmpeg calls.
The splitting logic mirrors `split_video_from_end` in models/AURA.py.

Usage:
    python presplit_videos.py \
        --anno_path data/ovo_bench_new.json \
        --chunked_dir data/chunked_videos \
        --chunked_1s_dir data/chunked_1s_videos \
        --max_segments 30 \
        --workers 16 \
        --task EPM ASI HLD STU OJR ATR ACR OCR FPD REC SSR CRR
"""

import argparse
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

BACKWARD_TASKS = ["EPM", "ASI", "HLD"]
REALTIME_TASKS = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
FORWARD_TASKS = ["REC", "SSR", "CRR"]

def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def compute_segments(video_path: str, max_segments: int = 30):
    """Return list of (seg_start, seg_end, seg_path) that need cutting."""
    duration = get_video_duration(video_path)
    if duration < 0.01:
        return []

    n_segments = min(max_segments, math.floor(duration))
    if n_segments < 1:
        return []

    t_end = round(duration, 2)
    t_start = round(t_end - n_segments, 2)

    stem = os.path.splitext(os.path.basename(video_path))[0]
    segments = []
    for i in range(n_segments):
        seg_start = round(t_start + i, 2)
        seg_end = round(t_start + i + 1, 2)
        seg_name = f"{stem}_{seg_start:.2f}_{seg_end:.2f}.mp4"
        segments.append((seg_start, seg_end, seg_name))
    return segments


def cut_one_segment(video_path: str, seg_start: float, output_path: str):
    """Cut a single 1-second segment with ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-ss", f"{seg_start:.2f}",
            "-i", video_path,
            "-t", "1",
            "-c:v", "libx264",
            "-c:a", "aac",
            output_path,
        ],
        capture_output=True,
        check=True,
    )


def collect_chunk_video_paths(annotations, tasks, chunked_dir):
    """Collect all unique chunk video paths that will be used during inference."""
    paths = set()
    for anno in annotations:
        if anno["task"] not in tasks:
            continue
        if anno["task"] in BACKWARD_TASKS or anno["task"] in REALTIME_TASKS:
            paths.add(os.path.join(chunked_dir, f"{anno['id']}.mp4"))
        elif anno["task"] in FORWARD_TASKS:
            for i in range(len(anno["test_info"])):
                paths.add(os.path.join(chunked_dir, f"{anno['id']}_{i}.mp4"))
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description="Pre-split videos into 1-sec segments")
    parser.add_argument("--anno_path", type=str, default="data/ovo_bench_new.json")
    parser.add_argument("--chunked_dir", type=str, default="data/chunked_videos")
    parser.add_argument("--chunked_1s_dir", type=str, default="data/chunked_1s_videos")
    parser.add_argument("--max_segments", type=int, default=30)
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel ffmpeg workers")
    parser.add_argument("--task", type=str, nargs="+",
                        default=["EPM", "ASI", "HLD", "STU", "OJR", "ATR",
                                 "ACR", "OCR", "FPD", "REC", "SSR", "CRR"],
                        choices=["EPM", "ASI", "HLD", "STU", "OJR", "ATR",
                                 "ACR", "OCR", "FPD", "REC", "SSR", "CRR"])
    args = parser.parse_args()

    os.makedirs(args.chunked_1s_dir, exist_ok=True)

    with open(args.anno_path, "r") as f:
        annotations = json.load(f)

    chunk_paths = collect_chunk_video_paths(annotations, set(args.task), args.chunked_dir)
    print(f"Found {len(chunk_paths)} chunk videos to process")

    # Phase 1: collect all segment cutting jobs
    jobs = []  # (video_path, seg_start, output_path)
    skipped = 0
    missing = 0
    for vp in chunk_paths:
        if not os.path.exists(vp):
            missing += 1
            continue
        for seg_start, seg_end, seg_name in compute_segments(vp, args.max_segments):
            output_path = os.path.join(args.chunked_1s_dir, seg_name)
            if os.path.exists(output_path):
                skipped += 1
            else:
                jobs.append((vp, seg_start, output_path))

    print(f"Cached segments: {skipped}, missing source videos: {missing}, segments to cut: {len(jobs)}")
    if not jobs:
        print("All segments are ready. No cutting is needed.")
        return

    # Phase 2: multi-threaded cutting
    t0 = time.time()
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(cut_one_segment, vp, ss, op): op
            for vp, ss, op in jobs
        }
        for future in as_completed(futures):
            out = futures[future]
            try:
                future.result()
                done += 1
            except Exception as e:
                failed += 1
                print(f"[FAIL] {out}: {e}")
            if (done + failed) % 200 == 0:
                elapsed = time.time() - t0
                print(f"  Progress: {done + failed}/{len(jobs)}  "
                      f"success={done} failed={failed}  "
                      f"elapsed={elapsed:.1f}s")

    elapsed = time.time() - t0
    print(f"\nDone! success={done}, failed={failed}, total elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
