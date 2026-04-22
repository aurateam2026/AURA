"""
Pre-split all video clips into 1-second segments (multi-threaded).

Run BEFORE inference to avoid on-the-fly ffmpeg calls during AURA evaluation.
The splitting logic mirrors `split_video_from_end` in src/model/AURA.py.

This script reads annotation JSONs, cuts intermediate clips (tmp_60) from the
original videos in src/data/videos/, then splits those clips into 1-second
segments stored in --chunked_1s_dir.

Usage:
    python presplit_videos.py \
        --data_files ../src/data/questions_real.json ../src/data/questions_omni.json \
                     ../src/data/questions_sqa.json ../src/data/questions_proactive.json \
        --context_time -1 \
        --chunked_1s_dir ../src/data/chunked_1s_videos \
        --max_segments 30 \
        --workers 16
"""

import argparse
import ffmpeg
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMINGBENCH_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_CHUNKED_1S_DIR = os.path.join(
    STREAMINGBENCH_ROOT, "src", "data", "chunked_1s_videos"
)


def get_video_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "quiet",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def timestamp_to_seconds(ts: str) -> int:
    return sum(int(x) * 60 ** i for i, x in enumerate(reversed(ts.split(":"))))


def compute_segments(video_path: str, max_segments: int = 30):
    """Return list of (seg_start, seg_name) for a given clip video."""
    try:
        duration = get_video_duration(video_path)
    except Exception:
        print(f"[WARN] ffprobe cannot read, skipping: {video_path}")
        return []
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
        segments.append((seg_start, seg_name))
    return segments


def cut_one_segment(video_path: str, seg_start: float, output_path: str):
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


def ensure_clip(video_path: str, start_time: int, end_time: int) -> str:
    """Create the tmp_60 clip from the original video if it doesn't exist.

    Mirrors the logic in video_execution.py split_video().
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), "tmp_60")
    os.makedirs(output_dir, exist_ok=True)
    clip_path = os.path.join(output_dir, f"{video_name}_{start_time}_{end_time}.mp4")

    if not os.path.exists(clip_path):
        try:
            (
                ffmpeg
                .input(video_path, ss=int(start_time))
                .output(
                    clip_path,
                    t=(int(end_time) - int(start_time)),
                    vcodec="libx264",
                    acodec="aac",
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
        except ffmpeg.Error as e:
            raise RuntimeError(e.stderr.decode("utf-8")) from e
    return clip_path


def collect_clips_from_real_omni(data, context_time):
    """For real / omni benchmarks: data is list[dict]."""
    clips = set()
    for subset in data:
        video_path = subset["video_path"]
        for question in subset["questions"]:
            ts = timestamp_to_seconds(question["time_stamp"])
            start = max(0, ts - context_time) if context_time > 0 else 0
            clips.add((video_path, start, ts))
    return clips


def collect_clips_from_sqa(data, context_time):
    """For SQA benchmark: data is list[list[dict]]."""
    clips = set()
    for video_data in data:
        for subset in video_data:
            video_path = subset["video_path"]
            for question in subset["questions"]:
                ts = timestamp_to_seconds(question["time_stamp"])
                start = max(0, ts - context_time) if context_time > 0 else 0
                clips.add((video_path, start, ts))
    return clips


def collect_clips_from_proactive(data, context_time):
    """For proactive benchmark: each question polls every second from start to ground_truth+4."""
    clips = set()
    for subset in data:
        video_path = subset["video_path"]
        for question in subset["questions"]:
            start_time = timestamp_to_seconds(question["time_stamp"])
            gt_time = timestamp_to_seconds(question["ground_truth_time_stamp"])
            max_time = gt_time + 4

            current_time = start_time + 1
            while current_time <= max_time:
                s = max(start_time, current_time - context_time) if context_time > 0 else start_time
                clips.add((video_path, s, current_time))
                current_time += 1
    return clips


def detect_data_type(data):
    """Heuristic to detect data format: real/omni, sqa, or proactive."""
    if not data:
        return "unknown"
    first = data[0]
    if isinstance(first, list):
        return "sqa"
    if isinstance(first, dict):
        questions = first.get("questions", [])
        if questions and "ground_truth_time_stamp" in questions[0]:
            return "proactive"
        return "real_omni"
    return "unknown"


def main():
    parser = argparse.ArgumentParser(description="Pre-split StreamingBench clips into 1-sec segments")
    parser.add_argument("--data_files", type=str, nargs="+", required=True,
                        help="Annotation JSON files (questions_real.json, questions_omni.json, etc.)")
    parser.add_argument("--chunked_1s_dir", type=str, default=DEFAULT_CHUNKED_1S_DIR,
                        help="Output directory for cached 1-second video segments")
    parser.add_argument("--context_time", type=int, default=-1,
                        help="-1 means all context (0, query_time); >0 means (query_time - t, query_time)")
    parser.add_argument("--max_segments", type=int, default=30)
    parser.add_argument("--workers", type=int, default=16,
                        help="Number of parallel ffmpeg workers used in phase 1 and phase 3")
    args = parser.parse_args()

    os.makedirs(args.chunked_1s_dir, exist_ok=True)

    # Collect all (video_path, start, end) from annotation files
    all_clip_info = set()
    for data_file in args.data_files:
        print(f"Loading annotation file: {data_file}")
        with open(data_file, "r") as f:
            data = json.load(f)

        dtype = detect_data_type(data)
        if dtype == "real_omni":
            clips = collect_clips_from_real_omni(data, args.context_time)
        elif dtype == "sqa":
            clips = collect_clips_from_sqa(data, args.context_time)
        elif dtype == "proactive":
            clips = collect_clips_from_proactive(data, args.context_time)
        else:
            print(f"  Unrecognized data format, skipping: {data_file}")
            continue
        print(f"  type={dtype}, collected {len(clips)} clips")
        all_clip_info |= clips

    all_clip_info = sorted(all_clip_info)
    print(f"\nFound {len(all_clip_info)} unique clips to process")

    # Phase 1: create tmp_60 clips from original videos
    print("\n=== Phase 1: Creating tmp_60 clips from original videos ===")
    clip_paths = []
    clip_created = 0
    clip_cached = 0
    clip_failed = 0
    clip_jobs = []
    for video_path, start, end in all_clip_info:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        tmp_dir = os.path.join(os.path.dirname(video_path), "tmp_60")
        expected = os.path.join(tmp_dir, f"{video_name}_{start}_{end}.mp4")

        if os.path.exists(expected):
            clip_cached += 1
            clip_paths.append(expected)
            continue

        if not os.path.exists(video_path):
            clip_failed += 1
            print(f"[WARN] Source video not found: {video_path}")
            continue

        clip_jobs.append((video_path, start, end))

        if (clip_created + clip_cached + clip_failed) > 0 and (clip_created + clip_cached + clip_failed) % 100 == 0:
            print(f"  Progress: {clip_created + clip_cached + clip_failed}/{len(all_clip_info)}  "
                  f"created={clip_created} cached={clip_cached} failed={clip_failed}")

    if clip_jobs:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(ensure_clip, video_path, start, end): (video_path, start, end)
                for video_path, start, end in clip_jobs
            }
            for future in as_completed(futures):
                video_path, start, end = futures[future]
                try:
                    path = future.result()
                    clip_paths.append(path)
                    clip_created += 1
                except Exception as e:
                    clip_failed += 1
                    print(f"[FAIL] clip {video_path} [{start}-{end}]: {e}")

                if (clip_created + clip_cached + clip_failed) % 100 == 0:
                    print(f"  Progress: {clip_created + clip_cached + clip_failed}/{len(all_clip_info)}  "
                          f"created={clip_created} cached={clip_cached} failed={clip_failed}")

    print(f"Clips: created={clip_created}, cached={clip_cached}, failed={clip_failed}")

    # Phase 2: collect 1-second segment cutting jobs
    print("\n=== Phase 2: Splitting clips into 1-second segments ===")
    jobs = []       # (clip_path, seg_start, output_path)
    skipped = 0
    for cp in clip_paths:
        for seg_start, seg_name in compute_segments(cp, args.max_segments):
            output_path = os.path.join(args.chunked_1s_dir, seg_name)
            if os.path.exists(output_path):
                skipped += 1
            else:
                jobs.append((cp, seg_start, output_path))

    print(f"Cached segments: {skipped}, segments to cut: {len(jobs)}")
    if not jobs:
        print("All segments are ready. No cutting is needed.")
        return

    # Phase 3: multi-threaded segment cutting
    t0 = time.time()
    done = 0
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(cut_one_segment, cp, ss, op): op
            for cp, ss, op in jobs
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
