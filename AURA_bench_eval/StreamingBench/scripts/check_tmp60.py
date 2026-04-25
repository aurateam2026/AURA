"""Quick & dirty validator for tmp_60 clips.

Multi-threaded ffprobe over every .mp4 under TMP60_DIR.
Reports zero-byte files and files that ffprobe cannot read / report a
positive duration. Optionally deletes them so presplit_videos.py will
recreate them on the next run.

Edit the CONFIG block below and run:
    python check_tmp60.py
"""

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---------- CONFIG ----------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STREAMINGBENCH_ROOT = os.path.dirname(SCRIPT_DIR)
TMP60_DIR = os.path.join(STREAMINGBENCH_ROOT, "src", "data", "videos", "tmp_60")
WORKERS = 16
MIN_DURATION = 0.1   # seconds; anything shorter is treated as broken
DELETE_BROKEN = False  # set True to actually rm broken files
# ----------------------------


def probe(path):
    """Return (path, ok, reason). ok=True means file is fine."""
    try:
        size = os.path.getsize(path)
    except OSError as e:
        return path, False, f"stat failed: {e}"

    if size == 0:
        return path, False, "0 bytes"

    try:
        out = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return path, False, "ffprobe timeout"
    except Exception as e:
        return path, False, f"ffprobe crashed: {e}"

    if out.returncode != 0:
        msg = (out.stderr or "").strip().splitlines()
        tail = msg[-1] if msg else "unknown error"
        return path, False, f"ffprobe rc={out.returncode}: {tail}"

    text = out.stdout.strip()
    try:
        duration = float(text)
    except ValueError:
        return path, False, f"bad duration output: {text!r}"

    if duration < MIN_DURATION:
        return path, False, f"duration={duration:.3f}s"

    return path, True, ""


def main():
    if not os.path.isdir(TMP60_DIR):
        print(f"[ERR] TMP60_DIR not found: {TMP60_DIR}")
        return

    files = [
        os.path.join(TMP60_DIR, f)
        for f in os.listdir(TMP60_DIR)
        if f.endswith(".mp4")
    ]
    files.sort()
    print(f"Scanning {len(files)} files in {TMP60_DIR}")
    print(f"workers={WORKERS} min_duration={MIN_DURATION}s delete_broken={DELETE_BROKEN}")

    broken = []
    ok = 0
    checked = 0
    total = len(files)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(probe, p) for p in files]
        for fut in as_completed(futures):
            path, good, reason = fut.result()
            checked += 1
            if good:
                ok += 1
            else:
                broken.append((path, reason))
                print(f"[BAD] {os.path.basename(path)}  ->  {reason}")
            if checked % 200 == 0:
                print(f"  progress: {checked}/{total}  ok={ok}  bad={len(broken)}")

    print()
    print(f"Done. ok={ok}  bad={len(broken)}  total={total}")

    if broken:
        log = os.path.join(SCRIPT_DIR, "tmp60_broken.txt")
        with open(log, "w") as f:
            for p, r in sorted(broken):
                f.write(f"{p}\t{r}\n")
        print(f"Broken file list written to: {log}")

        if DELETE_BROKEN:
            removed = 0
            for p, _ in broken:
                try:
                    os.remove(p)
                    removed += 1
                except OSError as e:
                    print(f"[ERR] cannot remove {p}: {e}")
            print(f"Deleted {removed}/{len(broken)} broken files.")


if __name__ == "__main__":
    main()
