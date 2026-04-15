"""
    Inference and save results to results/[model]/
"""

import argparse
import os
import json
from models import *
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, "models"))

parser = argparse.ArgumentParser(description='Run OVOBench')
parser.add_argument("--anno_path", type=str, default="data/ovo_bench_new.json", help="Path to the annotations")
parser.add_argument("--video_dir", type=str, default="data/src_videos", help="Root directory of source videos")
parser.add_argument("--chunked_dir", type=str, default="data/chunked_videos", help="Root directory of chunked videos")
parser.add_argument("--chunked_1s_dir", type=str, default="data/chunked_1s_videos", help="Directory for cached 1-second video chunks")
parser.add_argument("--result_dir", type=str, default="results", help="Root directory of results")
parser.add_argument("--mode", type=str, required=True, choices=["online", "offline"], help="Online of Offline model for testing")
parser.add_argument("--task", type=str, required=False, nargs="+", \
                    choices=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"], \
                    default=["EPM", "ASI", "HLD", "STU", "OJR", "ATR", "ACR", "OCR", "FPD", "REC", "SSR", "CRR"], \
                    help="Tasks to evaluate")
parser.add_argument("--model", type=str, required=True, help="Model to evaluate")
parser.add_argument("--base_url", type=str, default="http://localhost:8028/v1", help="Base URL for OpenAI-compatible API")
parser.add_argument("--save_results", type=bool, default=True, help="Save results to a file")
args = parser.parse_args()

print(f"Inference Model: {args.model}; Task: {args.task}")

if args.model == "AURA":
    from models.AURA import EvalAURA
    model = EvalAURA(args)
else:
    raise ValueError(f"Unsupported model: {args.model}. Please implement the model.")

with open(args.anno_path, "r") as f:
    annotations = json.load(f)

for i, item in enumerate(annotations):
    annotations[i]["video"] = os.path.join(args.video_dir, item["video"])

backward_anno = []
realtime_anno = []
forward_anno = []
backward_tasks = ["EPM", "ASI", "HLD"]
realtime_tasks = ["STU", "OJR", "ATR", "ACR", "OCR", "FPD"]
forward_tasks = ["REC", "SSR", "CRR"]

for anno in annotations:
    if anno["task"] in args.task:
        if anno["task"] in backward_tasks:
            backward_anno.append(anno)
        if anno["task"] in realtime_tasks:
            realtime_anno.append(anno)
        if anno["task"] in forward_tasks:
            forward_anno.append(anno)

anno = {
    "backward": backward_anno,
    "realtime": realtime_anno,
    "forward": forward_anno
}

model.eval(anno, args.task, args.mode)