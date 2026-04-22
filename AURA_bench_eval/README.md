# AURA Benchmark Evaluation

This directory contains the benchmark evaluation setup for AURA.

## Quick Install

Run the following commands inside `AURA_bench_eval`:

```bash
uv venv --python 3.11 --seed
source .venv/bin/activate
uv pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0
uv pip install vllm==0.11.0 --torch-backend=auto
uv pip install transformers==4.57.1
uv pip install ffmpeg-python==0.2.0
```

## Required Transformers Patch

After installation, edit two lines in `.venv/lib/python3.11/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py` for AURA's default 1-second video chunks.

### 1. In `smart_resize`

If you install the same version as above (`transformers==4.57.1`), this change is at line 44.

Change:

```python
raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
```

to:

```python
num_frames = temporal_factor
```

### 2. In `Qwen3VLVideoProcessor`

If you install the same version as above (`transformers==4.57.1`), this change is at line 100.

Change:

```python
min_frames = 4
```

to:

```python
min_frames = 2
```

## Model Deployment

After the environment is ready and the patch above is applied, run:

```bash
bash deploy_aura_vllm.sh
```

This command starts a `vllm` server for AURA and should be kept running in a separate terminal while you run the benchmark scripts below. You can modify `CUDA_VISIBLE_DEVICES` and `PORT` in `deploy_aura_vllm.sh` according to your setup.

## OVO-Bench Evaluation

### 1. Prepare Data

Download the `chunked_videos.tar.parta[a~o]` files from [JoeLeelyf/OVO-Bench](https://huggingface.co/datasets/JoeLeelyf/OVO-Bench/tree/main), extract them, and place the extracted directory at: `OVO-Bench/data/chunked_videos`

### 2. Recommended: Pre-split 1-second Videos in Advance

Inference uses 1-second chunk videos. If these chunks are cut on the fly during inference, the extra `ffmpeg` calls can be slow. It is recommended to pre-split them with high parallelism in advance:

```bash
cd OVO-Bench
python presplit_videos.py \
  --anno_path data/ovo_bench_new.json \
  --chunked_dir data/chunked_videos \
  --chunked_1s_dir data/chunked_1s_videos \
  --max_segments 30 \
  --workers 32 \
  --task EPM ASI HLD STU OJR ATR ACR OCR FPD REC SSR CRR
```

You can increase `--workers` according to your CPU resources.

### 3. Run Inference

After pre-splitting, edit `scripts/inference/AURA.sh` to set `HOSTNAME` and `PORT` according to the IP address and port of your deployed AURA service (the default assumes a local deployment at `localhost:8028`), then run:

```bash
bash scripts/inference/AURA.sh
```

### 4. Run Scoring

After inference finishes, run scoring script:

```bash
bash scripts/score/AURA.sh
```

## StreamingBench Evaluation

### 1. Prepare Data

Download the StreamingBench dataset from [mjuicem/StreamingBench](https://huggingface.co/datasets/mjuicem/StreamingBench), extract the files, and place them under `StreamingBench/data` (for the exact dataset layout, see the official StreamingBench instructions: [THUNLP-MT/StreamingBench](https://github.com/THUNLP-MT/StreamingBench/tree/main)).

Then enter `StreamingBench/scripts` and run the preprocessing script to move videos and update paths in the annotation JSONs:

```bash
cd StreamingBench/scripts
bash preprocess.sh
```

### 2. Recommended: Pre-split 1-second Videos in Advance

AURA inference splits each video clip into 1-second segments. Pre-splitting with high parallelism avoids slow on-the-fly `ffmpeg` calls. The script reads annotation JSONs, creates intermediate clips (`tmp_60`) from the original videos, then splits those clips into 1-second segments. Run the following command from `StreamingBench/scripts`:

```bash
python presplit_videos.py \
  --data_files ../src/data/questions_real.json ../src/data/questions_omni.json ../src/data/questions_sqa.json ../src/data/questions_proactive.json \
  --chunked_1s_dir ../src/data/chunked_1s_videos \
  --context_time -1 \
  --max_segments 30 \
  --workers 32
```

You can increase `--workers` according to your CPU resources.

### 3. Run Inference

Still in `StreamingBench/scripts`, edit `eval.sh` to set `HOSTNAME` and `PORT` according to the IP address and port of your deployed AURA service (the default assumes a local deployment at `localhost:8028`), then run:

```bash
bash eval.sh
```

This will evaluate AURA on four tasks: real-time visual understanding, omni-source understanding, sequential question answering, and proactive output. The task grouping here follows the StreamingBench codebase used for evaluation. In our technical report, we follow the original task grouping in the StreamingBench paper; the underlying sub-tasks are the same, but their assignment to high-level task categories differs slightly. Please refer to the StreamingBench paper or our technical report for the exact mapping.

### 4. Run Scoring

After inference finishes, still in `StreamingBench/scripts`, run the scoring script to compute accuracy statistics:

```bash
bash stats.sh
```