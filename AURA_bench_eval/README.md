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
```

## Required Transformers Patch

After installation, edit two lines in `.venv/lib/python3.11/site-packages/transformers/models/qwen3_vl/video_processing_qwen3_vl.py` to make processing behave correctly for AURA's default 1-second video chunks.

### 1. Line 44

Change:

```python
raise ValueError(f"t:{num_frames} must be larger than temporal_factor:{temporal_factor}")
```

to:

```python
num_frames = temporal_factor
```

### 2. Line 100

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
  --workers 16 \
  --task EPM ASI HLD STU OJR ATR ACR OCR FPD REC SSR CRR
```

You can increase `--workers` according to your CPU resources.

### 3. Run Inference

After pre-splitting, run inference script:

```bash
bash scripts/inference/AURA.sh
```

### 4. Run Scoring

After inference finishes, run scoring script:

```bash
bash scripts/score/AURA.sh
```