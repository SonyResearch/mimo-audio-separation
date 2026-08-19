# Model Unwrapping & Evaluation Guide

This document details how to unwrap training checkpoints for deployment and run evaluations on test datasets.

---

## 1. Unwrapping Model Checkpoints

Training checkpoints contain full optimizer states, schedulers, discriminators, and EMA models. Before deployment or distribution, use `src/unwrap_model.py` to extract only the essential model weights (`backbone_model.pth`) and a minimal configuration (`config.yaml`).

### EMA vs. Online Model Weight Extraction
- If `ema.pth` exists in `--ckpt-dir`, weights are extracted from the **EMA model**.
- If `ema.pth` is absent, weights are copied directly from `backbone_model.pth`.

### Command

```bash
python src/unwrap_model.py \
    --ckpt-dir /path/to/runs/train/exp001/ckpt_dir/step_0100000 \
    --output-dir /path/to/unwrapped_models/exp001
```

### Unwrapped Output Layout

```text
unwrapped_models/exp001/
├── backbone_model.pth  # Clean PyTorch state dict of backbone model
└── config.yaml          # Minimal Hydra config containing model architecture parameters
```

---

## 2. Model Evaluation (`src/evaluate.py`)

`src/evaluate.py` runs inference on test tracks and evaluates separation performance. It supports both **unwrapped model directories** and **raw training checkpoint directories**.

### Test Data Structure

The test dataset directory must organize tracks in sub-folders containing stem `.wav` files:

```text
TEST_AUDIO_DIR/
├── Track_001/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
└── Track_002/
    ├── vocals.wav
    └── ...
```

### Execution Command

```bash
TEST_AUDIO_DIR="/path/to/MUSDB18_HQ/test"
CKPT_DIR="/path/to/unwrapped_models/exp001"  # Or raw checkpoint directory
OUTPUT_DIR="/path/to/runs/evaluate/exp001"

python src/evaluate.py \
    --ckpt-dir ${CKPT_DIR} \
    --iterations 1 2 3 \
    --input-audio-dir ${TEST_AUDIO_DIR} \
    --stems "vocals" "drums/bass/other" \
    --stem-names "vocals" "accompaniment" \
    --output-dir ${OUTPUT_DIR} \
    --sample-rate 44100 \
    --sample-size 352800 \
    --overlap-rate 0.5 \
    --chunk-size 44100 \
    --max-batch-size 10 \
    --use-museval \  # By removing this, the evaluation scheme in the paper is used instead of museval
    --save-audio
```

### Command Line Arguments

| Argument | Description |
| --- | --- |
| `--ckpt-dir` | Path to unwrapped or full checkpoint directory |
| `--iterations` | List of iteration step counts for inference (e.g. `1 2 3`) |
| `--input-audio-dir` | Directory containing ground-truth test track folders |
| `--stems` | Source patterns for ground-truth. `/` sums stems (e.g., `"drums/bass/other"`). `'other'` dynamically groups remaining stems |
| `--stem-names` | Target output names matching `--stems` (e.g., `"vocals" "accompaniment"`) |
| `--output-dir` | Directory where evaluation logs and separated audios will be saved |
| `--sample-rate` | Audio sampling rate in Hz (default: `44100`) |
| `--sample-size` | Chunk frame size for sliding-window inference (default: `352800` = 8 seconds) |
| `--overlap-rate` | Overlap ratio between adjacent window chunks (e.g., `0.5` for 50% overlap) |
| `--chunk-size` | Frame chunk size used when computing metrics to avoid CUDA OOM |
| `--max-batch-size` | Maximum batch size during sliding window inference |
| `--use-museval` | Compute standard SDR/ISR/SIR/SAR metrics using `museval` |
| `--save-audio` | Save separated output audio `.wav` files into `--output-dir` |

---

## 3. Re-evaluating Existing Audio Directories (`src/evaluate_from_dir.py`)

If you have already generated and saved estimated separated audio files, you can evaluate them directly against ground-truth targets using `src/evaluate_from_dir.py` without re-running model inference.

### Command

```bash
TARGET_DIR="/path/to/MUSDB18_HQ/test"
ESTIMATE_DIR="/path/to/runs/evaluate/exp001/estimates"

python src/evaluate_from_dir.py \
    --target-audio-dir ${TARGET_DIR} \
    --estimate-audio-dir ${ESTIMATE_DIR} \
    --stems vocals accompaniment \
    --sample-rate 44100 \
    --chunk-size 44100
```
