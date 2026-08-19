# Inference Guide

This document details how to run source separation on arbitrary mixture audio files using `src/inference.py` and trained / unwrapped model checkpoints.

---

## 1. Overview

`src/inference.py` allows you to apply a trained or unwrapped iterative separation model to separate mixture audio files into individual stems (e.g., vocals, drums, bass, other).

Key features:
- **Flexible Checkpoints**: Automatically loads from unwrapped directories (`backbone_model.pth` + `config.yaml`) or training run directories containing EMA checkpoints (`ema.pth`).
- **Recursive Audio Search**: Scans the input directory recursively for supported audio file formats (`.wav`, `.flac`, `.mp3`, etc.).
- **Automatic Audio Preprocessing**: Resamples input audio to the target sample rate and adjusts channel count (mono-to-stereo or channel clipping) to match the model.
- **Overlapping Chunked Inference**: Processes arbitrarily long audio tracks using sliding window inference with cross-fading to prevent seam artifacts.
- **Multi-Iteration Support**: Allows extracting separated audio stems at specific iterative refinement steps (e.g., iteration 1, 2, 3) or at the final step.

---

## 2. Quick Command Examples

### Standard Usage (Single Iteration)
Separate audio files in `/path/to/mixtures` using the model's maximum iteration step:

```bash
python src/inference.py \
    --ckpt-dir /path/to/unwrapped_models/exp001 \
    --input-audio-dir /path/to/mixtures \
    --output-dir /path/to/output_stems \
    --stem-names vocals drums bass other \
    --iterations 3
```

### Multi-Iteration Output
Save output stems at multiple intermediate iterative refinement steps (e.g., iterations 1, 2, and 3):

```bash
python src/inference.py \
    --ckpt-dir /path/to/unwrapped_models/exp001 \
    --input-audio-dir /path/to/mixtures \
    --output-dir /path/to/output_stems \
    --stem-names vocals drums bass other \
    --iterations 1 2 3 \
    --out-ext wav
```

---

## 3. Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--ckpt-dir` | `str` | **Required** | Path to checkpoint directory containing `config.yaml` and model weights (`ema.pth` or `backbone_model.pth`). |
| `--input-audio-dir` | `str` | **Required** | Input directory containing mixture audio files (scanned recursively). |
| `--output-dir` | `str` | **Required** | Output directory to save separated audio stems. |
| `--stem-names` | `str ...` | **Required** | List of stem names corresponding to model output sources (must match `model.num_sources`). E.g., `vocals drums bass other` or `vocals accompaniment`. |
| `--iterations` | `int ...` | `None` (uses `max_iter`) | Specific iterative refinement step(s) to output (e.g., `3` or `1 2 3`). |
| `--ext-audio` | `str ...` | `wav flac mp3` | Extensions of input audio files to search for. |
| `--out-ext` | `str` | `flac` | Output audio format extension (e.g., `flac`, `wav`). FLAC files are saved with 16-bit PCM. |
| `--sample-rate` | `int` | `44100` | Target sample rate (Hz) for model inference. Audio is resampled if needed. |
| `--sample-size` | `int` | `352800` | Chunk window size in samples for model inference (e.g., 352800 = 8.0s at 44.1 kHz). |
| `--overlap-rate` | `float` | `0.5` | Overlap ratio (0.0 to 1.0) between adjacent inference chunks for cross-fading. |
| `--max-batch-size` | `int` | `10` | Maximum batch size of audio chunks processed concurrently during inference. |

---

## 4. Output Directory Structure

The directory structure of the output depends on whether single or multiple iterations are specified in `--iterations`.

### Single Iteration Output (e.g., `--iterations 3`)
Stems are saved directly under the track sub-directory inside `--output-dir`:

```text
output_dir/
├── Track_01/
│   ├── vocals.flac
│   ├── drums.flac
│   ├── bass.flac
│   └── other.flac
└── Track_02/
    ├── vocals.flac
    ├── drums.flac
    ├── bass.flac
    └── other.flac
```

### Multiple Iteration Outputs (e.g., `--iterations 1 2 3`)
Stems for each specified iteration step are saved under separate `iter_<N>` sub-directories:

```text
output_dir/
├── Track_01/
│   ├── iter_1/
│   │   ├── vocals.flac
│   │   └── ...
│   ├── iter_2/
│   │   ├── vocals.flac
│   │   └── ...
│   └── iter_3/
│       ├── vocals.flac
│       └── ...
└── Track_02/
    ├── iter_1/
    │   ├── vocals.flac
    │   └── ...
    ├── iter_2/
    │   ├── vocals.flac
    │   └── ...
    └── iter_3/
        ├── vocals.flac
        └── ...
```

---

## 5. Technical Details & Audio Processing

1. **Automatic Weight Selection**:
   - `src/inference.py` checks for `ema.pth` first (Exponential Moving Average weights used during training).
   - If `ema.pth` is absent, it directly loads `backbone_model.pth`.
2. **Channel & Sample Rate Handling**:
   - Audio is automatically converted to match the model's channel requirements (e.g., mono audio is duplicated to stereo if the model expects 2 channels).
   - Input audio with non-matching sample rates is automatically resampled using `julius.resample_frac`.
3. **Sliding Window Chunk Processing**:
   - Long audio files are split into overlapping chunks of length `--sample-size` with `--overlap-rate` (default 50%).
   - Cross-fading windowing (`separate` function in `src/evaluate.py`) is applied to seamlessly merge chunks without boundary artifacts.
4. **Mixed Precision Support**:
   - Reads the mixed precision configuration (`trainer.amp`) from `config.yaml` and executes inference under PyTorch `torch.inference_mode()` with Accelerate.
