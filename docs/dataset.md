# Dataset Preparation & Structure

This document details the audio dataset folder structure, configuration parameters, and metadata generation scripts used in this repository.

---

## Data Structure

The dataloader (`data.AudioDataset`) expects each track to be stored in its own sub-directory under a root dataset folder. Each track directory should contain the individual audio stems.

```text
ROOT_DIR/
├── Track_001/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
├── Track_002/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   └── other.wav
└── metadata.csv  # Optional (recommended for fast initialization)
```

- Supported audio formats include `.wav`, `.flac`, `.mp3`, etc. (configurable via Hydra config).
- Stems for each track must match the stem names defined in the configuration (e.g., `inst_list: ['vocals', 'drums', 'bass', 'other']`).

---

## Generating Metadata CSV (`metadata.csv`)

Scanning large dataset directories recursively at startup can introduce significant overhead, especially on network file systems or multi-node training clusters.

To eliminate startup delay, you can pre-generate a `metadata.csv` file inside the dataset root directory using `script/make_musdb_metadata_csv.py`.

### Execution Command

```bash
python script/make_musdb_metadata_csv.py --root-dir /path/to/MUSDB18_HQ/train
```

### How It Works

The script performs the following steps:
1. Iterates over all immediate subdirectories (tracks) under `--root-dir`.
2. Inspects a representative stem file (`vocals.wav` by default) using `soundfile` to extract audio length, sample rate, and channel count.
3. Outputs `metadata.csv` into the dataset root directory with the following structure:

| Header | Description |
| --- | --- |
| `id` | Normalized track identifier |
| `path` | Relative path to the track sub-directory |
| `sample_rate` | Audio sampling rate in Hz (e.g., `44100`) |
| `num_frames` | Total audio frames in the track |
| `num_channels` | Number of audio channels (e.g., `2`) |

When `metadata.csv` exists in `root_dirs`, the dataloader loads metadata directly from the CSV, resulting in near-instant dataset initialization.

---

## Dataset Configuration via Hydra

Data configurations are managed using Hydra under `configs/data/`. Below is an example configuration for 2-source separation (Vocals vs. Accompaniment):

```yaml
train:
  _target_: data.AudioDataset
  sample_rate: 44100
  sample_length: 352800 # 8 seconds at 44.1 kHz
  out_channels: 'stereo'
  audio_modules:
    musdb18:
      root_dirs: ['/path/to/MUSDB18_HQ/train']
      metadata_name: 'metadata'
      inst_list: ['vocals', 'bass', 'drums', 'other']
      gather_list: [[0], [1, 2, 3]] # Stem 0 (vocals) vs. Stems 1, 2, 3 (accompaniment)
      ext: 'wav'
      random_shift: true
      aug_mix: true
      aug_volume: true
      aug_flip: true
      aug_ch_shuffle: true
      aug_silence: true
      ensure_non_silent: true
```

### Key Parameters Explained

- **`sample_rate`**: Target sampling rate. Audio files will be resampled automatically if needed.
- **`sample_length`**: Length of audio segments extracted during training (in frames). Set to `null` for full-track processing during validation.
- **`inst_list`**: List of individual stem audio filenames (without extension).
- **`gather_list`**: Index groups defining target sources. In the example above:
  - Group `[0]` corresponds to `vocals`.
  - Group `[1, 2, 3]` sums `bass`, `drums`, and `other` to form `accompaniment`.
- **Augmentation options**:
  - `aug_mix`: Random channel mixing across different tracks.
  - `aug_volume`: Random volume gain adjustments.
  - `aug_flip`: Random phase inversion.
  - `aug_ch_shuffle`: Random stereo channel swapping.
  - `aug_silence`: Random insertion of silent segments.
