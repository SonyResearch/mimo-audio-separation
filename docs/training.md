# Model Training Guide

This document provides step-by-step instructions for training iterative separation models using Hydra and PyTorch Accelerate.

---

## 1. Environment Setup

It is strongly recommended to use a containerized environment (Singularity/Apptainer) to ensure consistent CUDA, PyTorch, and dependency versions across clusters.

### Building the Singularity Image

```bash
cd container
./build_singularity.bash
```

This generates `container/iterative-separation.sif`.

### Weights & Biases (WandB) Setup

Training metrics and validation samples are logged to [Weights & Biases](https://wandb.ai/). Obtain your API key from [wandb.ai/authorize](https://wandb.ai/authorize) and export it as an environment variable:

```bash
export WANDB_API_KEY="your_wandb_api_key_here"
```

---

## 2. Training from Scratch

Training is launched via `src/train.py` using [Hydra](https://hydra.cc/) configurations. PyTorch Distributed Data Parallel (DDP) is managed via `torchrun` and HuggingFace Accelerate.

### Launching Single-Node Multi-GPU Training

```bash
ROOT_DIR="/path/to/this/repository"
DATASET_DIR="/path/to/dataset"
CONTAINER_PATH="${ROOT_DIR}/container/iterative-separation.sif"

MASTER_ADDR=$(hostname)
MASTER_PORT=12345
export WANDB_API_KEY="your_wandb_api_key_here"

singularity exec --nv --pwd ${ROOT_DIR} \
    -B ${ROOT_DIR} -B ${DATASET_DIR} \
    --env HYDRA_FULL_ERROR=1 \
    --env MASTER_PORT=${MASTER_PORT} \
    --env MASTER_ADDR=${MASTER_ADDR} \
    --env WANDB_API_KEY=${WANDB_API_KEY} \
    ${CONTAINER_PATH} \
torchrun --nproc_per_node gpu \
${ROOT_DIR}/src/train.py \
    model=bsroformer-2src_base_71M \
    data=musdb18_src-2_sec-8 \
    optimizer=separator_and_disc \
    data.train.audio_modules.musdb18.root_dirs=[/path/to/MUSDB18_HQ/train] \
    trainer.output_dir=/path/to/runs/train/exp001 \
    trainer.batch_size=48 \
    trainer.num_workers=4 \
    trainer.logger.project_name=iterative-separation \
    trainer.logger.run_name=exp001
```

### Key Config Arguments Explained

| Parameter | Description |
| --- | --- |
| `model=` | Model backbone configuration under `configs/model/` (e.g., `bsroformer-2src_base_71M`, `scnet-4src_large`, `melroformer-2src_small`) |
| `data=` | Dataset configuration under `configs/data/` (e.g., `musdb18_src-2_sec-8`) |
| `optimizer=` | Optimizer & scheduler settings under `configs/optimizer/` (e.g., `separator_and_disc`, `separator_only`) |
| `data.train.audio_modules.<name>.root_dirs` | Target training dataset directory list |
| `trainer.output_dir` | Directory where checkpoints and logs will be saved |
| `trainer.batch_size` | Global batch size across all GPUs |
| `trainer.num_workers` | PyTorch DataLoader worker threads per GPU process |

---

## 3. Resuming Training & Fine-Tuning

### Checkpoint Structure

During training, checkpoints are saved under `trainer.output_dir` in the following directory layout:

```text
ckpt_dir/
├── backbone_model.pth
├── optimizer.pth
├── scheduler.pth
├── (discriminator files if GAN loss enabled)
├── ema.pth
├── config.yaml
└── states.json
```

### Resuming Training

To resume training from an interrupted state (retaining optimizer and scheduler states):

```bash
torchrun --nproc_per_node gpu ${ROOT_DIR}/src/train.py \
    trainer.ckpt_dir=/path/to/runs/train/exp001/ckpt_dir/step_0050000
```

### Fine-Tuning

To initialize training from existing model weights while **reinitializing** optimizer and learning rate scheduler states, pass `trainer.finetuning=true`:

```bash
torchrun --nproc_per_node gpu ${ROOT_DIR}/src/train.py \
    trainer.ckpt_dir=/path/to/runs/train/exp001/ckpt_dir/step_0050000 \
    trainer.finetuning=true
```
