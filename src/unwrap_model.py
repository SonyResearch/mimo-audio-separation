"""
Copyright (C) 2026 Yukara Ikemiya

====================
Extract only model weights required for inference from a model checkpoint directory.
- If the checkpoint contains an EMA model, the model weights are extracted from that.
- If not, the model weights are extracted from the online model.
"""

import sys
from pathlib import Path
import argparse
import shutil

import torch
import hydra
from omegaconf import OmegaConf
from ema_pytorch import EMA

sys.dont_write_bytecode = True

KEEP_CONFIG = ["data.train.sample_rate", "data.train.sample_length", "data.train.out_channels",
               "trainer.amp"]


def argparse_setup():
    parser = argparse.ArgumentParser(description="Unwrap a model checkpoint for inference.")
    parser.add_argument("--ckpt-dir", type=str, required=True, help="Checkpoint directory.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory to save the unwrapped checkpoint.")
    return parser.parse_args()


def main():
    args = argparse_setup()
    ckpt_dir = Path(args.ckpt_dir)
    output_dir = Path(args.output_dir)
    assert ckpt_dir.is_dir(), f"Checkpoint directory {ckpt_dir} does not exist."
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading a model from {ckpt_dir}...")

    cfg_path = ckpt_dir / 'config.yaml'
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found in {ckpt_dir}. Expected 'config.yaml'.")

    cfg = OmegaConf.load(cfg_path)
    model = hydra.utils.instantiate(cfg.model)

    if (ckpt_dir / 'ema.pth').exists():
        ckpt_path = ckpt_dir / 'ema.pth'
        ema = EMA(model, **cfg.trainer.ema)
        ema.load_state_dict(torch.load(ckpt_path, map_location='cpu', weights_only=False))
        model = ema.ema_model
        state_dict = model.state_dict()
        print(state_dict.keys())
        # extract backbone_model weights from the state_dict
        state_dict = {k.replace('backbone_model.', ''): v for k, v in state_dict.items() if k.startswith('backbone_model.')}

        # dict to state_dict (just in case)
        # model.backbone_model.load_state_dict(state_dict)
        # state_dict = model.backbone_model.state_dict()

        # save the state_dict to the output directory
        torch.save(state_dict, output_dir / 'backbone_model.pth')
    elif (ckpt_dir / 'backbone_model.pth').exists():
        ckpt_path = ckpt_dir / 'backbone_model.pth'
        # copy the pth file to the output directory
        shutil.copy(ckpt_path, output_dir / 'backbone_model.pth')
    else:
        raise FileNotFoundError(f"No checkpoint file found in {ckpt_dir}. Expected 'ema.pth' or 'backbone_model.pth'.")

    # save config.yaml
    cfg_model = OmegaConf.create({"model": cfg.model})
    for key in KEEP_CONFIG:
        value = OmegaConf.select(cfg, key)
        if value is not None:
            OmegaConf.update(cfg_model, key, value)

    OmegaConf.save(cfg_model, output_dir / 'config.yaml')

    print(f"Unwrapped model saved to: {output_dir}")


if __name__ == "__main__":
    main()
