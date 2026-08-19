"""
Copyright (C) 2026 Yukara Ikemiya
"""

import sys
sys.dont_write_bytecode = True

import os
import argparse
import warnings
warnings.filterwarnings("ignore")

import hydra
import torch
import torchaudio
import julius
from omegaconf import OmegaConf
from accelerate import Accelerator

from evaluate import separate
from utils.torch_common import print_once


def main():
    parser = argparse.ArgumentParser(description="Inference script to separate mixture audio files.")
    parser.add_argument('--ckpt-dir', type=str, required=True, help="Checkpoint directory containing config.yaml and model weights.")
    parser.add_argument('--input-audio-dir', type=str, required=True, help="Input directory containing mixture audio files.")
    parser.add_argument('--output-dir', type=str, required=True, help="Output directory to save separated stems.")
    parser.add_argument('--stem-names', type=str, required=True, nargs='+', help="Names of output stems (e.g. vocals drums bass other).")
    parser.add_argument('--iterations', type=int, nargs='+', default=None, help="Iteration list for inference.")
    parser.add_argument('--ext-audio', type=str, nargs='+', default=['wav', 'flac', 'mp3'], help="Extensions of input audio files.")
    parser.add_argument('--out-ext', type=str, default='flac', help="Extension of output separated stems (e.g. flac, wav).")
    parser.add_argument('--sample-rate', type=int, default=44100, help="Sample rate for audio processing.")
    parser.add_argument('--sample-size', type=int, default=352800, help="Sample size for model inference.")
    parser.add_argument('--overlap-rate', type=float, default=0.5, help="Overlap rate for cross-fading.")
    parser.add_argument('--max-batch-size', type=int, default=10, help="Max batch size for inference.")

    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    input_audio_dir = args.input_audio_dir
    output_dir = args.output_dir
    stem_names = args.stem_names
    iterations = args.iterations
    sample_rate = args.sample_rate
    sample_size = args.sample_size
    overlap_rate = args.overlap_rate
    max_batch_size = args.max_batch_size
    out_ext = args.out_ext.lstrip('.')
    overlap = int(sample_size * overlap_rate)

    # Normalize extensions list
    ext_list = args.ext_audio if isinstance(args.ext_audio, list) else [args.ext_audio]
    ext_audio_list = [ext.lstrip('.').lower() for ext in ext_list]

    cfg_ckpt = OmegaConf.load(os.path.join(ckpt_dir, 'config.yaml'))
    amp = getattr(cfg_ckpt.trainer, 'amp', 'no') or 'no'

    # Get max iteration from config
    max_iter = getattr(cfg_ckpt.model, 'max_iter', 1)
    if iterations is None:
        iterations = [max_iter]
    elif max(iterations) > max_iter:
        warnings.warn(f"Max iteration is {max_iter}, but got iterations={iterations}. Using max_iter.")
        iterations = [max_iter]

    # Check the number of stem names
    assert len(stem_names) == cfg_ckpt.model.num_sources, \
        f"The number of stem names must be equal to the number of sources. " \
        f"Expected: {cfg_ckpt.model.num_sources}, Got: {len(stem_names)}"

    accel = Accelerator(mixed_precision=amp)
    device = accel.device

    print_once(f"Checkpoint dir  : {ckpt_dir}")
    print_once(f"Input audio dir : {input_audio_dir}")
    print_once(f"Output dir      : {output_dir}")

    # Load model
    model = hydra.utils.instantiate(cfg_ckpt.model)
    ema_path = os.path.join(ckpt_dir, "ema.pth")
    backbone_path = os.path.join(ckpt_dir, "backbone_model.pth")

    if os.path.exists(ema_path):
        print_once("->-> Loading EMA checkpoint.")
        from ema_pytorch import EMA
        ema = EMA(model, **cfg_ckpt.trainer.ema)
        ema.load_state_dict(torch.load(ema_path, weights_only=False))
        model = ema.ema_model
    elif os.path.exists(backbone_path):
        print_once("->-> No EMA checkpoint found. Directly loading the backbone model.")
        model.backbone_model.load_state_dict(torch.load(backbone_path, weights_only=False))
    else:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")

    model.eval()
    print_once("->-> Successfully loaded a checkpoint.")

    model = accel.prepare(model)
    model = accel.unwrap_model(model)

    # Collect all audio files in input_audio_dir
    audio_files = []
    for root, _, files in os.walk(input_audio_dir):
        for f in files:
            ext = os.path.splitext(f)[1].lstrip('.').lower()
            if ext in ext_audio_list:
                audio_files.append(os.path.join(root, f))

    audio_files.sort()
    print_once(f"->-> Found {len(audio_files)} audio files in {input_audio_dir}")

    if len(audio_files) == 0:
        print_once("No audio files found. Exiting.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for idx, file_path in enumerate(audio_files):
        rel_path = os.path.relpath(file_path, input_audio_dir)
        original_name = os.path.splitext(rel_path)[0]

        print_once(f"{idx + 1}/{len(audio_files)}: Processing {file_path}")

        mixture, sr = torchaudio.load(file_path)  # (ch, L)
        if sr != sample_rate:
            mixture = julius.resample_frac(mixture, sr, sample_rate)

        num_channels = getattr(model, "num_channels", mixture.shape[0])
        if mixture.shape[0] != num_channels:
            if mixture.shape[0] == 1 and num_channels == 2:
                mixture = mixture.repeat(2, 1)
            elif mixture.shape[0] > num_channels:
                mixture = mixture[:num_channels]

        mixture = mixture.to(device)
        ch, L = mixture.shape

        with torch.inference_mode():
            audios_sep = separate(
                model=model,
                audio=mixture,
                sample_size=sample_size,
                overlap=overlap,
                batch_size=max_batch_size,
                iterations=iterations,
                accel=accel
            )  # (n_iterations, n_src, ch, L)

        n_src = audios_sep.shape[1]

        for it_idx, it in enumerate(iterations):
            if len(iterations) == 1:
                target_output_dir = os.path.join(output_dir, original_name)
            else:
                target_output_dir = os.path.join(output_dir, original_name, f"iter_{it}")

            os.makedirs(target_output_dir, exist_ok=True)

            for i_src in range(n_src):
                stem_filename = f"{stem_names[i_src]}.{out_ext}"
                out_path = os.path.join(target_output_dir, stem_filename)

                audio_stem = audios_sep[it_idx, i_src].cpu()
                if out_ext == "flac":
                    torchaudio.save(out_path, audio_stem, sample_rate=sample_rate,
                                    encoding="PCM_S", bits_per_sample=16)
                else:
                    torchaudio.save(out_path, audio_stem, sample_rate=sample_rate)

        print_once(f"-> Saved stems to {os.path.join(output_dir, original_name)}")


if __name__ == '__main__':
    main()
