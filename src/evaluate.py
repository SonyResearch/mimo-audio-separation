"""
Copyright (C) 2026 Yukara Ikemiya
"""

import sys
sys.dont_write_bytecode = True

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import argparse
import math
import yaml
import typing as tp
import glob

import hydra
import torch
from torch.nn import functional as F
from torch import nn
import numpy as np
import torchaudio
from accelerate import Accelerator
from omegaconf import OmegaConf
from einops import rearrange
from ema_pytorch import EMA
import julius

from utils.torch_common import get_rank, print_once
from eval import BSSEval, SpecEval, SilentSegmentEval

sys.dont_write_bytecode = True

# Disable user warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def make_audio_batch(audio, sample_size: int, overlap: int, return_last_short: bool = True):
    """
    audio : (ch, L)
    """
    assert 0 <= overlap < sample_size
    L = audio.shape[-1]
    shift = sample_size - overlap

    if return_last_short:
        n_split = math.ceil(max(L - overlap, 0) / shift) + 1
    else:
        assert L >= sample_size
        n_split = math.floor((L - sample_size) / shift) + 1

    full_L = (n_split - 1) * shift + sample_size
    audio = torch.nn.functional.pad(audio, (0, full_L - L)) if full_L > L else audio[:, :full_L]

    batch = []
    for n in range(n_split):
        b = audio[:, n * shift: n * shift + sample_size]
        batch.append(b)

    batch = torch.stack(batch, dim=0)  # (n_split, ch, sample_size)

    return batch, L


def cross_fade(preds, overlap: int, L: int):
    """
    preds: (bs, ch, sample_size)
    """
    bs, ch, sample_size = preds.shape
    shift = sample_size - overlap
    full_L = sample_size + (bs - 1) * shift

    if overlap > 0:
        win = torch.bartlett_window(overlap * 2, device=preds.device)

    buf = torch.zeros(ch, full_L, device=preds.device)
    for idx in range(bs):
        pred = preds[idx]  # (1, sample_size)
        ofs = idx * shift

        if overlap > 0:
            if idx != 0:
                pred[:, :overlap] *= win[None, :overlap]
            if idx != bs - 1:
                pred[:, -overlap:] *= win[None, overlap:]

        buf[:, ofs:ofs + sample_size] += pred

    buf = buf[..., :L]

    return buf


def make_silent_segment_to_zero(
    audio: torch.Tensor, chunk_size: int,
    th_amp: float = 0.01, th_rms: float = 1e-6,
    diable_amp_threshold: bool = False
):
    """
    Explicitly make silent segments in audio to zero
    for better representation of separation performance
    and removal of unexpected leakage sounds in test dataset (e.g. MUSDB18)

    Reference: Eetu Tunturi, "Score-informed separation of classical music",
    https://trepo.tuni.fi/bitstream/handle/10024/229112/TunturiEetu.pdf

    audio: (n_src, ch, L)
    """
    n_src, ch, L = audio.shape
    chunks, L = make_audio_batch(audio.view(-1, L), chunk_size, 0, return_last_short=True)
    chunks = rearrange(chunks, 'bs (s c) l -> bs s c l', c=ch, s=n_src)  # (n_chunk, n_src, ch, chunk_size)

    # remove DC component to compute amplitude/rms correctly
    chunks_nodc = chunks - chunks.mean(dim=3, keepdim=True)

    max_amps = chunks_nodc.abs().max(dim=3)[0].max(dim=2)[0]  # (n_chunk, n_src)
    rms = chunks_nodc.pow(2).mean(dim=[2, 3]).sqrt()  # (n_chunk, n_src)
    mask = (rms < th_rms) if diable_amp_threshold else ((max_amps < th_amp) | (rms < th_rms))   # (n_chunk, n_src)

    # make silent chunks to zero
    chunks[mask[..., None, None].expand_as(chunks)] = 0.0

    # chunks to audio
    chunks = rearrange(chunks, 'bs s c l -> bs (s c) l')  # (n_chunk, ch * n_src, chunk_size)
    audio = cross_fade(chunks, 0, L)  # (ch * n_src, L)
    audio = rearrange(audio, ' (s c) l -> s c l', s=n_src, c=ch)  # (n_src, ch, L)

    return audio, ~mask


def separate(
    model: nn.Module,
    audio: torch.Tensor,
    sample_size: int,
    overlap: int,
    batch_size: int,
    iterations: tp.List[int],
    accel,
    return_init: bool = False
):
    ch, L = audio.shape
    strides = make_audio_batch(audio, sample_size, overlap)[0]  # (bs, ch, sample_size)

    n_iter = math.ceil(strides.shape[0] / batch_size)
    pred_strides = []
    for it in range(n_iter):
        batch = strides[it * batch_size: (it + 1) * batch_size]  # (bs, ch, sample_size)
        with torch.no_grad(), accel.autocast():
            pred_batch = model.inference(batch, iters=iterations, return_init=return_init)  # List[(bs, n_src, ch, sample_size)]

        if isinstance(pred_batch, list):
            pred_batch = torch.stack(pred_batch, dim=0)  # (n_iterations, bs, n_src, ch, sample_size)
        else:
            pred_batch = pred_batch.unsqueeze(0)  # (1, bs, n_src, ch, sample_size)

        pred_strides.append(pred_batch)

    pred_strides = torch.cat(pred_strides, dim=1)  # (iters, bs, n_src, ch, sample_size)
    n_src = pred_strides.shape[2]
    pred_strides = rearrange(pred_strides, 'i b s c l -> (i s) b c l')  # (iters * n_src, bs, ch, sample_size)

    # Overlap-add
    audios_sep = []
    for i in range(pred_strides.shape[0]):
        pred = cross_fade(pred_strides[i], overlap, L)  # (ch, L)
        audios_sep.append(pred)

    audios_sep = torch.stack(audios_sep, dim=0)
    audios_sep = rearrange(audios_sep, '(i s) c l -> i s c l', s=n_src)  # (iters, n_src, ch, L)

    return audios_sep.contiguous()


class FlowList(list):
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


def get_stem_audio(root_dir: str, stems: tp.List[tp.List[str]], ext_audio: str, sample_rate: int):
    # Load stem audios
    # ['other'] is a special case where the stem audios which are not included in the other stem groups are summed to create a stem audio for 'other'

    # find all stems in the directory
    all_stems = [p.split("/")[-1].split(".")[0] for p in glob.glob(f"{root_dir}/*.{ext_audio}")]

    # exclude 'mixture' stem if exists
    if 'mixture' in all_stems:
        all_stems.remove('mixture')

    has_other = ['other'] in stems
    if has_other:
        # get files for 'other' stem
        other_stem = all_stems.copy()
        for stem_group in stems:
            if stem_group == ['other']:
                continue
            for stem in stem_group:
                if stem in other_stem:
                    other_stem.remove(stem)

        print_once(f"->-> The 'other' stem includes the following stems: {other_stem}")

    stem_audios = []
    for stem_group in stems:
        if stem_group == ['other']:
            stem_group = other_stem

        stem_group_audios = []
        for stem in stem_group:
            path = f"{root_dir}/{stem}.{ext_audio}"
            audio, sr = torchaudio.load(path)
            if sr != sample_rate:
                audio = julius.resample_frac(audio, sr, sample_rate)
            stem_group_audios.append(audio)

        # pad len
        max_len = max([a.shape[1] for a in stem_group_audios])
        stem_group_audios = [F.pad(a, (0, max_len - a.shape[1])) for a in stem_group_audios]

        stem_group_audio = torch.stack(stem_group_audios, dim=0).sum(dim=0)  # (ch, L)
        stem_audios.append(stem_group_audio)

    # pad len
    max_len = max([a.shape[1] for a in stem_audios])
    stem_audios = [F.pad(a, (0, max_len - a.shape[1])) for a in stem_audios]

    stem_audios = torch.stack(stem_audios, dim=0)  # (n_stem_groups, ch, L)

    return stem_audios


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt-dir', type=str, help="Checkpoint directory.")
    parser.add_argument('--iterations', type=int, nargs='+', default=[1, 2], help="Iteration list for inference.")
    parser.add_argument('--input-audio-dir', type=str, help="Input audio directory.")
    parser.add_argument('--stems', type=str, nargs='+', default=None, help="Stem patterns for evaluation (e.g. vocals drums/bass/other).")
    parser.add_argument('--stem-names', type=str, nargs='+', default=None, help="Names of the stems (e.g. vocals accompaniment).")
    parser.add_argument('--ext-audio', type=str, default='wav', help="Extension of audio files for evaluation.")
    parser.add_argument('--output-dir', type=str, help="Output directory.")
    parser.add_argument('--sample-rate', type=int, default=44100, help="Sample rate for evaluation.")
    parser.add_argument('--sample-size', type=int, default=220500, help="Sample size for inference.")
    parser.add_argument('--overlap-rate', type=float, default=0.5, help="Overlap rate for inference.")
    parser.add_argument('--chunk-size', type=int, default=44100, help="Chunk size for evaluation. If 0, no chunking is applied.")
    parser.add_argument('--max-batch-size', type=int, default=10, help="Max batch size for inference.")
    parser.add_argument('--use-original-name', default=True, type=bool, help="Whether to use an original file name as an output name.")
    parser.add_argument('--save-audio', action='store_true', help="Whether to save separated audios.")
    parser.add_argument('--save-target', action='store_true', help="Whether to save target audios.")
    parser.add_argument('--save-init', action='store_true', help="Whether to save the initial input audios.")
    parser.add_argument('--disable-amp-threshold', action='store_true', help="Disable amplitude threshold.")
    # museval evaluation
    parser.add_argument('--use-museval', action='store_true', help="Whether to use museval for evaluation.")

    args = parser.parse_args()

    ckpt_dir = args.ckpt_dir
    iterations = args.iterations
    input_audio_dir = args.input_audio_dir
    stems = args.stems
    stem_names = args.stem_names
    ext_audio = args.ext_audio
    output_dir = args.output_dir
    sample_rate = args.sample_rate
    sample_size = args.sample_size
    overlap_rate = args.overlap_rate
    chunk_size = args.chunk_size
    max_batch_size = args.max_batch_size
    use_original_name = args.use_original_name
    save_audio = args.save_audio
    save_target = args.save_target
    save_init = args.save_init
    disable_amp_threshold = args.disable_amp_threshold
    use_museval = args.use_museval
    overlap = int(sample_size * overlap_rate)

    assert len(stems) == len(stem_names), "The number of stem patterns and stem names must be the same."

    if use_museval:
        import museval
        # check museval version
        print_once(f"Museval version: {museval.version._version}")

    th_amp = 0.01
    th_rms = 1e-5

    cfg_ckpt = OmegaConf.load(f'{ckpt_dir}/config.yaml')
    amp = cfg_ckpt.trainer.amp if cfg_ckpt.trainer.amp else "no"

    # Distributed inference
    accel = Accelerator(mixed_precision=amp)
    device = accel.device
    rank = get_rank()

    print_once(f"Checkpoint dir  : {ckpt_dir}")
    print_once(f"Input audio dir : {input_audio_dir}")
    print_once(f"Output dir      : {output_dir}")

    # Load checkpoint
    model = hydra.utils.instantiate(cfg_ckpt.model)
    if os.path.exists(f"{ckpt_dir}/ema.pth"):
        print_once("->-> Loading EMA checkpoint.")
        ema = EMA(model, **cfg_ckpt.trainer.ema)
        ema.load_state_dict(torch.load(f"{ckpt_dir}/ema.pth", weights_only=False))
        model = ema.ema_model
    else:
        print_once("->-> No EMA checkpoint found. Directly loading the backbone model.")
        model.backbone_model.load_state_dict(torch.load(f"{ckpt_dir}/backbone_model.pth", weights_only=False))
    model.eval()
    print_once("->-> Successfully loaded a checkpoint.")

    model = accel.prepare(model)
    model = accel.unwrap_model(model)

    # Evaluation dataset

    stems = [stem.split("/") for stem in stems]  # List[List[str]]

    # Get valid audio directories
    audio_dirs = []
    for root, dirs, files in os.walk(input_audio_dir):
        # all required audio files exist in the directory
        if all(all(f"{stem}.{ext_audio}" in files for stem in stem_group) for stem_group in stems if stem_group != ['other']):
            audio_dirs.append(root)

    audio_dirs.sort()
    print_once(f"->-> Found evaluation data: {len(audio_dirs)}")

    os.makedirs(output_dir, exist_ok=True)

    # Evaluation metrics
    evals = {
        "bss_eval": BSSEval().to(device),
        "spec_eval": SpecEval().to(device),
        # "bleedfull_eval": BleedFull(sr=sample_rate, n_fft=4096, hop_length=1024, n_mels=512).to(device)
        "silent_eval": SilentSegmentEval().to(device),
    }

    # Metrics to be converted to dB after averaging / before printing
    db_convert_metrics = ["silence_ratio[silence]"]

    # Execution

    all_results = [{} for _ in iterations]
    if use_museval:
        results_museval = museval.EvalStore(frames_agg='median', tracks_agg='median')

    for idx, audio_dir in enumerate(audio_dirs):
        track_id = os.path.basename(audio_dir)
        print_once(f"{idx}: Processing {audio_dir}, Stems ({stem_names})-> {stems}")

        audio = get_stem_audio(audio_dir, stems, ext_audio, sample_rate)  # (n_src, ch, L)

        max_amp_ = audio.abs().max()
        if max_amp_ > 1.0:
            print_once(f"[Warning]: max amplitude ({max_amp_}) is greater than 1.0.")

        audio = audio.to(device)
        n_src, ch, L = audio.shape

        if not use_museval:
            # make silent segments to zero
            # https://trepo.tuni.fi/bitstream/handle/10024/229112/TunturiEetu.pdf
            # chunk size = 1 sec, no overlap
            audio, _ = make_silent_segment_to_zero(
                audio, chunk_size=sample_rate, th_amp=th_amp, th_rms=th_rms,
                diable_amp_threshold=disable_amp_threshold
            )

        if save_target:
            dir_name = track_id if use_original_name else f"sample_{idx}"
            target_output_dir = f"{args.output_dir}/target/{dir_name}"
            os.makedirs(target_output_dir, exist_ok=True)

            for i_src in range(n_src):
                out_path = f"{target_output_dir}/{stem_names[i_src]}.flac"
                torchaudio.save(out_path, audio[i_src].cpu(), sample_rate=sample_rate,
                                encoding="PCM_S", bits_per_sample=16)

        # Separation
        mixture = audio.sum(dim=0)  # (ch, L)
        audios_sep = separate(
            model=model,
            audio=mixture,  # mix
            sample_size=sample_size,
            overlap=overlap,
            batch_size=max_batch_size,
            iterations=iterations,
            return_init=save_init,
            accel=accel
        )  # (n_iterations, n_src, ch, L)

        if save_init and audios_sep.shape[0] > 1:
            audios_init = audios_sep[0]  # (n_src, ch, L)
            audios_sep = audios_sep[1:]

            dir_name = track_id if use_original_name else f"sample_{idx}"
            init_output_dir = f"{args.output_dir}/init/{dir_name}"
            os.makedirs(init_output_dir, exist_ok=True)

            for i_src in range(n_src):
                out_path = f"{init_output_dir}/{stem_names[i_src]}.flac"
                torchaudio.save(out_path, audios_init[i_src].cpu(), sample_rate=sample_rate,
                                encoding="PCM_S", bits_per_sample=16)

        if use_museval:
            # museval evaluation
            ref_np = audio.cpu().numpy().transpose(0, 2, 1)  # (n_src, L, ch)
            est_np = audios_sep[-1].cpu().numpy().transpose(0, 2, 1)  # (n_src, L, ch)
            print_once(f"Reference shape: {ref_np.shape}, Estimate shape: {est_np.shape}")

            win_sec = chunk_size / sample_rate
            SDR, ISR, SIR, SAR = museval.evaluate(
                ref_np, est_np, win=chunk_size, hop=chunk_size, mode='v4',
            )

            print_once(f"{len(SDR[0])} frames / {ref_np.shape[1] / sample_rate:.2f} sec")

            data = museval.TrackStore(track_id, win=win_sec, hop=win_sec, frames_agg="median")
            for i, name in enumerate(stem_names):
                values = {
                    "SDR": SDR[i].tolist(),
                    "SIR": SIR[i].tolist(),
                    "ISR": ISR[i].tolist(),
                    "SAR": SAR[i].tolist(),
                }

                data.add_target(target_name=name, values=values)

            results_museval.add_track(data)

            # log
            print_once(data)
            continue

        chunk_size_ = chunk_size if chunk_size > 0 else L
        chunks_tgt, _ = make_audio_batch(audio.view(-1, L), chunk_size_, 0, return_last_short=False)
        chunks_tgt = rearrange(chunks_tgt, 'bs (s c) l -> bs s c l', c=ch, s=n_src)  # (n_chunk, n_src, ch, chunk_size)

        # remove chunks including silence
        rms = chunks_tgt.pow(2).mean(dim=[2, 3]).sqrt()
        mask = rms > th_rms  # (n_chunk, n_src)

        num_valid_chunks = mask.sum(dim=0)  # (n_src,)
        print_once(f"Input shape: {audio.shape}, Valid chunks: {num_valid_chunks.tolist()}/{chunks_tgt.shape[0]}")

        # Evaluation
        for it_idx, it in enumerate(iterations):
            audios_sep_it = audios_sep[it_idx]  # (n_src, ch, L)

            # chunk split
            chunks, _ = make_audio_batch(audios_sep_it.view(-1, L), chunk_size_, 0, return_last_short=False)
            chunks = rearrange(chunks, 'bs (s c) l -> bs s c l', c=ch, s=n_src)  # (n_chunk, n_src, ch, chunk_size)

            eval_results_ = {}
            for eval in evals.values():
                eval_results_.update(
                    eval(
                        estimate=chunks,
                        reference=chunks_tgt,
                        mask=mask
                    )
                )  # dict of (bs, n_src)

            # median and mean
            eval_results = {f"{k} (median)": v.nanmedian(dim=0)[0] for k, v in eval_results_.items()}
            eval_results.update({f"{k} (mean)": v.nanmean(dim=0) for k, v in eval_results_.items()})  # (n_src)

            for k, v in eval_results.items():
                all_results[it_idx][k] = all_results[it_idx].get(k, []) + [v]

            # log
            if it_idx == len(iterations) - 1:  # log only for the last iteration
                print(f"Iter {it}: {eval_results}")

        # save audio
        if save_audio:
            dir_name = track_id if use_original_name else f"sample_{idx}"
            for it_idx, it in enumerate(iterations):
                output_dir = f"{args.output_dir}/iter_{it}/{dir_name}"
                os.makedirs(output_dir, exist_ok=True)

                for i_src in range(n_src):
                    out_path = f"{output_dir}/{stem_names[i_src]}.flac"
                    torchaudio.save(out_path, audios_sep[it_idx, i_src].cpu(), sample_rate=sample_rate,
                                    encoding="PCM_S", bits_per_sample=16)

    if use_museval:
        print("\n\n==== Final evaluation results using museval ====")
        print(results_museval)

        return

    # compute final results
    results_all_iters = {}
    for it_idx, it in enumerate(iterations):
        print_once(f"==== Final evaluation results (Iteration: {it}) ====")
        results_iter = {}
        for k, v in all_results[it_idx].items():
            v = torch.stack(v, dim=0)  # (n_samples, n_src)

            met_name = k.split(" ")[0]
            if "median" in k:
                results_iter[f"{met_name} (median-of-median)"] = v.nanmedian(dim=0)[0].tolist()
                results_iter[f"{met_name} (mean-of-median)"] = v.nanmean(dim=0).tolist()
            elif "mean" in k:
                results_iter[f"{met_name} (mean-of-mean)"] = v.nanmean(dim=0).tolist()

            for i_src in range(v.shape[1]):
                if f"iter_{it}" not in results_all_iters:
                    results_all_iters[f"iter_{it}"] = {}
                if k not in results_all_iters[f"iter_{it}"]:
                    results_all_iters[f"iter_{it}"][k] = {}
                results_all_iters[f"iter_{it}"][k][f"src_{i_src}"] = FlowList(v[:, i_src].cpu().tolist())

        # convert to dB
        for k in results_iter.keys():
            for db_key in db_convert_metrics:
                if db_key in k:
                    results_iter[k] = (20. * np.log10(results_iter[k])).tolist()

        # print results
        print_once("[Median-of-Median]")
        for k, v in results_iter.items():
            if "median-of-median" in k:
                print_once(f"\t{k}: {v}")
        print_once("[Mean-of-Median]")
        for k, v in results_iter.items():
            if "mean-of-median" in k:
                print_once(f"\t{k}: {v}")
        print_once("[Mean-of-Mean]")
        for k, v in results_iter.items():
            if "mean-of-mean" in k:
                print_once(f"\t{k}: {v}")

    # save as yaml
    with open(f"{args.output_dir}/results_all_samples.yaml", "w") as f:
        yaml.dump(results_all_iters, f)

    print(f"[rank-{rank}] ->->-> Finished evaluation.")

    return


if __name__ == "__main__":
    main()
