"""
Copyright (C) 2024 Yukara Ikemiya
"""

import os
import sys
import argparse
import math
import yaml
import typing as tp

import torch
from torch import nn
import torchaudio
import numpy as np
import julius
from einops import rearrange

from utils.torch_common import print_once
from eval import BSSEval, SpecEval, BleedFull, SilentSegmentEval
from evaluate import make_audio_batch, cross_fade, make_silent_segment_to_zero

sys.dont_write_bytecode = True


class FlowList(list):
    pass


def flow_list_representer(dumper, data):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)


yaml.add_representer(FlowList, flow_list_representer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target-audio-dir', type=str, help="Target (reference) audio directory.")
    parser.add_argument('--estimate-audio-dir', type=str, help="Estimate audio directory.")
    parser.add_argument('--stems', nargs='+', type=str, default=['vocals', 'accompaniment'],
                        help="List of target stems to be separated.")
    parser.add_argument("--audio-ext", type=str, default='wav', help="Audio extension to process in input_folder")
    parser.add_argument('--chunk-size', type=int, default=44100, help="Chunk size for evaluation.")
    parser.add_argument('--sample-rate', type=int, default=44100, help="Sampling rate of audio files.")
    parser.add_argument('--disable-amp-threshold', action='store_true', help="Disable amplitude threshold.")
    args = parser.parse_args()

    target_dir = args.target_audio_dir
    estimate_dir = args.estimate_audio_dir
    stems = args.stems
    ext = args.audio_ext
    chunk_size = args.chunk_size
    sr = args.sample_rate
    disable_amp_threshold = args.disable_amp_threshold
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    th_amp = 0.01
    th_rms = 1e-5

    print_once(f"== Estimate audio directory: {estimate_dir} ==")
    print_once(f"== Target audio directory: {target_dir} ==")

    # Evaluation modules
    evals = {
        "bss_eval": BSSEval().to(device),
        "spec_eval": SpecEval().to(device),
        # "bleedfull_eval": BleedFull(sr=sr, n_fft=4096, hop_length=1024, n_mels=512).to(device),
        "silent_eval": SilentSegmentEval().to(device),
    }

    # Metrics to be converted to dB after averaging / before printing
    db_convert_metrics = ["silence_ratio[silence]"]

    # find all valid sub-directories under target_dir (recursively)
    # a valid dir must contain all target stem files AND have a matching estimate dir with all stems
    valid_rel_dirs = []
    for root, dirs, files in os.walk(target_dir):
        if not all(f"{stem}.{ext}" in files for stem in stems):
            continue
        rel = os.path.relpath(root, target_dir)
        est_dir = os.path.join(estimate_dir, rel)
        if not os.path.isdir(est_dir):
            continue
        est_files = set(os.listdir(est_dir))
        if not all(f"{stem}.{ext}" in est_files for stem in stems):
            continue
        valid_rel_dirs.append(rel)

    valid_rel_dirs.sort()
    print(f"Found {len(valid_rel_dirs)} valid sub-directories under {target_dir}.")

    # Execution

    all_results = {}
    for rel_dir in valid_rel_dirs:
        print(f"--- Processing {rel_dir}")

        # load audios
        audios_t = []
        audios_e = []
        for stem in stems:
            file_path_t = os.path.join(target_dir, rel_dir, f"{stem}.{ext}")
            file_path_e = os.path.join(estimate_dir, rel_dir, f"{stem}.{ext}")
            audio_t, sr_t = torchaudio.load(file_path_t)  # (ch, L)
            audio_e, sr_e = torchaudio.load(file_path_e)  # (ch, L)

            # resample to the specified sample rate if needed
            if sr_t != sr:
                audio_t = julius.resample_frac(audio_t, sr_t, sr)
            if sr_e != sr:
                audio_e = julius.resample_frac(audio_e, sr_e, sr)

            audios_t.append(audio_t)
            audios_e.append(audio_e)

        # align lengths (in case of small mismatch after resampling)
        min_L = min(min(a.shape[-1] for a in audios_t), min(a.shape[-1] for a in audios_e))
        audios_t = [a[..., :min_L] for a in audios_t]
        audios_e = [a[..., :min_L] for a in audios_e]

        audios_t = torch.stack(audios_t, dim=0).to(device)
        audios_e = torch.stack(audios_e, dim=0).to(device)  # (n_src, ch, L)
        assert audios_t.shape == audios_e.shape, f"Shape mismatch between target and estimate in {rel_dir}"

        n_src, ch, L = audios_t.shape

        # make silent segments to zero
        # https://trepo.tuni.fi/bitstream/handle/10024/229112/TunturiEetu.pdf
        #
        # NOTE: Please be sure that the same preprocessing is also applied at separation time.
        audios_t, _ = make_silent_segment_to_zero(
            audios_t, chunk_size=sr, th_amp=th_amp, th_rms=th_rms,
            diable_amp_threshold=disable_amp_threshold
        )

        chunk_size_ = chunk_size if chunk_size > 0 else L
        chunks_tgt, _ = make_audio_batch(audios_t.view(-1, L), chunk_size_, 0, return_last_short=False)
        chunks_tgt = rearrange(chunks_tgt, 'bs (s c) l -> bs s c l', c=ch, s=n_src)  # (n_chunk, n_src, ch, chunk_size)

        # remove chunks including silence
        rms = chunks_tgt.pow(2).mean(dim=[2, 3]).sqrt()
        mask = rms > th_rms  # (n_chunk, n_src)

        num_valid_chunks = mask.sum(dim=0)  # (n_src,)
        print_once(f"Input shape: {audios_t.shape}, Valid chunks: {num_valid_chunks.tolist()}/{chunks_tgt.shape[0]}")

        # Evaluation

        # chunk split
        chunks_est, _ = make_audio_batch(audios_e.view(-1, L), chunk_size_, 0, return_last_short=False)
        chunks_est = rearrange(chunks_est, 'bs (s c) l -> bs s c l', c=ch, s=n_src)  # (n_chunk, n_src, ch, chunk_size)

        eval_results_ = {}
        for eval in evals.values():
            eval_results_.update(
                eval(
                    estimate=chunks_est,
                    reference=chunks_tgt,
                    mask=mask
                )
            )  # dict of (bs, n_src)

        # median and mean
        eval_results = {f"{k} (median)": v.nanmedian(dim=0)[0] for k, v in eval_results_.items()}
        eval_results.update({f"{k} (mean)": v.nanmean(dim=0) for k, v in eval_results_.items()})  # (n_src)

        for k, v in eval_results.items():
            all_results[k] = all_results.get(k, []) + [v]

    # compute final results
    results_all_samples = {}
    results_final = {}
    print_once("==== Final evaluation results ====")
    for k, v in all_results.items():
        v = torch.stack(v, dim=0)  # (n_samples, n_src)

        met_name = k.split(" ")[0]
        if "median" in k:
            results_final[f"{met_name} (median-of-median)"] = v.nanmedian(dim=0)[0].tolist()
            results_final[f"{met_name} (mean-of-median)"] = v.nanmean(dim=0).tolist()
        elif "mean" in k:
            results_final[f"{met_name} (mean-of-mean)"] = v.nanmean(dim=0).tolist()

        # for i_src in range(v.shape[1]):
        #     if k not in results_all_samples:
        #         results_all_samples[k] = {}

        #     results_all_samples[k][f"({tag}) src_{i_src}"] = v_agg[i_src].item()
        #     results_all_samples[k][f"src_{i_src}"] = FlowList(v[:, i_src].cpu().tolist())

        # # sort results_all_samples[k]
        # results_all_samples[k] = dict(sorted(results_all_samples[k].items()))

    # save as yaml
    # with open(f"{estimate_dir}/sep_evaluation.yaml", "w") as f:
    #     yaml.dump(results_all_samples, f)

    # convert to dB
    for k in results_final.keys():
        for db_key in db_convert_metrics:
            if db_key in k:
                results_final[k] = (20. * np.log10(results_final[k])).tolist()

    # print results
    print_once("[Median-of-Median]")
    for k, v in results_final.items():
        if "median-of-median" in k:
            print_once(f"\t{k}: {v}")
    print_once("[Mean-of-Median]")
    for k, v in results_final.items():
        if "mean-of-median" in k:
            print_once(f"\t{k}: {v}")
    print_once("[Mean-of-Mean]")
    for k, v in results_final.items():
        if "mean-of-mean" in k:
            print_once(f"\t{k}: {v}")

    print("\n[ Finished evaluation. ]")

    return


if __name__ == "__main__":
    main()
