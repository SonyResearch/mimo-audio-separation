"""
Copyright (C) 2024 Yukara Ikemiya
"""

import math

from torch.nn import functional as F
import torchaudio
from torchaudio import transforms as T
import soundfile as sf


def get_audio_metadata(filepath):
    try:
        info_ = sf.info(filepath)
        sample_rate = info_.samplerate
        num_channels = info_.channels
        num_frames = info_.frames

        info = {
            'sample_rate': sample_rate,
            'num_frames': num_frames,
            'num_channels': num_channels
        }
    except Exception as e:
        # error : cannot open an audio file
        print(f"Failed to load metadata for {filepath}: {e}")
        info = None

    return info


def load_audio_with_pad(filepath, info: dict, sr: int, n_samples: int, offset: int, return_info: bool = False):
    sr_in, num_frames = info['sample_rate'], info['num_frames']
    n_samples_in = int(math.ceil(n_samples * (sr_in / sr))) if n_samples is not None else num_frames

    # load audio
    ext = filepath.split(".")[-1]
    out_frames = min(n_samples_in, num_frames - offset)

    audio, _ = torchaudio.load(
        filepath, frame_offset=offset, num_frames=out_frames,
        format=ext, backend='soundfile')

    # resample
    if sr_in != sr:
        resample_tf = T.Resample(sr_in, sr)
        audio = resample_tf(audio)
        if n_samples is not None:
            audio = audio[..., :n_samples]

    # zero pad
    if n_samples is not None:
        L = audio.shape[-1]
        if L < n_samples:
            audio = F.pad(audio, (0, n_samples - L), value=0.)

    if return_info:
        info = {"num_frames": audio.shape[-1]}
        return audio, info

    return audio
