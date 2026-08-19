"""
Copyright (C) 2024 Yukara Ikemiya
"""

import random

import torch
from torch import nn
import numpy as np


# Channels

class Mono(nn.Module):
    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return torch.mean(x, dim=0, keepdims=True) if len(x.shape) > 1 else x


class Stereo(nn.Module):
    def __call__(self, x: torch.Tensor):
        x_shape = x.shape
        assert len(x_shape) <= 2
        # Check if it's mono
        if len(x_shape) == 1:  # s -> 2, s
            x = x.unsqueeze(0).repeat(2, 1)
        elif len(x_shape) == 2:
            if x_shape[0] == 1:  # 1, s -> 2, s
                x = x.repeat(2, 1)
            elif x_shape[0] > 2:  # ?, s -> 2,s
                x = x[:2, :]

        return x


# Augmentation

class PhaseFlipper(nn.Module):
    """Randomly invert the phase of a signal"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) <= 2
        return -x if (random.random() < self.p) else x


class VolumeChanger(nn.Module):
    """Randomly change volume (amplitude) of a signal"""

    def __init__(self, min_db: float = -3., max_db: float = 3., max_amplitude: float = 1.0):
        super().__init__()
        self.min_db = min_db
        self.max_db = max_db
        self.max_amplitude = max_amplitude
        self.rng = np.random.default_rng()

    def __call__(self, x: torch.Tensor):
        max_db = min(self.max_db, 20 * np.log10(self.max_amplitude / (x.abs().max() + 1.0e-8)))  # amp <= 1.0
        min_db = min(self.min_db, max_db)
        db = self.rng.uniform(min_db, max_db)
        gain = 10 ** (db / 20.)

        return x * gain


class ChannelShuffler(nn.Module):
    """Randomly shuffle channels of a multi-channel signal"""

    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor):
        assert len(x.shape) == 2  # (ch, sample_size)
        ch = x.shape[0]
        if random.random() < self.p and ch > 1:
            ch_indices = list(range(ch))
            random.shuffle(ch_indices)
            x = x[ch_indices, :]

        return x


class Silence(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor):
        return torch.zeros_like(x) if (random.random() < self.p) else x


class SilenceMultiSrc(nn.Module):
    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def __call__(self, x: torch.Tensor):
        # (num_src, ch, sample_size)
        n_src = x.shape[0]
        if n_src == 1:
            return x

        # randomly choose one source not to be silent
        idx_ns = random.randint(0, n_src - 1)
        for i in range(n_src):
            if i != idx_ns and (random.random() < self.p):
                x[i] = torch.zeros_like(x[i])

        return x
