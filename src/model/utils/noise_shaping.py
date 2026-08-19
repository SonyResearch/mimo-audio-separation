"""
Copyright (C) 2025 Yukara Ikemiya
"""

import torch
from torch import nn
import torch.nn.functional as F


class EnergyEnvelopeNoiseShaping(nn.Module):
    """
    Noise shaping from FLOSS [2025, Google]
    https://arxiv.org/abs/2505.16119
    """

    def __init__(
        self,
        window_size: int = 1025,  # for 44.1khz
        scale: float = 1.0
    ):
        super().__init__()
        self.scale = scale

        window_size = window_size if window_size % 2 == 1 else window_size + 1
        hamm_win = torch.hamming_window(window_size)
        hamm_win /= hamm_win.sum()
        self.register_buffer("hamm_win", hamm_win, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: reference audio, (bs, sample_length)
        Returns: noise shape, (bs, sample_length)
        """
        assert x.dim() == 2

        # get energy envelope
        envelope = self.get_envelope(x)
        weight = envelope.sqrt() * self.scale

        return weight

    def get_envelope(self, x: torch.Tensor):
        """
        Get the energy envelope of the input signal.
        x: (bs, sample_length)
        Returns: (bs, sample_length)
        """
        x_e = x ** 2.
        x_e = x_e.unsqueeze(1)  # (bs, 1, sample_length)

        # Hamming window filter
        hamm_win = self.hamm_win.flip(0).view(1, 1, -1)  # (1, 1, window_size)
        envelope = F.conv1d(x_e, hamm_win, padding=hamm_win.size(-1) // 2)

        return envelope.squeeze(1)  # (bs, sample_length)
