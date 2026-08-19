"""
Copyright (C) 2025 Yukara Ikemiya

-----------------------------------------------------
Noise shaping module for initial noise of flow-matching.
"""

import torch
from torch import nn
import torch.nn.functional as F


class EnergyEnvelopeNoiseShaping(nn.Module):
    """
    Noise shaping from Sec.3.4
    """

    def __init__(
        self,
        window_size: int = 1025,
        use_average: bool = True,
        eps: float = 1e-7,
        # output scale factor
        scale: float = 1.0
    ):
        super().__init__()
        self.use_average = use_average
        self.eps = eps
        self.scale = scale

        window_size = window_size if window_size % 2 == 1 else window_size + 1
        hamm_win = torch.hamming_window(window_size)
        hamm_win /= hamm_win.sum()

        self.register_buffer("hamm_win", hamm_win, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        x: reference audio, (bs, sample_length)
        Returns: noise shape, (bs, sample_length) or (bs, 1)
        """
        assert x.dim() == 2

        # get energy envelope
        envelope = self.get_envelope(x)

        if self.use_average:
            # average the envelope
            weight = self.__average(envelope)  # (bs)
            weight = weight.unsqueeze(-1)
        else:
            weight = envelope.sqrt()

        return weight * self.scale

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

    def __average(self, envelope: torch.Tensor):
        """
        x: (bs, sample_length)
        Returns: (bs)
        """
        mask = envelope > self.eps
        masked_sum = (envelope * mask).sum(dim=1)
        masked_count = mask.sum(dim=1)

        masked_avg = torch.where(
            masked_count > 0,
            masked_sum / masked_count,
            torch.zeros_like(masked_sum)
        )

        return masked_avg  # (bs)
