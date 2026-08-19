"""
Copyright (C) 2025 Yukara Ikemiya

-----------------------------------------------------
Time sampler for Flow-Matching training (Sec.3.2)
"""

import torch


class ZeroWeightTimestepSampler:
    def __init__(self, zero_weight: float = 0.5):
        """
        zero_weight: Proportion of samples for 't = 0' (default: 0.5)
        NOTE: If zero_weight is 0.5, half of the samples will have t = 0 as in the paper.
        """
        assert 0 <= zero_weight <= 1.0
        self.zero_weight = zero_weight

    def sample(self, N: int, device):
        """
        Sample timesteps for training.
        Returns:
            t (N,): Tensor of timesteps within [0, 1].
        """
        t = torch.rand(N, device=device)
        idxs_zero = torch.rand(N, device=device) <= self.zero_weight
        t[idxs_zero] = 0.0

        return t


class SNRWeightTimestepSampler:
    def __init__(self, snr_range: list = [-80, 100], zero_weight: float = 0.01):
        """
        snr_range: Range of SNR values for sampling (default: [-80, 100])
        zero_weight: Proportion of samples for 't = 0' (default: 0.01)
        """
        self.a, self.b = snr_range
        self.zero_weight = zero_weight

    def sample(self, N: int, device):
        """
        Sample timesteps for training.
        Returns:
            t (N,): Tensor of timesteps within [0, 1].
        """
        r = (self.b - self.a) * torch.rand(N, device=device) + self.a
        t = 1. / (1. + 10 ** (-r / 20.0))
        idxs_zero = torch.rand(N, device=device) <= self.zero_weight
        t[idxs_zero] = 0.0

        return t
