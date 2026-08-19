"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under MIT License.
https://github.com/starrytong/SCNet

-----------------------------------------------------
SCNet backbone architecture.
https://arxiv.org/pdf/2401.13276.pdf
"""

import math

import torch
import torch.nn as nn
from torch.nn.modules.rnn import LSTM

from utils.torch_common import exists


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn(
            [out_features // 2, in_features]) * std)

    def forward(self, input):
        """
        input: (..., in_features)
        Returns: (..., out_features)
        """
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class FeatureConversion(nn.Module):
    """
    Integrates into the adjacent Dual-Path layer.

    Args:
        channels (int): Number of input channels.
        inverse (bool): If True, uses ifft; otherwise, uses rfft.
    """

    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse = inverse
        self.channels = channels

    def forward(self, x):
        # B, C, F, T = x.shape
        if self.inverse:
            x = x.float()
            x_r = x[:, :self.channels // 2, :, :]
            x_i = x[:, self.channels // 2:, :, :]
            x = torch.complex(x_r, x_i)
            x = torch.fft.irfft(x, dim=3, norm="ortho")
        else:
            x = x.float()
            x = torch.fft.rfft(x, dim=3, norm="ortho")
            x_real = x.real
            x_imag = x.imag
            x = torch.cat([x_real, x_imag], dim=1)

        return x


class DualPathRNN(nn.Module):
    """
    Dual-Path RNN in Separation Network.

    Args:
        d_model (int): The number of expected features in the input (input_size).
        expand (int): Expansion factor used to calculate the hidden_size of LSTM.
        bidirectional (bool): If True, becomes a bidirectional LSTM.
    """

    def __init__(self, d_model, expand, bidirectional=True):
        super(DualPathRNN, self).__init__()

        self.d_model = d_model
        self.hidden_size = d_model * expand
        self.bidirectional = bidirectional
        # Initialize LSTM layers and normalization layers
        self.lstm_layers = nn.ModuleList([self._init_lstm_layer(self.d_model, self.hidden_size) for _ in range(2)])
        self.linear_layers = nn.ModuleList([nn.Linear(self.hidden_size * 2, self.d_model) for _ in range(2)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])

    def _init_lstm_layer(self, d_model, hidden_size):
        return LSTM(d_model, hidden_size, num_layers=1, bidirectional=self.bidirectional, batch_first=True)

    def forward(self, x):
        B, C, F, T = x.shape

        # Process dual-path rnn
        original_x = x
        # Frequency-path
        x = self.norm_layers[0](x)
        x = x.transpose(1, 3).contiguous().view(B * T, F, C)
        x, _ = self.lstm_layers[0](x)
        x = self.linear_layers[0](x)
        x = x.view(B, T, F, C).transpose(1, 3)
        x = x + original_x

        original_x = x
        # Time-path
        x = self.norm_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B * F, C, T).transpose(1, 2)
        x, _ = self.lstm_layers[1](x)
        x = self.linear_layers[1](x)
        x = x.transpose(1, 2).contiguous().view(B, F, C, T).transpose(1, 2)
        x = x + original_x

        return x


class SeparationNet(nn.Module):
    """
    Implements a simplified Sparse Down-sample block in an encoder architecture.

    Args:
    - channels (int): Number input channels.
    - expand (int): Expansion factor used to calculate the hidden_size of LSTM.
    - num_layers (int): Number of dual-path layers.
    """

    def __init__(
        self,
        channels,
        expand=1,
        num_layers=6,
        # time conditioning
        use_time_embed: bool = False,
        dim_time_emb: int = 256,
    ):
        super().__init__()
        self.use_time_embed = use_time_embed
        if self.use_time_embed:
            self.time_embedding = nn.Sequential(
                FourierFeatures(1, dim_time_emb),
                nn.Linear(dim_time_emb, channels, bias=True),
                nn.SiLU(),
                nn.Linear(channels, channels, bias=True)
            )

        self.num_layers = num_layers

        self.dp_modules = nn.ModuleList([
            DualPathRNN(channels * (2 if i % 2 == 1 else 1), expand) for i in range(num_layers)
        ])

        self.feature_conversion = nn.ModuleList([
            FeatureConversion(channels * 2, inverse=False if i % 2 == 0 else True) for i in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        # time conditioning
        if self.use_time_embed:
            assert exists(t) and t.dim() == 1 and t.shape[0] == x.shape[0]
            t_emb = self.time_embedding(t[:, None])  # (B, dim_time_emb)
            x = x + t_emb[:, :, None, None]

        for i in range(self.num_layers):
            x = self.dp_modules[i](x)
            x = self.feature_conversion[i](x)
        return x
