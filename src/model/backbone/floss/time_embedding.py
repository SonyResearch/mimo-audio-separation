"""
Copyright (C) 2025 Yukara Ikemiya
"""

import torch
from torch import nn


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.):
        super().__init__()
        assert out_features % 2 == 0
        self.in_features = in_features
        self.pi = 3.141592653589793
        self.weight = nn.Parameter(torch.randn([in_features, out_features // 2]) * std)

    def forward(self, x: torch.Tensor):
        """
        x: (..., in_features), elements assumes to be between 0 -- 1
        return : (..., out_features)
        """
        assert x.shape[-1] == self.in_features
        f = 2 * self.pi * x @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)
