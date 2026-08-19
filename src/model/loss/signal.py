"""
Copyright (C) 2025 Yukara Ikemiya

-------------
Signal-domain loss
"""

import torch
from torch import nn
from torch.nn import functional as F


class WaveLoss(nn.Module):
    def __init__(
        self,
        loss_type: str = "L2",
        decibel: bool = True,
        eps: float = 1e-4
    ):
        super().__init__()
        assert loss_type in ["L1", "L2"], "loss_type must be 'L1' or 'L2'"
        self.loss_type = loss_type
        self.decibel = decibel
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, n_src, C, L)
        """
        assert x.shape == y.shape

        # x = x.view(-1, x.shape[-1])
        # y = y.view(-1, y.shape[-1])

        if self.loss_type == "L1":
            loss = self.__l1(x, y)
        else:  # L2
            loss = self.__l2(x, y)

        name = f"{'l1' if self.loss_type == 'L1' else 'l2'}_wave{'_db' if self.decibel else ''}"
        output = {name: loss}

        return output

    def __l1(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x: prediction, (B, n_src, C, L)
        y: target
        """
        if not self.decibel:
            loss = F.l1_loss(x, y, reduction='mean')
        else:
            # source-wise computation
            loss = (x - y).abs().mean(dim=[2, 3])
            loss = loss / (y.abs().mean(dim=[2, 3]) + self.eps)
            loss = (20 * loss.log10()).mean()

        return loss

    def __l2(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if not self.decibel:
            loss = F.mse_loss(x, y, reduction='mean')
        else:
            loss = (x - y).pow(2).mean(dim=[2, 3]).sqrt()
            loss = loss / (y.pow(2).mean(dim=[2, 3]).sqrt() + self.eps)
            loss = (20 * loss.log10()).mean()

        return loss


class SISNRLoss(nn.Module):
    """
    (Negative) Scale-Invariant Signal-to-Noise Ratio (SI-SNR) loss proposed in Conv-TasNet
    [https://arxiv.org/abs/1809.07454]
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, n_src, C, L)
        """
        assert x.shape == y.shape

        # source-wise computation
        scale_y = (x * y).sum(dim=[2, 3]) / ((y ** 2).sum(dim=[2, 3]) + self.eps)
        y = y * scale_y[:, :, None, None]

        e_noise = x - y
        loss = (e_noise ** 2).sum(dim=[2, 3]) / ((y ** 2).sum(dim=[2, 3]) + self.eps)
        loss = (10 * loss.log10()).mean()

        return {'n-sisnr': loss}


class SNRLoss(nn.Module):
    """
    (Negative) Signal-to-Noise Ratio (SNR) loss in Universal Sound Separation
    [https://arxiv.org/abs/1905.03330]
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: (B, n_src, C, L)
        """
        assert x.shape == y.shape

        e_noise = x - y
        loss = (e_noise ** 2).sum(dim=[2, 3]) / ((y ** 2).sum(dim=[2, 3]) + self.eps)
        loss = (10 * loss.log10()).mean()

        return {'n-snr': loss}
