"""
Copyright (C) 2025 Yukara Ikemiya

----------------
Base class for Single-Input-Single-Output (SISO) models
"""
import typing as tp

import torch
from torch import nn


def centering(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x - x.mean(dim=dim, keepdim=True)


class SISOBase(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_sources: int,
        backbone_model: nn.Module,
        discriminator=None,  # dummy
        # losses
        loss_modules: tp.Dict[str, nn.Module] = {},
        loss_lambdas: tp.Dict[str, float] = {},
        # training/inference config
        input_normalize: bool = True,
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_sources = num_sources
        # modules
        self.backbone_model = backbone_model
        self.discriminator = discriminator  # dummy
        # losses
        self.loss_modules = loss_modules
        self.loss_lambdas = loss_lambdas
        # training config
        self.input_normalize = input_normalize

    def forward(
        self,
        x: torch.Tensor,
        **kwargs
    ) -> tp.List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input multi-source audio, (B, C, L)
        Returns:
            sep (torch.Tensor): (B, C, T)
        """
        # prediction
        pred = self.backbone_model(x, **kwargs)

        return pred

    def train_step(
        self,
        sources: torch.Tensor
    ):
        """
        Args:
            sources (torch.Tensor): sources, (B, n_src, C, L)
        """
        self.train()
        output = {}

        bs, n_src, ch, L = sources.shape
        assert n_src == self.num_sources and ch == self.num_channels, \
            f"Input sources shape mismatch: expected n_src={self.num_sources}, ch={self.num_channels}, got n_src={n_src}, ch={ch}"

        # scale normalization
        if self.input_normalize:
            scales = self.__get_normalize_scale(sources.sum(1))  # (B,)
            sources = sources * scales.view(bs, 1, 1, 1)

        # target
        # NOTE: target is always the first source for SISO
        target = sources[:, 0, :, :]  # (B, C, L)

        # mixture (summed)
        mixture = sources.sum(dim=1)  # (B, C, L)

        # prediction
        pred = self(mixture)

        # add stem dimension for loss computation
        pred = pred.unsqueeze(1)  # (B, 1, C, L)
        target = target.unsqueeze(1)  # (B, 1, C, L)

        # losses
        losses = {}
        for name, module in self.loss_modules.items():
            losses.update(module(pred, target))

        # weighted sum of losses
        loss = 0.
        for k in self.loss_lambdas.keys():
            loss += losses[k] * self.loss_lambdas[k]

        output['G/loss'] = loss
        output.update({f"G/{k}": v.detach() for k, v in losses.items()})

        return output

    @torch.no_grad()
    def test(
        self,
        sources: torch.Tensor,
        **kwargs
    ):
        """
        Args:
            sources (torch.Tensor): sources, (B, n_src, C, L)
        """
        self.eval()

        bs, n_src, ch, L = sources.shape
        assert n_src == self.num_sources and ch == self.num_channels, \
            f"Input sources shape mismatch: expected n_src={self.num_sources}, ch={self.num_channels}, got n_src={n_src}, ch={ch}"

        # scale normalization
        if self.input_normalize:
            scales = self.__get_normalize_scale(sources.sum(1))  # (B,)
            sources = sources * scales.view(bs, 1, 1, 1)

        target = sources[:, 0, :, :]  # (B, C, L)
        mixture = sources.sum(dim=1)  # (B, C, L)

        # prediction
        pred = self(mixture)

        # add stem dimension for loss computation
        pred = pred.unsqueeze(1)  # (B, 1, C, L)
        target = target.unsqueeze(1)  # (B, 1, C, L)

        # loss evaluation
        losses = {}
        for name, module in self.loss_modules.items():
            losses.update(module(pred, target))

        # loss weight
        for k in self.loss_lambdas.keys():
            losses[k] = losses[k] * self.loss_lambdas[k]

        # rescale to original scale
        pred = pred.squeeze(1)  # (B, C, L)
        if self.input_normalize:
            pred /= scales.view(bs, 1, 1)
            mixture = mixture / scales.view(bs, 1, 1)

        # get "residual" stem
        residual = mixture - pred
        pred = torch.stack([pred, residual], dim=1)  # (B, 2, C, L)

        info = {'mixture': mixture, 'losses': losses}

        return pred, info

    @torch.no_grad()
    def inference(
        self,
        mixture: torch.Tensor,
        **kwargs
    ):
        """
        Args:
            mixture (torch.Tensor): mixture, (B, C, L)
        Returns:
            pred (torch.Tensor): separated sources (target and residual), (B, 2, C, L)
        """
        self.eval()

        bs, ch, L = mixture.shape

        # scale normalization
        if self.input_normalize:
            scales = self.__get_normalize_scale(mixture)  # (B,)
            mixture = mixture * scales.view(bs, 1, 1)

        # prediction
        pred = self(mixture)
        residual = mixture - pred
        pred = torch.stack([pred, residual], dim=1)  # (B, 2, C, L)

        # rescale to original scale
        if self.input_normalize:
            pred = pred / scales.view(bs, 1, 1, 1)

        return pred

    def __get_normalize_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio to have max amplitude of 1.
        x: (B, C, L)
        """
        max_amps = x.abs().amax(dim=(-1, -2))  # (B,)

        # silence: no normalization, voiced: 1 / (max_amp + 1e-8)
        max_amps = torch.where(max_amps < 1e-8, torch.full((len(max_amps),), 1.0, device=x.device),
                               max_amps + 1e-8)

        scales = 1 / max_amps  # (B,)
        return scales
