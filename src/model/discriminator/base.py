"""
Copyright (C) 2025 Yukara Ikemiya

-----------------------------------------------------
Base classes for discriminators.
"""
import typing as tp
from abc import ABC, abstractmethod

import torch
from torch import nn
import torch.nn.functional as F

from utils.torch_common import checkpoint


class MultiDiscriminator(ABC, nn.Module):
    @abstractmethod
    def forward(self, x: torch.Tensor, return_feature: bool = True):
        pass

    def compute_G_loss(self, x_fake, x_real, gradient_checkpointing: bool = False):
        assert x_fake.shape == x_real.shape

        logits_f, feats_f = self(x_fake, return_feature=True, gradient_checkpointing=gradient_checkpointing)
        with torch.no_grad():
            logits_r, feats_r = self(x_real, return_feature=True)

        num_D = len(self.model)
        losses = {
            'disc_gan_loss': 0.,
            'disc_feat_loss': 0.
        }

        for i_d in range(num_D):
            n_layer = len(feats_f[i_d])

            # GAN loss
            losses['disc_gan_loss'] += (1 - logits_f[i_d]).relu().mean()

            # Feature-matching loss
            # eq.(8)
            feat_loss = 0.
            for i_l in range(n_layer):
                feat_loss += F.l1_loss(feats_f[i_d][i_l], feats_r[i_d][i_l])

            losses['disc_feat_loss'] += feat_loss / (n_layer)

        losses['disc_gan_loss'] /= num_D
        losses['disc_feat_loss'] /= num_D

        return losses

    def compute_D_loss(self, x, mode: str, gradient_checkpointing: bool = False):
        assert mode in ['fake', 'real']
        sign = 1 if mode == 'fake' else -1

        logits = self(x, return_feature=False, gradient_checkpointing=gradient_checkpointing)

        num_D = len(self.model)
        losses = {'loss': 0.}

        for i_d in range(num_D):
            # Hinge loss
            losses['loss'] += (1 + sign * logits[i_d]).relu().mean()

        losses['loss'] /= num_D

        return losses


class CombDiscriminator(nn.Module):
    """
    Combination of any discriminators.
    """

    def __init__(
        self,
        D: tp.Dict[str, MultiDiscriminator],
        loss_lambdas: tp.Optional[tp.Dict[str, float]] = None,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        self.D = nn.ModuleDict(D)
        self.loss_lambdas = loss_lambdas
        self.gradient_checkpointing = gradient_checkpointing
        if self.loss_lambdas is None:
            self.loss_lambdas = {k: 1.0 for k in D.keys()}
        else:
            for k in loss_lambdas.keys():
                assert k in D.keys(), f"Loss lambda key '{k}' not found in discriminators."

    def compute_G_loss(self, x_fake, x_real):
        losses = {}
        # NOTE: All discs assume to return the same keys (losses)
        for name, disc in self.D.items():
            # if self.gradient_checkpointing and self.training:
            #     disc_losses = checkpoint(disc.compute_G_loss, x_fake, x_real)
            # else:
            #     disc_losses = disc.compute_G_loss(x_fake, x_real)
            disc_losses = disc.compute_G_loss(x_fake, x_real, gradient_checkpointing=self.gradient_checkpointing)
            for k, v in disc_losses.items():
                losses[f"{name}/{k}"] = self.loss_lambdas[name] * v

        return losses

    def compute_D_loss(self, x, mode: str):
        losses = {'loss': 0.}
        for name, disc in self.D.items():
            if self.gradient_checkpointing and self.training:
                disc_losses = checkpoint(disc.compute_D_loss, x, mode)
            else:
                disc_losses = disc.compute_D_loss(x, mode)

            # Layer-wise checkpointing (very heavy!)
            # disc_losses = disc.compute_D_loss(x, mode, gradient_checkpointing=self.gradient_checkpointing)

            losses[f"{name}/loss"] = disc_losses['loss'].detach()
            losses['loss'] += disc_losses['loss']

        return losses
