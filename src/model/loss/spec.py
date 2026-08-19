"""
Copyright (C) 2024 Yukara Ikemiya
"""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def complex_abs(x: torch.Tensor):
    abs_diff = torch.sqrt(x.real**2 + x.imag**2)
    return abs_diff


class MRSTFTLoss(nn.Module):
    """
    Multi-resolution STFT loss
    """

    def __init__(
        self,
        n_ffts: tp.List[int] = [512, 1024, 2048],
        hop_sizes: tp.List[int] = [80, 150, 300],
        win_sizes: tp.Optional[tp.List[int]] = None,
        window: str = "hann_window",
        decibel_mae: bool = False,
        eps_amp: float = 1e-4
    ):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes if win_sizes is not None else n_ffts
        assert len(self.n_ffts) == len(self.hop_sizes) == len(self.win_sizes)
        self.win_func = getattr(torch, window)
        self.decibel_mae = decibel_mae

        # minimum value of spectral amplitude for numerical stability
        self.eps_amp = eps_amp

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ):
        """
        x, y: (B, n_src, C, L)
        """
        assert x.shape == y.shape  # (B, 1, L)
        name_mae = 'mrstft/l1' if not self.decibel_mae else 'mrstft/l1_db'
        losses = {
            name_mae: 0.,
            # only for logging
            'mrstft/sc': 0.,
            'mrstft/logmag': 0.
        }

        B, n_src, n_ch, L = x.shape
        x = rearrange(x, 'B n_src n_ch L -> (B n_src n_ch) L')
        y = rearrange(y, 'B n_src n_ch L -> (B n_src n_ch) L')

        for n_fft, win_size, hop_size in zip(self.n_ffts, self.win_sizes, self.hop_sizes):
            # STFT
            spec_x = self.__stft(x, n_fft, win_size, hop_size)
            spec_y = self.__stft(y, n_fft, win_size, hop_size)

            # Complex MAE loss
            if not self.decibel_mae:
                cmae_loss = F.l1_loss(spec_x, spec_y)
            else:
                diff_a = complex_abs(spec_y - spec_x)  # (B*n_src*n_ch, F, T)
                y_a = complex_abs(spec_y)
                diff_a = rearrange(diff_a, '(B n_src n_ch) F T -> B n_src n_ch F T', B=B, n_src=n_src, n_ch=n_ch)
                y_a = rearrange(y_a, '(B n_src n_ch) F T -> B n_src n_ch F T', B=B, n_src=n_src, n_ch=n_ch)

                # source-wise computation
                cmae_loss = diff_a.mean(dim=[2, 3, 4]) / (y_a.mean(dim=[2, 3, 4]) + self.eps_amp)
                cmae_loss = (20 * cmae_loss.log10()).mean()

            aspec_x = torch.view_as_real(spec_x).pow(2).sum(-1).sqrt()
            aspec_y = torch.view_as_real(spec_y).pow(2).sum(-1).sqrt()  # (B, F, T)

            aspec_x = aspec_x.clamp(min=self.eps_amp)
            aspec_y = aspec_y.clamp(min=self.eps_amp)

            aspec_x = rearrange(aspec_x, '(B n_src n_ch) F T -> B n_src n_ch F T', B=B, n_src=n_src, n_ch=n_ch)
            aspec_y = rearrange(aspec_y, '(B n_src n_ch) F T -> B n_src n_ch F T', B=B, n_src=n_src, n_ch=n_ch)

            # spectral convergence
            # source-wise computation
            sc_loss = (aspec_y - aspec_x).norm(p=2, dim=[2, 3, 4]) / aspec_y.norm(p=2, dim=[2, 3, 4])
            sc_loss = sc_loss.mean()

            # magnitude loss
            logmag_loss = F.l1_loss(aspec_y.log(), aspec_x.log())

            losses[name_mae] += cmae_loss
            losses['mrstft/sc'] += sc_loss.detach()
            losses['mrstft/logmag'] += logmag_loss.detach()

        losses[name_mae] /= len(self.n_ffts)
        losses['mrstft/sc'] /= len(self.n_ffts)
        losses['mrstft/logmag'] /= len(self.n_ffts)

        return losses

    def __stft(self, x, n_fft, win_size, hop_size):
        window = self.win_func(win_size, device=x.device)
        stft_spec = torch.stft(
            x, n_fft, hop_length=hop_size, win_length=win_size, window=window,
            center=True, normalized=False, onesided=True, return_complex=True)

        return stft_spec


class MRSTFT_RMSELoss(nn.Module):
    """
    Multi-resolution STFT RMSE loss
    """

    def __init__(
        self,
        n_ffts: tp.List[int] = [512, 1024, 2048],
        hop_sizes: tp.List[int] = [80, 150, 300],
        win_sizes: tp.Optional[tp.List[int]] = None,
        window: str = "hann_window",
        eps: float = 1e-8
    ):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes if win_sizes is not None else n_ffts
        assert len(self.n_ffts) == len(self.hop_sizes) == len(self.win_sizes)
        self.win_func = getattr(torch, window)
        self.eps = eps

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor
    ):
        """
        x, y: (B, n_src, C, L)
        """
        assert x.shape == y.shape  # (B, 1, L)
        losses = {
            'mrstft/rmse': 0.,
        }

        B, n_src, n_ch, L = x.shape
        x = rearrange(x, 'B n_src n_ch L -> (B n_src n_ch) L')
        y = rearrange(y, 'B n_src n_ch L -> (B n_src n_ch) L')

        for n_fft, win_size, hop_size in zip(self.n_ffts, self.win_sizes, self.hop_sizes):
            # STFT
            spec_x = self.__stft(x, n_fft, win_size, hop_size)
            spec_y = self.__stft(y, n_fft, win_size, hop_size)

            spec_x = torch.view_as_real(spec_x)
            spec_y = torch.view_as_real(spec_y)  # (B, F, T, 2)

            spec_x = rearrange(spec_x, '(B n_src n_ch) F T c -> B n_src n_ch F T c', B=B, n_src=n_src, n_ch=n_ch)
            spec_y = rearrange(spec_y, '(B n_src n_ch) F T c -> B n_src n_ch F T c', B=B, n_src=n_src, n_ch=n_ch)

            loss = F.mse_loss(spec_x, spec_y, reduction='none')

            # RMSE
            loss = (loss.mean(dim=(2, 3, 4, 5)).sqrt() + self.eps).mean(dim=(0, 1))

            losses['mrstft/rmse'] += loss

        losses['mrstft/rmse'] /= len(self.n_ffts)

        return losses

    def __stft(self, x, n_fft, win_size, hop_size):
        window = self.win_func(win_size, device=x.device)
        stft_spec = torch.stft(
            x, n_fft, hop_length=hop_size, win_length=win_size, window=window,
            center=True, normalized=False, onesided=True, return_complex=True)

        return stft_spec
