
import typing as tp

import torch
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from einops import rearrange

from .base import MultiDiscriminator
from utils.torch_common import checkpoint


def WNConv2d(*args, **kwargs):
    act = kwargs.pop("act", True)
    conv = weight_norm(nn.Conv2d(*args, **kwargs))
    if not act:
        return conv
    return nn.Sequential(conv, nn.LeakyReLU(0.1))


class MRD(nn.Module):

    BANDS = [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]

    def __init__(
        self,
        num_channels: int,
        hidden_dim: int = 32,
        window_length: int = 1024,
        hop_factor: float = 0.25,
        bands: list = BANDS,
        num_feat_layers: int = 5
    ):
        """
        Complex multi-band spectrogram discriminator proposed in the DAC paper.
        [https://arxiv.org/abs/2306.06546]
        """
        super().__init__()

        self.num_channels = num_channels
        self.window_length = window_length
        self.hop_factor = hop_factor
        self.num_feat_layers = num_feat_layers
        self.stft_params = {
            "window_length": window_length,
            "hop_length": int(window_length * hop_factor),
            "window": torch.hann_window(window_length),
        }

        n_fft = window_length // 2 + 1
        bands = [(int(b[0] * n_fft), int(b[1] * n_fft)) for b in bands]
        self.bands = bands

        ch_in = 2 * num_channels
        ch = hidden_dim

        def convs(): return nn.ModuleList(
            [
                WNConv2d(ch_in, ch, (3, 9), (1, 1), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4)),
                WNConv2d(ch, ch, (3, 3), (1, 1), padding=(1, 1)),
            ]
        )
        self.band_convs = nn.ModuleList([convs() for _ in range(len(self.bands))])
        self.conv_post = WNConv2d(ch, 1, (3, 3), (1, 1), padding=(1, 1), act=False)

    def spectrogram(self, x):
        """
        x: (B, T)
        out: list of (B, 2, T', F_band)
        """
        x = torch.stft(
            x,
            n_fft=self.window_length,
            hop_length=int(self.window_length * self.hop_factor),
            window=torch.hann_window(self.window_length, device=x.device),
            return_complex=True,
        )
        x = torch.view_as_real(x)  # shape: (B, F, T, 2)
        x = x.permute(0, 3, 2, 1)  # (B, 2, T, F)

        # Split into bands
        x_bands = [x[..., b[0]: b[1]] for b in self.bands]
        return x_bands

    def forward(self, x: torch.Tensor, return_feature: bool = False,
                gradient_checkpointing: bool = False):
        """
        x: (B, C, T)
        """
        bs, n_ch, L = x.shape
        assert n_ch == self.num_channels

        x = rearrange(x, "b c t -> (b c) t")
        x_bands = self.spectrogram(x)

        out = []
        feats = []
        for band, stack in zip(x_bands, self.band_convs):
            band = rearrange(band, "(b c) p t f -> b (c p) t f", c=n_ch)
            # for idx_l, layer in enumerate(stack):
            #     band = layer(band)
            #     if idx_l < self.num_feat_layers:
            #         feats.append(band)
            # out.append(band)
            if self.training and gradient_checkpointing:
                out_, feats_ = checkpoint(self.__process_band, band, stack)
            else:
                out_, feats_ = self.__process_band(band, stack)

            out.append(out_)
            feats.extend(feats_)

        out = torch.cat(out, dim=-1)
        logits = self.conv_post(out)

        return (logits, feats) if return_feature else logits

    def __process_band(self, band, stack):
        feats = []
        for idx_l, layer in enumerate(stack):
            band = layer(band)
            if idx_l < self.num_feat_layers:
                feats.append(band)

        return band, feats


class MixingMRD(MultiDiscriminator):
    """
    MRDs that detect whether samples are mixed or not.
    """

    def __init__(
        self,
        num_channels: int,
        num_sources: int,
        fft_sizes: tp.List[int],
        hidden_dim: int = 32,
        hop_factor: float = 0.25,
        num_feat_layers: int = 4
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_sources = num_sources

        ch_in = num_channels * num_sources
        self.model = nn.ModuleList(
            [MRD(num_channels=ch_in, window_length=fft_size, hidden_dim=hidden_dim,
                 hop_factor=hop_factor, num_feat_layers=num_feat_layers)
             for fft_size in fft_sizes]
        )

    def forward(self, x: torch.Tensor, return_feature: bool = True,
                gradient_checkpointing: bool = False):
        """
        x: (B, n_src, n_ch, L)
        """
        bs, n_src, n_ch, L = x.shape
        assert n_ch == self.num_channels and n_src == self.num_sources, f"{n_ch} != {self.num_channels} or {n_src} != {self.num_sources}"
        x = rearrange(x, "b s c t -> b (s c) t")

        logits = []
        feats = []
        for mrd in self.model:
            output = mrd(x, return_feature=return_feature, gradient_checkpointing=gradient_checkpointing)
            if return_feature:
                logit, feat = output
                feats.append(feat)
            else:
                logit = output

            logits.append(logit)

        return (logits, feats) if return_feature else logits


class StemMRD(MultiDiscriminator):
    """
    MRDs that detect whether each stem is real or not.
    """

    def __init__(
        self,
        num_channels: int,
        num_sources: int,
        fft_sizes: tp.List[int],
        hidden_dim: int = 32,
        hop_factor: float = 0.25,
        num_feat_layers: int = 4
    ):
        super().__init__()
        self.num_channels = num_channels
        self.num_sources = num_sources
        self.fft_sizes = fft_sizes

        ch_in = num_channels
        # len(self.model) == num_sources * len(fft_sizes)
        self.model = nn.ModuleList()
        for _ in range(num_sources):
            self.model.extend(
                [MRD(num_channels=ch_in, window_length=fft_size, hidden_dim=hidden_dim,
                     hop_factor=hop_factor, num_feat_layers=num_feat_layers)
                 for fft_size in fft_sizes]
            )

    def forward(self, x: torch.Tensor, return_feature: bool = True,
                gradient_checkpointing: bool = False):
        """
        x: (B, n_src, n_ch, L)
        """
        bs, n_src, n_ch, L = x.shape
        assert n_ch == self.num_channels and n_src == self.num_sources, f"{n_ch} != {self.num_channels} or {n_src} != {self.num_sources}"
        num_ffts = len(self.fft_sizes)

        logits = []
        feats = []
        for idx_src in range(n_src):
            x_src = x[:, idx_src, :, :]  # (B, C, L)
            for idx_fft in range(num_ffts):
                mrd = self.model[idx_src * num_ffts + idx_fft]
                output = mrd(x_src, return_feature=return_feature, gradient_checkpointing=gradient_checkpointing)
                if return_feature:
                    logit, feat = output
                    feats.append(feat)
                else:
                    logit = output

                logits.append(logit)

        return (logits, feats) if return_feature else logits
