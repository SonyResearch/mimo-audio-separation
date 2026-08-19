"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under MIT License.
https://github.com/lucidrains/BS-RoFormer/

-----------------------------------------------------
Learnable Mel-band split module for audio processing.
"""

import typing as tp

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa import filters
from einops import rearrange, repeat

from utils.torch_common import print_once, default


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


def MLP(
    dim_in,
    dim_out,
    dim_hidden=None,
    depth=1,
    activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * (depth - 1)), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class BandSplit(nn.Module):
    def __init__(
        self,
        dim,
        dim_inputs: tp.List[int]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x: torch):
        """
        x: (batch, n_time, sum(dim_inputs))
        Returns: (batch, n_time, len(dim_inputs), dim)
        """
        assert x.shape[-1] == sum(self.dim_inputs), \
            f"Input dimension mismatch: expected {sum(self.dim_inputs)}, got {x.shape[-1]}"

        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


class BandUnsplit(nn.Module):
    def __init__(
        self,
        dim,
        dim_inputs: tp.List[int],
        depth: int = 1,
        output_stream: int = 1
    ):
        """
        Inverse transforme of BandSplit.
        """
        super().__init__()

        self.dim = dim
        self.dim_inputs = dim_inputs
        self.output_stream = output_stream

        self.to_freqs = nn.ModuleList([])
        for dim_in in dim_inputs:
            net = nn.Sequential(
                MLP(dim, dim_in * 2 * output_stream, dim_hidden=dim * 4, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(net)

    def forward(self, x: torch.Tensor):
        """
        x: (batch, n_time, len(dim_inputs), dim)
        Returns: (batch, n_time, output_stream, sum(dim_inputs))
        """
        assert x.shape[-2:] == (len(self.dim_inputs), self.dim)

        outs = []
        for idx, to_freq in enumerate(self.to_freqs):
            split_output = to_freq(x[:, :, idx, :])  # (batch, n_time, dim_inputs[idx] * output_stream)
            split_output = rearrange(split_output, 'b t (s f) -> b t s f', s=self.output_stream)
            outs.append(split_output)

        return torch.cat(outs, dim=-1)


class MelBandSplit(nn.Module):
    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        n_mels: int = 80,
        ch_in: int = 1,
        depth_unsplit: int = 1,
        output_stream: int = 1,
        win_func: str = 'hann_window',
        sample_rate: int = 24000,
        compress_exp: float = 0.33
    ):
        super().__init__()

        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.ch_in = ch_in
        self.output_stream = output_stream
        self.compress_exp = compress_exp

        # stft
        self.stft_kwargs = dict(n_fft=n_fft, hop_length=hop_length)

        # mel filterbank
        mel_filter_np = filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
        mel_filter = torch.from_numpy(mel_filter_np)
        mel_filter[0][0] = mel_filter[0][1] * 0.25
        mel_filter[-1, -1] = mel_filter[-1, -2] * 0.25

        # binary filter as in the MelBandRoformer paper
        n_bin = n_fft // 2 + 1
        freqs_per_band = mel_filter > 0  # (n_mels, n_bin)
        num_freqs_per_band = freqs_per_band.sum(dim=1)
        num_bands_per_freq = freqs_per_band.sum(dim=0)

        repeated_freq_indices = repeat(torch.arange(n_bin), 'f -> b f', b=n_mels)
        freq_indices = repeated_freq_indices[freqs_per_band]

        # multi-channel setting
        freq_indices = repeat(freq_indices, 'f -> f s', s=ch_in)
        freq_indices = freq_indices * ch_in + torch.arange(ch_in)
        freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        # Band split
        # NOTE: The real/imag components of complex spectral are jointly converted into embeddings.
        freqs_per_bands_with_complex = tuple(2 * f * ch_in for f in num_freqs_per_band.tolist())
        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        # Band unsplit
        self.band_unsplit = BandUnsplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            depth=depth_unsplit,
            output_stream=output_stream
        )

        win_fft = getattr(torch, win_func)(n_fft)
        self.register_buffer('win_fft', win_fft, persistent=False)
        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)

    def forward(self, x):
        """
        x: (batch, ch, time)
        Returns:
            x: (batch, n_time, n_mels, emb_dim)
            stft_repr: (batch, ch, n_freq, n_time, 2)
        """
        bs, ch, L = x.shape
        assert ch == self.ch_in, f"expected {self.ch_in} channels, got {ch}"

        # STFT
        x = rearrange(x, 'b c t -> (b c) t')
        stft_repr = torch.stft(x, **self.stft_kwargs, window=self.win_fft, return_complex=True)

        # Compress amplitude
        if self.compress_exp != 1.0:
            stft_repr = stft_repr.abs().pow(self.compress_exp) * torch.exp(1j * stft_repr.angle())
        stft_repr = torch.view_as_real(stft_repr)  # (bs, F, T, 2)

        # output
        stft_repr = rearrange(stft_repr, '(b s) f t c -> b s f t c', b=bs, s=ch)

        # merge multi-channels into the frequency
        stft_repr_ = rearrange(stft_repr, 'b s f t c -> b (f s) t c', b=bs, s=ch)
        batch_arange = torch.arange(bs, device=stft_repr_.device)[..., None]
        x = stft_repr_[batch_arange, self.freq_indices]  # (bs, F_mel, T, 2)

        # fold the complex (real and imag)
        x = rearrange(x, 'b f t c -> b t (f c)')

        # band split
        x = self.band_split(x)  # (bs, T, n_mels, dim)

        return x, stft_repr

    def melband_unsplit(self, z):
        """
        Invert embeddings into STFT domain.

        z: (batch, n_time, n_mels, emb_dim)
        Returns: List of (batch, ch, n_freq, n_time) or (batch, ch, n_freq, n_time)
        """
        assert z.shape[-2:] == (self.n_mels, self.dim)
        bs, T = z.shape[:2]
        F = self.n_fft // 2 + 1

        # band unsplit
        z = self.band_unsplit(z)  # (bs, T, output_stream, sum(dim_inputs))
        z = rearrange(z, 'b t o (f c) -> b o f t c', f=len(self.freq_indices), c=2)  # (bs, output_stream, F_mel, T, 2)

        if z.dtype == torch.bfloat16:
            z = z.to(torch.float32)

        z = torch.view_as_complex(z)  # (bs, output_stream, F_mel, T)

        # to spectrogram
        x = torch.zeros(bs, self.output_stream, F * self.ch_in, T, device=z.device, dtype=torch.complex64)
        x[:, :, self.freq_indices] += z

        x = rearrange(x, 'b o (f s) t -> b o s f t', s=self.ch_in)  # (bs, output_stream, ch, F, T)

        # averaged by the number of overlapped mel bands
        x = x / self.num_bands_per_freq[None, None, None, :, None]

        # split into list
        x = x.unbind(dim=1)  # list of (bs, ch, F, T)

        return x if self.output_stream > 1 else x[0]

    def istft(self, x, complex_input: bool = True, original_length: int = None):
        """
        Inverse STFT.

        x: (batch, n_freq, n_time) or (batch, n_freq, n_time, 2)
        Returns: (batch, time)
        """
        assert x.shape[1] == self.n_fft // 2 + 1

        # to complex value
        if not complex_input:
            assert x.shape[-1] == 2
            x = torch.view_as_complex(x)

        # revert compression
        if self.compress_exp != 1.0:
            x = x.abs().pow(1.0 / self.compress_exp) * torch.exp(1j * x.angle())

        # inverse stft
        x = torch.istft(x, **self.stft_kwargs, window=self.win_fft, length=original_length)

        return x
