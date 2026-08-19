"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under MIT License.
https://github.com/lucidrains/BS-RoFormer/

-----------------------------------------------------
BS-RoFormer backbone architecture.
https://arxiv.org/abs/2309.02612
"""

from __future__ import annotations
from functools import partial
from packaging import version
import typing as tp
import math

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from beartype import beartype
from einops import rearrange, pack, unpack
from rotary_embedding_torch import RotaryEmbedding
from hyper_connections import get_init_and_expand_reduce_stream_functions

from utils.torch_common import checkpoint, exists, default, print_once

try:
    assert torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')
except AssertionError as e:
    raise e

# helper functions


def pack_one(t, pattern):
    return pack([t], pattern)


def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]


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


class Attend(nn.Module):
    def __init__(
        self,
        dropout=0.,
        flash=False,
        scale=None
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash

        # determine efficient attention configs for cuda and cpu
        self.cpu_config = [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]
        self.cuda_config = [SDPBackend.FLASH_ATTENTION] if self.flash else [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def flash_attn(self, q, k, v):
        is_cuda = q.is_cuda

        if exists(self.scale):
            default_scale = q.shape[-1] ** -0.5
            q = q * (self.scale / default_scale)

        # Check if there is a compatible device for flash attention

        config = self.cuda_config if is_cuda else self.cpu_config

        # pytorch 2.0 flash attn: q, k, v, mask, dropout, softmax_scale

        with sdpa_kernel(config):
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.
            )

        return out

    def forward(self, q, k, v):
        """
        einstein notation
        b - batch
        h - heads
        n, i, j - sequence length (base sequence length, source, target)
        d - feature dimension
        """

        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        # similarity

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale

        # attention

        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # aggregate values

        out = einsum("b h i j, b h j d -> b h i d", attn, v)

        return out


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.scale * self.gamma


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult=4,
        dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        dropout=0.,
        rotary_embed=None,
        flash=True,
        learned_value_residual_mix=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_value_residual_mix = nn.Linear(dim, heads) if learned_value_residual_mix else None

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x, value_residual=None):
        x = self.norm(x)

        q, k, v = rearrange(self.to_qkv(x), 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.heads)

        orig_v = v

        if exists(self.to_value_residual_mix):
            mix = self.to_value_residual_mix(x)
            mix = rearrange(mix, 'b n h -> b h n 1').sigmoid()

            assert exists(value_residual)
            v = v.lerp(value_residual, mix)

        if exists(self.rotary_embed):
            q = self.rotary_embed.rotate_queries_or_keys(q)
            k = self.rotary_embed.rotate_queries_or_keys(k)

        out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * rearrange(gates, 'b n h -> b h n 1').sigmoid()

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), orig_v


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        dim_head=64,
        heads=8,
        attn_dropout=0.,
        ff_dropout=0.,
        ff_mult=4,
        norm_output=True,
        rotary_embed=None,
        flash_attn=True,
        add_value_residual=False,
        num_residual_streams=1,
        num_residual_fracs=1
    ):
        super().__init__()
        self.layers = nn.ModuleList()

        init_hyper_conn, *_ = get_init_and_expand_reduce_stream_functions(num_residual_streams, num_fracs=num_residual_fracs)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                init_hyper_conn(dim=dim, branch=Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                rotary_embed=rotary_embed, flash=flash_attn, learned_value_residual_mix=add_value_residual)),
                init_hyper_conn(dim=dim, branch=FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout))
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x, value_residual=None):

        first_values = None

        for attn, ff in self.layers:
            x, next_values = attn(x, value_residual=value_residual)

            first_values = default(first_values, next_values)

            x = ff(x)

        return self.norm(x), first_values

# bandsplit module


class BandSplit(nn.Module):
    @beartype
    def __init__(
        self,
        dim,
        dim_inputs: tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_features = nn.ModuleList()

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def forward(self, x):
        x = x.split(self.dim_inputs, dim=-1)

        outs = []
        for split_input, to_feature in zip(x, self.to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)  # (bs, num_frames, num_bands, dim)


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


class MaskEstimator(nn.Module):
    @beartype
    def __init__(
        self,
        dim: int,
        dim_inputs: tuple[int, ...],
        depth: int,
        mlp_expansion_factor: tp.Union[int, float] = 4.
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self.to_freqs = nn.ModuleList()
        dim_hidden = int(round(dim * mlp_expansion_factor))

        for dim_in in dim_inputs:

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def forward(self, x):
        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)

# main class


DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)  # NOTE: only for stft_n_fft=2048


class BSRoformer(nn.Module):

    @beartype
    def __init__(
        self,
        dim: int,
        audio_channels: int = 2,
        *,
        depth: int = 12,
        num_stems=1,
        use_time_embed: bool = False,  # time embedding (0~1) conditioning
        time_transformer_depth=1,
        freq_transformer_depth=1,
        freqs_per_bands: tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,  # in the paper, they divide into ~60 bands, test with 1 for starters
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        flash_attn: bool = True,
        num_residual_streams: int = 1,  # set to 1. to disable hyper connections
        num_residual_fracs: int = 1,   # can be used as an alternative to residual streams for memory efficiency while retaining benefits of hyper connections
        # STFT parameters
        stft_n_fft: int = 2048,
        stft_win_length: int = 2048,
        # @faroit recommends // 2 or // 4 for better reconstruction
        stft_hop_length: int = 512,  # near 10ms at 44100Hz, from sections 4.1, 4.4 in the paper
        stft_window_fn: str = "hann_window",
        # mask config
        mask_estimator_depth: int = 2,
        mask_estimator_mult: tp.Union[int, float] = 4.,
        tanh_mask: bool = False,
        # training
        gradient_checkpointing: bool = True,
        checkpoint_every: int = 1
    ):
        super().__init__()

        self.audio_channels = audio_channels
        self.num_stems = num_stems
        self.num_bands = len(freqs_per_bands)
        self.use_time_embed = use_time_embed
        self.tanh_mask = tanh_mask
        self.gradient_checkpointing = gradient_checkpointing
        self.checkpoint_every = max(checkpoint_every, 1)

        _, self.expand_stream, self.reduce_stream = get_init_and_expand_reduce_stream_functions(
            num_residual_streams, disable=(num_residual_streams == 1))

        # Time embedding

        if self.use_time_embed:
            dim_f = 256
            # Time embeddings for each of time/freq transformers
            self.time_embedding_t = nn.Sequential(
                FourierFeatures(1, dim_f), nn.Linear(dim_f, dim, bias=True), nn.SiLU(), nn.Linear(dim, dim, bias=True)
            )
            self.time_embedding_f = nn.Sequential(
                FourierFeatures(1, dim_f), nn.Linear(dim_f, dim, bias=True), nn.SiLU(), nn.Linear(dim, dim, bias=True)
            )

        # Main transformer layers

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            num_residual_streams=num_residual_streams,
            num_residual_fracs=num_residual_fracs,
            norm_output=False,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        self.layers = nn.ModuleList()
        for layer_index in range(depth):
            is_first = layer_index == 0

            self.layers.append(nn.ModuleList([
                Transformer(depth=time_transformer_depth, rotary_embed=time_rotary_embed, add_value_residual=not is_first, **transformer_kwargs),
                Transformer(depth=freq_transformer_depth, rotary_embed=freq_rotary_embed, add_value_residual=not is_first, **transformer_kwargs)
            ]))

        self.final_norm = RMSNorm(dim)

        # Band splitting and un-splitting

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=False
        )

        stft_window_fn = getattr(torch, stft_window_fn)
        self.stft_window_fn = partial(stft_window_fn, stft_win_length)

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList()
        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mask_estimator_mult
            )

            self.mask_estimators.append(mask_estimator)

    def forward(
        self,
        raw_audio: torch.Tensor,
        t: tp.Optional[torch.Tensor] = None,
        # training
        gradient_checkpointing: bool = True,
        checkpoint_every: tp.Optional[int] = None
    ):
        """
        Args:
        raw_audio: (batch, audio_channels, sample_length)
        t: (batch,) or None
            time embedding between 0 and 1 if use_time_embed is True

        einops

        b - batch
        f - freq
        t - time
        s - audio channel
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """
        bs, channels, L = raw_audio.shape
        assert channels == self.audio_channels, f"expected {self.audio_channels} channels, got {channels}"

        # overwrite training configs if provided
        gradient_checkpointing = (gradient_checkpointing and self.gradient_checkpointing)
        checkpoint_every = default(checkpoint_every, self.checkpoint_every)

        # to stft
        raw_audio, batch_audio_channel_packed_shape = pack_one(raw_audio, '* t')

        stft_window = self.stft_window_fn(device=raw_audio.device)

        stft_repr = torch.stft(raw_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = unpack_one(stft_repr, batch_audio_channel_packed_shape, '* f t c')
        stft_repr = rearrange(stft_repr, 'b s f t c -> b (f s) t c')

        # band split
        x = rearrange(stft_repr, 'b f t c -> b t (f c)')
        x = self.band_split(x)  # (bs, time, num_bands, dim)

        # prepend time embeddings
        if self.use_time_embed:
            assert exists(t) and len(t) == bs, "Time embedding t must be provided with shape(batch, )."
            time_emb_t = self.time_embedding_t(t[:, None])
            time_emb_f = self.time_embedding_f(t[:, None])  # (bs, dim)
            time_emb_t_r = time_emb_t[:, None, None, :].expand(-1, 1, x.shape[2], -1)
            time_emb_f_r = time_emb_f[:, None, None, :].expand(-1, x.shape[1] + 1, 1, -1)
            x = torch.cat([time_emb_t_r, x], dim=1)
            x = torch.cat([time_emb_f_r, x], dim=2)  # (bs, time+1, num_bands+1, dim)

        # maybe expand residual streams
        x = self.expand_stream(x)

        # time / frequency attention
        time_v_residual = None
        freq_v_residual = None
        for idx_l, (time_transformer, freq_transformer) in enumerate(self.layers):
            is_checkpoint = gradient_checkpointing and (((idx_l + 1) % checkpoint_every) == 0)
            if is_checkpoint and self.training:
                x, time_v_residual, freq_v_residual = checkpoint(
                    self.__tf_block,
                    x, time_transformer, freq_transformer, time_v_residual, freq_v_residual
                )
            else:
                x, time_v_residual, freq_v_residual = self.__tf_block(
                    x, time_transformer, freq_transformer, time_v_residual, freq_v_residual
                )

        # maybe reduce residual streams
        x = self.reduce_stream(x)

        # remove time embeddings
        if self.use_time_embed:
            x = x[:, 1:, 1:, :]  # (bs, time, num_bands, dim)

        # final norm
        x = self.final_norm(x)

        num_stems = len(self.mask_estimators)

        mask = torch.stack([fn(x) for fn in self.mask_estimators], dim=1)
        mask = rearrange(mask, 'b n t (f c) -> b n f t c', c=2)

        # add 'stem' dimension
        stft_repr = rearrange(stft_repr, 'b f t c -> b 1 f t c')

        # NOTE: STFT and iSTFT are computed with complex64
        if mask.dtype != torch.float32:
            mask = mask.to(torch.float32)

        if self.tanh_mask:
            # apply tanh to mask for bounded values
            mask = mask.tanh()

        # masking
        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask)
        stft_repr = stft_repr * mask

        # istft
        stft_repr = rearrange(stft_repr, 'b n (f s) t -> (b n s) f t', s=self.audio_channels)
        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False, length=L)
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=num_stems)

        if num_stems == 1:
            recon_audio = recon_audio.squeeze(1)

        return recon_audio

    def __tf_block(self, x, time_transformer, freq_transformer, time_v_residual, freq_v_residual):
        x = rearrange(x, 'b t f d -> b f t d')

        # time transformer
        x, ps = pack([x], '* t d')
        x, next_time_v_residual = time_transformer(x, value_residual=time_v_residual)
        time_v_residual = default(time_v_residual, next_time_v_residual)
        x, = unpack(x, ps, '* t d')  # (b, f, t, d)

        x = rearrange(x, 'b f t d -> b t f d')

        # frequency transformer
        x, ps = pack([x], '* f d')
        x, next_freq_v_residual = freq_transformer(x, value_residual=freq_v_residual)
        freq_v_residual = default(freq_v_residual, next_freq_v_residual)
        x, = unpack(x, ps, '* f d')

        return x, time_v_residual, freq_v_residual

    def __tf_blocks(self, x, layers, time_v_residual=None, freq_v_residual=None):
        for time_transformer, freq_transformer in layers:
            x, time_v_residual, freq_v_residual = self.__tf_block(
                x, time_transformer, freq_transformer, time_v_residual, freq_v_residual
            )

        return x, time_v_residual, freq_v_residual


class MultiSourceBSRoformer(BSRoformer):
    def __init__(
        self,
        n_src: int,
        use_time_embed: bool = True,
        input_mixture: bool = False,  # whether to concatenate the input mixture to the model input as prior information
        **kwargs
    ):
        super().__init__(num_stems=n_src, use_time_embed=use_time_embed, **kwargs)
        assert n_src > 1, f"n_src must be greater than 1, got {n_src}"
        self.n_src = n_src
        self.use_time_embed = use_time_embed
        self.input_mixture = input_mixture

    def forward(
        self,
        x_t: torch.Tensor,
        t: tp.Optional[torch.Tensor] = None,
        **kwargs
    ):
        """
        x_t: (bs, n_src, n_ch, sample_length)
        t: (bs)
        """
        assert self.use_time_embed == exists(t)
        bs, n_src, n_ch, L = x_t.shape
        ch_in = self.audio_channels
        assert (n_src + int(self.input_mixture)) * n_ch == ch_in, \
            f"input channels (({n_src} + {int(self.input_mixture)}) * {n_ch}) must match model's configured audio channels ({ch_in})"

        mixture = x_t.sum(dim=1)  # (bs, n_ch, L)
        x_t = x_t.view(bs, n_src * n_ch, L)
        if self.input_mixture:
            x_t = torch.cat([x_t, mixture], dim=1)

        out = super().forward(x_t, t, **kwargs)  # (bs, n_src, (n_src + int(self.input_mixture)) * n_ch, L)
        out = out.view(bs, n_src, n_src + int(self.input_mixture), n_ch, L)

        # prediction of each source is a weighted mixture (by masks) of all input sources

        out = out.sum(dim=2)  # (bs, n_src, n_ch, L)

        return out
