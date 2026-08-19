"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under MIT License.
https://github.com/lucidrains/BS-RoFormer/

-----------------------------------------------------
Mel-Band RoFormer backbone architecture.
https://arxiv.org/abs/2310.01809
"""

from __future__ import annotations
from packaging import version
import typing as tp

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from beartype import beartype
from einops import rearrange, pack, unpack
from rotary_embedding_torch import RotaryEmbedding
from hyper_connections import get_init_and_expand_reduce_stream_functions

from utils.torch_common import checkpoint, exists, default
from .bsroformer import RMSNorm, Transformer, FourierFeatures
from .mel_band_split import MelBandSplit

try:
    assert torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')
except AssertionError as e:
    raise e


class MelRoformer(nn.Module):

    @beartype
    def __init__(
        self,
        dim: int,
        audio_channels: int = 2,
        num_bands: int = 60,
        sample_rate: int = 44100,
        *,
        depth: int = 12,
        num_stems: int = 1,
        use_time_embed: bool = False,  # time embedding (0~1) conditioning
        time_transformer_depth=1,
        freq_transformer_depth=1,
        dim_head: int = 64,
        heads: int = 8,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        flash_attn: bool = True,
        num_residual_streams: int = 4,  # set to 1. to disable hyper connections
        num_residual_fracs: int = 1,   # can be used as an alternative to residual streams for memory efficiency while retaining benefits of hyper connections
        stft_n_fft: int = 2048,
        # @faroit recommends // 2 or // 4 for better reconstruction
        stft_hop_length: int = 512,  # near 10ms at 44100Hz, from sections 4.1, 4.4 in the paper
        stft_window_fn: str = "hann_window",
        # mask config
        mask_estimator_depth: int = 2,
        tanh_mask: bool = False,
        # training
        gradient_checkpointing: bool = True
    ):
        super().__init__()

        self.audio_channels = audio_channels
        self.num_stems = num_stems
        self.num_bands = num_bands
        self.use_time_embed = use_time_embed
        self.tanh_mask = tanh_mask
        self.gradient_checkpointing = gradient_checkpointing

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

        self.melband_split = MelBandSplit(
            dim=dim, ch_in=audio_channels, n_fft=stft_n_fft, hop_length=stft_hop_length,
            n_mels=num_bands, depth_unsplit=mask_estimator_depth, output_stream=num_stems,
            win_func=stft_window_fn, sample_rate=sample_rate, compress_exp=1.0
        )

    def forward(
        self,
        raw_audio: torch.Tensor,
        t: tp.Optional[torch.Tensor] = None
    ):
        """
        raw_audio: (bs, audio_channels, sample_length)
        t: (batch,) or None
            time embedding between 0 and 1 if use_time_embed is True

        Returns: (bs, audio_channels, sample_length)

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

        # x: (batch, n_time, n_mels, dim)
        # stft_repr: (batch, ch, n_freq, n_time, 2)
        x, stft_repr = self.melband_split(raw_audio)

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
        for time_transformer, freq_transformer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, time_v_residual, freq_v_residual = checkpoint(
                    self.__tf_block, x, time_transformer, freq_transformer, time_v_residual, freq_v_residual)
            else:
                x = rearrange(x, 'b t f d -> b f t d')
                x, ps = pack([x], '* t d')

                x, next_time_v_residual = time_transformer(x, value_residual=time_v_residual)

                time_v_residual = default(time_v_residual, next_time_v_residual)

                x, = unpack(x, ps, '* t d')
                x = rearrange(x, 'b f t d -> b t f d')
                x, ps = pack([x], '* f d')

                x, next_freq_v_residual = freq_transformer(x, value_residual=freq_v_residual)

                freq_v_residual = default(freq_v_residual, next_freq_v_residual)

                x, = unpack(x, ps, '* f d')

        # maybe reduce residual streams
        x = self.reduce_stream(x)

        # remove time embeddings
        if self.use_time_embed:
            x = x[:, 1:, 1:, :]  # (bs, time, num_mels, dim)

        # final norm
        x = self.final_norm(x)

        # mel-band unsplit
        mask = self.melband_split.melband_unsplit(x)  # complex: num_stems x (bs, ch, F, T)
        if self.num_stems == 1:
            mask = mask.unsqueeze(1)  # (bs, 1, ch, F, T)
        else:
            mask = torch.stack(mask, dim=1)  # (bs, num_stems, ch, F, T)

        if self.tanh_mask:
            mask = torch.view_as_real(mask).tanh()
            mask = torch.view_as_complex(mask)

        # add 'stem' dimension
        stft_repr = rearrange(stft_repr, 'b s f t c -> b 1 s f t c')

        # masking
        stft_repr = torch.view_as_complex(stft_repr)
        stft_repr = stft_repr * mask  # (batch, num_stems, ch, F, T)

        # istft
        stft_repr = rearrange(stft_repr, 'b n s f t -> (b n s) f t')
        recon_audio = self.melband_split.istft(stft_repr, complex_input=True, original_length=L)
        recon_audio = rearrange(recon_audio, '(b n s) t -> b n s t', s=self.audio_channels, n=self.num_stems)

        if self.num_stems == 1:
            recon_audio = recon_audio.squeeze(1)

        return recon_audio

    def __tf_block(self, x, time_transformer, freq_transformer, time_v_residual, freq_v_residual):
        x = rearrange(x, 'b t f d -> b f t d')
        x, ps = pack([x], '* t d')

        x, next_time_v_residual = time_transformer(x, value_residual=time_v_residual)

        time_v_residual = default(time_v_residual, next_time_v_residual)

        x, = unpack(x, ps, '* t d')
        x = rearrange(x, 'b f t d -> b t f d')
        x, ps = pack([x], '* f d')

        x, next_freq_v_residual = freq_transformer(x, value_residual=freq_v_residual)

        freq_v_residual = default(freq_v_residual, next_freq_v_residual)

        x, = unpack(x, ps, '* f d')

        return x, time_v_residual, freq_v_residual


class MultiSourceMelRoformer(MelRoformer):
    def __init__(
        self,
        n_src: int,
        use_time_embed: bool = False,
        **kwargs
    ):
        super().__init__(num_stems=n_src, use_time_embed=use_time_embed, **kwargs)
        assert n_src > 1, f"n_src must be greater than 1, got {n_src}"
        self.n_src = n_src
        self.use_time_embed = use_time_embed

    def forward(
        self,
        x_t: torch.Tensor,
        t: tp.Optional[torch.Tensor] = None
    ):
        """
        x_t: (bs, n_src, n_ch, sample_length)
        t: (bs)
        """
        assert self.use_time_embed == exists(t)
        bs, n_src, n_ch, L = x_t.shape
        ch_in = self.audio_channels
        assert n_src * n_ch == ch_in, f"input channels ({n_src} * {n_ch}) must match model's configured audio channels ({ch_in})"

        x_t = x_t.view(bs, n_src * n_ch, L)

        out = super().forward(x_t, t)  # (bs, n_src, n_src * n_ch, L)
        out = out.view(bs, n_src, n_src, n_ch, L)

        # prediction of each source is a weighted mixture (by masks) of all input sources

        out = out.sum(dim=2)  # (bs, n_src, n_ch, L)

        return out
