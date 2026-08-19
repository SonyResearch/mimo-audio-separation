"""
Copyright (C) 2025 Yukara Ikemiya

-----------------------------------------------------
FLOSS backbone architecture.
(FLOSS: Source Separation by Flow Matching.)
https://arxiv.org/abs/2505.16119
"""

import typing as tp

import torch
from torch import nn
from einops import rearrange

from ..bsroformer.mel_band_split import MelBandSplit
from .tflocoformer_modules import Attention, RMSGroupNorm, SwiGLUConvDeconv1d
from .time_embedding import ScaleShiftTimeEmbedder, apply_scale_shift
# from .noise_shaping import EnergyEnvelopeNoiseShaping
from utils.torch_common import checkpoint, exists, print_once


class BSJA(nn.Module):
    """
    Band-source Joint Attention (BSJA) module in Fig.2.
    """

    def __init__(
        self,
        dim_in: int,
        dim_attn: int,
        dim_ffn: int,
        n_heads: int,
        conv_kernel: int = 5,
        args_input: dict = {
            'n_mels': 80,
            'n_src': 2
        },
        flash_attention: bool = True
    ):
        super().__init__()
        assert dim_attn % n_heads == 0, "dim_attn must be divisible by n_heads"

        self.dim_in = dim_in
        self.dim_attn = dim_attn
        self.n_heads = n_heads
        self.args_input = args_input

        # norm layer
        self.norm_a = RMSGroupNorm(dim=dim_in, num_groups=4)
        self.norm_b = RMSGroupNorm(dim=dim_in, num_groups=4)

        # time embedding
        self.time_embedder = ScaleShiftTimeEmbedder(dim_in, num_embed=6)

        # attention layers ([time] x [source, band])
        self.attn = Attention(
            dim_in=dim_in, dim_attn=dim_attn, n_heads=n_heads,
            use_rope=True, conv_kernel=conv_kernel,
            dropout=0.0, flash_attention=flash_attention, input_shape='1d'
        )

        # FFN (Conv-SwishGLU)
        self.mlp = SwiGLUConvDeconv1d(
            dim=dim_in, dim_inner=dim_ffn, conv1d_kernel=4, conv1d_shift=1, dropout=0.0
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ):
        """
        x: (bs, n_src, n_band, n_time, dim_in)
        t: (bs)
        """

        bs, n_src, n_band, n_time, dim_in = x.shape
        assert bs == t.shape[0], "Batch size of x and t must match"
        assert n_band == self.args_input['n_mels']
        assert dim_in == self.dim_in, f"Input dimension {dim_in} does not match expected {self.dim_in}"

        # get time parameters
        scale_a1, shift_a1, scale_a2, scale_b1, shift_b1, scale_b2 = self.time_embedder(t)

        # norm
        h = self.norm_a(x)
        # 1st scale/shift
        h = apply_scale_shift(h, scale_a1, shift_a1)
        # attention
        h = rearrange(h, 'n s b t d -> (n s b) t d')
        h = self.attn(h)
        h = rearrange(h, '(n s b) t d -> n s b t d', n=bs, s=n_src, b=n_band)
        # 2nd scale
        h = apply_scale_shift(h, scale_a2)

        # shortcut
        x = x + h

        # norm
        h = self.norm_b(x)
        # 1st scale/shift
        h = apply_scale_shift(h, scale_b1, shift_b1)
        # FFN
        h = rearrange(h, 'n s b t d -> (n s) b t d')
        h = self.mlp(h)
        h = rearrange(h, '(n s) b t d -> n s b t d', n=bs, s=n_src)
        # 2nd scale
        h = apply_scale_shift(h, scale_b2)

        # shortcut
        x = x + h

        return x


class TSPA(nn.Module):
    """
    Time-source Parallel Attention (TSPA) module in Fig.2.
    """

    def __init__(
        self,
        dim_in: int,
        dim_attn_mhsa: int,
        n_heads_mhsa: int,
        dim_attn_cmhsa: int,
        n_heads_cmhsa: int,
        dim_ffn: int,
        conv_kernel_cmhsa: tp.Tuple[int, int] = (7, 5),
        args_input: dict = {
            'n_mels': 80,
            'n_src': 2,
            'n_time': 500
        },
        flash_attention: bool = True
    ):
        super().__init__()
        assert dim_attn_mhsa % n_heads_mhsa == dim_attn_cmhsa % n_heads_cmhsa == 0, "dim_attn must be divisible by n_heads"
        self.dim_in = dim_in
        self.dim_attn_mhsa = dim_attn_mhsa
        self.n_heads_mhsa = n_heads_mhsa
        self.dim_attn_cmhsa = dim_attn_cmhsa
        self.n_heads_cmhsa = n_heads_cmhsa
        self.args_input = args_input

        # norm layer
        self.norm_a = RMSGroupNorm(dim=dim_in, num_groups=4)
        self.norm_b = RMSGroupNorm(dim=dim_in, num_groups=4)

        # time embedding
        self.time_embedder = ScaleShiftTimeEmbedder(dim_in, num_embed=6)

        # MHSA attention layers ([source, band] x [time])
        self.attn_mhsa = Attention(
            dim_in=dim_in, dim_attn=self.dim_attn_mhsa, n_heads=self.n_heads_mhsa,
            use_rope=True, conv_kernel=None,
            dropout=0.0, flash_attention=flash_attention, input_shape='1d'
        )

        # CMHSA attention layers ([time, band] x [source])
        self.attn_cmhsa = Attention(
            dim_in=dim_in, dim_attn=self.dim_attn_cmhsa, n_heads=self.n_heads_cmhsa,
            use_rope=True, conv_kernel=conv_kernel_cmhsa,
            dropout=0.0, flash_attention=flash_attention, input_shape='2d'
        )

        # FFN (Conv-SwishGLU)
        self.mlp = SwiGLUConvDeconv1d(
            dim=dim_in, dim_inner=dim_ffn, conv1d_kernel=4, conv1d_shift=1, dropout=0.0
        )

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor
    ):
        """
        x: (bs, n_src, n_band, n_time, dim_in)
        t: (bs)
        """
        bs, n_src, n_band, n_time, dim_in = x.shape
        assert bs == t.shape[0], "Batch size of x and t must match"
        assert n_band == self.args_input['n_mels']
        assert dim_in == self.dim_in, f"Input dimension {dim_in} does not match expected {self.dim_in}"

        # get time parameters
        scale_a1, shift_a1, scale_a2, scale_b1, shift_b1, scale_b2 = self.time_embedder(t)

        # norm
        h = self.norm_a(x)
        # 1st scale/shift
        h = apply_scale_shift(h, scale_a1, shift_a1)
        # MHSA
        h1 = rearrange(h, 'n s b t d -> (n t) (s b) d')
        h1 = self.attn_mhsa(h1)
        h1 = rearrange(h1, '(n t) (s b) d -> n s b t d', n=bs, s=n_src, b=n_band)
        # CMHSA (time/band order)
        h2 = rearrange(h, 'n s b t d -> (n s) t b d')
        h2 = self.attn_cmhsa(h2)
        h2 = rearrange(h2, '(n s) t b d -> n s b t d', n=bs, s=n_src, b=n_band)
        h = h1 + h2
        # 2nd scale
        h = apply_scale_shift(h, scale_a2)

        # shortcut
        x = x + h

        # norm
        h = self.norm_b(x)
        # 1st scale/shift
        h = apply_scale_shift(h, scale_b1, shift_b1)
        # FFN
        h = rearrange(h, 'n s b t d -> (n s) b t d')
        h = self.mlp(h)
        h = rearrange(h, '(n s) b t d -> n s b t d', n=bs, s=n_src)
        # 2nd scale
        h = apply_scale_shift(h, scale_b2)

        # shortcut
        x = x + h

        return x


def centering(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x - x.mean(dim=dim, keepdim=True)


class FLOSS(nn.Module):
    """
    FLOSS-like architecture for source separation.
    """

    def __init__(
        self,
        n_src: int,
        audio_channels: int,
        dim: int,
        n_blocks: int,
        sample_length: int,
        sample_rate: int,
        args_mels: tp.Dict[str, tp.Any],
        args_bsja: tp.Dict[str, tp.Any],
        args_tspa: tp.Dict[str, tp.Any],
        flash_attention: bool = True,
        gradient_checkpointing: bool = True
    ):
        super().__init__()

        self.n_src = n_src
        self.ch = audio_channels
        self.dim = dim
        self.sample_length = sample_length
        self.sample_rate = sample_rate
        self.gradient_checkpointing = gradient_checkpointing

        n_time = 1 + sample_length // args_mels['hop_length']
        args_input = {'n_mels': args_mels['n_mels'], 'n_src': n_src, 'n_time': n_time}

        # Modules

        # Mel-band split
        self.mel_band_split = MelBandSplit(**args_mels)

        # TBS blocks
        self.tbs_blocks = nn.ModuleList()
        for _ in range(n_blocks):
            bsja = BSJA(dim_in=dim, **args_bsja, args_input=args_input, flash_attention=flash_attention)
            tspa = TSPA(dim_in=dim, **args_tspa, args_input=args_input, flash_attention=flash_attention)
            self.tbs_blocks.append(nn.ModuleList([bsja, tspa]))

        self.final_norm = RMSGroupNorm(dim, num_groups=4)

    def forward(
        self,
        x_t: torch.Tensor,
        t: tp.Optional[torch.Tensor] = None
    ):
        """
        x_t: (bs, n_src, n_ch, sample_length)
        t: (bs)
        """
        assert x_t.shape[1:] == (self.n_src, self.ch, self.sample_length), \
            f"Input shape mismatch: expected ({self.n_src}, 1, {self.sample_length}), got {x_t.shape[1:]}"
        if exists(t):
            assert t.shape[0] == x_t.shape[0], "Batch size of x_t and t must match"
        else:
            t = torch.zeros(x_t.shape[0], dtype=torch.float, device=x_t.device)

        bs, n_src, ch, L = x_t.shape

        inputs = x_t

        # mel-band split
        inputs = rearrange(inputs, 'b s c t -> (b s) c t')
        # x: (bs, n_time, n_mels, emb_dim)
        # stft_repr: (bs, ch, n_freq, n_time, 2)
        x, stft_repr = self.mel_band_split(inputs)
        x = rearrange(x, '(b s) t f d -> b s f t d', b=bs, s=n_src)  # (bs, n_src, n_band, n_time, dim)
        stft_repr = rearrange(stft_repr, '(b s) c f t i -> b s c f t i', b=bs, s=n_src)

        # TBS blocks
        for idx_block, (bsja, tspa) in enumerate(self.tbs_blocks):
            if self.gradient_checkpointing and self.training:
                def tbs_block(x_, t_):
                    x_ = bsja(x_, t_)
                    x_ = tspa(x_, t_)
                    return x_
                x = checkpoint(tbs_block, x, t)
            else:
                x = bsja(x, t)
                x = tspa(x, t)

        # final norm
        x = self.final_norm(x)

        # band unsplit
        x = rearrange(x, 'b s f t d -> (b s) t f d')
        x = self.mel_band_split.melband_unsplit(x)  # complex, (b s) ch f t

        # masking
        mask = rearrange(x, '(b s) c f t -> b s c f t', b=bs, s=self.n_src)
        stft_repr = torch.view_as_complex(stft_repr)  # (bs, n_src, ch, n_freq, n_time)

        stft_repr = stft_repr * mask

        # iSTFT
        stft_repr = rearrange(stft_repr, 'b s c f t -> (b s c) f t')
        out = self.mel_band_split.istft(stft_repr, complex_input=True, original_length=L)
        out = rearrange(out, '(b s c) t -> b s c t', b=bs, s=n_src)

        return out
