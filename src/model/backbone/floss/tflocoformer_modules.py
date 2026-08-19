"""
Copyright (C) 2025 Yukara Ikemiya

Adapted from the following repo's code under Apache-2.0 licenses respectively.
https://github.com/merlresearch/tf-locoformer

-----------------------------------------------------
Modules from TF-Locoformer method.
"""

import typing as tp
from packaging import version
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from einops import rearrange
from rotary_embedding_torch import RotaryEmbedding

try:
    assert torch.cuda.is_available() and version.parse(torch.__version__) >= version.parse('2.0.0')
except AssertionError as e:
    raise e

from utils.torch_common import exists


def checkpoint(function, *args, **kwargs):
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        x = x.permute(*self.dims).contiguous()
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim_in: int,
        dim_attn: int,
        n_heads: int,
        use_rope: bool = True,
        conv_kernel: tp.Optional[tp.List[int]] = None,
        dropout=0.0,
        flash_attention: bool = True,
        input_shape: str = '1d'
    ):
        super().__init__()
        assert input_shape in ['1d', '2d'], f"Unsupported input shape: {input_shape}"

        self.n_heads = n_heads
        self.dropout = dropout
        self.input_shape = input_shape

        if exists(conv_kernel):
            # use conv instead of linear for qkv
            conv_module = nn.Conv2d if input_shape == '2d' else nn.Conv1d
            # Use Sequential to handle transpose/pre/post-processing and convolution
            if input_shape == '1d':
                self.qkv = nn.Sequential(
                    Permute(0, 2, 1),  # (b, seq, c) -> (b, c, seq)
                    conv_module(dim_in, dim_attn * 3, kernel_size=conv_kernel, stride=1, padding='same', bias=False),
                    Permute(0, 2, 1)   # (b, c, seq) -> (b, seq, c)
                )
            else:
                self.qkv = nn.Sequential(
                    Permute(0, 3, 1, 2),  # (b, h, w, c) -> (b, c, h, w)
                    conv_module(dim_in, dim_attn * 3, kernel_size=conv_kernel, stride=1, padding='same', bias=False),
                    Permute(0, 2, 3, 1)   # (b, c, h, w) -> (b, h, w, c)
                )
        else:
            self.qkv = nn.Linear(dim_in, dim_attn * 3, bias=False)

        self.rope = RotaryEmbedding(dim_attn // n_heads) if use_rope else None
        self.aggregate_heads = nn.Sequential(nn.Linear(dim_attn, dim_in, bias=False), nn.Dropout(dropout))

        self.flash_attention_config = [SDPBackend.FLASH_ATTENTION] if flash_attention else [SDPBackend.MATH, SDPBackend.EFFICIENT_ATTENTION]

    def forward(self, input: torch.Tensor):
        """
        input: (batch, seq_len, dim_in) or (batch, height, width, dim_in)
        """
        assert input.ndim == {'1d': 3, '2d': 4}[self.input_shape]

        # convert dimension size
        emb_qkv = self.qkv(input)

        if self.input_shape == '2d':
            emb_qkv = rearrange(emb_qkv, 'b h w d -> b (h w) d')

        # get query, key, and value
        query, key, value = self.get_qkv(emb_qkv)

        # rotary positional encoding
        if exists(self.rope):
            query, key = self.apply_rope(query, key)

        # pytorch 2.0 flash attention: q, k, v, mask, dropout, softmax_scale
        with sdpa_kernel(self.flash_attention_config):
            output = F.scaled_dot_product_attention(
                query, key, value,
                dropout_p=self.dropout if self.training else 0.0,
            )  # (batch, head, seq_len, -1)

        output = output.transpose(1, 2)  # (batch, seq_len, head, -1)
        output = output.reshape(output.shape[:2] + (-1,))
        output = self.aggregate_heads(output)

        if self.input_shape == '2d':
            output = rearrange(output, 'b (h w) d -> b h w d', h=input.shape[1], w=input.shape[2])

        return output

    def get_qkv(self, input):
        """
        input: (batch, seq_len, dim_head x n_heads x 3)
        """
        n_batch, seq_len = input.shape[:2]
        x = input.reshape(n_batch, seq_len, 3, self.n_heads, -1)
        x = x.movedim(-2, 1)  # (batch, head, seq_len, 3, -1)
        query, key, value = x[..., 0, :], x[..., 1, :], x[..., 2, :]
        return query, key, value

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def apply_rope(self, query, key):
        query = self.rope.rotate_queries_or_keys(query)
        key = self.rope.rotate_queries_or_keys(key)
        return query, key


class RMSGroupNorm(nn.Module):
    def __init__(
        self,
        dim,
        num_groups: int = 4,
        eps=1e-8,
        bias=False
    ):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.

        """
        super().__init__()

        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.eps = eps
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.ones(dim, dtype=torch.float32))

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.zeros(dim, dtype=torch.float32))

    @torch.amp.autocast(device_type="cuda", enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(
        self,
        dim,
        dim_inner,
        conv1d_kernel,
        conv1d_shift,
        dropout=0.0
    ):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        b, s1, s2, h = x.shape
        x = x.contiguous().view(b * s1, s2, h)
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner:, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks: self.diff_ks + s2, :]

        return self.dropout(x).view(b, s1, s2, h)
