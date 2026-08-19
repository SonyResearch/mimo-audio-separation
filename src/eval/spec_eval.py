"""
Copyright (C) 2025 Yukara Ikemiya

-------------
Evaluation metrics on spectral domain
"""
import typing as tp

import torch
import torch.nn.functional as F
from einops import rearrange


class SpecEval(torch.nn.Module):
    """
    Calculate spectral evaluation metrics (spectral L1 loss and log-magnitude loss)
    between reference and estimated audio signals.
    """

    def __init__(
        self,
        n_ffts: tp.List[int] = [512, 1024, 2048],
        hop_sizes: tp.List[int] = [256, 512, 1024],
        window: str = "hann_window",
        eps_amp: float = 1e-6
    ):
        super().__init__()
        self.n_ffts = n_ffts
        self.hop_sizes = hop_sizes
        self.eps_amp = eps_amp
        self.win_func = getattr(torch, window)

    @torch.no_grad()
    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        reference, estimate (bs, n_src, ch, L) : torch.Tensor
        """
        est = estimate
        ref = reference
        assert est.shape == ref.shape, f"Estimated and reference signals must have the same shape. Got {est.shape} and {ref.shape}."
        assert est.dim() == 4, f"Expected 4D tensors (bs, n_src, ch, L), got {est.dim()}D"

        bs, n_src, ch, L = est.shape
        device = est.device

        if mask is not None:
            assert mask.shape == (bs, n_src), f"Mask shape must be (bs, n_src). Got {mask.shape}."
        else:
            mask = torch.ones((bs, n_src), dtype=torch.bool, device=device)

        est = rearrange(est, 'bs n_src ch L -> (bs ch n_src) L')
        ref = rearrange(ref, 'bs n_src ch L -> (bs ch n_src) L')

        results = {'spec_l1': 0., 'spec_logmag': 0.}
        for n_fft, hop_size in zip(self.n_ffts, self.hop_sizes):
            win = self.win_func(n_fft, device=device)
            est_spec = torch.stft(est, n_fft=n_fft, hop_length=hop_size, window=win, return_complex=True)
            ref_spec = torch.stft(ref, n_fft=n_fft, hop_length=hop_size, window=win, return_complex=True)
            est_spec = rearrange(est_spec, '(bs ch n_src) F T -> (bs ch) n_src F T', bs=bs, ch=ch, n_src=n_src)
            ref_spec = rearrange(ref_spec, '(bs ch n_src) F T -> (bs ch) n_src F T', bs=bs, ch=ch, n_src=n_src)  # (bs*ch, n_src, F, T)

            # spec L1
            spec_l1 = F.l1_loss(est_spec, ref_spec, reduction='none').mean(dim=(-2, -1))  # (bs*ch, n_src)
            spec_l1 = rearrange(spec_l1, '(b c) s -> b s c', b=bs, c=ch).mean(dim=-1)  # (bs, n_src)

            # log-magnitude
            est_aspec = torch.view_as_real(est_spec).pow(2).sum(-1).sqrt()
            ref_aspec = torch.view_as_real(ref_spec).pow(2).sum(-1).sqrt()  # (bs*ch, n_src, F, T)

            ref_mask = ref_aspec > self.eps_amp
            est_aspec = est_aspec.clamp(min=self.eps_amp)
            ref_aspec = ref_aspec.clamp(min=self.eps_amp)

            logmag = F.l1_loss(est_aspec.log(), ref_aspec.log(), reduction='none')

            # mask out silent bins
            logmag = logmag.masked_fill(~ref_mask, float('nan'))

            logmag = logmag.nanmean(dim=(-2, -1))  # (bs*ch, n_src)
            logmag = rearrange(logmag, '(b c) s -> b s c', b=bs, c=ch).mean(dim=-1)  # (bs, n_src)

            results['spec_l1'] += spec_l1
            results['spec_logmag'] += logmag

        results = {k: v / len(self.n_ffts) for k, v in results.items()}

        # replace metrics for silent sources with None
        results = {k: v.masked_fill(~mask, float('nan')) for k, v in results.items()}

        return results
