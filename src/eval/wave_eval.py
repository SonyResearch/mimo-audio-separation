"""
Copyright (C) 2025 Yukara Ikemiya

-------------
Evaluation metrics based on waveform domain.
"""

import torch


class WaveEval(torch.nn.Module):
    """
    Waveform domain metrics.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        reference, estimate (bs, n_src, ch, L) : torch.Tensor
        mask (bs, n_src) : torch.Tensor or None,
            Metrics are computed only for sources where mask is True.
        """
        est = estimate
        ref = reference
        assert est.shape == ref.shape, f"Estimated and reference signals must have the same shape. Got {est.shape} and {ref.shape}."
        assert est.dim() == 4, f"Expected 4D tensors (bs, n_src, ch, L), got {est.dim()}"

        bs, n_src, ch, L = est.shape
        device = est.device

        if mask is not None:
            assert mask.shape == (bs, n_src), f"Mask shape must be (bs, n_src). Got {mask.shape}."
        else:
            mask = torch.ones((bs, n_src), dtype=torch.bool, device=device)

        # RMSE
        res = est - ref
        rmse = torch.sqrt((res.pow(2).mean(dim=[2, 3])))  # (bs, n_src)

        # replace metrics for silent sources with None
        rmse = rmse.masked_fill(~mask, float('nan'))

        return {
            'rmse': rmse
        }


class SilentSegmentEval(torch.nn.Module):
    """
    Metrics for silent segments.
    """

    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        reference, estimate (bs, n_src, ch, L) : torch.Tensor
        mask (bs, n_src) : torch.Tensor or None.
        """
        est = estimate
        ref = reference
        assert est.shape == ref.shape, f"Estimated and reference signals must have the same shape. Got {est.shape} and {ref.shape}."
        assert est.dim() == 4, f"Expected 4D tensors (bs, n_src, ch, L), got {est.dim()}"

        bs, n_src, ch, L = est.shape
        device = est.device

        if mask is not None:
            assert mask.shape == (bs, n_src), f"Mask shape must be (bs, n_src). Got {mask.shape}."
        else:
            mask = torch.ones((bs, n_src), dtype=torch.bool, device=device)

        # RMSE
        res = est - ref
        rmse = torch.sqrt((res.pow(2).mean(dim=[2, 3])))  # (bs, n_src)

        # Silent frame energy (silent frame evaluation)
        energy_nonsilence = ref.pow(2).sum(dim=[2, 3])  # (bs, n_src)
        energy_nonsilence = energy_nonsilence.masked_fill(~mask, float('nan'))
        energy_nonsilence_median = torch.nanmedian(energy_nonsilence, dim=0)[0]  # (n_src,)
        energy_silence_pred = est.pow(2).sum(dim=[2, 3])  # (bs, n_src)
        energy_silence_pred = energy_silence_pred.masked_fill(mask, float('nan'))
        silence_ratio = (energy_silence_pred / energy_nonsilence_median[None, :]).sqrt()  # (bs, n_src)

        # replace metrics for non-silent chunks with None
        rmse = rmse.masked_fill(mask, float('nan'))

        return {
            'rmse[silence]': rmse,
            'silence_ratio[silence]': silence_ratio
        }
