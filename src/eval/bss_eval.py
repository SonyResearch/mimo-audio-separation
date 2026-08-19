"""
Copyright (C) 2025 Yukara Ikemiya

-------------
Evaluation metrics based on BSS Eval
"""

import torch
from einops import rearrange


class BSSEval(torch.nn.Module):
    """
    Calculate BSS Eval metrics (SDR, SI-SDR, SIR, SAR) between reference and estimated audio signals.
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

        # Batch-wise computation
        scales = (est * ref).sum(dim=3) / (ref**2).sum(dim=3)  # (bs, n_src, ch)
        ref_scaled = ref * scales[:, :, :, None]  # for SI-SDR

        res = est - ref
        res_s = est - ref_scaled

        # SDR and SI-SDR
        # NOTE: SI-SDR is computed as average over channels
        sdr = 10 * (ref.pow(2).sum(dim=[2, 3]) / res.pow(2).sum(dim=[2, 3])).log10()
        si_sdr = 10 * (ref_scaled.pow(2).sum(dim=3) / res_s.pow(2).sum(dim=3)).log10().mean(dim=2)  # (bs, n_src)

        est = rearrange(est, 'bs n_src ch L -> (bs ch) n_src L')
        ref = rearrange(ref, 'bs n_src ch L -> (bs ch) n_src L')
        res = rearrange(res, 'bs n_src ch L -> (bs ch) n_src L')

        # SIR and SAR
        sir = torch.zeros(bs * ch, n_src, device=device)
        sar = torch.zeros(bs * ch, n_src, device=device)

        # Batch computation for SIR and SAR
        Rss = torch.bmm(ref, ref.transpose(1, 2))  # (bs, n_src, n_src)

        # stable computation
        eps = 1e-12 * ref.shape[-1]
        Rss += eps * torch.eye(n_src, device=device, dtype=Rss.dtype)[None, :, :]

        for j in range(n_src):
            Rsr = torch.bmm(ref, res[:, j:j + 1].transpose(1, 2))  # (bs, n_src, 1)
            b_coeff = torch.linalg.solve(Rss, Rsr)  # (bs, n_src, 1)
            e_interf = torch.bmm(ref.transpose(1, 2), b_coeff).squeeze(-1)  # (bs, L)
            e_artif = res[:, j] - e_interf

            p_ref = ref[:, j].pow(2).sum(dim=1)  # (bs,)
            sir[:, j] = 10 * ((p_ref + eps) / (e_interf.pow(2).sum(dim=1) + eps)).log10()
            sar[:, j] = 10 * ((p_ref + eps) / (e_artif.pow(2).sum(dim=1) + eps)).log10()

        sir = rearrange(sir, '(b c) s -> b s c', b=bs, c=ch).mean(dim=-1)
        sar = rearrange(sar, '(b c) s -> b s c', b=bs, c=ch).mean(dim=-1)  # (bs, n_src)

        # replace metrics for silent sources with None
        sdr = sdr.masked_fill(~mask, float('nan'))
        si_sdr = si_sdr.masked_fill(~mask, float('nan'))
        sir = sir.masked_fill(~mask, float('nan'))
        sar = sar.masked_fill(~mask, float('nan'))

        return {
            'sdr': sdr,
            'si_sdr': si_sdr,
            'sir': sir,
            'sar': sar,
        }
