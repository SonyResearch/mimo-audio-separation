"""
Copyright (C) 2025 Yukara Ikemiya
"""
import typing as tp

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from utils.torch_common import exists
from .utils.noise_shaping import EnergyEnvelopeNoiseShaping


def centering(x: torch.Tensor, dim: int) -> torch.Tensor:
    return x - x.mean(dim=dim, keepdim=True)


class MIMOBase(nn.Module):
    def __init__(
        self,
        num_channels: int,
        num_sources: int,
        backbone_model: nn.Module,
        max_iter: int,
        model_output_style: str = "diff",  # "direct" or "diff"
        mixture_consistency: bool = True,
        use_time_emb: bool = False,
        noise_addition: bool = False,
        # discriminator
        discriminator: tp.Optional[nn.Module] = None,
        # losses
        loss_modules: tp.Dict[str, nn.Module] = {},
        loss_lambdas: tp.Dict[str, float] = {},
        # training/inference config
        input_normalize: bool = True,
        training_config: tp.Dict[str, tp.Any] = {
            "iter_weights": None,       # ratios of iterations during training
            "iter_loss_weights": None,  # loss weights for each iteration during training
            "checkpoint_from_iter": 1
        },
        noise_config: tp.Dict[str, tp.Any] = {
            "window_size": 1025,        # for 44.1khz
            "scale": 1.0
        }
    ):
        super().__init__()
        assert model_output_style in ["direct", "diff"]
        self.num_channels = num_channels
        self.num_sources = num_sources
        self.max_iter = max_iter
        self.model_output_style = model_output_style
        self.mixture_consistency = mixture_consistency
        self.use_time_emb = use_time_emb
        self.noise_addition = noise_addition
        # modules
        self.backbone_model = backbone_model
        self.discriminator = discriminator
        # losses
        self.loss_modules = loss_modules
        self.loss_lambdas = loss_lambdas
        # training config
        self.input_normalize = input_normalize
        self.iter_weights = training_config["iter_weights"]
        self.iter_loss_weights = training_config["iter_loss_weights"] if exists(training_config["iter_loss_weights"]) else [1.0] * max_iter
        self.checkpoint_from_iter = training_config["checkpoint_from_iter"]
        assert len(self.iter_loss_weights) == max_iter

        if self.use_time_emb:
            self.register_buffer("t_list", torch.linspace(0, 1, steps=self.max_iter))  # (max_iter)

        if self.noise_addition:
            self.noise_shaping = EnergyEnvelopeNoiseShaping(**noise_config)

    def forward(
        self,
        x: torch.Tensor,
        t: tp.Optional[torch.Tensor] = None,
        **kwargs
    ) -> tp.List[torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input multi-source audio, (B, n_src, C, L)
            t (torch.Tensor): Status embedding, (B,) or None
        Returns:
            sep (torch.Tensor): (B, n_src, C, T)
        """
        # prediction
        pred = self.backbone_model(x, t, **kwargs)

        # mixture consistency
        if self.mixture_consistency:
            if self.model_output_style == "diff":
                # zero-sum projection
                pred = centering(pred, dim=1)  # (B, n_src, C, L)
                x = (x + pred).contiguous()
            elif self.model_output_style == "direct":
                # mixture consistency projection
                n_src = x.shape[1]
                mix_in = x.sum(1, keepdim=True)
                mix_out = pred.sum(1, keepdim=True)  # (B, 1, C, L)
                avg_diff = (mix_in - mix_out) / n_src
                x = (pred + avg_diff).contiguous()
        else:
            x = pred

        return x

    def train_step(
        self,
        sources: torch.Tensor
    ):
        """
        Args:
            sources (torch.Tensor): sources, (B, n_src, C, L)
        """
        self.train()
        output = {}

        bs, n_src, ch, L = sources.shape
        device = sources.device
        assert n_src == self.num_sources and ch == self.num_channels, \
            f"Input sources shape mismatch: expected n_src={self.num_sources}, ch={self.num_channels}, got n_src={n_src}, ch={ch}"

        # scale normalization
        if self.input_normalize:
            scales = self.__get_normalize_scale(sources.sum(1))  # (B,)
            sources = sources * scales.view(bs, 1, 1, 1)

        # mixture (averaged)
        mixture = sources.mean(dim=1, keepdim=True).repeat(1, n_src, 1, 1)  # (B, n_src, C, L)

        # noise addition for generative framework
        if self.noise_addition:
            noise_weight = self.noise_shaping(mixture[:, 0].contiguous().view(-1, L)).view(bs, ch, L).unsqueeze(1)  # (B, 1, C, L)
            noise = torch.randn_like(mixture) * noise_weight
            # centering (zero-sum)
            noise = centering(noise, dim=1)
            mixture = mixture + noise

        # debug
        err = (sources.sum(1) - mixture.sum(1)).abs().mean().item()
        assert err < 1e-8, f"Mixture is not sum of sources (err={err})"

        num_iter = 1 + torch.multinomial(torch.tensor(self.iter_weights, dtype=torch.float32, device=device), num_samples=1).item() \
            if exists(self.iter_weights) else self.max_iter

        # prediction
        x = mixture  # initial input
        t = None
        preds = []
        for iter in range(num_iter):
            if self.use_time_emb:
                t = self.t_list[iter].expand(bs)  # (B,)

            gradient_checkpointing = True if iter + 1 >= self.checkpoint_from_iter else False
            x = self(x, t, gradient_checkpointing=gradient_checkpointing)

            # NaN check
            # NOTE: If you encounter too much exception, you should consider using BF16 or FP32 training
            # try:
            assert torch.isfinite(x).all(), "Non-finite values detected in separation output"
            # except AssertionError as e:
            #     print(e)
            #     # tentative fix: skip this batch
            #     x = sources.clone().detach()

            # err = (sources.sum(1) - x.sum(1)).abs().mean().item()
            # assert err < 1e-5, (
            #     f"Mixture is not sum of sources [Iter: {iter+1}] "
            #     f"(err={err}, x_sum={x.sum().item()}, src_sum={sources.sum().item()})"
            # )

            preds.append(x)

            # To avoid gradient loop
            x = x.detach()

        # losses
        losses = {}
        intermidiates = {}
        for i, pred in enumerate(preds):
            losses_i = {}
            for name, module in self.loss_modules.items():
                losses_i.update(module(pred, sources))

            if exists(self.discriminator):
                # TODO: Fix this to reduce duplicated computation?
                losses_i.update(self.discriminator.compute_G_loss(pred, sources))

            # loss weight of this iteration
            loss_weight_i = self.iter_loss_weights[i] / sum(self.iter_loss_weights[:num_iter])
            for k, v in losses_i.items():
                losses[k] = losses.get(k, 0.) + v * loss_weight_i

            # logs
            losses_i = {f"I/{k}/iter-{i+1}": v.detach() for k, v in losses_i.items()}
            intermidiates.update(losses_i)

        # weighted sum of losses
        loss = 0.
        for k in self.loss_lambdas.keys():
            loss += losses[k] * self.loss_lambdas[k]

        # Discriminator loss
        if exists(self.discriminator):
            losses_real = self.discriminator.compute_D_loss(sources, mode='real')
            losses_real = {f"{k}_real": v for k, v in losses_real.items()}
            loss_real = losses_real.pop('loss_real')

            losses_fake = {}
            for i, pred in enumerate(preds):
                losses_fake_i = self.discriminator.compute_D_loss(pred.detach().clone(), mode='fake')
                for k, v in losses_fake_i.items():
                    losses_fake[f"{k}_fake"] = losses_fake.get(f"{k}_fake", 0.) + v / len(preds)

            loss_fake = losses_fake.pop('loss_fake')

            output.update({
                'D/loss': loss_real + loss_fake,
                'D/loss_real': loss_real.detach(),
                'D/loss_fake': loss_fake.detach()
            })
            output.update({f"D/{k}": v.detach() for k, v in losses_real.items()})
            output.update({f"D/{k}": v.detach() for k, v in losses_fake.items()})

        output['G/loss'] = loss
        output.update({f"G/{k}": v.detach() for k, v in losses.items()})
        output.update(intermidiates)

        return output

    @torch.no_grad()
    def test(
        self,
        sources: torch.Tensor,
        iters: tp.List[int] = [1, 3]
    ):
        """
        Args:
            sources (torch.Tensor): sources, (B, n_src, C, L)
        """
        self.eval()

        bs, n_src, ch, L = sources.shape
        assert n_src == self.num_sources and ch == self.num_channels, \
            f"Input sources shape mismatch: expected n_src={self.num_sources}, ch={self.num_channels}, got n_src={n_src}, ch={ch}"
        max_iter = max(iters)
        assert max_iter <= self.max_iter, f"Requested max_iter {max_iter} exceeds model's max_iter {self.max_iter}"

        # scale normalization
        if self.input_normalize:
            scales = self.__get_normalize_scale(sources.sum(1))  # (B,)
            sources = sources * scales.view(bs, 1, 1, 1)

        mixture = sources.mean(dim=1, keepdim=True).repeat(1, n_src, 1, 1)

        # noise addition for generative framework
        if self.noise_addition:
            noise_weight = self.noise_shaping(mixture[:, 0].contiguous().view(-1, L)).view(bs, ch, L).unsqueeze(1)  # (B, 1, C, L)
            noise = torch.randn_like(mixture) * noise_weight
            # centering (zero-sum)
            noise = centering(noise, dim=1)
            mixture = mixture + noise

            # debug
            err = (sources.sum(1) - mixture.sum(1)).abs().mean().item()
            assert err < 1e-8, f"Mixture is not sum of sources (err={err})"

        # prediction
        x = mixture  # initial input
        t = None
        preds = []
        for iter in range(max_iter):
            if self.use_time_emb:
                t = self.t_list[iter].expand(bs)  # (B,)

            x = self(x, t)

            if (iter + 1) in iters:
                preds.append(x)

        # loss evaluation
        losses = {}
        for i, pred in zip(iters, preds):
            losses_i = {}
            for name, module in self.loss_modules.items():
                losses_i.update(module(pred, sources))

            if exists(self.discriminator):
                losses_i.update(self.discriminator.compute_G_loss(pred, sources))

            # loss weight
            for k in self.loss_lambdas.keys():
                losses_i[k] = losses_i[k] * self.loss_lambdas[k]

            losses_i = {f"iter-{i}_{k}": v.detach() for k, v in losses_i.items()}
            losses.update(losses_i)

        # rescale to original scale
        if self.input_normalize:
            preds = [p / scales.view(bs, 1, 1, 1) for p in preds]
            mixture = mixture / scales.view(bs, 1, 1, 1)

        info = {'mixture': mixture, 'losses': losses}

        return preds, info

    @torch.no_grad()
    def inference(
        self,
        mixture: torch.Tensor,
        iters: tp.List[int] = [1, 3],
        return_init: bool = False
    ):
        """
        Args:
            mixture (torch.Tensor): mixture, (B, C, L)
        Returns:
            preds (List[torch.Tensor]): List of separated sources at each iteration, each (B, n_src, C, L)
        """
        self.eval()

        bs, ch, L = mixture.shape
        n_src = self.num_sources
        max_iter = max(iters)
        # assert max_iter <= self.max_iter, f"Requested max_iter {max_iter} exceeds model's max_iter {self.max_iter}"

        # scale normalization
        if self.input_normalize:
            scales = self.__get_normalize_scale(mixture)  # (B,)
            mixture = mixture * scales.view(bs, 1, 1)

        input = (mixture / n_src).unsqueeze(1).repeat(1, n_src, 1, 1)  # (B, n_src, C, L)

        # noise addition for generative framework
        if self.noise_addition:
            noise_weight = self.noise_shaping(input[:, 0].contiguous().view(-1, L)).view(bs, ch, L).unsqueeze(1)  # (B, 1, C, L)
            noise = torch.randn_like(input) * noise_weight
            # centering (zero-sum)
            noise = centering(noise, dim=1)
            input = input + noise

        # prediction
        x = input  # initial input
        t = None
        preds = []

        if return_init:
            preds.append(x)

        for iter in range(max_iter):
            # NOTE: # The last iteration's model is used for iterations beyond max_iter
            if iter >= self.max_iter:
                # print_once(f"[Warning]: iter {iter+1} exceeds model's max_iter {self.max_iter}, using the last iteration's model for inference.")
                iter = self.max_iter - 1

            if self.use_time_emb:
                t = self.t_list[iter].expand(bs)  # (B,)

            x = self(x, t)

            if (iter + 1) in iters:
                preds.append(x)

        # rescale to original scale
        if self.input_normalize:
            preds = [p / scales.view(bs, 1, 1, 1) for p in preds]

        return preds

    def __get_normalize_scale(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize audio to have max amplitude of 1.
        x: (B, C, L)
        """
        max_amps = x.abs().amax(dim=(-1, -2))  # (B,)

        # silence: no normalization, voiced: 1 / (max_amp + 1e-8)
        max_amps = torch.where(max_amps < 1e-8, torch.full((len(max_amps),), 1.0, device=x.device),
                               max_amps + 1e-8)

        scales = 1 / max_amps  # (B,)
        return scales
