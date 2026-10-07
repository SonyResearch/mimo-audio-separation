"""
Copyright (C) 2025 Yukara Ikemiya
"""

import os
import typing as tp

import torch
import torch.distributed as dist
import wandb
import gc

from utils.logging import MetricsLogger
from utils.torch_common import exists, sort_dict, print_once
from evaluate import separate, BSSEval, SpecEval


class Trainer:
    def __init__(
        self,
        model,              # model
        ema,                # exponential moving average
        optimizer,          # optimizer
        scheduler,          # scheduler
        optimizer_d,        # optimizer for discriminator
        scheduler_d,        # scheduler for discriminator
        train_dataloader,
        valid_dataset,
        accel,              # Accelerator object
        cfg,                # Configurations
        ckpt_dir=None
    ):
        self.model = accel.unwrap_model(model)
        self.ema = ema
        self.opt = optimizer
        self.sche = scheduler
        self.train_dataloader = train_dataloader
        self.valid_dataset = valid_dataset
        self.accel = accel
        self.cfg = cfg
        self.cfg_t = cfg.trainer
        self.EPS = 1e-8
        self.device = accel.device
        self.sr = cfg.data.train.sample_rate

        # discriminator
        self.have_disc = exists(optimizer_d) and exists(scheduler_d)
        self.opt_d = optimizer_d
        self.sche_d = scheduler_d

        self.logger = MetricsLogger()           # Logger for WandB
        self.logger_metrics = MetricsLogger()   # Logger for metrics
        self.logger_print = MetricsLogger()     # Logger for printing
        self.logger_test = MetricsLogger()      # Logger for test

        self.states = {'global_step': 0, 'best_metrics': float('inf'), 'latest_metrics': float('inf')}

        # Evaluation modules
        self.evals = {
            "bss_eval": BSSEval().to(self.device),
            "spec_eval": SpecEval().to(self.device)
        }

        # time measurement
        self.s_event = torch.cuda.Event(enable_timing=True)
        self.e_event = torch.cuda.Event(enable_timing=True)

        # resume training
        if ckpt_dir is not None:
            self.__load_ckpt(ckpt_dir)

    def start_training(self):
        """
        Start training with infinite loops
        """
        self.model.train()
        self.s_event.record()

        print_once("\n[ Started training ]\n")

        while True:
            for batch in self.train_dataloader:
                # Update
                metrics = self.run_step(batch)

                self.logger.add(metrics)
                self.logger_metrics.add(metrics)
                self.logger_print.add(metrics)

                # Log / Print -- must run on ALL ranks since metrics are reduced across ranks
                if self.__its_time(self.cfg_t.logging.n_step_log):
                    self.__log_metrics()

                if self.__its_time(self.cfg_t.logging.n_step_print):
                    self.__print_metrics()

                if self.accel.is_main_process:
                    # Save checkpoint
                    if self.__its_time(self.cfg_t.logging.n_step_ckpt):
                        self.__save_ckpt()

                    # Sample
                    if self.__its_time(self.cfg_t.logging.n_step_sample):
                        self.__sampling()

                    # Validation
                    if self.__its_time(self.cfg_t.logging.n_step_val) and exists(self.valid_dataset):
                        self.__validation()

                self.states['global_step'] += 1

    def run_step(self, batch, train: bool = True):
        """ One training step """

        # srouces: (bs, n_src, n_ch, sample_length)
        sources, _ = batch

        bs, n_src, n_ch, sample_length = sources.shape

        # Update

        if train:
            self.opt.zero_grad()
            if self.have_disc:
                self.opt_d.zero_grad()

        output = self.model.train_step(sources)

        if train:
            # Separator update
            params_g = self._optimizer_params(self.opt)
            self.accel.backward(output['G/loss'])
            self._all_reduce_grads(params_g)
            if self.accel.sync_gradients:
                self.accel.clip_grad_norm_(params_g, self.cfg_t.max_grad_norm)

            self.opt.step()
            self.sche.step()

            # Discriminator update
            if self.have_disc:
                params_d = self._optimizer_params(self.opt_d)
                # Drop the grads that G/loss (GAN / feature-matching) left on the discriminator.
                self.opt_d.zero_grad()
                self.accel.backward(output['D/loss'])
                self._all_reduce_grads(params_d)
                if self.accel.sync_gradients:
                    self.accel.clip_grad_norm_(params_d, self.cfg_t.max_grad_norm)

                self.opt_d.step()
                self.sche_d.step()

            # EMA
            if exists(self.ema):
                self.ema.update()

        return {k: v.detach() for k, v in output.items()}

    @staticmethod
    def _optimizer_params(opt) -> tp.List[torch.nn.Parameter]:
        return [p for group in opt.param_groups for p in group['params']]

    @torch.no_grad()
    def _all_reduce_grads(self, params: tp.List[torch.nn.Parameter], bucket_numel: int = 2 ** 25):
        """Average .grad over ranks; needed because train_step runs on the unwrapped model, bypassing DDP's reducer."""
        world_size = self.accel.num_processes
        if world_size == 1:
            return

        params = [p for p in params if p.requires_grad]
        # As in DDP, params unused on every rank keep grad=None; params unused only on some ranks get zeros.
        has_grad = torch.tensor([p.grad is not None for p in params], dtype=torch.uint8, device=self.device)
        dist.all_reduce(has_grad, op=dist.ReduceOp.MAX)
        grads = []
        for p, used in zip(params, has_grad.tolist()):
            if used:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                grads.append(p.grad)

        buckets_by_dtype: tp.Dict[torch.dtype, tp.List[tp.List[torch.Tensor]]] = {}
        for g in grads:
            buckets = buckets_by_dtype.setdefault(g.dtype, [[]])
            if buckets[-1] and sum(x.numel() for x in buckets[-1]) + g.numel() > bucket_numel:
                buckets.append([])
            buckets[-1].append(g)

        for buckets in buckets_by_dtype.values():
            for bucket in buckets:
                flat = torch._utils._flatten_dense_tensors(bucket)
                dist.all_reduce(flat)
                flat.div_(world_size)
                for g, synced in zip(bucket, torch._utils._unflatten_dense_tensors(flat, bucket)):
                    g.copy_(synced)

    @torch.no_grad()
    def _pop_reduced(self, logger: MetricsLogger) -> tp.Dict[str, torch.Tensor]:
        """Pop metrics averaged over logged steps and all ranks. Must be called on ALL ranks."""
        counts = dict(logger.counts)
        sums = logger.pop(mean=False)
        if self.accel.num_processes == 1:
            return {k: v / counts[k] for k, v in sums.items()}

        # Key sets can differ across ranks (e.g. randomly sampled number of iterations)
        all_keys = [None] * self.accel.num_processes
        dist.all_gather_object(all_keys, sorted(sums.keys()))
        keys = sorted(set().union(*all_keys))
        if len(keys) == 0:
            return {}

        zero = torch.zeros((), device=self.device)
        local_sums = torch.stack([sums[k].float().to(self.device) if k in sums else zero for k in keys])
        local_cnts = torch.tensor([float(counts.get(k, 0)) for k in keys], device=self.device)
        buf = torch.cat([local_sums, local_cnts])
        dist.all_reduce(buf)
        g_sums, g_cnts = buf.split(len(keys))

        return {k: g_sums[i] / g_cnts[i].clamp(min=1.0) for i, k in enumerate(keys)}

    @torch.no_grad()
    def __sampling(self):
        # sampling from EMA model
        target_device = self.accel.device
        target_dtype = next(self.model.parameters()).dtype
        if exists(self.ema):
            model = self.ema.ema_model
            model.to(device=target_device, dtype=target_dtype)
        else:
            model = self.model
        model.eval()

        steps: list = self.cfg_t.logging.steps
        n_sample: int = self.cfg_t.logging.n_samples_per_step
        n_src: int = self.model.num_sources

        # randomly select samples
        dataset = self.train_dataloader.dataset
        # dataset = self.valid_dataset if exists(self.valid_dataset) else self.train_dataloader.dataset
        sr = dataset.sample_rate
        idxs = torch.randperm(len(dataset))[:n_sample]
        audios = torch.stack([dataset[idx][0] for idx in idxs], dim=0).to(self.accel.device)  # (n_sample, n_src, n_ch, L)
        # mix = audios.sum(dim=1)  # (n_sample, n_ch, L)

        # sampling
        with torch.no_grad(), self.accel.autocast():
            gen_samples, info = model.test(sources=audios, iters=steps)  # List of (n_sample, n_src, ch, L), or (n_sample, n_src, ch, L)

        is_mimo = isinstance(gen_samples, list)
        if is_mimo:
            columns = [f"mix-{i} (audio)" for i in range(n_src)] + [f"gt-{i} (audio)" for i in range(n_src)]
        else:
            gen_samples = [gen_samples]
            steps = [1]
            columns = [f"mix (audio)"] + [f"gt-{i} (audio)" for i in range(n_src)]

        columns += [item for pair in [[f"sep-{i} (audio)"] for i in range(n_src)] for item in pair]
        table_audio = wandb.Table(columns=columns)

        mixture = info['mixture']
        losses = info['losses']
        losses = {f"test/{k}": v.detach() for k, v in losses.items()}

        for idx_iter, step in enumerate(steps):
            for idx in range(n_sample):
                if is_mimo:
                    data = [wandb.Audio(mixture[idx, i].cpu().numpy().T, sample_rate=sr) for i in range(n_src)]
                else:
                    data = [wandb.Audio(mixture[idx].cpu().numpy().T, sample_rate=sr)]
                # ground truth sources
                data += [wandb.Audio(audios[idx, i].cpu().numpy().T, sample_rate=sr) for i in range(n_src)]
                # separated sources
                data += [wandb.Audio(gen_samples[idx_iter][idx, i].cpu().numpy().T, sample_rate=sr) for i in range(n_src)]

                table_audio.add_data(*data)

        # log
        self.accel.log(losses, step=self.states['global_step'])

        self.accel.log({'Samples': table_audio}, step=self.states['global_step'])

        self.model.train()

        print("\t->->-> Sampled.")

        del audios, gen_samples, info
        gc.collect()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def __validation(self):
        # validation on EMA model
        target_device = self.accel.device
        target_dtype = next(self.model.parameters()).dtype
        if exists(self.ema):
            model = self.ema.ema_model
            model.to(device=target_device, dtype=target_dtype)
        else:
            model = self.model
        model.eval()

        dataset = self.valid_dataset
        n_src = model.num_sources
        sample_size = self.cfg.data.train.sample_length
        overlap = int(sample_size * 0.5)
        max_iter = self.cfg.model.get('max_iter', 1)
        iterations = [1] if max_iter == 1 else [1, max_iter]
        n_samples_val = min(self.cfg_t.logging.n_samples_val, len(dataset))

        self.s_event.record()

        print(f"[ Validation ({n_samples_val} samples) ]")

        for idx in range(n_samples_val):
            audio, meta = dataset[idx]
            audio = audio.to(self.accel.device)
            n_src, ch, L = audio.shape

            # Separation
            audios_sep = separate(
                model=model,
                audio=audio.sum(dim=0),  # mix
                sample_size=sample_size,
                overlap=overlap,
                batch_size=20,
                iterations=iterations,
                accel=self.accel
            )  # (n_iterations, n_src, ch, L)

            tgt = audio.unsqueeze(0)  # (1, n_src, ch, L)

            # Evaluation
            iterations = [1] if audios_sep.shape[0] == 1 else iterations
            all_results = [{} for _ in iterations]
            for it_idx, it in enumerate(iterations):
                audios_sep_it = audios_sep[it_idx].unsqueeze(0)  # (1, n_src, ch, L)

                eval_results_ = {}
                for eval in self.evals.values():
                    eval_results_.update(
                        eval(
                            estimate=audios_sep_it,
                            reference=tgt
                        )
                    )  # dict of (1, n_src)

                eval_results_ = {k: v.mean(dim=0) for k, v in eval_results_.items()}  # (n_src)

                for k, v in eval_results_.items():
                    all_results[it_idx][k] = all_results[it_idx].get(k, []) + [v]

        metrics = {}
        for it_idx, it in enumerate(iterations):
            for k, v in all_results[it_idx].items():
                v = torch.stack(v, dim=0).mean(dim=0)  # (n_src)
                for src_idx in range(n_src):
                    metrics[f"iter-{it}/{k}/src-{src_idx}"] = v[src_idx].cpu().item()

        metrics = {f"val/{k}": v for k, v in metrics.items()}
        self.accel.log(metrics, step=self.states['global_step'])

        self.e_event.record()
        torch.cuda.synchronize()
        val_time = self.s_event.elapsed_time(self.e_event) / 1000.  # [sec]

        print(f"\t->->-> Validation completed ({val_time:.1e} [sec]).")

        self.model.train()
        self.s_event.record()

        gc.collect()
        torch.cuda.empty_cache()

    def __save_ckpt(self):
        import shutil
        import json
        from omegaconf import OmegaConf

        out_dir = self.cfg_t.output_dir + '/ckpt'

        # save latest ckpt
        latest_dir = out_dir + '/latest'
        os.makedirs(latest_dir, exist_ok=True)

        # online model
        torch.save(self.opt.state_dict(), f"{latest_dir}/optimizer.pth")
        torch.save(self.sche.state_dict(), f"{latest_dir}/scheduler.pth")
        torch.save(self.model.backbone_model.state_dict(), f"{latest_dir}/backbone_model.pth")

        if exists(self.model.discriminator):
            torch.save(self.opt_d.state_dict(), f"{latest_dir}/optimizer_d.pth")
            torch.save(self.sche_d.state_dict(), f"{latest_dir}/scheduler_d.pth")
            torch.save(self.model.discriminator.state_dict(), f"{latest_dir}/discriminator.pth")

        # EMA
        if exists(self.ema):
            torch.save(self.ema.state_dict(), f"{latest_dir}/ema.pth")

        # save states and configuration
        OmegaConf.save(self.cfg, f"{latest_dir}/config.yaml")
        with open(f"{latest_dir}/states.json", mode="wt", encoding="utf-8") as f:
            json.dump(self.states, f, indent=2)

        # save best ckpt
        if self.states['latest_metrics'] == self.states['best_metrics']:
            shutil.copytree(latest_dir, out_dir + '/best', dirs_exist_ok=True)

        print("\t->->-> Saved checkpoints.")

    def __load_ckpt(self, dir: str):
        import json

        resume_type = "Finetuning" if self.cfg_t.finetuning else "Resuming"
        print_once(f"\n[{resume_type} training from the checkpoint directory] -> {dir}")

        self.model.backbone_model.load_state_dict(torch.load(f"{dir}/backbone_model.pth", weights_only=False))
        if not self.cfg_t.finetuning:
            self.opt.load_state_dict(torch.load(f"{dir}/optimizer.pth", weights_only=False))
            self.sche.load_state_dict(torch.load(f"{dir}/scheduler.pth", weights_only=False))

        if exists(self.model.discriminator):
            self.model.discriminator.load_state_dict(torch.load(f"{dir}/discriminator.pth", weights_only=False))
            if not self.cfg_t.finetuning:
                self.opt_d.load_state_dict(torch.load(f"{dir}/optimizer_d.pth", weights_only=False))
                self.sche_d.load_state_dict(torch.load(f"{dir}/scheduler_d.pth", weights_only=False))

        if not self.cfg_t.finetuning:
            with open(f"{dir}/states.json", mode="rt", encoding="utf-8") as f:
                self.states.update(json.load(f))
                # reset metrics
                self.states['best_metrics'] = float('inf')
                self.states['latest_metrics'] = float('inf')

        # EMA
        if exists(self.ema):
            if os.path.exists(f"{dir}/ema.pth"):
                self.ema.load_state_dict(torch.load(f"{dir}/ema.pth", weights_only=False))
            else:
                self.ema.model = self.model
                self.ema.init_ema()

    def __log_metrics(self, sort_by_key: bool = True):
        metrics = self._pop_reduced(self.logger)
        # learning rate
        metrics['G/lr'] = self.sche.get_last_lr()[0]
        if sort_by_key:
            metrics = sort_dict(metrics)

        self.accel.log(metrics, step=self.states['global_step'])

        # update states
        if self.logger_metrics.cnt >= self.cfg_t.logging.n_step_metrics:
            metrics = self._pop_reduced(self.logger_metrics)
            m_for_ckpt = self.cfg_t.logging.metrics_for_best_ckpt
            m_latest = float(sum([metrics[k].detach() for k in m_for_ckpt]))
            self.states['latest_metrics'] = m_latest
            if m_latest < self.states['best_metrics']:
                self.states['best_metrics'] = m_latest

    def __print_metrics(self, sort_by_key: bool = True):
        self.e_event.record()
        torch.cuda.synchronize()
        p_time = self.s_event.elapsed_time(self.e_event) / 1000.  # [sec]

        metrics = self._pop_reduced(self.logger_print)
        # tensor to scalar
        metrics = {k: v.item() for k, v in metrics.items()}
        if sort_by_key:
            metrics = sort_dict(metrics)

        step = self.states['global_step']
        s = f"Step {step} ({p_time:.1e} [sec]): " + ' / '.join([f"[{k}] - {v:.3e}" for k, v in metrics.items()])
        if self.accel.is_main_process:
            print(s)

        self.s_event.record()

    def __its_time(self, itv: int):
        return (self.states['global_step'] - 1) % itv == 0
