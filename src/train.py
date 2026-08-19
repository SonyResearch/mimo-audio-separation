"""
Copyright (C) 2026 Yukara Ikemiya
"""

import sys
sys.dont_write_bytecode = True

# ignore warnings
import warnings
warnings.filterwarnings("ignore")

# DDP
from accelerate import Accelerator, DistributedDataParallelKwargs, DataLoaderConfiguration
from accelerate.utils import ProjectConfiguration

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from ema_pytorch import EMA

from utils.torch_common import get_world_size, get_rank, count_parameters, set_seed, print_once
from trainer import Trainer


@hydra.main(version_base=None, config_path='../configs/', config_name="default.yaml")
def main(cfg: DictConfig):

    # Update config if ckpt_dir is specified (training resumption)

    is_resumption = (cfg.trainer.ckpt_dir is not None)
    if is_resumption:
        cfg_ckpt = OmegaConf.load(f'{cfg.trainer.ckpt_dir}/config.yaml')

        # Keep cfg.data and cfg.trainer
        overrides = {}
        if "trainer" in cfg.keys():
            trainer_conf = OmegaConf.to_container(cfg.trainer, resolve=True)
            overrides_trainer = [f'trainer.{k}={v}' for k, v in trainer_conf.items()]
            overrides_trainer = OmegaConf.from_dotlist(overrides_trainer)
            print_once(f"Keep trainer config at resumption: {overrides_trainer}")
            overrides.update(overrides_trainer)
        if "data" in cfg.keys():
            cfg_ckpt.data = cfg.data  # Keep data config as is (e.g. for dataset size)
            print_once(f"Keep data config at resumption: {cfg.data}")

        if cfg.trainer.finetuning:
            # Keep cfg.optimizer
            assert "optimizer" in cfg.keys(), "optimizer config must be specified for finetuning."
            optimizer_conf = OmegaConf.to_container(cfg.optimizer, resolve=True)
            overrides_optimizer = [f'optimizer.{k}={v}' for k, v in optimizer_conf.items()]
            overrides_optimizer = OmegaConf.from_dotlist(overrides_optimizer)
            print_once(f"Keep optimizer config at finetuning: {overrides_optimizer}")
            overrides.update(overrides_optimizer)

        override_conf = OmegaConf.create(overrides)

        # Load checkpoint configuration
        cfg = OmegaConf.merge(cfg_ckpt, override_conf)

    # HuggingFace Accelerate for distributed training

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    dl_config = DataLoaderConfiguration(split_batches=True)
    p_config = ProjectConfiguration(project_dir=cfg.trainer.output_dir)
    amp = cfg.trainer.amp if cfg.trainer.amp else "no"
    accel = Accelerator(
        mixed_precision=amp,
        dataloader_config=dl_config,
        project_config=p_config,
        kwargs_handlers=[ddp_kwargs],
        log_with='wandb'
    )

    accel.init_trackers(cfg.trainer.logger.project_name, config=OmegaConf.to_container(cfg),
                        init_kwargs={"wandb": {"name": cfg.trainer.logger.run_name, "dir": cfg.trainer.output_dir}})

    print(f"->->-> Rank: {get_rank()} / World size: {get_world_size()}")

    set_seed(cfg.trainer.seed)

    # Dataset

    batch_size = cfg.trainer.batch_size
    num_workers = cfg.trainer.num_workers
    # Train dataset and dataloader
    train_dataset = hydra.utils.instantiate(cfg.data.train)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers > 0))
    # Validation dataset
    val_dataset = hydra.utils.instantiate(cfg.data.val) if ('val' in cfg.data) else None

    # Model

    model = hydra.utils.instantiate(cfg.model)

    # EMA

    if is_resumption:
        # At resumption, update EMA from the beginning
        cfg.trainer.ema.update_after_step = 0

    ema = None
    if accel.is_main_process:
        ema = EMA(model, **cfg.trainer.ema)
        ema.to(accel.device)

    # Optimizer

    if 'optimizer_d' in cfg.optimizer:
        opt = hydra.utils.instantiate(cfg.optimizer.optimizer)(params=model.backbone_model.parameters())
        sche = hydra.utils.instantiate(cfg.optimizer.scheduler)(optimizer=opt)
        opt_d = hydra.utils.instantiate(cfg.optimizer.optimizer_d)(params=model.discriminator.parameters())
        sche_d = hydra.utils.instantiate(cfg.optimizer.scheduler_d)(optimizer=opt_d)
    else:
        opt = hydra.utils.instantiate(cfg.optimizer.optimizer)(params=model.parameters())
        sche = hydra.utils.instantiate(cfg.optimizer.scheduler)(optimizer=opt)
        opt_d = None
        sche_d = None

    # Log

    model.train()
    have_disc = (opt_d is not None and sche_d is not None)
    num_params = count_parameters(model.backbone_model) / 1e6
    if have_disc:
        num_params_d = count_parameters(model.discriminator) / 1e6

    if accel.is_main_process:
        print("=== Parameters ===")
        print(f"\tSeparator:\t{num_params:.2f} [million]")
        if have_disc:
            print(f"\tDiscriminator:\t{num_params_d:.2f} [million]")
        print("=== Dataset ===")
        print(f"\tBatch size: {cfg.trainer.batch_size}")
        print("\tTrain data:")
        print(f"\t\tChunks:  {len(train_dataset)}")
        print(f"\t\tBatches: {max(1, len(train_dataset)//cfg.trainer.batch_size)}")

    # Prepare for DDP

    train_dataloader, model, backbone_model, opt, sche, opt_d, sche_d = \
        accel.prepare(train_dataloader, model, model.backbone_model, opt, sche, opt_d, sche_d)

    # Start training

    trainer = Trainer(
        model,
        ema,
        opt, sche, opt_d, sche_d,
        train_dataloader,
        val_dataset,
        accel,
        cfg,
        ckpt_dir=cfg.trainer.ckpt_dir
    )

    trainer.start_training()


if __name__ == '__main__':
    main()
    print("[Training finished.]")
