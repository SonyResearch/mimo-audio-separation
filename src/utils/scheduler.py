"""
Copyright (C) 2025 Yukara Ikemiya

------
Learning rate schedulers.
"""
import math

import torch


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=0.0, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1

        if step < self.warmup_steps:
            # warmup
            return [base_lr * step / self.warmup_steps for base_lr in self.base_lrs]
        else:
            # cosine decay (base_lr -> min_lr)
            progress = min(1, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps))

            return [
                self.min_lr + (base_lr - self.min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))
                for base_lr in self.base_lrs
            ]
