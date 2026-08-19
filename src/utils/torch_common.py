"""
Copyright (C) 2024 Yukara Ikemiya
"""

import os
import random

import numpy as np
import torch


def exists(x: torch.Tensor):
    return x is not None


def default(v, d):
    return v if exists(v) else d


def get_world_size():
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 1
    else:
        return torch.distributed.get_world_size()


def get_rank():
    """Get rank of current process."""

    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return 0
    else:
        return torch.distributed.get_rank()


def print_once(*args):
    if get_rank() == 0:
        print(*args)


def set_seed(seed: int = 0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def count_parameters(model: torch.nn.Module, include_buffers: bool = False):
    n_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_buffers = sum(p.numel() for p in model.buffers()) if include_buffers else 0
    return n_trainable_params + n_buffers


def sort_dict(D: dict):
    s_keys = sorted(D.keys())
    return {k: D[k] for k in s_keys}


def checkpoint(function, *args, **kwargs):
    """ Gradient checkpointing """
    kwargs.setdefault("use_reentrant", False)
    return torch.utils.checkpoint.checkpoint(function, *args, **kwargs)
