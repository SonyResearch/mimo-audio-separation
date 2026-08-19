"""
Copyright (C) 2024 Yukara Ikemiya
"""

import typing as tp

import torch

from utils.torch_common import print_once, exists
from .datamodule import MUSDB18DataModule, MUSDB18FastDataModule, MUSDB18TestDataModule

AUDIO_MODULES = {
    "musdb18": MUSDB18DataModule,
    "musdb18fast": MUSDB18FastDataModule,
    "musdb18test": MUSDB18TestDataModule,
}


class AudioDataset(torch.utils.data.Dataset):
    """
    A audio dataset class which consists of a list of audio modules 
    """

    def __init__(
        self,
        audio_modules: tp.Dict[str, dict],
        sample_rate: int,
        sample_length: int,
        out_channels="mono",
        # explicitly specify dataset size (e.g. case of batch size > dataset size)
        dataset_size: int = None
    ):
        assert out_channels in ['mono', 'stereo']

        super().__init__()
        self.sample_rate = sample_rate
        self.sample_size = sample_length
        self.out_channels = out_channels

        print_once('[Dataset instantiation]')

        self.modules = []
        for name, cfg in audio_modules.items():
            assert name in AUDIO_MODULES, f"Audio module '{name}' is not supported."
            print_once(f'\t->-> Instantiating audio module: {name}')
            module = AUDIO_MODULES[name](sample_rate=sample_rate, sample_length=sample_length, out_channels=out_channels, **cfg)
            self.modules.append(module)

        self.module_lengths = [len(m) for m in self.modules]

        self.dataset_size = self.actual_len()
        if exists(dataset_size):
            self.dataset_size = max(dataset_size, self.dataset_size)

    def actual_len(self):
        return sum(self.module_lengths)

    def __len__(self):
        return self.dataset_size

    def __get_idx(self, idx):
        idx = idx % self.actual_len()  # wrap around if idx exceeds actual dataset size

        """Get module index and internal index from global index."""
        assert 0 <= idx < len(self)
        for idx_m, l in enumerate(self.module_lengths):
            if idx < l:
                return idx_m, idx
            idx -= l

    def __getitem__(self, idx):
        # select other sources
        idx_m, idx_local = self.__get_idx(idx)

        # get audio
        audio, info = self.modules[idx_m].get_data(idx_local)
        return audio, info
