"""
Copyright (C) 2025 Yukara Ikemiya
"""
from abc import ABC, abstractmethod

from .modification import Stereo, Mono


class AudioDataModule(ABC):
    def __init__(self, sample_rate: int, sample_length: int, out_channels: str):
        super().__init__()
        assert out_channels in ['mono', 'stereo']
        self.sample_rate = sample_rate
        self.sample_length = sample_length
        self.out_channels = out_channels
        self.ch_encoding = Mono() if out_channels == 'mono' else Stereo()

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def get_data(self, idx: int):
        pass

    def encode_channels(self, x):
        return self.ch_encoding(x)
