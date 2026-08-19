"""
Copyright (C) 2025 Yukara Ikemiya
"""
import os
import typing as tp
import random

import torch

from .base import AudioDataModule
from .audio_io import load_audio_with_pad, get_audio_metadata
from .modification import PhaseFlipper, VolumeChanger, ChannelShuffler, SilenceMultiSrc
from utils.files import get_info_from_csv
from utils.torch_common import print_once, exists


class MUSDB18DataModule(AudioDataModule):

    def __init__(
        self,
        root_dirs: tp.List[str],
        sample_rate: int,
        sample_length: int,
        metadata_name: str = "metadata",
        out_channels: str = "mono",
        inst_list: tp.List[str] = ['vocals', 'drums', 'bass', 'other'],
        gather_list: tp.Optional[tp.List[tp.List[int]]] = None,  # [[0], [1, 2, 3]]
        ext: str = 'wav',
        random_shift: bool = True,
        aug_mix: bool = True,
        aug_volume: bool = True,
        aug_flip: bool = True,
        aug_ch_shuffle: bool = True,
        aug_silence: bool = True,
        rate_silence: float = 0.1,  # valid only when aug_silence is True
        ensure_non_silent: bool = True,  # valid only when aug_mix is True
    ):
        assert out_channels in ['mono', 'stereo']
        super().__init__(sample_rate, sample_length, out_channels)

        self.root_dirs = root_dirs
        self.inst_list = inst_list
        self.gather_list = gather_list
        self.ext = ext
        self.random_shift = random_shift
        self.aug_mix = aug_mix
        self.aug_volume = aug_volume
        self.aug_flip = aug_flip
        self.aug_ch_shuffle = aug_ch_shuffle
        self.aug_silence = aug_silence
        self.ensure_non_silent = ensure_non_silent
        self.silencer = SilenceMultiSrc(p=rate_silence) if aug_silence else None

        if exists(gather_list):
            # check if all indices are valid
            assert all(0 <= i < len(inst_list) for g in gather_list for i in g)
            self.num_stems = len(gather_list)
        else:
            self.num_stems = len(inst_list)

        # preprocessing and augmentations
        self.preprocess = torch.nn.Sequential(
            self.ch_encoding,
            # Augmentation (BSRoFormer setting)
            PhaseFlipper(p=0.5) if aug_flip else torch.nn.Identity(),
            ChannelShuffler(p=0.5) if aug_ch_shuffle else torch.nn.Identity(),
            VolumeChanger(-3., 3.) if aug_volume else torch.nn.Identity()
        )

        # find all sub-directories in root_dirs
        self.all_dirs = []
        self.all_metas = []
        for root_dir in self.root_dirs:
            # check metadata csv
            metadata_csv_path = f"{root_dir}/{metadata_name}.csv"
            if os.path.exists(metadata_csv_path):
                dirs, metas = get_info_from_csv(metadata_csv_path, main_tag='path',
                                                other_info_tags=['sample_rate', 'num_frames', 'num_channels'])
                self.all_dirs.extend([os.path.join(root_dir, d) for d in dirs])
                self.all_metas.extend(metas)
            else:
                for subdir in os.listdir(root_dir):
                    full_path = os.path.join(root_dir, subdir)
                    if os.path.isdir(full_path):
                        # check if all inst_list are in the directory
                        if all(os.path.exists(os.path.join(full_path, inst + f'.{self.ext}')) for inst in self.inst_list):
                            self.all_dirs.append(full_path)

                            # get metadata from one of the files
                            metadata = get_audio_metadata(os.path.join(full_path, inst_list[0] + f'.{self.ext}'))
                            self.all_metas.append(metadata)

        assert len(self.all_dirs) > 0, f'No valid data directories found in {self.root_dirs}'
        assert len(self.all_dirs) == len(self.all_metas), 'Mismatch between directories and metadata lengths'

        print_once(f'\t[Root directories] : {self.root_dirs}')
        print_once(f'\tFound {len(self.all_dirs)} valid directories.')

    @property
    def num_sources(self):
        return len(self.inst_list)

    def __len__(self):
        return len(self.all_dirs)

    def get_offset(self, meta):
        max_ofs = max(0, meta['num_frames'] - self.sample_length)
        offset = random.randint(0, max_ofs) if (self.random_shift and max_ofs) else 0
        return offset

    def get_audio(self, idx: int, inst: str):
        meta = self.all_metas[idx]
        offset = self.get_offset(meta)
        filepath = os.path.join(self.all_dirs[idx], inst + f'.{self.ext}')
        audio = load_audio_with_pad(filepath, meta, self.sample_rate, self.sample_length, offset)
        return audio, filepath, offset

    def get_data(self, idx: int):
        audios = []
        if self.aug_mix:
            # select random directories for each instrument (including 'idx' as one of them)
            selected_indices = random.choices(range(len(self.all_dirs)), k=len(self.inst_list))
            selected_indices[random.randint(0, len(self.inst_list) - 1)] = idx  # ensure idx is included
        else:
            selected_indices = [idx] * len(self.inst_list)

        filepaths = []
        offsets = []
        silence_flags = []
        silence_flag = False
        for inst, sel_idx in zip(self.inst_list, selected_indices):
            silence_flag = True
            while (self.ensure_non_silent and silence_flag) and self.aug_mix:
                audio, filepath, offset = self.get_audio(sel_idx, inst)  # (ch, L)
                silence_flag = (audio.abs().mean() < 1e-5).item()
                # another index
                sel_idx = random.randint(0, len(self) - 1)

            # audio, filepath, offset = self.get_audio(sel_idx, inst)  # (ch, L)

            audio = self.preprocess(audio)

            audios.append(audio)
            filepaths.append(filepath)
            offsets.append(offset)
            silence_flags.append(silence_flag)

        if all(silence_flags):
            # if all sources are silent, re-try loading data
            new_idx = idx if self.aug_mix else random.randint(0, len(self) - 1)
            return self.get_data(new_idx)

        audios = torch.stack(audios, dim=0)  # (num_src, ch, sample_size)

        if exists(self.gather_list):
            gathered_audios = []
            for gather_indices in self.gather_list:
                gathered_audio = audios[gather_indices, :, :]  # (n_src_, ch, sample_size)
                if self.aug_silence:
                    # augment instrument combination by silencing some sources before gathering
                    gathered_audio = self.silencer(gathered_audio)

                gathered_audios.append(gathered_audio.sum(dim=0))  # (ch, sample_size)
            audios = torch.stack(gathered_audios, dim=0)  # (num_stems, ch, sample_size)

        if self.aug_silence:
            # randomly make some sources silent
            audios = self.silencer(audios)

        info = {'filepaths': filepaths, 'offsets': offsets}
        return audios, info


class MUSDB18FastDataModule(AudioDataModule):
    """
    Specific datamodule for faster data-loading of MUSDB18
    by pre-removing silence segments in each source audio.

    - Only supports random mixing of different tracks.
    """

    def __init__(
        self,
        root_dirs: tp.List[str],
        sample_rate: int,
        sample_length: int,
        metadata_name: str = "metadata",
        out_channels: str = "mono",
        inst_list: tp.List[str] = ['vocals', 'drums', 'bass', 'other'],
        gather_list: tp.Optional[tp.List[tp.List[int]]] = None,  # [[0], [1, 2, 3]]
        ext: str = 'wav',
        random_shift: bool = True,
        aug_mix: bool = True,
        aug_volume: bool = True,
        aug_flip: bool = True,
        aug_ch_shuffle: bool = True,
        aug_silence: bool = True
    ):
        assert out_channels in ['mono', 'stereo']
        super().__init__(sample_rate, sample_length, out_channels)

        self.root_dirs = root_dirs
        self.inst_list = inst_list
        self.gather_list = gather_list
        self.ext = ext
        self.random_shift = random_shift
        self.aug_mix = aug_mix
        self.aug_volume = aug_volume
        self.aug_flip = aug_flip
        self.aug_ch_shuffle = aug_ch_shuffle
        self.aug_silence = aug_silence
        self.silencer = SilenceMultiSrc(p=0.1) if aug_silence else None

        if exists(gather_list):
            # check if all indices are valid
            assert all(0 <= i < len(inst_list) for g in gather_list for i in g)
            self.num_stems = len(gather_list)
        else:
            self.num_stems = len(inst_list)

        # preprocessing and augmentations
        self.preprocess = torch.nn.Sequential(
            self.ch_encoding,
            # Augmentation (BSRoFormer setting)
            PhaseFlipper(p=0.5) if aug_flip else torch.nn.Identity(),
            ChannelShuffler(p=0.5) if aug_ch_shuffle else torch.nn.Identity(),
            VolumeChanger(-3., 3.) if aug_volume else torch.nn.Identity()
        )

        # find all sub-directories in root_dirs
        self.all_dirs = []
        self.all_metas = []
        for root_dir in self.root_dirs:
            # check metadata csv
            metadata_csv_path = f"{root_dir}/{metadata_name}.csv"
            dirs, metas = get_info_from_csv(metadata_csv_path, main_tag='path',
                                            other_info_tags=['sample_rate', 'num_frames', 'num_channels'])
            self.all_dirs.extend([os.path.join(root_dir, d) for d in dirs])
            self.all_metas.extend(metas)

        assert len(self.all_dirs) > 0, f'No valid data directories found in {self.root_dirs}'
        assert len(self.all_dirs) == len(self.all_metas), 'Mismatch between directories and metadata lengths'

        print_once(f'\t[Root directories] : {self.root_dirs}')
        print_once(f'\tFound {len(self.all_dirs)} valid directories.')

        # Load all audios into CPU memory
        print_once('\t ->-> Pre-loading all audio files into memory...')
        self.loaded_audios = []
        for idx in range(len(self.all_dirs)):
            if idx % 10 == 0:
                print_once(f'\t\t- Loading data {idx + 1}/{len(self.all_dirs)}')
            audios = {}
            for idx_inst, inst in enumerate(self.inst_list):
                p = os.path.join(self.all_dirs[idx], f'{inst}.{self.ext}')
                audio, info = load_audio_with_pad(p, self.all_metas[idx], self.sample_rate,
                                                  n_samples=None, offset=0, return_info=True)
                audios[inst] = audio
                if idx_inst == 0:
                    # update metadata with actual loaded info
                    self.all_metas[idx]["path"] = p
                    self.all_metas[idx]["sample_rate"] = sample_rate
                    self.all_metas[idx]["num_frames"] = info["num_frames"]

            self.loaded_audios.append(audios)

    @property
    def num_sources(self):
        return len(self.inst_list)

    def __len__(self):
        return len(self.all_dirs)

    def get_offset(self, meta):
        max_ofs = max(0, meta['num_frames'] - self.sample_length)
        offset = random.randint(0, max_ofs) if (self.random_shift and max_ofs) else 0
        return offset

    def get_audio(self, idx: int, inst: str):
        meta = self.all_metas[idx]
        offset = self.get_offset(meta)
        filepath = meta['path']
        audio = self.loaded_audios[idx][inst][:, offset:offset + self.sample_length].clone()
        return audio, filepath, offset

    def get_data(self, idx: int):
        if self.aug_mix:
            # select random directories for each instrument (including 'idx' as one of them)
            selected_indices = random.choices(range(len(self.all_dirs)), k=len(self.inst_list))
            selected_indices[random.randint(0, len(self.inst_list) - 1)] = idx  # ensure idx is included
        else:
            selected_indices = [idx] * len(self.inst_list)

        audios = []
        filepaths = []
        offsets = []
        for inst, sel_idx in zip(self.inst_list, selected_indices):
            audio, filepath, offset = self.get_audio(sel_idx, inst)  # (ch, L)
            audio = self.preprocess(audio)

            audios.append(audio)
            filepaths.append(filepath)
            offsets.append(offset)

        audios = torch.stack(audios, dim=0)  # (num_src, ch, sample_size)

        if exists(self.gather_list):
            gathered_audios = []
            for gather_indices in self.gather_list:
                gathered_audio = audios[gather_indices, :, :]  # (n_src_, ch, sample_size)
                if self.aug_silence:
                    # augment instrument combination by silencing some sources before gathering
                    gathered_audio = self.silencer(gathered_audio)

                gathered_audios.append(gathered_audio.sum(dim=0))  # (ch, sample_size)
            audios = torch.stack(gathered_audios, dim=0)  # (num_stems, ch, sample_size)

        if self.aug_silence:
            # randomly make some sources silent
            audios = self.silencer(audios)

        info = {'filepaths': filepaths, 'offsets': offsets}
        return audios, info


class MUSDB18TestDataModule(AudioDataModule):
    """
    DataModule that returns a full track for test/evaluation.
    """

    def __init__(
        self,
        root_dirs: tp.List[str],
        metadata_name: str,
        sample_rate: int,
        sample_length=None,
        out_channels: str = "mono",
        inst_list: tp.List[str] = ['vocals', 'drums', 'bass', 'other'],
        ext: str = 'wav',
    ):
        assert out_channels in ['mono', 'stereo']
        super().__init__(sample_rate, sample_length, out_channels)

        self.root_dirs = root_dirs
        self.inst_list = inst_list
        self.ext = ext

        # find all sub-directories in root_dirs
        self.all_dirs = []
        self.all_metas = []
        for root_dir in self.root_dirs:
            # check metadata csv
            metadata_csv_path = f"{root_dir}/{metadata_name}.csv"
            if os.path.exists(metadata_csv_path):
                dirs, metas = get_info_from_csv(metadata_csv_path, main_tag='file_dir',
                                                other_info_tags=['sample_rate', 'num_frames', 'num_channels'])
                self.all_dirs.extend([os.path.join(root_dir, d) for d in dirs])
                self.all_metas.extend(metas)
            else:
                for subdir in os.listdir(root_dir):
                    full_path = os.path.join(root_dir, subdir)
                    if os.path.isdir(full_path):
                        # check if all inst_list are in the directory
                        if all(os.path.exists(os.path.join(full_path, inst + f'.{self.ext}')) for inst in self.inst_list):
                            self.all_dirs.append(full_path)

                            # get metadata from one of the files
                            metadata = get_audio_metadata(os.path.join(full_path, inst_list[0] + f'.{self.ext}'))
                            self.all_metas.append(metadata)

        assert len(self.all_dirs) > 0, f'No valid data directories found in {self.root_dirs}'
        assert len(self.all_dirs) == len(self.all_metas), 'Mismatch between directories and metadata lengths'

        print_once(f'\t[Root directories] : {self.root_dirs}')
        print_once(f'\tFound {len(self.all_dirs)} valid directories.')

    @property
    def num_sources(self):
        return len(self.inst_list)

    def __len__(self):
        return len(self.all_dirs)

    def get_audio(self, idx: int, inst: str):
        meta = self.all_metas[idx]
        filepath = os.path.join(self.all_dirs[idx], inst + f'.{self.ext}')
        audio = load_audio_with_pad(filepath, meta, self.sample_rate, meta['num_frames'], 0)
        return audio, filepath

    def get_data(self, idx: int):
        audios = []

        filepaths = []
        for inst in self.inst_list:
            audio, filepath = self.get_audio(idx, inst)  # (ch, L)

            audios.append(audio)
            filepaths.append(filepath)

        audios = torch.stack(audios, dim=0)  # (num_src, ch, num_frames)

        if self.out_channels == 'mono':
            audios = torch.mean(audios, dim=1, keepdim=True)  # (num_src, 1, num_frames)

        info = {'filepaths': filepaths}

        return audios, info
