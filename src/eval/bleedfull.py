"""
Copyright (C) 2025 Yukara Ikemiya

-------------
Evaluation metrics based on bleed and fullness using mel spectrograms.
https://arxiv.org/abs/2305.07489
"""

import torch
from torchaudio.transforms import AmplitudeToDB
import librosa
from einops import rearrange


class BleedFull(torch.nn.Module):
    """
    Calculate the 'bleed' and 'fullness' metrics between a reference and an estimated audio signal.

    The 'bleed' metric measures how much the estimated signal bleeds into the reference signal,
    while the 'fullness' metric measures how much the estimated signal retains its distinctiveness
    in relation to the reference signal, both using mel spectrograms and decibel scaling.
    """

    def __init__(
        self,
        sr: int = 44100,
        n_fft: int = 4096,
        hop_length: int = 1024,
        n_mels: int = 512
    ):
        super().__init__()
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Register mel filter bank as buffer
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        self.register_buffer('mel_filter_bank', torch.from_numpy(mel_basis))  # (n_mels, F)

        self.amplitude_to_db = AmplitudeToDB(stype="magnitude", top_db=80)

    @torch.no_grad()
    def forward(
        self,
        estimate: torch.Tensor,
        reference: torch.Tensor,
        mask: torch.Tensor = None
    ):
        """
        Parameters:
        ----------
        reference, estimate (bs, n_src, ch, L) : torch.Tensor

        Returns:
        -------
        tuple
            A tuple containing two values:
            - `bleedless` (torch.Tensor): (bs,),  A score indicating how much 'bleeding' the estimated signal has (higher is better).
            - `fullness` (torch.Tensor): (bs,), A score indicating how 'full' the estimated signal is (higher is better).
        """
        assert reference.shape == estimate.shape, f"Reference and estimate must have the same shape. Got {reference.shape} and {estimate.shape}."
        device = reference.device
        window = torch.hann_window(self.n_fft).to(device)

        bs, n_src, ch, L = reference.shape
        reference = rearrange(reference, 'bs n_src ch L -> (bs n_src ch) L')
        estimate = rearrange(estimate, 'bs n_src ch L -> (bs n_src ch) L')

        # Compute STFTs with the Hann window
        D1 = torch.abs(torch.stft(reference, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                                  return_complex=True, pad_mode="constant"))
        D2 = torch.abs(torch.stft(estimate, n_fft=self.n_fft, hop_length=self.hop_length, window=window,
                                  return_complex=True, pad_mode="constant"))  # (B, F, T)

        S1_mel = torch.matmul(self.mel_filter_bank, D1)
        S2_mel = torch.matmul(self.mel_filter_bank, D2)  # (B, n_mels, T)

        S1_db = self.amplitude_to_db(S1_mel)
        S2_db = self.amplitude_to_db(S2_mel)

        diff = S2_db - S1_db

        # Reshape to separate batch and channel dimensions for batch processing
        diff = rearrange(diff, '(bs n_src ch) n_mels T -> (bs n_src) ch n_mels T', ch=ch, n_src=n_src)

        # Calculate metrics per batch item
        bleedless_list = []
        fullness_list = []

        for i in range(diff.shape[0]):
            diff_i = diff[i]  # (ch, n_mels, T)

            positive_diff = diff_i[diff_i > 0]
            negative_diff = diff_i[diff_i < 0]

            average_positive = positive_diff.mean() if positive_diff.numel() > 0 else torch.tensor(0.0, device=device)
            average_negative = negative_diff.mean() if negative_diff.numel() > 0 else torch.tensor(0.0, device=device)

            bleedless_list.append(100 * 1 / (average_positive + 1))
            fullness_list.append(100 * 1 / (-average_negative + 1))

        bleedless = rearrange(torch.stack(bleedless_list), '(bs n_src) -> bs n_src', bs=bs, n_src=n_src)
        fullness = rearrange(torch.stack(fullness_list), '(bs n_src) -> bs n_src', bs=bs, n_src=n_src)  # (bs, n_src)

        return {
            "bleedless": bleedless,
            "fullness": fullness
        }
