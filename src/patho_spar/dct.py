"""Differentiable low-frequency DCT field used for concentration modulation."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor, nn


class LowFrequencyDCTField(nn.Module):
    """Generate a smooth, bounded spatial field from a low-frequency DCT block."""

    def __init__(
        self,
        batch_size: int,
        num_frequencies: int = 4,
        init_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.num_frequencies = int(num_frequencies)
        self.raw_coefficients = nn.Parameter(
            torch.randn(batch_size, num_frequencies, num_frequencies) * init_scale
        )

    def forward(self, spatial_size: Tuple[int, int]) -> Tensor:
        height, width = spatial_size
        nf = min(self.num_frequencies, height, width)

        # The spatial field has zero DC. Global H/E concentration changes are
        # represented separately by two stain-specific scalar variables.
        low_frequency = self.raw_coefficients[:, :nf, :nf]
        mask = torch.ones_like(low_frequency)
        mask[:, 0, 0] = 0.0

        full_coefficients = low_frequency.new_zeros(self.batch_size, height, width)
        full_coefficients[:, :nf, :nf] = low_frequency * mask
        spatial_field = self.idct_2d(full_coefficients) * (height * width)
        return torch.tanh(spatial_field)

    @staticmethod
    def idct_1d(coefficients: Tensor) -> Tensor:
        """Type-III DCT along the final dimension."""

        length = coefficients.shape[-1]
        indices = torch.arange(length, device=coefficients.device, dtype=coefficients.dtype)
        phase = 1j * math.pi * indices / (2 * length)
        rotation = torch.exp(phase)

        weighted = coefficients.clone()
        weighted[..., 0] = weighted[..., 0] * 0.5
        reordered = torch.fft.ifft(weighted * rotation, dim=-1).real * length

        output = torch.empty_like(reordered)
        output[..., ::2] = reordered[..., : (length + 1) // 2]
        output[..., 1::2] = reordered[..., (length + 1) // 2 :].flip(dims=[-1])
        return output / length

    @classmethod
    def idct_2d(cls, coefficients: Tensor) -> Tensor:
        first = cls.idct_1d(coefficients)
        return cls.idct_1d(first.transpose(-2, -1)).transpose(-2, -1)
