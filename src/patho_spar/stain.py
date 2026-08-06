"""H&E optical-density decomposition for the Patho-SPAR renderer."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as functional
from torch import Tensor

RGB_EPSILON = 1e-6
REFERENCE_HEMATOXYLIN = (0.6500, 0.7040, 0.2860)
REFERENCE_EOSIN = (0.0720, 0.9900, 0.1050)


def rgb_to_od(rgb: Tensor) -> Tensor:
    """Map normalized RGB intensities to optical density."""

    return -torch.log10(rgb.clamp(min=RGB_EPSILON))


def od_to_rgb(od: Tensor) -> Tensor:
    """Apply the inverse Beer--Lambert transform and return RGB in ``[0, 1]``."""

    return torch.pow(10.0, -od).clamp(0.0, 1.0)


def reference_stain_matrix(device: torch.device, dtype: torch.dtype) -> Tensor:
    hematoxylin = torch.tensor(REFERENCE_HEMATOXYLIN, device=device, dtype=dtype)
    eosin = torch.tensor(REFERENCE_EOSIN, device=device, dtype=dtype)
    hematoxylin = hematoxylin / (hematoxylin.norm() + 1e-8)
    eosin = eosin / (eosin.norm() + 1e-8)
    return torch.stack((hematoxylin, eosin), dim=1)


def estimate_stain_matrix(
    od_image: Tensor,
    od_threshold: float = 0.15,
    angular_percentile: float = 99.0,
) -> Tensor:
    """Estimate one H&E stain matrix with a Macenko-style OD-plane fit."""

    if od_image.ndim != 3 or od_image.shape[0] != 3:
        raise ValueError(f"expected an OD image with shape [3,H,W], got {od_image.shape}")

    pixels = od_image.reshape(3, -1).T
    foreground = pixels[pixels.sum(dim=1) > od_threshold]
    if foreground.shape[0] < 10:
        return reference_stain_matrix(od_image.device, od_image.dtype)

    centered = foreground - foreground.mean(dim=0, keepdim=True)
    try:
        _, _, principal_directions = torch.linalg.svd(centered, full_matrices=False)
    except RuntimeError:
        return reference_stain_matrix(od_image.device, od_image.dtype)

    plane = principal_directions[:2, :]
    projected = foreground @ plane.T
    angles = torch.atan2(projected[:, 1], projected[:, 0])
    low = torch.quantile(angles, (100.0 - angular_percentile) / 100.0)
    high = torch.quantile(angles, angular_percentile / 100.0)

    direction_low = torch.stack((torch.cos(low), torch.sin(low)))
    direction_high = torch.stack((torch.cos(high), torch.sin(high)))
    stain_low = (plane.T @ direction_low).abs()
    stain_high = (plane.T @ direction_high).abs()
    stain_low = stain_low / (stain_low.norm() + 1e-8)
    stain_high = stain_high / (stain_high.norm() + 1e-8)

    reference_h = od_image.new_tensor(REFERENCE_HEMATOXYLIN)
    if torch.dot(stain_low, reference_h) >= torch.dot(stain_high, reference_h):
        return torch.stack((stain_low, stain_high), dim=1)
    return torch.stack((stain_high, stain_low), dim=1)


def estimate_stain_matrices(
    od_images: Tensor,
    od_threshold: float = 0.15,
    angular_percentile: float = 99.0,
) -> Tensor:
    """Estimate an independent H&E stain matrix for every image in a batch."""

    if od_images.ndim != 4 or od_images.shape[1] != 3:
        raise ValueError(f"expected OD images with shape [B,3,H,W], got {od_images.shape}")
    matrices = [
        estimate_stain_matrix(image, od_threshold, angular_percentile) for image in od_images
    ]
    return torch.stack(matrices, dim=0)


def decompose_he(od_images: Tensor, stain_matrices: Tensor) -> Tuple[Tensor, Tensor]:
    """Solve H/E concentration maps with a batched pseudoinverse."""

    batch, _, height, width = od_images.shape
    if stain_matrices.shape != (batch, 3, 2):
        raise ValueError(
            f"expected stain matrices with shape {(batch, 3, 2)}, got {tuple(stain_matrices.shape)}"
        )
    flattened = od_images.reshape(batch, 3, -1)
    pseudoinverse = torch.linalg.pinv(stain_matrices)
    concentrations = torch.einsum("bsc,bcn->bsn", pseudoinverse, flattened)
    concentrations = functional.leaky_relu(concentrations, negative_slope=0.01)
    hematoxylin = concentrations[:, 0].reshape(batch, height, width)
    eosin = concentrations[:, 1].reshape(batch, height, width)
    return hematoxylin, eosin
