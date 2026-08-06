"""Differentiable H&E stain-formation renderer used by Patho-SPAR."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from .config import PathoSPARConfig
from .dct import LowFrequencyDCTField
from .stain import decompose_he, estimate_stain_matrices, od_to_rgb, rgb_to_od


class TangentPlaneStainRotation(nn.Module):
    """Rotate one unit stain vector using two bounded tangent-plane coordinates."""

    def __init__(self, batch_size: int, max_angle: float, init_scale: float) -> None:
        super().__init__()
        self.batch_size = int(batch_size)
        self.max_angle = float(max_angle)
        self.raw_tangent = nn.Parameter(torch.randn(batch_size, 2) * init_scale)

    @staticmethod
    def _basis(stain_vectors: Tensor) -> tuple[Tensor, Tensor]:
        least_aligned_axis = stain_vectors.abs().argmin(dim=1)
        canonical = torch.zeros_like(stain_vectors)
        canonical[
            torch.arange(stain_vectors.shape[0], device=stain_vectors.device),
            least_aligned_axis,
        ] = 1.0
        first = canonical - (canonical * stain_vectors).sum(dim=1, keepdim=True) * stain_vectors
        first = first / (first.norm(dim=1, keepdim=True) + 1e-8)
        second = torch.cross(stain_vectors, first, dim=1)
        second = second / (second.norm(dim=1, keepdim=True) + 1e-8)
        return first, second

    @property
    def effective_angle(self) -> Tensor:
        return self.max_angle * torch.tanh(self.raw_tangent.norm(dim=1))

    def forward(self, stain_vectors: Tensor) -> Tensor:
        if stain_vectors.shape != (self.batch_size, 3):
            raise ValueError(
                f"expected stain vectors with shape {(self.batch_size, 3)}, "
                f"got {tuple(stain_vectors.shape)}"
            )
        vectors = stain_vectors / (stain_vectors.norm(dim=1, keepdim=True) + 1e-8)
        first, second = self._basis(vectors)
        norm = self.raw_tangent.norm(dim=1, keepdim=True)
        direction = self.raw_tangent / (norm + 1e-8)
        tangent = direction[:, :1] * first + direction[:, 1:] * second
        tangent = tangent / (tangent.norm(dim=1, keepdim=True) + 1e-8)
        angle = self.max_angle * torch.tanh(norm)
        rotated = vectors * torch.cos(angle) + tangent * torch.sin(angle)
        return rotated / (rotated.norm(dim=1, keepdim=True) + 1e-8)


class StainFormationRenderer(nn.Module):
    """Render bounded Patho-SPAR variations of a fixed batch of H&E patches.

    The renderer estimates each patch's stain matrix once, then keeps the stain
    matrix, base concentration maps, and OD residual fixed during optimization.
    Only the variables defined by the Patho-SPAR threat model are learnable.
    """

    def __init__(self, reference_images: Tensor, config: PathoSPARConfig) -> None:
        super().__init__()
        if reference_images.ndim != 4 or reference_images.shape[1] != 3:
            raise ValueError("reference_images must be an RGB tensor with shape [B,3,H,W]")
        if not reference_images.is_floating_point():
            raise TypeError("reference_images must use a floating-point dtype")
        if reference_images.min() < 0 or reference_images.max() > 1:
            raise ValueError("reference_images must lie in [0, 1]")

        self.config = config
        self.batch_size = int(reference_images.shape[0])
        self.spatial_size = (int(reference_images.shape[2]), int(reference_images.shape[3]))

        with torch.no_grad():
            reference_od = rgb_to_od(reference_images.detach())
            stain_matrices = estimate_stain_matrices(
                reference_od,
                od_threshold=config.od_threshold,
                angular_percentile=config.angular_percentile,
            )
            concentration_h, concentration_e = decompose_he(reference_od, stain_matrices)
            stain_h = stain_matrices[:, :, 0].view(self.batch_size, 3, 1, 1)
            stain_e = stain_matrices[:, :, 1].view(self.batch_size, 3, 1, 1)
            reconstructed = (
                concentration_h.unsqueeze(1) * stain_h + concentration_e.unsqueeze(1) * stain_e
            )
            od_residual = reference_od - reconstructed

        self.register_buffer("reference_images", reference_images.detach().clone())
        self.register_buffer("stain_matrices", stain_matrices)
        self.register_buffer("concentration_h", concentration_h)
        self.register_buffer("concentration_e", concentration_e)
        self.register_buffer("od_residual", od_residual)

        self.rotation_h = TangentPlaneStainRotation(
            self.batch_size, config.rotation_bound_radians, config.init_scale
        )
        self.rotation_e = TangentPlaneStainRotation(
            self.batch_size, config.rotation_bound_radians, config.init_scale
        )
        self.spatial_field = LowFrequencyDCTField(
            self.batch_size, config.num_frequencies, config.init_scale
        )
        self.raw_global_concentration_h = nn.Parameter(
            torch.randn(self.batch_size, 1, 1) * config.init_scale
        )
        self.raw_global_concentration_e = nn.Parameter(
            torch.randn(self.batch_size, 1, 1) * config.init_scale
        )
        # The formal search initializes the patch-wide OD offset at identity.
        self.raw_od_offset = nn.Parameter(torch.zeros(self.batch_size))

    def _modulate_concentration(
        self, concentration: Tensor, raw_scale: Tensor, field: Tensor
    ) -> Tensor:
        global_multiplier = 1.0 + self.config.global_concentration_bound * torch.tanh(raw_scale)
        spatial_multiplier = 1.0 + self.config.spatial_concentration_bound * field
        modulated = concentration * global_multiplier * spatial_multiplier
        return functional.leaky_relu(modulated, negative_slope=0.01)

    def forward(self) -> Tensor:
        field = self.spatial_field(self.spatial_size)
        concentration_h = self._modulate_concentration(
            self.concentration_h, self.raw_global_concentration_h, field
        )
        concentration_e = self._modulate_concentration(
            self.concentration_e, self.raw_global_concentration_e, field
        )

        stain_h = self.rotation_h(self.stain_matrices[:, :, 0]).view(self.batch_size, 3, 1, 1)
        stain_e = self.rotation_e(self.stain_matrices[:, :, 1]).view(self.batch_size, 3, 1, 1)
        adversarial_od = (
            concentration_h.unsqueeze(1) * stain_h
            + concentration_e.unsqueeze(1) * stain_e
            + self.od_residual
        )
        offset = self.config.od_offset_bound * torch.tanh(self.raw_od_offset)
        adversarial_od = adversarial_od + offset.view(self.batch_size, 1, 1, 1)
        return od_to_rgb(adversarial_od)

    @torch.no_grad()
    def effective_parameters(self) -> Dict[str, Tensor]:
        """Return bounded physical variables for diagnostics and serialization."""

        return {
            "rotation_h_radians": self.rotation_h.effective_angle.detach().clone(),
            "rotation_e_radians": self.rotation_e.effective_angle.detach().clone(),
            "global_concentration_h": (
                self.config.global_concentration_bound
                * torch.tanh(self.raw_global_concentration_h.squeeze(-1).squeeze(-1))
            )
            .detach()
            .clone(),
            "global_concentration_e": (
                self.config.global_concentration_bound
                * torch.tanh(self.raw_global_concentration_e.squeeze(-1).squeeze(-1))
            )
            .detach()
            .clone(),
            "global_od_offset": (self.config.od_offset_bound * torch.tanh(self.raw_od_offset))
            .detach()
            .clone(),
        }
