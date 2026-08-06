import torch
from torch import nn

from patho_spar import PathoSPAR, PathoSPARConfig


class MeanColorClassifier(nn.Module):
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        red = images[:, 0].mean(dim=(1, 2))
        blue = images[:, 2].mean(dim=(1, 2))
        score = red - blue
        return torch.stack((score, -score), dim=1)


def test_attack_returns_finite_batched_result() -> None:
    torch.manual_seed(2)
    images = torch.rand(2, 3, 20, 20) * 0.5 + 0.25
    config = PathoSPARConfig(steps=2)
    result = PathoSPAR(MeanColorClassifier(), config, normalize=None)(images)
    assert result.adversarial_images.shape == images.shape
    assert result.clean_predictions.shape == (2,)
    assert result.success.dtype == torch.bool
    assert torch.isfinite(result.adversarial_images).all()
    assert torch.all((0 <= result.adversarial_images) & (result.adversarial_images <= 1))


def test_unsuccessful_samples_return_the_reference_image() -> None:
    torch.manual_seed(3)
    images = torch.rand(1, 3, 20, 20) * 0.5 + 0.25
    config = PathoSPARConfig(
        steps=2,
        spatial_concentration_bound=0.0,
        global_concentration_bound=0.0,
        rotation_bound_radians=0.0,
        od_offset_bound=0.0,
    )
    result = PathoSPAR(MeanColorClassifier(), config, normalize=None)(images)
    assert not bool(result.success[0])
    torch.testing.assert_close(result.adversarial_images, images, rtol=1e-4, atol=2e-5)
