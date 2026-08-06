import torch

from patho_spar import PathoSPARConfig
from patho_spar.renderer import StainFormationRenderer


def _he_like_batch(batch_size: int = 2) -> torch.Tensor:
    torch.manual_seed(1)
    base = torch.rand(batch_size, 3, 24, 24)
    base[:, 0] = 0.55 + 0.35 * base[:, 0]
    base[:, 1] = 0.35 + 0.45 * base[:, 1]
    base[:, 2] = 0.50 + 0.40 * base[:, 2]
    return base.clamp(0, 1)


def test_zero_parameters_reconstruct_reference() -> None:
    images = _he_like_batch()
    renderer = StainFormationRenderer(images, PathoSPARConfig())
    with torch.no_grad():
        for parameter in renderer.parameters():
            parameter.zero_()
    rendered = renderer()
    # The formal renderer applies the fixed-slope non-negativity map both after
    # decomposition and after concentration modulation. The resulting zero-state
    # reconstruction is numerically within the renderer's 1e-3 identity tolerance.
    torch.testing.assert_close(rendered, images, rtol=2e-3, atol=1e-3)


def test_effective_parameters_respect_budgets() -> None:
    config = PathoSPARConfig()
    renderer = StainFormationRenderer(_he_like_batch(), config)
    with torch.no_grad():
        for parameter in renderer.parameters():
            parameter.fill_(100.0)
    effective = renderer.effective_parameters()
    assert torch.all(effective["rotation_h_radians"] <= config.rotation_bound_radians)
    assert torch.all(effective["rotation_e_radians"] <= config.rotation_bound_radians)
    assert torch.all(effective["global_concentration_h"].abs() <= config.global_concentration_bound)
    assert torch.all(effective["global_concentration_e"].abs() <= config.global_concentration_bound)
    assert torch.all(effective["global_od_offset"].abs() <= config.od_offset_bound)
