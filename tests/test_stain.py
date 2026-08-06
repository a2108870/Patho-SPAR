import torch

from patho_spar.stain import od_to_rgb, rgb_to_od


def test_rgb_od_round_trip() -> None:
    torch.manual_seed(0)
    images = torch.rand(2, 3, 16, 16).clamp_min(1e-3)
    recovered = od_to_rgb(rgb_to_od(images))
    torch.testing.assert_close(recovered, images, rtol=1e-5, atol=1e-6)
