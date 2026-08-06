"""Run Patho-SPAR on the bundled real H&E patch without a checkpoint.

This example validates installation, image loading, differentiable rendering,
and gradient-based optimization.  Its tiny colour-based classifier is not a
pathology model and the output must not be interpreted as an experimental
result.
"""

from pathlib import Path

import torch
from torch import nn

from patho_spar import PathoSPAR, PathoSPARConfig
from patho_spar.data import load_rgb_image, save_rgb_image


class ColorBalanceClassifier(nn.Module):
    """A deterministic two-logit model used only to exercise the API."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        red = images[:, 0].mean(dim=(1, 2))
        blue = images[:, 2].mean(dim=(1, 2))
        score = red - blue
        return torch.stack((score, -score), dim=1)


def main() -> None:
    torch.manual_seed(42)
    example_dir = Path(__file__).resolve().parent
    image_path = example_dir / "nct_crc_tum_example.png"
    output_path = Path("outputs") / "smoke_test_patho_spar.png"

    image = load_rgb_image(image_path).unsqueeze(0)
    config = PathoSPARConfig(steps=10)
    result = PathoSPAR(ColorBalanceClassifier(), config, normalize=None)(image)
    save_rgb_image(result.adversarial_images[0], output_path)

    print(f"Input:  {image_path}")
    print(f"Output: {output_path}")
    print(f"Prediction: {int(result.clean_predictions[0])} -> "
          f"{int(result.adversarial_predictions[0])}")
    print(f"Attack successful: {bool(result.success[0])}")
    print("The script completed successfully; this is an installation check, not a benchmark.")


if __name__ == "__main__":
    main()
