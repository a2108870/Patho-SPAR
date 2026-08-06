"""Attack one H&E image with a local classifier checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patho_spar import PathoSPAR
from patho_spar.data import load_rgb_image, save_rgb_image

from .common import add_attack_arguments, add_model_arguments, load_runtime, sha256_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    add_attack_arguments(parser)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--report",
        type=Path,
        help="JSON report path; defaults to the output image path with a .json suffix",
    )
    parser.add_argument("--image-size", type=int, default=224)
    return parser


def resolve_report_path(output: Path, report: Path | None) -> Path:
    """Choose an explicit report destination or the image sidecar path."""

    return report if report is not None else output.with_suffix(".json")


def main() -> None:
    args = build_parser().parse_args()
    device, config, model, normalize = load_runtime(args)
    image = load_rgb_image(args.image, args.image_size).unsqueeze(0).to(device)
    result = PathoSPAR(model, config, normalize=normalize)(image)
    save_rgb_image(result.adversarial_images[0], args.output)
    report_path = resolve_report_path(args.output, args.report)

    report = {
        "input": str(args.image),
        "output": str(args.output),
        "report": str(report_path),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture": args.architecture,
        "num_classes": args.num_classes,
        "normalization": args.normalization,
        "image_size": args.image_size,
        "clean_prediction": int(result.clean_predictions[0].item()),
        "adversarial_prediction": int(result.adversarial_predictions[0].item()),
        "success": bool(result.success[0].item()),
        "first_success_step": int(result.first_success_step[0].item()),
        "steps_run": result.steps_run,
        "seed": args.seed,
        "configuration": config.to_dict(),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
