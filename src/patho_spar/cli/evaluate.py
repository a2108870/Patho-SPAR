"""Evaluate Patho-SPAR attack success rate on a labeled image manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from patho_spar import PathoSPAR, imagenet_normalize
from patho_spar.data import ManifestDataset, save_rgb_image

from .common import (
    add_attack_arguments,
    add_model_arguments,
    load_runtime,
    sha256_file,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_model_arguments(parser)
    add_attack_arguments(parser)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--save-successful",
        type=Path,
        help="optional directory for successful adversarial images",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device, config, model = load_runtime(args)
    dataset = ManifestDataset(args.manifest, args.data_root, args.image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    attack = PathoSPAR(model, config)

    total = 0
    clean_correct = 0
    successful = 0
    transition_counts: Dict[str, int] = {}
    saved = 0

    for images, labels, paths in tqdm(loader, desc="Patho-SPAR"):
        images = images.to(device)
        labels = labels.to(device)
        total += int(labels.numel())
        with torch.no_grad():
            clean_predictions = model(imagenet_normalize(images)).argmax(dim=1)
        correct = clean_predictions.eq(labels)
        clean_correct += int(correct.sum().item())
        if not bool(correct.any()):
            continue

        selected_images = images[correct]
        selected_paths = [path for path, keep in zip(paths, correct.cpu().tolist()) if keep]
        result = attack(selected_images)
        successful += int(result.success.sum().item())

        for before, after in zip(
            result.clean_predictions.cpu().tolist(),
            result.adversarial_predictions.cpu().tolist(),
        ):
            key = f"{before}->{after}"
            transition_counts[key] = transition_counts.get(key, 0) + 1

        if args.save_successful is not None:
            for image, source, is_success in zip(
                result.adversarial_images,
                selected_paths,
                result.success.cpu().tolist(),
            ):
                if not is_success:
                    continue
                destination = args.save_successful / f"{saved:06d}_{Path(source).name}"
                save_rgb_image(image, destination)
                saved += 1

    attack_success_rate = successful / clean_correct if clean_correct else float("nan")
    report = {
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "architecture": args.architecture,
        "num_classes": args.num_classes,
        "total_images": total,
        "initially_correct": clean_correct,
        "successful_attacks": successful,
        "attack_success_rate": attack_success_rate,
        "transition_counts": transition_counts,
        "seed": args.seed,
        "configuration": config.to_dict(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
