"""Shared command-line helpers."""

from __future__ import annotations

import argparse
import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from patho_spar.config import PathoSPARConfig
from patho_spar.modeling import load_timm_classifier


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--architecture", required=True, help="timm model name")
    parser.add_argument("--num-classes", required=True, type=int)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument(
        "--device",
        default="auto",
        help="PyTorch device, or 'auto' to select CUDA when available",
    )
    parser.add_argument(
        "--non-strict-checkpoint",
        action="store_true",
        help="allow missing or unexpected checkpoint keys",
    )


def add_attack_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        help="YAML attack configuration; omitted values use the paper defaults",
    )
    parser.add_argument("--seed", type=int, default=42)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_runtime(args: argparse.Namespace):
    device = resolve_device(args.device)
    set_seed(args.seed)
    config = (
        PathoSPARConfig.from_yaml(args.config) if args.config is not None else PathoSPARConfig()
    )
    model = load_timm_classifier(
        args.architecture,
        args.num_classes,
        args.checkpoint,
        device,
        strict=not args.non_strict_checkpoint,
    )
    return device, config, model
