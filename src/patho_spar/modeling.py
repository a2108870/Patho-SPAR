"""Checkpoint loading helpers for the public command-line examples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch import Tensor, nn


def _state_dict(checkpoint: object) -> Dict[str, Tensor]:
    if not isinstance(checkpoint, dict):
        raise TypeError("checkpoint must contain a state dictionary")
    for key in ("state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    if not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in checkpoint.items()
    ):
        raise TypeError("could not identify a tensor state dictionary in the checkpoint")
    state = dict(checkpoint)
    for prefix in ("module.", "model."):
        if state and all(key.startswith(prefix) for key in state):
            state = {key[len(prefix) :]: value for key, value in state.items()}
    return state


def load_timm_classifier(
    architecture: str,
    num_classes: int,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> nn.Module:
    """Instantiate a timm classifier and load a local checkpoint."""

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise ImportError("timm is required by the command-line model loader") from exc

    model = timm.create_model(architecture, pretrained=False, num_classes=num_classes)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    incompatibility = model.load_state_dict(_state_dict(checkpoint), strict=strict)
    if not strict and (incompatibility.missing_keys or incompatibility.unexpected_keys):
        print(
            "warning: non-strict checkpoint load; "
            f"missing={incompatibility.missing_keys}, "
            f"unexpected={incompatibility.unexpected_keys}"
        )
    model.to(device).eval()
    return model
