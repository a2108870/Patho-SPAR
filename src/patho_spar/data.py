"""Minimal image and manifest utilities for attack evaluation."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset


def load_rgb_image(path: str | Path, image_size: int = 224) -> Tensor:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image = image.resize((image_size, image_size), Image.Resampling.BICUBIC)
        array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def save_rgb_image(image: Tensor, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (
        (image.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    )
    Image.fromarray(array, mode="RGB").save(path)


class ManifestDataset(Dataset[Tuple[Tensor, int, str]]):
    """Dataset backed by a CSV file with ``path`` and integer ``label`` columns."""

    def __init__(
        self,
        manifest: str | Path,
        data_root: str | Path | None = None,
        image_size: int = 224,
    ) -> None:
        self.manifest = Path(manifest)
        self.data_root = Path(data_root) if data_root is not None else self.manifest.parent
        self.image_size = int(image_size)
        self.records: List[Tuple[Path, int]] = []

        with self.manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or not {"path", "label"}.issubset(reader.fieldnames):
                raise ValueError("manifest must contain 'path' and 'label' columns")
            for row in reader:
                image_path = Path(row["path"])
                if not image_path.is_absolute():
                    image_path = self.data_root / image_path
                self.records.append((image_path, int(row["label"])))
        if not self.records:
            raise ValueError(f"manifest contains no image records: {self.manifest}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, str]:
        path, label = self.records[index]
        return load_rgb_image(path, self.image_size), label, str(path)
