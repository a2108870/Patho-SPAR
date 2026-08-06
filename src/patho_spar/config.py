"""Configuration for the Patho-SPAR worst-case stain search."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Mapping


@dataclass(frozen=True)
class PathoSPARConfig:
    """Attack parameters used by the differentiable stain renderer.

    The defaults reproduce the threat-model and optimization budgets reported
    for the main Patho-SPAR robustness evaluation.
    """

    num_frequencies: int = 4
    spatial_concentration_bound: float = 0.20
    global_concentration_bound: float = 0.20
    rotation_bound_radians: float = 0.10
    od_offset_bound: float = 0.10
    steps: int = 100
    learning_rate: float = 0.10
    margin: float = 0.10
    gradient_clip: float = 1.0
    init_scale: float = 0.01
    od_threshold: float = 0.15
    angular_percentile: float = 99.0

    def __post_init__(self) -> None:
        if self.num_frequencies < 1:
            raise ValueError("num_frequencies must be positive")
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive")
        if not 0 < self.angular_percentile < 100:
            raise ValueError("angular_percentile must lie in (0, 100)")
        for name in (
            "spatial_concentration_bound",
            "global_concentration_bound",
            "rotation_bound_radians",
            "od_offset_bound",
            "margin",
            "init_scale",
            "od_threshold",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PathoSPARConfig":
        allowed = {field.name for field in fields(cls)}
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise ValueError(f"unknown Patho-SPAR configuration keys: {unknown}")
        return cls(**dict(values))

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PathoSPARConfig":
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - dependency is declared
            raise ImportError("PyYAML is required to read an attack configuration") from exc

        with Path(path).open("r", encoding="utf-8") as handle:
            values = yaml.safe_load(handle) or {}
        if not isinstance(values, Mapping):
            raise ValueError("the YAML configuration must contain a mapping")
        return cls.from_mapping(values)
