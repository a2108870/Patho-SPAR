import argparse
from pathlib import Path

from patho_spar.cli.common import resolve_normalizer
from patho_spar.cli.attack_image import resolve_report_path
from patho_spar.cli.evaluate import build_parser as build_evaluate_parser


def test_normalizer_options_are_explicit() -> None:
    assert resolve_normalizer("imagenet") is not None
    assert resolve_normalizer("none") is None


def test_evaluate_cli_defaults_to_paper_preprocessing() -> None:
    parser: argparse.ArgumentParser = build_evaluate_parser()
    args = parser.parse_args(
        [
            "--architecture",
            "resnet50",
            "--num-classes",
            "2",
            "--checkpoint",
            "checkpoint.pth",
            "--manifest",
            "manifest.csv",
            "--output",
            "report.json",
        ]
    )
    assert args.normalization == "imagenet"


def test_single_image_report_defaults_to_an_image_sidecar() -> None:
    assert resolve_report_path(Path("outputs/patch.png"), None) == Path("outputs/patch.json")
    assert resolve_report_path(Path("outputs/patch.png"), Path("reports/result.json")) == Path(
        "reports/result.json"
    )
