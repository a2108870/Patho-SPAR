"""Classifier-guided Patho-SPAR worst-case stain search."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import Tensor, nn

from .config import PathoSPARConfig
from .normalization import imagenet_normalize
from .renderer import StainFormationRenderer

NormalizeFn = Callable[[Tensor], Tensor]


@dataclass
class AttackResult:
    """Outputs from one batched Patho-SPAR search."""

    adversarial_images: Tensor
    clean_predictions: Tensor
    adversarial_predictions: Tensor
    success: Tensor
    first_success_step: Tensor
    steps_run: int


def _extract_logits(output: object) -> Tensor:
    if isinstance(output, Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        return output[0]
    if isinstance(output, dict):
        for key in ("logits", "out"):
            value = output.get(key)
            if isinstance(value, Tensor):
                return value
    raise TypeError("the classifier must return a logits tensor or a container with logits")


@contextmanager
def _fixed_classifier(model: nn.Module) -> Iterator[None]:
    was_training = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, state in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(state)
        model.train(was_training)


class PathoSPAR:
    """Untargeted Patho-SPAR attack against a fixed pathology classifier.

    The clean prediction is the reference class. The search retains the first
    stain-perturbed iterate whose prediction differs from that reference.
    """

    def __init__(
        self,
        model: nn.Module,
        config: PathoSPARConfig | None = None,
        normalize: NormalizeFn | None = imagenet_normalize,
    ) -> None:
        self.model = model
        self.config = config or PathoSPARConfig()
        self.normalize = normalize or (lambda images: images)

    def _logits(self, images: Tensor) -> Tensor:
        return _extract_logits(self.model(self.normalize(images)))

    def __call__(self, images: Tensor) -> AttackResult:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        reference = images.detach()
        renderer = StainFormationRenderer(reference, self.config).to(reference.device)
        parameters = list(renderer.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.config.learning_rate)

        batch_size = reference.shape[0]
        best = reference.clone()
        success = torch.zeros(batch_size, dtype=torch.bool, device=reference.device)
        first_success_step = torch.full(
            (batch_size,), -1, dtype=torch.long, device=reference.device
        )
        steps_run = 0

        with _fixed_classifier(self.model):
            with torch.no_grad():
                clean_predictions = self._logits(reference).argmax(dim=1)

            for step in range(self.config.steps):
                optimizer.zero_grad(set_to_none=True)
                candidate = renderer()
                candidate = torch.where(torch.isfinite(candidate), candidate, reference).clamp(0, 1)
                logits = self._logits(candidate)
                reference_logits = logits.gather(1, clean_predictions[:, None]).squeeze(1)
                competing_logits = (
                    logits.masked_fill(
                        torch.nn.functional.one_hot(
                            clean_predictions, num_classes=logits.shape[1]
                        ).bool(),
                        float("-inf"),
                    )
                    .max(dim=1)
                    .values
                )
                loss = torch.relu(reference_logits - competing_logits + self.config.margin).sum()
                if not torch.isfinite(loss):
                    break
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.config.gradient_clip)
                optimizer.step()
                steps_run = step + 1

                with torch.no_grad():
                    candidate = renderer()
                    candidate = torch.where(torch.isfinite(candidate), candidate, reference).clamp(
                        0, 1
                    )
                    predictions = self._logits(candidate).argmax(dim=1)
                    newly_successful = (predictions != clean_predictions) & ~success
                    best[newly_successful] = candidate[newly_successful]
                    first_success_step[newly_successful] = step + 1
                    success |= newly_successful
                if bool(success.all()):
                    break

            adversarial_images = (
                torch.where(success.view(batch_size, 1, 1, 1), best, reference).detach().clamp(0, 1)
            )
            with torch.no_grad():
                adversarial_predictions = self._logits(adversarial_images).argmax(dim=1)

        result = AttackResult(
            adversarial_images=adversarial_images,
            clean_predictions=clean_predictions.detach(),
            adversarial_predictions=adversarial_predictions.detach(),
            success=success.detach(),
            first_success_step=first_success_step.detach(),
            steps_run=steps_run,
        )
        del optimizer, renderer
        return result
