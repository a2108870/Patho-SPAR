# Patho-SPAR

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Review status](https://img.shields.io/badge/status-anonymous%20reviewer%20release-lightgrey.svg)](#double-blind-review)

Reference implementation of **Patho-SPAR**, the stain-physics threat model
introduced in the anonymous manuscript <em>Beyond &#8467;<sub>p</sub> Perturbations:
A Stain-Physics Threat Model for Digital Pathology</em>.

Patho-SPAR evaluates a fixed pathology classifier by optimizing bounded H&E
stain-formation variables in optical-density (OD) space. This repository
contains the differentiable renderer, classifier-guided worst-case search,
single-image interface, manifest-based evaluation, tests, and the attack
configuration used in the manuscript.

## Double-blind review

This repository is anonymized for peer review. Citation metadata, package
metadata, release history, and commit authorship intentionally omit identifying
information. Author information and the final citation will be restored after
the review process.

Datasets, classifier checkpoints, training pipelines, online augmentation,
competing attacks, and private experiment orchestration are not distributed.

## Installation

Python 3.8 or later is required. Install a PyTorch build compatible with the
local CUDA environment, then install Patho-SPAR:

```bash
git clone https://github.com/a2108870/Patho-SPAR.git
cd Patho-SPAR
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For development:

```bash
python -m pip install -e ".[dev]"
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
```

## Quick start

The classifier must accept normalized tensors and return logits. Patho-SPAR
expects floating-point RGB tensors in `[0, 1]` with shape
`[batch, 3, height, width]`.

```python
from patho_spar import PathoSPAR, PathoSPARConfig

model = ...   # fixed classifier on the same device as images
images = ...  # RGB tensor in [0, 1]

attack = PathoSPAR(model, PathoSPARConfig())
result = attack(images)

adversarial = result.adversarial_images
success = result.success
first_success_step = result.first_success_step
```

ImageNet normalization is applied by default because the evaluated classifiers
use ImageNet-pretrained ResNet-50 and Swin-Tiny backbones. For a classifier that
performs its own preprocessing, pass `normalize=None`.

## Command-line interfaces

The model loader supports standard `timm` architectures and checkpoints stored
as a state dictionary or under a `state_dict` or `model` key.

### Attack one image

```bash
patho-spar-attack \
  --architecture resnet50 \
  --num-classes 2 \
  --checkpoint checkpoints/cam17_resnet50_224.pth \
  --image examples/patch.png \
  --output outputs/patch_patho_spar.png \
  --config configs/paper_attack.yaml \
  --device cuda:0
```

The command saves the adversarial patch and reports the clean prediction,
adversarial prediction, success status, and first successful optimization step.

### Evaluate a manifest

Dataset evaluation reads a CSV manifest with two required columns:

```csv
path,label
relative/path/to/patch_0001.png,0
relative/path/to/patch_0002.png,1
```

Paths are resolved relative to `--data-root`, or relative to the manifest when
`--data-root` is omitted. Labels must be zero-based integer class indices.

```bash
patho-spar-evaluate \
  --architecture resnet50 \
  --num-classes 2 \
  --checkpoint checkpoints/cam17_resnet50_224.pth \
  --manifest manifests/cam17_ood_test.csv \
  --data-root /path/to/camelyon17 \
  --output outputs/cam17_resnet50.json \
  --config configs/paper_attack.yaml \
  --batch-size 32 \
  --device cuda:0
```

The evaluator attacks only initially correct samples. It defines the attack
success rate as:

> **ASR = successful attacks on initially correct samples / initially correct samples.**

The JSON output records the configuration, clean-correct count, successful
attack count, and class-transition counts.

## Method summary

For an RGB H&E patch `x`, Patho-SPAR estimates an H/E stain basis and two
concentration maps while preserving the OD residual outside the fitted
two-stain subspace. The search optimizes three groups of variables:

1. **Stain-vector orientation:** two tangent-plane coordinates independently
   rotate the hematoxylin and eosin stain vectors.
2. **Concentration-field modulation:** H and E have independent global scales
   and share a smooth zero-DC field represented by a low-frequency DCT block.
3. **Global OD offset:** one patch-wide scalar models uniform darkening or
   lightening in OD space.

The classifier remains fixed. Adam minimizes an untargeted hinge margin until
the prediction changes or the iteration budget is exhausted. The first
successful iterate is retained; otherwise, the clean input is returned.

## Data and evaluation protocol

No image data are redistributed. Download each dataset from its official source
and comply with its license and access conditions.

| Dataset | Role | Protocol |
|---|---|---|
| [Camelyon17-WILDS](https://wilds.stanford.edu/datasets/) | Binary lymph-node metastasis detection and cross-center evaluation | The official WILDS split is followed. The complete center-2 OOD-test split contains 85,054 patches. |
| [NCT-CRC-HE-100K](https://doi.org/10.5281/zenodo.1214456) | Nine-class colorectal tissue classification | A class-stratified 70/15/15 split with seed 42 yields 70,000 training, 15,000 validation, and 15,000 test patches. |
| [PLISM](https://doi.org/10.25452/figshare.plus.c.6773925.v2) | Registered-pair validation under real staining variations | GIVH/AT2 is the reference and nine same-scanner target staining conditions form 1,800 registered pairs across 200 tissue locations. PLISM is not required by this attack-only package. |

For Camelyon17-WILDS, export the center-2 test records to the two-column
manifest format. For NCT-CRC-HE-100K, preserve the original nine folder labels;
the reported evaluation does not collapse the eight non-TUM classes.

## Models and preprocessing

The manuscript evaluates ImageNet-pretrained ResNet-50 and Swin-Tiny
classifiers. Their `timm` identifiers are:

```text
resnet50
swin_tiny_patch4_window7_224
```

All patches are resized to `224 x 224`, converted to RGB tensors in `[0, 1]`,
and normalized immediately before classifier inference with ImageNet mean
`(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
The renderer always operates on unnormalized RGB values in `[0, 1]`.

Classifier checkpoints are not included. The SHA-256 digest of the reference
Camelyon17 ResNet-50 checkpoint is:

```text
73979011531532742587d61d2d9f039f06327bec4e62b99ab2b83e906e07df98
```

Exact numerical reproduction requires the same checkpoint, dataset split,
preprocessing, attack seed, and software stack.

## Reference results

The following attack success rates are reported for the complete evaluation
splits and fixed reference checkpoints:

| Dataset | ResNet-50 | Swin-Tiny |
|---|---:|---:|
| Camelyon17-WILDS OOD test | 75.42% | 71.12% |
| NCT-CRC-HE-100K test | 89.68% | 83.11% |

The mean across the four dataset-architecture settings is 79.83%.

## Main attack configuration

[`configs/paper_attack.yaml`](configs/paper_attack.yaml) records the main
robustness-evaluation setting.

| Parameter | Value |
|---|---:|
| Shared DCT block | `4 x 4`, DC fixed to zero |
| Spatial concentration bound | `0.20` |
| H/E global concentration bound | `0.20` |
| H/E stain-vector rotation bound | `0.10 rad` |
| Global OD-offset bound | `0.10` |
| Optimizer | Adam |
| Learning rate | `0.10` |
| Hinge margin | `0.10` |
| Gradient-norm clip | `1.0` |
| Maximum search steps | `100` |
| Evaluation seed | `42` |

These are threat-model parameters rather than pixel-norm budgets. Results for
pixel or general appearance attacks therefore correspond to different
admissible perturbation spaces, not a shared attack radius.

## Repository layout

```text
Patho-SPAR/
|-- .github/                      # CI and contribution templates
|-- configs/paper_attack.yaml     # Manuscript attack configuration
|-- examples/manifest.example.csv # Evaluation-manifest schema
|-- src/patho_spar/               # Renderer, search, data, and CLI code
|-- tests/                        # Numerical and API tests
|-- CONTRIBUTING.md
|-- SECURITY.md
`-- pyproject.toml
```

## Scope and limitations

Patho-SPAR parameterizes bounded variations in the modeled H&E stain-formation
factors while preserving tissue layout and a fixed OD residual. It does not
model every digital-pathology deployment shift. Scanner-dependent color
response, focus variation, compression artifacts, tissue deformation, and
other acquisition effects remain outside the attack space. Generated examples
should not be interpreted as proof that every optimized perturbation is
physically realizable in a laboratory.

## Development and contribution

Install the development dependencies and run the same checks as CI:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
python -m pytest -q
python -m build
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution workflow and
[SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Citation

The manuscript is under double-blind review. `CITATION.cff` contains temporary
anonymous metadata and will be updated after review.

## License

Patho-SPAR is released under the [MIT License](LICENSE). Dataset licenses and
access conditions are governed by their respective providers.
