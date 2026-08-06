# Examples

`nct_crc_tum_example.png` is a 224 by 224, Macenko-normalized H&E tile from
the `TUM` class of NCT-CRC-HE-100K. It was converted from the source file
`TUM-AACHTSSN.tif` without resizing or pixel edits. The tile is included only
to let users verify the package on a real pathology image; it is not an
evaluation sample or a model input reference. The source dataset is
distributed under CC BY 4.0. Please cite and follow the terms at
<https://doi.org/10.5281/zenodo.1214456> when reusing it.

`smoke_test.py` runs the attack on this tile using a deliberately minimal
colour-based classifier. It verifies the software path only and makes no
claim about pathology-model robustness.

`manifest.example.csv` documents the schema expected by
`patho-spar-evaluate`; its paths and labels are placeholders.
