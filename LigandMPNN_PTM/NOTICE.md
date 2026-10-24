# Licensing of `LigandMPNN_PTM/`

This directory is a **fork of [dauparas/LigandMPNN](https://github.com/dauparas/LigandMPNN)** (Dauparas et al.). Two licences apply, split by authorship:

| What | Licence | File |
|---|---|---|
| **Our additions and the PTM fine-tune** — `ptm_utils.py`, `ptm_data/`, `examples/` (PTM examples), the fine-tuned weights `model_params_ptm/`, `requirements_tested.txt`, `README.md`, and **our modifications of upstream files** (`run.py`, `score.py`, `data_utils.py`, `model_utils.py`, the NumPy fix in `openfold/np/residue_constants.py`) | **PolyForm Noncommercial License 1.0.0** — free for research, education and other noncommercial use; commercial use needs a separate license from the authors | [`LICENSE-PTM-NONCOMMERCIAL`](LICENSE-PTM-NONCOMMERCIAL) |
| **Upstream LigandMPNN** — the original code, the vendored `openfold/` code, `sc_utils.py`, `get_model_params.sh`, `README_upstream.md` | **MIT** — *Copyright (c) 2024 Justas Dauparas*; the notice must stay with every copy | [`LICENSE`](LICENSE) |

Notes
* The MIT licence of the upstream portions is not changed and not removed: anyone may keep using those portions under MIT. Where a file contains both (the modified upstream files), our modifications are offered under PolyForm Noncommercial 1.0.0 on top of the MIT-licensed original.
* The **fine-tuned weights** `model_params_ptm/ligandmpnn_ptm_ft_v2A.pt` are a derivative of the released LigandMPNN weights (Dauparas et al., downloaded separately by `get_model_params.sh`, not included here); the terms of those parent weights are the upstream authors'. Our fine-tune is released under PolyForm Noncommercial 1.0.0.
* Please cite the original work (see `README.md`, *Credits*) in addition to this fork.

*Required notice for the PolyForm part: Copyright 2026 Lukas Gerstung (PTM-Binder-Design).*
