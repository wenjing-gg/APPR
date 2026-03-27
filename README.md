<div align="center">

# APPR

**Anatomy-aware Pretraining and Pseudo-label Rectification for Semi-Supervised Maxillary Sinus Segmentation**

PyTorch / MONAI implementation for low-label bilateral maxillary sinus segmentation.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![MONAI](https://img.shields.io/badge/MONAI-3D%20Medical%20Imaging-7C3AED?style=flat-square)

<img src="assets/appr-overview.svg" alt="APPR overview" width="100%" />

</div>

## Overview

APPR is a three-stage semi-supervised framework designed for anatomically constrained bilateral maxillary sinus segmentation.

- **Stage I – Anatomy-aware pretraining** initializes the segmentation network with robust aerated-region priors extracted from unlabeled scans.
- **Stage II – Semantic rectification** improves labeled supervision with side-aware constraints that reduce left-right semantic confusion.
- **Stage III – Pseudo-label co-refinement** progressively fuses anatomical priors and model predictions to obtain cleaner unlabeled supervision.

This repository contains the core training, evaluation, prior modeling, loss design, and 3D backbone implementation used by APPR.

## Highlights

- Unified `train.py` entry for `unsup`, `sup`, and `corefined` stages.
- Explicit bilateral semantic handling for left/right maxillary sinus consistency.
- Anatomy-aware prior extraction implemented in `PriorNet.py`.
- MONAI-based 3D training and sliding-window inference pipeline.
- Lightweight evaluation entry in `eval.py` for checkpoint-based testing.

## Repository Layout

```text
APPR/
├── PriorNet.py      # anatomy-aware prior extraction and pretraining modules
├── VISTA3D.py       # 3D segmentation backbone builder
├── dataset.py       # data loading, transforms, split helpers, orientation checks
├── loss.py          # side-aware loss definitions
├── train.py         # unified three-stage training entry
├── eval.py          # checkpoint evaluation script
├── requirements.txt # minimal runtime dependencies
└── README.md
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are using CUDA, install the PyTorch build that matches your local driver before installing the remaining packages.

## Data Preparation

The code expects a dataset root and a split file:

```text
dataset_root/
├── images/...
├── labels/...
└── splits.json
```

Key notes:

- `dataset.py` contains orientation consistency checks to keep bilateral semantics aligned.
- `--split-json` defaults to `<dataset-root>/splits.json`.
- `--split-ratio` supports settings such as `5`, `10`, `20`, `50`, and `100`.

## Training

### Stage I: anatomy-aware pretraining

```bash
python train.py \
  --stage unsup \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --gpu 0 \
  --amp
```

### Stage II: supervised semantic rectification

```bash
python train.py \
  --stage sup \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --init-checkpoint /path/to/stage1_checkpoint.pt \
  --gpu 0 \
  --amp
```

### Stage III: collaborative pseudo-label refinement

```bash
python train.py \
  --stage corefined \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --init-checkpoint /path/to/stage2_checkpoint.pt \
  --gpu 0 \
  --amp
```

## Evaluation

```bash
python eval.py \
  --checkpoint /path/to/checkpoint.pt \
  --dataset-root /path/to/dataset_root \
  --split-ratio 20 \
  --eval-key test \
  --gpu 0 \
  --amp
```

## Citation

If you use this repository in your work, please cite the APPR paper:

```bibtex
@article{appr2025,
  title   = {APPR: Anatomy-aware Pretraining and Pseudo-label Rectification for Semi-Supervised Maxillary Sinus Segmentation},
  author  = {Anonymous},
  journal = {Under preparation},
  year    = {2025}
}
```

## Acknowledgement

- Built with `PyTorch` and `MONAI` for 3D medical image segmentation research.
- README presentation is organized in a polished GitHub style inspired by `mileswyn/SAMIHS`.
