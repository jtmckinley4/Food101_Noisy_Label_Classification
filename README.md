# Noise-Robust Food Image Classification

Fine-grained food image classification on Food-101 using EfficientNetV2-S transfer learning. This project investigates how label noise in training data affects model performance on a clean test set, and which noise-robust training strategies best recover accuracy.

**Authors:** Julian McKinley · Cristhian Cruz  
**Course:** CAP5636 — Advanced Artificial Intelligence for Data Analytics, University of Central Florida

---

## Overview

The [Food-101](https://www.kaggle.com/datasets/kmader/food41) dataset contains 101,000 food images across 101 categories. Training images were collected automatically without human review (introducing label noise), while test images are manually verified and clean. This asymmetry makes it an ideal benchmark for studying noise-robust training.

We compare five experimental configurations:

| ID | Backbone | Loss Function | Purpose |
|---|---|---|---|
| E0 | ResNet-50 | Cross-Entropy | Architecture baseline |
| E1 | EfficientNetV2-S | Cross-Entropy | Noise-method baseline |
| E2 | EfficientNetV2-S | Label Smoothing (ε=0.1) | Soft-target robustness |
| E3 | EfficientNetV2-S | Symmetric CE (α=0.1, β=1.0) | Loss-level robustness |
| E4 | EfficientNetV2-S | CutMix (α=1.0) | Augmentation-level robustness |

---

## Results

| Configuration | Top-1 (%) | Top-5 (%) |
|---|---|---|
| E0 — ResNet-50 + CE | — | — |
| E1 — EfficientNetV2-S + CE | — | — |
| E2 — EfficientNetV2-S + Label Smoothing | — | — |
| E3 — EfficientNetV2-S + SCE | — | — |
| E4 — EfficientNetV2-S + CutMix | — | — |

*Results to be filled in after training.*

---

## Project Structure
---

## Setup

### Requirements

```bash
pip install torch torchvision matplotlib seaborn scikit-learn tqdm pandas
```

For the Jupyter notebook, also install:

```bash
pip install notebook
```

> **Windows users:** Replace `run_all.sh` with individual `python train.py` commands in PowerShell.

> **PyTorch with GPU (Windows/Linux):** Install the CUDA-enabled build from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the default pip wheel.

---

## Usage

### Option A — Jupyter Notebook (recommended for exploration)

Open `Food101_NoisyLabel_Classification.ipynb` in VS Code or Jupyter. The notebook runs end-to-end with `FAST_MODE = True` by default (~15 min on GPU, 10 classes). Set `FAST_MODE = False` for the full 101-class experiment.

### Option B — Command line (individual experiment)

```bash
# EfficientNetV2-S + label smoothing (primary config)
python train.py --backbone efficientnet_v2_s --loss label_smoothing

# ResNet-50 baseline
python train.py --backbone resnet50 --loss ce --warmup_epochs 0

# Compute-constrained fallback (EfficientNet-B0)
python train.py --backbone efficientnet_b0 --loss sce --epochs 15
```

### Option C — Run all experiments

```bash
chmod +x run_all.sh
./run_all.sh
```

This trains all 5 configurations sequentially, then generates comparison plots and a summary table in `./outputs/`.

### Evaluation + Grad-CAM

```bash
python evaluate.py \
  --checkpoint outputs/efficientnet_v2_s_label_smoothing_best.pth \
  --backbone efficientnet_v2_s \
  --loss label_smoothing \
  --gradcam
```

---

## Training Details

- **Optimizer:** AdamW (lr = 1e-4, weight_decay = 1e-2)
- **Schedule:** Cosine annealing over fine-tune epochs
- **Phase 1 — Warm-up:** 5 epochs, head only, lr = 1e-3
- **Phase 2 — Fine-tune:** 30 epochs, top 3 EfficientNetV2-S blocks + head, lr = 1e-4
- **Augmentation:** RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- **Image size:** 224 × 224
- **Batch size:** 32

---

## References

1. Bossard et al. (2014). Food-101 — Mining discriminative components with random forests. ECCV.
2. Tan & Le (2019). EfficientNet: Rethinking model scaling for CNNs. ICML.
3. Tan & Le (2021). EfficientNetV2: Smaller models and faster training. ICML.
4. Wang et al. (2019). Symmetric Cross Entropy for Robust Learning With Noisy Labels. ICCV.
5. Yun et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers. ICCV.
6. Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks. ICCV.
