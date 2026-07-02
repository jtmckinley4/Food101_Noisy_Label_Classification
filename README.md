# Food101_Noisy_Label_Classification
Fine-grained food image classification on Food-101 using EfficientNet-B4 transfer learning. Compares noise-robust training strategies (label smoothing, SCE, CutMix) against a standard cross-entropy baseline on intentionally noisy training labels.

# Noise-Robust Food Image Classification

Fine-grained food image classification on Food-101 using EfficientNet-B4 transfer learning. This project investigates how label noise in training data affects model performance on a clean test set, and which noise-robust training strategies best recover accuracy.

**Authors:** Julian McKinley · Cristhian Cruz  
**Course:** CAP5636 — Advanced Artificial Intelligence for Data Analytics, University of Central Florida

---

## Overview

The [Food-101](https://www.kaggle.com/datasets/kmader/food41) dataset contains 101,000 food images across 101 categories. Training images were collected automatically without human review (introducing label noise), while test images are manually verified and clean. This asymmetry makes it an ideal benchmark for studying noise-robust training.

We compare five experimental configurations:

| ID | Backbone | Loss Function | Purpose |
|---|---|---|---|
| E0 | ResNet-50 | Cross-Entropy | Architecture baseline |
| E1 | EfficientNet-B4 | Cross-Entropy | Noise-method baseline |
| E2 | EfficientNet-B4 | Label Smoothing (ε=0.1) | Soft-target robustness |
| E3 | EfficientNet-B4 | Symmetric CE (α=0.1, β=1.0) | Loss-level robustness |
| E4 | EfficientNet-B4 | CutMix (α=1.0) | Augmentation-level robustness |

---

## Results

| Configuration | Top-1 (%) | Top-5 (%) |
|---|---|---|
| E0 — ResNet-50 + CE | — | — |
| E1 — EfficientNet-B4 + CE | — | — |
| E2 — EfficientNet-B4 + Label Smoothing | — | — |
| E3 — EfficientNet-B4 + SCE | — | — |
| E4 — EfficientNet-B4 + CutMix | — | — |

*Results to be filled in after training.*

---

## Project Structure

food101_classifier/

├── train.py              # Main training script (two-phase fine-tuning)

├── evaluate.py           # Top-1/5 accuracy, confusion matrix, Grad-CAM

├── losses.py             # SCE loss and CutMix augmentation helpers

├── model.py              # Backbone factory + freeze/unfreeze logic

├── compare_results.py    # Aggregate results into plots and summary table

├── run_all.sh            # Run all 5 experiments sequentially (Mac/Linux)

├── Food101_NoisyLabel_Classification.ipynb  # Interactive Jupyter notebook

└── outputs/              # Checkpoints and result JSON files (auto-created)

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

> **Windows users:** Replace `run_all.sh` with `python run_all.py` (coming soon), or run each `train.py` command individually in PowerShell.

> **PyTorch with GPU (Windows/Linux):** Install the CUDA-enabled build from [pytorch.org](https://pytorch.org/get-started/locally/) instead of the default pip wheel.

---

## Usage

### Option A — Jupyter Notebook (recommended for exploration)

Open `Food101_NoisyLabel_Classification.ipynb` in VS Code or Jupyter. The notebook runs end-to-end with `FAST_MODE = True` by default (~15 min on GPU, 10 classes). Set `FAST_MODE = False` for the full 101-class experiment.

### Option B — Command line (individual experiment)

```bash
# EfficientNet-B4 + label smoothing
python train.py --backbone efficientnet_b4 --loss label_smoothing

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
  --checkpoint outputs/efficientnet_b4_label_smoothing_best.pth \
  --backbone efficientnet_b4 \
  --loss label_smoothing \
  --gradcam
```

---

## Training Details

- **Optimizer:** AdamW (lr = 1e-4, weight_decay = 1e-2)
- **Schedule:** Cosine annealing over fine-tune epochs
- **Phase 1 — Warm-up:** 5 epochs, head only, lr = 1e-3
- **Phase 2 — Fine-tune:** 30 epochs, top 3 EfficientNet blocks + head, lr = 1e-4
- **Augmentation:** RandomResizedCrop, RandomHorizontalFlip, ColorJitter
- **Image size:** 224 × 224
- **Batch size:** 32

---

## References

1. Bossard et al. (2014). Food-101 — Mining discriminative components with random forests. ECCV.
2. Tan & Le (2019). EfficientNet: Rethinking model scaling for CNNs. ICML.
3. Wang et al. (2019). Symmetric Cross Entropy for Robust Learning With Noisy Labels. ICCV.
4. Yun et al. (2019). CutMix: Regularization Strategy to Train Strong Classifiers. ICCV.
5. Selvaraju et al. (2017). Grad-CAM: Visual explanations from deep networks. ICCV.
