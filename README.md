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
