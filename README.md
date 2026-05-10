# Second-Order Trainable Nonlinear Reaction Diffusion for Poisson Image Denoising

This repository contains the implementation of a second-order diffusion framework designed for denoising images corrupted by Poisson noise. The research extends the first-order Trainable Nonlinear Reaction Diffusion (TNRD) model by introducing a second-order Telegraph Diffusion formulation with damping.

---

## Overview

Poisson noise is signal-dependent, meaning the noise variance equals the underlying intensity value. This is common in medical imaging, astronomical photography, and low-light conditions. While standard TNRD models rely on first-order PDEs, they may lack the "memory" required to propagate complex structural information effectively.

This project implements a second-order model inspired by wave dynamics, introducing a velocity state variable. This momentum-based approach improves information exchange across unrolled stages and accelerates convergence.

---

## Model Architectures

### Model 1: First-Order TNRD Baseline
A replication of the TRDPD model (Feng et al.). It serves as the performance benchmark, utilizing a first-order diffusion process unrolled into a feed-forward network.

---

### Model 2: Second-Order Telegraph Model (Joint-Training)
This model introduces a velocity state \( v_t \) and a damping coefficient \( \gamma \). It utilizes intermediate supervision, where loss is accumulated at every stage output rather than just the final layer. This ensures stable gradient propagation to early layers during end-to-end training.

---

### Model 3: Second-Order Telegraph Model (Greedy + Joint)
This variant employs a two-phase training strategy:

- **Greedy Pre-training:** Each stage is trained sequentially for 3 epochs while previous stages are frozen  
- **Joint Fine-tuning:** All stages are unfrozen and trained simultaneously for 30 epochs  

---

## Key Technical Specifications

- **Proximal Operator:** Closed-form solution used for stability in low-light regions  
- **Damping Coefficient (\( \gamma \))**  
  - 0.7 for Model 2  
  - 0.5 for Model 3  
  - As \( \gamma \to 1 \), behavior becomes first-order  

- **Learned Parameters per stage:**  
  - Filters (\( k \))  
  - Influence functions (\( \phi \)) via RBFs  
  - Proximal parameter (\( \lambda \))  

- **Configuration:**  
  4–8 stages, 7×7 filters  

---

## Results

| Method                    | Peak=1 | Peak=4 | Peak=40 |
|--------------------------|--------|--------|---------|
| BM3D                     | 21.01  | 23.54  | 28.20   |
| Model 2 (Joint + IS)     | 18.80  | 22.66  | 27.80   |
| Model 3 (Greedy + Joint) | 19.58  | 22.90  | 27.88   |
| TRDPD (Reference)        | 21.60  | 23.84  | 28.50   |

Second-order models use fewer filters but show improved structural propagation, especially at higher photon counts.

---

## Installation

### Requirements
- Python 3.8+
- PyTorch
- Torch
- Torchvision
- Pillow
- NumPy

### Run location
Model 1 code and data folders are now under `Model_1/`:
- `Model_1/poisson_tnr/`
- `Model_1/TNRD-Codes/`
- `Model_1/checkpoints/`
- `Model_1/run_demo.py`
- `Model_1/train_pipeline.py`

To keep existing relative dataset/checkpoint paths working, run commands from inside `Model_1/`:

```bash
cd Model_1
```

---

## Training

```bash
python train.py --model model3 --peak 4 --phases greedy joint
```
## Evaluation
```bash
python evaluate.py --checkpoint path/to/model.pth --input path/to/image.png --peak 4
```
