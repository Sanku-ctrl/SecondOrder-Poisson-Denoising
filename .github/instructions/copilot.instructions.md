---
description: Instructions for MA3221 Project 10 - PDE-based Image Processing (Telegraph Equation for Poisson Noise)
applyTo: "**/*.{py,m}"
---

Provide project context and coding guidelines that AI should follow when generating code, answering questions, or reviewing changes.

# Project Context

This workspace is for an academic project (MA3221 - Project 10): **"Develop a learnable architecture for $u_{tt} + \gamma u_t = \text{div}(c(|\nabla u|)\nabla u)$ for Poisson noise removal. Fix $\gamma$ and learn other parameters stage-wise."**

We are using a concept called "Algorithm Unrolling," where each layer of a PyTorch neural network corresponds to a single time-step ($t$) of a discretized Partial Differential Equation (PDE).

1. **The Reference Paper:** `ref-Fast_and_Accurate_Poisson_Denoising_With_TND.pdf`.
2. **Key Equations from Paper:**
   - **Equation 15:** The forward pass for a single stage of their network (1st-order PDE).
   - **Equation 14:** The Proximal Operator formula handling the Poisson noise reaction term. Standard gradient descent fails due to division by $u$, so this proximal operator is strictly required for stability.
3. **Our Novelty (Project Goal):** Upgrading the paper's 1st-order PDE ($u_t$) to a 2nd-order Telegraph Equation ($u_{tt} + \gamma u_t$) to introduce wave momentum.

# Coding Guidelines & Phased Workflow

When assisting me, **do not generate the entire codebase at once**. Follow these strict phases and wait for my prompts to move between them. My workspace currently has ~400 images and a large folder of provided baseline MATLAB (`.m`) code which we must translate to Python/PyTorch.

### Phase 1: MATLAB Translation & Navigation

- Help me identify which `.m` files are critical (e.g., data loading, baseline TNRD filter generation) and which can be ignored.
- Translate the necessary mathematical logic from MATLAB into standard Python/NumPy/PyTorch.

### Phase 2: Core PyTorch Architecture (Build TWO Models)

**Crucial:** We are building TWO models. Do not skip to Model B until Model A is functional.

- **Model A (The Baseline):** Build the exact 1st-order model from the 2018 paper based on **Equation 15** and the Proximal Operator (**Equation 14**). Implement learnable filters ($k_i$) and activation functions ($\phi_i$).
- **Model B (The Project Goal - 2nd Order PDE):** Duplicate Model A and upgrade it to our target equation. Apply finite differences to $u_{tt}$ and $u_t$ so that layer $t+1$ depends on BOTH layer $t$ and layer $t-1$:
  `u_{t+1} ≈ (2-γ)u_t + (γ-1)u_{t-1} + Diffusion(u_t) - Reaction(using Eq 14)`
- **Strict Constraint:** In Model B, ensure the parameter $\gamma$ is a fixed hyperparameter (`requires_grad=False`), while filters and activation functions remain learnable.

### Phase 3: Data and Training Pipeline

- Write PyTorch `Dataset` and `DataLoader` classes using the ~400 images.
- Write a robust Python function to inject **Poisson Noise** (ensure proper scaling as Poisson is signal-dependent).
- Write the **Stage-wise Training Loop**. We must train Stage 1, freeze it, train Stage 2, freeze it, etc. Do NOT write standard end-to-end training loops initially.

### Phase 4: Ablation Studies and Evaluation

- Implement evaluation metrics (PSNR and SSIM).
- Write an evaluation script for an **Ablation Study**. We will loop through different fixed values of $\gamma$ (e.g., $\gamma = 0.1, 0.5, 0.9$) to prove mathematically why adding the $u_{tt}$ momentum term improves upon Model A.

**Whenever I ask a question, refer back to these guidelines to ensure our code aligns perfectly with the two-model approach, the $u_{tt} + \gamma u_t$ Telegraph equation, and the stage-wise training constraint.**
