# Project Progress Report: Poisson TNRD Unrolling (MA3221 Project)

## 1) Initial Project Setup and Phase 1 Start

### Workspace and structure

- Root folder: `/home/sanku/Documents/mmip_project/project`
- Key existing resources:
  - `TNRD-Codes/` (baseline MATLAB code from FOE/TNRD)
  - `ref-Fast_and_Accurate_Poisson_Denoising_With_TND.pdf` (paper reference)
  - `.github/instructions/copilot.instructions.md` (project direction and phased plan)

### Phase 1 objective

- Identify critical MATLAB files for translation.
- Build minimal Python code to run a simple stage-wise denoising model.
- Test end-to-end with Poisson noise dataset.

### Important MATLAB files referenced

- `TNRD-Codes/TrainingCodes4denoising/JointTrainingFoEwithMFs/denoisingOneStepGMixMFs.m`
- `TNRD-Codes/TrainingCodes4denoising/JointTrainingFoEwithMFs/loss_specific_model.m`
- `TNRD-Codes/TrainingCodes4denoising/JointTrainingFoEwithMFs/loss_with_gradient_joint_stages.m`
- `TNRD-Codes/TrainingCodes4denoising/JointTrainingFoEwithMFs/JointTraining.m`

These files gave us the core update formula:

- gradient-step diffusion: `u - g` where `g` includes data fidelity and filter responses
- filter responses via convolution + learned nonlinear mapping
- stage-wise model training with learned filters and MFs.

## 2) Python Package and Module Creation

### Created files

1. `poisson_tnr/__init__.py`
2. `poisson_tnr/model.py`
3. `poisson_tnr/data.py`
4. `poisson_tnr/train.py`
5. `poisson_tnr/eval.py`
6. `run_demo.py`
7. `requirements.txt`

### Purpose of each file

- `poisson_tnr/model.py`: Defines Model A (baseline 1st order) and Model B (2nd-order telegraph) modules, stage blocks, and proximal operator.
- `poisson_tnr/data.py`: Defines `PoissonDataset` and `add_poisson_noise()` for image patches.
- `poisson_tnr/train.py`: Defines stage-wise training logic with freezing/unfreezing stage parameters.
- `poisson_tnr/eval.py`: Defines PSNR and SSIM functions for evaluation.
- `run_demo.py`: Demo driver that loads data, trains Model A stage-wise, and runs Model B forward pass.
- `requirements.txt`: Python dependencies list.

### Key design decisions

- Use PyTorch for differentiable unrolled model.
- Implement `TNRDStageA` block with learnable convolution filters + pointwise nonlinear network.
- Implement Poisson proximal from Eq. 14: closed-form update based on solving quadratic.
- Stage-wise train by enabling only current stage parameters and proximal/data weights.
- For Model B, implement telegraph update formula:
  - `u_{t+1} = (2-γ)u_t + (γ-1)u_{t-1} - data_weight*(u_t-y) - diffusion` then prox.
- `gamma` fixed for Model B using `requires_grad=False`.

## 3) Environment setup, venv, and package installation

### Virtual environment setup

- Created venv at `/home/sanku/Documents/mmip_project/project/.venv`.
- Activation command used in terminal: `. .venv/bin/activate`

### Dependencies installed

- `torch`
- `torchvision`
- `numpy`
- `Pillow`

### Notes

- We ran installation inside venv because of external-managed environment restrictions.

## 4) Debugging and results from running the demo

### First run issue

- Runtime error: `ValueError: num_samples should be a positive integer value, but got num_samples=0`
- Cause: dataset path pointed to MATLAB code directory, not actual images.
- Fix: changed `image_dir` in `run_demo.py` to `TNRD-Codes/TrainingCodes4denoising/FoETrainingSets180`.

### Successful run

- Run `python run_demo.py` after fix produced:
  - "Training stage 0 of ModelA for 1 epoch..."
  - `Loss: 0.0067`
  - `PSNR after demo run: 21.45`

### Successive updated demo with full stage-wise training

- After implementing stagewise training freeze logic and updated model formulas, run output:
  - Stage 0 loss: 0.006737
  - Stage 1 loss: 0.006850
  - Stage 2 loss: 0.006665
  - PSNR after ModelA demo run: 20.46
  - ModelB output PSNR: 20.40

## 5) Current model implementation details

### ModelA

- `num_stages` stage blocks, each with:
  - kernel convolution responses
  - pointwise phi network (1x1 conv + ReLU)
  - transpose reconstruction
  - learned diffusion step `step`
- Data fidelity and proximal updates inside each stage.

### ModelB

- Uses same stage blocks for diffusion.
- Telegrapher update uses `u_prev` and `u`:
  - `u_pred = (2-γ)u + (γ-1)u_prev - data_weight*(u-y) - diffusion`
  - proximal step applied to `u_pred`.

## 6) Files currently present in Python package

- `poisson_tnr/__init__.py`
- `poisson_tnr/data.py`
- `poisson_tnr/model.py`
- `poisson_tnr/train.py`
- `poisson_tnr/eval.py`
- `run_demo.py`
- `requirements.txt`
- `.venv/` (virtual env, not in git maybe)
- `PROJECT_PROGRESS.md` (this summary file)

## 7) What we still need before Phase 3 training

1. Finalize exact equations from ref paper Eq. 15 and Eq. 14 into model updates. (We already have a close approximation.)
2. Build robust dataset class to support full training and validation splits.
3. Develop stage-wise training schedule script with checkpointing and per-stage metrics.
4. Add evaluation / ablation script for different fixed `gamma` values.

## 8) What to run next (quick commands)

```bash
cd /home/sanku/Documents/mmip_project/project
. .venv/bin/activate
python run_demo.py
```

If this run is successful, we can proceed to Phase 3 full train/eval scripts.

---

## 9) Phase 3 Implementation (Data Pipeline, Training Loop, Checkpointing)

### Environment (this machine)

- venv created at `.venv/` using `python3 -m venv .venv`
- Packages installed: `torch==2.10.0`, `torchvision==0.25.0`, `numpy==2.4.3`, `Pillow==12.1.1`
- Activation: `source .venv/bin/activate`

### Changes to `poisson_tnr/data.py`

- **`add_poisson_noise(clean, peak)`** — renamed `scale` → `peak` for physical clarity. Poisson noise is signal-dependent: pixel values are scaled by `peak`, sampled from a Poisson distribution, then divided back. Higher `peak` = less noise (e.g. `peak=60` is moderate).
- **`train_val_split(image_dir, val_ratio=0.1, seed=42)`** — splits the 400 FoE images into 360 train / 40 val deterministically.
- **`PoissonDataset`** now accepts a list of image paths (or directory string). New parameters:
  - `patches_per_image` (default 8) — extracts multiple random patches per image per epoch → 360×8 = 2880 training patches/epoch instead of 360.
  - `augment` (default True) — random horizontal flip + 90° rotation for data augmentation.
- Added helper `_collect_image_paths()` and `_load_grayscale()`.

### Changes to `poisson_tnr/train.py`

- **`_freeze_all(model)`** — sets `requires_grad=False` for all parameters.
- **`_unfreeze_stage(model, stage_idx)`** — enables only the current stage block + shared params (`alpha`, `prox`).
- **`_train_one_epoch()`** — training loop for one epoch at a given stage index.
- **`_validate()`** — validation loop returning (val_loss, val_PSNR).
- **`stagewise_train()`** — legacy wrapper kept for backward compatibility with `run_demo.py`.
- **`train_model_stagewise()`** — full pipeline function:
  - Iterates over each stage index.
  - Freeze/unfreeze logic per stage.
  - Fresh Adam optimizer per stage (only trainable params).
  - Trains for `epochs_per_stage` epochs with per-epoch validation PSNR printed.
  - After each stage: freezes that stage permanently, saves checkpoint to `save_dir/model_name_stageN.pt`.
  - After all stages: saves final model + training history as JSON.
  - Returns history dict `{train_losses, val_losses, val_psnrs}` (list of lists, outer = stage).

### New file: `train_pipeline.py`

- Full CLI training script accepting all hyperparameters as arguments.
- Key arguments: `--model A|B`, `--stages`, `--epochs`, `--lr`, `--batch`, `--patch`, `--peak`, `--patches_per_image`, `--val_ratio`, `--gamma`, `--workers`, `--save_dir`, `--seed`.
- Builds train/val `DataLoader` from `train_val_split`.
- Calls `train_model_stagewise()`.
- Runs final PSNR + SSIM evaluation on full validation set after training.
- Saves training history JSON.

### Updated `poisson_tnr/__init__.py`

- Added exports: `train_val_split`, `train_model_stagewise`.

### Updated `run_demo.py`

- Updated to use new `PoissonDataset(train_paths, ..., peak=60.0)` API.

### Complete test results (3 stages × 3 epochs each, `--batch 8 --patches_per_image 2 --workers 0`)

**Model A (1st-order baseline)** — 1036 trainable params

| Stage | Epoch | Train Loss | Val Loss | Val PSNR     |
| ----- | ----- | ---------- | -------- | ------------ |
| 1     | 1     | 0.006656   | 0.007190 | 21.63 dB     |
| 1     | 2     | 0.006784   | 0.007149 | 21.63 dB     |
| 1     | 3     | 0.006707   | 0.006974 | 21.75 dB     |
| 2     | 1     | 0.006648   | 0.006824 | 21.83 dB     |
| 2     | 2     | 0.004281   | 0.003267 | 25.12 dB     |
| 2     | 3     | 0.003029   | 0.003118 | 25.31 dB     |
| 3     | 1     | 0.003016   | 0.003125 | 25.35 dB     |
| 3     | 2     | 0.002963   | 0.002974 | 25.60 dB     |
| 3     | 3     | 0.002863   | 0.002850 | **25.86 dB** |

**Final val PSNR: 25.85 dB | Final val SSIM: 0.6469**

**Model B (2nd-order telegraph, γ=0.5)** — 1036 trainable params, gamma frozen

| Stage | Epoch | Train Loss | Val Loss | Val PSNR     |
| ----- | ----- | ---------- | -------- | ------------ |
| 1     | 1     | 0.006769   | 0.007208 | 21.60 dB     |
| 1     | 2     | 0.006750   | 0.007125 | 21.68 dB     |
| 1     | 3     | 0.006574   | 0.006857 | 21.82 dB     |
| 2     | 1     | 0.006579   | 0.006590 | 21.97 dB     |
| 2     | 2     | 0.004195   | 0.003339 | 25.02 dB     |
| 2     | 3     | 0.002975   | 0.003094 | 25.36 dB     |
| 3     | 1     | 0.003073   | 0.003190 | 25.22 dB     |
| 3     | 2     | 0.003045   | 0.003162 | 25.27 dB     |
| 3     | 3     | 0.003030   | 0.003181 | **25.24 dB** |

**Final val PSNR: 25.23 dB | Final val SSIM: 0.5984**

Note: these are quick smoke-test results (only 3 epochs/stage). Full training uses `--epochs 30`.

- `gamma.requires_grad = False` confirmed for Model B (1037 total params, 1036 trainable).
- All checkpoints saved to `checkpoints/` — per-stage `.pt` files + `_final.pt` + `_history.json`.

### Commands to run full training

```bash
cd SecondOrder-Poisson-Denoising
source .venv/bin/activate

# Train Model A
python train_pipeline.py --model A --stages 5 --epochs 30 --peak 60

# Train Model B (gamma=0.5)
python train_pipeline.py --model B --stages 5 --epochs 30 --peak 60 --gamma 0.5
```

## 10) What to do next (Phase 4)

- Write ablation study script that loops over `gamma` values (e.g. 0.1, 0.5, 0.9) for Model B.
- Compare each `gamma` against Model A baseline using PSNR and SSIM.
- Produce a summary table showing the effect of the $u_{tt}$ momentum term.
