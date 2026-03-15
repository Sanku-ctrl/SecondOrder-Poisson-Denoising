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

### Note
You can now continue with Phase 3. I can implement the stage-wise training driver and ablation evaluation next, using this exact environment and models.
