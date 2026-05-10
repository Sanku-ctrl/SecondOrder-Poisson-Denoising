import os
import torch
from .eval import compute_psnr


# ---------------------------------------------------------------------------
# Freeze / unfreeze helpers
# ---------------------------------------------------------------------------

def _freeze_all(model):
    """Set requires_grad=False for every parameter."""
    for p in model.parameters():
        p.requires_grad = False


def _unfreeze_stage(model, stage_idx):
    """Enable gradients only for the given stage block plus shared params."""
    _freeze_all(model)
    # Shared params that should always be trainable during any stage
    for p in model.prox.parameters():
        p.requires_grad = True
    if hasattr(model, 'alpha'):
        model.alpha.requires_grad = True
    # Current stage's filters + activations
    if 0 <= stage_idx < len(model.stages):
        for p in model.stages[stage_idx].parameters():
            p.requires_grad = True


# ---------------------------------------------------------------------------
# Single-epoch train / validate
# ---------------------------------------------------------------------------

def _train_one_epoch(model, loader, optimizer, criterion, device, stage_idx):
    model.train()
    total_loss = 0.0
    count = 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy, noisy, training_stage=stage_idx)
        loss = criterion(output, clean)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * noisy.size(0)
        count += noisy.size(0)
    return total_loss / max(count, 1)


@torch.no_grad()
def _validate(model, loader, criterion, device, stage_idx):
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0
    for noisy, clean in loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy, noisy, training_stage=stage_idx)
        loss = criterion(output, clean)
        psnr = compute_psnr(output, clean)
        total_loss += loss.item() * noisy.size(0)
        total_psnr += psnr * noisy.size(0)
        count += noisy.size(0)
    return total_loss / max(count, 1), total_psnr / max(count, 1)


# ---------------------------------------------------------------------------
# Legacy wrapper (keeps run_demo.py working)
# ---------------------------------------------------------------------------

def stagewise_train(model, dataloader, optimizer, criterion, device, stage_idx):
    """Train one stage for one epoch (backward-compatible wrapper)."""
    _unfreeze_stage(model, stage_idx)
    return _train_one_epoch(model, dataloader, optimizer, criterion, device, stage_idx)


# ---------------------------------------------------------------------------
# Full stage-wise training pipeline
# ---------------------------------------------------------------------------

def train_model_stagewise(
    model,
    train_loader,
    val_loader,
    device,
    num_stages,
    epochs_per_stage=30,
    lr=1e-3,
    save_dir="checkpoints",
    model_name="model",
):
    """Train a model (ModelA or ModelB) stage-by-stage.

    For each stage:
      1. Freeze all parameters except current stage + shared (alpha, prox).
      2. Create a fresh Adam optimizer for the trainable parameters.
      3. Train for ``epochs_per_stage`` epochs, tracking validation PSNR.
      4. Save a checkpoint after the stage finishes.

    Args:
        model: ModelA or ModelB instance (already on ``device``).
        train_loader: DataLoader yielding (noisy, clean) pairs.
        val_loader: DataLoader for validation (can be smaller).
        device: torch device.
        num_stages: how many stages to train (should match model.num_stages).
        epochs_per_stage: training epochs for each individual stage.
        lr: learning rate for Adam.
        save_dir: directory for checkpoint .pt files.
        model_name: prefix for checkpoint filenames.

    Returns:
        dict with keys 'train_losses', 'val_losses', 'val_psnrs'
        (each a list of lists, outer index = stage, inner = epoch).
    """
    os.makedirs(save_dir, exist_ok=True)
    criterion = torch.nn.MSELoss()

    history = {"train_losses": [], "val_losses": [], "val_psnrs": []}

    for stage_idx in range(num_stages):
        print(f"\n{'='*60}")
        print(f"  Stage {stage_idx + 1}/{num_stages}")
        print(f"{'='*60}")

        _unfreeze_stage(model, stage_idx)

        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable, lr=lr)

        stage_train, stage_val, stage_psnr = [], [], []

        for epoch in range(epochs_per_stage):
            train_loss = _train_one_epoch(
                model, train_loader, optimizer, criterion, device, stage_idx
            )
            val_loss, val_psnr = _validate(
                model, val_loader, criterion, device, stage_idx
            )

            stage_train.append(train_loss)
            stage_val.append(val_loss)
            stage_psnr.append(val_psnr)

            print(
                f"  Epoch {epoch + 1:3d}/{epochs_per_stage}"
                f"  train_loss={train_loss:.6f}"
                f"  val_loss={val_loss:.6f}"
                f"  val_PSNR={val_psnr:.2f} dB"
            )

        history["train_losses"].append(stage_train)
        history["val_losses"].append(stage_val)
        history["val_psnrs"].append(stage_psnr)

        # Freeze finished stage permanently before moving on
        for p in model.stages[stage_idx].parameters():
            p.requires_grad = False

        # Checkpoint
        ckpt_path = os.path.join(save_dir, f"{model_name}_stage{stage_idx}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"  -> Saved checkpoint: {ckpt_path}")

    # Save final model
    final_path = os.path.join(save_dir, f"{model_name}_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\nFinal model saved to {final_path}")

    return history
