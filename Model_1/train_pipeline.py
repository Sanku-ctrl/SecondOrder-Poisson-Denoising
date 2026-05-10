"""Phase 3: Full stage-wise training pipeline for ModelA and ModelB.

Usage examples:
    # Train Model A (baseline, 1st-order PDE)
    python train_pipeline.py --model A --stages 5 --epochs 30 --peak 60

    # Train Model B (telegraph equation, 2nd-order PDE)
    python train_pipeline.py --model B --stages 5 --epochs 30 --peak 60 --gamma 0.5

    # Quick smoke-test (small run)
    python train_pipeline.py --model A --stages 3 --epochs 2 --batch 8
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

from poisson_tnr.data import PoissonDataset, train_val_split
from poisson_tnr.model import ModelA, ModelB
from poisson_tnr.train import train_model_stagewise
from poisson_tnr.eval import compute_psnr, compute_ssim


def parse_args():
    p = argparse.ArgumentParser(description="Stage-wise Poisson denoising training")
    p.add_argument("--image_dir", type=str,
                   default="TNRD-Codes/TrainingCodes4denoising/FoETrainingSets180",
                   help="Path to training image folder")
    p.add_argument("--model", type=str, choices=["A", "B"], default="A",
                   help="Model variant: A (1st-order baseline) or B (2nd-order telegraph)")
    p.add_argument("--stages", type=int, default=5,
                   help="Number of unrolled stages")
    p.add_argument("--filters", type=int, default=8,
                   help="Number of convolution filters per stage")
    p.add_argument("--kernel", type=int, default=5,
                   help="Convolution kernel size")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="Fixed damping parameter for Model B (ignored for Model A)")
    p.add_argument("--epochs", type=int, default=30,
                   help="Training epochs per stage")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate (Adam)")
    p.add_argument("--batch", type=int, default=4,
                   help="Batch size")
    p.add_argument("--patch", type=int, default=64,
                   help="Training patch size")
    p.add_argument("--peak", type=float, default=60.0,
                   help="Poisson noise peak (higher = less noise)")
    p.add_argument("--patches_per_image", type=int, default=8,
                   help="Random patches extracted per image per epoch")
    p.add_argument("--val_ratio", type=float, default=0.1,
                   help="Fraction of images reserved for validation")
    p.add_argument("--workers", type=int, default=2,
                   help="DataLoader workers")
    p.add_argument("--save_dir", type=str, default="checkpoints",
                   help="Directory for checkpoint files")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducibility")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ----- Data -----
    train_paths, val_paths = train_val_split(
        args.image_dir, val_ratio=args.val_ratio, seed=args.seed
    )
    print(f"Images: {len(train_paths)} train, {len(val_paths)} val")

    train_ds = PoissonDataset(
        train_paths, patch_size=args.patch, peak=args.peak,
        patches_per_image=args.patches_per_image, augment=True,
    )
    val_ds = PoissonDataset(
        val_paths, patch_size=args.patch, peak=args.peak,
        patches_per_image=4, augment=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
    )
    print(f"Train patches/epoch: {len(train_ds)}, Val patches/epoch: {len(val_ds)}")

    # ----- Model -----
    if args.model == "A":
        model = ModelA(
            num_stages=args.stages, num_filters=args.filters,
            kernel_size=args.kernel,
        ).to(device)
        model_name = f"modelA_s{args.stages}_p{int(args.peak)}"
    else:
        model = ModelB(
            num_stages=args.stages, num_filters=args.filters,
            kernel_size=args.kernel, gamma=args.gamma,
        ).to(device)
        model_name = f"modelB_s{args.stages}_g{args.gamma}_p{int(args.peak)}"

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model {args.model}: {total_params} params ({trainable_params} trainable)")

    # ----- Train stage-wise -----
    history = train_model_stagewise(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        num_stages=args.stages,
        epochs_per_stage=args.epochs,
        lr=args.lr,
        save_dir=args.save_dir,
        model_name=model_name,
    )

    # ----- Final evaluation on full validation set -----
    print(f"\n{'='*60}")
    print("  Final Evaluation (all stages, full val set)")
    print(f"{'='*60}")
    model.eval()
    all_psnr, all_ssim, count = 0.0, 0.0, 0
    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy, noisy)
            all_psnr += compute_psnr(output, clean) * noisy.size(0)
            all_ssim += compute_ssim(output, clean) * noisy.size(0)
            count += noisy.size(0)
    avg_psnr = all_psnr / max(count, 1)
    avg_ssim = all_ssim / max(count, 1)
    print(f"  Val PSNR: {avg_psnr:.2f} dB")
    print(f"  Val SSIM: {avg_ssim:.4f}")

    # ----- Save training history -----
    hist_path = os.path.join(args.save_dir, f"{model_name}_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {hist_path}")


if __name__ == "__main__":
    main()
