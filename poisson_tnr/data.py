import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def add_poisson_noise(clean, peak):
    """Add signal-dependent Poisson noise to image tensor in [0,1].

    Poisson noise is signal-dependent: Var(noisy) ~ clean/peak.
    Higher peak = less noise (cleaner). Typical values: 1, 4, 8, 20, 60.

    Args:
        clean: tensor [B,C,H,W] or [C,H,W] in [0, 1].
        peak: peak value controlling noise severity.

    Returns:
        noisy tensor in [0, 1], same shape and device as input.
    """
    x_np = clean.detach().cpu().numpy()
    scaled = np.clip(x_np * peak, 0.0, None)
    noisy = np.random.poisson(scaled).astype(np.float64) / peak
    noisy = np.clip(noisy, 0.0, 1.0).astype(np.float32)
    return torch.from_numpy(noisy).to(clean.device)


def _collect_image_paths(image_dir):
    """Recursively collect image file paths from a directory."""
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    paths = []
    for root, _, files in os.walk(image_dir):
        for f in sorted(files):
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return paths


def _load_grayscale(path):
    """Load an image as float32 grayscale array in [0, 1]."""
    img = Image.open(path).convert("L")
    return np.array(img).astype(np.float32) / 255.0


def train_val_split(image_dir, val_ratio=0.1, seed=42):
    """Split image paths into train and validation sets.

    Returns:
        (train_paths, val_paths)
    """
    paths = _collect_image_paths(image_dir)
    rng = np.random.RandomState(seed)
    rng.shuffle(paths)
    n_val = max(1, int(len(paths) * val_ratio))
    return paths[n_val:], paths[:n_val]


class PoissonDataset(Dataset):
    """Dataset that yields (noisy, clean) patch pairs with Poisson noise.

    Args:
        image_paths: list of image file paths.
        patch_size: spatial size of extracted patches.
        peak: Poisson peak parameter (higher = less noise).
        patches_per_image: number of random patches to extract per image
            per epoch. Total dataset length = len(image_paths) * patches_per_image.
        augment: if True, apply random flips and 90-degree rotations.
    """

    def __init__(self, image_paths, patch_size=64, peak=60.0,
                 patches_per_image=8, augment=True):
        super().__init__()
        if isinstance(image_paths, str):
            # Allow passing a directory path directly for convenience
            image_paths = _collect_image_paths(image_paths)
        self.image_paths = image_paths
        self.patch_size = patch_size
        self.peak = peak
        self.patches_per_image = patches_per_image
        self.augment = augment

    def __len__(self):
        return len(self.image_paths) * self.patches_per_image

    def __getitem__(self, idx):
        img_idx = idx // self.patches_per_image
        path = self.image_paths[img_idx]
        img = _load_grayscale(path)
        h, w = img.shape

        # Pad if image is smaller than patch_size
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(self.patch_size - h, 0)
            pad_w = max(self.patch_size - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = img.shape

        # Random crop
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        patch = img[top:top + self.patch_size, left:left + self.patch_size]

        # Data augmentation: random flip + 90-degree rotation
        if self.augment:
            if np.random.rand() < 0.5:
                patch = np.flip(patch, axis=1)
            k = np.random.randint(0, 4)
            patch = np.rot90(patch, k)

        patch = np.ascontiguousarray(patch)
        clean = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        noisy = add_poisson_noise(clean, peak=self.peak)
        return noisy, clean
