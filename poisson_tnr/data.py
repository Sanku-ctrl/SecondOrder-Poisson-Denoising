import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np


def add_poisson_noise(x, scale=1.0):
    """Add signal-dependent Poisson noise to image tensor x in [0,1]."""
    # x [B,C,H,W] or [C,H,W]
    x_np = x.detach().cpu().numpy()
    noisy = np.random.poisson(x_np * scale) / float(scale)
    noisy = np.clip(noisy, 0.0, 1.0)
    return torch.from_numpy(noisy).type_as(x)


class PoissonDataset(Dataset):
    def __init__(self, image_dir, patch_size=128, transform=None, scale=1.0):
        super().__init__()
        self.image_paths = []
        for root, dirs, files in os.walk(image_dir):
            for f in files:
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                    self.image_paths.append(os.path.join(root, f))
        self.patch_size = patch_size
        self.transform = transform
        self.scale = scale

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("L")
        img = np.array(img).astype(np.float32) / 255.0
        h, w = img.shape
        if h < self.patch_size or w < self.patch_size:
            pad_h = max(self.patch_size - h, 0)
            pad_w = max(self.patch_size - w, 0)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect")
            h, w = img.shape
        top = np.random.randint(0, h - self.patch_size + 1)
        left = np.random.randint(0, w - self.patch_size + 1)
        patch = img[top:top+self.patch_size, left:left+self.patch_size]
        clean = torch.tensor(patch, dtype=torch.float32).unsqueeze(0)
        noisy = add_poisson_noise(clean, scale=self.scale)
        return noisy, clean
