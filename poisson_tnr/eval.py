import torch
import torch.nn.functional as F


def compute_psnr(output, target, max_val=1.0):
    mse = F.mse_loss(output, target, reduction='none')
    mse = mse.mean(dim=[1,2,3])
    psnr = 10.0 * torch.log10(max_val * max_val / (mse + 1e-12))
    return psnr.mean().item()


def compute_ssim(output, target, max_val=1.0):
    # simple SSIM approximation using torchmetrics is preferred, but implement baseline
    C1 = (0.01 * max_val) ** 2
    C2 = (0.03 * max_val) ** 2
    mu_x = F.avg_pool2d(output, 3, 1, 1)
    mu_y = F.avg_pool2d(target, 3, 1, 1)
    sigma_x = F.avg_pool2d(output * output, 3, 1, 1) - mu_x * mu_x
    sigma_y = F.avg_pool2d(target * target, 3, 1, 1) - mu_y * mu_y
    sigma_xy = F.avg_pool2d(output * target, 3, 1, 1) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x * mu_x + mu_y * mu_y + C1) * (sigma_x + sigma_y + C2))
    return ssim_map.mean().item()
