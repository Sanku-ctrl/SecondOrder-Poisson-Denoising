import torch
import torch.nn as nn
import torch.nn.functional as F

class TNRDStageA(nn.Module):
    def __init__(self, num_filters=8, kernel_size=5):
        super().__init__()
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.kernels = nn.Parameter(torch.randn(num_filters, 1, kernel_size, kernel_size) * 0.01)
        self.phi = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=1, bias=True),
        )
        self.step = nn.Parameter(torch.tensor(0.2))

    def forward(self, u):
        # u: [B,1,H,W]
        # Diffusion term: sum_i K_i^T phi_i(K_i u)
        Ku = F.conv2d(u, self.kernels, padding=self.kernel_size // 2)
        phi = self.phi(Ku)
        # backproject
        diff = F.conv_transpose2d(phi, self.kernels, padding=self.kernel_size // 2)
        return self.step * diff

class ProximalPoisson(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, y, alpha):
        # proximal operator for Poisson data term: min_u 0.5||u-v||^2 + alpha*(u - y log u)
        b = x - alpha
        inside = b * b + 4 * alpha * y
        inside = torch.clamp(inside, min=self.eps)
        u = 0.5 * (b + torch.sqrt(inside))
        return torch.clamp(u, min=0.0)

class ModelA(nn.Module):
    """1st-order TNRD-style model (baseline)."""
    def __init__(self, num_stages=5, num_filters=8, kernel_size=5):
        super().__init__()
        self.num_stages = num_stages
        self.stages = nn.ModuleList([TNRDStageA(num_filters, kernel_size) for _ in range(num_stages)])
        self.prox = ProximalPoisson()
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, init_u, y, training_stage=-1):
        u = init_u
        for i, stage in enumerate(self.stages):
            diff = stage(u)
            u_pred = u - diff
            u = self.prox(u_pred, y, alpha=torch.relu(self.alpha) + 1e-6)
            if training_stage >= 0 and i == training_stage:
                break
        return u

class ModelB(nn.Module):
    """2nd-order telegraph model using u_{t-1}, u_t updates."""
    def __init__(self, num_stages=5, num_filters=8, kernel_size=5, gamma=0.5):
        super().__init__()
        self.num_stages = num_stages
        self.gamma = nn.Parameter(torch.tensor(gamma), requires_grad=False)
        self.stages = nn.ModuleList([TNRDStageA(num_filters, kernel_size) for _ in range(num_stages)])
        self.prox = ProximalPoisson()
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward(self, init_u, y, training_stage=-1):
        u_prev = init_u
        u = init_u
        for i, stage in enumerate(self.stages):
            diff = stage(u)
            u_pred = (2 - self.gamma) * u + (self.gamma - 1) * u_prev - diff
            u_next = self.prox(u_pred, y, alpha=torch.relu(self.alpha) + 1e-6)
            u_prev, u = u, u_next
            if training_stage >= 0 and i == training_stage:
                break
        return u
