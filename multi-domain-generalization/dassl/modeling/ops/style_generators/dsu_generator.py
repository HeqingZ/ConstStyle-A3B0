import torch
import torch.nn as nn


class A1DSUStyleGenerator(nn.Module):
    """
    A1 = DSU-style Distribution Uncertainty generator.

    It estimates uncertainty from observed style statistics and samples
    mean/std by Gaussian perturbation.

    Output convention:
      const_mean: [B, C, 1, 1]
      const_std:  [B, C, 1, 1]
    """

    def __init__(self, const_mean, const_cov=None, factor=0.5, eps=1e-6):
        super().__init__()

        const_value = torch.reshape(const_mean.float(), (2, -1))
        base_mean = const_value[0]
        base_std = const_value[1].abs().clamp_min(eps)

        self.register_buffer("base_mean", base_mean)
        self.register_buffer("base_std", base_std)

        self.factor = factor
        self.eps = eps

    def forward(self, x):
        device = x.device
        batch_size = x.size(0)

        base_mean = self.base_mean.to(device)
        base_std = self.base_std.to(device)

        mean_noise_scale = base_std.std().clamp_min(self.eps)
        std_noise_scale = base_std.std().clamp_min(self.eps)

        sampled_mean = base_mean.unsqueeze(0) + self.factor * torch.randn(
            batch_size, base_mean.numel(), device=device
        ) * mean_noise_scale

        sampled_std = base_std.unsqueeze(0) + self.factor * torch.randn(
            batch_size, base_std.numel(), device=device
        ) * std_noise_scale

        sampled_std = sampled_std.abs().clamp_min(self.eps)

        const_mean = sampled_mean.reshape(batch_size, base_mean.numel(), 1, 1)
        const_std = sampled_std.reshape(batch_size, base_std.numel(), 1, 1)

        return const_mean, const_std