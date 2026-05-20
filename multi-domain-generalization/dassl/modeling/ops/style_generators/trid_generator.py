import torch
import torch.nn as nn


class A2TriDStyleGenerator(nn.Module):
    """
    A2 = TriD-style feature statistics randomization generator.

    It samples mean/std from a uniform interval around the estimated
    ConstStyle statistics. This keeps the same output interface as A0/A1/A3.

    Output convention:
      const_mean: [B, C, 1, 1]
      const_std:  [B, C, 1, 1]
    """

    def __init__(self, const_mean, const_cov=None, factor=1.0, eps=1e-6):
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

        mean_radius = self.factor * base_std.std().clamp_min(self.eps)
        std_radius = self.factor * base_std.std().clamp_min(self.eps)

        mean_low = base_mean - mean_radius
        mean_high = base_mean + mean_radius

        std_low = (base_std - std_radius).clamp_min(self.eps)
        std_high = base_std + std_radius

        sampled_mean = mean_low.unsqueeze(0) + torch.rand(
            batch_size, base_mean.numel(), device=device
        ) * (mean_high - mean_low).unsqueeze(0)

        sampled_std = std_low.unsqueeze(0) + torch.rand(
            batch_size, base_std.numel(), device=device
        ) * (std_high - std_low).unsqueeze(0)

        sampled_std = sampled_std.abs().clamp_min(self.eps)

        const_mean = sampled_mean.reshape(batch_size, base_mean.numel(), 1, 1)
        const_std = sampled_std.reshape(batch_size, base_std.numel(), 1, 1)

        return const_mean, const_std