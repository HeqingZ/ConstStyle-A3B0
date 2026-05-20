import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal


class A0OriginalStyleGenerator(nn.Module):
    """
    A0 = original ConstStyle sampler.

    It uses the clustered style mean/covariance estimated by ConstStyle.cal_mean_std(),
    then samples style vectors from MultivariateNormal.

    Output convention:
      const_mean: [B, C, 1, 1]
      const_std:  [B, C, 1, 1]
    """

    def __init__(self, const_mean, const_cov):
        super().__init__()

        self.register_buffer("const_mean", const_mean.float())
        self.register_buffer("const_cov", const_cov.float())

    def forward(self, x):
        device = x.device

        mean = self.const_mean.to(device)
        cov = self.const_cov.to(device)

        # Small diagonal jitter for numerical stability.
        eye = torch.eye(cov.size(0), device=device, dtype=cov.dtype)
        cov = cov + 1e-6 * eye

        dist = MultivariateNormal(mean, covariance_matrix=cov)
        style = dist.sample((x.size(0),))

        style = torch.reshape(style, (style.shape[0], 2, -1))

        const_mean = style[:, 0, :].float()
        const_std = style[:, 1, :].float()

        const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1))
        const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1))

        return const_mean, const_std