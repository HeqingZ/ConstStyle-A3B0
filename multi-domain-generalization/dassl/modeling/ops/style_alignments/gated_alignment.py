import torch.nn as nn


class B1GatedAlignment(nn.Module):
    """
    B1 = residual/gated ConstStyle alignment.

    It first computes the original B0 stylized feature:

        stylized = x_normed * const_std + const_mean

    Then applies a residual interpolation:

        out = x + alpha * (stylized - x)

    alpha=1.0 makes it equivalent to B0.
    alpha<1.0 makes the style transfer milder.
    """

    def __init__(self, eps=1e-6, alpha=0.5):
        super().__init__()
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, const_mean, const_std):
        mu = x.mean(dim=[2, 3], keepdim=True).detach()
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt().detach()

        x_normed = (x - mu) / sig
        stylized = x_normed * const_std + const_mean

        out = x + self.alpha * (stylized - x)

        return out