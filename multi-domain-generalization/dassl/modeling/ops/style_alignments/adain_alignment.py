import torch.nn as nn


class B0AdaINAlignment(nn.Module):
    """
    B0 = original ConstStyle AdaIN-like alignment.

    out = x_normed * const_std + const_mean
    """

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x, const_mean, const_std):
        mu = x.mean(dim=[2, 3], keepdim=True).detach()
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt().detach()

        x_normed = (x - mu) / sig
        out = x_normed * const_std + const_mean

        return out
