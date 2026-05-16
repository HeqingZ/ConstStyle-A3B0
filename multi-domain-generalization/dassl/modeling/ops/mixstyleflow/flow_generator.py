import torch
import torch.nn as nn
from types import SimpleNamespace

from .nflows import get_normalizing_flow


class FlowStyleGenerator(nn.Module):
    """
    A3 style generator for ConstStyle.

    It replaces the original BayesianGaussianMixture + MultivariateNormal
    style sampler with a normalizing-flow-based sampler.

    Input/output convention:
      style vector shape: [B, 2*C]
      first C dims  = style mean
      second C dims = style std
    """

    def __init__(
        self,
        style_dim,
        hidden_size=None,
        n_flows=4,
        clamp_value=2.0,
        temp=1.0,
        normalize_input=False,
        use_gpu=True,
    ):
        super().__init__()

        self.style_dim = style_dim

        network_config = SimpleNamespace(
            normalize_input=normalize_input,
            latent_size=style_dim,
            hidden_size=hidden_size,
            temp=temp,
            n_flows=n_flows,
            clamp_value=clamp_value,
        )

        self.flow = get_normalizing_flow(network_config, use_gpu=use_gpu)

    @torch.no_grad()
    def sample(self, num_samples, device=None):
        styles = self.flow.get_sample(num_samples)

        if device is not None:
            styles = styles.to(device)

        return styles
