import torch
import torch.nn as nn

from dassl.modeling.ops.mixstyleflow.flow_generator import FlowStyleGenerator


class A3FlowStyleGenerator(nn.Module):
    """
    A3 = MixStyleFlow / normalizing-flow-based style generator.

    Output convention:
      const_mean: [B, C, 1, 1]
      const_std:  [B, C, 1, 1]
    """

    def __init__(self, cfg, style_dim):
        super().__init__()
        self.cfg = cfg
        self.style_dim = int(style_dim)

        self.generator = FlowStyleGenerator(
            style_dim=self.style_dim,
            hidden_size=None,
            n_flows=getattr(cfg.TRAINER.CONSTSTYLE, "FLOW_N_FLOWS", 4),
            clamp_value=getattr(cfg.TRAINER.CONSTSTYLE, "FLOW_CLAMP_VALUE", 2.0),
            temp=getattr(cfg.TRAINER.CONSTSTYLE, "FLOW_TEMP", 1.0),
            normalize_input=getattr(cfg.TRAINER.CONSTSTYLE, "FLOW_NORMALIZE_INPUT", False),
            use_gpu=torch.cuda.is_available(),
        )

    def forward(self, x):
        style = self.generator.sample(x.size(0), device=x.device)
        style = torch.reshape(style, (style.shape[0], 2, -1))

        const_mean = style[:, 0, :].float()
        const_std = style[:, 1, :].float()

        const_mean = torch.reshape(const_mean, (const_mean.shape[0], const_mean.shape[1], 1, 1))
        const_std = torch.reshape(const_std, (const_std.shape[0], const_std.shape[1], 1, 1))

        return const_mean, const_std
