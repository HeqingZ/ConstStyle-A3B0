import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from types import SimpleNamespace
from dassl.modeling.ops.conststyle import ConstStyle


def main():
    cfg = SimpleNamespace(
        CLUSTER="gmm",
        NUM_CLUSTERS=2,
        TRAINER=SimpleNamespace(
            CONSTSTYLE=SimpleNamespace(
                ALPHA_TEST=None,
                PROB=1.0,
                STYLE_GENERATOR="A1",
                STYLE_ALIGNMENT="B1",
                B1_ALPHA=0.5,
                DSU_FACTOR=0.5,
                FLOW_N_FLOWS=2,
                FLOW_CLAMP_VALUE=2.0,
                FLOW_TEMP=1.0,
                FLOW_NORMALIZE_INPUT=False,
            )
        )
    )

    cs = ConstStyle(idx=0, cfg=cfg)

    x_store = torch.randn(8, 64, 8, 8)
    domain_store = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2])

    cs.store_style(x_store, domain_store)
    cs.cal_mean_std()

    x = torch.randn(4, 64, 8, 8, requires_grad=True)
    domain = torch.tensor([0, 1, 2, 0])

    out = cs(x, domain=domain, apply_conststyle=True)

    loss = out.mean()
    loss.backward()

    print("out shape:", out.shape)
    print("finite:", torch.isfinite(out).all().item())
    print("backward ok:", x.grad is not None)
    print("grad finite:", torch.isfinite(x.grad).all().item())


if __name__ == "__main__":
    main()
