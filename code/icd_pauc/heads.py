"""Lightweight heads for the frozen-PanDerm ICD+pAUC method.

  BaseHead : CE linear probe (stage-1 anchor, frozen after fit).
  ICDHead  : logits-space inter-class de-confusion residual (borrowed from LDC,
             CVPR2025, clip_ldc/model.py:424-433,585-587), retargeted to a frozen
             ViT feature + frozen CE anchor.

Forward (stage-2):
  base_logits = base_head(z).detach()      # [B,C] frozen constant
  a = adapter_1(base_logits)               # C -> d
  b = adapter_2(z)                         # 1024 -> d
  residual = adapter_3(gelu(a + b))        # d -> C
  final_logits = base_logits + residual
"""
import torch
import torch.nn as nn


class BaseHead(nn.Module):
    """Stage-1 CE linear probe = prior anchor."""
    def __init__(self, in_dim=1024, num_classes=7):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, z):
        return self.fc(z)


class ICDHead(nn.Module):
    """Stage-2 inter-class de-confusion residual head (~0.53M params for d=512)."""
    def __init__(self, num_classes=7, in_dim=1024, d=512):
        super().__init__()
        self.adapter_1 = nn.Linear(num_classes, d)   # anchor logits -> hidden
        self.adapter_2 = nn.Linear(in_dim, d)         # frozen ViT feature -> hidden
        self.adapter_3 = nn.Linear(d, num_classes)    # hidden -> residual logits
        self.act = nn.GELU()

    def forward(self, base_logits, z):
        a = self.adapter_1(base_logits)
        b = self.adapter_2(z)
        residual = self.adapter_3(self.act(a + b))
        return base_logits + residual                 # final_logits


def num_trainable_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
