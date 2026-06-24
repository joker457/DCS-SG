from __future__ import annotations

import torch
import torch.nn as nn

from model.IQFormer import IQFormer


class ResidualAdapterBranch(nn.Module):
    """A zero-start residual classifier branch for IQFormer.

    The last linear layer is initialized to zero, so this branch contributes
    exactly zero logits at initialization. When added to a pretrained IQFormer,
    the initial predictions are therefore identical to the base model.
    """

    def __init__(self, num_classes: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.iq_encoder = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.stft_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(5, 7), padding=(2, 3), bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=(3, 5), padding=(1, 2), bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(96, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.reset_residual_head()

    def reset_residual_head(self) -> None:
        last = self.classifier[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, iq: torch.Tensor, stft: torch.Tensor) -> torch.Tensor:
        iq_feat = self.iq_encoder(iq)
        stft_feat = self.stft_encoder(stft)
        return self.classifier(torch.cat([iq_feat, stft_feat], dim=1))


class IQFormerResidualAdapter(nn.Module):
    def __init__(
        self,
        base: IQFormer,
        num_classes: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base
        self.adapter = ResidualAdapterBranch(num_classes, hidden_dim=hidden_dim, dropout=dropout)
        self.freeze_base = bool(freeze_base)
        if self.freeze_base:
            self.freeze_base_parameters()

    def freeze_base_parameters(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
        return self

    def forward(self, iq: torch.Tensor, stft: torch.Tensor) -> torch.Tensor:
        if self.freeze_base:
            with torch.no_grad():
                base_logits = self.base(iq, stft)
        else:
            base_logits = self.base(iq, stft)
        return base_logits + self.adapter(iq, stft)

    def forward_base(self, iq: torch.Tensor, stft: torch.Tensor) -> torch.Tensor:
        return self.base(iq, stft)
