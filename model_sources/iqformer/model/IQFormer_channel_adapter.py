from __future__ import annotations

import torch
import torch.nn as nn

from model.IQFormer import IQFormer


class ZeroFIRChannelCompensator(nn.Module):
    """Lossless-start I/Q front-end for channel-aware correction.

    The 2x2 mixing matrix, FIR convolution, and bias all start at zero, so the
    module is exactly an identity map at initialization. During training it can
    learn small phase/amplitude/IQ-mixing and local equalization corrections.
    """

    def __init__(self, kernel_size: int = 9):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve sequence length.")
        self.mix = nn.Parameter(torch.zeros(2, 2))
        self.fir = nn.Conv1d(2, 2, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.bias = nn.Parameter(torch.zeros(1, 2, 1))
        nn.init.zeros_(self.fir.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mixed = torch.einsum("ij,bjl->bil", self.mix, x)
        return x + mixed + self.fir(x) + self.bias


class ZeroSTFTCompensator(nn.Module):
    """Lossless-start T-F residual correction for the original IQFormer STFT input."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(1, 2), bias=True)
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv(x)


class ZeroTokenAdapter(nn.Module):
    """Bottleneck residual adapter with zero output at initialization."""

    def __init__(self, channels: int, bottleneck: int = 16, dropout: float = 0.0):
        super().__init__()
        bottleneck = max(1, int(bottleneck))
        self.norm = nn.LayerNorm(channels)
        self.down = nn.Linear(channels, bottleneck)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck, channels)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.transpose(1, 2)
        y = self.up(self.drop(self.act(self.down(self.norm(y)))))
        return x + y.transpose(1, 2)


class ZeroHeadAdapter(nn.Module):
    """Small residual classifier head; starts as exactly zero logits."""

    def __init__(self, channels: int, num_classes: int, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        last = self.net[-1]
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class IQFormerChannelAdapter(nn.Module):
    """Channel-aware, feature-level lossless adapter for pretrained IQFormer.

    Compared with a logits-only residual branch, this wrapper inserts small
    zero-start modules at the signal front-end and IQFormer token pathway. The
    pretrained IQFormer can remain frozen, while gradients still flow through
    it to train the adapters.
    """

    def __init__(
        self,
        base: IQFormer,
        num_classes: int,
        bottleneck: int = 16,
        dropout: float = 0.05,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base
        self.freeze_base = bool(freeze_base)
        self.iq_compensator = ZeroFIRChannelCompensator(kernel_size=9)
        self.stft_compensator = ZeroSTFTCompensator()
        self.fusion_adapter = ZeroTokenAdapter(32, bottleneck=bottleneck, dropout=dropout)
        self.lstm_adapter = ZeroTokenAdapter(64, bottleneck=bottleneck, dropout=dropout)
        self.encoder_adapter = ZeroTokenAdapter(64, bottleneck=bottleneck, dropout=dropout)
        self.head_adapter = ZeroHeadAdapter(64, num_classes, hidden_dim=max(32, bottleneck * 4), dropout=dropout)
        if self.freeze_base:
            self.freeze_base_parameters()

    def freeze_base_parameters(self) -> None:
        for param in self.base.parameters():
            param.requires_grad = False
        if hasattr(self.base, "patch_LSTM"):
            self.base.patch_LSTM.dropout = 0.0
        self.base.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_base:
            self.base.eval()
            if mode and hasattr(self.base, "patch_LSTM"):
                self.base.patch_LSTM.train(True)
        return self

    def forward_base(self, iq: torch.Tensor, stft: torch.Tensor) -> torch.Tensor:
        return self.base(iq, stft)

    def forward(self, iq: torch.Tensor, stft: torch.Tensor) -> torch.Tensor:
        iq = self.iq_compensator(iq)
        stft = self.stft_compensator(stft)

        x = self.base.BN(iq)
        stft = self.base.BN_stft(stft)
        x = self.base.patch_embedIQ(x)
        stft = torch.squeeze(self.base.patch_embedSTFT(stft), 2)
        x = self.base.fusion(x, stft)
        x = self.fusion_adapter(x)

        x, _ = self.base.patch_LSTM(x.permute(0, 2, 1))
        x = self.lstm_adapter(x.permute(0, 2, 1))
        x = self.base.forward_tokens(x)
        x = self.encoder_adapter(x)
        x = self.base.norm(x)
        pooled = self.base.globalavgpool(x)
        return self.base.head(pooled) + self.head_adapter(pooled)
