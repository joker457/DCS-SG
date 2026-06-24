from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def matlab_pad(x: torch.Tensor, padding: tuple[int, int, int, int]) -> torch.Tensor:
    """MATLAB uses [top bottom left right]; torch uses [left right top bottom]."""
    top, bottom, left, right = padding
    if top == bottom == left == right == 0:
        return x
    return F.pad(x, (left, right, top, bottom))


class ConvRelu(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(matlab_pad(x, self.padding)))


class MatlabPool2d(nn.Module):
    def __init__(
        self,
        mode: str,
        kernel_size: tuple[int, int],
        stride: tuple[int, int],
        padding: tuple[int, int, int, int] = (0, 0, 0, 0),
    ):
        super().__init__()
        if mode not in {"max", "avg"}:
            raise ValueError("mode must be 'max' or 'avg'")
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = matlab_pad(x, self.padding)
        if self.mode == "max":
            return F.max_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=self.stride)


class DownsampleMBlock(nn.Module):
    def __init__(self, in_channels: int, branch_pool: str = "avg"):
        super().__init__()
        self.conv_a = ConvRelu(in_channels, 32, (1, 1))
        self.conv_b = ConvRelu(32, 48, (3, 1), padding=(1, 1, 0, 0))
        self.pool_b = MatlabPool2d(branch_pool, (1, 3), (1, 2), padding=(0, 0, 1, 1))
        self.conv_c = ConvRelu(32, 48, (1, 3), stride=(1, 2), padding=(0, 0, 1, 1))
        self.conv_d = ConvRelu(32, 32, (1, 1), stride=(1, 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_a(x)
        b = self.pool_b(self.conv_b(x))
        c = self.conv_c(x)
        d = self.conv_d(x)
        return torch.cat([c, b, d], dim=1)


class SameMBlock(nn.Module):
    def __init__(self, in_channels: int, d_channels: int = 32, b_channels: int = 48, c_channels: int = 48):
        super().__init__()
        self.conv_a = ConvRelu(in_channels, 32, (1, 1))
        self.conv_b = ConvRelu(32, b_channels, (3, 1), padding=(1, 1, 0, 0))
        self.conv_c = ConvRelu(32, c_channels, (1, 3), padding=(0, 0, 1, 1))
        self.conv_d = ConvRelu(32, d_channels, (1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_a(x)
        b = self.conv_b(x)
        c = self.conv_c(x)
        d = self.conv_d(x)
        return torch.cat([c, b, d], dim=1)


class MCNet(nn.Module):
    """PyTorch implementation of the MATLAB MCNet layer graph.

    The input tensor is expected to have shape [B, 1, 2, 1024].
    """

    def __init__(self, num_classes: int = 24, dropout: float = 0.5):
        super().__init__()
        self.stem = nn.Sequential(
            ConvRelu(1, 64, (3, 7), stride=(1, 2), padding=(1, 1, 3, 3)),
            MatlabPool2d("max", (1, 3), (1, 2), padding=(0, 0, 1, 1)),
        )
        self.pre_a = nn.Sequential(
            ConvRelu(64, 32, (3, 1), padding=(1, 1, 0, 0)),
            MatlabPool2d("avg", (1, 3), (1, 2), padding=(0, 0, 1, 1)),
        )
        self.pre_b = ConvRelu(64, 32, (1, 3), stride=(1, 2), padding=(0, 0, 1, 1))

        self.jump_a = nn.Sequential(
            ConvRelu(64, 128, (1, 1), stride=(1, 2)),
            MatlabPool2d("max", (1, 3), (1, 2), padding=(0, 0, 1, 1)),
        )
        self.post_pool = MatlabPool2d("max", (1, 3), (1, 2), padding=(0, 0, 1, 1))
        self.block_a = DownsampleMBlock(64, branch_pool="avg")
        self.block_b = SameMBlock(128)

        self.jump_c = MatlabPool2d("max", (2, 2), (1, 2), padding=(1, 0, 0, 0))
        self.block_c = DownsampleMBlock(128, branch_pool="avg")
        self.block_d = SameMBlock(128)

        self.jump_e = MatlabPool2d("max", (2, 2), (1, 2), padding=(1, 0, 0, 0))
        self.block_e = DownsampleMBlock(128, branch_pool="max")
        self.block_f = SameMBlock(128, d_channels=64, b_channels=96, c_channels=96)

        self.global_pool = MatlabPool2d("avg", (2, 8), (1, 1))
        self.fc = nn.Linear(384, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.cat([self.pre_b(x), self.pre_a(x)], dim=1)

        jump = self.jump_a(x)
        x = self.post_pool(x)
        x = self.block_a(x) + jump
        x = self.block_b(x) + x

        jump = self.jump_c(x)
        x = self.block_c(x) + jump
        x = self.block_d(x) + x

        jump = self.jump_e(x)
        x = self.block_e(x) + jump
        mix_f = self.block_f(x)
        x = torch.cat([mix_f, x], dim=1)

        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return self.dropout(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


OBS_LEVEL_LENGTHS = (2048, 1024, 512, 256, 128, 64)


class ResidualChannelGate(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        gate = self.fc(pooled).view(x.shape[0], x.shape[1], 1, 1)
        return x * (1.0 + self.scale * gate)


class ResidualTemporalGate(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.context = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=True),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2, groups=channels, bias=True),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.scale = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        temporal = x.mean(dim=2)
        gate = self.context(temporal).unsqueeze(2)
        return x * (1.0 + self.scale * gate)


class MCNetEnhanced(nn.Module):
    """MCNet trunk with conservative robustness additions.

    The convolutional stem and M-block topology intentionally mirrors MCNet.
    The added temporal/channel gates are residual and initialized with zero
    scale, so the model starts from the public MCNet computation. Datagen
    boundary experiments can pass the demand tensor to enable observation
    length-aware global pooling for padded short-window signals.
    """

    uses_demand = True

    def __init__(
        self,
        num_classes: int = 24,
        dropout: float = 0.2,
        use_obs_mask: bool = True,
        use_channel_gate: bool = True,
        use_temporal_gate: bool = True,
    ):
        super().__init__()
        self.use_obs_mask = use_obs_mask
        self.stem = nn.Sequential(
            ConvRelu(1, 64, (3, 7), stride=(1, 2), padding=(1, 1, 3, 3)),
            MatlabPool2d("max", (1, 3), (1, 2), padding=(0, 0, 1, 1)),
        )
        self.pre_a = nn.Sequential(
            ConvRelu(64, 32, (3, 1), padding=(1, 1, 0, 0)),
            MatlabPool2d("avg", (1, 3), (1, 2), padding=(0, 0, 1, 1)),
        )
        self.pre_b = ConvRelu(64, 32, (1, 3), stride=(1, 2), padding=(0, 0, 1, 1))

        self.jump_a = nn.Sequential(
            ConvRelu(64, 128, (1, 1), stride=(1, 2)),
            MatlabPool2d("max", (1, 3), (1, 2), padding=(0, 0, 1, 1)),
        )
        self.post_pool = MatlabPool2d("max", (1, 3), (1, 2), padding=(0, 0, 1, 1))
        self.block_a = DownsampleMBlock(64, branch_pool="avg")
        self.block_b = SameMBlock(128)

        self.jump_c = MatlabPool2d("max", (2, 2), (1, 2), padding=(1, 0, 0, 0))
        self.block_c = DownsampleMBlock(128, branch_pool="avg")
        self.block_d = SameMBlock(128)

        self.jump_e = MatlabPool2d("max", (2, 2), (1, 2), padding=(1, 0, 0, 0))
        self.block_e = DownsampleMBlock(128, branch_pool="max")
        self.block_f = SameMBlock(128, d_channels=64, b_channels=96, c_channels=96)

        self.channel_gate = ResidualChannelGate(384) if use_channel_gate else nn.Identity()
        self.temporal_gate = ResidualTemporalGate(384) if use_temporal_gate else nn.Identity()
        self.global_pool = MatlabPool2d("avg", (2, 8), (1, 1))
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.fc = nn.Linear(384, num_classes)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = torch.cat([self.pre_b(x), self.pre_a(x)], dim=1)

        jump = self.jump_a(x)
        x = self.post_pool(x)
        x = self.block_a(x) + jump
        x = self.block_b(x) + x

        jump = self.jump_c(x)
        x = self.block_c(x) + jump
        x = self.block_d(x) + x

        jump = self.jump_e(x)
        x = self.block_e(x) + jump
        mix_f = self.block_f(x)
        x = torch.cat([mix_f, x], dim=1)
        x = self.channel_gate(x)
        x = self.temporal_gate(x)
        return x

    def _obs_width_mask(self, demand: torch.Tensor, width: int, input_width: int, dtype: torch.dtype) -> torch.Tensor:
        obs = demand[:, 1].long().clamp(0, len(OBS_LEVEL_LENGTHS) - 1)
        lengths = torch.tensor(OBS_LEVEL_LENGTHS, device=demand.device, dtype=dtype)[obs]
        lengths = lengths.clamp(max=float(input_width))
        starts = (float(input_width) - lengths) * 0.5
        ends = starts + lengths
        centers = (torch.arange(width, device=demand.device, dtype=dtype) + 0.5) * (float(input_width) / float(width))
        mask = (centers[None, :] >= starts[:, None]) & (centers[None, :] < ends[:, None])
        center_idx = width // 2
        mask[:, center_idx] = True
        return mask.to(dtype)

    def _pool_features(self, x: torch.Tensor, demand: torch.Tensor | None, input_width: int) -> torch.Tensor:
        if demand is None or not self.use_obs_mask:
            return self.global_pool(x).flatten(1)
        width = x.shape[-1]
        mask = self._obs_width_mask(demand.to(x.device), width, input_width, x.dtype)
        masked = x * mask[:, None, None, :]
        denom = (mask.sum(dim=1) * x.shape[2]).clamp_min(1.0)
        return masked.sum(dim=(2, 3)) / denom[:, None]

    def forward(self, x: torch.Tensor, demand: torch.Tensor | None = None) -> torch.Tensor:
        input_width = int(x.shape[-1])
        x = self.extract_features(x)
        x = self._pool_features(x, demand, input_width)
        x = self.dropout(x)
        return self.fc(x)


def build_mcnet(
    variant: str = "original",
    num_classes: int = 24,
    dropout: float = 0.5,
) -> nn.Module:
    variant = (variant or "original").lower()
    if variant in {"original", "mcnet"}:
        return MCNet(num_classes=num_classes, dropout=dropout)
    if variant in {"enhanced", "mcnet-enhanced", "mcnet_v2", "v2"}:
        return MCNetEnhanced(num_classes=num_classes, dropout=dropout)
    raise ValueError(f"Unknown MCNet variant: {variant}")
