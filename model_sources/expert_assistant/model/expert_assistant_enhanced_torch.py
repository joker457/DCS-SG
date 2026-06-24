from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.expert_assistant_torch import temporal_shuffle


@dataclass
class LoadReport:
    loaded: int
    skipped: list[str]
    missing: list[str]


def _zero_last_linear(module: nn.Module) -> None:
    linears = [m for m in module.modules() if isinstance(m, nn.Linear)]
    if not linears:
        return
    nn.init.zeros_(linears[-1].weight)
    if linears[-1].bias is not None:
        nn.init.zeros_(linears[-1].bias)


class PhaseFrequencyCompensator(nn.Module):
    """Identity-start phase/frequency compensator inspired by PET.

    The final linear layer is initialized to zero, so the predicted phase and
    frequency offsets are exactly zero at initialization. This keeps a loaded
    RAW E-A checkpoint numerically unchanged before fine-tuning.
    """

    def __init__(self, hidden_dim: int = 32, max_phase: float = math.pi, max_freq: float = 0.25):
        super().__init__()
        self.max_phase = float(max_phase)
        self.max_freq = float(max_freq)
        self.estimator = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )
        _zero_last_linear(self.estimator)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, L)
        mean = x.mean(dim=-1)
        std = x.std(dim=-1, unbiased=False)
        power = (x * x).mean(dim=-1)
        iq_corr = (x[:, 0] * x[:, 1]).mean(dim=-1, keepdim=True)
        stats = torch.cat([mean, std, power, iq_corr], dim=1)
        offsets = self.estimator(stats)
        phase = self.max_phase * torch.tanh(offsets[:, 0])
        freq = self.max_freq * torch.tanh(offsets[:, 1])

        length = x.shape[-1]
        t = torch.linspace(-0.5, 0.5, length, device=x.device, dtype=x.dtype)
        theta = phase[:, None] + (2.0 * math.pi) * freq[:, None] * t[None, :]
        c = torch.cos(theta)
        s = torch.sin(theta)
        i = x[:, 0]
        q = x[:, 1]
        return torch.stack((i * c - q * s, i * s + q * c), dim=1)


class SpectralResidualBranch(nn.Module):
    """Small frequency-domain residual classifier.

    This branch uses an FFT magnitude summary instead of a full STFT to keep the
    E-A improvement lightweight. Its final classifier is zero-initialized, so it
    starts as an exact zero-logit residual.
    """

    def __init__(self, num_classes: int, channels: int = 32, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(2, channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels, num_classes),
        )
        _zero_last_linear(self.net)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spec = torch.fft.rfft(x.float(), dim=-1)
        mag = torch.log1p(torch.abs(spec)).to(dtype=x.dtype)
        return self.net(mag)


class ExpertAssistantEnhanced(nn.Module):
    """Lossless-start enhanced Expert-Assistant network.

    The RAW E-A layers deliberately keep the same names as
    `ExpertAssistant`, so a RAW checkpoint can be loaded with `strict=False`.
    New PET/spectral/feature residual modules are initialized to zero effect.
    """

    def __init__(
        self,
        input_length: int = 128,
        num_classes: int = 11,
        pet_hidden: int = 32,
        spectral_channels: int = 32,
        feature_hidden: int = 64,
        residual_dropout: float = 0.1,
        use_pet: bool = True,
        use_spectral: bool = True,
        use_feature_residual: bool = True,
    ):
        super().__init__()
        if input_length % 2 != 0:
            raise ValueError("input_length must be divisible by 2 for K=2 assistants")
        self.input_length = input_length
        self.num_classes = num_classes
        self.use_pet = bool(use_pet)
        self.use_spectral = bool(use_spectral)
        self.use_feature_residual = bool(use_feature_residual)

        # RAW E-A trunk. Names match model.expert_assistant_torch.ExpertAssistant.
        self.conv_b1 = nn.Conv2d(1, 75, kernel_size=(2, 8))
        self.conv_b2 = nn.Conv2d(75, 24, kernel_size=(1, 5))
        self.pool = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_a1out = nn.Conv2d(1, 12, kernel_size=(2, 8))
        self.conv1_a2out = nn.Conv2d(1, 12, kernel_size=(2, 8))
        self.conv2_a1out = nn.Conv2d(12, 4, kernel_size=(1, 5))
        self.conv2_a2out = nn.Conv2d(12, 4, kernel_size=(1, 5))

        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.dense_class = nn.Linear(64, num_classes)

        self.pet = PhaseFrequencyCompensator(hidden_dim=pet_hidden) if self.use_pet else nn.Identity()
        self.spectral_residual = (
            SpectralResidualBranch(num_classes=num_classes, channels=spectral_channels, dropout=residual_dropout)
            if self.use_spectral
            else None
        )
        self.feature_residual = (
            nn.Sequential(
                nn.LayerNorm(64),
                nn.Dropout(residual_dropout),
                nn.Linear(64, feature_hidden),
                nn.GELU(),
                nn.Linear(feature_hidden, num_classes),
            )
            if self.use_feature_residual
            else None
        )
        if self.feature_residual is not None:
            _zero_last_linear(self.feature_residual)

    def forward_features(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x_in = self.pet(inputs) if self.use_pet else inputs

        expert = x_in.unsqueeze(1)
        expert = F.pad(expert, (3, 4, 0, 0))
        expert = F.relu(self.conv_b1(expert))
        expert = F.pad(expert, (2, 2, 0, 0))
        expert = F.relu(self.conv_b2(expert))
        expert = self.pool(expert)

        half = self.input_length // 2
        a1 = x_in[:, :, :half].unsqueeze(1)
        a2 = x_in[:, :, half:].unsqueeze(1)
        a1 = F.pad(a1, (3, 4, 0, 0))
        a2 = F.pad(a2, (3, 4, 0, 0))
        a1 = F.relu(self.conv1_a1out(a1))
        a2 = F.relu(self.conv1_a2out(a2))

        assistant = torch.cat([a1, a2], dim=1)
        assistant = temporal_shuffle(assistant)

        a1 = F.pad(assistant[:, :12], (2, 2, 0, 0))
        a2 = F.pad(assistant[:, 12:], (2, 2, 0, 0))
        a1 = F.relu(self.conv2_a1out(a1))
        a2 = F.relu(self.conv2_a2out(a2))
        assistant = torch.cat([a1, a2], dim=1)

        features = torch.cat([expert, assistant], dim=1)
        features = features.squeeze(2).permute(0, 2, 1).contiguous()
        _, h_n = self.gru(features)
        pooled = h_n[-1]
        return pooled, x_in

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features, compensated = self.forward_features(inputs)
        logits = self.dense_class(features)
        if self.feature_residual is not None:
            logits = logits + self.feature_residual(features)
        if self.spectral_residual is not None:
            logits = logits + self.spectral_residual(compensated)
        return logits

    def base_parameters(self):
        names = {
            "conv_b1",
            "conv_b2",
            "conv1_a1out",
            "conv1_a2out",
            "conv2_a1out",
            "conv2_a2out",
            "gru",
            "dense_class",
        }
        for name, module in self.named_children():
            if name in names:
                yield from module.parameters()

    def enhancement_parameters(self):
        base_ids = {id(p) for p in self.base_parameters()}
        for p in self.parameters():
            if id(p) not in base_ids:
                yield p

    def set_base_trainable(self, trainable: bool) -> None:
        for p in self.base_parameters():
            p.requires_grad = trainable

    def load_compatible_state_dict(self, state_dict: dict[str, torch.Tensor]) -> LoadReport:
        own_state = self.state_dict()
        filtered = {}
        skipped = []
        for key, value in state_dict.items():
            clean_key = key[7:] if key.startswith("module.") else key
            if clean_key in own_state and own_state[clean_key].shape == value.shape:
                filtered[clean_key] = value
            else:
                skipped.append(key)
        result = self.load_state_dict(filtered, strict=False)
        return LoadReport(loaded=len(filtered), skipped=skipped, missing=list(result.missing_keys))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
