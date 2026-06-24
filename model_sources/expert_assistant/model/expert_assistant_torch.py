from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def temporal_shuffle(x: torch.Tensor) -> torch.Tensor:
    """Shuffle assistant channels across temporal segments.

    Input and output use PyTorch layout (B, C, H, W). For K=2 and 12 channels
    per assistant, channel order [a1_c0..a1_c11, a2_c0..a2_c11] becomes
    [a1_c0, a2_c0, a1_c1, a2_c1, ...].
    """
    batch, channels, height, width = x.shape
    if channels != 24:
        raise ValueError(f"temporal_shuffle expects 24 channels, got {channels}")
    x = x.reshape(batch, 2, 12, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    return x.reshape(batch, channels, height, width)


class ExpertAssistant(nn.Module):
    """PyTorch implementation of the raw Expert-Assistant AMR network."""

    def __init__(self, input_length: int = 128, num_classes: int = 11):
        super().__init__()
        if input_length % 2 != 0:
            raise ValueError("input_length must be divisible by 2 for K=2 assistants")
        self.input_length = input_length
        self.num_classes = num_classes

        self.conv_b1 = nn.Conv2d(1, 75, kernel_size=(2, 8))
        self.conv_b2 = nn.Conv2d(75, 24, kernel_size=(1, 5))
        self.pool = nn.AvgPool2d(kernel_size=(1, 2))

        self.conv1_a1out = nn.Conv2d(1, 12, kernel_size=(2, 8))
        self.conv1_a2out = nn.Conv2d(1, 12, kernel_size=(2, 8))
        self.conv2_a1out = nn.Conv2d(12, 4, kernel_size=(1, 5))
        self.conv2_a2out = nn.Conv2d(12, 4, kernel_size=(1, 5))

        self.gru = nn.GRU(input_size=32, hidden_size=64, batch_first=True)
        self.dense_class = nn.Linear(64, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # inputs: (B, 2, L)
        expert = inputs.unsqueeze(1)  # (B, 1, 2, L)
        expert = F.pad(expert, (3, 4, 0, 0))
        expert = F.relu(self.conv_b1(expert))
        expert = F.pad(expert, (2, 2, 0, 0))
        expert = F.relu(self.conv_b2(expert))
        expert = self.pool(expert)  # (B, 24, 1, L/2)

        half = self.input_length // 2
        a1 = inputs[:, :, :half].unsqueeze(1)
        a2 = inputs[:, :, half:].unsqueeze(1)
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
        assistant = torch.cat([a1, a2], dim=1)  # (B, 8, 1, L/2)

        x = torch.cat([expert, assistant], dim=1)  # (B, 32, 1, L/2)
        x = x.squeeze(2).permute(0, 2, 1).contiguous()  # (B, L/2, 32)
        _, h_n = self.gru(x)
        return self.dense_class(h_n[-1])


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
