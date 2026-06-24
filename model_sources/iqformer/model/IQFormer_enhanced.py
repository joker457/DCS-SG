from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

from model.IQFormer import Stage, stemIQ


class FlexibleSTFTStem(nn.Module):
    """Compress multi-channel STFT features into time-aligned tokens."""

    def __init__(self, freq_bins: int, in_chans: int, out_chans: int):
        super().__init__()
        self.freq_bins = int(freq_bins)
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=(self.freq_bins, 1), stride=1, bias=False),
            nn.BatchNorm2d(out_chans),
            nn.GELU(),
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            trunc_normal_(module.weight, std=0.02)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return torch.squeeze(x, 2)


class GatedFusion(nn.Module):
    """Fuse IQ and STFT tokens with a learned modality gate."""

    def __init__(self, token_dim: int, out_dim: int, drop: float):
        super().__init__()
        in_dim = token_dim * 2
        self.gate = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=1),
            nn.BatchNorm1d(in_dim),
            nn.GELU(),
            nn.Conv1d(in_dim, in_dim, kernel_size=1),
            nn.Sigmoid(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(in_dim, out_dim, kernel_size=1),
            nn.BatchNorm1d(out_dim),
            nn.GELU(),
            nn.Conv1d(out_dim, out_dim, kernel_size=1),
        )
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm1d):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, iq_tokens: torch.Tensor, stft_tokens: torch.Tensor) -> torch.Tensor:
        if stft_tokens.shape[-1] != iq_tokens.shape[-1]:
            stft_tokens = F.interpolate(
                stft_tokens,
                size=iq_tokens.shape[-1],
                mode="linear",
                align_corners=False,
            )
        fused = torch.cat((iq_tokens, stft_tokens), dim=1)
        fused = fused * self.gate(fused)
        return self.drop(self.mix(fused))


class IQFormerEnhanced(nn.Module):
    """IQFormer with richer STFT features and gated early fusion.

    This variant keeps the original staged IQFormer backbone, while improving
    the front end for difficult channel and synchronization conditions.
    """

    def __init__(
        self,
        layers,
        embed_dims=None,
        mlp_ratios=4,
        act_layer=nn.GELU,
        num_classes=11,
        down_patch_size=5,
        down_stride=3,
        down_pad=1,
        drop_rate=0.0,
        drop_path_rate=0.0,
        use_layer_scale=True,
        layer_scale_init_value=1e-5,
        fork_feat=False,
        vit_num=1,
        stft_in_chans: int = 12,
        stft_freq_bins: int = 32,
    ):
        super().__init__()
        embed_dims = embed_dims or [64, 64, 64]
        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        token_dim = embed_dims[0] // 8
        fusion_dim = embed_dims[0] // 2
        self.BN = nn.BatchNorm1d(2)
        self.BN_stft = nn.BatchNorm2d(stft_in_chans)
        self.patch_embedIQ = stemIQ(2, token_dim * 2)
        self.patch_embedSTFT = FlexibleSTFTStem(stft_freq_bins, stft_in_chans, token_dim)
        self.fusion = GatedFusion(token_dim, fusion_dim, drop_rate)

        network = []
        for i in range(len(layers)):
            stage = Stage(
                embed_dims[i],
                i,
                layers,
                mlp_ratio=mlp_ratios,
                act_layer=act_layer,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
                vit_num=vit_num,
            )
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if embed_dims[i] != embed_dims[i + 1]:
                from model.IQFormer import Embedding

                network.append(
                    Embedding(
                        patch_size=down_patch_size,
                        stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i],
                        embed_dim=embed_dims[i + 1],
                    )
                )

        self.network = nn.ModuleList(network)
        self.patch_LSTM = nn.LSTM(
            input_size=fusion_dim,
            hidden_size=fusion_dim,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=drop_rate,
        )
        self.norm = nn.BatchNorm1d(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.globalavgpool = nn.Sequential(nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv1d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.network:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor, stft: torch.Tensor) -> torch.Tensor:
        x = self.BN(x)
        stft = self.BN_stft(stft)
        x = self.patch_embedIQ(x)
        stft = self.patch_embedSTFT(stft)
        x = self.fusion(x, stft)
        x, _ = self.patch_LSTM(x.permute(0, 2, 1))
        x = self.forward_tokens(x.permute(0, 2, 1))
        x = self.norm(x)
        return self.head(self.globalavgpool(x))
