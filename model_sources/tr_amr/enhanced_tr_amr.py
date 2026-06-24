from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from vit_model_2018 import VisionTransformer


def _zero_init(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear)):
        nn.init.zeros_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    return module


class ResidualSignalConditioner(nn.Module):
    """Learnable I/Q frontend that is exactly identity at initialization."""

    def __init__(self, hidden_channels: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=(1, 9), padding=(0, 4), bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=(2, 1), padding=(0, 0), bias=False),
            nn.GELU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=(2, 1), bias=False),
            nn.GELU(),
            _zero_init(nn.Conv2d(hidden_channels, 1, kernel_size=(1, 7), padding=(0, 3))),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TokenScaleAdapter(nn.Module):
    """Multi-scale temporal token adapter with zero residual output initially."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(32, embed_dim // 2)
        self.in_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.branch3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=1)
        self.branch5 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2, groups=1)
        self.branch9 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=9, padding=4, groups=1)
        self.out_proj = _zero_init(nn.Conv1d(hidden_dim * 3, embed_dim, kernel_size=1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = tokens.transpose(1, 2)
        x = F.gelu(self.in_proj(x))
        x = torch.cat(
            [F.gelu(self.branch3(x)), F.gelu(self.branch5(x)), F.gelu(self.branch9(x))],
            dim=1,
        )
        return self.out_proj(x).transpose(1, 2)


class TokenDenoiseAdapter(nn.Module):
    """Identity-initialized local token denoiser for boundary-robust training."""

    def __init__(self, embed_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.branch3 = nn.Conv1d(embed_dim, embed_dim, kernel_size=3, padding=1, groups=embed_dim)
        self.branch5 = nn.Conv1d(embed_dim, embed_dim, kernel_size=5, padding=2, groups=embed_dim)
        self.branch9 = nn.Conv1d(embed_dim, embed_dim, kernel_size=9, padding=4, groups=embed_dim)
        self.mix = _zero_init(nn.Conv1d(embed_dim * 3, embed_dim, kernel_size=1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens).transpose(1, 2)
        x = torch.cat(
            [F.gelu(self.branch3(x)), F.gelu(self.branch5(x)), F.gelu(self.branch9(x))],
            dim=1,
        )
        return self.mix(x).transpose(1, 2)


class TokenPoolAdapter(nn.Module):
    """Zero-residual adapter that supplements cls-token decisions with patch statistics."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(64, embed_dim)
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 4),
            nn.Linear(embed_dim * 4, hidden_dim),
            nn.GELU(),
            _zero_init(nn.Linear(hidden_dim, embed_dim)),
        )

    def forward(self, cls_feature: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        mean = patch_tokens.mean(dim=1)
        maxv = patch_tokens.amax(dim=1)
        std = patch_tokens.var(dim=1, unbiased=False).clamp_min(1e-8).sqrt()
        pooled = torch.cat([cls_feature, mean, maxv, std], dim=-1)
        return self.proj(pooled)


class BoundaryTokenAdapter(nn.Module):
    """Lossless token mixer for boundary data without raw-signal shortcuts."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(64, embed_dim // 2)
        self.norm = nn.LayerNorm(embed_dim)
        self.in_proj = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.local3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.local7 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=7, padding=3, groups=hidden_dim)
        self.dilated = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2, groups=hidden_dim)
        self.mix = _zero_init(nn.Conv1d(hidden_dim * 3, embed_dim, kernel_size=1))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.norm(tokens).transpose(1, 2)
        x = F.gelu(self.in_proj(x))
        x = torch.cat(
            [F.gelu(self.local3(x)), F.gelu(self.local7(x)), F.gelu(self.dilated(x))],
            dim=1,
        )
        return self.mix(x).transpose(1, 2)


class AttentiveTokenPoolAdapter(nn.Module):
    """Zero-residual context adapter using mean/max/std and attentive patch pooling."""

    def __init__(self, embed_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden_dim = hidden_dim or max(64, embed_dim)
        self.score = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.proj = nn.Sequential(
            nn.LayerNorm(embed_dim * 5),
            nn.Linear(embed_dim * 5, hidden_dim),
            nn.GELU(),
            _zero_init(nn.Linear(hidden_dim, embed_dim)),
        )

    def forward(self, cls_feature: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        mean = patch_tokens.mean(dim=1)
        maxv = patch_tokens.amax(dim=1)
        std = patch_tokens.var(dim=1, unbiased=False).clamp_min(1e-8).sqrt()
        attn = torch.softmax(self.score(patch_tokens).squeeze(-1), dim=1)
        attentive = torch.sum(patch_tokens * attn.unsqueeze(-1), dim=1)
        pooled = torch.cat([cls_feature, mean, maxv, std, attentive], dim=-1)
        return self.proj(pooled)


class RawIQFeatureAdapter(nn.Module):
    """Feature adapter from raw I/Q, magnitude, and phase-difference cues."""

    def __init__(self, embed_dim: int, hidden_dim: int = 64):
        super().__init__()
        in_ch = 4
        self.branch7 = nn.Sequential(
            nn.Conv1d(in_ch, hidden_dim, kernel_size=7, padding=3),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch15 = nn.Sequential(
            nn.Conv1d(in_ch, hidden_dim, kernel_size=15, padding=7),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch31 = nn.Sequential(
            nn.Conv1d(in_ch, hidden_dim, kernel_size=31, padding=15),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.out = _zero_init(nn.Linear(hidden_dim * 3, embed_dim))

    @staticmethod
    def _features(x: torch.Tensor) -> torch.Tensor:
        iq = x.squeeze(1)
        i = iq[:, 0]
        q = iq[:, 1]
        mag = torch.sqrt(i.square() + q.square() + 1e-8)
        i_prev = F.pad(i[:, :-1], (1, 0))
        q_prev = F.pad(q[:, :-1], (1, 0))
        cross = i_prev * q - q_prev * i
        dot = i_prev * i + q_prev * q
        dphi = torch.atan2(cross, dot + 1e-8) / torch.pi
        return torch.stack([i, q, mag, dphi], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self._features(x)
        pooled = torch.cat(
            [
                self.branch7(feat).squeeze(-1),
                self.branch15(feat).squeeze(-1),
                self.branch31(feat).squeeze(-1),
            ],
            dim=1,
        )
        return self.out(pooled)


class TimeFrequencyAdapter(nn.Module):
    """Small spectro-temporal adapter inspired by IQ/time-frequency fusion."""

    def __init__(
        self,
        embed_dim: int,
        n_fft: int = 64,
        hop_length: int = 16,
        hidden_channels: int = 24,
    ):
        super().__init__()
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(n_fft)
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out = _zero_init(nn.Linear(hidden_channels, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        iq = x.squeeze(1).float()
        z = torch.complex(iq[:, 0], iq[:, 1])
        spec = torch.stft(
            z,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window.to(device=x.device, dtype=torch.float32),
            return_complex=True,
            center=True,
        )
        mag = torch.log1p(spec.abs()).unsqueeze(1)
        pooled = self.net(mag).flatten(1).to(dtype=x.dtype)
        return self.out(pooled)


class EnhancedTrAMR(nn.Module):
    """Backward-compatible Tr-AMR with identity-initialized robustness adapters."""

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 64,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 16,
        num_classes: int = 24,
        use_signal_conditioner: bool = True,
        use_token_adapter: bool = True,
        use_raw_adapter: bool = True,
        use_tf_adapter: bool = True,
        use_logit_adapter: bool = True,
        num_groups: int = 6,
    ):
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=(2, patch_size),
            in_c=1,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.signal_conditioner = ResidualSignalConditioner() if use_signal_conditioner else nn.Identity()
        self.token_adapter = TokenScaleAdapter(embed_dim) if use_token_adapter else None
        self.raw_adapter = RawIQFeatureAdapter(embed_dim) if use_raw_adapter else None
        self.tf_adapter = TimeFrequencyAdapter(embed_dim) if use_tf_adapter else None
        self.logit_adapter = (
            nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim // 2),
                nn.GELU(),
                _zero_init(nn.Linear(embed_dim // 2, num_classes)),
            )
            if use_logit_adapter
            else None
        )
        self.group_head = nn.Linear(embed_dim, num_groups)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.signal_conditioner(x)
        tokens = self.backbone.patch_embed(x)
        if self.token_adapter is not None:
            tokens = tokens + self.token_adapter(tokens)

        cls_token = self.backbone.cls_token.expand(tokens.shape[0], -1, -1)
        if self.backbone.dist_token is None:
            tokens = torch.cat((cls_token, tokens), dim=1)
        else:
            dist = self.backbone.dist_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_token, dist, tokens), dim=1)

        tokens = self.backbone.pos_drop(tokens + self.backbone.pos_embed)
        tokens = self.backbone.blocks(tokens)
        tokens = self.backbone.norm(tokens)
        if self.backbone.dist_token is None:
            features = self.backbone.pre_logits(tokens[:, 0])
        else:
            features = self.backbone.pre_logits((tokens[:, 0] + tokens[:, 1]) * 0.5)
        return features

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self._forward_features(x)
        if self.raw_adapter is not None:
            features = features + self.raw_adapter(x)
        if self.tf_adapter is not None:
            features = features + self.tf_adapter(x)
        return features

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.forward_features(x)
        logits = self.backbone.head(features)
        if self.logit_adapter is not None:
            logits = logits + self.logit_adapter(features)
        if not return_aux:
            return logits
        return logits, {"features": features, "group_logits": self.group_head(features)}


class EnhancedTrAMRV2(nn.Module):
    """Conservative Tr-AMR enhancement focused on difficult datagen boundaries.

    V2 deliberately removes the global raw-IQ/STFT shortcut adapters used by V1.
    The model stays lossless when initialized from an original Tr-AMR checkpoint,
    while adding two local, token-level residual paths that can be learned during
    boundary training.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 64,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 16,
        num_classes: int = 24,
        use_token_adapter: bool = True,
        use_pool_adapter: bool = True,
        num_groups: int = 6,
    ):
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=(2, patch_size),
            in_c=1,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.token_adapter = TokenDenoiseAdapter(embed_dim) if use_token_adapter else None
        self.pool_adapter = TokenPoolAdapter(embed_dim) if use_pool_adapter else None
        self.group_head = nn.Linear(embed_dim, num_groups)

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.patch_embed(x)
        if self.token_adapter is not None:
            tokens = tokens + self.token_adapter(tokens)

        cls_token = self.backbone.cls_token.expand(tokens.shape[0], -1, -1)
        if self.backbone.dist_token is None:
            tokens = torch.cat((cls_token, tokens), dim=1)
        else:
            dist = self.backbone.dist_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_token, dist, tokens), dim=1)

        tokens = self.backbone.pos_drop(tokens + self.backbone.pos_embed)
        tokens = self.backbone.blocks(tokens)
        return self.backbone.norm(tokens)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._forward_tokens(x)
        if self.backbone.dist_token is None:
            cls_feature = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            cls_feature = (tokens[:, 0] + tokens[:, 1]) * 0.5
            patch_tokens = tokens[:, 2:]
        if self.pool_adapter is not None:
            cls_feature = cls_feature + self.pool_adapter(cls_feature, patch_tokens)
        return self.backbone.pre_logits(cls_feature)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.forward_features(x)
        logits = self.backbone.head(features)
        if not return_aux:
            return logits
        return logits, {"features": features, "group_logits": self.group_head(features)}


class EnhancedTrAMRV3(nn.Module):
    """Boundary-oriented Tr-AMR enhancement.

    V3 keeps the original Tr-AMR decision path as the only non-residual path and
    adds lossless local token/context adapters. It intentionally avoids the V1
    raw-IQ, STFT, and logit shortcuts that can overfit source-like data while
    hurting difficult datagen boundaries.
    """

    def __init__(
        self,
        img_size: int = 1024,
        patch_size: int = 64,
        embed_dim: int = 256,
        depth: int = 8,
        num_heads: int = 16,
        num_classes: int = 24,
        use_token_adapter: bool = True,
        use_post_adapter: bool = True,
        use_pool_adapter: bool = True,
        num_groups: int = 6,
    ):
        super().__init__()
        self.backbone = VisionTransformer(
            img_size=img_size,
            patch_size=(2, patch_size),
            in_c=1,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            qkv_bias=True,
        )
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.token_adapter = BoundaryTokenAdapter(embed_dim) if use_token_adapter else None
        self.post_adapter = BoundaryTokenAdapter(embed_dim) if use_post_adapter else None
        self.pool_adapter = AttentiveTokenPoolAdapter(embed_dim) if use_pool_adapter else None
        self.group_head = nn.Linear(embed_dim, num_groups)

    def _forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone.patch_embed(x)
        if self.token_adapter is not None:
            tokens = tokens + self.token_adapter(tokens)

        cls_token = self.backbone.cls_token.expand(tokens.shape[0], -1, -1)
        if self.backbone.dist_token is None:
            tokens = torch.cat((cls_token, tokens), dim=1)
        else:
            dist = self.backbone.dist_token.expand(tokens.shape[0], -1, -1)
            tokens = torch.cat((cls_token, dist, tokens), dim=1)

        tokens = self.backbone.pos_drop(tokens + self.backbone.pos_embed)
        tokens = self.backbone.blocks(tokens)
        tokens = self.backbone.norm(tokens)
        if self.post_adapter is not None:
            tokens = tokens + self.post_adapter(tokens)
        return tokens

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._forward_tokens(x)
        if self.backbone.dist_token is None:
            cls_feature = tokens[:, 0]
            patch_tokens = tokens[:, 1:]
        else:
            cls_feature = (tokens[:, 0] + tokens[:, 1]) * 0.5
            patch_tokens = tokens[:, 2:]
        if self.pool_adapter is not None:
            cls_feature = cls_feature + self.pool_adapter(cls_feature, patch_tokens)
        return self.backbone.pre_logits(cls_feature)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        features = self.forward_features(x)
        logits = self.backbone.head(features)
        if not return_aux:
            return logits
        return logits, {"features": features, "group_logits": self.group_head(features)}


def build_enhanced_tr_amr(args: Namespace | object) -> EnhancedTrAMR:
    variant = str(getattr(args, "enhanced_variant", "v1")).lower()
    if variant == "v2":
        return EnhancedTrAMRV2(
            patch_size=int(getattr(args, "patch_size", 64)),
            embed_dim=int(getattr(args, "embed_dim", 256)),
            depth=int(getattr(args, "depth", 8)),
            num_heads=int(getattr(args, "num_heads", 16)),
            num_classes=int(getattr(args, "num_classes", 24)),
            use_token_adapter=not bool(getattr(args, "disable_token_adapter", False)),
            use_pool_adapter=not bool(getattr(args, "disable_pool_adapter", False)),
        )
    if variant == "v3":
        return EnhancedTrAMRV3(
            patch_size=int(getattr(args, "patch_size", 64)),
            embed_dim=int(getattr(args, "embed_dim", 256)),
            depth=int(getattr(args, "depth", 8)),
            num_heads=int(getattr(args, "num_heads", 16)),
            num_classes=int(getattr(args, "num_classes", 24)),
            use_token_adapter=not bool(getattr(args, "disable_token_adapter", False)),
            use_post_adapter=not bool(getattr(args, "disable_post_adapter", False)),
            use_pool_adapter=not bool(getattr(args, "disable_pool_adapter", False)),
        )
    if variant != "v1":
        raise ValueError(f"Unsupported enhanced_variant={variant!r}; expected 'v1', 'v2', or 'v3'.")
    return EnhancedTrAMR(
        patch_size=int(getattr(args, "patch_size", 64)),
        embed_dim=int(getattr(args, "embed_dim", 256)),
        depth=int(getattr(args, "depth", 8)),
        num_heads=int(getattr(args, "num_heads", 16)),
        num_classes=int(getattr(args, "num_classes", 24)),
        use_signal_conditioner=not bool(getattr(args, "disable_signal_conditioner", False)),
        use_token_adapter=not bool(getattr(args, "disable_token_adapter", False)),
        use_raw_adapter=not bool(getattr(args, "disable_raw_adapter", False)),
        use_tf_adapter=not bool(getattr(args, "disable_tf_adapter", False)),
        use_logit_adapter=not bool(getattr(args, "disable_logit_adapter", False)),
    )


def split_enhanced_parameter_groups(
    model: nn.Module,
    lr: float,
    weight_decay: float,
    backbone_lr_scale: float = 0.25,
    adapter_lr_scale: float = 1.0,
):
    backbone_params = []
    adapter_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            adapter_params.append(param)
    groups = []
    if backbone_params:
        groups.append(
            {
                "params": backbone_params,
                "lr": lr * float(backbone_lr_scale),
                "weight_decay": weight_decay,
                "name": "backbone",
            }
        )
    if adapter_params:
        groups.append(
            {
                "params": adapter_params,
                "lr": lr * float(adapter_lr_scale),
                "weight_decay": weight_decay,
                "name": "adapters",
            }
        )
    return groups


def _extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def load_tramr_or_enhanced_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    strict: bool = False,
):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state = _extract_state_dict(checkpoint)
    if any(key.startswith("backbone.") for key in state):
        missing, unexpected = model.load_state_dict(state, strict=strict)
    else:
        mapped = {f"backbone.{key}": value for key, value in state.items()}
        missing, unexpected = model.load_state_dict(mapped, strict=False)
    return checkpoint, list(missing), list(unexpected)


def freeze_backbone(model: nn.Module, frozen: bool = True) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = not frozen
