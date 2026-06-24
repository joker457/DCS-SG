from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import stft


@dataclass(frozen=True)
class StftFeatureConfig:
    iq_mode: str = "iq"
    feature_mode: str = "magphase"
    windows: tuple[int, ...] = (17, 31, 63)
    nfft: int = 128
    freq_bins: int = 32

    @property
    def source_channels(self) -> tuple[int, ...]:
        if self.iq_mode == "i-only":
            return (0,)
        if self.iq_mode == "q-only":
            return (1,)
        if self.iq_mode == "iq":
            return (0, 1)
        raise ValueError(f"Unsupported stft iq mode: {self.iq_mode}")

    @property
    def features_per_source(self) -> int:
        if self.feature_mode == "real":
            return 1
        if self.feature_mode in {"ri", "magphase"}:
            return 2
        raise ValueError(f"Unsupported stft feature mode: {self.feature_mode}")

    @property
    def output_channels(self) -> int:
        return len(self.source_channels) * self.features_per_source * len(self.windows)


def parse_windows(text: str | tuple[int, ...] | list[int]) -> tuple[int, ...]:
    if isinstance(text, tuple):
        values = text
    elif isinstance(text, list):
        values = tuple(int(x) for x in text)
    else:
        values = tuple(int(x) for x in str(text).replace(",", " ").split() if x.strip())
    if not values:
        raise ValueError("At least one STFT window is required.")
    for value in values:
        if value < 3:
            raise ValueError(f"STFT window must be >= 3, got {value}.")
    return values


def _fit_time_axis(feature: np.ndarray, target_frames: int) -> np.ndarray:
    current = int(feature.shape[-1])
    if current == target_frames:
        return feature
    if current > target_frames:
        start = (current - target_frames) // 2
        return feature[..., start : start + target_frames]
    pad_left = (target_frames - current) // 2
    pad_right = target_frames - current - pad_left
    return np.pad(feature, [(0, 0), (0, 0), (pad_left, pad_right)], mode="constant")


def compute_stft_features(iq: np.ndarray, config: StftFeatureConfig) -> np.ndarray:
    """Build model-ready STFT features with shape (C, freq_bins, time).

    The original IQFormer code uses only the real part of the I-channel STFT.
    This helper keeps that path available via iq_mode="i-only",
    feature_mode="real", windows=(31,), while also enabling richer variants.
    """
    iq_np = np.asarray(iq, dtype=np.float32)
    if iq_np.ndim != 2 or iq_np.shape[0] != 2:
        raise ValueError(f"Expected IQ sample with shape (2, L), got {iq_np.shape}.")

    target_frames = int(iq_np.shape[-1])
    channels: list[np.ndarray] = []
    for window in config.windows:
        noverlap = max(0, window - 1)
        for source_channel in config.source_channels:
            _, _, spectrum = stft(
                iq_np[source_channel],
                fs=1.0,
                window="blackman",
                nperseg=window,
                noverlap=noverlap,
                nfft=config.nfft,
                boundary="zeros",
                padded=True,
            )
            spectrum = spectrum[: config.freq_bins, :]
            if config.feature_mode == "real":
                parts = [np.real(spectrum)]
            elif config.feature_mode == "ri":
                parts = [np.real(spectrum), np.imag(spectrum)]
            elif config.feature_mode == "magphase":
                parts = [np.log1p(np.abs(spectrum)), np.angle(spectrum) / np.pi]
            else:
                raise ValueError(f"Unsupported stft feature mode: {config.feature_mode}")
            for part in parts:
                part = np.asarray(part, dtype=np.float32)
                part = _fit_time_axis(part[None, :, :], target_frames)
                channels.append(part[0])
    return np.stack(channels, axis=0).astype(np.float32, copy=False)
