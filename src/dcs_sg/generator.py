from __future__ import annotations

import hashlib
import math
from pathlib import Path
import warnings

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore", message="Plan failed with a cudnnException.*")

from dcs_sg.config import (
    ANALOG_TYPES,
    CPM_TYPES,
    DEMAND_DIMS,
    MOD_NAMES,
    NUM_TAPS,
    OBS_LEVELS,
    OQPSK_TYPES,
    ROLLOFF,
    RRC_BUFFER_SYMBOLS,
    SPS,
    ModulationType,
    snr_db_to_level,
)


DEFAULT_TEXT_SOURCE = (
    "The signal bears a message before it becomes a waveform. "
    "Words, pauses, numbers, and repeated phrases create structure in the bit stream. "
    "A receiver sees only samples, but the transmitter began with language, timing, "
    "and small variations that are not quite white noise. "
    "We use this public domain style corpus as a compact stand in for the text source "
    "used by classic RadioML generators. "
)


class TorchBatchGenerator:
    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
        deterministic: bool = True,
        source_mode: str = "natural",
        text_source: str | Path | None = None,
    ):
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        if source_mode not in {"random", "natural"}:
            raise ValueError("source_mode must be 'random' or 'natural'")
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self.source_mode = source_mode
        if self.device.type == "cuda":
            if deterministic:
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
                torch.set_float32_matmul_precision("highest")
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
        self.dtype = dtype
        self.complex_dtype = torch.complex64 if dtype == torch.float32 else torch.complex128
        self.rrc = self._rrc_filter(SPS, ROLLOFF, NUM_TAPS).to(self.device, self.dtype)
        self.constellations = self._make_constellations()
        self.source_bits = self._load_source_bits(text_source).to(self.device)
        self.source_text_sha1 = self._source_text_sha1(text_source)

    def _source_text_bytes(self, text_source: str | Path | None) -> bytes:
        if text_source is not None:
            path = Path(text_source)
            if not path.exists():
                raise FileNotFoundError(f"text_source does not exist: {path}")
            data = path.read_bytes()
            if data:
                return data
        return DEFAULT_TEXT_SOURCE.encode("utf-8")

    def _source_text_sha1(self, text_source: str | Path | None) -> str:
        data = self._source_text_bytes(text_source)
        return hashlib.sha1(data).hexdigest()

    def _load_source_bits(self, text_source: str | Path | None) -> torch.Tensor:
        data = self._source_text_bytes(text_source)
        raw = np.frombuffer(data, dtype=np.uint8)
        bits = np.unpackbits(raw, bitorder="big")
        if bits.size == 0:
            bits = np.unpackbits(np.frombuffer(DEFAULT_TEXT_SOURCE.encode("utf-8"), dtype=np.uint8), bitorder="big")
        return torch.tensor(bits.astype(np.uint8), dtype=torch.uint8)

    @torch.no_grad()
    def generate_batch(
        self,
        mod_type: ModulationType,
        snr_db: float,
        obs_level: int,
        chan_levels: torch.Tensor,
        off_levels: torch.Tensor,
        sample_seeds: torch.Tensor,
        impairment_seeds: torch.Tensor | None = None,
    ) -> dict:
        sample_seeds = sample_seeds.to(self.device, torch.long)
        if impairment_seeds is None:
            impairment_seeds = sample_seeds
        else:
            impairment_seeds = impairment_seeds.to(self.device, torch.long)
        chan_levels = chan_levels.to(self.device, torch.long)
        off_levels = off_levels.to(self.device, torch.long)
        batch = int(chan_levels.numel())
        leff = OBS_LEVELS[obs_level]
        context = max(8 * SPS, leff // 2)
        burst_len = leff + 2 * context
        generation_len = burst_len + SPS

        shaped = self._generate_modulated(mod_type, batch, generation_len, sample_seeds)
        symbol_offsets = self._randint(impairment_seeds, (), 10, SPS)
        shaped = self._batched_take(shaped, symbol_offsets, burst_len)
        initial_phase = (self._rand_uniform(impairment_seeds, (), 11) * 2.0 - 1.0) * math.pi
        shaped = shaped * torch.exp(1j * initial_phase.to(self.complex_dtype))[:, None]

        shaped = self._apply_channel(shaped, chan_levels, impairment_seeds)
        shaped = self._apply_sync(shaped, off_levels, impairment_seeds)
        power = shaped.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-12)
        shaped = shaped / torch.sqrt(power)

        rx_len = burst_len + 2 * context
        burst_offsets = self._randint(impairment_seeds, (), 20, context + 1)
        rx = torch.zeros((batch, rx_len), dtype=self.complex_dtype, device=self.device)
        idx = burst_offsets[:, None] + torch.arange(burst_len, device=self.device)[None, :]
        rx.scatter_(1, idx, shaped)
        rx = self._add_awgn(rx, snr_db, impairment_seeds)

        crop_start = self._sample_crop_starts(burst_offsets, burst_len, leff, rx_len, off_levels, impairment_seeds)
        cropped = self._batched_take(rx, crop_start, leff)
        x = torch.stack([cropped.real, cropped.imag], dim=1).to(torch.float32).cpu().numpy()

        y = np.full((batch,), int(mod_type), dtype=np.int32)
        demand = np.full((batch, len(DEMAND_DIMS)), -1, dtype=np.int8)
        demand[:, 0] = snr_db_to_level(snr_db)
        demand[:, 1] = obs_level
        demand[:, 2] = chan_levels.cpu().numpy().astype(np.int8)
        demand[:, 3] = off_levels.cpu().numpy().astype(np.int8)
        demand[:, 4] = int(mod_type)
        snr = np.full((batch,), float(snr_db), dtype=np.float32)
        return {"X": x, "Y": y, "SNR": snr, "demand": demand}

    def _generate_modulated(self, mod_type: ModulationType, batch: int, generation_len: int, sample_seeds: torch.Tensor) -> torch.Tensor:
        mt = ModulationType(mod_type)
        if mt in ANALOG_TYPES:
            return self._generate_analog(mt, batch, generation_len, sample_seeds)
        if mt in CPM_TYPES:
            return self._generate_gmsk(batch, generation_len, sample_seeds)
        if mt in OQPSK_TYPES:
            return self._generate_oqpsk(batch, generation_len, sample_seeds)
        return self._generate_digital(mt, batch, generation_len, sample_seeds)

    def _generate_digital(self, mt: ModulationType, batch: int, generation_len: int, sample_seeds: torch.Tensor) -> torch.Tensor:
        num_symbols = generation_len // SPS + RRC_BUFFER_SYMBOLS
        const = self.constellations[mt]
        symbols = self._source_symbols(sample_seeds, num_symbols, int(const.numel()), 100)
        baseband = const[symbols]
        return self._pulse_shape(baseband)

    def _generate_oqpsk(self, batch: int, generation_len: int, sample_seeds: torch.Tensor) -> torch.Tensor:
        num_symbols = generation_len // SPS + RRC_BUFFER_SYMBOLS
        symbols = self._source_symbols(sample_seeds, num_symbols, 4, 110)
        i_syms = (2.0 * torch.bitwise_right_shift(symbols, 1).to(self.dtype) - 1.0) / math.sqrt(2.0)
        q_syms = (2.0 * torch.bitwise_and(symbols, 1).to(self.dtype) - 1.0) / math.sqrt(2.0)
        i_shaped = self._pulse_shape_real(i_syms)
        q_shaped = self._pulse_shape_real(q_syms)
        q_delay = torch.zeros_like(q_shaped)
        half = SPS // 2
        q_delay[:, half:] = q_shaped[:, :-half]
        return torch.complex(i_shaped, q_delay)

    def _generate_gmsk(self, batch: int, generation_len: int, sample_seeds: torch.Tensor) -> torch.Tensor:
        num_symbols = generation_len // SPS + RRC_BUFFER_SYMBOLS
        symbols = self._source_symbols(sample_seeds, num_symbols, 2, 120).to(self.dtype) * 2.0 - 1.0
        up = torch.zeros(batch, num_symbols * SPS, device=self.device, dtype=self.dtype)
        up[:, ::SPS] = symbols
        span = 3
        t = torch.arange(-span * SPS, span * SPS + 1, device=self.device, dtype=self.dtype) / SPS
        bt = 0.3
        sigma = math.sqrt(math.log(2.0)) / (2.0 * math.pi * bt)
        g = (1.0 / (math.sqrt(2.0 * math.pi) * sigma)) * torch.exp(-(t ** 2) / (2.0 * sigma ** 2))
        g = g / g.sum()
        phase_inc = self._conv_full_real(up, g)[:, : generation_len]
        phase = math.pi * 0.5 * self._cumsum_time(phase_inc)
        return torch.exp(1j * phase.to(self.complex_dtype))

    def _generate_analog(self, mt: ModulationType, batch: int, generation_len: int, sample_seeds: torch.Tensor) -> torch.Tensor:
        bandwidth = 0.1 if mt == ModulationType.FM else 0.2
        x = self._generate_message(batch, generation_len, bandwidth, sample_seeds)
        if mt == ModulationType.AM_DSB_SC:
            return torch.complex(x, torch.zeros_like(x))
        if mt == ModulationType.AM_DSB_WC:
            return torch.complex(1.0 + 0.8 * x, torch.zeros_like(x))
        if mt in (ModulationType.AM_SSB_SC, ModulationType.AM_SSB_WC):
            analytic = self._hilbert(x)
            if mt == ModulationType.AM_SSB_WC:
                analytic = 1.0 + 0.8 * analytic
            return analytic.to(self.complex_dtype)
        phase = 2.0 * math.pi * 0.5 * self._cumsum_time(x)
        return torch.exp(1j * phase.to(self.complex_dtype))

    def _generate_message(self, batch: int, n: int, bandwidth_factor: float, sample_seeds: torch.Tensor) -> torch.Tensor:
        if self.source_mode == "natural":
            return self._generate_audio_like_message(batch, n, bandwidth_factor, sample_seeds)
        raw = self._randn(sample_seeds, (n,), 130)
        kernel_size = max(int(n * bandwidth_factor), 8)
        kernel = torch.ones(kernel_size, device=self.device, dtype=self.dtype) / kernel_size
        filtered = self._conv_same_real(raw, kernel)
        filtered = filtered - filtered.mean(dim=1, keepdim=True)
        peak = filtered.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        return filtered / peak

    def _generate_audio_like_message(self, batch: int, n: int, bandwidth_factor: float, sample_seeds: torch.Tensor) -> torch.Tensor:
        t = torch.arange(n, device=self.device, dtype=self.dtype)[None, :]
        max_freq = max(0.02, min(0.18, float(bandwidth_factor)))
        f1 = 0.006 + self._rand_uniform(sample_seeds, (), 131) * (0.030 - 0.006)
        f2 = 0.025 + self._rand_uniform(sample_seeds, (), 132) * max(max_freq - 0.025, 0.005)
        f3 = 0.045 + self._rand_uniform(sample_seeds, (), 133) * max(max_freq - 0.045, 0.005)
        p1 = self._rand_uniform(sample_seeds, (), 134) * 2.0 * math.pi
        p2 = self._rand_uniform(sample_seeds, (), 135) * 2.0 * math.pi
        p3 = self._rand_uniform(sample_seeds, (), 136) * 2.0 * math.pi
        a2 = 0.25 + 0.35 * self._rand_uniform(sample_seeds, (), 137)
        a3 = 0.10 + 0.25 * self._rand_uniform(sample_seeds, (), 138)
        voiced = (
            torch.sin(2.0 * math.pi * f1[:, None] * t + p1[:, None])
            + a2[:, None] * torch.sin(2.0 * math.pi * f2[:, None] * t + p2[:, None])
            + a3[:, None] * torch.sin(2.0 * math.pi * f3[:, None] * t + p3[:, None])
        )
        env_freq = 0.0015 + 0.004 * self._rand_uniform(sample_seeds, (), 139)
        env_phase = self._rand_uniform(sample_seeds, (), 140) * 2.0 * math.pi
        envelope = 0.60 + 0.30 * torch.sin(2.0 * math.pi * env_freq[:, None] * t + env_phase[:, None])
        breath = self._randn(sample_seeds, (n,), 141)
        kernel_size = max(int(n * max(0.025, bandwidth_factor * 0.35)), 8)
        kernel = torch.hann_window(kernel_size, device=self.device, dtype=self.dtype)
        kernel = kernel / kernel.sum().clamp_min(1e-12)
        breath = self._conv_same_real(breath, kernel)
        msg = envelope * voiced + 0.12 * breath
        msg = msg - msg.mean(dim=1, keepdim=True)
        peak = msg.abs().amax(dim=1, keepdim=True).clamp_min(1e-6)
        return msg / peak

    def _source_symbols(self, sample_seeds: torch.Tensor, num_symbols: int, constellation_size: int, stream: int) -> torch.Tensor:
        if self.source_mode == "random":
            return self._randint(sample_seeds, (num_symbols,), stream, constellation_size)
        bits_per_symbol = int(math.ceil(math.log2(constellation_size)))
        bits = self._natural_bits(sample_seeds, num_symbols * bits_per_symbol, stream)
        bit_groups = bits.reshape(bits.shape[0], num_symbols, bits_per_symbol)
        weights = (2 ** torch.arange(bits_per_symbol - 1, -1, -1, device=self.device, dtype=torch.long))[None, None, :]
        symbols = torch.sum(bit_groups.to(torch.long) * weights, dim=2)
        return symbols.remainder(constellation_size)

    def _natural_bits(self, sample_seeds: torch.Tensor, length: int, stream: int) -> torch.Tensor:
        source_len = int(self.source_bits.numel())
        offsets = self._randint(sample_seeds, (), stream + 700, source_len)
        idx = (offsets[:, None] + torch.arange(length, device=self.device, dtype=torch.long)[None, :]).remainder(source_len)
        text_bits = self.source_bits[idx]
        scrambler_len = 256
        scrambler = self._randint(sample_seeds, (scrambler_len,), stream + 701, 2).to(torch.uint8)
        scrambler_idx = torch.arange(length, device=self.device, dtype=torch.long).remainder(scrambler_len)
        return torch.bitwise_xor(text_bits, scrambler[:, scrambler_idx])

    def _apply_channel(self, signal: torch.Tensor, levels: torch.Tensor, sample_seeds: torch.Tensor) -> torch.Tensor:
        batch, n = signal.shape
        max_paths = 12
        delays, powers, k_factor, doppler = self._sample_channel_params(levels, max_paths, sample_seeds)
        out = torch.zeros_like(signal)
        sample_idx = torch.arange(n, device=self.device)[None, :]
        time = torch.arange(n, device=self.device, dtype=self.dtype)[None, :]
        for p in range(max_paths):
            delay = delays[:, p]
            valid_path = powers[:, p] > 0
            src_idx = sample_idx - delay[:, None]
            valid = (src_idx >= 0) & valid_path[:, None]
            src_idx = src_idx.clamp(0, n - 1)
            delayed = torch.gather(signal, 1, src_idx)
            delayed = torch.where(valid, delayed, torch.zeros_like(delayed))

            phase0 = self._rand_uniform(sample_seeds, (), 300 + 3 * p)[:, None] * 2.0 * math.pi
            fd = doppler[:, None] * (self._rand_uniform(sample_seeds, (), 301 + 3 * p)[:, None] * 2.0 - 1.0)
            scatter = torch.exp(1j * (2.0 * math.pi * fd * time / SPS + phase0).to(self.complex_dtype))
            los_phase = self._rand_uniform(sample_seeds, (), 302 + 3 * p)[:, None] * 2.0 * math.pi
            los = torch.exp(1j * los_phase.to(self.complex_dtype))
            finite_k = torch.isfinite(k_factor)
            k = torch.where(finite_k, k_factor, torch.zeros_like(k_factor))[:, None]
            coeff_rician = torch.sqrt(k / (k + 1.0)) * los + torch.sqrt(1.0 / (k + 1.0)) * scatter
            coeff = torch.where(finite_k[:, None], coeff_rician, scatter)
            coeff = torch.where(levels[:, None] == 0, torch.ones_like(coeff), coeff)
            out = out + torch.sqrt(powers[:, p].clamp_min(0.0))[:, None].to(self.complex_dtype) * coeff * delayed
        power = out.abs().square().mean(dim=1, keepdim=True).clamp_min(1e-12)
        return out / torch.sqrt(power)

    def _sample_channel_params(self, levels: torch.Tensor, max_paths: int, sample_seeds: torch.Tensor):
        batch = levels.numel()
        delays = torch.zeros(batch, max_paths, device=self.device, dtype=torch.long)
        powers = torch.zeros(batch, max_paths, device=self.device, dtype=self.dtype)
        k_factor = torch.full((batch,), float("inf"), device=self.device, dtype=self.dtype)
        doppler = torch.zeros(batch, device=self.device, dtype=self.dtype)
        cfg = self._channel_cfg()
        for level, (pmin, pmax, dmin, dmax, fdmin, fdmax, kmin, kmax) in cfg.items():
            mask = levels == level
            count = int(mask.sum().item())
            if count == 0:
                continue
            local_seeds = sample_seeds[mask]
            stream = 200 + 20 * level
            n_paths = self._randint(local_seeds, (), stream, pmax - pmin + 1) + pmin
            max_delay_sym = self._rand_uniform(local_seeds, (), stream + 1) * (dmax - dmin) + dmin
            max_delay = torch.clamp((max_delay_sym * SPS).round().long(), min=1)
            local_delays = self._rand_uniform(local_seeds, (max_paths,), stream + 2).mul(max_delay[:, None].to(self.dtype)).round().long()
            local_delays[:, 0] = 0
            path_idx = torch.arange(max_paths, device=self.device)[None, :]
            active = path_idx < n_paths[:, None]
            local_delays = torch.where(active, local_delays, torch.zeros_like(local_delays))
            delay_sym = local_delays.to(self.dtype) / SPS
            decay = 1.0 + 2.0 * self._rand_uniform(local_seeds, (1,), stream + 3)
            local_powers = torch.exp(-delay_sym / decay) * active.to(self.dtype)
            local_powers = local_powers / local_powers.sum(dim=1, keepdim=True).clamp_min(1e-12)
            delays[mask] = local_delays
            powers[mask] = local_powers
            doppler[mask] = self._rand_uniform(local_seeds, (), stream + 4) * (fdmax - fdmin) + fdmin
            if math.isnan(kmin):
                k_factor[mask] = float("nan")
            else:
                k_db = self._rand_uniform(local_seeds, (), stream + 5) * (kmax - kmin) + kmin
                k_factor[mask] = torch.pow(torch.tensor(10.0, device=self.device, dtype=self.dtype), k_db / 10.0)
        k_factor = torch.where(torch.isnan(k_factor), torch.full_like(k_factor, float("nan")), k_factor)
        return delays, powers, k_factor, doppler

    def _channel_cfg(self):
        return {
            0: (1, 1, 0.0, 0.0, 0.0, 0.0, float("inf"), float("inf")),
            1: (1, 2, 0.0, 0.25, 0.0, 0.003, 12.0, 20.0),
            2: (2, 3, 0.15, 0.60, 0.003, 0.010, 5.0, 12.0),
            3: (4, 6, 0.80, 2.00, 0.018, 0.050, -2.0, 5.0),
            4: (6, 10, 1.50, 4.00, 0.045, 0.110, -8.0, 2.0),
            5: (8, 12, 2.50, 6.00, 0.080, 0.180, float("nan"), float("nan")),
        }

    def _apply_sync(self, signal: torch.Tensor, levels: torch.Tensor, sample_seeds: torch.Tensor) -> torch.Tensor:
        batch, n = signal.shape
        q = self._sample_sync_components(levels, sample_seeds)
        cfo_scale, timing_scale, sco_scale = self._sync_scales()
        cfo = cfo_scale * q[:, 0]
        cpo = math.pi * q[:, 1]
        to_samples = timing_scale * SPS * q[:, 2]
        sco_ppm = sco_scale * q[:, 3]

        signal = self._apply_sco(signal, sco_ppm)
        signal = self._apply_fractional_delay(signal, to_samples)
        t = torch.arange(n, device=self.device, dtype=self.dtype)[None, :] / SPS
        phase = 2.0 * math.pi * cfo[:, None] * t + cpo[:, None]
        return signal * torch.exp(1j * phase.to(self.complex_dtype))

    def _sample_sync_components(self, levels: torch.Tensor, sample_seeds: torch.Tensor) -> torch.Tensor:
        ranges = torch.tensor(self._sync_ranges(), device=self.device, dtype=self.dtype)
        lo = ranges[levels, 0]
        hi = ranges[levels, 1]
        target = lo + self._rand_uniform(sample_seeds, (), 400) * (hi - lo)
        weights = torch.tensor([0.35, 0.20, 0.30, 0.15], device=self.device, dtype=self.dtype)
        raw = self._rand_uniform(sample_seeds, (4,), 401).clamp_min(1e-6)
        raw = raw / raw.sum(dim=1, keepdim=True)
        q = target[:, None] * raw / weights[None, :]
        return q.clamp(0.0, max(1.0, float(ranges.max().item())))

    def _sync_ranges(self):
        return [
            [0.0, 0.0],
            [0.0, 0.10],
            [0.10, 0.25],
            [0.35, 0.70],
            [0.70, 1.10],
            [1.10, 1.55],
        ]

    def _sync_scales(self):
        return 0.09, 1.10, 800.0

    def _apply_fractional_delay(self, signal: torch.Tensor, delay_samples: torch.Tensor) -> torch.Tensor:
        n = signal.shape[1]
        freqs = torch.fft.fftfreq(n, device=self.device, dtype=self.dtype)[None, :]
        h = torch.exp((-1j * 2.0 * math.pi * delay_samples[:, None] * freqs).to(self.complex_dtype))
        return torch.fft.ifft(torch.fft.fft(signal, dim=1) * h, dim=1)

    def _apply_sco(self, signal: torch.Tensor, ppm: torch.Tensor) -> torch.Tensor:
        batch, n = signal.shape
        ratio = 1.0 + ppm * 1e-6
        pos = torch.arange(n, device=self.device, dtype=self.dtype)[None, :] * ratio[:, None]
        pos0 = torch.floor(pos).long().clamp(0, n - 1)
        pos1 = (pos0 + 1).clamp(0, n - 1)
        frac = (pos - pos0.to(self.dtype)).clamp(0.0, 1.0)
        y0 = torch.gather(signal, 1, pos0)
        y1 = torch.gather(signal, 1, pos1)
        return y0 * (1.0 - frac) + y1 * frac

    def _rand_uniform(self, sample_seeds: torch.Tensor, shape_tail: tuple[int, ...] | int, stream: int) -> torch.Tensor:
        if isinstance(shape_tail, int):
            shape_tail = (shape_tail,)
        elif shape_tail == ():
            shape_tail = tuple()
        else:
            shape_tail = tuple(shape_tail)
        batch = int(sample_seeds.numel())
        tail_numel = int(math.prod(shape_tail)) if shape_tail else 1
        mod = 2_147_483_647
        seeds = sample_seeds.to(self.device, torch.long).view(batch, 1).remainder(mod)
        idx = torch.arange(tail_numel, device=self.device, dtype=torch.long).view(1, tail_numel)
        x = (seeds + (int(stream) + 1) * 1_000_003 + (idx + 1) * 97_531).remainder(mod)
        x = (x * 48_271 + 1).remainder(mod)
        x = (x * 69_621 + 7).remainder(mod)
        out = (x.to(self.dtype) + 0.5) / float(mod)
        return out.reshape((batch,) + shape_tail) if shape_tail else out.reshape(batch)

    def _randint(self, sample_seeds: torch.Tensor, shape_tail: tuple[int, ...] | int, stream: int, high: int) -> torch.Tensor:
        if high <= 0:
            raise ValueError("high must be positive")
        u = self._rand_uniform(sample_seeds, shape_tail, stream)
        return torch.floor(u * int(high)).long().clamp(max=int(high) - 1)

    def _randn(self, sample_seeds: torch.Tensor, shape_tail: tuple[int, ...] | int, stream: int) -> torch.Tensor:
        u1 = self._rand_uniform(sample_seeds, shape_tail, stream).clamp_min(1e-7)
        u2 = self._rand_uniform(sample_seeds, shape_tail, stream + 10_000)
        return torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)

    def _add_awgn(self, signal: torch.Tensor, snr_db: float, sample_seeds: torch.Tensor) -> torch.Tensor:
        snr_linear = 10 ** (float(snr_db) / 10.0)
        noise_power = 1.0 / snr_linear
        noise_real = self._randn(sample_seeds, (signal.shape[1],), 500)
        noise_imag = self._randn(sample_seeds, (signal.shape[1],), 501)
        noise = torch.complex(noise_real, noise_imag) * math.sqrt(noise_power / 2.0)
        return signal + noise

    def _sample_crop_starts(self, burst_offsets: torch.Tensor, burst_len: int, leff: int, rx_len: int, off_levels: torch.Tensor, sample_seeds: torch.Tensor) -> torch.Tensor:
        overlap = torch.tensor([0.75, 0.75, 0.68, 0.58, 0.48, 0.38], device=self.device, dtype=self.dtype)[off_levels]
        min_samples = torch.ceil(overlap * float(leff)).long()
        lo = torch.clamp(burst_offsets - leff + min_samples, min=0)
        hi = torch.clamp(burst_offsets + burst_len - min_samples, max=rx_len - leff)
        span = (hi - lo + 1).clamp_min(1)
        return lo + torch.floor(self._rand_uniform(sample_seeds, (), 600) * span.to(self.dtype)).long()

    def _batched_take(self, x: torch.Tensor, starts: torch.Tensor, length: int) -> torch.Tensor:
        idx = starts[:, None] + torch.arange(length, device=self.device)[None, :]
        return torch.gather(x, 1, idx)

    def _pulse_shape(self, symbols: torch.Tensor) -> torch.Tensor:
        up = torch.zeros(symbols.shape[0], symbols.shape[1] * SPS, device=self.device, dtype=self.complex_dtype)
        up[:, ::SPS] = symbols
        return self._conv_full_complex(up, self.rrc)

    def _pulse_shape_real(self, symbols: torch.Tensor) -> torch.Tensor:
        up = torch.zeros(symbols.shape[0], symbols.shape[1] * SPS, device=self.device, dtype=self.dtype)
        up[:, ::SPS] = symbols
        return self._conv_full_real(up, self.rrc)

    def _conv_full_complex(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        real = self._conv_full_real(x.real, h)
        imag = self._conv_full_real(x.imag, h)
        return torch.complex(real, imag)

    def _conv_full_real(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if not self.deterministic:
            return F.conv1d(x[:, None, :], h[None, None, :], padding=h.numel() - 1)[:, 0, :]
        out = torch.zeros(x.shape[0], x.shape[1] + h.numel() - 1, device=self.device, dtype=x.dtype)
        for i in range(h.numel()):
            out[:, i:i + x.shape[1]] = out[:, i:i + x.shape[1]] + x * h[i]
        return out

    def _conv_same_real(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        full = self._conv_full_real(x, h)
        start = (h.numel() - 1) // 2
        return full[:, start:start + x.shape[1]]

    def _cumsum_time(self, x: torch.Tensor) -> torch.Tensor:
        if not (self.deterministic and x.is_cuda):
            return torch.cumsum(x, dim=1)
        acc = torch.zeros_like(x[:, 0])
        out = torch.empty_like(x)
        for i in range(x.shape[1]):
            acc = acc + x[:, i]
            out[:, i] = acc
        return out

    def _hilbert(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[1]
        X = torch.fft.fft(x, dim=1)
        h = torch.zeros(n, device=self.device, dtype=self.dtype)
        if n % 2 == 0:
            h[0] = 1.0
            h[n // 2] = 1.0
            h[1:n // 2] = 2.0
        else:
            h[0] = 1.0
            h[1:(n + 1) // 2] = 2.0
        return torch.fft.ifft(X * h[None, :], dim=1)

    def _rrc_filter(self, sps: int, rolloff: float, num_taps: int) -> torch.Tensor:
        ntaps = sps * num_taps + 1
        t = torch.arange(ntaps, dtype=torch.float64) - ntaps // 2
        t = t / sps
        h = torch.zeros(ntaps, dtype=torch.float64)
        alpha = rolloff
        for i, ti in enumerate(t.tolist()):
            if abs(ti) < 1e-12:
                h[i] = 1.0 - alpha + 4.0 * alpha / math.pi
            elif abs(abs(ti) - 1.0 / (4.0 * alpha)) < 1e-12:
                h[i] = (alpha / math.sqrt(2.0)) * ((1 + 2 / math.pi) * math.sin(math.pi / (4 * alpha)) + (1 - 2 / math.pi) * math.cos(math.pi / (4 * alpha)))
            else:
                num = math.sin(math.pi * ti * (1 - alpha)) + 4 * alpha * ti * math.cos(math.pi * ti * (1 + alpha))
                den = math.pi * ti * (1 - (4 * alpha * ti) ** 2)
                h[i] = num / den
        h = h / torch.sqrt(torch.sum(h ** 2) / sps)
        return h.to(torch.float32)

    def _make_constellations(self):
        def norm(c):
            c = np.asarray(c, dtype=np.complex64)
            return torch.tensor(c / np.sqrt(np.mean(np.abs(c) ** 2)), device=self.device, dtype=self.complex_dtype)
        const = {}
        for mt in [ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8, ModulationType.PSK16, ModulationType.PSK32]:
            m = {ModulationType.BPSK: 2, ModulationType.QPSK: 4, ModulationType.PSK8: 8, ModulationType.PSK16: 16, ModulationType.PSK32: 32}[mt]
            const[mt] = norm(np.exp(1j * 2 * np.pi * np.arange(m) / m))
        const[ModulationType.OOK] = norm(np.array([0.0, 1.0]))
        const[ModulationType.ASK4] = norm(2.0 * np.arange(4) - 3)
        const[ModulationType.ASK8] = norm(2.0 * np.arange(8) - 7)
        for mt, m in [(ModulationType.QAM16, 16), (ModulationType.QAM64, 64), (ModulationType.QAM256, 256)]:
            n = int(np.sqrt(m)); vals = 2.0 * np.arange(n) - (n - 1); const[mt] = norm((vals[:, None] + 1j * vals[None, :]).ravel())
        for mt, shape in [(ModulationType.QAM32, (4, 8)), (ModulationType.QAM128, (8, 16))]:
            vi = 2.0 * np.arange(shape[0]) - (shape[0] - 1); vq = 2.0 * np.arange(shape[1]) - (shape[1] - 1); const[mt] = norm((vi[:, None] + 1j * vq[None, :]).ravel())
        apsk = {
            ModulationType.APSK16: [(4, 1.0), (12, 2.72)],
            ModulationType.APSK32: [(4, 1.0), (12, 2.0), (16, 3.17)],
            ModulationType.APSK64: [(4, 1.0), (12, 1.8), (20, 2.6), (28, 3.5)],
            ModulationType.APSK128: [(4, 1.0), (12, 1.6), (20, 2.2), (28, 2.8), (64, 3.6)],
        }
        for mt, rings in apsk.items():
            pts = []
            for ri, (num, radius) in enumerate(rings):
                offset = 0.0 if ri == 0 else np.pi / num
                for k in range(num):
                    pts.append(radius * np.exp(1j * (2 * np.pi * k / num + offset)))
            const[mt] = norm(np.array(pts))
        return const
