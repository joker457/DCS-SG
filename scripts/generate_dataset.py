#!/usr/bin/env python
import argparse
import itertools
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from dcs_sg.config import (
    GRANULARITY_LEVEL_MODS,
    MOD_NAMES,
    OBS_LEVELS,
    SNR_VALUES,
    ModulationType,
    snr_db_to_level,
)
from dcs_sg.generator import TorchBatchGenerator
from dcs_sg.storage import BucketedSNRWriter


RAND_MOD = 2_147_483_647


def seed_everything(seed: int, deterministic: bool) -> None:
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")


def _splitmix64(x: np.ndarray) -> np.ndarray:
    x = x + np.uint64(0x9E3779B97F4A7C15)
    z = x.copy()
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def derive_sample_seeds(
    global_seed: int,
    snr_db: float,
    obs_level: int,
    mod: ModulationType,
    chan: np.ndarray,
    off: np.ndarray,
    rep: np.ndarray,
) -> np.ndarray:
    snr_key = int(round((float(snr_db) + 100.0) * 10.0))
    with np.errstate(over="ignore"):
        x = np.full(chan.shape, np.uint64(int(global_seed) & 0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        x ^= np.uint64(snr_key) * np.uint64(0xD6E8FEB86659FD93)
        x ^= np.uint64(int(obs_level) + 1) * np.uint64(0xA5A3564E27F1A2D1)
        x ^= np.uint64(int(mod) + 1) * np.uint64(0x9E3779B185EBCA87)
        x ^= chan.astype(np.uint64) * np.uint64(0xC2B2AE3D27D4EB4F)
        x ^= off.astype(np.uint64) * np.uint64(0x165667B19E3779F9)
        x ^= rep.astype(np.uint64) * np.uint64(0x85EBCA77C2B2AE63)
        mixed = _splitmix64(x)
    return (mixed % np.uint64(RAND_MOD - 1) + np.uint64(1)).astype(np.int64)


def derive_impairment_seeds(
    global_seed: int,
    snr_db: float,
    obs_level: int,
    chan: np.ndarray,
    off: np.ndarray,
    rep: np.ndarray,
) -> np.ndarray:
    snr_key = int(round((float(snr_db) + 100.0) * 10.0))
    with np.errstate(over="ignore"):
        x = np.full(chan.shape, np.uint64(int(global_seed) & 0xFFFFFFFFFFFFFFFF), dtype=np.uint64)
        x ^= np.uint64(0xA0761D6478BD642F)
        x ^= np.uint64(snr_key) * np.uint64(0xE7037ED1A0B428DB)
        x ^= np.uint64(int(obs_level) + 1) * np.uint64(0x8EBC6AF09C88C6E3)
        x ^= chan.astype(np.uint64) * np.uint64(0x589965CC75374CC3)
        x ^= off.astype(np.uint64) * np.uint64(0x1D8E4E27C47D124F)
        x ^= rep.astype(np.uint64) * np.uint64(0xEB44ACCAB455D165)
        mixed = _splitmix64(x)
    return (mixed % np.uint64(RAND_MOD - 1) + np.uint64(1)).astype(np.int64)


def parse_csv_values(text: str | None, cast, allowed: set | None = None) -> list:
    if text is None or str(text).strip().lower() in {"", "all", "*"}:
        return []
    values = []
    for item in str(text).replace(",", " ").split():
        value = cast(item)
        if allowed is not None and value not in allowed:
            raise argparse.ArgumentTypeError(f"Unsupported value {value!r}; allowed={sorted(allowed)}")
        values.append(value)
    return values


def parse_mods(text: str | None, granularity_level: int | None) -> list[ModulationType]:
    selected = set(range(len(MOD_NAMES)))
    if text is not None and str(text).strip().lower() not in {"", "all", "*"}:
        selected = set()
        name_to_id = {name.lower(): idx for idx, name in enumerate(MOD_NAMES)}
        for item in str(text).replace(",", " ").split():
            if item.isdigit():
                idx = int(item)
            else:
                key = item.lower()
                if key not in name_to_id:
                    raise argparse.ArgumentTypeError(f"Unknown modulation {item!r}; available={MOD_NAMES}")
                idx = name_to_id[key]
            if idx < 0 or idx >= len(MOD_NAMES):
                raise argparse.ArgumentTypeError(f"Modulation id out of range: {idx}")
            selected.add(idx)
    if granularity_level is not None:
        names = GRANULARITY_LEVEL_MODS[granularity_level]
        name_to_id = {name: idx for idx, name in enumerate(MOD_NAMES)}
        selected &= {name_to_id[name] for name in names}
    if not selected:
        raise argparse.ArgumentTypeError("No modulation remains after applying --mods/--granularity-level")
    return [ModulationType(idx) for idx in sorted(selected)]


def make_level_arrays(reps: int, chan_levels: list[int], off_levels: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    chan = []
    off = []
    rep_ids = []
    for r in range(reps):
        for c, o in itertools.product(chan_levels, off_levels):
            chan.append(c)
            off.append(o)
            rep_ids.append(r)
    return np.asarray(chan, dtype=np.int64), np.asarray(off, dtype=np.int64), np.asarray(rep_ids, dtype=np.int64)


def generate_snr_file(
    snr_db: float,
    output_dir: str,
    reps: int,
    batch_size: int,
    device: str,
    progress_interval: float,
    seed: int,
    deterministic: bool,
    obs_levels: list[int],
    mods: list[ModulationType],
    chan_levels: list[int],
    off_levels: list[int],
    source_mode: str,
    text_source: str | None,
) -> None:
    samples_per_obs = len(mods) * len(chan_levels) * len(off_levels) * reps
    samples_per_file = samples_per_obs * len(obs_levels)
    filename = f"amc_snr_{int(snr_db):+03d}dB.h5"
    path = Path(output_dir) / filename
    print(f"  SNR={snr_db:+.0f} dB: {samples_per_file:,} samples -> {filename}")

    generator = TorchBatchGenerator(device=device, deterministic=deterministic, source_mode=source_mode, text_source=text_source)
    writer = BucketedSNRWriter(
        path,
        snr_db,
        reps,
        seed=seed,
        deterministic=deterministic,
        obs_level_ids=obs_levels,
        mod_ids=[int(m) for m in mods],
        chan_levels=chan_levels,
        off_levels=off_levels,
        source_mode=source_mode,
        source_text_sha1=generator.source_text_sha1,
    )
    base_chan, base_off, base_rep = make_level_arrays(reps, chan_levels, off_levels)
    t0 = time.time()
    progress = tqdm(
        total=samples_per_file,
        desc=f"    SNR {snr_db:+.0f} dB",
        unit="samp",
        unit_scale=True,
        dynamic_ncols=True,
        mininterval=progress_interval,
    )
    try:
        for obs_level in obs_levels:
            for mod in mods:
                total = len(base_chan)
                for start in range(0, total, batch_size):
                    end = min(start + batch_size, total)
                    chan_np = base_chan[start:end]
                    off_np = base_off[start:end]
                    rep_np = base_rep[start:end]
                    chan = torch.from_numpy(chan_np)
                    off = torch.from_numpy(off_np)
                    sample_seeds = torch.from_numpy(derive_sample_seeds(seed, snr_db, obs_level, mod, chan_np, off_np, rep_np))
                    impairment_seeds = torch.from_numpy(derive_impairment_seeds(seed, snr_db, obs_level, chan_np, off_np, rep_np))
                    result = generator.generate_batch(mod, snr_db, obs_level, chan, off, sample_seeds, impairment_seeds)
                    writer.write_batch(obs_level, result)
                    progress.update(end - start)
    finally:
        writer.close()
        progress.close()
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.1f}s ({samples_per_file / elapsed:.0f} samples/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DCS-SG scale-controlled AMC datasets with a batched PyTorch backend")
    snr_group = parser.add_mutually_exclusive_group(required=True)
    snr_group.add_argument("--snr", type=float, help="Single SNR value, e.g. 0")
    snr_group.add_argument("--snrs", default=None, help="Comma/space separated SNR values, e.g. '0,2,4,6'")
    snr_group.add_argument("--snr-levels", default=None, help="Comma/space separated SNR demand levels 0..5")
    snr_group.add_argument("--all", action="store_true", help="Generate all configured SNR files")
    parser.add_argument("--output-dir", default="data_gpu", help="Output directory")
    parser.add_argument("--reps", type=int, default=300, help="Realizations per (mod, obs, chan, off) cell")
    parser.add_argument("--obs-levels", default="all", help="Comma/space separated observation levels 0..5")
    parser.add_argument("--chan-levels", default="all", help="Comma/space separated channel levels 0..5")
    parser.add_argument("--off-levels", default="all", help="Comma/space separated sync mismatch levels 0..5")
    parser.add_argument("--mods", default="all", help="Comma/space separated modulation names or ids, or all")
    parser.add_argument("--granularity-level", default=None, help="Optional class granularity level 0..5")
    parser.add_argument("--batch-size", type=int, default=1024, help="CUDA batch size")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Generation device")
    parser.add_argument("--progress-interval", type=float, default=1.0, help="tqdm refresh interval in seconds")
    parser.add_argument("--seed", type=int, default=20260426, help="Global random seed for reproducible generation")
    parser.add_argument("--source-mode", choices=["natural", "random"], default="natural", help="Signal source mode. natural uses text-like digital bits and audio-like analog messages; random keeps the original iid source behavior.")
    parser.add_argument("--text-source", default=None, help="Optional text file used by --source-mode natural for digital bit streams, e.g. RML-gen/source_material/gutenberg_shakespeare.txt")
    parser.add_argument("--fast-nondeterministic", action="store_true", help="Enable faster CUDA settings that may break bitwise reproducibility")
    args = parser.parse_args()

    if args.reps < 1:
        parser.error("--reps must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.progress_interval <= 0:
        parser.error("--progress-interval must be > 0")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is not available")

    obs_levels = parse_csv_values(args.obs_levels, int, set(range(len(OBS_LEVELS)))) or list(range(len(OBS_LEVELS)))
    chan_levels = parse_csv_values(args.chan_levels, int, set(range(6))) or list(range(6))
    off_levels = parse_csv_values(args.off_levels, int, set(range(6))) or list(range(6))
    granularity_level = None
    if args.granularity_level is not None and str(args.granularity_level).strip().lower() not in {"", "all", "full", "none", "*"}:
        granularity_level = int(args.granularity_level)
        if granularity_level not in range(6):
            parser.error("--granularity-level must be 0..5, full, none, or omitted")
    mods = parse_mods(args.mods, granularity_level)

    deterministic = not args.fast_nondeterministic
    seed_everything(args.seed, deterministic)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.all:
        snr_list = list(SNR_VALUES)
    elif args.snr is not None:
        snr_list = [args.snr]
    elif args.snrs is not None:
        snr_list = parse_csv_values(args.snrs, float)
    else:
        requested = set(parse_csv_values(args.snr_levels, int, set(range(6))))
        snr_list = [snr for snr in SNR_VALUES if snr_db_to_level(snr) in requested]
    if not snr_list:
        parser.error("No SNR values selected")

    print(f"Generating {len(snr_list)} file(s), reps={args.reps}, batch_size={args.batch_size}, device={args.device}, seed={args.seed}, deterministic={deterministic}")
    print(f"Source mode: {args.source_mode}" + (f", text_source={args.text_source}" if args.text_source else ""))
    print(f"Demand selection: snrs={snr_list}, snr_levels={sorted({snr_db_to_level(x) for x in snr_list})}, obs_levels={obs_levels}, chan_levels={chan_levels}, off_levels={off_levels}")
    print(f"Modulations ({len(mods)}): {[MOD_NAMES[int(m)] for m in mods]}")
    if args.device == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    t0 = time.time()
    for snr_db in snr_list:
        generate_snr_file(
            snr_db,
            args.output_dir,
            args.reps,
            args.batch_size,
            args.device,
            args.progress_interval,
            args.seed,
            deterministic,
            obs_levels,
            mods,
            chan_levels,
            off_levels,
            args.source_mode,
            args.text_source,
        )
    print(f"\nAll done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
