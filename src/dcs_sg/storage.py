import json
from pathlib import Path

import h5py
import numpy as np

from dcs_sg.config import DEMAND_DIMS, MOD_NAMES, OBS_LEVELS


class BucketedSNRWriter:
    def __init__(
        self,
        path: str | Path,
        snr_db: float,
        reps: int,
        seed: int | None = None,
        deterministic: bool = True,
        obs_level_ids: list[int] | None = None,
        mod_ids: list[int] | None = None,
        chan_levels: list[int] | None = None,
        off_levels: list[int] | None = None,
        source_mode: str = "natural",
        source_text_sha1: str | None = None,
    ):
        self.path = str(path)
        self.snr_db = float(snr_db)
        self.reps = int(reps)
        self.seed = None if seed is None else int(seed)
        self.deterministic = bool(deterministic)
        self.obs_level_ids = list(range(len(OBS_LEVELS))) if obs_level_ids is None else [int(x) for x in obs_level_ids]
        self.mod_ids = list(range(len(MOD_NAMES))) if mod_ids is None else [int(x) for x in mod_ids]
        self.chan_levels = list(range(6)) if chan_levels is None else [int(x) for x in chan_levels]
        self.off_levels = list(range(6)) if off_levels is None else [int(x) for x in off_levels]
        self.source_mode = str(source_mode)
        self.source_text_sha1 = source_text_sha1
        self.f = h5py.File(self.path, "w", libver="latest")
        self.f.attrs["snr_db"] = self.snr_db
        self.f.attrs["storage"] = "bucketed_by_obs_level"
        self.f.attrs["modulations"] = json.dumps(MOD_NAMES)
        self.f.attrs["demand_dims"] = json.dumps(DEMAND_DIMS)
        self.f.attrs["num_active_dims"] = 5
        self.f.attrs["obs_levels"] = json.dumps(OBS_LEVELS)
        self.f.attrs["selected_obs_levels"] = json.dumps(self.obs_level_ids)
        self.f.attrs["selected_mod_ids"] = json.dumps(self.mod_ids)
        self.f.attrs["selected_modulations"] = json.dumps([MOD_NAMES[i] for i in self.mod_ids])
        self.f.attrs["selected_chan_levels"] = json.dumps(self.chan_levels)
        self.f.attrs["selected_off_levels"] = json.dumps(self.off_levels)
        if self.seed is not None:
            self.f.attrs["seed"] = self.seed
        self.f.attrs["deterministic"] = self.deterministic
        self.f.attrs["seed_formula"] = "signal seed includes mod_type; channel/sync/noise seed excludes mod_type"
        self.f.attrs["source_mode"] = self.source_mode
        if self.source_text_sha1 is not None:
            self.f.attrs["source_text_sha1"] = self.source_text_sha1

        self.groups = {}
        self.offsets = {}
        samples_per_obs = len(self.mod_ids) * len(self.chan_levels) * len(self.off_levels) * reps
        root = self.f.create_group("data")
        for obs_level in self.obs_level_ids:
            leff = OBS_LEVELS[obs_level]
            g = root.create_group(f"obs_{obs_level}")
            g.attrs["obs_level"] = obs_level
            g.attrs["leff"] = leff
            chunks = (min(512, max(samples_per_obs, 1)), 2, leff)
            g.create_dataset("X", (samples_per_obs, 2, leff), dtype="float32", chunks=chunks)
            g.create_dataset("Y", (samples_per_obs,), dtype="int32")
            g.create_dataset("SNR", (samples_per_obs,), dtype="float32")
            g.create_dataset("demand", (samples_per_obs, len(DEMAND_DIMS)), dtype="int8")
            g.create_dataset("mod_names", data=np.array(MOD_NAMES, dtype=h5py.string_dtype()))
            self.groups[obs_level] = g
            self.offsets[obs_level] = 0

    def write_batch(self, obs_level: int, result: dict) -> None:
        g = self.groups[obs_level]
        n = result["X"].shape[0]
        start = self.offsets[obs_level]
        end = start + n
        g["X"][start:end] = result["X"]
        g["Y"][start:end] = result["Y"]
        g["SNR"][start:end] = result["SNR"]
        g["demand"][start:end] = result["demand"]
        self.offsets[obs_level] = end

    def close(self) -> None:
        self.f.close()
