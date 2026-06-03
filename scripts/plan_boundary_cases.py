#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dcs_sg.config import GRANULARITY_LEVEL_MODS, SNR_VALUES, snr_db_to_level


DIM_ORDER = ("snr", "obs", "chan", "off", "gra")

MODEL_PROFILES = {
    "tramr": {"source_snrs": list(range(0, 31, 2)), "base_obs": 1, "base_chan": 2, "base_off": 1, "target_length": 1024},
    "mcnet": {"source_snrs": list(range(-20, 31, 2)), "base_obs": 1, "base_chan": 2, "base_off": 1, "target_length": 1024},
    "iqformer": {"source_snrs": list(range(-20, 19, 2)), "base_obs": 4, "base_chan": 2, "base_off": 1, "target_length": 128},
    "ea": {"source_snrs": list(range(-20, 31, 2)), "base_obs": 1, "base_chan": 2, "base_off": 1, "target_length": 1024},
}


@dataclass(frozen=True)
class Case:
    name: str
    stage: str
    dimension: str
    level: str
    snrs: tuple[int, ...] | None = None
    snr_levels: tuple[int, ...] | None = None
    obs_level: int = 1
    chan_levels: tuple[int, ...] = (2,)
    off_levels: tuple[int, ...] = (1,)
    granularity_level: int | None = None
    aux_level_cap: int | None = None
    aux_policy: str = "fixed_base"

    def expected_snrs(self) -> list[int]:
        if self.snrs is not None:
            return list(self.snrs)
        if self.snr_levels is not None:
            levels = set(self.snr_levels)
            return [snr for snr in SNR_VALUES if snr_db_to_level(snr) in levels]
        return list(SNR_VALUES)

    def gen_snr_args(self) -> list[str]:
        if self.snrs is not None:
            return [f"--snrs={','.join(str(x) for x in self.snrs)}"]
        if self.snr_levels is not None:
            return [f"--snr-levels={','.join(str(x) for x in self.snr_levels)}"]
        return [f"--snrs={','.join(str(x) for x in SNR_VALUES)}"]

    def row(self, profile_name: str, target_length: int) -> dict[str, str | int]:
        mods = GRANULARITY_LEVEL_MODS[self.granularity_level] if self.granularity_level is not None else None
        return {
            "profile": profile_name,
            "case": self.name,
            "stage": self.stage,
            "dimension": self.dimension,
            "level": self.level,
            "snrs": ",".join(str(x) for x in self.expected_snrs()),
            "obs_level": self.obs_level,
            "chan_levels": ",".join(str(x) for x in self.chan_levels),
            "off_levels": ",".join(str(x) for x in self.off_levels),
            "granularity_level": "" if self.granularity_level is None else self.granularity_level,
            "granularity_num_classes": "" if mods is None else len(mods),
            "granularity_mods": "" if mods is None else ",".join(mods),
            "aux_level_cap": "" if self.aux_level_cap is None else self.aux_level_cap,
            "aux_policy": self.aux_policy,
            "target_length_for_model_pack": target_length,
        }


def parse_csv_ints(text: str) -> list[int]:
    return [int(x) for x in str(text).replace(",", " ").split() if x.strip()]


def parse_csv_text(text: str) -> list[str]:
    return [x.strip() for x in str(text).replace(",", " ").split() if x.strip()]


def auxiliary_level_cap(level: int) -> int:
    level = int(level)
    return max(0, min(5, max(level // 2, level - 2)))


def fixed_aux_levels(case_dim: str, case_level: int) -> tuple[int, dict[str, int]]:
    cap = auxiliary_level_cap(case_level)
    return cap, {dim: cap for dim in DIM_ORDER if dim != case_dim}


def boundary_case(profile: dict, dim: str, level: int) -> Case:
    cap, aux = fixed_aux_levels(dim, level)
    return Case(
        name=f"bd_{dim}_l{level}",
        stage="boundary",
        dimension=dim,
        level=str(level),
        snrs=None,
        snr_levels=(level,) if dim == "snr" else (aux["snr"],),
        obs_level=level if dim == "obs" else aux["obs"],
        chan_levels=(level,) if dim == "chan" else (aux["chan"],),
        off_levels=(level,) if dim == "off" else (aux["off"],),
        granularity_level=level if dim == "gra" else aux["gra"],
        aux_level_cap=cap,
        aux_policy="fixed_cap",
    )


def build_boundary_cases(profile: dict, dims: list[str], levels: list[int]) -> list[Case]:
    cases = [
        Case(
            name="base_source_like",
            stage="boundary",
            dimension="base",
            level="source",
            snrs=tuple(profile["source_snrs"]),
            obs_level=profile["base_obs"],
            chan_levels=(profile["base_chan"],),
            off_levels=(profile["base_off"],),
        )
    ]
    for dim in DIM_ORDER:
        if dim in dims:
            cases.extend(boundary_case(profile, dim, level) for level in levels)
    return cases


def stress_case(profile: dict, selected: list[tuple[str, int]], source_snrs: tuple[int, ...]) -> Case:
    snr_levels = None
    obs_level = profile["base_obs"]
    chan_levels = (profile["base_chan"],)
    off_levels = (profile["base_off"],)
    granularity_level = None
    parts = []
    for dim, level in selected:
        parts.append(f"{dim}{level}")
        if dim == "snr":
            snr_levels = (level,)
        elif dim == "obs":
            obs_level = level
        elif dim == "chan":
            chan_levels = (level,)
        elif dim == "off":
            off_levels = (level,)
        elif dim == "gra":
            granularity_level = level
    return Case(
        name=f"stress_{'pair' if len(selected) == 2 else 'all'}_{'_'.join(parts)}",
        stage="stress",
        dimension="+".join(dim for dim, _ in selected),
        level="+".join(str(level) for _, level in selected),
        snrs=None if snr_levels is not None else source_snrs,
        snr_levels=snr_levels,
        obs_level=obs_level,
        chan_levels=chan_levels,
        off_levels=off_levels,
        granularity_level=granularity_level,
        aux_policy="stress_all_pairs_high",
    )


def build_stress_cases(profile: dict, dims: list[str], level: int) -> list[Case]:
    selected = [(dim, level) for dim in DIM_ORDER if dim in dims]
    source_snrs = tuple(profile["source_snrs"])
    pairs = [stress_case(profile, list(pair), source_snrs) for pair in itertools.combinations(selected, 2)]
    all_case = stress_case(profile, selected, source_snrs) if len(selected) > 2 else None
    return pairs + ([all_case] if all_case is not None else [])


def filter_cases(cases: list[Case], text: str | None) -> list[Case]:
    if not text:
        return cases
    tokens = parse_csv_text(text)
    return [case for case in cases if any(token in case.name for token in tokens)]


def generation_command(args: argparse.Namespace, case: Case) -> list[str]:
    script = Path("scripts") / "generate_dataset.py"
    output_dir = Path(args.output_root) / args.profile / case.name
    cmd = [
        args.python,
        str(script),
        *case.gen_snr_args(),
        "--output-dir",
        str(output_dir),
        "--reps",
        str(args.reps),
        "--obs-levels",
        str(case.obs_level),
        "--chan-levels",
        ",".join(str(x) for x in case.chan_levels),
        "--off-levels",
        ",".join(str(x) for x in case.off_levels),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--source-mode",
        args.source_mode,
    ]
    if case.granularity_level is not None:
        cmd += ["--granularity-level", str(case.granularity_level)]
    if args.text_source:
        cmd += ["--text-source", args.text_source]
    if args.fast_nondeterministic:
        cmd += ["--fast-nondeterministic"]
    return cmd


def write_plan(path: Path, cases: list[Case], profile_name: str, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [case.row(profile_name, profile["target_length"]) for case in cases]
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan or run DCS-SG boundary/stress case generation.")
    parser.add_argument("--profile", choices=sorted(MODEL_PROFILES), default="mcnet")
    parser.add_argument("--stage", choices=["plan", "boundary", "stress", "all"], default="plan")
    parser.add_argument("--dims", default="snr,obs,chan,off,gra")
    parser.add_argument("--levels", default="0,1,2,3,4,5")
    parser.add_argument("--stress-level", type=int, default=None)
    parser.add_argument("--case-filter", default=None)
    parser.add_argument("--plan-file", default="examples/boundary_plan.csv")
    parser.add_argument("--commands-file", default="examples/generate_commands.txt")
    parser.add_argument("--output-root", default="generated_data")
    parser.add_argument("--reps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--source-mode", choices=["natural", "random"], default="natural")
    parser.add_argument("--text-source", default=None)
    parser.add_argument("--fast-nondeterministic", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--run", action="store_true", help="Execute generation commands after writing the plan.")
    args = parser.parse_args()

    dims = parse_csv_text(args.dims)
    levels = parse_csv_ints(args.levels)
    invalid_dims = set(dims) - set(DIM_ORDER)
    invalid_levels = [level for level in levels if level < 0 or level > 5]
    if invalid_dims:
        raise ValueError(f"Unsupported dimensions: {sorted(invalid_dims)}")
    if invalid_levels:
        raise ValueError(f"Levels must be in 0..5: {invalid_levels}")

    profile = MODEL_PROFILES[args.profile]
    cases: list[Case] = []
    if args.stage in {"plan", "boundary", "all"}:
        cases.extend(build_boundary_cases(profile, dims, levels))
    if args.stage in {"stress", "all"}:
        stress_level = args.stress_level if args.stress_level is not None else max(levels)
        cases.extend(build_stress_cases(profile, dims, stress_level))
    cases = filter_cases(cases, args.case_filter)
    if not cases:
        raise RuntimeError("No cases selected.")

    plan_file = PROJECT_ROOT / args.plan_file
    commands_file = PROJECT_ROOT / args.commands_file
    write_plan(plan_file, cases, args.profile, profile)
    commands = [generation_command(args, case) for case in cases]
    commands_file.parent.mkdir(parents=True, exist_ok=True)
    commands_file.write_text("\n".join(" ".join(str(part) for part in cmd) for cmd in commands) + "\n", encoding="utf-8")
    print(f"Wrote {len(cases)} cases to {plan_file}")
    print(f"Wrote generation commands to {commands_file}")

    if args.run:
        for cmd in commands:
            print(" ".join(str(part) for part in cmd), flush=True)
            subprocess.run([str(part) for part in cmd], cwd=str(PROJECT_ROOT), check=True)


if __name__ == "__main__":
    main()
