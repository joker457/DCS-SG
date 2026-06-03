# DCS-SG: Demand-Capability-Scale Signal Generator

DCS-SG is a scale-controlled AMC signal generator for boundary and stress testing of automatic modulation classification models. It turns demand-scale levels into controllable IQ-signal test cases, so models can be evaluated by environment capability rather than only by average accuracy on a fixed dataset.

This repository keeps only the core generation code and case-planning logic. Generated HDF5 data, checkpoints, training logs, paper drafts, and model-specific training code are intentionally excluded.

## Core Controls

DCS-SG currently implements five signal-environment axes for demand-capability-scale analysis:

| Axis | Meaning | Level effect |
| --- | --- | --- |
| `snr` | low-SNR robustness | higher level selects lower SNR ranges |
| `obs` | observation budget | higher level shortens the effective observation length |
| `chan` | fading-channel robustness | higher level increases multipath/Doppler/fading severity |
| `off` | synchronization-mismatch robustness | higher level increases frequency/phase/timing offsets |
| `gra` | fine-grained class discrimination | higher level increases the modulation class set |

The HDF5 output also stores inactive auxiliary demand fields for compatibility with the full demand-capability-scale framework.

## Layout

```text
DCS-SG/
  src/dcs_sg/
    config.py              # modulation classes, scale levels, class-granularity sets
    generator.py           # batched PyTorch IQ-signal synthesis and impairments
    storage.py             # HDF5 writer bucketed by observation length
  scripts/
    generate_dataset.py    # raw DCS-SG HDF5 generation entry
    plan_boundary_cases.py # boundary/stress case planner for profile-based runs
  examples/
    *.csv, *.txt           # lightweight generated plans/commands
```

## Installation

```powershell
cd <path-to-your-DCS-SG-repository>
pip install -e .
```

Use CUDA for full-scale data generation. CPU mode is useful for smoke tests.

## Smoke Test

```powershell
cd <path-to-your-DCS-SG-repository>
python scripts\generate_dataset.py --snr 0 --output-dir generated_data\smoke --reps 4 --obs-levels 1 --chan-levels 2 --off-levels 1 --granularity-level 1 --batch-size 128 --device cpu --seed 20260426
```

The output is an HDF5 file such as `generated_data/smoke/amc_snr_+00dB.h5` with groups:

```text
data/obs_0/X, data/obs_0/Y, data/obs_0/SNR, data/obs_0/demand
data/obs_1/X, data/obs_1/Y, data/obs_1/SNR, data/obs_1/demand
...
```

`X` has shape `(N, 2, L)`, where the second dimension stores I and Q channels.

## Boundary and Stress Plans

Generate the case table and commands for a model profile:

```powershell
python scripts\plan_boundary_cases.py --profile mcnet --stage all --reps 512
```

Run the raw HDF5 generation for the selected cases:

```powershell
python scripts\plan_boundary_cases.py --profile mcnet --stage boundary --reps 512 --run --output-root generated_data
```

Built-in example profiles:

| Profile | Source-like SNR range | Base obs | Model pack length |
| --- | --- | --- | --- |
| `tramr` | 0 to 30 dB | 1 | 1024 |
| `mcnet` | -20 to 30 dB | 1 | 1024 |
| `iqformer` | -20 to 18 dB | 4 | 128 |
| `ea` | -20 to 30 dB | 1 | 1024 |

The profile only controls case planning and records a reference model packing length. DCS-SG itself outputs bucketed raw observations by `obs_level`; each model project can then pack, crop, pad, train, and evaluate the raw HDF5 files using its own input format.

## Reproducibility Notes

- `--source-mode natural` is the default and follows the RadioML-style idea of text-like digital bit streams and audio-like analog messages.
- `--seed` controls deterministic sample identity. Changing `--batch-size` does not change generated samples in deterministic mode.
- For negative SNR lists, prefer the equals form, for example `--snrs=-20,-18,-16`.
- Full experiment generation with `reps=512` can create large HDF5 files. Keep `generated_data/` outside commits or rely on `.gitignore`.

## Using DCS-SG with AMC Experiments

A typical model-specific experiment can use the case rules collected here as follows:

1. build single-axis boundary cases for `snr`, `obs`, `chan`, `off`, and `gra`;
2. optionally build bivariate high-level stress cases and an all-axis stress case;
3. call DCS-SG to generate raw HDF5 files;
4. pack the raw files into the model's input length and label convention;
5. train/evaluate the target model.

This repository preserves steps 1-3. Steps 4-5 remain model-specific and should live in the corresponding model or experiment repositories.
