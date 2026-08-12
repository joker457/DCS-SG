# DCS Scale Parameterization

The DCS levels are monotone ordinal stress bins for controlled baseband simulation. They are not universal categories calibrated to a particular standard, carrier band, mobility profile, or receiver device. Absolute physical values depend on the selected symbol rate and carrier frequency.

## Discrete Levels Used in the Study

| Scale | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 |
| --- | --- | --- | --- | --- | --- |
| SNR (dB) | 6, 8, 10 | 0, 2, 4 | -6, -4, -2 | -12, -10, -8 | -20, -18, -16, -14 |
| Observation length (I/Q samples) | 1024 | 512 | 256 | 128 | 64 |
| Class granularity | 8 | 12 | 16 | 20 | 24 |

The generator uses eight samples per symbol (`SPS=8`). Level 0 is an easier calibration setting and is omitted from the five-level demand profile in the paper.

## Channel-Fading Levels

| Level | Paths | Maximum delay (symbols) | Normalized Doppler (cycles/symbol) | Rician K factor (dB) |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1-2 | 0-0.25 | 0-0.003 | 12-20 |
| 2 | 2-3 | 0.15-0.60 | 0.003-0.010 | 5-12 |
| 3 | 4-6 | 0.80-2.00 | 0.018-0.050 | -2-5 |
| 4 | 6-10 | 1.50-4.00 | 0.045-0.110 | -8-2 |
| 5 | 8-12 | 2.50-6.00 | 0.080-0.180 | Rayleigh; no LOS term |

For symbol rate `Rs` symbols/s, a delay of `u` symbols corresponds to `u/Rs` seconds, and normalized Doppler `v` corresponds to `v*Rs` Hz. If a carrier frequency `fc` is supplied, an illustrative one-way radial speed is `c*v*Rs/fc`; the generator itself does not assume `fc`.

## Synchronization-Mismatch Levels

A sampled severity budget is distributed across carrier-frequency offset (CFO), carrier-phase offset, timing offset, and sampling-clock offset (SCO) with weights 0.35, 0.20, 0.30, and 0.15. The severity intervals are:

| Level | Severity interval | Minimum effective burst overlap |
| --- | ---: | ---: |
| 1 | 0-0.10 | 75% |
| 2 | 0.10-0.25 | 68% |
| 3 | 0.35-0.70 | 58% |
| 4 | 0.70-1.10 | 48% |
| 5 | 1.10-1.55 | 38% |

After weighted allocation and clipping, component multipliers are applied to maximum scales of 0.09 cycles/symbol for CFO, pi radians for phase, 1.10 symbols for timing, and 800 ppm for SCO. For symbol rate `Rs`, normalized CFO `q` becomes `q*Rs` Hz, timing `u` becomes `u/Rs` seconds, and SCO `p` ppm gives a sampling-rate ratio of `1+p*1e-6`.

## Interpretation Boundary

The bins support controlled comparison and monotone stress traversal in the reported simulator. They should be mapped to deployment-specific `Rs`, `fc`, channel statistics, and receiver tolerances before being interpreted as operational requirements. The source of truth is `src/dcs_sg/config.py` and the `_channel_cfg`, `_sync_ranges`, and `_sync_scales` methods in `src/dcs_sg/generator.py`.
