# DCS Signal Generator

**Demand-Capability Scaled Signal Generation for Automatic Modulation Classification**

The DCS signal generator (repository identifier: **DCS-SG**) is the signal-generation component of the Demand-Capability Scaled (DCS) Agent framework for automatic modulation classification (AMC). It converts demand-scale levels into controllable I/Q samples for capability probing under heterogeneous electromagnetic environments. The resulting probes expose scale-specific model strengths and failure modes that can be obscured by aggregate accuracy on a fixed benchmark.

Within the complete DCS Agent loop, the generator constructs scale-conditioned test sets; the agent evaluates candidate AMC algorithms, compares measured capability with the target demand profile, and uses unmet probes together with an AMC knowledge base to guide model revision and retesting.

This repository contains only the core generation and probe-planning logic. Generated HDF5 data, checkpoints, training logs, paper drafts, the agent implementation, and model-specific training code are intentionally excluded.

## Core Controls

The DCS signal generator currently implements five signal-environment scales:

| Axis | Meaning | Level effect |
| --- | --- | --- |
| `snr` | low-SNR robustness | higher level selects lower SNR ranges |
| `obs` | observation budget | higher level shortens the effective observation length |
| `chan` | fading-channel robustness | higher level increases multipath/Doppler/fading severity |
| `off` | synchronization-mismatch robustness | higher level increases frequency/phase/timing offsets |
| `gra` | fine-grained class discrimination | higher level increases the modulation class set |

The HDF5 output also stores inactive auxiliary demand fields for compatibility with the complete DCS framework.

## Layout

```text
DCS-SG/
  src/dcs_sg/
    config.py              # modulation classes, scale levels, class-granularity sets
    generator.py           # batched PyTorch IQ-signal synthesis and impairments
    storage.py             # HDF5 writer bucketed by observation length
  scripts/
    generate_dataset.py    # raw DCS signal generation entry
    plan_boundary_cases.py # boundary/stress case planner for profile-based runs
  examples/
    *.csv, *.txt           # lightweight generated plans/commands
  model_sources/
    tr_amr/                # agent-reproduced Tr-AMR and its DCS Agent revision
    mcnet/                 # agent-reproduced MCNet and MCNetEnhanced
    iqformer/              # IQFormer backbone and DCS Agent adapter variants
    expert_assistant/      # PyTorch E-A backbone and its DCS Agent revision
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

The profile only controls case planning and records a reference model packing length. The DCS signal generator outputs bucketed raw observations by `obs_level`; each model project can then pack, crop, pad, train, and evaluate the raw HDF5 files using its own input format.

## Reproducibility Notes

- `--source-mode natural` is the default and follows the RadioML-style idea of text-like digital bit streams and audio-like analog messages.
- `--seed` controls deterministic sample identity. Changing `--batch-size` does not change generated samples in deterministic mode.
- For negative SNR lists, prefer the equals form, for example `--snrs=-20,-18,-16`.
- Full experiment generation with `reps=512` can create large HDF5 files. Keep `generated_data/` outside commits or rely on `.gitignore`.

## Using the DCS Signal Generator with the DCS Agent

A typical model-specific experiment can use the case rules collected here as follows:

1. build single-axis boundary cases for `snr`, `obs`, `chan`, `off`, and `gra`;
2. optionally build bivariate high-level stress cases and an all-axis stress case;
3. call the DCS signal generator to generate raw HDF5 files;
4. pack the raw files into the model's input length and label convention;
5. train and evaluate the target model;
6. return scale-wise performance gaps to the DCS Agent for knowledge-constrained model revision and retesting.

The generation pipeline preserves steps 1-3. The architecture snapshots under `model_sources/` document the models used in steps 4-6, but their dataset packing, checkpoint loading, training, and evaluation entry points remain model-specific.

## Model Source Snapshots

`model_sources/` contains the model definitions used in the four-model study. Here, **reproduced backbone** denotes the implementation reconstructed by the agent from the corresponding paper and source repository; **DCS Agent revision** denotes the model obtained after feeding DCS probe failures back to knowledge-constrained code optimization. These files are source snapshots for inspection and reproduction and do not alter or depend on the DCS signal-generation path.

| Model | Reproduced backbone | DCS Agent revision used in the study | Source files |
| --- | --- | --- | --- |
| Tr-AMR | `VisionTransformer` / `vit_` | `EnhancedTrAMRV3`: zero-start local token, post-encoder, and attentive-pooling adapters | `model_sources/tr_amr/vit_model_2018.py`, `enhanced_tr_amr.py` |
| MCNet | `MCNet` | `MCNetEnhanced`: observation-aware pooling with residual channel and temporal gates | `model_sources/mcnet/mcnet_model.py` |
| IQFormer | `IQFormer` | `IQFormerChannelAdapter`: zero-start I/Q FIR, STFT, token, and classifier-head adapters | `model_sources/iqformer/model/IQFormer.py`, `IQFormer_channel_adapter.py` |
| E-A | `ExpertAssistant` | `ExpertAssistantEnhanced`: phase/frequency compensation with spectral and feature residual branches | `model_sources/expert_assistant/model/expert_assistant_torch.py`, `expert_assistant_enhanced_torch.py` |

The IQFormer snapshot additionally retains `IQFormer_enhanced.py`, `IQFormer_adapter.py`, and `utils/stft_features.py`, which were used to compare front-end/fusion and residual-adapter alternatives during development. The channel-adapter implementation listed in the table is the primary improved variant used for the reported DCS boundary and stress results.

The Tr-AMR, MCNet, and E-A snapshots require PyTorch. IQFormer additionally imports `einops` and `timm`. Install these optional dependencies with `pip install -r model_sources/requirements.txt`. They are deliberately kept separate from the core DCS-SG requirements because raw signal generation does not use the classifier implementations. Checkpoints, generated datasets, and model-specific training wrappers are not included.

## AMC Knowledge Base Used by the DCS Agent

The DCS Agent knowledge base contains **25 papers** covering convolutional, recurrent, Transformer, multi-representation, feature-based, data-augmentation, and denoising approaches to AMC. During feedback optimization, their structured records provide evidence for mapping an unmet demand scale to candidate source-code revisions. The first 21 references below are the most recent non-duplicate entries retained from `AMCagent/knowledge/json`; the final four are the representative algorithms evaluated in the paper. Because the legacy JSON records do not store persistent identifiers, their links use exact-title scholarly searches, whereas the four experimental papers use DOI links.

1. [Contrastive Learning for Robust Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22Contrastive+Learning+for+Robust+Automatic+Modulation+Classification%22) (2023)
2. [Multi-Scale Feature Fusion for Robust Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22Multi-Scale+Feature+Fusion+for+Robust+Automatic+Modulation+Classification%22) (2023)
3. [CGDNet: Contourlet-Based Multi-Scale Decomposition Network for AMC](https://scholar.google.com/scholar?q=%22CGDNet%3A+Contourlet-Based+Multi-Scale+Decomposition+Network+for+AMC%22) (2022)
4. [IQ and Constellation Diagram Fusion for Robust Modulation Classification](https://scholar.google.com/scholar?q=%22IQ+and+Constellation+Diagram+Fusion+for+Robust+Modulation+Classification%22) (2022)
5. [Automatic Modulation Classification Using Transformer](https://scholar.google.com/scholar?q=%22Automatic+Modulation+Classification+Using+Transformer%22) (2021)
6. [CNN-GRU Hybrid Network for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22CNN-GRU+Hybrid+Network+for+Automatic+Modulation+Classification%22) (2021)
7. [Hierarchical Deep Learning for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22Hierarchical+Deep+Learning+for+Automatic+Modulation+Classification%22) (2021)
8. [Multi-Task Learning for Joint Modulation Classification and SNR Estimation](https://scholar.google.com/scholar?q=%22Multi-Task+Learning+for+Joint+Modulation+Classification+and+SNR+Estimation%22) (2021)
9. [GAN-based Data Augmentation for Modulation Classification with ACGAN](https://scholar.google.com/scholar?q=%22GAN-based+Data+Augmentation+for+Modulation+Classification+with+ACGAN%22) (2021)
10. [Denoising Autoencoder for Robust Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22Denoising+Autoencoder+for+Robust+Automatic+Modulation+Classification%22) (2021)
11. [IC-AMCNet: Inception based CNN with Channel Attention for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22IC-AMCNet%3A+Inception+based+CNN+with+Channel+Attention+for+Automatic+Modulation+Classification%22) (2020)
12. [AMRNet: Attention-based Multi-scale Residual Network for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22AMRNet%3A+Attention-based+Multi-scale+Residual+Network+for+Automatic+Modulation+Classification%22) (2020)
13. [DenseNet for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22DenseNet+for+Automatic+Modulation+Classification%22) (2020)
14. [Bidirectional LSTM for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22Bidirectional+LSTM+for+Automatic+Modulation+Classification%22) (2020)
15. [CNN-SVM Hybrid Approach for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22CNN-SVM+Hybrid+Approach+for+Automatic+Modulation+Classification%22) (2019)
16. [VGG-style Deep Learning for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22VGG-style+Deep+Learning+for+Automatic+Modulation+Classification%22) (2019)
17. [Multi-Channel Long-term Deep Neural Network for Automatic Modulation Classification](https://scholar.google.com/scholar?q=%22Multi-Channel+Long-term+Deep+Neural+Network+for+Automatic+Modulation+Classification%22) (2019)
18. [Deep Learning-Based Modulation Recognition for Software-Defined Radio](https://scholar.google.com/scholar?q=%22Deep+Learning-Based+Modulation+Recognition+for+Software-Defined+Radio%22) (2018)
19. [Over-the-Air Deep Learning Based Radio Signal Classification](https://doi.org/10.1109/JSTSP.2018.2797022) (2018)
20. [Wavelet Transform Based Modulation Classification](https://scholar.google.com/scholar?q=%22Wavelet+Transform+Based+Modulation+Classification%22) (2005)
21. [Automatic Modulation Classification Using Higher-Order Cumulants](https://scholar.google.com/scholar?q=%22Automatic+Modulation+Classification+Using+Higher-Order+Cumulants%22) (2000)
22. [MCNet: An Efficient CNN Architecture for Robust Automatic Modulation Classification](https://doi.org/10.1109/LCOMM.2020.2968030) (2020)
23. [IQFormer: A Novel Transformer-Based Model With Multi-Modality Fusion for Automatic Modulation Recognition](https://doi.org/10.1109/TCCN.2024.3485118) (2025)
24. [An Expert-Assistant Network With Temporal Shuffling for Efficient Automatic Modulation Recognition](https://doi.org/10.1109/TWC.2025.3645099) (2026)
25. [Tr-AMR: A Lightweight Transformer With Enhanced Temporal Modeling for Automatic Modulation Recognition](https://doi.org/10.1002/dac.70447) (2026)
