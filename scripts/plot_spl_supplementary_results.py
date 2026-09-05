#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "docs" / "images"
EXPORT_DIR = DATA_DIR

MODELS = ["Tr-AMR", "MCNet", "IQFormer", "E-A"]
DIMENSIONS = ["snr", "obs", "chan", "off", "gra"]
DIMENSION_LABELS = {
    "snr": "SNR",
    "obs": "Observation",
    "chan": "Channel fading",
    "off": "Sync. offset",
    "gra": "Class granularity",
}
MODEL_COLORS = {
    "Tr-AMR": "#3B6FB6",
    "MCNet": "#C75146",
    "IQFormer": "#7A5AA6",
    "E-A": "#2E8B57",
}
STRESS_ORDER = [
    "snr+obs",
    "snr+chan",
    "snr+off",
    "snr+gra",
    "obs+chan",
    "obs+off",
    "obs+gra",
    "chan+off",
    "chan+gra",
    "off+gra",
    "snr+obs+chan+off+gra",
]
STRESS_LABELS = {
    "snr+obs": "SNR+Obs.",
    "snr+chan": "SNR+Fad.",
    "snr+off": "SNR+Sync.",
    "snr+gra": "SNR+Gran.",
    "obs+chan": "Obs.+Fad.",
    "obs+off": "Obs.+Sync.",
    "obs+gra": "Obs.+Gran.",
    "chan+off": "Fad.+Sync.",
    "chan+gra": "Fad.+Gran.",
    "off+gra": "Sync.+Gran.",
    "snr+obs+chan+off+gra": "All-5",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.dpi": 160,
            "savefig.dpi": 400,
        }
    )


def save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / f"{stem}.png"
    pdf_path = OUT_DIR / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def plot_single_scale_boundaries(boundary: pd.DataFrame) -> pd.DataFrame:
    work = boundary[boundary["level_num"].astype(int).between(1, 5)].copy()
    work["level_num"] = work["level_num"].astype(int)

    fig, axes_grid = plt.subplots(3, 2, figsize=(3.5, 5.15), sharey=True)
    axes = axes_grid.ravel()
    for ax, dimension in zip(axes, DIMENSIONS):
        for model in MODELS:
            rows = work[(work["dimension"] == dimension) & (work["model"] == model)].sort_values(
                "level_num"
            )
            levels = rows["level_num"].to_numpy()
            ax.plot(
                levels,
                rows["original_acc"].to_numpy(dtype=float) * 100,
                color=MODEL_COLORS[model],
                linestyle="--",
                linewidth=0.95,
                marker="o",
                markersize=2.2,
                alpha=0.72,
            )
            ax.plot(
                levels,
                rows["enhanced_acc"].to_numpy(dtype=float) * 100,
                color=MODEL_COLORS[model],
                linestyle="-",
                linewidth=1.25,
                marker="o",
                markersize=2.4,
            )
        ax.set_title(DIMENSION_LABELS[dimension], pad=3, fontsize=8)
        ax.set_xticks(range(1, 6))
        ax.set_xlabel("Demand level", fontsize=7)
        ax.tick_params(labelsize=6.5)
        ax.grid(color="#D6D6D6", linestyle="--", linewidth=0.55, alpha=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].axis("off")
    for index in (0, 2, 4):
        axes[index].set_ylabel("Accuracy (%)", fontsize=7)
    axes[0].set_ylim(0, 102)
    model_handles = [
        Line2D([0], [0], color=MODEL_COLORS[model], lw=1.4, label=model) for model in MODELS
    ]
    style_handles = [
        Line2D([0], [0], color="#333333", lw=1.5, linestyle="--", label="Original"),
        Line2D([0], [0], color="#333333", lw=1.9, linestyle="-", label="+Ours"),
    ]
    fig.legend(
        handles=model_handles + style_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=3,
        frameon=False,
        fontsize=6.5,
        columnspacing=1.0,
        handlelength=1.8,
    )
    fig.subplots_adjust(left=0.13, right=0.99, bottom=0.06, top=0.88, wspace=0.22, hspace=0.48)
    save_figure(fig, "single_scale_boundary_curves")

    export = work[
        ["model", "dimension", "level_num", "original_acc", "enhanced_acc", "delta_acc"]
    ].copy()
    export.to_csv(
        EXPORT_DIR / "single_scale_boundary_curves_data.csv", index=False, encoding="utf-8-sig"
    )
    return export


def stress_matrix(stress: pd.DataFrame, accuracy_column: str) -> np.ndarray:
    values = np.full((len(MODELS), len(STRESS_ORDER)), np.nan, dtype=float)
    for row_idx, model in enumerate(MODELS):
        for col_idx, dimension in enumerate(STRESS_ORDER):
            cell = stress[(stress["model"] == model) & (stress["dimension"] == dimension)]
            if not cell.empty:
                values[row_idx, col_idx] = float(cell.iloc[0][accuracy_column]) * 100
    return values


def annotate_heatmap(ax: plt.Axes, values: np.ndarray, image) -> None:
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if np.isnan(value):
                continue
            red, green, blue, _ = image.cmap(image.norm(value))
            luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
            color = "white" if luminance < 0.52 else "#202020"
            ax.text(col, row, f"{value:.1f}", ha="center", va="center", fontsize=7.3, color=color)


def plot_stress_accuracy(stress: pd.DataFrame) -> pd.DataFrame:
    original = stress_matrix(stress, "original_acc")
    enhanced = stress_matrix(stress, "enhanced_acc")

    fig, axes = plt.subplots(1, 2, figsize=(3.5, 3.85), sharey=True)
    for ax, values, title in zip(axes, [original, enhanced], ["(a) Original", "(b) +Ours"]):
        values = values.T
        image = ax.imshow(values, aspect="auto", cmap="YlGnBu", vmin=0, vmax=90)
        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=48, ha="right", fontsize=6.2)
        ax.set_yticks(range(len(STRESS_ORDER)))
        ax.set_yticklabels([STRESS_LABELS[item] for item in STRESS_ORDER], fontsize=6.1)
        ax.set_title(title, loc="left", pad=3, fontsize=8)
        ax.set_xticks(np.arange(-0.5, len(MODELS), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(STRESS_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                value = values[row, col]
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                color = "white" if luminance < 0.52 else "#202020"
                ax.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=4.7,
                    color=color,
                )
        ax.axhline(9.5, color="white", linewidth=1.6)

    axes[0].set_ylabel("Level-5 stress configuration", fontsize=7)
    colorbar_axis = fig.add_axes([0.26, 0.035, 0.55, 0.018])
    colorbar = fig.colorbar(image, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label("Accuracy (%)", fontsize=6.5)
    colorbar.ax.tick_params(labelsize=6)
    fig.subplots_adjust(left=0.25, right=0.995, bottom=0.17, top=0.96, wspace=0.10)
    save_figure(fig, "bivariate_stress_accuracy")

    export = stress[
        ["model", "case", "dimension", "level", "original_acc", "enhanced_acc", "delta_acc"]
    ].copy()
    export.to_csv(
        EXPORT_DIR / "bivariate_stress_accuracy_data.csv", index=False, encoding="utf-8-sig"
    )
    return export


def bubble_size(weight_mib: float, maximum: float) -> float:
    scaled = math.log10(weight_mib + 1.0) / math.log10(maximum + 1.0)
    return 120.0 + 500.0 * scaled


def plot_deployment_profile(json_path: Path) -> pd.DataFrame:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    rows = pd.DataFrame(payload["rows"])
    rows = rows[(rows["method"] == "Optimized") & rows["backbone"].isin(MODELS)].copy()
    rows["model_order"] = rows["backbone"].map({model: index for index, model in enumerate(MODELS)})
    rows = rows.sort_values("model_order").reset_index(drop=True)
    rows["stft_prep_B1_ms_sample"] = rows["stft_prep_B1_ms_sample"].fillna(0.0).astype(float)
    rows["fp32_weight_mib"] = rows["params"].astype(float) * 4.0 / (1024.0 * 1024.0)

    fig, ax_profile = plt.subplots(figsize=(3.5, 2.05))
    end_to_end = rows["cpu_e2e_B1_ms_sample"].to_numpy(dtype=float)
    max_latency = float(end_to_end.max())

    max_weight = float(rows["fp32_weight_mib"].max())
    for _, row in rows.iterrows():
        model = str(row["backbone"])
        x = float(row["cpu_e2e_B1_ms_sample"])
        y_value = float(row["input_ratio"])
        size = bubble_size(float(row["fp32_weight_mib"]), max_weight)
        ax_profile.scatter(
            x,
            y_value,
            s=size,
            color=MODEL_COLORS[model],
            alpha=0.82,
            edgecolor="white",
            linewidth=0.9,
        )
        if model == "IQFormer":
            dx, dy, ha = 0.05, -1.25, "left"
        elif model == "Tr-AMR":
            dx, dy, ha = 0.05, 0.70, "left"
        elif model == "MCNet":
            dx, dy, ha = 0.05, -0.55, "left"
        else:
            dx, dy, ha = -0.12, 0.55, "right"
        ax_profile.text(x + dx, y_value + dy, model, fontsize=7, ha=ha, va="center")

    ax_profile.set_xlabel("CPU end-to-end latency (ms/sample, B=1)", fontsize=7)
    ax_profile.set_ylabel("Input burden relative to raw I/Q", fontsize=7)
    ax_profile.set_xlim(0.65, max_latency * 1.22)
    ax_profile.set_ylim(0, 18.4)
    ax_profile.set_yticks([1, 5, 9, 13, 17])
    ax_profile.grid(color="#D6D6D6", linewidth=0.6)
    ax_profile.set_axisbelow(True)
    ax_profile.spines["top"].set_visible(False)
    ax_profile.spines["right"].set_visible(False)
    ax_profile.text(
        0.02,
        0.97,
        "Bubble area: complete FP32 model size",
        transform=ax_profile.transAxes,
        fontsize=6.5,
        color="#555555",
        va="top",
    )

    ax_profile.tick_params(labelsize=6.5)
    fig.subplots_adjust(left=0.17, right=0.98, bottom=0.19, top=0.98)
    save_figure(fig, "deployment_resource_profile")

    export_columns = [
        "backbone",
        "method",
        "params",
        "fp32_weight_mib",
        "input_ratio",
        "cpu_model_B1_ms_sample",
        "stft_prep_B1_ms_sample",
        "cpu_e2e_B1_ms_sample",
        "gpu_model_B1_ms_sample",
        "gpu_model_B64_ms_sample",
    ]
    export = rows[export_columns].copy()
    export.to_csv(
        EXPORT_DIR / "deployment_resource_profile_data.csv", index=False, encoding="utf-8-sig"
    )
    return export


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    boundary = pd.read_csv(DATA_DIR / "boundary_dimension_level_detail.csv")
    stress = pd.read_csv(DATA_DIR / "stress_case_detail.csv")

    plot_single_scale_boundaries(boundary)
    plot_stress_accuracy(stress)
    deployment = plot_deployment_profile(DATA_DIR / "inference_latency_benchmark.json")

    print(f"Saved supplementary figures and data to: {OUT_DIR}")
    print("Deployment profile:")
    print(
        deployment[["backbone", "params", "fp32_weight_mib", "input_ratio", "cpu_e2e_B1_ms_sample"]]
        .round(4)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
