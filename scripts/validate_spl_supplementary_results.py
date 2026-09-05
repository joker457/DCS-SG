#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODELS = ["Tr-AMR", "MCNet", "IQFormer", "E-A"]
BOUNDARY_COLUMNS = [
    "model",
    "dimension",
    "level_num",
    "original_acc",
    "enhanced_acc",
    "delta_acc",
]
STRESS_COLUMNS = [
    "model",
    "case",
    "dimension",
    "level",
    "original_acc",
    "enhanced_acc",
    "delta_acc",
]
DEPLOYMENT_COLUMNS = [
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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def verify_checksums() -> None:
    manifest = DATA_DIR / "spl_supplementary_checksums.sha256"
    for line in manifest.read_text(encoding="ascii").splitlines():
        if not line.strip():
            continue
        expected, relative_path = line.split(maxsplit=1)
        path = ROOT / relative_path
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        require(actual == expected, f"SHA-256 mismatch: {relative_path}")
    print("Paper-artifact SHA-256 checks: PASS")


def verify_boundary_export() -> pd.DataFrame:
    source = pd.read_csv(DATA_DIR / "boundary_dimension_level_detail.csv")
    published = pd.read_csv(DATA_DIR / "single_scale_boundary_curves_data.csv")
    expected = source[source["level_num"].astype(int).between(1, 5)][BOUNDARY_COLUMNS]
    expected = expected.reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, published, check_exact=True)
    require(len(published) == 100, "Expected 4 models x 5 dimensions x 5 levels")
    require(
        not published.duplicated(["model", "dimension", "level_num"]).any(),
        "Duplicate single-scale result key",
    )
    require(
        set(published["model"]) == set(MODELS),
        "Unexpected model set in single-scale results",
    )
    require(
        np.allclose(
            published["enhanced_acc"] - published["original_acc"], published["delta_acc"]
        ),
        "Single-scale deltas do not equal optimized minus original accuracy",
    )
    for column in ("original_acc", "enhanced_acc"):
        require(published[column].between(0.0, 1.0).all(), f"Out-of-range values in {column}")
    return source


def verify_stress_export() -> pd.DataFrame:
    source = pd.read_csv(DATA_DIR / "stress_case_detail.csv")
    published = pd.read_csv(DATA_DIR / "bivariate_stress_accuracy_data.csv")
    expected = source[STRESS_COLUMNS].reset_index(drop=True)

    pd.testing.assert_frame_equal(expected, published, check_exact=True)
    require(len(published) == 44, "Expected 4 models x (10 bivariate probes + All-5)")
    require(not published.duplicated(["model", "case"]).any(), "Duplicate stress result key")
    require(
        np.allclose(
            published["enhanced_acc"] - published["original_acc"], published["delta_acc"]
        ),
        "Stress deltas do not equal optimized minus original accuracy",
    )
    for column in ("original_acc", "enhanced_acc"):
        require(published[column].between(0.0, 1.0).all(), f"Out-of-range values in {column}")
    return source


def verify_deployment_export() -> None:
    payload = json.loads(
        (DATA_DIR / "inference_latency_benchmark.json").read_text(encoding="utf-8")
    )
    source = pd.DataFrame(payload["rows"])
    expected = source[(source["method"] == "Optimized") & source["backbone"].isin(MODELS)].copy()
    expected["model_order"] = expected["backbone"].map(
        {model: index for index, model in enumerate(MODELS)}
    )
    expected = expected.sort_values("model_order").reset_index(drop=True)
    expected["stft_prep_B1_ms_sample"] = (
        expected["stft_prep_B1_ms_sample"].fillna(0.0).astype(float)
    )
    expected["fp32_weight_mib"] = expected["params"].astype(float) * 4.0 / (1024.0**2)
    expected = expected[DEPLOYMENT_COLUMNS]
    published = pd.read_csv(DATA_DIR / "deployment_resource_profile_data.csv")

    pd.testing.assert_frame_equal(expected, published, check_exact=False, rtol=0.0, atol=1e-14)
    require(len(published) == 4, "Expected one optimized deployment row per model")
    require(
        np.allclose(
            published["cpu_model_B1_ms_sample"] + published["stft_prep_B1_ms_sample"],
            published["cpu_e2e_B1_ms_sample"],
        ),
        "CPU end-to-end latency does not include STFT preprocessing",
    )


def verify_aggregate_summary(boundary: pd.DataFrame, stress: pd.DataFrame) -> None:
    summary = pd.read_csv(DATA_DIR / "four_experiment_delta_summary.csv").set_index("model")
    require(set(summary.index) == set(MODELS), "Unexpected model set in aggregate summary")
    accuracy_columns = [
        column
        for column in summary.columns
        if column.startswith(("original_", "enhanced_")) and column.endswith("_acc")
    ]
    for column in accuracy_columns:
        require(summary[column].between(0.0, 1.0).all(), f"Out-of-range values in {column}")
    delta_columns = (
        ("original_base_acc", "enhanced_base_acc", "delta_base_acc"),
        ("original_boundary_mean_acc", "enhanced_boundary_mean_acc", "delta_boundary_mean_acc"),
        (
            "original_boundary_l5_mean_acc",
            "enhanced_boundary_l5_mean_acc",
            "delta_boundary_l5_mean_acc",
        ),
        (
            "original_stress_pair_mean_acc",
            "enhanced_stress_pair_mean_acc",
            "delta_stress_pair_mean_acc",
        ),
        ("original_stress_all_acc", "enhanced_stress_all_acc", "delta_stress_all_acc"),
        ("source_original_acc", "source_enhanced_acc", "delta_source_acc"),
    )
    for original_column, enhanced_column, delta_column in delta_columns:
        require(
            np.allclose(
                summary[enhanced_column] - summary[original_column],
                summary[delta_column],
                rtol=0.0,
                atol=1e-12,
            ),
            f"Aggregate delta mismatch for {delta_column}",
        )
    checks = [
        ("original_boundary_mean_acc", boundary, "original_acc"),
        ("enhanced_boundary_mean_acc", boundary, "enhanced_acc"),
        (
            "original_boundary_l5_mean_acc",
            boundary[boundary["level_num"].astype(int) == 5],
            "original_acc",
        ),
        (
            "enhanced_boundary_l5_mean_acc",
            boundary[boundary["level_num"].astype(int) == 5],
            "enhanced_acc",
        ),
        ("original_stress_pair_mean_acc", stress[stress["level"] == "5+5"], "original_acc"),
        ("enhanced_stress_pair_mean_acc", stress[stress["level"] == "5+5"], "enhanced_acc"),
        (
            "original_stress_all_acc",
            stress[stress["level"] == "5+5+5+5+5"],
            "original_acc",
        ),
        (
            "enhanced_stress_all_acc",
            stress[stress["level"] == "5+5+5+5+5"],
            "enhanced_acc",
        ),
    ]
    for output_column, rows, accuracy_column in checks:
        calculated = rows.groupby("model")[accuracy_column].mean().sort_index()
        reported = summary[output_column].sort_index()
        require(
            np.allclose(calculated, reported, rtol=0.0, atol=1e-12),
            f"Aggregate mismatch in {output_column}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the published SPL supplementary results")
    parser.add_argument(
        "--check-paper-hashes",
        action="store_true",
        help="also verify that tracked CSV, JSON, and PNG files match the audited paper artifacts",
    )
    args = parser.parse_args()

    boundary = verify_boundary_export()
    stress = verify_stress_export()
    verify_deployment_export()
    verify_aggregate_summary(boundary, stress)
    if args.check_paper_hashes:
        verify_checksums()
    print("Numeric and provenance checks: PASS (100 boundary points, 44 stress cells, 4 profiles)")


if __name__ == "__main__":
    main()
