"""Replay the representative MCNet DCS feedback decision from public records."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET_ACCURACY = 0.50
BACKBONE = "MCNet"


def load_rows() -> list[dict[str, str]]:
    with (ROOT / "probe_results.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalized_gaps(rows: list[dict[str, str]], field: str) -> dict[str, float]:
    weighted_gap: dict[str, float] = defaultdict(float)
    weighted_total: dict[str, float] = defaultdict(float)
    for row in rows:
        if row["record_type"] != "single_scale":
            continue
        level = int(row["level"])
        gap = max(0.0, TARGET_ACCURACY - float(row[field]))
        weighted_gap[row["scale"]] += gap * level
        weighted_total[row["scale"]] += TARGET_ACCURACY * level
    return {
        scale: weighted_gap[scale] / weighted_total[scale]
        for scale in sorted(weighted_gap)
    }


def select_action(actions: list[dict], highest_gap_scale: str) -> dict:
    candidates = [
        action
        for action in actions
        if action["required_backbone"] == BACKBONE
        and highest_gap_scale in action["target_scales"]
    ]
    if not candidates:
        raise RuntimeError(f"No compatible action for {highest_gap_scale}")
    return max(
        candidates,
        key=lambda action: (
            len(action["target_scales"]),
            action["evidence_score"],
            -action["implementation_cost"],
        ),
    )


def find_case(rows: list[dict[str, str]], case: str) -> dict[str, str]:
    return next(row for row in rows if row["case"] == case)


def main() -> None:
    rows = load_rows()
    with (ROOT / "action_library.json").open(encoding="utf-8") as handle:
        actions = json.load(handle)

    before_gaps = normalized_gaps(rows, "before_accuracy")
    after_gaps = normalized_gaps(rows, "after_accuracy")
    highest_gap_scale = max(before_gaps, key=before_gaps.get)
    action = select_action(actions, highest_gap_scale)

    source = find_case(rows, "RadioML2018.01A")
    observation_l3 = find_case(rows, "bd_obs_l3")
    bivariate = find_case(rows, "stress_pair_snr5_obs5")
    accepted = (
        after_gaps[highest_gap_scale] < before_gaps[highest_gap_scale]
        and float(source["after_accuracy"]) >= float(source["before_accuracy"])
    )

    trace = {
        "trace_id": "mcnet_dcs_feedback_representative",
        "target_accuracy": TARGET_ACCURACY,
        "gap_definition": "max(0, target_accuracy - case_accuracy)",
        "backbone": BACKBONE,
        "before_gap_by_scale": before_gaps,
        "highest_gap_scale": highest_gap_scale,
        "selected_action": action,
        "model_change": {
            "source": "model_sources/mcnet/mcnet_model.py",
            "before": "MCNet",
            "after": "MCNetEnhanced",
        },
        "validation": {
            "observation_level_3": {
                "before": float(observation_l3["before_accuracy"]),
                "after": float(observation_l3["after_accuracy"]),
            },
            "snr_observation_bivariate": {
                "before": float(bivariate["before_accuracy"]),
                "after": float(bivariate["after_accuracy"]),
            },
            "source_dataset": {
                "before": float(source["before_accuracy"]),
                "after": float(source["after_accuracy"]),
            },
            "after_gap_by_scale": after_gaps,
        },
        "decision": {
            "accepted": accepted,
            "stopping_reason": "one_round_iteration_budget" if accepted else "action_rejected",
        },
        "scope": {
            "split": "fixed_disjoint_70_30_development_split",
            "checkpoint_selected_on_evaluation_split": True,
            "independent_final_seed": False,
            "global_optimality_guarantee": False,
        },
    }

    output = ROOT / "generated_trace.json"
    output.write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
