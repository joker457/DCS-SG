# DCS Agent Feedback Protocol

This document specifies the auditable protocol represented by the DCS Agent experiments. The current implementation is a constrained language-agent workflow, not a globally optimal autonomous controller.

## Inputs

Each feedback cycle receives:

- source-dataset metrics;
- single-scale and bivariate DCS probe records;
- the nonnegative scale-gap vector computed with `max(0, target_accuracy - case_accuracy)`;
- model source files, run logs, configuration, and checkpoints;
- structured knowledge records that link AMC evidence to compatible revisions.

The reported study uses a target accuracy of 0.50. Level-weighted case gaps are normalized per scale so that larger values identify higher-priority empirical deficits.

## Action Selection and Execution

1. Rank scales by their normalized gaps and collect failed cases.
2. Retrieve knowledge records tagged with the highest-gap scales.
3. Discard actions incompatible with the model input, tensor shapes, or training protocol.
4. Rank the remaining actions by failed-scale coverage, evidence support, and implementation cost.
5. Apply the highest-ranked feasible action in an isolated model directory.
6. Train the revised model under its recorded model-specific initialization protocol, and retest the DCS and source-data cases.

The action rule is heuristic and traceable. It does not guarantee that the selected revision is globally optimal.

The language agent performs diagnosis, knowledge retrieval, candidate ranking, patch proposal, and command generation. An optional human approval gate authorizes code execution but does not change the computed gaps or the recorded action ranking.

## Acceptance and Stopping

An executed action is accepted when it reduces the target DCS gap and satisfies the prescribed source-accuracy tolerance. The reported runs use zero permitted source-accuracy decrease. The cycle stops when all target gaps close, no feasible action is accepted, or the iteration budget is exhausted. The study reports one feedback-update-retest cycle for each backbone.

The diagnosis, selected action, affected source files, and before/after measurements are written back as one trace record. See `examples/mcnet_end_to_end/` for a runnable representative record grounded in the reported MCNet runs. In that trace, the enhanced architecture was trained without an `init_from` checkpoint; initialization is therefore not represented as backbone-checkpoint continuation.

## Experimental Scope

The original and revised models see the same generated data for each DCS case, but the revised model receives additional adaptation computation and added module capacity. The current results therefore do not constitute an equal-budget robust-training control. Each case uses one deterministic seed and a fixed 70/30 development split, with the 30% subset also used for checkpoint selection. The records are development measurements, not independent multi-run confidence estimates or evidence of generalization to unseen scale combinations.
