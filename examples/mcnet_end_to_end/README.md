# Representative MCNet Feedback Cycle

This directory reconstructs one reported DCS Agent feedback cycle using compact records derived from the completed MCNet experiments. It exposes the four items required to audit the loop: feedback inputs, heuristic action selection, the resulting source-code change, and before/after validation.

The example is a lightweight trace replay, not a replacement for the full model-training runs. Large generated datasets and checkpoints are intentionally excluded from this public repository.

## Run

```powershell
python examples/mcnet_end_to_end/run_feedback_example.py
```

The command reads `probe_results.csv` and `action_library.json`, recomputes the nonnegative level-weighted gap for every scale, ranks compatible actions, applies the recorded acceptance rule, and rewrites the committed `generated_trace.json`.

## Audited Cycle

1. **Input evidence.** At target accuracy 0.50, observation length has the largest original normalized gap, 0.405270.
2. **Selected action.** The highest-ranked compatible action adds observation-aware masked pooling and zero-start residual channel and temporal gates.
3. **Model modification.** The reproduced and revised definitions are `MCNet` and `MCNetEnhanced` in [`../../model_sources/mcnet/mcnet_model.py`](../../model_sources/mcnet/mcnet_model.py). The revised implementation is localized in `ResidualChannelGate`, `ResidualTemporalGate`, `_obs_width_mask`, and `_masked_pool`.
4. **Validation.** Observation-Level-3 accuracy changes from 0.481964 to 0.890968; the SNR+observation bivariate case changes from 0.047674 to 0.388851; and source-dataset accuracy changes from 0.582136 to 0.593695.
5. **Decision.** The action is accepted because the target-scale gap decreases and source accuracy does not decrease. The reported cycle then stops at its one-round iteration budget.

The language agent computes and ranks the actions. A human approval gate may authorize filesystem execution, but it does not alter the gap calculation or recorded ranking. The rule is knowledge constrained and heuristic; it provides no global optimality guarantee.

## Data-Split Scope

Each generated case uses a fixed, disjoint 70% training and 30% evaluation split, and the reproduced and revised models use identical sample indices. The 30% subset also selects the best checkpoint. These values are therefore held-out-sample development measurements, not independently seeded final-test estimates. The enhanced MCNet run used its recorded architecture-specific training configuration and did not load an original-model checkpoint through `init_from`.
