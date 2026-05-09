# Family Loop Summary: open_volume_correlation

- target_profile: `raw_alpha`
- broad_stage_preset: `new_family_broad`
- focused_stage_preset: `focused_refine`
- broad_run_dir: `/root/workspace/zxy_workspace/AlphaRefinery/artifacts/examples/open_volume_correlation_current/runs/20260508_084643_open_volume_correlation/broad_runs/20260508_084644_open_volume_correlation`
- focused_run_dir: ``
- recommended_next_step: `return_to_broad`
- recommended_next_stage_preset: `new_family_broad`
- next_action_mode: `decision_engine_v1`
- reason: broad 阶段没有候选通过 anchor graduation gate
- broad_stop_reason: `round_failed`
- focused_stop_reason: ``

## Prompt Trace
- broad_stage_mode: `new_family_broad`
- broad_seed_stage_active: `True`
- broad_selected_parent_kind: `explicit_parent`
- broad_requested_candidate_count: ``
- broad_bootstrap_frontier_count: ``
- broad_donor_motifs_count: ``
- focused_stage_mode: ``
- focused_seed_stage_active: ``
- focused_selected_parent_kind: ``
- focused_requested_candidate_count: ``
- focused_bootstrap_frontier_count: ``
- focused_donor_motifs_count: ``

## Orchestration Trace (Advisory-driven)
- recommended_stage_mode: `terminate`
- round_strategy: `terminate`
- promotion_bias: `normal`
- parent_selection_bias: `best_node`
- termination_bias: `stop`
- confidence: `medium`

*Note: Stage transition is now table-policy driven. Legacy if/else output is retained only as an audit artifact.*

## Stage Transition Table Policy
- current_stage: `family_loop`
- next_stage: `terminate`
- action: `terminate`
- confidence: `medium`
- termination_bias: `stop`
- parent_selection_bias: `best_node`
- target_profile_bias: `keep_current`
- rationale_tags: `table_policy, table_rule:empty_or_low_frontier_broad, anchor_strength:none, winner_quality:none, corr_pressure:low, turnover_pressure:low, frontier_health:low, policy_config:stage_policy_v1`
- reason: broad search has no usable signal and low frontier health

## Legacy Transition Audit
- legacy_next_stage: `terminate`
- legacy_action: `freeze_or_switch_family`
- legacy_confidence: `medium`
- legacy_rationale_tags: `empty_flat_round`
- legacy_reason: round produced no children, no improvement, and no usable candidate
- table_vs_legacy_stage_agrees: `True`
- table_vs_legacy_action_agrees: `False`

## Stage Transition Signals
- anchor_strength: `none`
- winner_quality: `none`
- material_gain: `False`
- material_gain_score: `0.0`
- corr_pressure: `low`
- turnover_pressure: `low`
- frontier_health: `low`
- no_improve_count: `0`
- budget_exhausted: `False`
- frontier_exhausted: `False`
- model_consensus: `low`
- validation_fail_count: `0`

## Saturation Assessment
- grade: `medium`
- score: `0.14`
- recommended_escape_mode: `diversify_within_family`
- components: `corr=0.0, motif=0.0, turnover=0.0, plateau=0.0, frontier=0.7, anchor_reuse=0.7`
- advisory_only: `True`

## Round Transition Plan
- transition_authority: `advisory`
- control_effective: `False`
- execute_next_round: `False`
- next_stage_mode: `terminate`
- next_target_profile: `raw_alpha`
- policy_extension_granted: `False`
- policy_extension_count: `0`
- stop_reason: `stage_policy_terminate`
- reason: stage policy requested terminate
- budget_gate: `base_remaining=False, total_remaining=False, needs_extension=True`

## Broad Strongest
- factor: `llmgen.open_vol_amount_corr`
- role: `research_winner`
- metrics: IC=0.0488, ICIR=0.6686, Ann=1.3202, Excess=-0.3493, Sharpe=4.5023, TO=0.2553

## Broad Round-Level Best Candidate
- factor: ``
- status: ``
- metrics: (empty)

## Broad Global Best Keep
- factor: ``
- metrics: (empty)

## Broad Candidate Snapshot
- total_candidates: `0`
- promotable_candidates: `0`
- evaluated_candidate_count: `0`
- research_drop_count: `0`
- evaluation_failed_count: `0`

## Selected Anchor
- anchor_selection_mode: `decision_engine_v1`
- corr_mode: `heuristic_fallback`
- passed_count: `0`
- rejected_count: `0`
- no anchor passed graduation gate

## Focused Best
- factor: ``
- metrics: (empty)
- best_candidate: ``
- best_keep: ``

## Anchor -> Focused Delta
- no comparable focused delta

## Next Action Trace
- target_profile: `raw_alpha`
- stage_mode: `family_loop`
- broad_stop_reason: `round_failed`
- focused_stop_reason: ``
- passed_anchor_candidate_count: `0`
- strong_anchor: `False`
- focused_improved_vs_anchor: `False`
