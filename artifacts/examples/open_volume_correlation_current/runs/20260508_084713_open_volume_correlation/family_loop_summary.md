# Family Loop Summary: open_volume_correlation

- target_profile: `raw_alpha`
- broad_stage_preset: `new_family_broad`
- focused_stage_preset: `focused_refine`
- broad_run_dir: `/root/workspace/zxy_workspace/AlphaRefinery/artifacts/examples/open_volume_correlation_current/runs/20260508_084713_open_volume_correlation/broad_runs/20260508_084714_open_volume_correlation`
- focused_run_dir: ``
- recommended_next_step: `return_to_broad`
- recommended_next_stage_preset: `new_family_broad`
- next_action_mode: `decision_engine_v1`
- reason: broad 阶段没有候选通过 anchor graduation gate
- broad_stop_reason: `round_failed`
- focused_stop_reason: ``

## Prompt Trace
- broad_stage_mode: `focused_refine`
- broad_seed_stage_active: `False`
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
- recommended_stage_mode: `broad_followup`
- round_strategy: `reopen_broad`
- promotion_bias: `normal`
- parent_selection_bias: `diversify_branch`
- termination_bias: `normal`
- confidence: `low`

*Note: Stage transition is now table-policy driven. Legacy if/else output is retained only as an audit artifact.*

## Stage Transition Table Policy
- current_stage: `family_loop`
- next_stage: `broad_followup`
- action: `reopen_broad`
- confidence: `low`
- termination_bias: `normal`
- parent_selection_bias: `diversify_branch`
- target_profile_bias: `keep_current`
- rationale_tags: `table_policy, table_rule:broad_default_reopen, anchor_strength:none, winner_quality:none, corr_pressure:low, turnover_pressure:low, frontier_health:medium, policy_config:stage_policy_v1`
- reason: default broad-stage table action keeps search open

## Legacy Transition Audit
- legacy_next_stage: `broad_followup`
- legacy_action: `continue_broad_search`
- legacy_confidence: `medium`
- legacy_rationale_tags: `continue_broad`
- legacy_reason: no anchor-level evidence yet; continue broad search or diversify parent choice
- table_vs_legacy_stage_agrees: `True`
- table_vs_legacy_action_agrees: `False`

## Stage Transition Signals
- anchor_strength: `none`
- winner_quality: `none`
- material_gain: `False`
- material_gain_score: `0.0`
- corr_pressure: `low`
- turnover_pressure: `low`
- frontier_health: `medium`
- no_improve_count: `0`
- budget_exhausted: `False`
- frontier_exhausted: `False`
- model_consensus: `low`
- validation_fail_count: `0`

## Saturation Assessment
- grade: `medium`
- score: `0.12`
- recommended_escape_mode: `diversify_within_family`
- components: `corr=0.0, motif=0.0, turnover=0.0, plateau=0.0, frontier=0.45, anchor_reuse=0.7`
- advisory_only: `True`

## Round Transition Plan
- transition_authority: `advisory`
- control_effective: `False`
- execute_next_round: `False`
- next_stage_mode: `broad_followup`
- next_target_profile: `raw_alpha`
- policy_extension_granted: `False`
- policy_extension_count: `0`
- stop_reason: `max_total_rounds`
- reason: stage policy requested another round but max_total_rounds was reached
- budget_gate: `base_remaining=False, total_remaining=False, needs_extension=True`

## Broad Strongest
- factor: `llmgen.open_vol_corr_stable`
- role: `research_keep`
- metrics: IC=0.0420, ICIR=0.6163, Ann=1.2253, Excess=-0.3821, Sharpe=4.6216, TO=0.1883

## Broad Round-Level Best Candidate
- factor: ``
- status: ``
- metrics: (empty)

## Broad Global Best Keep
- factor: ``
- metrics: (empty)

## Broad Candidate Snapshot
- total_candidates: `2`
- promotable_candidates: `2`
- evaluated_candidate_count: `2`
- research_drop_count: `0`
- evaluation_failed_count: `0`

## Selected Anchor
- anchor_selection_mode: `decision_engine_v1`
- corr_mode: `true_correlation`
- passed_count: `0`
- rejected_count: `1`
- no anchor passed graduation gate

## Broad Decision Counts
- research_keep: 1
- research_keep_exploratory: 1

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
