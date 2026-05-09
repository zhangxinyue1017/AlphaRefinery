# Multi-Model Scheduler Summary: open_volume_correlation

- stage_mode: `new_family_broad`
- target_profile: `raw_alpha`
- scheduler_dir: `/root/workspace/zxy_workspace/AlphaRefinery/artifacts/examples/open_volume_correlation_current/runs/20260508_084713_open_volume_correlation/broad_runs/20260508_084714_open_volume_correlation`
- rounds_completed: `2`
- stop_reason: `round_failed`
- last_round_status: `failed`
- last_selected_parent_name: `llmgen.open_vol_amount_corr`

## Prompt Trace
- stage_mode: `focused_refine`
- prompt_template_version: ``
- seed_stage_active: `False`
- selected_parent_kind: `explicit_parent`
- requested_candidate_count: ``
- bootstrap_frontier_count: ``
- donor_motifs_count: ``

## Core Family Flow
- family_state: `exploring`
- recommended_action: `reopen_broad`
- action_reason: failed rounds should reopen/repair instead of advancing phase
- next_stage_mode: `broad_followup`
- next_target_profile: `raw_alpha`
- transfer_mode: `none`
- transfer_action: `reopen_broad`
- transfer_donor_motifs_count: `0`
- transfer_donor_families: ``

## Shared Context
- search_phase: `refining`
- exploration_pressure: `medium`
- redundancy_pressure: `low`
- prompt_constraint_style: `structured`
- memory_mode: `standard`
- examples_mode: `family_only`
- branching_bias: `stay_local`
- next_action_bias: `continue_focused`

## Orchestration Trace (Advisory-driven)
- recommended_stage_mode: `broad_followup`
- round_strategy: `reopen_broad`
- promotion_bias: `normal`
- parent_selection_bias: `diversify_branch`
- termination_bias: `normal`
- confidence: `low`
- rationale_tags: `table_policy, table_rule:round_failed_reopen, anchor_strength:none, winner_quality:none, corr_pressure:low, turnover_pressure:low, frontier_health:low, policy_config:stage_policy_v1`

*Note: Stage transition is now table-policy driven. Legacy if/else output is retained only as an audit artifact.*

## Stage Transition Table Policy
- current_stage: `focused_refine`
- next_stage: `broad_followup`
- action: `reopen_broad`
- confidence: `low`
- termination_bias: `normal`
- parent_selection_bias: `diversify_branch`
- target_profile_bias: `keep_current`
- rationale_tags: `table_policy, table_rule:round_failed_reopen, anchor_strength:none, winner_quality:none, corr_pressure:low, turnover_pressure:low, frontier_health:low, policy_config:stage_policy_v1`
- reason: failed rounds should reopen/repair instead of advancing phase

## Legacy Transition Audit
- legacy_next_stage: `focused_refine`
- legacy_action: `repair_or_retry`
- legacy_confidence: `low`
- legacy_rationale_tags: `last_round_failed`
- legacy_reason: last round failed; keep execution manual and inspect failure reasons
- table_vs_legacy_stage_agrees: `False`
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
- validation_fail_count: `0`

## Saturation Assessment
- grade: `medium`
- score: `0.14`
- recommended_escape_mode: `diversify_within_family`
- components: `corr=0.0, motif=0.0, turnover=0.0, plateau=0.0, frontier=0.7, anchor_reuse=0.7`
- advisory_only: `True`

## Round Transition Plan
- transition_authority: `guarded_control`
- control_effective: `True`
- execute_next_round: `True`
- next_stage_mode: `broad_followup`
- next_target_profile: `raw_alpha`
- policy_extension_granted: `True`
- policy_extension_count: `2`
- stop_reason: ``
- reason: guarded_control granted one policy extension
- budget_gate: `base_remaining=False, total_remaining=True, needs_extension=True`

## Runtime Evidence
- selected_parent_kind: `archive_winner`
- requested_candidate_count: `2`
- final_candidate_target: `2`
- has_decorrelation_targets: `False`

## Last Round Winner
- factor: ``
- status: ``
- metrics: (empty)

## Last Round Best Candidate
- factor: ``
- status: ``
- metrics: (empty)

## Last Round Best Keep
- factor: ``
- status: ``
- metrics: (empty)

## Search Best Node
- factor: `llmgen.open_vol_corr_stable`
- status: `research_keep`
- metrics: IC=0.0420, ICIR=0.6163, Ann=1.2253, Excess=-0.3821, Sharpe=4.6216, TO=0.1883

## Last Round Rollup
- child_stage_mode: `focused_refine`
- search_improved: `False`
- children_collected: `0`
- children_added_to_search: `0`
- successful_model_count: `0`
- failed_model_count: `1`
