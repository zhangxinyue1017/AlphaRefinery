# Family Loop Summary: qp_weighted_price_centroid

- target_profile: `raw_alpha`
- broad_stage_preset: `new_family_broad`
- focused_stage_preset: `focused_refine`
- broad_run_dir: `/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_family_loop/20260511_081152_qp_weighted_price_centroid/broad_runs/20260511_081153_qp_weighted_price_centroid`
- focused_run_dir: `/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_family_loop/20260511_081152_qp_weighted_price_centroid/focused_runs/20260511_130501_qp_weighted_price_centroid`
- recommended_next_step: `donor_mode`
- recommended_next_stage_preset: `donor_validation`
- next_action_mode: `decision_engine_v1`
- reason: focused 阶段没有继续抬高 anchor，但当前 anchor 足够强，适合转 donor/confirmation
- broad_stop_reason: `stage_policy_confirmation`
- focused_stop_reason: `stage_policy_confirmation`

## Prompt Trace
- broad_stage_mode: `focused_refine`
- broad_seed_stage_active: `False`
- broad_selected_parent_kind: `explicit_parent`
- broad_requested_candidate_count: `8`
- broad_bootstrap_frontier_count: `2`
- broad_donor_motifs_count: `0`
- focused_stage_mode: `focused_refine`
- focused_seed_stage_active: `False`
- focused_selected_parent_kind: `explicit_parent`
- focused_requested_candidate_count: `6`
- focused_bootstrap_frontier_count: `2`
- focused_donor_motifs_count: `0`

## Orchestration Trace (Advisory-driven)
- recommended_stage_mode: `focused_refine`
- round_strategy: `continue_focused`
- promotion_bias: `normal`
- parent_selection_bias: `best_node`
- termination_bias: `normal`
- confidence: `medium`

*Note: Stage transition is now table-policy driven. Legacy if/else output is retained only as an audit artifact.*

## Stage Transition Table Policy
- current_stage: `focused_refine`
- next_stage: `focused_refine`
- action: `continue_focused`
- confidence: `medium`
- termination_bias: `normal`
- parent_selection_bias: `best_node`
- target_profile_bias: `keep_current`
- rationale_tags: `table_policy, table_rule:focused_material_gain_continue, anchor_strength:strong, winner_quality:strong, corr_pressure:low, turnover_pressure:low, frontier_health:high, policy_config:stage_policy_v1`
- reason: focused result has usable quality and material incremental gain

## Legacy Transition Audit
- legacy_next_stage: `focused_refine`
- legacy_action: `continue_focused`
- legacy_confidence: `medium`
- legacy_rationale_tags: `focused_strong_winner`
- legacy_reason: focused round produced a strong winner; continue this mainline before freezing
- table_vs_legacy_stage_agrees: `True`
- table_vs_legacy_action_agrees: `True`

## Stage Transition Signals
- anchor_strength: `strong`
- winner_quality: `strong`
- material_gain: `True`
- material_gain_score: `8.5772`
- corr_pressure: `low`
- turnover_pressure: `low`
- frontier_health: `high`
- no_improve_count: `0`
- budget_exhausted: `False`
- frontier_exhausted: `False`
- model_consensus: `low`
- validation_fail_count: `2`

## Saturation Assessment
- grade: `low`
- score: `0.0`
- recommended_escape_mode: `continue_local`
- components: `corr=0.0, motif=0.0, turnover=0.0, plateau=0.0, frontier=0.0, anchor_reuse=0.0`
- advisory_only: `True`

## Round Transition Plan
- transition_authority: `advisory`
- control_effective: `False`
- execute_next_round: `False`
- next_stage_mode: `focused_refine`
- next_target_profile: `raw_alpha`
- policy_extension_granted: `False`
- policy_extension_count: `0`
- stop_reason: `max_total_rounds`
- reason: stage policy requested another round but max_total_rounds was reached
- budget_gate: `base_remaining=False, total_remaining=False, needs_extension=True`

## Broad Strongest
- factor: `llmgen.vwap_mean_decay_smooth_30`
- role: `research_winner`
- metrics: IC=0.0606, ICIR=0.4188, Ann=1.2914, Excess=-0.4400, Sharpe=2.6725, TO=0.0582

## Broad Round-Level Best Candidate
- factor: `llmgen.vwap_mean_decay_smooth_30`
- status: `research_winner`
- metrics: IC=0.0606, ICIR=0.4188, Ann=1.2914, Excess=-0.4400, Sharpe=2.6725, TO=0.0582

## Broad Global Best Keep
- factor: `llmgen.vwap_mean_ema_smooth_30`
- metrics: IC=0.0580, ICIR=0.3870, Ann=1.1929, Excess=-0.4472, Sharpe=2.4481, TO=0.0486

## Broad Candidate Snapshot
- total_candidates: `178`
- promotable_candidates: `95`
- evaluated_candidate_count: `163`
- research_drop_count: `51`
- evaluation_failed_count: `2`

## Selected Anchor
- anchor_selection_mode: `decision_engine_v1`
- corr_mode: `true_correlation`
- passed_count: `12`
- rejected_count: `62`
- factor: `llmgen.vwap_mean_volconfirm_20`
- metrics: IC=0.0887, ICIR=0.7324, Ann=4.6182, Excess=0.4880, Sharpe=6.0586, TO=0.2155
- similarity_to_parent: 0.0950
- true_corr_to_parent: 0.2446
- true_corr_to_stronger_candidate: 0.9073 vs `llmgen.vwap_mean_gap_turnover_gate`
- material_gain_vs_parent: True
- material_gain_vs_stronger_candidate: False
- auto_applied_promotion: False

## Broad Decision Counts
- drop_redundant_family: 5
- drop_redundant_family_exact: 13
- drop_redundant_parent: 14
- research_drop: 51
- research_keep: 55
- research_keep_exploratory: 21
- research_winner: 19

## Focused Best
- factor: `llmgen.llmgen_vwap_typical_medvol20`
- metrics: IC=0.0896, ICIR=0.7608, Ann=4.9904, Excess=0.6595, Sharpe=6.2954, TO=0.2907
- best_candidate: `llmgen.llmgen_vwap_typical_medvol20`
- best_keep: `llmgen.vwap_typical_turn_confirm20`

## Anchor -> Focused Delta
- delta_anchor_to_focused_quick_rank_ic_mean: 0.0009
- delta_anchor_to_focused_quick_rank_icir: 0.0284
- delta_anchor_to_focused_net_ann_return: 0.3722
- delta_anchor_to_focused_net_excess_ann_return: 0.1715
- delta_anchor_to_focused_net_sharpe: 0.2368
- delta_anchor_to_focused_mean_turnover: 0.0752

## Next Action Trace
- target_profile: `raw_alpha`
- stage_mode: `family_loop`
- broad_stop_reason: `stage_policy_confirmation`
- focused_stop_reason: `stage_policy_confirmation`
- passed_anchor_candidate_count: `12`
- strong_anchor: `True`
- focused_improved_vs_anchor: `False`
