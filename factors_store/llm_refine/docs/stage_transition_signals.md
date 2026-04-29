# Stage Transition Signals

This note documents the signal/table layer used for formal stage-transition
decisions. The table policy is now the execution-facing transition decision;
the legacy `resolve_stage_transition` if/else resolver is retained only as an
audit reference.

Stage transition and round transition are intentionally separate:

| Layer | Question | Output |
|---|---|---|
| Stage transition | What research stage/action should this family move toward? | `StageTransitionDecision` |
| Round transition | Should the runner execute another round under the current authority and budget? | `RoundTransitionPlan` |

`StageTransitionDecision` is a research-state decision. `RoundTransitionPlan`
is the execution/budget gate that decides whether the stage decision can become
another launched round.

Thresholds are centralized in `search/policy_config.py` under
`DEFAULT_POLICY_CONFIG`. The numeric values below document the default
`refine_policy_config_v1`; they should not be duplicated as new hard-coded
constants in policy modules.

To tune policy behavior, change the matching dataclass in `policy_config.py`
first, then let signal extraction / decorrelation / saturation consume that
shared value. This keeps new knobs visible and prevents policy drift across
modules.

## Signals

### `anchor_strength`

| Value | Condition |
|---|---|
| `strong` | `passed_anchor_count >= 2` and `best_anchor.anchor_quality_score >= 0.70` |
| `passed` | `passed_anchor_count >= 1` |
| `weak` | no passed anchor, but `best_anchor` exists and `anchor_quality_score >= 0.50` |
| `none` | otherwise |

### `winner_quality`

Turnover is intentionally excluded here and modeled separately by
`turnover_pressure`.

| Value | Condition |
|---|---|
| `strong` | `quick_rank_icir >= 0.50` and `net_sharpe >= 3.0` |
| `usable` | `quick_rank_icir >= 0.40` and `net_sharpe >= 2.0` |
| `weak` | `quick_rank_icir >= 0.30` or `net_sharpe >= 1.5` |
| `none` | otherwise |

### `material_gain`

`material_gain` keeps the legacy "any threshold passes" logic.

| Field | Definition |
|---|---|
| `material_gain` | `true` if excess gain `>= 0.02`, ICIR gain `>= 0.05`, or Sharpe gain `>= 0.25` |
| `material_gain_score` | `max(excess_gain / 0.02, icir_gain / 0.05, sharpe_gain / 0.25)` |

Focused-stage baseline prefers the broad anchor; broad-stage baseline uses the
available keep/anchor reference.

### `corr_pressure`

`has_decorrelation_targets` is reported in diagnostics but does not create
pressure by itself.

| Value | Condition |
|---|---|
| `critical` | `high_corr_count >= 3` and high family overlap |
| `high` | `high_corr_count > 0` |
| `medium` | motif saturation / family crowding is visible, but no high-corr trigger fired |
| `low` | otherwise |

Runtime diagnostics used here include `high_corr_count`,
`portfolio_max_similarity`, `motif_usage_count`, `motif_counts`,
`family_overlap`, and `family_motif_saturation_penalty`.

### `turnover_pressure`

| Value | Condition |
|---|---|
| `critical` | winner turnover `> 0.80`, or multiple high-turnover candidates are present |
| `high` | winner turnover `> 0.65`, or `high_turnover_count > 0` |
| `medium` | winner turnover in `[0.50, 0.65]` |
| `low` | otherwise |

### `frontier_health`

This is the explicit replacement for the older branch-diversity intuition.

| Value | Condition |
|---|---|
| `exhausted` | `frontier_exhausted = true` |
| `high` | `children_added_to_search >= 5` and there is motif, branch, or cross-model diversity |
| `medium` | `children_added_to_search >= 2` |
| `low` | otherwise |

### `model_consensus`

This is currently a weak signal based on repeated motif/skeleton support
across models.

| Value | Condition |
|---|---|
| `high` | the same motif/skeleton has support from at least three models |
| `medium` | the same motif/skeleton has support from at least two models, or appears at least three times |
| `low` | otherwise |

### Direct Carry-Through Signals

| Signal | Definition |
|---|---|
| `no_improve_count` | integer consecutive no-improve count |
| `budget_exhausted` | boolean budget exhaustion flag |
| `frontier_exhausted` | boolean frontier exhaustion flag |
| `validation_fail_count` | integer candidate-level evaluation failure count; one failed candidate does not mean the round failed |

## Stage Actions

The stage table emits only:

- `continue_focused`
- `reopen_broad`
- `switch_to_complementarity`
- `confirmation`
- `terminate`

Promotion and formalization are intentionally excluded from this table and stay
in their own decision layer.

## Rule Priority

Rules are sorted by `specificity` descending, with table order breaking ties.
The first matching rule wins; there is no voting.

Important priority choices:

- `frontier_exhausted` is handled before empty/flat fallback rules.
- High or critical turnover outranks strong-winner continuation.
- `material_gain=false` plus high turnover routes to complementarity rather
  than confirmation.
- Correlation pressure is derived from runtime overlap/crowding diagnostics, not
  from `has_decorrelation_targets` alone.

## Stage Policy Table

The table below mirrors the stage policy table in
`search/transition/table_policy.py`. It is intentionally small and auditable: each
row matches a phase, checks simple conditions against extracted signals, and
emits one stage-transition action.

| Specificity | Rule | Phase | Conditions | Stage action | Next stage | Intent |
|---:|---|---|---|---|---|---|
| 100 | `frontier_exhausted_terminal` | `any` | `frontier_exhausted=true` | `terminate` | `terminate` | Hard-stop exhausted frontier before empty-flat fallback. |
| 95 | `round_failed_reopen` | `any` | `last_round_status=failed` | `reopen_broad` | `broad_followup` | Treat failed rounds as repair/reopen, not phase advancement. |
| 90 | `focused_turnover_no_gain_switch` | `focused_refine` | `turnover_pressure>=high`, `material_gain=false` | `switch_to_complementarity` | `focused_refine` | High turnover plus no material gain should not confirm by default. |
| 85 | `focused_turnover_switch` | `focused_refine` | `turnover_pressure>=high` | `switch_to_complementarity` | `focused_refine` | Turnover pressure outranks strong-winner continuation. |
| 80 | `focused_corr_critical_switch` | `focused_refine` | `corr_pressure>=high` | `switch_to_complementarity` | `focused_refine` | Redundant focused branches move toward complementarity. |
| 78 | `broad_corr_critical_reopen` | `new_family_broad`, `broad_followup`, `family_loop`, `auto` | `corr_pressure>=critical` | `reopen_broad` | `broad_followup` | Critical broad-stage crowding should diversify before graduation. |
| 70 | `broad_anchor_continue` | `new_family_broad`, `broad_followup`, `family_loop`, `auto` | `anchor_strength>=passed`, `turnover_pressure<=medium` | `continue_focused` | `focused_refine` | Passed anchor with manageable turnover graduates to focused search. |
| 65 | `broad_usable_winner_continue` | `new_family_broad`, `broad_followup`, `family_loop`, `auto` | `winner_quality>=usable`, `turnover_pressure<=medium` | `continue_focused` | `focused_refine` | Usable broad winner with manageable turnover is exploited. |
| 64 | `focused_complementarity_confirm` | `focused_refine` | `target_profile=complementarity`, `winner_quality>=usable`, `turnover_pressure<=medium` | `confirmation` | `confirmation` | Usable complementarity result should be confirmed before more mining. |
| 60 | `focused_material_gain_continue` | `focused_refine` | `winner_quality>=usable`, `material_gain=true`, `turnover_pressure<=medium` | `continue_focused` | `focused_refine` | Usable focused winner with material gain can continue. |
| 58 | `candidate_eval_fail_no_winner_reopen` | `any` | `validation_fail_count>=2`, `winner_quality<=weak` | `reopen_broad` | `broad_followup` | Multiple candidate-level eval failures with no usable winner should reopen/repair. |
| 55 | `focused_usable_no_gain_confirm` | `focused_refine` | `winner_quality>=usable`, `material_gain=false`, `turnover_pressure<=medium`, `corr_pressure<=medium` | `confirmation` | `confirmation` | Usable focused result with no material gain and no major pressure enters confirmation. |
| 50 | `no_improve_reopen` | `any` | `no_improve_count>=2`, `frontier_health>=medium` | `reopen_broad` | `broad_followup` | Repeated no-improve with live frontier should reopen broad search. |
| 48 | `no_improve_terminate` | `any` | `no_improve_count>=2`, `frontier_health<=low` | `terminate` | `terminate` | Repeated no-improve with weak frontier should stop. |
| 40 | `empty_or_low_frontier_broad` | `new_family_broad`, `broad_followup`, `family_loop`, `auto` | `winner_quality=none`, `anchor_strength=none`, `frontier_health<=low` | `terminate` | `terminate` | Broad search has no usable signal and poor frontier health. |
| 1 | `broad_default_reopen` | `new_family_broad`, `broad_followup`, `family_loop`, `auto` | none | `reopen_broad` | `broad_followup` | Default broad-stage table action keeps search open. |
| 1 | `focused_default_reopen` | `focused_refine` | none | `reopen_broad` | `broad_followup` | Default focused-stage table action reopens broad search. |
| 1 | `terminal_phase_terminate` | `confirmation`, `donor_validation` | none | `terminate` | `terminate` | Confirmation and donor-validation are terminal for stage transition. |

### Rule Syntax

The rule matcher supports a deliberately small condition grammar:

- Equality: `field=value`, `field!=value`
- Ordered levels: `anchor_strength>=passed`, `turnover_pressure<=medium`
- Numeric comparisons: `no_improve_count>=2`, `validation_fail_count>=2`
- Phase sets: `phase in {new_family_broad,broad_followup}` when needed

Ordered levels are defined in `table_policy.py`:

| Signal | Order |
|---|---|
| `anchor_strength` | `none < weak < passed < strong` |
| `winner_quality` | `none < weak < usable < strong` |
| `corr_pressure` | `low < medium < high < critical` |
| `turnover_pressure` | `low < medium < high < critical` |
| `frontier_health` | `exhausted < low < medium < high` |
| `model_consensus` | `low < medium < high` |

## Saturation Assessment

Family-loop and scheduler summaries now also write an advisory
`saturation_assessment`. This is deliberately **not** a stage-transition action
yet; it is a continuous audit signal for deciding later whether a family is
being over-mined.

| Field | Meaning |
|---|---|
| `score` | Weighted continuous score in `[0, 1]` |
| `grade` | `low`, `medium`, `high`, or `critical` from score thresholds |
| `recommended_escape_mode` | Advisory hint: `continue_local`, `diversify_within_family`, `switch_to_complementarity`, `fork_new_seed`, or `retire_family` |
| `components.corr` | Pressure from high-correlation diagnostics |
| `components.motif` | Motif/family crowding pressure |
| `components.turnover` | Pressure from high-turnover runtime signals |
| `components.plateau` | Consecutive no-improve pressure |
| `components.frontier` | Weak/exhausted frontier pressure, with medium frontier refined by motif/branch/model diversity |
| `components.anchor_reuse` | Approximate repeated-anchor / no-anchor pressure |

Default saturation v1.1 weights live in `SaturationPolicyConfig`:

| Component | Weight |
|---|---:|
| `corr` | `0.25` |
| `motif` | `0.18` |
| `turnover` | `0.25` |
| `plateau` | `0.12` |
| `frontier` | `0.08` |
| `anchor_reuse` | `0.12` |

Default grade thresholds:

| Grade | Score floor |
|---|---:|
| `medium` | `0.10` |
| `high` | `0.25` |
| `critical` | `0.45` |

The main table policy does not consume `saturation_assessment` in v1. This keeps
the production decision surface small while collecting enough data for a later
scoring/value layer.

## Artifact Fields

Family-loop and scheduler artifacts include:

- `stage_transition`: formal table-policy decision.
- `stage_transition_policy`: alias of the formal table-policy decision.
- `stage_transition_policy_source`: currently `table_policy`.
- `stage_transition_legacy_audit`: legacy if/else decision retained for audit.
- `stage_transition_legacy_compare`: legacy-vs-table agreement flags.
- `stage_transition_signals`: extracted signal values and diagnostics.
- `saturation_assessment`: advisory continuous family-saturation score and component breakdown.
- `round_transition_plan`: execution plan produced from the stage decision, authority mode, and budget gates.
- `transition_authority`: `audit_only`, `advisory`, or `guarded_control`.
- `budget_gate_decision`: compact copy of the round-controller budget gate diagnostics.
- `stage_transition_shadow_table`: deprecated alias of the table-policy decision.
- `stage_transition_shadow_compare`: deprecated alias of legacy-vs-table agreement flags.

## Round Transition Authority

`search/transition/round_controller.py` converts the stage decision into a
round-level execution plan.

| Authority | Effect |
|---|---|
| `audit_only` | Writes the plan but never controls execution. |
| `advisory` | Writes the plan and shows whether budget would allow a next round; existing runner limits still control execution. |
| `guarded_control` | Allows the table-policy decision to launch another round inside hard budget gates. |

Guarded control keeps hard brakes:

- `frontier_exhausted` and `budget_exhausted` stop immediately.
- `max_total_rounds` cannot be exceeded.
- `max_policy_extensions` limits extra rounds after the base `max_rounds`.
- `continue_focused` extensions require usable-or-better winner quality, safe
  turnover/correlation pressure, and low/medium saturation. Focused-stage
  extensions also require `material_gain=true`.

Scheduler CLI knobs:

```bash
--transition-authority advisory
--transition-authority guarded_control
--max-policy-extensions 2
--max-total-rounds 4
```
