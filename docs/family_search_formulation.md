# Family-Level Staged Refinement as a Search Problem

This note formalizes AlphaRefinery's family-level staged refinement loop as a
sequential search / decision process.

The goal is not to introduce a new mechanism immediately. The goal is to make
the existing `llm_refine` workflow explicit enough that future implementation
changes can be evaluated against a stable problem definition.

---

## 1. Problem Definition

For a given factor family `F`, AlphaRefinery repeatedly decides how to continue
the search:

```text
S_t --choose A_t--> generated candidates --evaluate--> feedback R_t --update--> S_{t+1}
```

At each step:

* `S_t` is the observable family search state before a refinement action.
* `A_t` is the chosen candidate-generation action.
* `R_t` is the evaluation feedback produced by the action.
* `T(S_t, A_t, R_t)` is the transition that updates the family state.
* `pi(S_t)` is the search policy that selects the next action.
* `V(S_t, A_t)` is the continuation value of taking action `A_t` in state `S_t`.

This turns the family workflow from a fixed pipeline into a formal decision
process over parents, stages, objectives, branches, and budgets.

---

## 2. Family State `S_t`

`S_t` is the state of one family at the beginning of round `t`.

It should describe what is already known about the family, what remains
promising, and what risks are likely to block further promotion.

Recommended structure:

```text
S_t = {
  family_id,
  stage,
  target_profile,
  archive_summary,
  parent_set,
  best_node,
  frontier_nodes,
  branch_state,
  motif_state,
  redundancy_state,
  failure_state,
  promotion_state,
  budget_state
}
```

### `family_id`

The current factor family, such as:

```text
qp_apb_price_bias
abnormal_volume_attention
gp_historical_anchor_ratio
```

### `stage`

The current search stage.

Examples:

```text
new_family_broad
broad_followup
focused_refine
confirmation
donor_validation
```

The stage determines the action space. A broad stage should explore multiple
motifs; a focused stage should refine a selected parent; a confirmation stage
should verify stability and deployability.

### `target_profile`

The active optimization profile:

```text
raw_alpha
deployability
complementarity
robustness
```

This controls how evaluation metrics, redundancy, turnover, and continuation
value are weighted.

### `archive_summary`

A compact summary of historical candidates in the family:

```text
archive_summary = {
  candidate_count,
  evaluated_count,
  winner_count,
  keep_count,
  latest_round,
  metric_distributions,
  recent_best_metrics
}
```

This is derived from the local archive and completed run summaries.

### `parent_set`

The current set of possible parents for expansion.

Parents should not be limited to the single best factor. Useful parent classes
include:

```text
best_quality_parent
low_corr_parent
low_turnover_parent
failed_but_strong_parent
underexplored_branch_parent
donor_or_canonical_seed_parent
```

This distinction matters because the strongest parent is often also the parent
most likely to create high-correlation children.

### `best_node`

The current best node under the active policy objective.

This may differ by target profile:

* `raw_alpha`: best predictive quality.
* `deployability`: best quality after turnover/complexity penalties.
* `complementarity`: best quality under redundancy pressure.

### `frontier_nodes`

Nodes that still have continuation value.

A frontier node can be weaker than the best node but still valuable if it:

* sits in a different motif region,
* has lower correlation to existing factors,
* has lower turnover,
* or has not yet been sufficiently expanded.

### `branch_state`

Branch-level state tracks which search branches remain useful:

```text
branch_state = {
  branch_id,
  depth,
  best_score,
  latest_score,
  child_count,
  no_improve_count,
  saturation_score,
  continuation_score
}
```

This prevents a single strong branch from absorbing the entire search budget.

### `motif_state`

Motif-level state tracks recurring expression structures.

Examples:

```text
vwap_ratio
volume_spike
turnover_confirm
anchor_gap
price_position
decay_smooth
```

Recommended fields:

```text
motif_state = {
  motif_counts,
  motif_quality,
  motif_failure_modes,
  saturated_motifs,
  underexplored_motifs
}
```

Motif saturation is important when a family keeps producing high-quality but
highly correlated candidates.

### `redundancy_state`

Redundancy state captures correlation pressure.

```text
redundancy_state = {
  max_corr_to_family,
  max_corr_to_library,
  nearest_family_factor,
  nearest_library_factor,
  decorrelation_targets,
  saturated_corr_clusters
}
```

This is the key state for deciding whether to continue raw-alpha mining or
switch toward complementarity / decorrelation.

### `failure_state`

Recent failure modes should be part of the state.

Examples:

```text
failure_state = {
  parse_fail_count,
  validation_filter_count,
  high_turnover_count,
  high_corr_count,
  low_ic_count,
  low_icir_count,
  low_win_rate_count,
  no_new_winner_count,
  provider_error_count
}
```

This allows the next action to respond to actual failure reasons rather than
only to headline performance metrics.

### `promotion_state`

Promotion state tracks which candidates were formalized and which passed
downstream checks if those checks are available locally.

```text
promotion_state = {
  in_py_count,
  pending_count,
  admitted_count,
  rejected_count,
  rejection_reasons
}
```

The core `llm_refine` workflow does not require external admission. However,
local downstream outcomes can still be treated as optional feedback when
available.

### `budget_state`

Budget state tracks remaining compute and search budget:

```text
budget_state = {
  remaining_rounds,
  candidate_budget,
  model_budget,
  max_parallel,
  elapsed_time,
  stop_rule_state
}
```

---

## 3. Action `A_t`

`A_t` is the refinement action chosen at state `S_t`.

It is more than a CLI invocation. It is a structured decision:

```text
A_t = {
  stage_mode,
  parent_selection,
  target_profile,
  policy_preset,
  model_allocation,
  generation_constraints,
  decorrelation_targets,
  candidate_budget,
  stopping_rule
}
```

### Decision variables in `A_t`

The main decision variables are:

```text
stage_mode
parent_set
target_profile
policy_preset
models
n_candidates
max_rounds
max_parallel
decorrelation_targets
promotion_policy
stop_if_no_new_winner
```

These are the knobs the system or operator can choose before launching a round.

### Example action classes

```text
broad_explore
focused_quality_refine
focused_decorrelation_refine
deployability_refine
confirmation_run
donor_seed_probe
dual_parent_branch_run
```

These action classes can all map to existing CLI parameters.

---

## 4. Evaluation Feedback `R_t`

`R_t` is the feedback after executing `A_t`.

It should represent the value added by the action, not only the metrics of the
single best child.

Recommended structure:

```text
R_t = {
  generated_candidates,
  parse_feedback,
  evaluation_metrics,
  winner_set,
  keep_set,
  promotion_feedback,
  redundancy_feedback,
  failure_feedback,
  cost_feedback
}
```

### Useful reward components

```text
quality_gain
novelty_gain
admission_or_promotion_gain
deployability_gain
branch_diversity_gain
failure_penalty
cost_penalty
```

In practice:

* A high-IC child with very high correlation may have low net reward.
* A slightly weaker but low-correlation child may have high continuation value.
* A failed run can still be useful if it identifies a saturated motif or bad
  parent choice.

---

## 5. Transition Function

The transition updates the family state:

```text
S_{t+1} = T(S_t, A_t, R_t)
```

The transition should update:

* archive records,
* best node,
* frontier nodes,
* branch scores,
* motif saturation,
* redundancy pressure,
* failure histograms,
* promotion state,
* and remaining budget.

Stage progression is part of the transition, not only part of the policy:

```text
S_{t+1}.stage = g_stage(S_t, A_t, R_t)
```

`g_stage` should make the following decisions explicit.

### Stage Transition Logic

#### Broad termination

Broad exploration ends when at least one of the following is true:

* the broad-stage budget is exhausted,
* the broad frontier is exhausted,
* the broad loop reaches its no-new-winner patience,
* a candidate passes the anchor graduation gate,
* or the broad run fails and no recoverable frontier remains.

Broad does not end just because a single candidate is best by raw metrics. It
ends when there is enough evidence to either graduate an anchor or conclude
that the current broad budget did not find one.

#### Broad -> Anchor

A broad candidate becomes an anchor candidate when it passes a graduation gate:

```text
status in {research_winner, winner, research_keep, keep}
and eligible_for_best_node
and metrics_completeness >= threshold
and quick_rank_icir >= threshold
and net_sharpe >= threshold
and mean_turnover <= threshold
and not blocked by parent/sibling correlation guard without material gain
```

The selected anchor is the passing candidate with the highest anchor quality
score under the active target profile.

#### Anchor -> Focused

The process enters focused refinement when a best anchor exists. The focused
stage uses that anchor as the current parent and switches the action space from
motif exploration to local deepening, simplification, confirmation, or
decorrelation around the anchor.

If no anchor passes, the next state should be `broad_followup` or
`new_family_broad`, depending on whether the family still has recoverable broad
frontier.

#### Focused continuation

Focused refinement continues when the focused best node improves the anchor or
previous focused best node with material gain and without materially worsening
deployability:

```text
improved_metric = any(IC, ICIR, return, excess_return, Sharpe improves)
turnover_not_much_worse = focused_turnover <= baseline_turnover + tolerance
material_gain = excess_gain >= threshold
             or icir_gain >= threshold
             or sharpe_gain >= threshold
continue_focused = improved_metric
                and turnover_not_much_worse
                and material_gain
                and focused_budget_remains
```

If improvement exists but material gain is small, the next state should move to
`confirmation`.

#### Branch reopen

A retired or deprioritized branch should be reopened when its continuation
value rises again. Typical triggers are:

* another broad candidate passed the anchor gate but was not selected,
* the focused branch went flat and broad still has multiple viable anchors,
* a branch has lower redundancy or turnover than the current best branch,
* a branch has high expandability / branch value despite weaker current quality,
* a new target profile such as `complementarity` makes that branch more valuable,
* or new evaluation feedback reduces a previous failure reason.

Branch reopen is different from ordinary frontier reranking: it changes a
branch from inactive / saturated back into the eligible frontier.

#### Terminate / freeze

The family search should terminate or freeze when:

* no broad anchor passes and broad budget is exhausted,
* focused refinement fails to improve and no alternative broad anchor remains,
* the frontier is exhausted by depth, branch, or family budget limits,
* repeated no-improvement rounds hit patience,
* failure modes dominate the remaining budget,
* or the best available node should be kept as the family result without more
  generation.

This state can be represented as `freeze_anchor`, `terminate_family`, or
`return_later`, depending on whether a usable anchor exists.

#### Promote / confirmation / donor validation

Promotion is a downstream transition from search state to formalized factor
state. A candidate should move toward promotion or confirmation when it passes
the research keep/winner gate and the target profile's deployability and
redundancy checks are acceptable.

If the anchor is already strong but focused refinement is flat, the next state
can become `donor_validation`: the factor is useful as a donor or canonical
reference even if it is not worth spending more local search budget.

Current implementation pieces already perform parts of this transition, but
the logic is distributed rather than owned by a single transition controller:

```text
archive ingestion          -> candidate history
SearchEngine               -> frontier / parent selection state
SearchPolicy               -> scoring semantics
DecisionContext            -> action context
ContextResolver            -> stage/context recommendation
DecisionEngine             -> anchor and post-focused recommendation
promotion.py               -> py formalization and keep/winner gate
research_funnel.py         -> family-level outcome summaries
```

Current code coverage is partial:

* `SearchEngine.can_continue()` implements budget/frontier/no-improvement stop
  conditions inside one scheduler run.
* `SearchFrontier` implements branch and motif caps for active frontier
  selection, but not a first-class branch reopen state.
* `DecisionEngine.select_anchor()` implements the broad-to-anchor graduation
  gate.
* `DecisionEngine.recommend_next_action()` implements a v1 post-focused
  recommendation among `continue_focused`, `confirmation`, `donor_mode`,
  `return_to_broad`, and `freeze_anchor`.
* `resolve_orchestration_profile()` implements a lightweight next-stage
  recommendation from runtime evidence and last-round status.
* `run_refine_family_loop.py` wires a deterministic v1 sequence:
  `broad -> anchor selection -> focused -> summary`.
* `run_refine_multi_model_scheduler.py` records orchestration hints in run
  summaries, but it does not yet call a centralized `g_stage` function to own
  stage progression across runs.

The missing abstraction is a single explicit `FamilyState` view that gathers
these pieces before a decision is made, plus a single stage-transition
controller that owns `g_stage`.

In other words, stage logic exists in code, but it is not yet systematized as a
first-class transition policy. Today it is a combination of hard-coded family
loop wiring, context heuristics, search-engine stop rules, anchor selection,
and manual run configuration.

---

## 6. Continuation Value `V(S_t, A_t)`

`V(S_t, A_t)` estimates whether an action is worth taking.

A practical first version can be heuristic:

```text
V(S_t, A_t) =
  expected_quality_uplift
+ expected_promotion_probability
+ expected_diversity_gain
+ expected_deployability_gain
- expected_corr_risk
- expected_turnover_risk
- expected_parse_failure_risk
- compute_cost
```

This value explains decisions such as:

* whether to keep mining the same family,
* whether to switch to a new or legacy family,
* whether to select the strongest parent or a lower-correlation parent,
* whether to use `raw_alpha` or `complementarity`,
* and whether a branch should be preserved or retired.

---

## 7. Search Policy `pi(S_t)`

The policy maps state to action:

```text
pi(S_t) -> A_t
```

It should answer:

* Which stage should run next?
* Which parent or parents should be expanded?
* Which target profile should be active?
* Which models should be used?
* How much candidate budget should be allocated?
* Which decorrelation targets should be included?
* Which branches should be retained?

The existing `SearchPolicy` and scheduler implement a substantial part of this
logic. Formalizing `pi` makes it easier to compare policies and reason about
why one run configuration was selected over another.

---

## 8. Observations vs Decisions vs Outcomes

Separating observations, decisions, and outcomes avoids mixing state with
control knobs.

### Observed variables

Observed variables are facts known before selecting the next action:

```text
family history
candidate metrics
archive records
nearest correlations
turnover statistics
complexity statistics
stage history
recent failure reasons
motif counts
promotion outcomes
optional downstream check outcomes
```

### Decision variables

Decision variables are chosen by the policy or operator:

```text
stage_mode
target_profile
policy_preset
parent_selection
decorrelation_targets
models
n_candidates
max_rounds
max_parallel
stop_rule
promotion_gate
```

### Outcomes

Outcomes are produced after a run:

```text
generated candidates
parse success / failure
validation filter results
evaluation metrics
research_winner / research_keep labels
py promotion results
optional downstream check results
updated archive
updated frontier
updated family state
```

---

## 9. Mapping to Current Implementation

| Formal object | Current implementation surface |
| --- | --- |
| `S_t.family_id` | `--family`, seed pool family config |
| `S_t.archive_summary` | `llm_refine_archive.db`, run summaries |
| `S_t.parent_set` | selected parent, bootstrap frontier, dual-parent logic |
| `S_t.frontier_nodes` | `SearchEngine.frontier_snapshot()` |
| `S_t.redundancy_state` | decorrelation targets, redundancy scoring, correlation gates |
| `S_t.failure_state` | parser/validation logs, no-new-winner counters, run summaries |
| `A_t.stage_mode` | `--stage-mode` |
| `A_t.target_profile` | `--target-profile` |
| `A_t.policy_preset` | `--policy-preset` |
| `A_t.model_allocation` | `--models`, `--max-parallel` |
| `A_t.candidate_budget` | `--n-candidates`, `--max-rounds` |
| `R_t.evaluation_metrics` | evaluator summary CSV / JSON |
| `R_t.winner_set` | `research_winner`, `research_keep` |
| `T` | archive ingestion, scheduler update, promotion update |
| `pi` | scheduler parent selection, search policy, manual run configuration |
| `V` | current score / continuation heuristics |

---

## 10. Practical Implementation Path

The first code-level step is a read-only stage-transition advisory layer, not a
rewrite of the scheduler.

Implemented state/action/feedback objects:

```python
@dataclass
class FamilyState:
    family_id: str
    stage: str
    target_profile: str
    parent_set: list[ParentNode]
    best_node: ParentNode | None
    frontier_nodes: list[ParentNode]
    motif_state: MotifState
    redundancy_state: RedundancyState
    failure_state: FailureState
    promotion_state: PromotionState
    budget_state: BudgetState
```

```python
@dataclass
class RefinementAction:
    stage_mode: str
    target_profile: str
    policy_preset: str
    parent_selection: str
    decorrelation_targets: list[str]
    models: list[str]
    n_candidates: int
    max_rounds: int
```

```python
@dataclass
class EvaluationFeedback:
    status: str
    search_improved: bool
    winner: dict
    keep: dict
    best_anchor: dict
    passed_anchor_count: int
    focused_best_node: dict
    consecutive_no_improve: int
    high_corr_count: int
    high_turnover_count: int
    validation_fail_count: int
    budget_exhausted: bool
    frontier_exhausted: bool
```

Implemented transition decision object:

```python
@dataclass
class StageTransitionDecision:
    current_stage: str
    next_stage: str
    action: str
    confidence: str
    reason: str
    rationale_tags: list[str]
    parent_selection_bias: str
    target_profile_bias: str
    termination_bias: str
    branch_reopen_candidates: list[str]
```

Implemented resolver:

```python
def resolve_stage_transition_from_state(
    state: FamilyState,
    action: RefinementAction,
    feedback: EvaluationFeedback,
) -> StageTransitionDecision:
    ...
```

There is also a lower-level compatibility resolver:

```python
def resolve_stage_transition(evidence: StageTransitionEvidence) -> StageTransitionDecision:
    ...
```

Initial rule:

* `FamilyState` should be derived from existing archive/run artifacts.
* It should not own execution.
* It should not change current behavior.
* It should only make the scheduler's implicit context inspectable.
* `StageTransitionDecision` is first written into `summary.json` and
  `summary.md` as an audit artifact before it controls any automatic launch.

Once this is stable, `V(S_t, A_t)`, `pi(S_t)`, and `g_stage(S_t, A_t, R_t)` can
be implemented as explicit scoring and recommendation layers.

---

## 11. Design Boundary

This formulation is intentionally independent of any private downstream
admission system.

External or local library checks can provide optional feedback, especially for
correlation and promotion outcomes. However, the core family-level search
problem is fully defined by:

```text
family state
candidate generation action
evaluation feedback
state transition
search policy
continuation value
```

This keeps AlphaRefinery usable as a standalone LLM refinement framework while
still allowing local workflows to add downstream constraints when available.
