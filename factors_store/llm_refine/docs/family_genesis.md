# Family Genesis

`llm_refine` treats a family as a research unit, not as a Python module or a
random bucket of generated factors.

A family starts from one curated seed motif. The seed defines the economic
story, the variables that are allowed to move, the local edit space, and the
boundary that refined variants should not cross.

## Definition

A factor family is a group of seeds and refined variants that share:

- one core economic or microstructure hypothesis,
- a recognizable formula skeleton or variable interaction,
- a bounded set of allowed local edits,
- known weaknesses worth refining,
- and a non-duplication boundary versus neighboring families.

The machine-readable source of truth is
[`config/refinement_seed_pool.yaml`](../../../config/refinement_seed_pool.yaml).
Each `seed_groups` entry is a family definition.

## Seed Eligibility

A factor can become a `canonical_seed` when it satisfies the following checks:

| Check | Meaning |
|---|---|
| Interpretability | The factor has a clear economic or microstructure story. |
| Empirical viability | It is stable enough to justify refinement, not pure noise. |
| Editability | The formula has local mutation space such as window changes, proxy substitutions, smoothing, decay, normalization, or one auxiliary branch. |
| Non-duplication | It is not merely a near-copy of an existing family. |
| Boundary clarity | The family has known weaknesses, allowed edit types, and anti-patterns. |
| Public resolvability | The seed can be resolved from tracked code, public formula overrides, or optional local extensions. |

These checks are intentionally stricter than "the seed once backtested well".
The goal is to refine durable motifs, not chase isolated formulas.

## Canonical Seed, Aliases, And Preferred Seed

`canonical_seed` is the identity anchor for the family. It is the factor used to
name and explain the family.

`aliases` are nearby factors that express nearly the same motif. They are useful
for prompt context and boundary checks, but they do not create separate
families.

`preferred_refine_seed` may point to a stronger local parent when the canonical
seed is useful for identity but another known variant is a better starting
point for refinement.

## Family Boundary

The family boundary answers two questions:

- What must stay invariant?
- What would make a candidate belong to another family?

The default boundary is defined by `family_origin_defaults.family_boundary`.
Per-family overrides may narrow it.

Good refinements stay inside the family by preserving the seed story while
editing local mechanics:

- windows,
- proxy variables,
- normalization,
- neutralization,
- smoothing or decay,
- one auxiliary interaction branch.

Bad refinements cross the boundary when they become:

- pure sign flips,
- unrelated economic stories,
- duplicate variants of another family,
- complexity-only expansions.

## Discovery Modes

`family_origin.discovery_mode` records how a family entered the seed pool.

| Mode | Meaning |
|---|---|
| `manual_curated` | A human selected the family from known factors and wrote its boundary. |
| `metric_screened` | A candidate passed empirical screening before curation. |
| `motif_clustered` | A group of related formulas was clustered by motif and then curated. |

The current public seed pool is primarily `manual_curated`. That is acceptable
as long as the curation rules are explicit and versioned.

## Transfer

Transfer is family-to-family reuse of motifs. It should not copy a formula
verbatim.

`export_donor` means a mature family has a motif that may help neighbors.
`import_donor` means the current family can borrow that motif, but it must be
translated back into the target family's boundary.

This is why transfer uses family-level metadata instead of only factor-level
metrics.
