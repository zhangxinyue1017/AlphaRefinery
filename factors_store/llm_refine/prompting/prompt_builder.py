'''Prompt assembly logic for LLM refinement candidates.

Combines family metadata, memory, constraints, examples, roles, and output schema instructions.
'''

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from textwrap import dedent

import pandas as pd

from factors_store.contract import CORE_FIELDS, DERIVED_FIELDS, EXTENDED_DAILY_FIELDS, OPTIONAL_CONTEXT_FIELDS

from ..config import ARTIFACTS_DIR
from ..core.models import PromptBundle, SeedFamily, SeedPool
from ..core.seed_loader import (
    apply_direction_rule,
    resolve_factor_direction,
    resolve_family_formula,
    resolve_preferred_refine_seed,
)
from ..memory import build_memory_snapshot, render_memory_snapshot_prompt_block
from ..parsing.operator_contract import PROMPT_OPERATOR_DESCRIPTIONS
from .prompt_plan import PromptConstraintPlan, PromptExamplesPlan, build_prompt_plan
from ..search.transition.context_resolver import ContextProfile

DEFAULT_WINDOWS = (3, 5, 10, 14, 15, 20, 28, 40, 60, 100, 120, 180, 250, 375)
DEFAULT_OPERATORS = PROMPT_OPERATOR_DESCRIPTIONS
PROMPT_TEMPLATE_VERSIONS = ("current_compact", "legacy_v1")


def _prompt_expression_text(expression: str, *, max_chars: int = 240) -> str:
    text = str(expression or "").strip()
    if len(text) <= max_chars:
        return text
    head = max_chars // 2 - 8
    tail = max_chars - head - 5
    return f"{text[:head]} ... {text[-tail:]}"


def _normalize_current_prompt_indent(text: str) -> str:
    prefix = " " * 12
    return "\n".join(line[len(prefix) :] if line.startswith(prefix) else line for line in text.splitlines()).strip()


def _family_formula_lines(family: SeedFamily, *, limit: int | None = None) -> str:
    items = list(family.formulas.items())
    if limit is not None:
        items = items[: max(int(limit), 0)]
    return "\n".join(
        f"- {name}: {_prompt_expression_text(resolve_family_formula(family, name))}"
        for name, _ in items
    )


def _family_summary(family: SeedFamily) -> str:
    formulas = _family_formula_lines(family)
    weaknesses = "\n".join(f"- {item}" for item in family.likely_weaknesses)
    axes = "\n".join(f"- {item}" for item in family.refinement_axes)
    aliases = ", ".join(family.aliases) if family.aliases else "(none)"
    return dedent(
        f"""
        family: {family.family}
        canonical_seed: {family.canonical_seed}
        preferred_refine_seed: {family.preferred_refine_seed or family.canonical_seed}
        aliases: {aliases}
        direction_rule: {family.direction}
        relation_note: {family.relation_note or "(none)"}

        formulas:
        {formulas}

        interpretation:
        {family.interpretation}

        likely_weaknesses:
        {weaknesses}

        refinement_axes:
        {axes}
        """
    ).strip()


def _family_formulas_block(family: SeedFamily, *, limit: int | None = None) -> str:
    return _family_formula_lines(family, limit=limit)


def _family_weakness_block(family: SeedFamily) -> str:
    return "\n".join(f"{idx}. {item}" for idx, item in enumerate(family.likely_weaknesses, start=1))


def _family_axes_block(family: SeedFamily, *, max_items: int = 3) -> str:
    axes = list(family.refinement_axes[: max(int(max_items), 0)])
    if not axes:
        return "- (none)"
    return "\n".join(f"- {item}" for item in axes)


def _family_constraint_block(items: tuple[str, ...]) -> str:
    if not items:
        return "- (none)"
    return "\n".join(f"- {item}" for item in items)


def _decorrelation_target_block(
    targets: tuple[str, ...],
    *,
    family: SeedFamily,
) -> str:
    if not targets:
        return "- (none)"
    lines: list[str] = []
    for name in targets:
        expression = ""
        if name == family.canonical_seed or name in family.aliases or name in family.formulas:
            expression = resolve_family_formula(family, name)
        if expression:
            lines.append(f"- `{name}`: `{_prompt_expression_text(expression)}`")
        else:
            lines.append(f"- `{name}`")
    return "\n".join(lines)


def _decorrelation_guidance_block(targets: tuple[str, ...]) -> str:
    if not targets:
        return ""
    return dedent(
        """
        Decorrelation experiment requirements:
        - Reduce structural overlap and correlation with the target set while preserving signal quality.
        - Prefer denominator or anchor changes, alternative transforms, conditional regimes, or one orthogonal supporting signal.
        - Avoid cosmetic smoothing swaps, window-only changes, neighboring mean/EMA substitutions, and weak normalization that leaves the main skeleton unchanged.
        - Preserve the family's economic logic; distance from the parent is not an objective by itself.
        """
    ).strip()


def _family_operator_hint_block(family: SeedFamily) -> str:
    if family.family not in {
        "amplitude_structure",
        "qp_amplitude_sliced_momentum",
        "qp_high_amplitude_reversal",
    }:
        return ""
    return dedent(
        """
        Family implementation notes:
        - Use `bucket_sum(x, key, window, q, side)` for within-window buckets; `side` must be `"low"` or `"high"`.
        - Use `where(cond, x, y)` / `if_then_else(cond, x, y)` for branches.
        - Use `count(cond, window)`, `sumif(x, window, cond)`, and `gt/lt/ge/le` for conditional aggregation.
        - Example keys:
          `amplitude = sub(div(high, add(low, 1e-12)), 1)`
          `upper_shadow = sub(high, rowmax(open, close))`
          `lower_shadow = sub(rowmin(open, close), low)`
          `intraday_position = div(sub(close, low), add(sub(high, low), 1e-12))`
        - For an amplitude-sliced return, consider:
          `bucket_sum(delay(returns, 1), delay(sub(div(high, add(low, 1e-12)), 1), 1), 60, 0.3, "low")`
        - Preserve the sliced or state-conditioned structure instead of reducing it to a generic `mean/std/rank` variant.
        """
    ).strip()


def _family_expression_safety_block(family: SeedFamily) -> str:
    if family.family not in {
        "qp_low_price_accumulation_pressure",
        "qp_high_price_distribution_pressure",
    }:
        return ""
    return dedent(
        """
        Family expression safety:
        - Do not emit registered factor names or internal helpers such as:
          `qp_pressure.net_pressure_20`, `qp_pressure.low_price_volume_share_20`,
          `low_price_volume_share(...)`, `high_price_volume_share(...)`,
          `buy_pressure(...)`, `sell_pressure(...)`,
          `share_of_volume_traded_on_low_price_days(...)`.
        - These are logic labels, not executable tokens. Rewrite them using contract primitives such as:
          `if_then_else` + `ts_quantile` + `ts_sum` + `div` + `add` + `mul`.
        - A low-price-day volume share can be written as:
          `div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))`
        - Pass `ts_quantile`'s third argument positionally (`0.3/0.7`), not as `q=0.3`.
        - Do not emit dotted references or use peer factor names as expressions.
        """
    ).strip()


def _global_expression_engine_safety_block() -> str:
    return dedent(
        """
        Expression-engine syntax:
        - Do not use keyword arguments except `half_life` and `min_obs`.
        - Do not write `q=0.3`, `m=1`, `side="low"`, `window=20`, or `cond=...`.
        - Use positional arguments, for example:
          `ts_quantile(close, 20, 0.3)`
          `sma(x, 20, 1)`
          `bucket_sum(x, key, 60, 0.3, "low")`
        - Do not emit dotted references such as `family.factor_name`, `module.helper(...)`, or `qp_pressure.xxx`.
        - Use only listed fields and operators. Rewrite helper-dependent ideas with executable primitives.
        """
    ).strip()


def _candidate_role_block_english(roles: tuple[str, ...], n_candidates: int) -> str:
    roles = list(roles[: int(n_candidates)])
    labels = {
        "conservative": "preserve the parent logic and improve stability or turnover",
        "confirmation": "add one lightweight confirmation term",
        "decorrelating": "reduce structural redundancy with the parent and peers",
        "donor_transfer": "translate a donor motif into the current family logic",
        "simplifying": "remove unnecessary nesting while preserving the core signal",
        "stretch": "make a bolder change while preserving the economic hypothesis",
        "enhancing": "add one economical confirmation term to improve the signal",
    }
    lines = []
    for idx, role in enumerate(roles, start=1):
        letter = chr(ord("A") + idx - 1)
        lines.append(f"- Candidate {letter}: {role}; {labels.get(role, 'follow this role objective')}")
    return "\n".join(lines)


def _candidate_role_block(roles: tuple[str, ...], n_candidates: int) -> str:
    roles = list(roles[: int(n_candidates)])
    labels = {
        "conservative": "尽量保留 parent 逻辑，主要修稳定性/换手",
        "confirmation": "保持主逻辑，只增加一个轻量确认项",
        "decorrelating": "尽量与 parent / peer 拉开结构差异，主要解决冗余",
        "donor_transfer": "借 donor motif，但必须翻译到当前 family 逻辑里",
        "simplifying": "减少复杂度或压平不必要嵌套，保留有效主结构",
        "stretch": "允许更大胆改写，但仍必须保持 family 经济故事",
        "enhancing": "在不显著增加复杂度下增加一个确认项，主要提升表现上限",
    }
    lines = []
    for idx, role in enumerate(roles, start=1):
        letter = chr(ord("A") + idx - 1)
        lines.append(f"- Candidate {letter}: {role}；{labels.get(role, '按该角色目标生成候选')}")
    return "\n".join(lines)


def _bootstrap_frontier_block(frontier: list[dict[str, object]], *, limit: int | None = None) -> str:
    if not frontier:
        return "- (none)"
    entries = list(frontier)
    if limit is not None:
        entries = entries[: max(int(limit), 0)]
    lines: list[str] = []
    for idx, item in enumerate(entries, start=1):
        factor_name = str(item.get("factor_name", "")).strip() or f"bootstrap_{idx}"
        expression = _prompt_expression_text(str(item.get("expression", "")).strip())
        metric_parts: list[str] = []
        for label, value in (
            ("ICIR", item.get("quick_rank_icir")),
            ("Excess", item.get("net_excess_ann_return", item.get("net_ann_return"))),
            ("Sharpe", item.get("net_sharpe")),
            ("TO", item.get("mean_turnover")),
        ):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            metric_parts.append(f"{label}={float(value):.4f}")
        metrics = f" | {'; '.join(metric_parts)}" if metric_parts else ""
        lines.append(f"- `{factor_name}`: `{expression}`{metrics}")
    return "\n".join(lines)


def _donor_motif_block(donor_motifs: list[dict[str, object]]) -> str:
    if not donor_motifs:
        return "- (none)"
    return _donor_motif_block_limited(donor_motifs, limit=None)


def _donor_motif_block_limited(donor_motifs: list[dict[str, object]], *, limit: int | None = None) -> str:
    if not donor_motifs:
        return "- (none)"
    entries = list(donor_motifs)
    if limit is not None:
        entries = entries[: max(int(limit), 0)]
    lines: list[str] = []
    for item in entries:
        source_family = str(item.get("source_family", "")).strip() or "-"
        source_factor = str(item.get("source_factor_name", "")).strip() or "-"
        expression = _prompt_expression_text(str(item.get("source_expression", "")).strip())
        rationale = str(item.get("rationale", "")).strip()
        metric_parts: list[str] = []
        motif_score = item.get("motif_score")
        if motif_score is not None:
            metric_parts.append(f"MotifScore={float(motif_score):.2f}")
        for label, value in (
            ("ICIR", item.get("quick_rank_icir")),
            ("Excess", item.get("net_excess_ann_return")),
            ("Sharpe", item.get("net_sharpe")),
            ("TO", item.get("mean_turnover")),
        ):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                continue
            metric_parts.append(f"{label}={float(value):.4f}")
        metrics = f" | {'; '.join(metric_parts)}" if metric_parts else ""
        line = f"- `{source_family} / {source_factor}`: `{expression}`{metrics}"
        if rationale:
            line += f"\n  rationale: {rationale}"
        lines.append(line)
    return "\n".join(lines)


def _render_examples_section(
    *,
    family: SeedFamily,
    plan: PromptExamplesPlan,
    is_seed_stage: bool,
    bootstrap_frontier: list[dict[str, object]],
    donor_motifs: list[dict[str, object]],
) -> str:
    blocks: list[str] = []
    if plan.include_family_formulas:
        blocks.extend(
            [
                "Related factors in this family:",
                _family_formulas_block(family, limit=plan.family_formula_limit),
            ]
        )
    if plan.include_bootstrap_frontier and is_seed_stage:
        blocks.extend(
            [
                "Round1 bootstrap frontier:",
                _bootstrap_frontier_block(bootstrap_frontier, limit=plan.bootstrap_frontier_limit),
            ]
        )
    if plan.include_donor_motifs:
        blocks.extend(
            [
                "Cross-family donor motifs:",
                _donor_motif_block_limited(donor_motifs, limit=plan.donor_motif_limit),
            ]
        )
    return "\n\n".join(blocks).strip()


def _render_constraints_section(
    *,
    family: SeedFamily,
    plan: PromptConstraintPlan,
    decorrelation_targets: tuple[str, ...],
    donor_motifs: list[dict[str, object]],
    requested_count: int,
    final_count: int,
    active_roles: tuple[str, ...],
) -> str:
    lines: list[str] = []
    lines.extend(
        [
            "Research objectives:",
            f"- Primary: {family.primary_objective or 'improve_rank_icir'}",
            f"- Secondary: {family.secondary_objective or 'maintain_family_logic_while_lowering_redundancy'}",
        ]
    )
    lines.extend(["", "Core requirements:"])
    if plan.include_family_constraints:
        constraints_text = _family_constraint_block(family.hard_constraints)
        if constraints_text != "- (none)":
            lines.append(constraints_text)
    lines.extend(
        [
            "- Preserve the family's economic hypothesis.",
            "- Address the parent's clearest weakness with a small, testable change.",
            "- For a hybrid parent, adjust direction, weight, or smoothing before removing a branch.",
        ]
    )

    if plan.include_axes_guidance:
        lines.extend(
            [
                "",
                "Preferred refinement axes:",
                _family_axes_block(family),
            ]
        )

    lines.extend(
        [
            "",
            "Edit scope:",
            "- Direction, smoothing, normalization, or one confirmation term may change.",
            "- Keep the main structure recognizable and avoid unnecessary fields or nesting.",
            "",
            "Candidate roles:",
            "- Use distinct refinement approaches across candidates.",
            "- Avoid formulas that differ from the parent or peers only cosmetically.",
            "",
            f"Generate exactly {int(requested_count)} candidates. The first {len(active_roles)} must cover these role slots:",
            _candidate_role_block_english(active_roles, len(active_roles)),
        ]
    )

    if plan.include_allowed_edit_types:
        lines.extend(["", "Allowed edit types:", _allowed_edit_block(family)])

    lines.extend(["", _global_expression_engine_safety_block()])

    lines.extend(
        [
            "",
            "Balance:",
            "- predictive stability",
            "- turnover",
            "- within-family redundancy",
            "- clear economic interpretation",
        ]
    )
    if donor_motifs:
        lines.append("- At least one candidate must translate a donor motif into the current family semantics.")
    if requested_count > final_count:
        lines.append(
            f"- This seed stage will rerank the broader set to {int(final_count)} finalists; prioritize role coverage and structural diversity."
        )

    if plan.include_anti_patterns and family.anti_patterns:
        lines.extend(["", "Prohibited patterns:", _family_constraint_block(family.anti_patterns)])
    else:
        lines.extend(
            [
                "",
                "Prohibited patterns:",
                "- No name-only or window-only edits.",
                "- No future information, external fundamentals, or unavailable fields.",
                "- Do not leave the family hypothesis or cosmetically rename the parent or peers.",
            ]
        )
    if plan.include_decorrelation_guidance and decorrelation_targets:
        lines.extend(["", _decorrelation_guidance_block(decorrelation_targets)])
    return "\n".join(lines).strip()


def _allowed_edit_block(family: SeedFamily) -> str:
    items = family.allowed_edit_types or (
        "window_replacement",
        "operator_replacement",
        "normalization_insertion",
        "smoothing_insertion",
    )
    return "\n".join(f"- {item}" for item in items)


def _legacy_constraints_section(
    *,
    family: SeedFamily,
    requested_count: int,
    active_roles: tuple[str, ...],
) -> str:
    lines: list[str] = [
        "本轮优化优先级：",
        f"- 主目标：{family.primary_objective or 'improve_rank_icir'}",
        f"- 次目标：{family.secondary_objective or 'maintain_family_logic_while_lowering_redundancy'}",
        "",
        "优先考虑以下改进方向：",
        _family_axes_block(family),
        "",
        "同时你需要兼顾：",
        "- 提高稳定性",
        "- 降低换手",
        "- 降低家族内冗余",
        "- 保持经济解释清晰",
        "- 不要让公式复杂度无意义膨胀",
        "",
        f"你必须生成 {int(requested_count)} 个候选，且前 {len(active_roles)} 个候选优先覆盖这些角色：",
        _candidate_role_block(active_roles, len(active_roles)),
        "",
        "允许的 edit 类型：",
        _allowed_edit_block(family),
        "",
        "小步改写限制：",
        "- 最多只允许改动 parent 中 2~4 个核心子结构。",
        "- 不允许完全替换主逻辑分支。",
        "- 不允许新增超过 2 层嵌套。",
        "- 不允许引入超过 2 个新的原始字段。",
        "- 不允许只做符号翻转，除非明确说明理由。",
        "- 不允许窗口跨度跳跃过大，除非有明确经济理由。",
        "",
        "候选多样性要求：",
        "- 不允许所有候选都只做窗口微调。",
        "- 不允许所有候选共享同一个主 operator skeleton。",
        "- 至少 1 个候选主要面向降低 turnover。",
        "- 至少 1 个候选主要面向降低与 parent 的相关性。",
        "- 至少 1 个候选要改变主刻画方式，但不能脱离 family 逻辑。",
        "",
        "不要输出以下坏模式：",
        _family_constraint_block(family.anti_patterns),
    ]
    return "\n".join(lines).strip()


def _build_legacy_user_prompt(
    *,
    seed_pool: SeedPool,
    family: SeedFamily,
    available_fields: list[str],
    history_stage: str,
    effective_parent_name: str,
    effective_parent_expression: str,
    effective_parent_row: dict[str, object] | None,
    requested_count: int,
    active_roles: tuple[str, ...],
    additional_notes: str,
) -> str:
    return dedent(
        f"""
        你的目标是：在不明显增加公式复杂度的前提下，尽量提升因子的 RankIC、RankICIR、Sharpe 和 excess return，并尽量改善稳定性、降低冗余、降低不必要的高换手。

        # 1. 现有因子信息
        当前 parent 因子名称：{effective_parent_name}
        因子家族：{family.family}
        当前 parent 因子原始表达式：{_prompt_expression_text(effective_parent_expression)}
        {_effective_expression_block(effective_parent_expression, family, factor_name=effective_parent_name)}

        家族内已有公式：
        {_family_formulas_block(family)}

        经济假设：{family.interpretation}
        已知弱点：
        {_family_weakness_block(family)}

        相似/同家族因子：
        {_family_formulas_block(family)}

        历史表现：
        {_history_scope_block(seed_pool, history_stage)}

        {_history_block(family, history_stage=history_stage, current_parent_name=effective_parent_name, current_parent_row=effective_parent_row)}

        {additional_notes or ""}

        # 2. 可用资源
        可用字段：
        {json.dumps(available_fields, ensure_ascii=False)}

        可用算子：
        {json.dumps(list(DEFAULT_OPERATORS), ensure_ascii=False)}

        可用时间窗：
        {json.dumps(list(DEFAULT_WINDOWS), ensure_ascii=False)}

        注意：
        1. 原因子默认应继承当前 family 的有效符号方向：{family.direction}
        2. 不要只是机械调窗口。
        3. 必须保留当前 family 的核心经济逻辑，不要彻底换思路。
        4. 不要引入未来信息、外部财务字段、或当前 contract 中没有的字段。
        5. 候选之间要尽量 low corr、低冗余，不要给出三个几乎一样的公式。
        6. 要尽量降低与 parent factor 及同家族 peer 的相关性，至少在结构上做出清晰区分。

        # 3. 优化要求
        {_legacy_constraints_section(family=family, requested_count=requested_count, active_roles=active_roles)}

        # 4. 分析要求
        请先在内部完成分析，但不要展示推理过程。必须综合以下三个维度：
        1. Optimization Strategy
        2. Alpha Idea
        3. Factor Interpretation

        你最终输出的 explanation 必须把这三部分信息融合成自然语言，具体、直白、能让研究员快速理解。
        """
    ).strip()


def _latest_csv(pattern: str) -> Path | None:
    matches = sorted(ARTIFACTS_DIR.rglob(pattern))
    return matches[-1] if matches else None


def prompt_history_stage(seed_pool: SeedPool) -> str:
    if seed_pool.evaluation_protocol is None:
        return ""
    return str(seed_pool.evaluation_protocol.prompt_history_stage or "").strip()


def _protocol_stage_window_text(seed_pool: SeedPool, stage: str) -> str:
    protocol = seed_pool.evaluation_protocol
    if protocol is None or not stage:
        return ""
    window = getattr(protocol, stage, None)
    if window is None:
        return ""
    start = getattr(window, "start", "") or "(unspecified)"
    end = getattr(window, "end", "") or "latest"
    return f"{start} 至 {end}"


def _history_scope_block(seed_pool: SeedPool, history_stage: str) -> str:
    if not history_stage:
        return ""
    window_text = _protocol_stage_window_text(seed_pool, history_stage)
    keep_stage = (
        str(seed_pool.evaluation_protocol.keep_decision_stage or "").strip()
        if seed_pool.evaluation_protocol is not None
        else ""
    )
    promote_stage = (
        str(seed_pool.evaluation_protocol.promote_stage or "").strip()
        if seed_pool.evaluation_protocol is not None
        else ""
    )
    keep_label = keep_stage or "selection"
    promote_label = promote_stage or keep_label or "selection"
    return dedent(
        f"""
        历史表现时间边界说明：
        - 以下历史表现仅来自 `{history_stage}` 阶段。
        - `{history_stage}` 时间范围：{window_text}。
        - 这些指标只用于帮助你理解该 family 在搜索阶段的强项、短板和可能失败模式，不代表最终样本外结论。
        - `{keep_label}` 与 `{promote_label}` 阶段结果不在本 prompt 中，你不能假设已经看到了更晚时期的表现。
        - 请不要把 `{history_stage}` 阶段表现直接理解为最终保留或最终样本外表现。
        """
    ).strip()


def _history_scope_block_english(seed_pool: SeedPool, history_stage: str) -> str:
    if not history_stage:
        return ""
    protocol = seed_pool.evaluation_protocol
    window = getattr(protocol, history_stage, None) if protocol is not None else None
    start = getattr(window, "start", "") or "unspecified"
    end = getattr(window, "end", "") or "latest"
    keep_stage = str(getattr(protocol, "keep_decision_stage", "") or "selection")
    promote_stage = str(getattr(protocol, "promote_stage", "") or keep_stage)
    hidden_stages = " and ".join(f"`{item}`" for item in dict.fromkeys((keep_stage, promote_stage)))
    return dedent(
        f"""
        Historical-metric boundary:
        - Metrics below come only from the `{history_stage}` stage ({start} to {end}).
        - They describe search-stage strengths and failure modes, not final out-of-sample results.
        - {hidden_stages} metrics are excluded from this generation prompt.
        """
    ).strip()


def _summary_factor_name_candidates(factor_name: str, family: SeedFamily | None = None) -> tuple[str, ...]:
    candidates = [factor_name]
    underscored = factor_name.replace(".", "_")
    candidates.append(underscored)
    candidates.append(f"seed_baseline.{underscored}")
    if family is not None:
        if family.direction == "use_negative_sign":
            candidates.append(f"seed_baseline.neg_{underscored}")
        elif family.direction == "use_positive_sign":
            candidates.append(f"seed_baseline.pos_{underscored}")
    # Preserve order while removing duplicates.
    return tuple(dict.fromkeys(candidates))


def _read_summary_row(
    path: Path, factor_name: str, *, family: SeedFamily | None = None
) -> dict[str, object] | None:
    df = pd.read_csv(path)
    if "factor_name" not in df.columns:
        return None
    row = df[df["factor_name"].isin(_summary_factor_name_candidates(factor_name, family))]
    if row.empty:
        return None
    payload = row.iloc[0].to_dict()
    payload.pop("Unnamed: 0", None)
    payload["_source_file"] = path.name
    return payload


def _load_summary_row(factor_name: str, *, stage: str = "", family: SeedFamily | None = None) -> dict[str, object] | None:
    library = factor_name.split(".", 1)[0]
    if stage:
        preferred_patterns: list[str] = []
        if family is not None:
            seed_suffix = family.canonical_seed.split(".")[-1]
            preferred_patterns.extend(
                [
                    f"{family.family}_{stage}_summary_*.csv",
                    f"{family.family}_{stage}_summary.csv",
                    f"llm_refined_{seed_suffix}_family_{stage}_summary_*.csv",
                    f"llm_refined_{seed_suffix}_family_{stage}_summary.csv",
                ]
            )
        preferred_patterns.extend(
            [
                f"{library}_*_{stage}_summary_*.csv",
                f"{library}_{stage}_summary_*.csv",
            ]
        )
        if library.startswith("qp_"):
            preferred_patterns = [
                f"qp_*_{stage}_summary_*.csv",
                f"qp_{stage}_summary_*.csv",
                *preferred_patterns,
            ]
        if library == "gp_mined":
            preferred_patterns = [
                f"gp_mined_{stage}_summary_*.csv",
                f"gp_*_{stage}_summary_*.csv",
                *preferred_patterns,
            ]
    else:
        preferred_patterns = [
            f"{library}_full_from2018_summary_*.csv",
            f"{library}_all_backtest_summary_*.csv",
            f"{library}_*_summary_*.csv",
        ]
        if library.startswith("qp_"):
            preferred_patterns = [
                "qp_full_from2018_summary_*.csv",
                "qp_*_summary_*.csv",
                *preferred_patterns,
            ]
        if library == "gp_mined":
            preferred_patterns = [
                "gp_mined_backtest_summary_*.csv",
                "gp_*_summary_*.csv",
                *preferred_patterns,
            ]
    for pattern in preferred_patterns:
        path = _latest_csv(pattern)
        if path is None:
            continue
        payload = _read_summary_row(path, factor_name, family=family)
        if payload is not None:
            return payload
    return None


def load_prompt_history_row(seed_pool: SeedPool, factor_name: str, *, family: SeedFamily | None = None) -> dict[str, object] | None:
    return _load_summary_row(factor_name, stage=prompt_history_stage(seed_pool), family=family)


def _load_refined_family_snapshot(family: SeedFamily, *, stage: str = "", topk: int = 3) -> list[dict[str, object]]:
    seed_suffix = family.canonical_seed.split(".")[-1]
    patterns = (
        [f"llm_refined_{seed_suffix}_family_{stage}_summary_*.csv"]
        if stage
        else [f"llm_refined_{seed_suffix}_family_summary_*.csv"]
    )
    path = None
    for pattern in patterns:
        path = _latest_csv(pattern)
        if path is not None:
            break
    if path is None:
        return []
    df = pd.read_csv(path)
    df = df.drop(columns=["Unnamed: 0"], errors="ignore")
    keep_cols = [
        "factor_name",
        "quick_rank_ic_mean",
        "quick_rank_icir",
        "net_ann_return",
        "mean_turnover",
    ]
    existing = [col for col in keep_cols if col in df.columns]
    return df[existing].head(topk).to_dict(orient="records")


def _apply_direction_adjustment(row: dict[str, object], family: SeedFamily) -> dict[str, object]:
    adjusted = dict(row)
    if "quick_rank_ic_mean" not in adjusted and "quick_mean_ic" in adjusted:
        adjusted["quick_rank_ic_mean"] = adjusted.get("quick_mean_ic")
    if "quick_rank_icir" not in adjusted and "quick_icir" in adjusted:
        adjusted["quick_rank_icir"] = adjusted.get("quick_icir")
    if family.direction != "use_negative_sign":
        return adjusted
    for key in ("quick_rank_ic_mean", "quick_rank_icir", "net_ann_return"):
        value = adjusted.get(key)
        if isinstance(value, (int, float)) and not pd.isna(value):
            adjusted[key] = -float(value)
    return adjusted


def _format_metric_line(name: str, row: dict[str, object], family: SeedFamily) -> str:
    row = _apply_direction_adjustment(row, family)

    def _fmt(key: str) -> str:
        value = row.get(key)
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return "NA"
        if isinstance(value, (int, float)):
            return f"{float(value):.6f}"
        return str(value)

    return (
        f"- {name}: "
        f"RankIC={_fmt('quick_rank_ic_mean')}, "
        f"RankICIR={_fmt('quick_rank_icir')}, "
        f"AnnReturn={_fmt('net_ann_return')}, "
        f"ExcessReturn={_fmt('net_excess_ann_return')}, "
        f"Sharpe={_fmt('net_sharpe')}, "
        f"Turnover={_fmt('mean_turnover')}"
    )


def _history_block(
    family: SeedFamily,
    *,
    history_stage: str = "",
    current_parent_name: str | None = None,
    current_parent_row: dict[str, object] | None = None,
) -> str:
    lines: list[str] = []
    has_metrics = False
    if history_stage:
        lines.append(f"{history_stage} 阶段历史快照：")
    if current_parent_name and current_parent_row:
        lines.append("Current parent snapshot:")
        lines.append(_format_metric_line(current_parent_name, current_parent_row, family))
        has_metrics = True

    parent_row = _load_summary_row(family.canonical_seed, stage=history_stage, family=family)
    if parent_row and current_parent_name != family.canonical_seed:
        lines.append("Parent seed snapshot:")
        lines.append(_format_metric_line(family.canonical_seed, parent_row, family))
        has_metrics = True

    alias_rows = []
    for alias in family.aliases:
        row = _load_summary_row(alias, stage=history_stage, family=family)
        if row:
            alias_rows.append((alias, row))
    if alias_rows:
        lines.append("Peer/alias snapshots:")
        lines.extend(_format_metric_line(alias, row, family) for alias, row in alias_rows)
        has_metrics = True

    refined_rows = _load_refined_family_snapshot(family, stage=history_stage)
    if refined_rows:
        lines.append("Previous refined family snapshot:")
        lines.extend(_format_metric_line(str(row.get("factor_name", "candidate")), row, family) for row in refined_rows)
        has_metrics = True

    if not has_metrics:
        prefix = f"{history_stage} 阶段暂无运行时补充指标；" if history_stage else ""
        return f"{prefix}请主要依据 family 逻辑、弱点和优化方向生成方案。"
    return "\n".join(lines)


def _history_block_english(
    family: SeedFamily,
    *,
    history_stage: str = "",
    current_parent_name: str | None = None,
    current_parent_row: dict[str, object] | None = None,
) -> str:
    lines: list[str] = []
    has_metrics = False
    if history_stage:
        lines.append(f"{history_stage} stage snapshot:")
    if current_parent_name and current_parent_row:
        lines.extend(["Current parent snapshot:", _format_metric_line(current_parent_name, current_parent_row, family)])
        has_metrics = True

    parent_row = _load_summary_row(family.canonical_seed, stage=history_stage, family=family)
    if parent_row and current_parent_name != family.canonical_seed:
        lines.extend(["Parent seed snapshot:", _format_metric_line(family.canonical_seed, parent_row, family)])
        has_metrics = True

    alias_rows = []
    for alias in family.aliases:
        row = _load_summary_row(alias, stage=history_stage, family=family)
        if row:
            alias_rows.append((alias, row))
    if alias_rows:
        lines.append("Peer/alias snapshots:")
        lines.extend(_format_metric_line(alias, row, family) for alias, row in alias_rows)
        has_metrics = True

    refined_rows = _load_refined_family_snapshot(family, stage=history_stage)
    if refined_rows:
        lines.append("Previous refined family snapshot:")
        lines.extend(_format_metric_line(str(row.get("factor_name", "candidate")), row, family) for row in refined_rows)
        has_metrics = True

    if not has_metrics:
        prefix = f"No additional runtime metrics are available for `{history_stage}`. " if history_stage else ""
        return f"{prefix}Use the family hypothesis, weaknesses, and refinement axes."
    return "\n".join(lines)


def _effective_expression_block(expression: str, family: SeedFamily, *, factor_name: str = "") -> str:
    if not expression:
        return ""
    adjusted = apply_direction_rule(expression, resolve_factor_direction(family, factor_name or family.canonical_seed))
    if adjusted == expression:
        return ""
    if resolve_factor_direction(family, factor_name or family.canonical_seed) in {"use_negative_sign", "use_positive_sign"}:
        return f"当前 parent 有效方向表达式：{_prompt_expression_text(adjusted)}"
    return ""


def _effective_expression_block_english(expression: str, family: SeedFamily, *, factor_name: str = "") -> str:
    if not expression:
        return ""
    direction = resolve_factor_direction(family, factor_name or family.canonical_seed)
    adjusted = apply_direction_rule(expression, direction)
    if adjusted != expression and direction in {"use_negative_sign", "use_positive_sign"}:
        return f"Effective parent expression after direction rule: {_prompt_expression_text(adjusted)}"
    return ""


def _build_legacy_output_prompt(
    *,
    prompt_body: str,
    requested_count: int,
    active_roles: tuple[str, ...],
    output_schema: dict[str, object],
) -> str:
    return dedent(
        f"""
        {prompt_body}

        # 5. 输出格式要求（必须严格遵循）
        你必须生成 {int(requested_count)} 个 candidate，并且只能输出一个 JSON object。
        每个 candidate 都必须是一个“低冗余、可落地、与 parent family 同逻辑但有明显差异”的小步改写。
        每个 `name` 用 snake_case，简短稳定。
        Candidate 列表顺序很重要：前 {len(active_roles)} 个候选必须分别对应上面的 role slots，后续额外候选可自由竞争。
        每个 `candidate_role` 必须从 {json.dumps(list(active_roles), ensure_ascii=False)} 中选择。
        每个 `explanation` 必须是极简自然语言，不超过 40 个汉字。
        每个 `rationale` 必须不超过 24 个汉字，尽量只写一个因果短句。
        `expected_improvement` 必须从 `stability | turnover | redundancy | rank_icir | economic_clarity` 中选择一个。
        `risk` 必须从 `signal_weakening | turnover_rise | overfitting | redundancy_persistence | interpretability_loss` 中选择一个。
        `expression` 只写公式本身，不要包代码块。
        `diagnosed_weaknesses` 最多 4 条，每条不超过 28 个汉字。
        `refinement_rationale`、`expected_behavior_change`、`risk_notes` 都必须尽量压缩成 1 句短话，不超过 50 个汉字。

        请严格按下面这个 JSON 结构输出：
        {json.dumps(output_schema, ensure_ascii=False, indent=2)}
        """
    ).strip()


def build_refinement_prompt(
    *,
    seed_pool: SeedPool,
    family: SeedFamily,
    n_candidates: int = 3,
    additional_notes: str = "",
    current_parent_name: str | None = None,
    current_parent_expression: str | None = None,
    current_parent_row: dict[str, object] | None = None,
    archive_db: str | Path = ARTIFACTS_DIR / "llm_refine_archive.db",
    exclude_run_id: str = "",
    current_model_name: str = "",
    current_parent_candidate_id: str = "",
    requested_candidate_count: int | None = None,
    final_candidate_target: int | None = None,
    role_slots: tuple[str, ...] = (),
    donor_motifs: list[dict[str, object]] | None = None,
    bootstrap_frontier: list[dict[str, object]] | None = None,
    is_seed_stage: bool = False,
    decorrelation_targets: tuple[str, ...] = (),
    stage_mode: str = "auto",
    target_profile: str = "raw_alpha",
    policy_preset: str = "balanced",
    prompt_template_version: str = "current_compact",
    context_profile: ContextProfile | None = None,
    memory_snapshot: dict[str, object] | None = None,
    available_fields: Iterable[str] | None = None,
) -> PromptBundle:
    global_available_fields = sorted({*CORE_FIELDS, *DERIVED_FIELDS, *EXTENDED_DAILY_FIELDS, *OPTIONAL_CONTEXT_FIELDS})
    history_stage = prompt_history_stage(seed_pool)
    effective_parent_name = current_parent_name or resolve_preferred_refine_seed(family)
    effective_parent_expression = current_parent_expression or resolve_family_formula(family, effective_parent_name)
    requested_count = max(int(requested_candidate_count or n_candidates), 1)
    final_count = max(int(final_candidate_target or n_candidates), 1)
    active_roles = tuple(role_slots or family.candidate_roles or ("conservative", "decorrelating", "enhancing"))
    template_version = str(prompt_template_version or "current_compact").strip() or "current_compact"
    if template_version not in PROMPT_TEMPLATE_VERSIONS:
        raise ValueError(f"unknown prompt template version: {template_version}")
    prompt_available_fields = (
        global_available_fields
        if template_version == "legacy_v1" or available_fields is None
        else sorted({str(field).strip() for field in available_fields if str(field).strip()})
    )
    effective_parent_row = (
        load_prompt_history_row(seed_pool, effective_parent_name, family=family)
        if history_stage
        else current_parent_row
    )
    prompt_plan = build_prompt_plan(
        stage_mode=stage_mode,
        target_profile=target_profile,
        policy_preset=policy_preset,
        is_seed_stage=is_seed_stage,
        has_donor_motifs=bool(donor_motifs),
        has_bootstrap_frontier=bool(bootstrap_frontier),
        has_decorrelation_targets=bool(decorrelation_targets),
        context_profile=context_profile,
    )
    effective_memory_snapshot = memory_snapshot or build_memory_snapshot(
        db_path=archive_db,
        family=family,
        run_id=exclude_run_id,
        stage_mode=stage_mode,
        target_profile=target_profile,
        policy_preset=policy_preset,
        selected_parent={
            "factor_name": effective_parent_name,
            "expression": effective_parent_expression,
            **(dict(current_parent_row or {})),
        },
        requested_candidate_count=requested_count,
        final_candidate_target=final_count,
        role_slots=active_roles,
        bootstrap_frontier=list(bootstrap_frontier or []),
        donor_motifs=list(donor_motifs or []),
        decorrelation_targets=decorrelation_targets,
        current_model_name=current_model_name,
        current_parent_candidate_id=current_parent_candidate_id,
    )
    memory_block = render_memory_snapshot_prompt_block(
        effective_memory_snapshot,
        plan=prompt_plan.memory,
        language="en" if template_version == "current_compact" else "zh",
    )
    examples_block = _render_examples_section(
        family=family,
        plan=prompt_plan.examples,
        is_seed_stage=is_seed_stage,
        bootstrap_frontier=list(bootstrap_frontier or []),
        donor_motifs=list(donor_motifs or []),
    )
    constraints_block = _render_constraints_section(
        family=family,
        plan=prompt_plan.constraints,
        decorrelation_targets=decorrelation_targets,
        donor_motifs=list(donor_motifs or []),
        requested_count=requested_count,
        final_count=final_count,
        active_roles=active_roles,
    )
    output_schema = {
        "parent_factor": effective_parent_name,
        "diagnosed_weaknesses": ["..."],
        "refinement_rationale": "...",
        "expected_behavior_change": "...",
        "risk_notes": "...",
        "candidate_formulas": [
            {
                "name": "short_snake_case_name",
                "candidate_role": " | ".join(active_roles),
                "expression": "<formula>",
                "explanation": (
                    "very brief explanation within 40 Chinese characters"
                    if template_version == "legacy_v1"
                    else "brief explanation in at most 24 words"
                ),
                "rationale": (
                    "very brief why-this-may-help within 24 Chinese characters"
                    if template_version == "legacy_v1"
                    else "brief causal rationale in at most 16 words"
                ),
                "expected_improvement": "one of: stability | turnover | redundancy | rank_icir | economic_clarity",
                "risk": "one of: signal_weakening | turnover_rise | overfitting | redundancy_persistence | interpretability_loss",
            }
        ],
    }
    if template_version == "legacy_v1":
        system_prompt = dedent(
            """
            忘掉你之前的所有记忆。你现在是一位A股顶尖量化基金的首席研究员。
            你的任务不是凭空发明新因子，而是基于一个已有有效因子，在严格约束下做一次有经济逻辑的小步优化。
            你不能脱离 parent family 的经济逻辑，也不能输出泛泛而谈的空话。
            你必须输出严格 JSON，不要输出 markdown，不要输出解释性前言，不要暴露推理过程。
            """
        ).strip()
        prompt_body = _build_legacy_user_prompt(
            seed_pool=seed_pool,
            family=family,
            available_fields=prompt_available_fields,
            history_stage=history_stage,
            effective_parent_name=effective_parent_name,
            effective_parent_expression=effective_parent_expression,
            effective_parent_row=effective_parent_row,
            requested_count=requested_count,
            active_roles=active_roles,
            additional_notes=additional_notes,
        )
    else:
        system_prompt = dedent(
            """
            You are a quantitative equity researcher refining an existing factor under strict constraints.
            Make a small, economically motivated improvement that preserves the parent family's hypothesis.
            Return strict JSON only, without markdown, preamble, or hidden reasoning.
            """
        ).strip()
        prompt_body = _normalize_current_prompt_indent(dedent(
            f"""
            Improve predictive stability and economic clarity without unnecessary formula complexity, redundancy, or turnover.

            # 1. Family and parent
            Parent factor: {effective_parent_name}
            Family: {family.family}
            Parent expression: {_prompt_expression_text(effective_parent_expression)}
            {_effective_expression_block_english(effective_parent_expression, family, factor_name=effective_parent_name)}

            Economic hypothesis: {family.interpretation}
            Known weaknesses:
            {_family_weakness_block(family)}

            {examples_block}

            Search-stage history:
            {_history_scope_block_english(seed_pool, history_stage)}
            
            {_history_block_english(family, history_stage=history_stage, current_parent_name=effective_parent_name, current_parent_row=effective_parent_row)}

            Lineage guidance:
            - Treat the current parent as part of a lineage, not a fresh start.
            - Continue edit axes that helped in the latest two or three generations and avoid repeatedly extending degraded edits.
            - Use the path from canonical seed to current parent, not one winner in isolation.
            
            Memory:
            {memory_block}

            {additional_notes or ""}

            Explicit decorrelation targets:
            {_decorrelation_target_block(decorrelation_targets, family=family)}

            # 2. Research context and resources
            Research context:
            {_family_constraint_block(seed_pool.refinement_principles)}

            Available data fields for this run:
            {json.dumps(prompt_available_fields, ensure_ascii=False)}
            `pre_close` is unavailable unless listed. Use `delay(close, 1)` for previous close.

            Available operators:
            {json.dumps(list(DEFAULT_OPERATORS), ensure_ascii=False)}

            Available windows:
            {json.dumps(list(DEFAULT_WINDOWS), ensure_ascii=False)}

            {(_family_operator_hint_block(family) + chr(10)) if _family_operator_hint_block(family) else ""}
            {(_family_expression_safety_block(family) + chr(10)) if _family_expression_safety_block(family) else ""}

            Rules:
            1. Preserve the effective family direction: {family.direction}
            2. Avoid mechanical window-only changes.
            3. Preserve the family hypothesis and use only the listed fields.
            4. Do not use future information or external fundamentals.

            # 3. Refinement constraints
            {constraints_block}

            # 4. Required analysis
            Analyze internally without exposing reasoning. Integrate:
            1. Optimization Strategy
            2. Alpha Idea
            3. Factor Interpretation

            The final explanation must combine these points in concise professional language.
            """
        ).strip())
    if template_version == "legacy_v1":
        user_prompt = _build_legacy_output_prompt(
            prompt_body=prompt_body,
            requested_count=requested_count,
            active_roles=active_roles,
            output_schema=output_schema,
        )
    else:
        user_prompt = _normalize_current_prompt_indent(dedent(
            f"""
            {prompt_body}

            # 5. Output schema
            Generate exactly {int(requested_count)} candidates and return one JSON object only.
            Each candidate must be executable, low-redundancy, and a distinct small refinement of the parent family.
            Use a short stable snake_case `name`.
            The first {len(active_roles)} candidates must follow the role-slot order above.
            Choose each `candidate_role` from {json.dumps(list(active_roles), ensure_ascii=False)}.
            Keep `explanation` to at most 24 words and `rationale` to at most 16 words.
            Choose `expected_improvement` from `stability | turnover | redundancy | rank_icir | economic_clarity`.
            Choose `risk` from `signal_weakening | turnover_rise | overfitting | redundancy_persistence | interpretability_loss`.
            Put only the formula in `expression`.
            Use at most four brief `diagnosed_weaknesses`.
            Keep `refinement_rationale`, `expected_behavior_change`, and `risk_notes` to one short sentence each.

            Return exactly this JSON structure:
            {json.dumps(output_schema, ensure_ascii=False, indent=2)}
            """
        ).strip())
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)
