'''Prompt assembly logic for LLM refinement candidates.

Combines family metadata, memory, constraints, examples, roles, and output schema instructions.
'''

from __future__ import annotations

import json
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
from ..knowledge.retrieval import build_family_memory_payload, render_family_memory_block
from ..parsing.operator_contract import PROMPT_OPERATOR_DESCRIPTIONS
from .prompt_plan import PromptConstraintPlan, PromptExamplesPlan, PromptMemoryPlan, PromptPlan, build_prompt_plan
from ..search.context_resolver import ContextProfile

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
        去相关试验要求：
        - 这是一轮显式去相关试验。请优先减少与上面 target set 的同构和高相关，而不是只做常规局部优化。
        - 更推荐的改法：
          1. denominator / anchor swap：换分母、换锚点、换 reference price，而不是只改窗口。
          2. transform swap：把简单 ratio 改成 log-ratio、distance、zscore distance、signed deviation。
          3. conditional gate：加入 turnover / volatility / range / quantile regime 条件门控。
          4. orthogonal side signal：加入一个不同来源的辅助轴，但不要破坏当前 family 的核心经济逻辑。
        - 明确避免以下伪去相关：
          1. 只做 smooth / decay / ema 近邻替换。
          2. 只做窗口微调。
          3. 只做 mean / ema / close 之间的近邻互换。
          4. 只加一个弱归一化项却保持主 skeleton 不变。
        - 你的目标不是离 parent 越远越好，而是在保持质量的前提下，优先避开 target set 已经覆盖的结构。
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
        该 family 的实现提示：
        - 现在可以直接使用 `bucket_sum(x, key, window, q, side)` 做严格的窗口内分桶聚合；`side` 只能是 `"low"` 或 `"high"`。
        - 现在可以直接使用 `where(cond, x, y)` / `if_then_else(cond, x, y)` 构造条件分支。
        - 现在可以直接使用 `count(cond, window)`、`sumif(x, window, cond)`、`gt/lt/ge/le` 做条件计数与条件求和。
        - 可直接构造的 key 例子：
          `amplitude = sub(div(high, add(low, 1e-12)), 1)`
          `upper_shadow = sub(high, rowmax(open, close))`
          `lower_shadow = sub(rowmin(open, close), low)`
          `intraday_position = div(sub(close, low), add(sub(high, low), 1e-12))`
        - 如果要复刻 amplitude-sliced return，可以优先考虑：
          `bucket_sum(delay(returns, 1), delay(sub(div(high, add(low, 1e-12)), 1), 1), 60, 0.3, "low")`
        - 请优先用这些新原语保留 sliced / state-conditioned 结构，而不是退化成普通 `mean/std/rank` 变体。
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
        该 family 的表达式安全提示：
        - 不要直接输出已注册因子名或内部 helper 名，例如：
          `qp_pressure.net_pressure_20`、`qp_pressure.low_price_volume_share_20`、
          `low_price_volume_share(...)`、`high_price_volume_share(...)`、
          `buy_pressure(...)`、`sell_pressure(...)`、
          `share_of_volume_traded_on_low_price_days(...)`。
        - 上面这些名称只代表 family 逻辑标签，不是当前表达式引擎可直接调用的 token。
        - 请用当前 contract 的基础原语重写，例如：
          `if_then_else` + `ts_quantile` + `ts_sum` + `div` + `add` + `mul`。
        - 例如“低价日成交占比”应写成类似：
          `div(ts_sum(if_then_else(le(close, ts_quantile(close, 20, 0.3)), volume, 0), 20), add(ts_sum(volume, 20), 1e-12))`
        - `ts_quantile` 第三个参数请用位置参数 `0.3/0.7`，不要写成 `q=0.3`。
        - 不要输出带点号的属性式引用，不要把 peer factor 名直接当成 expression。
        """
    ).strip()


def _global_expression_engine_safety_block() -> str:
    return dedent(
        """
        当前表达式引擎语法限制：
        - 除 `half_life` 和 `min_obs` 外，不要使用任何 keyword argument。
        - 也就是说，不要写 `q=0.3`、`m=1`、`side="low"`、`window=20`、`cond=...`。
        - 请统一使用位置参数，例如：
          `ts_quantile(close, 20, 0.3)`
          `sma(x, 20, 1)`
          `bucket_sum(x, key, 60, 0.3, "low")`
        - 不要输出带点号的引用，不要写 `family.factor_name`、`module.helper(...)`、`qp_pressure.xxx` 这类 dotted reference。
        - 不要发明新的 helper 名、标签名、宏名；只能使用 contract 中已有字段和算子。
        - 如果一个想法需要 helper function 才能表达，请改写成基础原语组合，而不是直接输出 helper 名。
        """
    ).strip()


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
                "相似/同家族因子：",
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


def _render_memory_section(payload: dict[str, object], *, plan: PromptMemoryPlan) -> str:
    if not plan.include:
        return ""
    return render_family_memory_block(
        payload,
        max_winners=plan.max_winners,
        max_keeps=plan.max_keeps,
        max_failures=plan.max_failures,
        include_lineage=plan.include_lineage,
        include_reflection=plan.include_reflection,
    )


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
            "本轮优化优先级：",
            f"- 主目标：{family.primary_objective or 'improve_rank_icir'}",
            f"- 次目标：{family.secondary_objective or 'maintain_family_logic_while_lowering_redundancy'}",
        ]
    )
    lines.extend(["", "主线要求："])
    if plan.include_family_constraints:
        constraints_text = _family_constraint_block(family.hard_constraints)
        if constraints_text != "- (none)":
            lines.append(constraints_text)
    lines.extend(
        [
            "- 不要脱离当前 family 的核心经济逻辑，不要彻底换题。",
            "- 优先修 parent 当前最明显的弱点，不要同时解决太多问题。",
            "- 如果 parent 是两段式或 hybrid 结构，先调整方向、权重或平滑方式；不要默认删掉其中一段。",
        ]
    )

    if plan.include_axes_guidance:
        lines.extend(
            [
                "",
                "优先考虑以下改进方向：",
                _family_axes_block(family),
            ]
        )

    lines.extend(
        [
            "",
            "改写幅度：",
            "- 可以做方向修正、平滑、归一化、替换一个确认项。",
            "- 不要把整个主结构重写成另一类因子。",
            "- 改动要轻，不要无意义增加嵌套、字段和条件分支。",
            "",
            "候选分工：",
            "- 三个候选不要做成同一种改法。",
            "- 一个偏稳健，一个偏去重，一个偏增强。",
            "- 不要求每个候选同时兼顾稳健、去重和增强。",
            "- 不要输出和 parent / peer 只有表面差别的公式。",
            "",
            f"你必须生成 {int(requested_count)} 个候选，并优先保证前 {len(active_roles)} 个候选按下面这些 role slots 覆盖：",
            _candidate_role_block(active_roles, len(active_roles)),
        ]
    )

    if plan.include_allowed_edit_types:
        lines.extend(["", "允许的 edit 类型：", _allowed_edit_block(family)])

    lines.extend(["", _global_expression_engine_safety_block()])

    lines.extend(
        [
            "",
            "同时兼顾：",
            "- 提高稳定性",
            "- 降低换手",
            "- 降低家族内冗余",
            "- 保持经济解释清晰",
        ]
    )
    if donor_motifs:
        lines.append("- 至少 1 个候选要明确借用 donor motif，但必须翻译到当前 family 语义里")
    if requested_count > final_count:
        lines.append(
            f"- 系统会先收集更宽的 seed-stage 候选，再轻量 rerank 到最终保留的 {int(final_count)} 条；请优先保证 role 覆盖和结构差异"
        )

    if plan.include_anti_patterns and family.anti_patterns:
        lines.extend(["", "禁止项：", _family_constraint_block(family.anti_patterns)])
    else:
        lines.extend(
            [
                "",
                "禁止项：",
                "- 不要只改名字或只换窗口。",
                "- 不要引入未来信息、外部财务字段、或当前 contract 中没有的字段。",
                "- 不要脱离当前 family 的经济解释。",
                "- 不要把 parent 或 peer 只做表面重命名后重复输出。",
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


def _effective_expression_block(expression: str, family: SeedFamily, *, factor_name: str = "") -> str:
    if not expression:
        return ""
    adjusted = apply_direction_rule(expression, resolve_factor_direction(family, factor_name or family.canonical_seed))
    if adjusted == expression:
        return ""
    if resolve_factor_direction(family, factor_name or family.canonical_seed) in {"use_negative_sign", "use_positive_sign"}:
        return f"当前 parent 有效方向表达式：{_prompt_expression_text(adjusted)}"
    return ""


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
) -> PromptBundle:
    available_fields = sorted({*CORE_FIELDS, *DERIVED_FIELDS, *EXTENDED_DAILY_FIELDS, *OPTIONAL_CONTEXT_FIELDS})
    history_stage = prompt_history_stage(seed_pool)
    effective_parent_name = current_parent_name or resolve_preferred_refine_seed(family)
    effective_parent_expression = current_parent_expression or resolve_family_formula(family, effective_parent_name)
    requested_count = max(int(requested_candidate_count or n_candidates), 1)
    final_count = max(int(final_candidate_target or n_candidates), 1)
    active_roles = tuple(role_slots or family.candidate_roles or ("conservative", "decorrelating", "enhancing"))
    template_version = str(prompt_template_version or "current_compact").strip() or "current_compact"
    if template_version not in PROMPT_TEMPLATE_VERSIONS:
        raise ValueError(f"unknown prompt template version: {template_version}")
    effective_parent_row = (
        load_prompt_history_row(seed_pool, effective_parent_name, family=family)
        if history_stage
        else current_parent_row
    )
    memory_payload = build_family_memory_payload(
        db_path=archive_db,
        family=family,
        exclude_run_id=exclude_run_id,
        current_model_name=current_model_name,
        current_parent_candidate_id=current_parent_candidate_id,
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
    memory_block = _render_memory_section(memory_payload, plan=prompt_plan.memory)
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
                "explanation": "very brief explanation within 40 Chinese characters",
                "rationale": "very brief why-this-may-help within 24 Chinese characters",
                "expected_improvement": "one of: stability | turnover | redundancy | rank_icir | economic_clarity",
                "risk": "one of: signal_weakening | turnover_rise | overfitting | redundancy_persistence | interpretability_loss",
            }
        ],
    }
    system_prompt = dedent(
        """
        忘掉你之前的所有记忆。你现在是一位A股顶尖量化基金的首席研究员。
        你的任务不是凭空发明新因子，而是基于一个已有有效因子，在严格约束下做一次有经济逻辑的小步优化。
        你不能脱离 parent family 的经济逻辑，也不能输出泛泛而谈的空话。
        你必须输出严格 JSON，不要输出 markdown，不要输出解释性前言，不要暴露推理过程。
        """
    ).strip()
    if template_version == "legacy_v1":
        prompt_body = _build_legacy_user_prompt(
            seed_pool=seed_pool,
            family=family,
            available_fields=available_fields,
            history_stage=history_stage,
            effective_parent_name=effective_parent_name,
            effective_parent_expression=effective_parent_expression,
            effective_parent_row=effective_parent_row,
            requested_count=requested_count,
            active_roles=active_roles,
            additional_notes=additional_notes,
        )
    else:
        prompt_body = dedent(
            f"""
            你的目标是：在不明显增加公式复杂度的前提下，尽量提升因子的 RankIC、RankICIR、Sharpe 和 excess return，并尽量改善稳定性、降低冗余、降低不必要的高换手。

            # 1. 现有因子信息
            当前 parent 因子名称：{effective_parent_name}
            因子家族：{family.family}
            当前 parent 因子原始表达式：{_prompt_expression_text(effective_parent_expression)}
            {_effective_expression_block(effective_parent_expression, family, factor_name=effective_parent_name)}

            经济假设：{family.interpretation}
            已知弱点：
            {_family_weakness_block(family)}

            {examples_block}

            历史表现：
            {_history_scope_block(seed_pool, history_stage)}
            
            {_history_block(family, history_stage=history_stage, current_parent_name=effective_parent_name, current_parent_row=effective_parent_row)}

            改进轨迹提示：
            - 请结合当前 parent 所在的 lineage，不要把每一轮都当成从零开始。
            - 优先延续最近 2~3 代里被证明有效的 edit axis；如果某类改写在 lineage 中退化，请避免继续沿同一路径机械外推。
            - canonical seed 到当前 parent 的链路，比单个 winner 更重要。
            
            最近经验记忆：
            {memory_block}

            {additional_notes or ""}

            显式去相关 target set：
            {_decorrelation_target_block(decorrelation_targets, family=family)}

            # 2. 可用资源
            可用字段：
            {json.dumps(available_fields, ensure_ascii=False)}

            可用算子：
            {json.dumps(list(DEFAULT_OPERATORS), ensure_ascii=False)}

            可用时间窗：
            {json.dumps(list(DEFAULT_WINDOWS), ensure_ascii=False)}

            {(_family_operator_hint_block(family) + chr(10)) if _family_operator_hint_block(family) else ""}
            {(_family_expression_safety_block(family) + chr(10)) if _family_expression_safety_block(family) else ""}

            注意：
            1. 原因子默认应继承当前 family 的有效符号方向：{family.direction}
            2. 不要只是机械调窗口。
            3. 必须保留当前 family 的核心经济逻辑，不要彻底换思路。
            4. 不要引入未来信息、外部财务字段、或当前 contract 中没有的字段。

            # 3. 优化要求
            {constraints_block}

            # 4. 分析要求
            请先在内部完成分析，但不要展示推理过程。必须综合以下三个维度：
            1. Optimization Strategy
            2. Alpha Idea
            3. Factor Interpretation

            你最终输出的 explanation 必须把这三部分信息融合成自然语言，但要极短、直白、能让研究员快速理解。
            """
        ).strip()
    user_prompt = dedent(
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
    return PromptBundle(system_prompt=system_prompt, user_prompt=user_prompt)
