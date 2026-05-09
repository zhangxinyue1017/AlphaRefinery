# llm_refined Weighted Upper Shadow Distribution Report

## 1. 报告目的
- 这份报告按 `weighted_upper_shadow_distribution` 到 `2026-04-02` 为止的**正式因子结果**来整理，不按某一次 run 的算法 phase 来写。
- 当前正式代码位置：
  - [weighted_upper_shadow_distribution_family.py](../../../../factors_store/factors/llm_refined/weighted_upper_shadow_distribution_family.py)

## 2. Family 基本信息
- Family: `weighted_upper_shadow_distribution`
- Canonical seed: `factor365.weighted_upper_shadow_frequency_40_hl10`
- Canonical direction:
  - `use_negative_sign`
- 这组 family 现在已经稳定收敛到三条研究主线：
  - `upper-body rejection × amount`
  - `turnover / relative-turnover confirmation`
  - `shadow geometry / volatility normalization`

## 3. 当前总判断
- 这组 family 已经从 seed-only family 长成了一个**正式可维护的专题 family**。
- 当前正式写入 `py` 的因子数量已经从 `17` 条增加到 **`34` 条**。
- 到这一步为止，最明确的结论是：
  - `upper-body rejection × amount` 仍然是最强结果线
  - `turnover-confirmation` 已经从“支点线”升级成了**第二主线**
  - `geometry / volatility normalization` 是有效旁支，但还不是 headline 主线

## 4. 当前最重要的正式结果

| 角色 | 因子 | 模型 | RankIC | RankICIR | Net Ann | Net Excess | Sharpe | Turn | 说明 |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 首次主线突破 | `llm_refined.amt_weighted_shadow_10` | `kimi-k2` | 0.1010 | 0.6137 | 3.8818 | 0.3640 | 4.2253 | 0.1626 | 这条第一次把 family 从“事件/权重”推进到 amount-weighted 主线 |
| 当前价格结构代表 | `llm_refined.upper_body_reject_amt_10` | `gpt-5.4` | 0.1055 | 0.6558 | 3.6788 | 0.3079 | 4.0431 | 0.1988 | 把上影定义收紧到 `high - max(open, close)`，语义最清楚 |
| 当前正式最强条目 | `llm_refined.shadow_pos_confirm_15` | `qwen3.5-plus` | 0.0966 | 0.7409 | 5.2492 | 0.6531 | 5.9298 | 0.1443 | dual-parent scheduler 新增；当前最亮 confirmation winner |
| 当前 best-node 代表 | `llm_refined.llmgen_shadow_vol_rel_15` | `deepseek-v3.1` | 0.1001 | 0.6743 | 4.1525 | 0.3700 | 4.7100 | 0.1661 | dual-parent scheduler 新增；当前顶层 `best_node / last_winner` |
| 平滑型强点 | `llm_refined.shadow_wm_half_life_40` | `qwen3.5-plus` | 0.0922 | 0.6423 | 3.9703 | 0.2919 | 4.8103 | 0.1201 | 更长窗口 `weighted_mean + half_life` 延伸出稳定强点 |
| 相对换手确认代表 | `llm_refined.shadow_amt_relturn_15` | `gpt-5.4` | 0.0918 | 0.7349 | 3.9450 | 0.2857 | 5.3678 | 0.1765 | 用 `rel_turnover` 做确认，是 turnover 主线里很像正式资产的一条 |

## 5. 正式落盘的全量因子清单
- 当前正式写入这个 family module 的因子总数：**`34`**
- 可以分成四波理解：
  - `first_wave`
  - `round2p5_amt`
  - `round2p5_vol`
  - `dual_parent_scheduler`
  - `target_conditioned_ab`

| 因子 | 来源 | 模型 | Role | RankIC | RankICIR | Net Ann | Net Excess | Sharpe | Turn |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| `llm_refined.volume_weighted_shadow_10` | `first_wave` | `deepseek-v3.1` | `research_winner` | 0.0656 | 0.3174 | 0.3647 | -0.6946 | 0.8445 | 0.2668 |
| `llm_refined.shadow_turnover_weighted` | `first_wave` | `qwen3.5-plus` | `research_keep` | 0.0755 | 0.3476 | 0.8131 | -0.5845 | 1.3290 | 0.1585 |
| `llm_refined.shadow_turnover_filtered` | `first_wave` | `qwen3.5-plus` | `research_winner` | 0.0522 | 0.3348 | 0.3291 | -0.6801 | 0.8790 | 0.3886 |
| `llm_refined.shadow_rank_threshold` | `first_wave` | `qwen3.5-plus` | `research_keep` | 0.0776 | 0.4269 | 0.6829 | -0.6337 | 1.2872 | 0.2320 |
| `llm_refined.amt_weighted_shadow_10` | `first_wave` | `kimi-k2` | `research_winner` | 0.1010 | 0.6137 | 3.8818 | 0.3640 | 4.2253 | 0.1626 |
| `llm_refined.vol_amt_shadow_15` | `first_wave` | `kimi-k2` | `research_keep` | 0.0938 | 0.6003 | 3.6005 | 0.1856 | 4.2785 | 0.1494 |
| `llm_refined.amt_shadow_turnover_confirm` | `first_wave` | `claude-sonnet-4-6` | `research_winner` | 0.0733 | 0.3525 | 0.7978 | -0.5934 | 1.3403 | 0.1625 |
| `llm_refined.upper_body_reject_amt_10` | `round2p5_amt` | `gpt-5.4` | `research_winner` | 0.1055 | 0.6558 | 3.6788 | 0.3079 | 4.0431 | 0.1988 |
| `llm_refined.shadow_length_weighted_10` | `round2p5_amt` | `deepseek-v3.1` | `research_winner` | 0.0977 | 0.5665 | 3.6262 | 0.2734 | 3.9357 | 0.1359 |
| `llm_refined.wm_half_life_shadow_10` | `round2p5_amt` | `qwen3.5-plus` | `research_winner` | 0.1000 | 0.6079 | 3.8455 | 0.3228 | 4.2127 | 0.1496 |
| `llm_refined.turnover_conf_shadow_ema10` | `round2p5_amt` | `claude-sonnet-4-6` | `research_winner` | 0.0755 | 0.3476 | 0.8131 | -0.5845 | 1.3290 | 0.1585 |
| `llm_refined.amt_turn_confirm_shadow_10` | `round2p5_amt` | `kimi-k2` | `research_winner` | 0.0965 | 0.5489 | 2.6448 | -0.0713 | 3.1725 | 0.1351 |
| `llm_refined.shadow_amt_turnover_confirm_15` | `round2p5_vol` | `gpt-5.4` | `research_winner` | 0.0881 | 0.5226 | 2.2458 | -0.2455 | 3.0539 | 0.1143 |
| `llm_refined.relative_shadow_length_15` | `round2p5_vol` | `deepseek-v3.1` | `research_winner` | 0.0955 | 0.5864 | 3.6287 | 0.3048 | 4.1612 | 0.1168 |
| `llm_refined.shadow_range_norm_15` | `round2p5_vol` | `qwen3.5-plus` | `research_winner` | 0.0955 | 0.5864 | 3.6287 | 0.3048 | 4.1612 | 0.1168 |
| `llm_refined.turnover_shadow_confirm_20` | `round2p5_vol` | `claude-sonnet-4-6` | `research_winner` | 0.0577 | 0.3011 | 0.6839 | -0.6370 | 1.2926 | 0.1360 |
| `llm_refined.amt_turn_confirm_15` | `round2p5_vol` | `kimi-k2` | `research_winner` | 0.0940 | 0.6852 | 3.8988 | 0.2873 | 4.9297 | 0.1832 |
| `llm_refined.shadow_turn_confirm_ema` | `dual_parent_scheduler` | `gpt-5.4` | `research_winner` | 0.1029 | 0.6889 | 3.5659 | 0.1655 | 4.2970 | 0.1978 |
| `llm_refined.vol_scaled_shadow_amt_15` | `dual_parent_scheduler` | `deepseek-v3.1` | `research_winner` | 0.0995 | 0.6107 | 3.7465 | 0.2589 | 4.1047 | 0.1179 |
| `llm_refined.shadow_decay_amt_20` | `dual_parent_scheduler` | `qwen3.5-plus` | `research_winner` | 0.1010 | 0.6195 | 3.2140 | 0.0766 | 3.7821 | 0.1256 |
| `llm_refined.decay_shadow_turn_15` | `dual_parent_scheduler` | `deepseek-v3.1` | `research_winner` | 0.0851 | 0.6908 | 3.1226 | 0.0567 | 4.8896 | 0.2265 |
| `llm_refined.shadow_pos_confirm_15` | `dual_parent_scheduler` | `qwen3.5-plus` | `research_winner` | 0.0966 | 0.7409 | 5.2492 | 0.6531 | 5.9298 | 0.1443 |
| `llm_refined.shadow_amt_relturn_15` | `dual_parent_scheduler` | `gpt-5.4` | `research_winner` | 0.0918 | 0.7349 | 3.9450 | 0.2857 | 5.3678 | 0.1765 |
| `llm_refined.llmgen_shadow_vol_rel_15` | `dual_parent_scheduler` | `deepseek-v3.1` | `research_winner` | 0.1001 | 0.6743 | 4.1525 | 0.3700 | 4.7100 | 0.1661 |
| `llm_refined.shadow_wm_half_life_40` | `dual_parent_scheduler` | `qwen3.5-plus` | `research_winner` | 0.0922 | 0.6423 | 3.9703 | 0.2919 | 4.8103 | 0.1201 |
| `llm_refined.llmgen_shadow_std40_15` | `path_eval_v2` | `qwen3.5-plus` | `research_winner` | 0.0831 | 0.6100 | 2.7938 | 0.1827 | 5.3338 | 0.1777 |
| `llm_refined.shadow_turnover_conf_15` | `target_conditioned_ab` | `gpt-5.4` | `research_winner` | 0.1023 | 0.6847 | 3.4314 | 0.1325 | 4.2054 | 0.1620 |
| `llm_refined.shadow_rel_amount_confirm_15` | `target_conditioned_ab` | `deepseek-v3.1` | `research_winner` | 0.1012 | 0.7142 | 3.8609 | 0.2259 | 4.6621 | 0.1510 |
| `llm_refined.shadow_amt_ema_15_th0015` | `target_conditioned_ab` | `qwen3.5-plus` | `research_winner` | 0.1019 | 0.6287 | 3.1819 | 0.0240 | 3.8005 | 0.1374 |
| `llm_refined.shadow_count_amt_ema_15` | `target_conditioned_ab` | `gpt-5.4` | `research_winner` | 0.1010 | 0.6075 | 3.5225 | 0.2916 | 3.8570 | 0.1548 |
| `llm_refined.llmgen_shadow_turnover_conf_15` | `target_conditioned_ab` | `deepseek-v3.1` | `research_winner` | 0.1023 | 0.6847 | 3.4314 | 0.1325 | 4.2054 | 0.1620 |
| `llm_refined.shadow_wma_smooth_20` | `target_conditioned_ab` | `qwen3.5-plus` | `research_winner` | 0.1020 | 0.6246 | 3.3057 | 0.1018 | 3.8375 | 0.1407 |

## 6. 新增的 dual-parent scheduler 这 8 条怎么读
- 这 8 条里最重要的不是“又多了几个名字”，而是：
  - `turnover-confirmation` 这条线明显继续变强了
  - 而且开始分化出：
    - `position confirm`
    - `relative turnover confirm`
    - `relative amount confirm`
    - `weighted_mean / half-life`
    - `decay_linear`
- 这说明 dual-parent continuation 不是在原地打转，而是在 confirmation 主线内部继续长专题分支。

### 第一梯队
- `llm_refined.shadow_pos_confirm_15`
- `llm_refined.llmgen_shadow_vol_rel_15`
- `llm_refined.shadow_wm_half_life_40`
- `llm_refined.shadow_amt_relturn_15`

### 第二梯队
- `llm_refined.vol_scaled_shadow_amt_15`
- `llm_refined.shadow_turn_confirm_ema`

### 第三梯队
- `llm_refined.shadow_decay_amt_20`
- `llm_refined.decay_shadow_turn_15`

## 7. 这组 family 目前收敛到了哪几条主线

### 主线 1：`upper-body rejection × amount`
- 当前代表：
  - `llm_refined.amt_weighted_shadow_10`
  - `llm_refined.upper_body_reject_amt_10`
  - `llm_refined.shadow_length_weighted_10`
  - `llm_refined.wm_half_life_shadow_10`
  - `llm_refined.shadow_decay_amt_20`
- 这条线仍然是价格结构定义最清楚、解释最直观的主结果线。

### 主线 2：`turnover / relative-turnover confirmation`
- 当前代表：
  - `llm_refined.amt_turn_confirm_15`
  - `llm_refined.shadow_pos_confirm_15`
  - `llm_refined.shadow_amt_relturn_15`
  - `llm_refined.shadow_turn_confirm_ema`
  - `llm_refined.shadow_wm_half_life_40`
  - `llm_refined.llmgen_shadow_vol_rel_15`
- 这条线现在已经从“支点线”升级成第二主线，而且从结果看，风险调整后已经很强。

## 8.1 Target-Conditioned Search 这 6 条新补充怎么读
- 这 6 条来自 `raw_alpha vs complementarity` 的首轮 A/B smoke，意义不是“又多了 6 个名字”，而是：
  - 现在搜索目标已经开始影响 child 的展开方向
  - `complementarity` 不再只是口号，已经开始推向不同的结构表达
- 这批新增可以粗分成三类：
  - `turnover-confirm` 强化：
    - `shadow_turnover_conf_15`
    - `llmgen_shadow_turnover_conf_15`
  - `amount / rel_amount` 状态确认：
    - `shadow_rel_amount_confirm_15`
    - `shadow_count_amt_ema_15`
  - `same-parent 结构微分化`：
    - `shadow_amt_ema_15_th0015`
    - `shadow_wma_smooth_20`
- 从结果看：
  - `shadow_rel_amount_confirm_15` 和 `shadow_count_amt_ema_15` 更像值得继续观察的 confirmation 支线
  - `shadow_wma_smooth_20` 是这波里最能说明 `complementarity` 已经在起作用的一条
  - `shadow_amt_ema_15_th0015` 更像对已有主线的 exploitation 微调

### 主线 3：`geometry / volatility normalization`
- 当前代表：
  - `llm_refined.relative_shadow_length_15`
  - `llm_refined.shadow_range_norm_15`
  - `llm_refined.vol_scaled_shadow_amt_15`
- 这条线更像跨 regime 稳定性的结构化旁支。

## 9. 当前最值得继续 formal / admission 关注的因子

### 第一梯队
- `llm_refined.shadow_pos_confirm_15`
- `llm_refined.llmgen_shadow_vol_rel_15`
- `llm_refined.upper_body_reject_amt_10`
- `llm_refined.shadow_wm_half_life_40`
- `llm_refined.shadow_amt_relturn_15`
- `llm_refined.amt_turn_confirm_15`
- `llm_refined.shadow_rel_amount_confirm_15`
- `llm_refined.shadow_count_amt_ema_15`
- `llm_refined.shadow_wma_smooth_20`

### 第二梯队
- `llm_refined.vol_scaled_shadow_amt_15`
- `llm_refined.relative_shadow_length_15`
- `llm_refined.shadow_range_norm_15`
- `llm_refined.wm_half_life_shadow_10`
- `llm_refined.shadow_turnover_conf_15`
- `llm_refined.llmgen_shadow_turnover_conf_15`
- `llm_refined.shadow_amt_ema_15_th0015`

## 10. 当前 admission 口径下的状态
- 到这份报告为止，这组 family 的正式 `py` 因子数已经是 **`34`**。
- 已完成的一轮 full-family bridge ingest 结果是：
  - 正式 family 因子 `28`
  - 实际 promoted / inserted into library：**`6`**
- 当前已经入库的 6 条是：
  - `llm_refined.shadow_turnover_weighted`
  - `llm_refined.shadow_rank_threshold`
  - `llm_refined.amt_weighted_shadow_10`
  - `llm_refined.shadow_amt_turnover_confirm_15`
  - `llm_refined.shadow_turn_confirm_ema`
  - `llm_refined.shadow_vol_scaled_15`
- 今天新补进去的 6 条 `target_conditioned_ab` 因子，已经开始做单独 bridge admission，目的是看：
  - 这批新增能否再补出新的入库代表
  - 还是主要会因 family / library corr 被拦

## 11. 当前最真实的风险
- 这组 family 已经开始出现明显的同族近重复：
  - `relative_shadow_length_15` 和 `shadow_range_norm_15`
  - `turnover_conf_shadow_ema10` 和 `shadow_turn_confirm_ema`
  - `wm_half_life_shadow_10` 和 `shadow_wm_half_life_40`
  - `shadow_turnover_conf_15` 和 `llmgen_shadow_turnover_conf_15`
  - `shadow_amt_ema_15` 和 `shadow_amt_ema_15_th0015`
- 这不说明 family 差，反而说明：
  - 主线已经开始收敛
  - 下一步重点应该是 `dedup / clustering / admission-aware filtering`

## 12. 当前建议分层

### 核心保留
- `llm_refined.amt_weighted_shadow_10`
- `llm_refined.upper_body_reject_amt_10`
- `llm_refined.shadow_pos_confirm_15`
- `llm_refined.llmgen_shadow_vol_rel_15`
- `llm_refined.amt_turn_confirm_15`
- `llm_refined.shadow_wm_half_life_40`
- `llm_refined.shadow_amt_relturn_15`

### 观察候选
- `llm_refined.vol_amt_shadow_15`
- `llm_refined.vol_scaled_shadow_amt_15`
- `llm_refined.relative_shadow_length_15`
- `llm_refined.shadow_range_norm_15`
- `llm_refined.shadow_length_weighted_10`
- `llm_refined.shadow_turnover_conf_15`
- `llm_refined.llmgen_shadow_turnover_conf_15`
- `llm_refined.shadow_amt_ema_15_th0015`

### 近重复 / 后续应做 dedup
- `llm_refined.turnover_conf_shadow_ema10`
- `llm_refined.shadow_turn_confirm_ema`
- `llm_refined.relative_shadow_length_15`
- `llm_refined.shadow_range_norm_15`
- `llm_refined.wm_half_life_shadow_10`
- `llm_refined.shadow_wm_half_life_40`
- `llm_refined.shadow_turnover_conf_15`
- `llm_refined.llmgen_shadow_turnover_conf_15`
- `llm_refined.shadow_amt_ema_15`
- `llm_refined.shadow_amt_ema_15_th0015`

## 13. 一句话结论
- `weighted_upper_shadow_distribution` 现在已经不只是“有一批 formalized 因子”，而是已经形成了一个 **34 条正式因子、双主线明确、confirmation 主题明显变强** 的 family。
- 当前最强结果线是：
  - `upper-body rejection × amount`
- 当前最强扩展线已经升级为：
  - `turnover / relative-turnover confirmation`
- 下一步重点不该是再盲目加量，而应该是：
  - `admission shortlist + 同族近重复整理 + target-conditioned continuation`
