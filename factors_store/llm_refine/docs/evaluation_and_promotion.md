# Evaluation / Research Gate / Promotion

## research_keep

`research_keep` 是 broad gate。  
它回答的是：

- 这条 candidate 相对 parent 有没有研究价值

当前大意是：

- 先排除明显不合格项：
  - evaluation failed
  - same expression as parent
  - 被 redundancy 直接打成 `drop_redundant_*`
- 再看 candidate 相对 parent 在主指标上是否有改进
- 同时检查 turnover 不要恶化得太离谱

## research_winner

`research_winner` 不是单一 `RankICIR-first`。

当前是：

- 先从 broad gate 得到 `research_keep`
- 再在 keep 候选里按综合 winner score 选一条 winner

综合会看：

- `quick_rank_ic_mean`
- `quick_rank_icir`
- `net_ann_return`
- `net_excess_ann_return`
- `net_sharpe`
- `mean_turnover`

## NA-heavy 收紧

从最近版本开始，`NA-heavy keep` 已经明显收紧：

- `missing_core_metrics_count > 1`
  - 不进入 `research_keep`
- 如果同时缺：
  - `net_sharpe`
  - `mean_turnover`
  - 也不进入 `research_keep`

## best_node 完整性

除了 `research_keep` 之外，搜索层还会看：

- `metrics_completeness`
- `eligible_for_best_node`

所以现在：

- 一条 candidate 也许还能算研究上有意思
- 但如果 full metrics 缺太多
- 不应该轻易变成 `best_node`

## corr 的两层语义

### evaluation / admission 里的 corr

- 是真实因子序列相关性
- 用来做：
  - `drop_redundant_parent`
  - `drop_redundant_family`

### search 里的 corr 风险

- 是历史风险先验
- 用来做轻量 `corr_redundancy_penalty`
- 不是实时重算相关性矩阵

## promotion

evaluation 后，满足条件的 `research_winner` 会进入 pending promotion。  
最近版本也允许**少量高质量 `research_keep`** 一起进入 pending promotion。

当前宽口径大意是：

- `research_winner`
  - 默认进入 pending promotion
- `research_keep`
  - 只放少量
  - 需要 full metrics 完整
  - 不能缺 core metrics
  - turnover 不能太坏
  - 且需要有足够高的 `winner_score` 或基本质量指标

开：

- `--auto-apply-promotion`

时，会自动写入：

- `factors_store/factors/llm_refined/*.py`
- `llm_refined/__init__.py`

## auto_apply_promotion 的口径

它是：

- 研究层 winner 自动 formalize
- 以及少量高质量 keep 自动 formalize

不是：

- 最终 admission
- 最终库内去重

所以 `auto_apply_promotion` 可以比较宽，  
formal admission 仍然应该单独看。
