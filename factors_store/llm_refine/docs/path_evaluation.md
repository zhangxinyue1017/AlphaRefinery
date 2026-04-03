# Path Evaluation / Branch Value

## 为什么需要它

搜索里经常会出现两个不同的问题：

- 当前最强结果是谁
- 最值得继续扩展的 branch 是谁

这两者不一定一样。

所以 `Path Evaluation` 的目标，是给 branch 本身打分，而不只给单个 candidate 打分。

## v1: expandability

最早的 `expandability` 主要看：

- `children_kept`
- `children_winners`
- `child_best_gain`
- `child_model_support`
- `child_mutation_diversity`

然后得到：

- `expandability_score`
- `expandability_confidence`
- `effective_expandability_score`

它回答的是：

- 这条 parent 历史上会不会生出好 child

## v2: branch value

`Path Evaluation v2` 进一步回答：

- 这条 branch 继续投预算值不值

当前核心字段：

- `child_top3_mean_gain`
- `child_median_gain`
- `child_positive_gain_rate`
- `child_gain_std`
- `child_model_support_rate`
- `child_cross_model_convergence`
- `child_positive_excess_rate`
- `child_low_turnover_rate`
- `child_full_metrics_rate`
- `child_admission_friendly_rate`
- `child_high_quality_novel_count`
- `child_new_motif_success_rate`
- `last_success_round`
- `rounds_since_last_success`
- `branch_value_score`

## 这些字段大致代表什么

### 后代质量

- `child_top3_mean_gain`
  - top3 child 的平均升级幅度
- `child_median_gain`
  - 典型 child 的升级幅度
- `child_positive_gain_rate`
  - 有多少 child 综合上优于 parent

### 后代稳定性

- `child_gain_std`
  - gain 分布是否太飘
- `child_model_support_rate`
  - 有多少模型能支持这条线
- `child_cross_model_convergence`
  - 多模型是否收敛到同类结构

### admission 友好度

- `child_positive_excess_rate`
  - 正 excess 后代比例
- `child_low_turnover_rate`
  - 低换手后代比例
- `child_full_metrics_rate`
  - full metrics 完整后代比例
- `child_admission_friendly_rate`
  - 同时满足几项 formal 友好条件的比例

### 新意质量

- `child_high_quality_novel_count`
  - 新且好的 child 数量
- `child_new_motif_success_rate`
  - 跳出 parent motif 后仍能成功的比例

### 时效性

- `last_success_round`
- `rounds_since_last_success`

避免老 branch 永远吃历史功劳。

## 最终怎么用

`branch_value_score` 会作为：

- frontier bonus
- dual-parent secondary 选择信号

所以它不只是展示字段，而是真正开始影响 parent budget allocation。

## 它和 expandability 的区别

### `expandability_score`

- 更偏：
  - 会不会生孩子
  - 历史上是否产出过 winner / keep

### `branch_value_score`

- 更偏：
  - 后代整体质量
  - 稳定性
  - admission 友好度
  - 新意
  - 时效性

实践上可以理解成：

- `expandability`
  - “这条 parent 会不会长”
- `branch_value`
  - “这条 branch 值不值得继续投预算”
