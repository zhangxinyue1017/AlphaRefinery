# SearchPolicy 调参说明

## 先看什么

`SearchPolicy` 的运行时 dataclass 仍在 `search/core/policy.py`，但默认
权重、preset、target profile overlay、search mode overlay 已集中到
`search/policy_config.py` 的 `DEFAULT_POLICY_CONFIG.search`。

以后调权重优先改 `policy_config.py`，不要在 `core/policy.py` 里散写新数值。
`core/policy.py` 主要保留 typed runtime object 和构造逻辑。

真正常调的，主要是下面这些。

## 1. 探索强度

### `exploration_weight`

- 控制 UCB-lite 探索强度
- 越大越愿意给访问少的节点预算

### `novelty_bonus_weight`

- 给未扩展新节点奖励
- 越大越鼓励系统去试没扩过的点

### `motif_novelty_weight`

- 给 motif 稀缺性奖励
- 越大越不容易让同一类结构长期霸榜

## 2. branch / parent 价值

### `expandability_weight`

- 给 `effective_expandability_score` 的权重
- 更偏“这条 parent 历史上会不会生出好 child”

### `branch_value_weight`

- 给 `branch_value_score` 的权重
- 更偏“这条 branch 继续投预算值不值”
- 这是 `Path Evaluation v2` 的新主权重

## 3. 质量和风险平衡

### `turnover_penalty_weight`

- 高换手惩罚

### `complexity_penalty_weight`

- 复杂表达式惩罚

### `depth_penalty_weight`

- 深链路惩罚

### `redundancy_penalty_weight`

- motif 重复惩罚

### `family_motif_saturation_weight`

- family 内已被反复探索的 profile 再降权

### `correlation_redundancy_weight`

- 对 archive 历史高 corr 风险做轻量惩罚
- 这不是实时相关性矩阵

## 4. frontier 形状

### `branch_frontier_cap`

- 同一 branch 允许多少个节点同时留在 frontier

### `motif_frontier_cap`

- 同一 motif 允许多少个节点同时留在 frontier

### `selection_pool_size`

- parent selection 前先看多少个 top 节点

### `mmr_candidate_pool_size`

- MMR rerank 候选池大小

### `mmr_lambda`

- relevance 和 diversity 的平衡
- 越大越看原始分数
- 越小越看“别太像”

## 5. dual-parent 参数

### `dual_parent_enabled`

- 是否允许一轮内选两个 parent

### `dual_parent_max_parents`

- 当前 v1 基本就是 `2`

### `dual_parent_delta_threshold`

- top1 和候选 secondary 的分差阈值

### `dual_parent_similarity_threshold`

- primary / secondary 相似度阈值

### `dual_parent_min_expandability_advantage`

- secondary 至少要多出的 expandability / branch value 优势

## 6. 三个 preset

### `balanced`

- 默认
- 适合大多数正常 refine

### `exploratory`

- 更偏发散
- 更适合新 family / family 宽搜

### `conservative`

- 更偏质量和惩罚
- 更适合主线明确后的稳健深挖

## 7. target profile / mode overlay

`with_target_profile()` 现在从 `DEFAULT_POLICY_CONFIG.search.target_profiles`
读取 overlay。

- `raw_alpha`: 偏原始 alpha 质量，同时保留适度 constraint / portfolio 权重
- `deployability`: 提高 constraint 权重
- `complementarity`: 提高 portfolio / decorrelation 相关权重和 gate penalty
- `robustness`: 提高 regime 权重

`for_mode()` 现在从 `DEFAULT_POLICY_CONFIG.search.modes` 读取 overlay。

- `multi_model_best_first`: 保证多模型 best-first 的基础 frontier 宽度
- `family_breadth_first`: 增强探索、novelty，并限制单 branch 过度占用
- `local_best_first`: 降低探索，偏局部深挖

## 8. 实用调参建议

### 新 family 首跑

- 提高：
  - `exploration_weight`
  - `novelty_bonus_weight`
  - `motif_novelty_weight`
- 适度提高：
  - `expandability_weight`
  - `branch_value_weight`

### 主线明确后深挖

- 降低：
  - `exploration_weight`
  - `novelty_bonus_weight`
- 提高：
  - `turnover_penalty_weight`
  - `redundancy_penalty_weight`

### family 太拥挤

- 提高：
  - `family_motif_saturation_weight`
  - `correlation_redundancy_weight`
- 收紧：
  - `branch_frontier_cap`
  - `motif_frontier_cap`

### 想更容易触发 dual-parent

- 提高：
  - `dual_parent_delta_threshold`
  - `dual_parent_similarity_threshold`
- 降低：
  - `dual_parent_min_expandability_advantage`

### 想让 dual-parent 更保守

- 反过来调：
  - 降低 `dual_parent_delta_threshold`
  - 降低 `dual_parent_similarity_threshold`
  - 提高 `dual_parent_min_expandability_advantage`
