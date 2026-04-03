# Search / Frontier / Dual-Parent

## SearchEngine 在做什么

统一 `SearchEngine` 负责：

- 维护 `SearchNode / SearchEdge / SearchFrontier`
- 对 frontier 上的节点打分
- 选择下一轮 parent
- 回收 child 结果

## frontier score 现在看什么

当前 `frontier_score` 主要由这些部分组成：

- `quality_score`
- `novelty_score`
- `underexplored_bonus`
- `expandability_bonus`
- `branch_value_bonus`
- `redundancy_penalty`
- `complexity_penalty`
- `completeness_penalty`

直观理解：

- `quality_score`
  - 节点本身质量
- `novelty_score`
  - 新颖性
- `underexplored_bonus`
  - 未扩展节点奖励
- `expandability_bonus`
  - 历史上会不会生出好 child
- `branch_value_bonus`
  - 这条 branch 继续投预算值不值
- `redundancy_penalty`
  - 太像、太拥挤、历史高 corr 风险
- `complexity_penalty`
  - 表达式太复杂
- `completeness_penalty`
  - full metrics 缺失太多

## MMR rerank

frontier 现在支持一层 MMR 风格 rerank：

- 先按原始 frontier 分数拿候选池
- 再做“高分但别太像”的重排

核心作用：

- 避免前几个 parent 都是同一条线的轻微改写

## motif-aware similarity

相似度不只看 token overlap，还会看：

- `branch_key`
- `motif_signature`
- `mutation_class`
- `operator_skeleton`
- `economic_family_tags`

所以现在的“太像”更接近研究语义，而不只是字符串像。

## selection rationale

每轮现在会落盘：

- `selected_parent_score_breakdown`
- `selected_parent_reason_tags`
- `runner_up`
- `selected_vs_runnerup_delta`
- `selection_pool`

如果 dual-parent 触发，还会落：

- `dual_parent_triggered`
- `selected_parents`
- 每个 parent 的 `role`
- `similarity_to_primary`

## dual-parent v1

### 角色

- `quality_parent`
  - 当前更像最强结果线
- `expandability_parent`
  - 当前更像最值得继续扩展的 branch

### 触发条件

当前核心是：

- top1 / top2 分差不能太大
- 两者不能太像
- secondary 必须在 expandability / branch value 上有足够理由

### 当前执行语义

- 同一轮双 parent
- 两条 parent 子批次可并行启动
- 最后再统一 merge child

### 当前不是

- 不是 full BFS
- 不是全并行 multi-parent
- 更像窄宽度、best-first 偏置的 branch search

## best_node 和 research_winner 的区别

### `search.best_node`

- 更偏搜索语义
- 是“下一轮系统更可能扩谁”

### `research_winner`

- 更偏 evaluation
- 是“这一轮最好的研究产出是谁”

两者经常接近，但不保证一样。
