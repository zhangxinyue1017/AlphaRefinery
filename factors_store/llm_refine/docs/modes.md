# 运行模式

## 目标

这份文档只回答一个问题：

- 当前要解决的问题是什么
- 应该用哪一个入口

## 默认前置步骤

所有 `llm_refine` 入口在运行前，都默认先执行：

```bash
cd AlphaRefinery
cp -n ./llm_refine_provider_env.example.sh ./llm_refine_provider_env.sh
source ./llm_refine_provider_env.sh
```

不要把这一步省掉。

如果不先加载这个 env 文件，`run_refine_*` 会回退到 CLI 内置默认 provider / base_url，这通常不是当前研究想用的实际配置。

## 入口总览

### `run_refine_loop`

- 文件：
  - `cli/run_refine_loop.py`
- 适合：
  - 验证一条 family 能不能跑通
  - 看 prompt / parser / evaluator 是否正常
  - 手工围绕一个 parent 做单轮 refine

### `run_refine_multi_model`

- 文件：
  - `cli/run_refine_multi_model.py`
- 适合：
  - 围绕一个 parent，让多个模型同时出 proposal
  - 做 focused round / round2.5
- 特点：
  - 单轮
  - 单 parent
  - 多模型

### `run_refine_multi_model_scheduler`

- 文件：
  - `cli/run_refine_multi_model_scheduler.py`
- 适合：
  - 多模型 + 自动多轮
  - family 已经有一定基础，想让系统自己连续滚动
- 特点：
  - 接入统一 `SearchEngine`
  - 支持 `conditional dual-parent round v1`
  - 当前 dual-parent 是同轮双 parent、子批次可并行执行

### `run_refine_family_explore`

- 文件：
  - `cli/run_refine_family_explore.py`
- 适合：
  - seed 很少
  - 不确定主线
  - 想从多个 family seed 同时起跑做广搜
- 备注：
  - 这是当前 family 级 orchestration 的过渡入口
  - 后续会逐步让位给更完整的 `family_loop`

### `run_refine_family_loop`

- 文件：
  - `cli/run_refine_family_loop.py`
- 适合：
  - broad run 之后自动挑 1 个 best anchor
  - 自动再起一条 focused run
  - 想把“人工挑 strongest branch 再继续挖”的动作收成闭环
- 当前 `v1`：
  - deterministic graduation
  - 单 family
  - 单 anchor
  - broad -> focused -> summary

### `run_next_experiments`

- 文件：
  - `cli/run_next_experiments.py`
- 适合：
  - 不直接跑 refine
  - 先生成下一批 family / motif transfer 建议

## 最常见的选择

### 想看一条新 family 能不能跑

- 用 `run_refine_loop`

### 想围绕当前 winner 再挖一层

- 用 `run_refine_multi_model`

### 想让系统自己连续跑 2 到 3 轮

- 用 `run_refine_multi_model_scheduler`

### 想做 breadth-first family 探索

- 用 `run_refine_family_explore`

### 想把 broad -> focused 自动串起来

- 用 `run_refine_family_loop`

## parent 语义

### 默认

- 一轮只围绕一个 `current parent`

### dual-parent scheduler

- 同一轮可以有两个 parent：
  - `quality_parent`
  - `expandability_parent`
- 当前 `v1` 语义：
  - 同轮双 parent
  - 两条 parent 子批次可并行启动
  - 两边完成后统一 merge child

## parent 来源优先级

优先级从高到低：

- 显式指定：
  - `--current-parent-name`
  - `--current-parent-expression`
  - `--parent-candidate-id`
- `--auto-parent`
- family `canonical_seed`

## bootstrap parent

multi-model scheduler 现在支持：

- round1 bootstrap parent 强制起跑

这和“把候选塞进 selection pool 里碰碰运气”不一样。  
如果开了 bootstrap，round1 会按你给的 parent 直接开局。

## 推荐模型池

当前常用 5 模型池：

- `gpt-5.4`
- `deepseek-v3.1`
- `qwen3.5-plus`
- `claude-sonnet-4-6`
- `kimi-k2`

## 产物目录

### 单轮

- `artifacts/runs/llm_refine_single/`

### 多模型

- `artifacts/runs/llm_refine_multi/`

### 多模型 scheduler

- `artifacts/runs/llm_refine_multi_scheduler/`

### family explore

- `artifacts/runs/llm_refine_family_explore/`

### family loop

- `artifacts/runs/llm_refine_family_loop/`
