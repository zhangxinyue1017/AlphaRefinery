# llm_refine
> AlphaRefinery 中负责 family 级因子 refine / search / evaluation 的子系统说明

## 这份 README 的角色

这份文档不再承担“仓库首页”的职责。  
仓库级说明请看：

- [AlphaRefinery/README.md](../../README.md)

这份 README 只回答三个问题：

1. `llm_refine` 这个子系统是做什么的
2. 什么时候该用哪个入口
3. 代码和文档应该从哪里继续往下看

## 子系统定位

`llm_refine` 负责把：

- `seed family`
- `parent selection`
- `LLM proposal`
- `parser / repair`
- `evaluation / redundancy`
- `archive / promotion`
- `search state update`

串成一个可复用的研究闭环。

它当前更像：

- family 级因子研究引擎
- unified best-first / dual-parent search 层
- 研究产物沉淀与下一轮 budget allocation 的控制层

## 运行前默认前置步骤

凡是运行 `llm_refine` CLI，都默认先执行：

```bash
cd /root/workspace/zxy_workspace/AlphaRefinery
cp -n ./llm_refine_provider_env.example.sh ./llm_refine_provider_env.sh
source ./llm_refine_provider_env.sh
```

原因：

- `run_refine_*` CLI 如果没有显式 provider 参数，会读取环境变量
- 如果环境变量也没有加载，就会退回到 CLI 内置默认值
- 那个默认值只适合本地兜底，不应当被当成日常研究 workflow 的默认配置

所以现在的约定是：

- **先从 `llm_refine_provider_env.example.sh` 复制出本地 `llm_refine_provider_env.sh`**
- **再 `source ./llm_refine_provider_env.sh`**
- **再运行任意 `run_refine_*` / scheduler / family explore 命令**

## 当前关键能力

- `run_refine_loop`
  - 单轮 smoke / 单 parent refine
- `run_refine_multi_model`
  - focused multi-model 单轮
- `run_refine_multi_model_scheduler`
  - unified search + 多轮 scheduler
- `run_refine_family_explore`
  - multi-seed breadth explore
  - 当前是 family 级 orchestration 的过渡入口
- `run_refine_family_loop`
  - broad -> anchor graduation -> focused 的 family controller v1
- MMR rerank
- dual-parent round v1
- Path Evaluation v2
- Target-Conditioned Search v1
- Context-aware decision layer v1
  - round-level rerank
  - family-loop anchor selection
  - next action recommendation
- De-correlation refine support
  - target-aware prompt block
  - de-correlation diagnostics
  - rerank hook
- `NA-heavy keep / best_node` 收紧

## 目录结构

```text
factors_store/llm_refine/
├── README.md
├── docs/
├── cli/
├── core/
├── prompting/
├── parsing/
├── evaluation/
├── knowledge/
└── search/
```

### 各层职责

- `config.py`
  - `llm_refine` 的共享默认值入口，集中管理常用 run/provider/path 默认参数
- `cli/`
  - 运行入口与 scheduler orchestration
- `core/`
  - provider / archive / model / seed_loader
- `prompting/`
  - prompt 构建与导出
  - PromptPlan / prompt block structuring
- `parsing/`
  - parser / validator / repair / expression engine
- `evaluation/`
  - evaluator / redundancy / promotion
- `knowledge/`
  - retrieval / reflection / next experiment planning
- `search/`
  - frontier / policy / engine / objectives / path evaluation
- `docs/`
  - 子系统设计与调参说明

## 最常用入口

| 入口 | 适合场景 |
|------|----------|
| `run_refine_loop` | 验证 family 能不能跑 |
| `run_refine_multi_model` | 围绕一个 parent 做 focused round |
| `run_refine_multi_model_scheduler` | 让系统自动连续跑多轮 |
| `run_refine_family_explore` | 新 family 广搜、主线未明 |
| `run_refine_family_loop` | broad 结束后自动挑 anchor 并继续 focused |

## 模式分层说明

`llm_refine` 不同层参数解释：

- `stage_mode`
- `target_profile`
- `policy_preset`
- `mode`


| 层 | 回答的问题 | 典型字段 |
|---|---|---|
| 阶段层 | 这轮现在在整个流程里处于什么阶段 | `stage_mode` |
| 目标层 | 这轮最终想优化什么类型的因子 | `target_profile` |
| 风格层 | 搜索时更激进还是更保守 | `policy_preset` |
| 搜索器层 | 搜索器如何组织候选与 frontier | `mode` |

### 补充：context-aware decision layer

除了上面的“模式分层”，`llm_refine` 最近还开始把原来分散的选择逻辑往一层统一 decision layer 收敛。

当前这层主要统一了：

- multi-model round 的 `best_candidate / best_keep / rerank preview`
- family loop 的 `anchor selection`
- family loop 的 `next action recommendation`

它读取的上下文包括：

- `stage_mode`
- `target_profile`
- `policy_preset`
- optional `decorrelation_targets`
- neutralized evaluation diagnostics

目标不是引入新的大而杂规则，而是让 broad / focused / family loop 在同一套上下文下做更一致的决策。

### 补充：shared context resolver

在 `DecisionContext` 和 `PromptPlan` 之间，最近又补了一层更轻的共享解释层：

- `ContextEvidence`
- `ContextProfile`
- `resolve_context_profile(...)`
- `OrchestrationProfile`
- `resolve_orchestration_profile(...)`

它的目标不是替代原始运行参数，而是把：

- `stage_mode`
- `target_profile`
- `policy_preset`
- seed-stage / bootstrap / donor / decorrelation 等运行时证据

先统一解析成一份共享的中间画像，再分别提供给：

- prompt block planning
- decision trace
- 后续 family / scheduler orchestration

当前这层还比较克制，主要先统一：

- `search_phase`
- `exploration_pressure`
- `redundancy_pressure`
- `prompt_constraint_style`
- `memory_mode`
- `examples_mode`
- `branching_bias`
- `next_action_bias`

它现在已经会随 run 一起记录到 `prompt_trace` 中，便于后续排查：

- 为什么某轮 prompt 更偏 guided / strict
- 为什么 donor / bootstrap block 被打开或关闭
- 当前上下文更像 opening / refining / confirming 的哪一种

在 scheduler / family-loop 这一层，最近还新增了更动作导向的 `OrchestrationProfile`。

它当前不是用来全面接管 orchestration，而是先服务于：

- stage resolve recommendation
- round strategy trace
- promotion / parent selection / termination bias 的显式记录
- 顶层 `summary.json / summary.md` 中的 orchestration-aware 可读展示

这层的目标是：

- 先让 orchestration 和 prompt / decision 说同一种上下文语言
- 再逐步把最稳定、最重复、最容易解释的判断收进自动流程

### 补充：de-correlation refine

de-correlation 不是默认全局强制目标，而是一种可显式启用的 refine 方向。

当前已落地的部分包括：

- prompt 中的 de-correlation target set
- evaluator 中的 nearest-target / average-target correlation diagnostics
- rerank 中的 soft adjustment hook

它更适合用在：

- family 已有主线
- admission / library 冗余明显
- 想继续产出“质量不差且不那么像”的支线因子时

### 补充：PromptPlan / prompt block structuring

最近 prompt 层开始做一层更轻量的结构化整理，但**不改变模型最终看到的输入格式**。

当前新增的不是另一套“强控制 prompt 引擎”，而是一层较克制的 `PromptPlan`：

- 只控制 prompt block 的 inclusion / budget / style
- 暂时不压缩 family 语义
- 仍然把最终内容 render 成自然语言

当前 `PromptPlan` 主要覆盖三块：

- `memory`
  - 是否展示 recent winners / keeps / failures / lineage / reflection
  - 以及各自的数量预算
- `constraints`
  - 约束区块的组织方式与是否展示 anti-pattern / allowed edit types / decorrelation guidance
- `examples`
  - family examples / bootstrap frontier / donor motifs 是否出现，以及展示多少

这层的目的不是替代原始上下文，而是把原来散在 prompt builder 里的 block-level if/else 显式化、可追踪化。

当前 run artifact 里的 `prompt_trace` 也已经会一并记录：

- `stage_mode`
- `target_profile`
- `policy_preset`
- `context_evidence`
- `context_profile`
- `prompt_plan`

这样后面调 prompt 时，可以直接看到：

- 为什么某轮开了 memory block
- 为什么 donor motifs 没开
- 当前 constraints 是按什么 style 组织的

而不需要只靠读最终 prompt 反推。

### 1. `stage_mode`

`stage_mode` 是“任务语义标签”，决定这一轮在整个 family research loop 里扮演什么角色。

| `stage_mode` | 含义 | 典型行为 | 适合场景 |
|---|---|---|---|
| `auto` | 让系统根据上下文自动判断阶段 | 是否开 seed-stage 由上下文决定 | 兼容旧 workflow、普通试跑 |
| `new_family_broad` | 新 family 的首轮 broad | 强制 seed-stage、开启 bootstrap / donor / 更丰富 role slots | 新 family 第一次启动 |
| `broad_followup` | broad 的后续轮 | 仍然偏 search-space opening，但不再是首轮 seed-stage | broad 第 2 轮及以后 |
| `focused_refine` | 围绕已有 parent 做局部精修 | 倾向小步改写、减少无谓发散 | 已找到 anchor 或主线 |
| `confirmation` | 对已有候选做确认 | 更偏稳定性与可持续性验证 | 准备 freeze / promote 前 |
| `donor_validation` | 做 donor / transfer 假设验证 | 更关注跨 family motif 是否成立 | donor 试验、迁移验证 |

即：

- `stage_mode` 决定的是“**这轮在干嘛**”。

### 2. `target_profile`

`target_profile` 是“目标函数偏好”，决定这轮更希望找到哪类因子。

当前可选项来自 [policy.py](./search/policy.py)：

| `target_profile` | 含义 | 更偏重什么 | 适合场景 |
|---|---|---|---|
| `raw_alpha` | 优先找原始 alpha 强度高的因子 | `Ann` / `Excess` / `ICIR` | broad、focused 早中期，先找强信号 |
| `deployability` | 优先找更容易落地、约束更友好的因子 | constraint / 可部署性 | 后期收口、实盘友好筛选 |
| `complementarity` | 优先找与已有库互补的因子 | 低冗余、组合补充价值 | admission 前、去冗余 refine |
| `robustness` | 优先找更稳健的因子 | regime / 稳定性 | confirmation、后期稳健性检查 |

即：

- `target_profile` 决定的是“**这轮想要什么**”。

### 3. `policy_preset`

`policy_preset` 是“搜索风格包”，决定同样的搜索器更保守还是更激进。

当前可选项同样来自 [policy.py](./search/policy.py)：

| `policy_preset` | 含义 | 典型特征 | 适合场景 |
|---|---|---|---|
| `balanced` | 平衡型默认配置 | 探索、质量、冗余控制都保持中间水平 | 大多数普通 run |
| `exploratory` | 更愿意打开结构空间 | 更高 exploration / novelty bonus、更大的候选池 | broad、新 family、想多开分支时 |
| `conservative` | 更注重稳健局部改进 | 更重 turnover / complexity penalty，更少发散 | focused、confirmation、主线已明时 |

即：

- `policy_preset` 决定的是“**这轮怎么搜得更激进/更保守**”。

### 4. `mode`

`mode` 更偏“搜索器组织方式”，决定 frontier 与 child record 如何被组织和 rerank。

目前在 [policy.py](./search/policy.py) 里可以看到的典型 `mode` 包括：

| `mode` | 含义 | 典型行为 | 适合场景 |
|---|---|---|---|
| `multi_model_best_first` | 多模型 best-first 搜索 | 从多模型 child 中统一挑当前更值得继续的记录 | multi-model / scheduler 主路径 |
| `family_breadth_first` | family 级 breadth-first 搜索 | 更优先扩 branch / motif，而不是只追 top1 邻域 | broad、family explore、新 family 起步 |

即：

- `mode` 决定的是“**搜索树怎么走**”。

## 推荐组合

下面这张表给出当前最常用的组合方式。

| 任务 | `stage_mode` | `target_profile` | `policy_preset` | 常见 `mode` |
|---|---|---|---|---|
| 新 family 首轮 broad | `new_family_broad` | `raw_alpha` | `exploratory` | `multi_model_best_first` 或 `family_breadth_first` |
| broad 后续轮 | `broad_followup` | `raw_alpha` | `exploratory` 或 `balanced` | `multi_model_best_first` |
| 已有主线 focused | `focused_refine` | `raw_alpha` | `balanced` | `multi_model_best_first` |
| admission 前去冗余 refine | `focused_refine` | `complementarity` | `balanced` | `multi_model_best_first` |
| 后期确认 / freeze 前检查 | `confirmation` | `robustness` 或 `deployability` | `conservative` | `multi_model_best_first` |


## dual-parent 当前语义

当 dual-parent 触发时，同一轮会保留两条线：

- `quality_parent`
- `expandability_parent`

当前执行方式是：

- 同轮双 parent
- 两条 parent 子批次可并行启动
- 完成后统一 merge child

## Target-Conditioned Search v1

当前已支持 target profile：

- `raw_alpha`
- `deployability`
- `complementarity`
- `robustness`

当前已落地的 aware module：

- `Constraint-aware`
- `Portfolio-aware`

当前先保留接口：

- `Regime-aware`
- `Transfer-aware`

## 先看哪份文档

### 想知道该用哪个入口

- [docs/modes.md](./docs/modes.md)

### 想看 frontier / MMR / dual-parent

- [docs/search_and_dual_parent.md](./docs/search_and_dual_parent.md)

### 想调 `SearchPolicy`

- [docs/policy_tuning.md](./docs/policy_tuning.md)

### 想看 `research_keep / research_winner / promotion`

- [docs/evaluation_and_promotion.md](./docs/evaluation_and_promotion.md)

### 想看 `Path Evaluation v2`

- [docs/path_evaluation.md](./docs/path_evaluation.md)

## 快速示例

所有示例都默认已经执行过：

```bash
cd /root/workspace/zxy_workspace/AlphaRefinery
cp -n ./llm_refine_provider_env.example.sh ./llm_refine_provider_env.sh
source ./llm_refine_provider_env.sh
```

### 单轮 smoke

```bash
PYTHONPATH=/root/workspace/zxy_workspace/AlphaRefinery \
python -m factors_store.llm_refine.cli.run_refine_loop \
  --family salience_panic_score \
  --n-candidates 3 \
  --auto-parent
```

### focused multi-model

```bash
PYTHONPATH=/root/workspace/zxy_workspace/AlphaRefinery \
python -m factors_store.llm_refine.cli.run_refine_multi_model \
  --family weighted_upper_shadow_distribution \
  --models gpt-5.4,deepseek-v3.1,qwen3.5-plus \
  --policy-preset balanced \
  --target-profile complementarity \
  --n-candidates 6
```

## 产物目录

- 单轮：
  - `artifacts/runs/llm_refine_single/`
- 多模型：
  - `artifacts/runs/llm_refine_multi/`
- 多模型 scheduler：
  - `artifacts/runs/llm_refine_multi_scheduler/`
- family explore：
  - `artifacts/runs/llm_refine_family_explore/`
- family loop：
  - `artifacts/runs/llm_refine_family_loop/`
