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
