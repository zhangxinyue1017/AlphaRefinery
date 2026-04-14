# AlphaRefinery Project Map

这份文档只做一件事：

- 把 `/root/workspace/zxy_workspace/AlphaRefinery` 当前到底有哪些层、各自负责什么，讲清楚。

它不是设计文档，也不是开发计划，而是一个“先别迷路”的项目地图。

和根目录 README 的分工是：

- [README.md](/root/workspace/zxy_workspace/AlphaRefinery/README.md)
  - 讲这个仓库是什么、为什么存在、当前系统怎么协同
- [PROJECT_MAP.md](/root/workspace/zxy_workspace/AlphaRefinery/PROJECT_MAP.md)
  - 讲东西具体都放在哪、每一层各自负责什么

一句话理解：

- `README` 负责回答“这是什么”
- `PROJECT_MAP` 负责回答“东西都在哪”

## 1. 顶层目录

```text
AlphaRefinery/
├── README.md
├── PROJECT_MAP.md
├── requirements.txt
├── llm_refine_provider_env.example.sh
├── run_refine.sh
├── config/
├── factors_store/
├── artifacts/
└── factor_eval_output/
```

### 顶层文件说明

- [requirements.txt](/root/workspace/zxy_workspace/AlphaRefinery/requirements.txt)
  - 主流程依赖清单，方便一次安装运行环境
- [llm_refine_provider_env.example.sh](/root/workspace/zxy_workspace/AlphaRefinery/llm_refine_provider_env.example.sh)
  - provider env 模板；本地复制为 `llm_refine_provider_env.sh` 后再填写真实密钥
- [README.md](/root/workspace/zxy_workspace/AlphaRefinery/README.md)
  - 整个仓库的总入口说明
- [PROJECT_MAP.md](/root/workspace/zxy_workspace/AlphaRefinery/PROJECT_MAP.md)
  - 当前这份项目地图
- [run_refine.sh](/root/workspace/zxy_workspace/AlphaRefinery/run_refine.sh)
  - 历史上的 refine 启动脚本，偏快捷入口
- `llm_refine_provider_env.sh`
  - 本地 secret 文件，不入库；通常由 `llm_refine_provider_env.example.sh` 复制得到

## 2. 真正的代码目录：`factors_store/`

```text
factors_store/
├── __init__.py
├── __main__.py
├── _bootstrap.py
├── contract.py
├── data.py
├── eval.py
├── operators.py
├── registry.py
├── factors/
├── llm_refine/
└── autofactorset_bridge/
```

这是整个项目最核心的代码层。

### 基础框架层

- [contract.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/contract.py)
  - 数据字段 contract，决定哪些字段是核心字段、扩展字段、可选字段
- [data.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/data.py)
  - 把 `/root/dmd/BaoStock/panel.parquet` 等数据源整理成统一数据结构
- [operators.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/operators.py)
  - 宽表/时序/cross-sectional 常用算子
- [registry.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/registry.py)
  - 因子注册器，统一管理 `alpha / qp / gp / llm_refined`
- [eval.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/eval.py)
  - 日频 quick eval / backtest 主入口
- [__main__.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/__main__.py)
  - `python -m factors_store` 的 CLI 入口

### 因子实现层：`factors/`

```text
factors_store/factors/
├── alpha101_like.py
├── alpha158_like.py
├── alpha191_like.py
├── alpha360_like.py
├── qp_kline.py
├── qp_momentum.py
├── qp_volatility.py
├── qp_behavior.py
├── qp_salience.py
├── qp_chip.py
├── gp_mined.py
├── seed_baselines.py
└── llm_refined/
```

这里放的是“正式可注册”的因子代码。

- `alpha*.py`
  - 经典 alpha 因子库
- `qp_*.py`
  - Quants Playbook 整理出来的专题因子
- `gp_mined.py`
  - GP 挖掘出来的基础 seed 因子
- `seed_baselines.py`
  - baseline / seed 基线因子
- `llm_refined/`
  - 本地或私有沉淀的 refined family 因子包
  - 公开仓库中允许只保留包骨架，不包含私有 `*_family.py`

### `llm_refined/` 的定位

```text
factors_store/factors/llm_refined/
├── __init__.py
├── common.py
└── *_family.py  (local/private optional)
```

这是本地 / 私有可选的 LLM refined 因子层。

一句话理解：

- `artifacts/llm_refine_*` 是研究产物
- `factors/llm_refined/*.py` 可以是本地私有代码资产

## 3. LLM refine 子系统：`llm_refine/`

```text
factors_store/llm_refine/
├── README.md
├── cli/
├── core/
├── prompting/
├── parsing/
├── evaluation/
└── search/
```

这是最近最容易让人混乱的一层，因为功能已经从单脚本扩成一个小系统了。现在已经按职责拆包，顶层旧模块名只作为兼容 shim 保留。
这也是最近最容易让人迷路的一层，因为功能已经从单脚本扩成了一个子系统。

### 这层的职责拆分

- [config.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/config.py)
  - `llm_refine` 共享默认值中枢；常用 run/provider/path 默认参数优先看这里
- [core/seed_loader.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/core/seed_loader.py)
  - 读取 seed family pool
- [prompting/prompt_builder.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/prompting/prompt_builder.py)
  - 拼 prompt
- [parsing/parser.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/parsing/parser.py)
  - 解析 LLM 返回
- [parsing/validator.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/parsing/validator.py)
  - 表达式白名单 / 预校验
- [parsing/operator_contract.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/parsing/operator_contract.py)
  - prompt / validator / evaluator 的统一算子契约
- [parsing/expression_engine.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/parsing/expression_engine.py)
  - 表达式执行器
- [core/providers.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/core/providers.py)
  - 当前主要是 OpenAI-compatible provider
- [evaluation/redundancy.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/evaluation/redundancy.py)
  - family 冗余过滤
- [evaluation/evaluator.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/evaluation/evaluator.py)
  - staged evaluation + research gate
- [core/archive.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/core/archive.py)
  - sqlite archive + run artifact 写入
- [evaluation/promotion.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/evaluation/promotion.py)
  - pending promotion / auto-apply promotion
- [search/](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search)
  - 统一 `SearchEngine`、frontier、policy、normalization、run ingest

### `search/` 现在承载什么

这一层现在已经不是“预留目录”，而是当前 refine 的统一搜索骨架，主要包括：

- [search/state.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search/state.py)
  - `SearchNode / SearchEdge / SearchBudget`
- [search/policy.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search/policy.py)
  - `balanced / exploratory / conservative`
- [search/scoring.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search/scoring.py)
  - 搜索分项打分
- [search/frontier.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search/frontier.py)
  - frontier rerank / branch-aware selection
- [search/engine.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search/engine.py)
  - 统一搜索执行入口
- [search/normalization.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search/normalization.py)
  - archive-based percentile normalization
- [search/run_ingest.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/search/run_ingest.py)
  - 把 single/multi run 结果回灌进 search state

当前这条线的关键口径是：

- `search.best_node`
  - 搜索器认为最值得继续扩展的节点
- `research_winner`
  - evaluation 认为当前轮最值得沉淀的候选

二者相关，但不再强制等同。

### 和根 README 的关系

如果根 [README.md](/root/workspace/zxy_workspace/AlphaRefinery/README.md) 已经告诉你：

- `llm_refine` 是 family 级研究引擎

那这一节补的是：

- 这个引擎在代码里到底拆成了哪些目录
- 真正需要改搜索、改 parser、改 evaluation 时应该去哪里

### refine 运行入口

- [cli/run_refine_loop.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/cli/run_refine_loop.py)
  - 单轮单 parent
- [cli/run_refine_multi_model.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/cli/run_refine_multi_model.py)
  - 单 seed 多模型并行
- [cli/run_refine_multi_model_scheduler.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/cli/run_refine_multi_model_scheduler.py)
  - 单 seed 多模型自动多轮
- [cli/run_refine_family_explore.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/cli/run_refine_family_explore.py)
  - 多 seed family-level breadth-first explore，后续会收口到 family loop
- [cli/run_refine_family_loop.py](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/cli/run_refine_family_loop.py)
  - broad -> anchor graduation -> focused 的 family controller v1

## 4. Admission / 入库桥接层：`autofactorset_bridge/`

```text
factors_store/autofactorset_bridge/
├── evaluate_registry_manifest.py
└── batch_import_from_manifest.py
```

这是“研究结果 -> formal library admission”的桥。

- `evaluate_registry_manifest.py`
  - 对 manifest 中的因子做评估
  - 可选 `--insert-promoted`
- `batch_import_from_manifest.py`
  - 批量从 manifest 做导入

这层的定位不是 refine 本身，而是：
- 把 registry 中的因子，再走一层 admission 流程

## 5. 共享配置与产物目录

这是整个项目第二个容易混乱的地方，因为它既有：
- 回测结果
- refine 运行产物
- promotion
- manifest
- 报告

现在这一层已经补了：

- [artifacts/README.md](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/README.md)
- [reports/README.md](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/reports/README.md)
- [logs/README.md](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/logs/README.md)

运行所需的共享静态配置已经从 `artifacts/` 抽出来，避免和生成产物混在一起。

### `config/`

- [refinement_seed_pool.yaml](/root/workspace/zxy_workspace/AlphaRefinery/config/refinement_seed_pool.yaml)
  - 当前 `llm_refine` 最核心的 seed family 配置

### `artifacts/runs/`

- 新的 canonical run 根目录
- 未来统一承载：
  - `llm_refine_single/`
  - `llm_refine_multi/`
  - `llm_refine_multi_scheduler/`
  - `llm_refine_family_explore/`
  - `llm_refine_family_loop/`
- 历史 run 目录暂时保留在旧位置，避免打断已有结果引用

### `artifacts/llm_refine_runs/`

- 历史单轮 refine 的 run 目录
- 一个目录对应一次 `run_refine_loop`
- 新的 canonical 写入路径是：
  - `artifacts/runs/llm_refine_single/`
- 说明文件：
  - [llm_refine_runs/README.md](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/llm_refine_runs/README.md)

### `artifacts/runs/llm_refine_multi/`

- 单次 multi-model run 的 canonical 目录
- 一个目录对应一次 `run_refine_multi_model`

### `artifacts/runs/llm_refine_multi_scheduler/`

- 多模型自动多轮 scheduler 的 canonical 总控目录
- 一个目录对应一次 `run_refine_multi_model_scheduler`

### `artifacts/llm_refine_promotions/`

- pending promotion 产物
- auto promotion 相关草案和记录
- 说明文件：
  - [llm_refine_promotions/README.md](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/llm_refine_promotions/README.md)

### `artifacts/backtests/`

- 家族级 / 分库级 backtest summary
- `alpha / qp / gp / llm_refined / seed_baselines` 各自分桶

### `artifacts/autofactorset_ingest/`

- manifest admission 这一层的 manifests / validated
- 新的 bridge run 默认写入：
  - `artifacts/runs/autofactorset_ingest/`

### `artifacts/reports/`

- 报告层
- 当前已经分桶：
  - `family/`
  - `design/`
  - `system/`
  - `research_notes/`

### `artifacts/logs/`

- 偏原始运行日志
- 当前已经分桶：
  - `backtests/`
  - `batch_jobs/`
  - `system/`

## 6. 目前最容易混的 4 个点

### 1. `factors_store/factors/llm_refined/` vs `artifacts/llm_refine_*`

- `artifacts/llm_refine_*`
  - 研究运行产物
- `factors/llm_refined/*.py`
  - 正式沉淀进 registry 的代码

这是最核心的边界，千万不要混。

### 2. `run_refine_loop` / `multi_model` / `family_explore` / `family_loop`

现在 refine 入口已经不少：

- `run_refine_loop`
- `run_refine_multi_model`
- `run_refine_multi_model_scheduler`
- `run_refine_family_explore`
- `run_refine_family_loop`

这层如果没有文档，会非常容易忘记“我现在是在单 parent 深挖，还是在多 seed 广搜”。

但现在它们已经共享同一个 `SearchEngine` 语义：

- `run_refine_loop`
- `run_refine_multi_model`
- `run_refine_multi_model_scheduler`
- `run_refine_family_explore`
- `run_refine_family_loop`

区别更多在于：

- mode 不同
- budget 不同
- parent / frontier 的组织方式不同

### 3. `artifacts/logs/` 和 `artifacts/reports/`

- `logs`
  - 偏运行记录
- `reports`
  - 偏人工阅读总结

目前 `logs` 已经很多，后面如果不分层，找东西会越来越累。

### 4. `autofactorset_ingest` 和 `llm_refine` 的边界

- `llm_refine`
  - 负责 research / generation / evaluation / promotion
- `autofactorset_ingest`
  - 负责 manifest / admission / library side

两层会接起来，但不是一回事。

## 7. 我建议的整理方向

先不动代码逻辑，只讲目录认知上的整理方向。

### 第一优先级：文档分层

- 根 [README.md](/root/workspace/zxy_workspace/AlphaRefinery/README.md)
  - 讲整个仓库是什么、主线是什么、系统如何协同
- [llm_refine/README.md](/root/workspace/zxy_workspace/AlphaRefinery/factors_store/llm_refine/README.md)
  - 专讲 refine 子系统是什么、入口怎么选
- [PROJECT_MAP.md](/root/workspace/zxy_workspace/AlphaRefinery/PROJECT_MAP.md)
  - 专讲目录地图和落点

### 第二优先级：artifacts 认知分层

以后看 `artifacts/` 时，建议脑子里先分成：

- `config`
- `runs`
- `promotion`
- `admission`
- `reports`
- `logs`

### 第三优先级：继续维护文档和运行口径一致

当前已经完成了这些：

- `reports/` 和 `logs/` 的第一层分桶
- `artifacts/runs/` 作为 canonical run 根目录
- `llm_refine` 的分层拆包
- `SearchEngine` / policy preset / percentile normalization

后面更重要的不是继续搬目录，而是持续保证：

- README 口径
- artifact 路径
- search / evaluation 语义

三者保持一致。

## 8. 相关但不在本仓库里的评估链路

分钟频率因子评估主要在：

- [single_factor_eval/README.md](/root/gp_factor_qlib/evaluation/single_factor_eval/README.md)
- [examples/README.md](/root/gp_factor_qlib/evaluation/single_factor_eval/examples/README.md)

这条链路和 `AlphaRefinery` 不是同一仓库，但它和这里的：

- 日频 factor evaluation
- `llm_refine` 候选研究
- formal admission

会共享同一批研究语境。

## 9. 一句话版

如果只记一句：

- `factors_store/factors/` 是正式代码资产
- `factors_store/llm_refine/` 是自动优化系统
- `artifacts/` 是研究运行产物和报告

这三层先分清，整个项目就不会再那么乱。
