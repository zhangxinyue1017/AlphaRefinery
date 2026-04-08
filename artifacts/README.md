# artifacts

这里是整个 `factors_store` 的运行产物根目录。

当前建议按下面几类理解：

- [backtests/](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/backtests)
  - 稳定 summary / 回测结果归档
- [runs/](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs)
  - 新的 canonical run 根目录
- [llm_refine_promotions/](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/llm_refine_promotions)
  - promotion pending / auto-apply 中间层
- [autofactorset_ingest/](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/autofactorset_ingest)
  - registry factor admission 配置、manifests、validated 清单
  - 实际 bridge 运行结果默认写入 `runs/autofactorset_ingest/`
- [reports/](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/reports)
  - 面向阅读的报告
- [logs/](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/logs)
  - 原始日志

和 `artifacts/` 平级的共享静态配置现在统一放在：

- [config/](/root/workspace/zxy_workspace/AlphaRefinery/config)
  - 受版本控制的运行前配置，例如 `refinement_seed_pool.yaml`

路径口径：

- 新 run：统一默认写入 `artifacts/runs/...`
- 报告：统一写入 `artifacts/reports/...`
- promotion 中间产物：统一写入 `artifacts/llm_refine_promotions/...`

不要混淆这两个边界：

- `artifacts/*`
  - 研究运行产物
- `config/*`
  - 受版本控制的共享静态配置
- `factors_store/factors/llm_refined/*.py`
  - 正式沉淀进 registry 的因子代码
