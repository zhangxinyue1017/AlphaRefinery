# autofactorset ingest

这个目录专门用来整理从 `factors_store` 侧进入 `autofactorset` 的候选清单与中间产物。

当前推荐分工：

- `manifests/`
  - 人工维护的候选清单
  - 主要来源于 family report、手工筛选结论、seed onboarding 结果
- `validated/`
  - 由 bridge 脚本校验后导出的结构化清单
  - 可作为后续真正入库批处理的上游输入

为什么单独放在 `factors_store`：

- 这些候选本来就是从 `factors_store` registry 中来的
- 因子名、family、source model、selection 指标都在这里更容易维护
- `autofactorset` 当前原生入口主要吃 `expr_json/node_dict`，而这批候选大量是 registry factor
- 因此更自然的做法是：
  1. 在 `factors_store` 管理 manifest
  2. 用 bridge 脚本校验 factor_name 与 registry 一致性
  3. 后续再决定是走 registry bridge 评估，还是导出成 `expr_json` 再提交给 `autofactorset`

当前文件：

- `manifests/llm_refined_reports_20260325_candidates.yaml`
  - 基于两个 `20260325` split report 整理出的第一批入库候选
- `validated/`
  - 预留给 `batch_import_from_manifest.py` 的校验输出
- 运行结果现在默认落到：
  - [artifacts/runs/autofactorset_ingest/](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/autofactorset_ingest)
  - 旧的 `autofactorset_ingest/runs/` 仅保留给历史结果

建议工作流：

1. 先维护 manifest
2. 跑 bridge 脚本做 dry-run 校验
3. 确认 factor_name / family / source_model / role 没问题
4. 再决定是否接下一步真正的批量入库

当前可用脚本：

- `python -m factors_store.autofactorset_bridge.batch_import_from_manifest`
  - 做 registry 一致性校验
  - 不跑回测
 - `python -m factors_store.autofactorset_bridge.evaluate_registry_manifest`
  - 直接按 `factor_name` 从 `factors_store` registry 计算因子
  - 复用 `single_factor_backtest + autofactorset promotion rules`
  - 默认只做评测和 promotion 判定
  - 加 `--insert-promoted` 后才会正式写入 `autofactorset` SQLite 库
  - 新的默认 run 根目录是：
    - `artifacts/runs/autofactorset_ingest/`

常用示例：

```bash
PYTHONPATH=/root/workspace/zxy_workspace/AlphaRefinery:/root \
python -m factors_store.autofactorset_bridge.batch_import_from_manifest \
  --manifest /root/workspace/zxy_workspace/AlphaRefinery/artifacts/autofactorset_ingest/manifests/llm_refined_reports_20260325_candidates.yaml \
  --output /root/workspace/zxy_workspace/AlphaRefinery/artifacts/autofactorset_ingest/validated/llm_refined_reports_20260325_candidates.validated.json
```

```bash
PYTHONPATH=/root/workspace/zxy_workspace/AlphaRefinery:/root \
python -m factors_store.autofactorset_bridge.evaluate_registry_manifest \
  --manifest /root/workspace/zxy_workspace/AlphaRefinery/artifacts/autofactorset_ingest/manifests/llm_refined_reports_20260325_candidates.yaml
```

```bash
PYTHONPATH=/root/workspace/zxy_workspace/AlphaRefinery:/root \
python -m factors_store.autofactorset_bridge.evaluate_registry_manifest \
  --manifest /root/workspace/zxy_workspace/AlphaRefinery/artifacts/autofactorset_ingest/manifests/llm_refined_reports_20260325_candidates.yaml \
  --insert-promoted
```
