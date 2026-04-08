# runs

这里是新的 canonical run 根目录。

新的 `llm_refine` 相关运行默认写入这里：

- `llm_refine_single/`
- `llm_refine_multi/`
- `llm_refine_multi_scheduler/`
- `llm_refine_family_explore/`
- `llm_refine_family_loop/`

整理原则：

- 新 run 统一写这里
- 历史 `llm_refine_scheduler/`、`llm_refine_queue/` 结果仍保留，只是不再作为当前主工作流入口
- 报告、日志、promotion 仍然留在各自独立分桶下
