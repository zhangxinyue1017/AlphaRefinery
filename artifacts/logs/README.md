# logs

这里放的是**原始日志**，不是最终报告。

当前分桶：

- [backtests/](/root/workspace/zxy_workspace/factors_store/artifacts/logs/backtests)
  - search / selection / final_oos / family backtest 相关日志
- [batch_jobs/](/root/workspace/zxy_workspace/factors_store/artifacts/logs/batch_jobs)
  - 批量回测、全量筛选、刷新任务日志
- [system/](/root/workspace/zxy_workspace/factors_store/artifacts/logs/system)
  - 其余系统级、遗留或单次任务日志

规则：

- 这里偏 stdout/stderr 原始输出
- 如果某份内容已经升格为结论，请写到 `artifacts/reports/`
- 不要把研究总结和日志混放
