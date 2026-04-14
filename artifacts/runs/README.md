# runs

这里是新的 canonical run 根目录。

现在 README 会按 bucket 自动汇总，并在每个子目录下按 `family/seed` 生成索引，方便快速定位每天正在跑的目录。

## Buckets

- [ab_tests](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/ab_tests)  
  runs=`13`; running=`0`; families=`2`
- [autofactorset_ingest](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/autofactorset_ingest)  
  runs=`22`; running=`1`; families=`10`
- [llm_refine_family_explore](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_family_explore)  
  runs=`0`; running=`0`; families=`0`
- [llm_refine_family_loop](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_family_loop)  
  runs=`1`; running=`0`; families=`1`
- [llm_refine_multi](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_multi)  
  runs=`8`; running=`0`; families=`3`
- [llm_refine_multi_scheduler](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_multi_scheduler)  
  runs=`24`; running=`0`; families=`7`
- [llm_refine_scheduler](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_scheduler)  
  runs=`1`; running=`0`; families=`1`
- [llm_refine_single](/root/workspace/zxy_workspace/AlphaRefinery/artifacts/runs/llm_refine_single)  
  runs=`3`; running=`0`; families=`2`

整理原则：

- 新 run 统一写这里
- README 只做索引，不改真实目录结构
- 历史 bucket 继续保留，但会在各自 README 里按 family/seed 聚合
