# 真实源码的独立选择

`dace.source_selection` 为真实源码确认之后的独立选择提供固定时点入口。它本身不自动诊断、生成候选或分配整场优化预算；这些由 `source_optimization` 串联，仍不是完整 Full DACE。已知源码故障下的真实单轮整链已完成，选择任务fix-git的奖励为0/1，但仅一题、未达到晋级门槛，没有晋级；不能据此声明性能提升。

## 入围证据

`confirmation_certificate(evidence_jsonl)` 直接读取候选自己的确认账本，不依赖 `summary.json` 中的声明。它重新校验：

- 源码/父版本/冻结契约与真实修改键集相符，全部固定任务各有一个成对观察。
- 原始执行、官方reward文件与观察相符，来源为无oracle的真实agent，记录了执行token。
- 样本按原契约预算各计费一次，确认预算已结清，无失败终止或执行失效事件。
- 用原预测和观察重新计算的固定时点结果与账本一致，候选确有自身 `MINIMAL_OBSERVED` 激活证据。

仅有单项确认不能替代组合工件确认；这里接受的是确切候选版本。贡献尚未显著不阻止进入独立效用选择，但已有源端修复/补偿标签不保证晋级。判别实验中的foreign-action声明不进入获胜者的自身部署归因。

## 计划与预算

所有候选必须来自相同的直接父版本、模型、runner、研究/lineage和执行锁。`make_selection_plan` 接受确认账本路径、固定SELECT任务、逐任务能力族、数据集与协议参数：

```python
from dace.source_selection import make_selection_plan

plan = make_selection_plan(
    confirmations=[confirmation_a / 'evidence.jsonl', confirmation_b / 'evidence.jsonl'],
    tasks=select_tasks,
    families={task_id: 'primary' for task_id in select_tasks},
    dataset=dataset, command=runner_argv,
    model='deepseek/deepseek-v4.1-flash',
    study=study_binding, runtime_lock=shared_runtime_lock,
    alpha=.05, delta=0., epsilon=0.)
```

执行锁v2可在确认前为整个预定任务范围生成：`capture_runtime_lock(runner_argv, dataset, confirmation_tasks + select_tasks)`。每个阶段可以只执行该范围的子集，但不能临时加入未冻结的任务。确认与选择必须共用同一锁；不能更新代码后把旧确认追认为新运行环境的证据。

每个SELECT任务只执行一次基线，再执行所有固定候选；N个任务、K个候选，预留并实际按 `(K+1)*N` 个执行成本单位计费。基线成本在候选间平均分摊用于选择成本比较，总账不重复收取，也不把节省的执行隐藏成零成本。美元/token保留在逐执行记录中。

选择任务必须位于冻结研究的SELECT角色，并与候选确认任务分离；EVAL角色不能用于选择。整场优化的确认/晋级预算及跨轮alpha分配仍须由外层协议提前冻结，本入口仅冻结本次独立选择阶段。

## 运行与恢复

```powershell
python tools/run_source_selection.py --plan selection-plan.json --output runs/source-selection --batch 2
python tools/run_source_selection.py --plan selection-plan.json --output runs/source-selection
```

每次batch处理完整的任务单元，包括该任务的基线和全部候选。到达预定样本量前只报告 `PENDING_SELECTION`，不提前选赢家。最终对所有候选的总体收益和能力族非劣性使用同时校正的配对Hoeffding界；样本太少或无收益时不晋级。

崩溃发生在任务启动之后、完整单元提交之前时，恢复终止为 `TERMINATED_UNRESOLVED`，不重跑基线或换题。观察全部提交、最终判定尚未落盘时，恢复只补完固定时点判断。已经完成的选择原样恢复，不新增执行或账本事件。

`summary.json` 返回确切 `deployed_artifact`；有赢家时就是原候选Artifact，没有赢家时保持原基线。下一轮可将该完整Artifact直接传给源码确认的 `make_plan`，不会重新包装成初始BASELINE。获胜者自身的版本/作用域/修复或补偿声明单独保留，不能因为效用晋级而补造根因修复结论。

此处的SELECT成绩用于优化决策，不是最终Benchmark Pass@1。顺序当前固定为基线后候选；实际容器镜像证明、自动源码机制搜索及完整真实演化闭环仍未完成。

2026-10-27真实链路尝试在上游确认阶段超时，按协议未进入SELECT；因此尚不能宣称独立选择的真实模型集成已验证。详见 `runs/20261027-integrity-fixes/real-source-selection-smoke/verification.json`。受控选择/恢复测试与真实模型成功验证分开报告。
