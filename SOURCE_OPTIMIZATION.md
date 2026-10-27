# 原生源码多轮编排

`dace/source_optimization.py` 将原生诊断、源码提案、成对确认和独立选择串联；入口为 `tools/run_source_optimization.py`。它操作真实 `SourcePolicy` 源码槽位，不使用模拟配置开关。显式传入 `level` 启用v3：持久CMVS、C0–C5/Contribution-only反馈投影、版本失效与控制器联合目标。任意语义作用域分裂、其留出验证、四配置交互识别及保留候选子链搜索仍未完整接入原生循环；不能据此宣称整个方法复现完成。

## 冻结计划

先用 `TaskRegistry` 冻结整个数据集的 DIAG/CONFIRM/SELECT/EVAL 分区及 lineage，再用 `tools/freeze_source_runtime.py` 冻结覆盖所有轮次任务的同一执行环境。不能把已披露任务重新命名为独立任务。

Python API：

```python
import json
from dataclasses import asdict
from pathlib import Path
from dace.agents.policy import SourcePolicy
from dace.core import Artifact
from dace.source_optimization import make_optimization_plan

# task IDs 必须来自上述研究登记表；每个任务在整个计划中只出现一次。
rounds = [{
    'diagnostic': ['diagnostic-task'],
    'confirmation': [['confirm-slot-0-task'], ['confirm-slot-1-task']],
    'selection': ['selection-task'],
    'families': {'selection-task': 'declared-family'},
}]
plan = make_optimization_plan(
    Artifact.create(SourcePolicy().content, kind='BASELINE'),
    rounds, dataset, command_argv, 'deepseek/deepseek-v4.1-flash',
    study_binding, runtime_lock,
    mechanism_cap=4, action_cap=4, intervention_cap=2,
    alpha=.05, coding_max_tokens=12000, level='C5',
    # 可选：eval_tasks 必须来自研究中独立的 EVAL 角色。
    # eval_tasks=['blind-evaluation-task'],
)
Path(plan_path).write_text(json.dumps(plan, indent=2), encoding='utf-8')
```

示例中的变量须绑定为真实本地绝对路径、冻结登记信息及完整 runner argv；示例任务名不是可运行 TB2 任务配置。单个选择任务通常不能通过统计晋级门槛，不能用它证明收益。

不传 `level` 保持v2兼容路径。v3可选 `C0/C1/C2/C3/C4/C5/CONTRIBUTION_ONLY`。C0不生成机制；每级都保留基本验证结果用于反思。C1有持久竞争空间，C2读取故障声明，C3独立读取故障/贡献，C4才生成联合目标，C5再按明确冻结的预测分歧调度。机制状态按父版本和完整假设revision隔离，来源相容本身不能提高优先级；只有实际判别证据支持的预测才得到该优先级。溢出机制归档而不删除历史，晋级使旧父版本记录和目标过期。

机制 revision 同时绑定该假设的全部动作 Artifact，不能因动作名称相同就把旧补丁证据转给新代码。原生调度器在同一父版本下考虑所有仍活跃提案；下一轮没有新提案时，仍可在新的预定确认槽位上检验既有提案。每个动作保留其原始 batch 的冻结预测，不补造跨 batch 预测。同一 Artifact 本轮只调度一次，先比较冻结预测分歧，再按提案新旧和动作 ID 确定顺序。C0只使用当轮动作；已归档及旧父版本动作不参与调度。这是同父版本提案保留，不是在未部署候选上继续扩展子链。

C5 的实际判别记录覆盖冻结的干预边界和中介谓词：同一族、同一字段的两个相反预测分别达到 `CONSISTENT` 和 `CONTRADICTED` 时才记录支持/冲突。不能从另一谓词族借用状态，也不能把经验匹配或聚合标签作为该字段的统计证据。记录按固定字段顺序生成，以保证跨进程检查点重建。

```powershell
python tools/run_source_optimization.py --plan D:/path/to/plan.json --output D:/path/to/run
```

同一命令恢复同一计划和目录。运行锁覆盖源码，执行期间不要修改代码；发生漂移会拒绝继续使用该证据。

## 预算与隔离

- 每轮诊断上限为 `N_diag + 10` 单位，10 为一次 Coding API 请求的固定记账单位。
- 每个确认槽位为 `2 * N_confirm_slot`，基线与候选各执行一次。槽位数由独立参数 `intervention_cap` 限制，不超过 `action_cap`。候选先按冻结预测分歧排序，再绑定预定槽位。
- 每轮选择预留 `(intervention_cap + 1) * N_select`；实际只比较有有效确认凭证的候选，同一选择任务共享一次基线。
- 整个运行先预留所有轮次的上述额度，再启动第一条诊断。未使用余额最后释放；不能借用下一阶段额度扩大本阶段搜索。
- 全局 alpha 均分给所有预定确认槽位和选择阶段；即使某槽位最终不用，也不事后重分配显著性预算。确认器还会在本次竞争预测间分配自身 alpha。
- 新建优化计划默认v2，确认阶段使用不可变 `PredicateProtocol` 的固定样本精确二项区间；默认违约容差为0.1，必须在执行前确定。每个机制预测的alpha再分给失败/干预边界/中介三个谓词族及效用对照，各占四分之一；族内按字段分配。单条匹配不被统计确认。区间要求独立同分布二元任务单位，当前没有聚类/分层替代。旧v1计划仅恢复其原有样本匹配协议，不追溯升级。
- 第二轮直接继承第一轮独立选择的精确获胜 Artifact；没有胜者则继承原父版本。下一轮反馈带原任务、版本与作用域；部署选择向优化器披露之前登记 SELECT 来源。EVAL 不会传给 Coding Agent。
- 若提供 `eval_tasks`，额外预留基线及每轮结束时各一次完整评估，共 `(轮数+1)*N_eval`；所有检查点使用同一事先冻结的EVAL样本。优化器不会读取其奖励、轨迹或聚合分数；重复检查点不是新的独立样本。摘要分别列优化与评估支出。

这些预算是执行/请求成本单位，不是美元或 token 上限。`execution_usage` 与 `optimizer_usage` 分别读取子账本的供应商费用；任一字段缺失时相应总费用保持未知。外层镜像预算仅用于总额度审计，不能与子阶段执行费用相加。

## 恢复与证据

外层账本在执行前冻结子计划，逐笔以子账本扣费事件哈希镜像全局支出。若子阶段已经完成、外层尚未写回执，恢复时读取子阶段原始证据完成记账，不重新请求模型或执行任务。已开始但没有提交观测的任务由子阶段终止为未解决，外层随之终止，保留此前部署版本。

每份完成回执绑定子账本末尾哈希；完成后修改、追加或丢失子证据都被拒绝。`summary.json` 是派生视图，恢复不读取它来决定成功。外层终止后只重建该视图，不运行下一轮。

受控测试覆盖两轮晋级与父版本继承、全局额度、alpha 分配、任务隔离、诊断/确认/选择中断，以及子结果和外层记账之间的崩溃。它们替换执行器和模型，不能作为真实 TB2 或 Full DACE 成绩。

后续 `native-optimization-controlled-smoke` 已完成真实单轮整链：明确注入源码故障，官方奖励依次为诊断0、确认0/0、独立选择0/1。选择只有一题，未通过冻结统计门槛，没有修改部署父版本。5次执行消耗15/15单位，终止后恢复没有追加调用或改动账本；容器/网络均已清空。该运行使用冻结v1协议，仅支持操作级修复和链路验证，不能追认为v2统计确认或正式E1–E5结果。原生真实多轮及原生C0–C5仍待验证。

## 不复跑任务的原生报告

```sh
python tools/report_native_run.py --run runs/study --output runs/study-report
```

报告核验外层回执、子账本哈希、原始官方reward及EVAL观察，输出 `curve.csv`、`stages.csv` 和 `report.json`。曲线只使用EVAL，SELECT奖励不会被改名为Pass@1；没有盲评时曲线为空。阶段开销从子账本读取，不把外层预算镜像重复加成费用。跨种子置信带、正式主实验对照和论文图仍需后续真实运行与汇总。
