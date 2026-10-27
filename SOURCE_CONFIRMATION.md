# 真实源码确认入口

本入口执行真实 Terminal-Bench 任务，并把测得的边界送入现有固定时点确认控制器。
它是完整真实演化流程的基础接口，不是 Full DACE 或 AutoSaddler 实现。
确认之后的独立选择入口见 [SOURCE_SELECTION.md](SOURCE_SELECTION.md)。已知源码故障下的真实单轮整链已完成；独立选择只有一题，未通过晋级门槛，不能据此声明整体性能改善。

## 修改范围

`dace.agents.terminus.EditableTerminus` 基于 Harbor 0.25.0 的 Terminus 2，提供三个版本化槽位：

| 槽位 | 实际执行位置 | 可观察事实 |
| --- | --- | --- |
| response_code | 模型响应交给原解析器之前 | 原始/变换后响应解析是否成功 |
| command_code | 已解析命令交给 terminal 之前 | 输出命令是否符合结构契约 |
| instruction_suffix | instruction 交给原 agent 之前 | 指定后缀的 Hook 是否实际执行、是否改变输入 |

`response_code` 和 `command_code` 是 `def transform(value)` 形式的受限 Python 函数，支持纯字符串、列表与字典变换。它们不能导入模块、读取文件、访问私有属性或执行任意调用。该限制定义初始编辑范围，不是针对恶意代码的完整资源隔离沙箱。

Trace 分开记录 Hook 执行和输入是否改变。激活只说明当前修改的指定 Hook 被调用，不证明修改改善任务结果。官方 verifier 是唯一效用来源；不会用解析成功、Hook 激活或候选配置替代官方 reward。

## 冻结计划

用 `dace.source_confirmation.make_plan` 生成 JSON 计划。参数为两个 `SourcePolicy`、预先声明的 `Prediction` 列表、唯一任务 ID、真实数据集目录、runner argv 和执行模型。

```python
import json
import sys
from pathlib import Path
from dace.agents.policy import SourcePolicy
from dace.core import Prediction
from dace.predicate_evidence import PredicateProtocol
from dace.source_confirmation import make_plan

root = Path.cwd()
baseline = SourcePolicy()
candidate = SourcePolicy(response_code='def transform(value):\n    return value.strip()\n')
prediction = Prediction(
    'response-normalization', 'raw_response_parse_valid', ('model', 'parser'),
    (('raw_response_parse_valid', False),),
    (('raw_response_parse_valid', False), ('response_parse_valid', True)),
    (('command_batch_valid', True),), 'COMPENSATION')
# 在查看确认结果前，根据独立诊断冻结这些谓词及任务。
plan = make_plan(baseline, candidate, [prediction], ['YOUR_FRESH_TASK_ID'],
    root/'datasets/terminal-bench-2',
    [sys.executable, str(root/'tools/tb2_live_runner.py'), '--harbor',
     'D:/path/to/venv/Scripts/harbor.exe', '--max-turns', '8'],
    'deepseek/deepseek-v4.1-flash', predicate_protocol=PredicateProtocol(max_violation_rate=.1))
Path('plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
```

只允许真实 Hook 的布尔谓词：`raw_response_parse_valid`、`response_parse_valid`、`command_batch_valid`、`terminal_dispatch_completed`。这些是逐 trial 聚合的执行事实，不是完整的语义正确性或根因真值。

显式传入 `PredicateProtocol` 时生成v2统计契约，固定违约容差、完整样本时点及任务级精确二项区间。三个谓词族与效用对照各取得该预测alpha的四分之一，谓词族内再按字段分配；置信区间上界不超过容差时才标为 `CONSISTENT`，下界超过容差才标为 `CONTRADICTED`，否则保持未解决。缺失观测不因其他样本匹配而被忽略。它要求独立同分布二元任务单位，尚未实现聚类/分层替代；不适用时不能据此作正式统计结论。

为恢复历史账本，未传入协议的直接确认API仍生成v1，其 `CONSISTENT` 仅表示已观察样本的谓词匹配。不能把v1单条匹配追认为总体机制支持。v2使用独立 `StatisticalContract`，不会改写v1契约ID；经验相容或源端桥接也不声称唯一根因或中介效应识别。

计划绑定完整 baseline/candidate artifact、直接父版本、实际修改键集、任务 ID、instruction/manifest/environment/tests 文件指纹、预测、模型、runner argv 与确认预算。指纹只读取这些执行输入，不读取 `solution`。每个成对单元预扣两次任务执行的成本单位，真实 token/美元另见逐执行账本。

## 运行与恢复

```powershell
python tools/run_source_confirmation.py --plan plan.json --output runs/source-confirm --batch 1
python tools/run_source_confirmation.py --plan plan.json --output runs/source-confirm
```

第二条命令续跑同一个计划。不能修改任务、代码、预测、预算或 runner argv 后继续沿用原账本。已完成的任务不再次执行；固定样本量达到前不产生最终声明。运行已启动但观察尚未提交的中断属于不明确状态，恢复时终止整个契约为 `TERMINATED_UNRESOLVED`，不重新执行或换题。独占 writer 锁阻止两个进程同时推进同一确认账本。

产物包括 `evidence.jsonl`、`summary.json` 和各次执行的官方 Harbor trial、模型轨迹、verifier reward 与身份绑定的 Hook Trace。恢复失败时释放 writer 锁；被截断或完整性不符的账本会拒绝恢复，不自动修复。

摘要中的 `execution_usage` 区分 `recorded_cost_usd` 和完整 `cost_usd`。发生超时、失败、未提交单元或缺失供应商字段时，总费用为null；不能把已经返回的基线费用当成包含失败候选的全部费用。原始供应商usage仍保留。

宿主超时覆盖准备、agent执行和verifier全部阶段；900秒宿主上限不等于让agent执行900秒再额外完成验证。正式协议需为这些阶段固定足够的时间包络。超时/中断会终止拥有的宿主进程树；异常结束后仅尝试清理原生trial配置和精确compose标签能证实归属的Docker资源，清理未验证会单独报告，不产生任务分数。

## 跨会话任务隔离

`TaskRegistry` 使用共享 SQLite 事务登记一项研究的完整任务角色划分。准备 JSON allocation，显式将数据集中的每个任务分入 `DIAG`、`CONFIRM`、`SELECT` 或 `EVAL`，不能重复、遗漏或把相同内容改名当作独立任务。

```powershell
python tools/manage_study.py --registry D:/UserData/lulu/Projects/dace/runs/study.sqlite3 freeze --id pilot --dataset datasets/terminal-bench-2 --allocation allocation.json
python tools/manage_study.py --registry D:/UserData/lulu/Projects/dace/runs/study.sqlite3 lineage --study pilot --id full-seed0
```

将以下绑定传给 `make_plan(..., study=...)`：

```python
study = {'registry': 'D:/UserData/lulu/Projects/dace/runs/study.sqlite3',
         'id': 'pilot', 'lineage': 'full-seed0'}
```

同一优化上下文的所有轮次使用同一个 lineage；派生上下文须通过 `lineage ... --parent full-seed0` 继承历史。真正隔离的重复运行才可创建新的根 lineage，共用同一固定角色划分；不能换 ID 消除已有知识。子 lineage 预约之后若父 lineage 获得相关任务反馈，执行前会再次检查并拒绝使用。

整个会话的任务在第一次执行前原子预约，其他会话不能借用；换输出目录、丢弃本地账本或执行失败都不释放任务。允许恢复同一会话，但每个成对单元的启动只登记一次。EVAL 可以在冻结的同一评估池上做多个检查点，但其任何任务或聚合反馈都不能披露给优化器。

优化器收到诊断、确认或选择反馈时，须通过 `TaskRegistry.disclose` 登记派生数据的真实来源任务和字段；尚未启动的任务不能冒充数据来源。该接口已接入真实源码确认，自动源码优化器尚未接通。

## 执行实现冻结

`capture_runtime_lock(command, dataset, tasks)` 记录项目 Python 源码、runner 文件和启动器哈希、Harbor 源码哈希、Python 信息、已安装包版本以及相关本地 Docker 镜像 ID/RepoDigest。可用 `tools/freeze_source_runtime.py --help` 查看无模型调用的预检入口。需显式指定独立环境的 `--harbor`，仅有 Dockerfile、尚未准备镜像的任务会被拒绝，不能假装已经固定环境。

传入 `make_plan(..., runtime_lock=lock)` 后，计划创建、每次模型执行前和执行后都会核对该锁，并重新检查任务输入。后置检查失败会保留真实执行 usage 和原始产物、追加失效事件，终止为未解决；不会接受该对任务分数。

```powershell
python tools/run_source_confirmation.py --plan plan.json --output runs/source-confirm --require-study --require-runtime-lock
```

两个必需开关可防止遗漏研究绑定或冻结配置。当前镜像保证是执行前后的本地身份核对，尚无已删除容器实际 image ID 的独立证明；完整二进制环境、网络响应及托管模型权重也不是该锁的保证范围。

## 正式实验前的限制

- 此确认入口没有自动诊断/竞争机制搜索、LLM源码提案或C0–C5真实演化主循环；原主循环仍使用合成布尔修改目录。独立选择已有单独入口，未被包装成完整演化闭环。
- 未绑定 study 的旧计划仅有本会话隔离，会明确标记 `task_isolation=session-only`；它们不能作为已完成跨会话隔离的正式实验。不能把已用于 smoke 的 `fix-git` 当作全新确认任务。
- 顺序固定为 baseline 后 candidate；环境随机性、实际容器镜像证明、完整二进制依赖锁及正式预算分配仍需冻结。没有 runtime_lock 的旧计划不能防止固定 argv 所指脚本被外部修改。
- 单题、重复同题及八轮限制只能验证集成；不能用于正式 Pass@1 表、统计显著收益、根因定位准确率或有效补丁单位成本结论。
- 完整演化主循环和晋级状态的重启恢复仍待实现；这里恢复的仅是已冻结确认义务。
