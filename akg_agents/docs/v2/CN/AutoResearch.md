[English](../AutoResearch.md)

# AutoResearch 工作流

## 1. 概述

AutoResearch 是 AKG Agents 中的 Agent 驱动迭代优化工作流。与单次代码生成（KernelGen）或种群搜索（Evolve/AdaptiveSearch）不同，AutoResearch 使用 ReAct Agent 循环，由 LLM 自主规划优化策略、编辑代码、评测结果，并在多轮迭代中自动保留或回滚更改。

注册为可调用工作流（`call_autoresearch_workflow`），可通过 KernelAgent 的 ToolExecutor 调用，也可通过 `LangGraphTask(workflow="autoresearch")` 直接调用。

### 适用场景

| 场景 | 推荐工作流 |
|------|-----------|
| 快速初稿，无需验证 | KernelGen |
| 生成 + 验证单个候选 | KernelGenOnly |
| 并行探索多种变体 | Evolve / AdaptiveSearch |
| **单个 kernel 的深度迭代优化** | **AutoResearch** |

AutoResearch 最适合以下情况：
- 已有初始 kernel，需要深度性能调优
- 优化空间复杂，需要多轮试错
- Agent 自主能力（策略规划、按需阅读文档、诊断子 Agent）比种群多样性更有价值

## 2. 架构

```
KernelAgent (ReAct)
  │
  ├─ call_autoresearch_workflow(op_name, task_desc, dsl, ...)
  │
  ▼
AutoresearchWorkflow (LangGraph 单节点)
  │
  ├─ 1. 预检（fail-closed: 无 worker → 中止）
  │     a) 获取 worker + 创建 KernelVerifier
  │     b) 验证参考实现: check_task_desc_static + check_task_desc_runtime
  │     c) 解析 seed:
  │        有 previous_code?
  │          → 运行时验证（参考实现 + kernel 联合编译）
  │            → 通过: 直接使用
  │            → 失败: 落入 KernelGen
  │        KernelGen 循环 (×3):
  │          → 生成 → CodeChecker（静态）→ 运行时验证
  │          → 失败: 错误回喂给 KernelGen 重试
  │     d) 释放预检 worker
  │
  ├─ 2. 知识组装: Skill 系统 + 硬件文档 + API 文档
  │     Layer 0（系统提示）: fundamental/reference skills + 硬件信息
  │     Layer 1（docs/ 目录）: guides、examples、API 文档（按需读取）
  │
  ├─ 3. scaffold_task_dir: 生成任务目录 + git init
  │
  ├─ 4. AgentLoop: 基于 Plan 的自主优化循环
  │     ┌────────────────────────────────────────────────────────┐
  │     │                                                        │
  │     │  ┌─ LLM 回合 ─────────────────────────────────┐       │
  │     │  │  update_plan → patch_file/write_file        │       │
  │     │  │  read_file   → compact   → finish           │       │
  │     │  └──────────────────────────────────────────────┘       │
  │     │         │                                              │
  │     │    FeedbackBuilder（Plan 状态机）                       │
  │     │    [no_plan] → [active] → [replanning] → ...          │
  │     │         │                                              │
  │     │    ┌─ TurnExecutor（自动流水线）────────────────┐       │
  │     │    │  edit → quick_check → eval_fn → settle   │       │
  │     │    │           │             │          │      │       │
  │     │    │       仅语法检查   KernelVerifier  keep/  │       │
  │     │    │                   验证+profile   discard  │       │
  │     │    │                                   (git)   │       │
  │     │    └───────────────────────────────────────────┘       │
  │     │         │                                              │
  │     │    SessionStore（快照 / 恢复 / 心跳）                   │
  │     │         │                                              │
  │     │    自动诊断（连续失败 N 次触发）                         │
  │     │    → 诊断子 Agent → require_replan                     │
  │     │         │                                              │
  │     │    上下文压缩（microcompact + auto_compact）             │
  │     │                                                        │
  │     └────────────────────────────────────────────────────────┘
  │
  └─ 5. 读取最优结果 → {coder_code, verifier_result, profile_res}
```

### 模块布局

```
op/autoresearch/              # Autoresearch 核心（可独立运行）
  agent/                      # AgentLoop、TurnExecutor、工具、反馈、压缩
    loop.py                   # 主 Agent 循环（ReAct: LLM → 工具 → 评测）
    turn.py                   # 单次 LLM 对话执行器（edit → quick_check → eval 流水线）
    tools.py                  # 工具定义 + 执行 + 护栏
    feedback.py               # FeedbackBuilder: Plan 状态机 + 评测反馈构造
    session.py                # SessionStore: 文件快照、会话恢复、心跳、逐轮日志
    compress.py               # 上下文压缩（microcompact + auto_compact）
    llm_client.py             # 独立 LLM 客户端（Anthropic/OpenAI，调试模式用）
    file_logger.py            # stdout 镜像写入 agent.log
    prompts/                  # 系统提示模板
  framework/                  # 配置、评测、实验运行器
    config.py                 # TaskConfig, AgentConfig, EvalResult, RoundRecord
    evaluator.py              # 评测执行器（子进程或注入的 eval_fn）
    runner.py                 # ExperimentRunner: 评测 → 判定 → git → 记录
    logger.py                 # JSONL + Markdown 双格式轮次日志
    report.py                 # 最终报告生成（文本 + 可选图表）
  adapters/                   # AKG 集成层（核心代码不 import 此层）
    llm_adapter.py            # AkgLLMAdapter: 包装 LLMClient → ConversationAdapter
    task_scaffolder.py        # 为 AKG workflow 模式生成任务目录

op/workflows/
  autoresearch_workflow.py    # LangGraph 工作流（AKG 胶水层，单文件）
```

### 依赖方向

```
autoresearch_workflow.py ──imports──→ autoresearch 核心（agent/, framework/）
autoresearch 核心 ──不 import──→ AKG 任何模块

AKG 集成通过注入实现:
  llm_adapter  → AgentLoop（llm_adapter= 参数）
  eval_fn      → ExperimentRunner → run_eval_robust（eval_fn= 参数）
```

## 3. 工具配置

| 属性 | 值 |
|------|---|
| TOOL_NAME | `call_autoresearch_workflow` |
| 用途 | 对已有 kernel 进行深度迭代优化 |
| 输出 | 优化后的 kernel 代码 + profile 结果（latency、speedup） |

### 参数

| 参数 | 类型/必填 | 说明 |
|------|----------|------|
| op_name | str（必填） | 算子名称 |
| task_desc | str（必填） | 框架代码（Model/get_inputs/get_init_inputs 格式） |
| dsl | str（必填） | 目标 DSL: `triton_cuda`, `triton_ascend`, `torch`, `cuda_c`, `cpp`, `ascendc` 等 |
| framework | str（必填） | 目标框架: `torch` |
| backend | str（必填） | 目标后端: `cuda`, `ascend` |
| arch | str（必填） | 目标架构: `a100`, `ascend910b4` 等 |
| previous_code | str（可选） | 初始 kernel 代码（未提供时由 KernelGen 生成种子） |
| max_rounds | int（可选） | 最大评测轮数（默认: 20） |

## 4. 核心流程

### 4.1 预检验证

工作流在 AgentLoop 启动前执行 fail-closed 预检。任何步骤失败都立即中止，不浪费 eval 轮次。

1. **获取 worker**（必须）。无可用 worker → 直接中止。

2. **验证参考实现**（`task_desc`）：
   - `check_task_desc_static()`: AST 级结构检查（class Model、get_inputs、get_init_inputs）
   - `check_task_desc_runtime()`: 在 worker 上实际执行参考实现，确认能运行
   - 任一失败 → 中止。参考实现是 ground truth，必须能跑。

3. **解析 seed**（初始 kernel）：
   - **用户提供 kernel**（`previous_code`）：通过 `KernelVerifier.run()` 运行时验证。失败则落入 KernelGen — 用户的 kernel 作为上下文保留。
   - **KernelGen 路径**：每次尝试三层验证：生成 → CodeChecker（静态）→ KernelVerifier（运行时）。错误回喂给 KernelGen 重试（最多 2 次）。

4. **释放 worker**。预检 worker 在整个阶段持有，通过 `finally` 块释放。

### 4.2 知识组装

DSL 知识分两层组装：

- **Layer 0（系统提示，始终可见）**: Skill 系统中的 fundamental/reference skills + 硬件文档。保持紧凑，遵守 40K 字符上下文预算。
- **Layer 1（docs/ 目录，Agent 按需读取）**: guide skills、example skills、API 文档（如 triton_ascend API）。Agent 通过 `read_file` 工具按需访问。

对于没有 skill 包的 DSL（torch、ascendc、swft、tilelang_npuir），回退到旧的 `_DSL_DOCS_DIR_MAP` 文档目录。

### 4.3 评测（eval_fn）

评测以 KernelVerifier 为唯一权威。工作流构造一个 `eval_fn` 闭包：

1. 从 WorkerManager 借用 worker（按次借用，非整个循环占用）
2. 从 task_dir 读取当前可编辑文件
3. 调用 `verifier.run()` 进行正确性验证
4. 调用 `verifier.run_profile()` 进行延迟测量
5. 返回 `EvalResult(correctness, metrics, error)`
6. 归还 worker

**优化措施：**
- **每轮单次 eval 调用**: `eval_fn` 每轮只调用一次（verify + profile）。profiler 内部的 `run_times` / `warmup_times` 控制重复运行。
- **base_time 缓存**: 参考实现不变，因此 `base_profile` 只运行一次。后续 profile 使用 `use_reference_data + override_base_time_us` 跳过冗余基线测量。

### 4.4 LLM 可用工具

Agent 可使用六个工具。注意 `quick_check` 和 `eval` 不是 LLM 工具 — 它们在编辑后由 TurnExecutor 自动触发。

| 工具 | 说明 |
|------|------|
| `update_plan` | 提交结构化计划（`- [ ]` 格式），条目自动分配 ID（p1, p2, ...），第一条激活 |
| `patch_file` | 对可编辑文件执行精准字符串替换。增量编辑首选 |
| `write_file` | 全文件重写。用于初始生成或大规模重构 |
| `read_file` | 读取可编辑文件或 docs/ 上下文文件，支持全文和范围模式 |
| `compact` | 手动触发上下文压缩 |
| `finish` | 显式终止优化循环并输出总结 |

### 4.5 基于 Plan 的执行模型

Agent 运行在由 `FeedbackBuilder` 管理的 **Plan 状态机**下。这确保了顺序化、可追踪的执行，而非随意编辑。

**三个阶段：**

| 阶段 | 含义 | 允许操作 |
|------|------|---------|
| `no_plan` | 初始状态，未提交计划 | 仅 `update_plan` |
| `active` | 一个计划条目正在执行 | `patch_file`/`write_file`（需匹配 `plan_item_id`）、`read_file` |
| `replanning` | 所有条目已结算 | `update_plan`（新计划）或 `finish` |

**Plan 条目生命周期：**
1. Agent 调用 `update_plan(plan="- [ ] 尝试 BLOCK_SIZE=512\n- [ ] 融合 epilogue")`
2. 创建条目：`p1`（激活）、`p2`（待定）
3. Agent 用 `plan_item_id="p1"` 编辑代码 → TurnExecutor 自动运行 quick_check → eval
4. Eval 结果自动结算 p1（KEEP → `done_ok`，FAIL/DISCARD → `done_fail`）
5. p2 自动激活 → 重复
6. 所有条目结算 → 进入 `replanning` → Agent 提交新计划或调用 `finish`

**编辑拦截**: 非 `active` 阶段或 `plan_item_id` 不匹配时，编辑请求会被拒绝。

### 4.6 自动 Post-Edit 流水线

每次成功编辑（patch_file / write_file）后，TurnExecutor 自动执行：

1. **quick_check** — 语法验证（py_compile）、DSL 合规检查、import 验证、可选 smoke test。失败 → 立即回滚，不消耗 eval。
2. **eval**（通过 `eval_fn`）— 完整正确性验证 + 性能 profiling。返回 KEEP / DISCARD / FAIL。
3. **自动结算** — 根据 eval 结果结算当前活跃的 plan 条目。
4. **反馈注入** — 结构化评测结果 + 性能排名 + 累积 diff（KEEP 时）注入对话上下文。

### 4.7 三态模型

每轮评测产生三种结果之一：

| 状态 | 条件 | 动作 |
|------|------|------|
| **KEEP** | 正确性通过 + 满足约束 + 优于历史最优 | Git commit，更新最优，重置失败计数 |
| **DISCARD** | 正确性通过但无改进 | Git rollback，重置失败计数 |
| **FAIL** | 正确性失败或崩溃 | Git rollback，递增失败计数 |

### 4.8 自动诊断子 Agent

当 Agent 累积连续失败达到阈值（每 `diagnose_suggest_threshold` 次，默认 3 次）时，系统自动：

1. 启动**诊断子 Agent** — 使用全新 LLM 上下文，只有代码文件的只读访问权限
2. 子 Agent 分析错误上下文和当前代码，返回结构化诊断（根因 + 修复建议 + 应避免的模式）
3. 调用 `require_replan()` — **阻止所有编辑**直到 Agent 提交新计划
4. 诊断结果作为强制方向变更注入对话

这防止 Agent 无限重复相同的失败方法。

### 4.9 上下文压缩

长时间优化会话可能超出 LLM 上下文窗口。三种压缩机制保持对话可控：

- **microcompact**: 自动截断较旧的工具结果（低于 `microcompact_min_chars`），仅保留最近 `microcompact_keep_recent` 条完整结果。
- **auto_compact**: 估算 token 数超过 `context_limit × compression_threshold` 时，触发基于 LLM 的摘要，用紧凑摘要替换旧消息。
- **compact 工具**: Agent 可通过 `compact` 工具手动触发压缩。

### 4.10 会话持久化

`SessionStore` 提供崩溃恢复和可观测性：

- **快照**: 每轮编辑前对可编辑文件做快照，失败时原子回滚。
- **会话保存/加载**: 完整 Agent 状态（计划、eval 次数、失败次数、消息）持久化，支持 `--resume`。
- **心跳**: PID 锁 + 状态文件，用于监控运行中的 Agent。
- **逐轮日志**: 每轮追加写入 JSONL，供事后分析。

## 5. 配置

AutoResearch 使用标准 AKG 配置系统（`load_config` + `build_langgraph_task_config`）。

### 任务级配置

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `max_step` | 20 | 最大评测轮数（可被 `max_rounds` 参数覆盖） |
| `agent_model_config.coder` | "standard" | Agent 使用的 LLM 模型级别 |
| `eval_timeout` | 120 | 单轮评测超时（秒） |
| `workflow_timeout` | 动态计算 | 整体工作流超时。在 workflow 初始化时自动计算为 `max(1800, max_rounds * (eval_timeout + 60) + 300)`，确保循环有足够时间完成。可在 config 中显式覆盖 |
| `profile_settings.run_times` | 50 | Profiling 重复次数 |
| `profile_settings.warmup_times` | 5 | Profiling 预热次数 |

### AgentConfig（框架行为参数）

通过 `task.yaml` 的 `agent.config` 块覆盖。大多数任务使用默认值即可。

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| **实验控制** | | |
| `max_consecutive_failures` | 10 | 连续失败 N 次后中止 |
| `max_no_edit_turns` | 3 | 连续 N 轮无编辑后强制提示 |
| `max_turns_multiplier` | 8 | 安全上限：最大 LLM 调用次数 = max_rounds × 此值 |
| **LLM** | | |
| `llm_max_tokens` | 8192 | LLM 响应最大 token 数 |
| `thinking_budget` | 8000 | 扩展思考预算（0 = 禁用） |
| `call_timeout` | 120.0 | API 调用超时（秒） |
| **上下文压缩** | | |
| `context_limit` | None | 模型上下文窗口 token 数（None = 禁用自动压缩） |
| `compression_threshold` | 0.75 | 达到 context_limit 的此比例时触发 auto_compact |
| `microcompact_keep_recent` | 3 | 保留最近 N 条工具结果不压缩 |
| **诊断** | | |
| `diagnose_suggest_threshold` | 3 | 每连续失败 N 次触发诊断子 Agent |
| `subagent_max_iterations` | 15 | 诊断子 Agent 最大迭代次数 |
| **截断** | | |
| `editable_file_truncate` | 8000 | 系统提示中可编辑文件内容截断长度 |
| `system_context_total_truncate` | 40000 | 系统上下文总字符预算 |
| `raw_output_tail` | 2048 | 反馈中 eval 原始输出尾部长度 |
| `cumulative_diff_truncate` | 10000 | KEEP 时累积 diff 长度 |

## 6. 使用方式

### 通过 KernelAgent（生产路径）

KernelAgent 在用户请求深度迭代优化时自动选择 AutoResearch。工具调用经过 ToolExecutor → `prepare_config()` → `build_initial_state()` → 工作流执行。

### 通过 LangGraphTask（直接路径）

```python
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.config.config_validator import load_config
from akg_agents.core.worker.manager import register_local_worker
from akg_agents.utils.task_label import resolve_task_label

await register_local_worker([0], backend="cuda", arch="a100")

config = load_config(dsl="triton_cuda", backend="cuda")
config["task_label"] = resolve_task_label(op_name="my_op", parallel_index=1)
config["max_step"] = 20

task = LangGraphTask(
    op_name="my_op",
    task_desc=open("reference.py").read(),
    task_id="my_op_001",
    backend="cuda", arch="a100",
    dsl="triton_cuda", framework="torch",
    config=config,
    workflow="autoresearch",
)

op_name, success, final_state = await task.run()
```

### 通过 CLI 脚本

```bash
# 自然语言描述（LLM 生成参考实现，KernelGen 生成初始 kernel）
python scripts/run_autoresearch.py --desc "fused ReLU + LayerNorm, input shape (32, 1024), fp16" --backend cuda

# 自然语言 + 初始 kernel（LLM 生成参考实现，跳过 KernelGen）
python scripts/run_autoresearch.py --desc "fused ReLU + LayerNorm, input shape (32, 1024), fp16" --kernel path/to/kernel.py --backend cuda

# 参考实现文件（KernelGen 生成初始 kernel）
python scripts/run_autoresearch.py --ref path/to/reference.py --backend cuda

# 参考实现 + 初始 kernel（跳过所有生成）
python scripts/run_autoresearch.py --ref path/to/reference.py --kernel path/to/kernel.py --backend cuda
```

## 7. 与其他工作流对比

| 特性 | KernelGenOnly | Evolve | AdaptiveSearch | **AutoResearch** |
|------|--------------|--------|----------------|-----------------|
| 策略 | 单次生成 | 种群进化 | UCB 多臂老虎机 | Agent 驱动 ReAct |
| 验证 | 一次 | 每候选一次 | 每候选一次 | 每轮一次 |
| 迭代 | 1 | 多次（并行） | 多次（并行） | 多次（顺序） |
| 自主性 | 无 | 模板驱动 | 模板驱动 | 完全自主（Agent 规划、诊断、按需阅读文档） |
| 适用 | 快速初稿 | 广泛探索 | 策略选择 | 深度优化 |
| DSL 知识 | 静态注入 | 静态注入 | 静态注入 | 按需读取（read_file） |
| 失败恢复 | 无 | 种群吸收 | 种群吸收 | 诊断子 Agent + 强制 replan |

## 8. 测试

| 测试 | 路径 | 覆盖范围 |
|------|------|---------|
| LangGraphTask 直接路径 | `tests/op/st/test_autoresearch.py` | Ascend: seed → scaffold → AgentLoop → eval_fn → KernelVerifier |
| LangGraphTask 直接路径 | `tests/op/st/test_autoresearch_cuda.py` | CUDA 变体 |
| ToolExecutor 生产路径 | `tests/op/st/test_autoresearch_toolexecutor.py` | KernelAgent → ToolExecutor → prepare_config → build_initial_state → workflow + config 隔离 |
| ToolExecutor 生产路径 | `tests/op/st/test_autoresearch_toolexecutor_cuda.py` | CUDA 变体 |
