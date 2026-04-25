# op/ — 算子/内核生成

## 职责

内核代码生成的完整场景层：从用户需求到生成代码、验证、profiling 的端到端链路。对应 CLI 的 `akg_cli op` 命令。

> **架构定位**：`op/` 是 `core_v2/` 之上的第一个业务场景层（算子级）。后续将新增图优化等更上层编译场景的场景层，与 `op/` 平级，同样构建在 `core_v2/` 之上。新增场景层时应参考 `op/` 的目录组织模式（agents/workflows/verifier/resources/config）。

## 目录结构

| 子目录 | 职责 | 规范 |
|--------|------|------|
| `agents/` | 算子侧 Agent（KernelGen、KernelDesigner 等） | [SPEC.md](agents/SPEC.md) |
| `workflows/` | 算子工作流（继承 OpBaseWorkflow） | [SPEC.md](workflows/SPEC.md) |
| `langgraph_op/` | LangGraphTask、KernelGenState、nodes、routers | — |
| `verifier/` | 内核正确性验证 + 多后端适配器 | [SPEC.md](verifier/SPEC.md) |
| `adaptive_search/` | UCB 自适应搜索 | — |
| `resources/` | Prompt（.j2）、Skill、DSL 文档、模板 | [SPEC.md](resources/SPEC.md) |
| `config/` | 各 DSL 默认 yaml 与校验 | [SPEC.md](config/SPEC.md) |
| `database/` | coder 数据库、向量库 | — |
| `skill/` | 算子侧 skill 选择 | — |
| `tools/` | OpToolExecutor、task_constructor 等领域工具 | — |
| `utils/` | 算子专用工具（evolve 子包等） | — |

## 开发约定

### 关键入口

- `LangGraphTask`（`langgraph_op/task.py`）：算子任务执行入口，内含 `WORKFLOW_REGISTRY`
- `OpBaseWorkflow`（`workflows/base_workflow.py`）：所有算子工作流的基类
- `KernelVerifier`（`verifier/kernel_verifier.py`）：验证入口，组合 backend/dsl/framework 三类适配器

### Prompt 路径

`get_prompt_path()`（`utils/common_utils.py`）固定指向 `op/resources/prompts/`，所有 Agent 的 prompt 模板放在这里。

## 不做什么

- **不要**在此写通用框架逻辑（Agent 基类、Skill 系统等）——归 `core_v2/`
- **不要**在此写 CLI 逻辑——归 `cli/`
- **不要**把测试代码放在这里——归 `tests/op/`
- **不要**在此写其他场景层（如图优化）的逻辑——应新建平级场景目录，参考 `op/` 的组织模式

## 参考

- `docs/v2/KernelGen.md`、`docs/v2/KernelDesigner.md`、`docs/v2/KernelAgent.md`
- `docs/v2/OpOptimizer.md`
