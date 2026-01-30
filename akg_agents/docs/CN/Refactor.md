#  模块迁移重构文档

## 重构背景

原有代码库中算子生成相关的代码散布在多个目录中（`config/`、`core/evolve.py`、`core/adaptive_search/`、`utils/evolve/`、`resources/`、`workflows/`、`utils/langgraph/` 等）。为了实现更好的模块化和支持未来的通用场景，需要将代码重新组织：

1. **通用框架层** (`core_v2/langgraph_base/`) - 领域无关的 LangGraph 基础设施
2. **算子专用层** (`op/`) - 算子生成场景的所有专用逻辑

## 新架构

```
akg_agents/
├── core_v2/
│   └── langgraph_base/              # 通用 LangGraph 框架
│       ├── __init__.py
│       ├── base_state.py            # 通用状态基类 BaseState
│       ├── base_workflow.py         # 通用工作流基类 BaseWorkflow
│       ├── base_task.py             # 通用任务基类 BaseLangGraphTask
│       ├── base_routers.py          # 通用路由工具函数
│       ├── visualizer.py            # 可视化工具 WorkflowVisualizer
│       └── node_tracker.py          # 节点追踪装饰器 track_node
│
├── op/                             # 算子场景专用 (完整移动)
│   ├── __init__.py
│   ├── config/                      # 算子配置文件 (从 config/ 迁移)
│   │   ├── __init__.py
│   │   ├── config_validator.py
│   │   └── *.yaml                   # 所有配置文件
│   │
│   ├── evolve.py                    # 进化式生成 (从 core/evolve.py 迁移)
│   │
│   ├── adaptive_search/             # 自适应搜索 (从 core/adaptive_search/ 迁移)
│   │   ├── __init__.py
│   │   ├── adaptive_search.py
│   │   ├── controller.py
│   │   ├── success_db.py
│   │   ├── task_generator.py
│   │   ├── task_pool.py
│   │   └── ucb_selector.py
│   │
│   ├── langgraph_op/                # 算子 LangGraph 组件
│   │   ├── __init__.py
│   │   ├── state.py                 # KernelGenState (继承 BaseState)
│   │   ├── nodes.py                 # NodeFactory (算子节点)
│   │   ├── routers.py               # RouterFactory (算子路由)
│   │   ├── task.py                  # LangGraphTask (继承 BaseLangGraphTask)
│   │   ├── conversational_state.py  # 对话式算子生成状态
│   │   └── op_task_builder_state.py # OpTaskBuilder 状态
│   │
│   ├── workflows/                   # 算子工作流
│   │   ├── __init__.py
│   │   ├── base_workflow.py         # OpBaseWorkflow (继承 BaseWorkflow)
│   │   ├── default_workflow.py      # Designer → Coder ↔ Verifier
│   │   ├── coder_only_workflow.py   # Coder → Verifier
│   │   ├── verifier_only_workflow.py
│   │   ├── connect_all_workflow.py
│   │   └── op_task_builder_workflow.py
│   │
│   ├── utils/                       # 算子专用工具 (从 utils/ 部分迁移)
│   │   ├── __init__.py
│   │   ├── evolve/                  # 进化工具 (从 utils/evolve/ 迁移)
│   │   │   ├── __init__.py
│   │   │   ├── evolution_core.py
│   │   │   ├── evolution_processors.py
│   │   │   ├── evolution_utils.py
│   │   │   ├── runner_manager.py
│   │   │   └── result_collector.py
│   │   ├── handwrite_loader.py
│   │   ├── case_generator.py
│   │   ├── space_sampler.py
│   │   ├── workflow_manager.py
│   │   └── ... (其他算子专用工具)
│   │
│   ├── verifier/                    # Verifier (从 core/verifier/ 迁移)
│   │   ├── __init__.py
│   │   ├── kernel_verifier.py
│   │   ├── profiler.py
│   │   └── adapters/                # DSL/Backend/Framework 适配器
│   │       ├── factory.py
│   │       ├── dsl/
│   │       ├── backend/
│   │       └── framework/
│   │
│   ├── tools/                       # 运行工具脚本 (从 akg_agents/tools/ 迁移)
│   │   ├── __init__.py
│   │   ├── run_single_evolve.py
│   │   ├── run_batch_evolve.py
│   │   ├── run_single_adaptive_search.py
│   │   ├── run_batch_adaptive_search.py
│   │   ├── batch_run/
│   │   ├── random_cases_test/
│   │   └── op_task_builder/
│   │
│   └── resources/                   # 算子资源文件 (从 resources/ 迁移)
│       ├── docs/                    # 文档 (API, 建议, 示例)
│       ├── prompts/                 # Prompt 模板
│       ├── skills/                  # 技能定义
│       └── templates/               # 验证模板
│
├── tests/
│   ├── op/                         # 算子测试 (从 tests/ 迁移)
│   │   ├── st/                      # 系统测试
│   │   ├── ut/                      # 单元测试
│   │   ├── bench/                   # 性能测试
│   │   └── resources/               # 测试资源
│   └── v2/                          # 通用测试 (保留)
│
└── [已删除的旧目录]                   # 已迁移到 op/，旧位置已删除
    ├── config/                      # → op/config
    ├── core/evolve.py               # → op/evolve.py
    ├── core/adaptive_search/        # → op/adaptive_search/
    ├── core/verifier/               # → op/verifier/
    ├── core/langgraph_task.py       # → op/langgraph_op/task.py
    ├── utils/evolve/                # → op/utils/evolve/
    ├── resources/                   # → op/resources/
    ├── workflows/                   # → op/workflows/
    └── utils/langgraph/             # → op/langgraph_op/
```

## 继承关系

```
core_v2/langgraph_base/                 op/langgraph_op/
┌─────────────────┐                ┌──────────────────┐
│   BaseState     │ ◄──────────────│  KernelGenState  │
└─────────────────┘                └──────────────────┘

┌─────────────────┐                ┌──────────────────┐
│  BaseWorkflow   │ ◄──────────────│  OpBaseWorkflow  │
└─────────────────┘                └──────────────────┘
                                          ▲
                                          │
                        ┌─────────────────┼─────────────────┐
                        │                 │                 │
               ┌────────┴──────┐ ┌───────┴───────┐ ┌──────┴──────┐
               │DefaultWorkflow│ │CoderOnlyWf    │ │ConnectAllWf │
               └───────────────┘ └───────────────┘ └─────────────┘

┌─────────────────────┐            ┌──────────────────┐
│ BaseLangGraphTask   │ ◄──────────│  LangGraphTask   │
└─────────────────────┘            └──────────────────┘
```

## 核心组件说明

### 1. 通用框架 (core_v2/langgraph_base/)

#### BaseState
```python
class BaseState(TypedDict, total=False):
    """通用状态基类，仅包含框架级字段"""
    task_id: str
    task_label: str
    session_id: str
    iteration: int
    step_count: int
    max_iterations: int
    agent_history: Annotated[List[str], add]
    success: bool
    error_message: Optional[str]
```

#### BaseWorkflow
```python
class BaseWorkflow(ABC, Generic[StateType]):
    """通用工作流基类"""
    def __init__(self, config: dict, trace=None):
        ...
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """子类实现：构建图结构"""
        pass
    
    def compile(self):
        """编译图"""
        return self.build_graph().compile()
```

#### BaseLangGraphTask
```python
class BaseLangGraphTask(ABC):
    """通用任务基类"""
    
    @abstractmethod
    def _init_workflow(self):
        """子类实现：初始化工作流"""
        pass
    
    @abstractmethod
    def _prepare_initial_state(self, init_info) -> Dict[str, Any]:
        """子类实现：准备初始状态"""
        pass
    
    async def run(self, init_info=None) -> Tuple[bool, dict]:
        """执行任务"""
        ...
```

### 2. 算子专用层 (op/)

#### KernelGenState
```python
class KernelGenState(BaseState, total=False):
    """算子生成状态，继承通用状态"""
    # 算子基础信息
    op_name: str
    task_desc: str
    dsl: str
    framework: str
    backend: str
    arch: str
    
    # Agent 输出
    designer_code: Optional[str]
    coder_code: Optional[str]
    verifier_result: bool
    verifier_error: str
    ...
```

#### LangGraphTask
```python
class LangGraphTask(BaseLangGraphTask):
    """算子生成任务"""
    def __init__(self, op_name, task_desc, ...):
        # 算子专用参数
        self.op_name = op_name
        ...
        super().__init__(task_id, config, workflow)
        self._init_agents()
        self._init_workflow()
```

## 迁移指南

### 旧路径 → 新路径

#### LangGraph 模块

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `utils.langgraph.state.KernelGenState` | `op.langgraph_op.state.KernelGenState` | 算子状态 |
| `utils.langgraph.nodes.NodeFactory` | `op.langgraph_op.nodes.NodeFactory` | 算子节点 |
| `utils.langgraph.routers.RouterFactory` | `op.langgraph_op.routers.RouterFactory` | 算子路由 |
| `utils.langgraph.visualizer` | `core_v2.langgraph_base.visualizer` | 通用可视化 |
| `utils.langgraph.node_tracker` | `core_v2.langgraph_base.node_tracker` | 通用追踪 |
| `workflows.BaseWorkflow` | `op.workflows.base_workflow.OpBaseWorkflow` | 算子工作流基类 |
| `workflows.DefaultWorkflow` | `op.workflows.default_workflow.DefaultWorkflow` | 默认工作流 |
| `core.langgraph_task.LangGraphTask` | `op.langgraph_op.task.LangGraphTask` | 算子任务 |

#### 配置模块

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `config.config_validator.ConfigValidator` | `op.config.config_validator.ConfigValidator` | 配置校验器 |
| `config.config_validator.load_config` | `op.config.config_validator.load_config` | 配置加载 |

#### 进化与搜索模块

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `core.evolve.evolve` | `op.evolve.evolve` | 进化式生成 |
| `core.adaptive_search.*` | `op.adaptive_search.*` | 自适应搜索 |
| `utils.evolve.*` | `op.utils.evolve.*` | 进化工具 |

#### Verifier 模块

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `core.verifier.kernel_verifier.KernelVerifier` | `op.verifier.kernel_verifier.KernelVerifier` | Kernel 验证器 |
| `core.verifier.profiler.*` | `op.verifier.profiler.*` | 性能分析器 |
| `core.verifier.adapters.factory.*` | `op.verifier.adapters.factory.*` | 适配器工厂 |

#### 工具模块

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `akg_agents/tools/run_single_evolve.py` | `op/tools/run_single_evolve.py` | 单任务进化 |
| `akg_agents/tools/run_batch_evolve.py` | `op/tools/run_batch_evolve.py` | 批量进化 |
| `akg_agents/tools/run_single_adaptive_search.py` | `op/tools/run_single_adaptive_search.py` | 单任务自适应搜索 |
| `akg_agents/tools/run_batch_adaptive_search.py` | `op/tools/run_batch_adaptive_search.py` | 批量自适应搜索 |
| `utils.handwrite_loader` | `op.utils.handwrite_loader` | 手写代码加载器 |
| `utils.case_generator` | `op.utils.case_generator` | 用例生成器 |
| `utils.space_sampler` | `op.utils.space_sampler` | 空间采样器 |

#### 资源模块

| 旧路径 | 新路径 | 说明 |
|--------|--------|------|
| `resources.*` | `op.resources.*` | 资源文件 |

### 代码迁移示例

**旧代码：**
```python
from akg_agents.utils.langgraph.state import KernelGenState
from akg_agents.workflows.default_workflow import DefaultWorkflow
from akg_agents.core.langgraph_task import LangGraphTask
from akg_agents.config.config_validator import load_config
from akg_agents.core.evolve import evolve
from akg_agents.core.adaptive_search import adaptive_search
from akg_agents.core_v2.langgraph.base_state import BaseState
```

**新代码：**
```python
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.workflows.default_workflow import DefaultWorkflow
from akg_agents.op.langgraph_op.task import LangGraphTask
from akg_agents.op.config import load_config
from akg_agents.op.evolve import evolve
from akg_agents.op.adaptive_search import adaptive_search
from akg_agents.core_v2.langgraph_base.base_state import BaseState
```

### 注意事项

**重要**: 旧路径已删除，不再支持向后兼容。所有代码必须使用新的 `op/` 路径。

如果遇到 `ModuleNotFoundError`，请检查导入路径是否已更新到新位置。

## 扩展新场景

### 示例: 文档修复场景 (examples/build_a_simple_workflow)

`examples/build_a_simple_workflow/` 是一个完整的示例，展示了如何基于 `core_v2/langgraph_base/` 构建新场景：

```
examples/build_a_simple_workflow/
├── __init__.py
├── state.py                  # DocFixerState (继承 BaseState)
├── workflow.py               # DocFixerWorkflow (继承 BaseWorkflow)
├── task.py                   # DocFixerTask (继承 BaseLangGraphTask)
├── run_example.py            # 运行示例脚本
├── agents/                   # 场景专用 Agent
│   ├── __init__.py
│   ├── typo_fixer.py         # 错别字修复 Agent
│   └── beautifier.py         # 文档美化 Agent
└── resources/
    └── sample_doc_with_typos.md  # 示例输入文档
```

**工作流程**: `TypoFixer(修复错别字)` → `Beautifier(美化格式)` → `完成`

**运行示例**:
```bash
# 使用内置示例文档
python examples/build_a_simple_workflow/run_example.py

# 使用自定义文档
python examples/build_a_simple_workflow/run_example.py --input /path/to/doc.md

# 保存输出结果
python examples/build_a_simple_workflow/run_example.py --output /path/to/output.md
```

### 如何添加新场景

1. 创建新的场景目录（如 `examples/my_scenario/` 或 `my_package/my_scenario/`）
2. 继承 `core_v2/langgraph_base/` 的基类
3. 实现场景专用的状态、节点、工作流

```python
# my_scenario/state.py
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "python"))

from akg_agents.core_v2.langgraph_base.base_state import BaseState

class MyState(BaseState, total=False):
    """自定义场景状态"""
    input_content: str
    processed_content: str
    ...

# my_scenario/workflow.py
from akg_agents.core_v2.langgraph_base.base_workflow import BaseWorkflow

class MyWorkflow(BaseWorkflow[MyState]):
    def build_graph(self) -> StateGraph:
        workflow = StateGraph(MyState)
        # 添加节点和边
        workflow.add_node("agent1", self.agent1_node)
        workflow.add_node("agent2", self.agent2_node)
        workflow.add_edge("agent1", "agent2")
        workflow.add_edge("agent2", END)
        workflow.set_entry_point("agent1")
        return workflow

# my_scenario/task.py
from akg_agents.core_v2.langgraph_base.base_task import BaseLangGraphTask

class MyTask(BaseLangGraphTask):
    def _init_workflow(self):
        self.workflow = MyWorkflow(config=self.config)
        self.app = self.workflow.compile()
    
    def _prepare_initial_state(self, init_info):
        return {"task_id": self.task_id, ...}
```

## 文件清单

### 新增文件示例

| 路径 | 说明 |
|------|------|
| `core_v2/langgraph_base/__init__.py` | 模块导出 |
| `core_v2/langgraph_base/base_state.py` | 通用状态 |
| `core_v2/langgraph_base/base_workflow.py` | 通用工作流 |
| `core_v2/langgraph_base/base_task.py` | 通用任务 |
| `core_v2/langgraph_base/base_routers.py` | 通用路由 |
| `core_v2/langgraph_base/visualizer.py` | 可视化 |
| `core_v2/langgraph_base/node_tracker.py` | 节点追踪 |
| `op/__init__.py` | 算子模块 |
| `op/langgraph_op/__init__.py` | 算子 LangGraph |
| `op/langgraph_op/state.py` | 算子状态 |
| `op/langgraph_op/nodes.py` | 算子节点 |
| `op/langgraph_op/routers.py` | 算子路由 |
| `op/langgraph_op/task.py` | 算子任务 |
| `op/langgraph_op/conversational_state.py` | 对话状态 |
| `op/langgraph_op/op_task_builder_state.py` | TaskBuilder 状态 |
| `op/workflows/__init__.py` | 算子工作流 |
| `op/workflows/base_workflow.py` | 算子工作流基类 |
| `op/workflows/default_workflow.py` | 默认工作流 |
| `op/workflows/coder_only_workflow.py` | Coder-only |
| `op/workflows/verifier_only_workflow.py` | Verifier-only |
| `op/workflows/connect_all_workflow.py` | 全连接 |
| `op/workflows/op_task_builder_workflow.py` | TaskBuilder |

### 已删除文件示例

| 旧路径 | 新位置 |
|--------|--------|
| `config/` | `op/config/` |
| `resources/` | `op/resources/` |
| `workflows/` | `op/workflows/` |
| `utils/langgraph/` | `op/langgraph_op/` |
| `utils/evolve/` | `op/utils/evolve/` |
| `core/evolve.py` | `op/evolve.py` |
| `core/adaptive_search/` | `op/adaptive_search/` |
| `core/verifier/` | `op/verifier/` |
| `core/langgraph_task.py` | `op/langgraph_op/task.py` |

## 变更日志

- **2026-01-26**: LangGraph 模块重命名
  - 重命名 `core_v2/langgraph/` → `core_v2/langgraph_base/`（避免与 `langgraph` 包名冲突）
  - 重命名 `op/langgraph/` → `op/langgraph_op/`（避免与 `langgraph` 包名冲突）
  - 更新所有相关导入路径
  - 在 `run_single_adaptive_search.py` 和 `runner_manager.py` 中添加 `agent_model_config` 自动补齐逻辑

- **2026-01-26**: 路径修复完成
  - 修复所有 yaml 配置文件中的 `docs_dir` 路径：`resources/` → `op/resources/`
  - 修复所有 yaml 配置文件中的 `workflow_config_path` 路径：`config/` → `op/config/`
  - 修复 `kernel_verifier.py` 中的模板路径
  - 修复 `sub_agent_registry.py` 中的 evolve/adaptive_search 配置路径
  - 修复 `core/skills/loader.py`, `core/agent/react_agent.py`, `core/agent/coder.py` 中的资源路径
  - 修复 `cli/constants.py` 中的 logo 路径
  - 修复 `utils/hardware_utils.py` 中的硬件文档路径
  - 修复 `core/checker/code_checker.py` 中的文档路径
  - 修复 `op/utils/workflow_manager.py` 中的 workflow 配置路径解析
  - 修复 `core/worker/local_worker.py` 中的 verifier 导入路径（`..verifier.profiler_utils` → `.verifier.profiler_utils`）
  - 修复 `tests/op/utils.py` 中的路径计算（测试文件移到 `tests/op/` 后需要多上溯一级）
  - 修复 `tests/op/` 下所有测试文件中的配置文件路径
  - 移动 `swft_docs_loader.py` 回 `utils/` 目录（供多个 agent 使用）

- **2026-01-26**: 完整迁移完成，删除所有向后兼容层
  - 更新 `core/`, `cli/`, `server/` 中所有旧导入路径到 `op/`
  - 更新 `tests/op/` 中所有测试文件的导入路径
  - 删除所有向后兼容层文件
  - 删除配置文件中的 `agent_model_config` (不再使用)

- **2026-01-26**: Verifier 与 Tools 迁移
  - 迁移 `core/verifier/` → `op/verifier/`
  - 迁移 `akg_agents/tools/` → `op/tools/` (运行脚本)

- **2026-01-26**: 核心模块迁移
  - 迁移 `config/` → `op/config/`
  - 迁移 `core/evolve.py` → `op/evolve.py`
  - 迁移 `core/adaptive_search/` → `op/adaptive_search/`
  - 迁移 `utils/evolve/` → `op/utils/evolve/`
  - 迁移算子专用 utils 文件 → `op/utils/`
  - 迁移 `resources/` → `op/resources/`
  - 迁移 `workflows/` → `op/workflows/`
  - 迁移 `utils/langgraph/` → `op/langgraph_op/`
  - 迁移 `tests/st/`, `tests/ut/`, `tests/bench/`, `tests/resources/` → `tests/op/`

- **2025-01-26**: 初始重构
  - 创建 `core_v2/langgraph_base/` 通用框架（原名 `core_v2/langgraph/`，后重命名避免包名冲突）
  - 创建 `op/` 算子专用层 (LangGraph 部分位于 `op/langgraph_op/`)

