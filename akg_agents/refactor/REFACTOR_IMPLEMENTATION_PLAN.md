# AIKG 重构实施方案（测试驱动）

> 基于最终架构的渐进式迁移方案
> 更新时间: 2026-01-20

## 一、代码复用分析

### 1.1 可直接复用（✅ 不改或微改）

| 模块 | 现有代码 | 是否需要改动 | 说明 |
|-----|---------|------------|------|
| **LangGraph Workflows** | `workflows/base_workflow.py`<br>`workflows/default_workflow.py`<br>`workflows/coder_only_workflow.py` | ❌ 不改 | 作为 tools 被 KernelAgent 调用 |
| **SubAgents** | `core/sub_agent_registry.py`<br>`agents/designer.py`<br>`agents/coder.py`<br>`agents/verifier.py` | ❌ 不改 | 作为 tools 被 KernelAgent 调用 |
| **基础 Tools** | `core/tools/basic_tools.py`<br>`core/tools/file_tools.py` | ❌ 不改 | 直接复用 |
| **WorkerManager** | `core/worker/manager.py` | ❌ 不改 | 直接复用 |
| **LLMClient** | `core/llm/client.py` | ⚠️ 微改 | 可能需要支持结构化输出 |
| **Trace** | `core/trace.py` | ⚠️ 微改 | 需要增强为 Trace System |

### 1.2 需要新开发（🆕 全新模块）

| 模块 | 功能 | 优先级 | 预计时间 |
|-----|------|-------|---------|
| **KernelAgent** | 主循环 + OpenAI function calling | P0 | 3-4天 |
| **PlanAgent** | 初始规划 + 每10步重新规划 | P0 | 2-3天 |
| **FileSystemState** | FileSystem 状态管理 | P0 | 2-3天 |
| **Tool 封装层** | Workflow/SubAgent → tools | P0 | 2天 |
| **ask_user Tool** | HITL 支持 | P1 | 1天 |
| **Trace System** | 增强版 Trace | P1 | 2-3天 |

### 1.3 需要适配（🔧 改动较大）

| 模块 | 改动内容 | 优先级 | 预计时间 |
|-----|---------|-------|---------|
| **CLI 入口** | 适配新的 KernelAgent | P0 | 1-2天 |
| **配置系统** | 支持 FileSystem 路径等 | P1 | 1天 |

---

## 二、分批开发计划（测试驱动）

### 阶段0: 准备阶段（1周）

**目标**: 搭建测试框架，确保现有代码正常工作

#### 任务清单

1. **创建测试框架**
   ```bash
   core_v2/
   ├── __init__.py
   ├── agents/
   │   ├── __init__.py
   │   └── kernel_agent.py  # 空框架
   ├── plan/
   │   ├── __init__.py
   │   └── plan_agent.py    # 空框架
   └── state/
       ├── __init__.py
       └── file_system_state.py  # 空框架
   
   tests/unit/core_v2/
   └── agents/
       └── test_kernel_agent.py  # 测试框架
   ```

2. **验证现有代码可用**
   - ✅ 测试现有 Workflows 可正常编译执行
   - ✅ 测试现有 SubAgents 可正常执行
   - ✅ 测试基础 Tools 正常工作

3. **建立回归测试基线**
   ```bash
   # 跑现有所有测试
   pytest tests/ -v
   
   # 记录基线结果
   pytest tests/ --baseline
   ```

**交付物**:
- ✅ 空的 `core_v2/` 目录结构
- ✅ 测试框架搭建完成
- ✅ 现有代码测试全部通过

---

### 阶段1: FileSystemState（第1-2周）

**目标**: 实现 FileSystem 状态管理（最基础的依赖）

#### 1.1 TDD 开发流程

**步骤1: 写测试（Red）**

```python
# tests/unit/core_v2/state/test_file_system_state.py

def test_create_conversation():
    """测试创建会话目录"""
    fs = FileSystemState(task_id="test_task_001")
    
    # 验证目录创建
    assert fs.conversation_dir.exists()
    assert (fs.conversation_dir / "trace.json").exists()
    assert (fs.conversation_dir / "state.json").exists()
    assert (fs.conversation_dir / "actions").is_dir()
    assert (fs.conversation_dir / "code").is_dir()

def test_save_and_load_state():
    """测试状态保存和加载"""
    fs = FileSystemState(task_id="test_task_002")
    
    # 保存状态
    state = {"op_name": "relu", "step": 5}
    fs.save_state(state)
    
    # 加载状态
    loaded_state = fs.load_state()
    assert loaded_state["op_name"] == "relu"
    assert loaded_state["step"] == 5

def test_save_action():
    """测试保存 action"""
    fs = FileSystemState(task_id="test_task_003")
    
    action = {
        "turn": 1,
        "tool_name": "call_designer",
        "tool_args": {"task_desc": "ReLU"},
        "result": {"design": "..."}
    }
    
    fs.save_action(action)
    
    # 验证文件创建
    action_file = fs.conversation_dir / "actions" / "turn_001_call_designer.json"
    assert action_file.exists()
    
    # 验证内容
    loaded_action = json.loads(action_file.read_text())
    assert loaded_action["tool_name"] == "call_designer"

def test_save_thinking():
    """测试保存 PlanAgent 的 thinking"""
    fs = FileSystemState(task_id="test_task_004")
    
    thinking = {
        "plan": "1. 设计算法 2. 生成代码 3. 验证",
        "todo_list": ["设计", "编码", "验证"],
        "next_steps": ["call_designer"]
    }
    
    fs.save_thinking(thinking)
    
    # 验证文件
    thinking_file = fs.conversation_dir / "thinking.json"
    assert thinking_file.exists()
    
    # 加载验证
    loaded = fs.load_thinking()
    assert loaded["plan"] == thinking["plan"]

def test_get_state_summary():
    """测试获取状态摘要"""
    fs = FileSystemState(task_id="test_task_005")
    
    # 模拟一些状态
    fs.save_state({"op_name": "relu", "step": 3})
    fs.save_action({
        "turn": 1,
        "tool_name": "call_designer",
        "result": {"design": "..."}
    })
    
    # 获取摘要
    summary = fs.get_state_summary()
    
    assert "op_name" in summary
    assert "recent_actions" in summary
    assert len(summary["recent_actions"]) == 1
```

**步骤2: 实现代码（Green）**

```python
# core_v2/state/file_system_state.py

from pathlib import Path
import json
from typing import Dict, Any, List
from datetime import datetime

class FileSystemState:
    """FileSystem 状态管理
    
    目录结构:
    ~/.akg/conversations/{task_id}/
    ├─ trace.json
    ├─ state.json
    ├─ thinking.json
    ├─ actions/
    ├─ code/
    └─ logs/
    """
    
    def __init__(self, task_id: str, base_dir: Path = None):
        self.task_id = task_id
        self.base_dir = base_dir or Path.home() / ".aikg" / "conversations"
        self.conversation_dir = self.base_dir / task_id
        
        # 创建目录结构
        self._init_directories()
    
    def _init_directories(self):
        """初始化目录结构"""
        self.conversation_dir.mkdir(parents=True, exist_ok=True)
        (self.conversation_dir / "actions").mkdir(exist_ok=True)
        (self.conversation_dir / "code").mkdir(exist_ok=True)
        (self.conversation_dir / "logs").mkdir(exist_ok=True)
        
        # 初始化空文件
        for filename in ["trace.json", "state.json", "thinking.json"]:
            filepath = self.conversation_dir / filename
            if not filepath.exists():
                filepath.write_text(json.dumps({}, indent=2))
    
    def save_state(self, state: Dict[str, Any]):
        """保存当前状态"""
        state["updated_at"] = datetime.now().isoformat()
        filepath = self.conversation_dir / "state.json"
        filepath.write_text(json.dumps(state, indent=2, ensure_ascii=False))
    
    def load_state(self) -> Dict[str, Any]:
        """加载状态"""
        filepath = self.conversation_dir / "state.json"
        if not filepath.exists():
            return {}
        return json.loads(filepath.read_text())
    
    def save_action(self, action: Dict[str, Any]):
        """保存 action"""
        turn = action.get("turn", 0)
        tool_name = action.get("tool_name", "unknown")
        filename = f"turn_{turn:03d}_{tool_name}.json"
        filepath = self.conversation_dir / "actions" / filename
        
        action["saved_at"] = datetime.now().isoformat()
        filepath.write_text(json.dumps(action, indent=2, ensure_ascii=False))
    
    def save_thinking(self, thinking: Dict[str, Any]):
        """保存 PlanAgent 的 thinking"""
        thinking["updated_at"] = datetime.now().isoformat()
        filepath = self.conversation_dir / "thinking.json"
        filepath.write_text(json.dumps(thinking, indent=2, ensure_ascii=False))
    
    def load_thinking(self) -> Dict[str, Any]:
        """加载 thinking"""
        filepath = self.conversation_dir / "thinking.json"
        if not filepath.exists():
            return {}
        return json.loads(filepath.read_text())
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取状态摘要（用于构建上下文）"""
        state = self.load_state()
        thinking = self.load_thinking()
        
        # 获取最近的 actions
        actions_dir = self.conversation_dir / "actions"
        action_files = sorted(actions_dir.glob("*.json"))[-5:]  # 最近5个
        recent_actions = []
        for action_file in action_files:
            action = json.loads(action_file.read_text())
            recent_actions.append({
                "turn": action.get("turn"),
                "tool_name": action.get("tool_name"),
                "result_summary": str(action.get("result", {}))[:200]
            })
        
        return {
            "current_state": state,
            "current_plan": thinking.get("plan", ""),
            "recent_actions": recent_actions
        }
```

**步骤3: 重构（Refactor）**

- 优化代码结构
- 添加错误处理
- 添加日志

**步骤4: 集成测试**

```python
def test_file_system_state_integration():
    """集成测试：完整的状态管理流程"""
    fs = FileSystemState(task_id="integration_test")
    
    # 1. 初始状态
    fs.save_state({"op_name": "relu", "step": 0})
    
    # 2. 保存 thinking
    fs.save_thinking({
        "plan": "设计 → 编码 → 验证",
        "todo_list": ["设计", "编码", "验证"]
    })
    
    # 3. 保存多个 actions
    for i in range(3):
        fs.save_action({
            "turn": i,
            "tool_name": f"action_{i}",
            "result": {"data": i}
        })
    
    # 4. 获取摘要
    summary = fs.get_state_summary()
    
    assert summary["current_state"]["op_name"] == "relu"
    assert "设计" in summary["current_plan"]
    assert len(summary["recent_actions"]) == 3
```

**交付物**:
- ✅ `core_v2/state/file_system_state.py` 实现完成
- ✅ 所有单元测试通过
- ✅ 集成测试通过

---

### 阶段2: PlanAgent（第2-3周）

**目标**: 实现 PlanAgent（初始规划 + 动态触发重新规划）

**PlanAgent 触发条件（三种）**:
1. **定期触发**: 每 10 步自动触发（兜底）
2. **ask_user 触发**: 用户反馈后立即触发（保险机制）
3. **主动检测**: 每轮开始前检查 plan 是否过期（规则检测）

#### 2.1 TDD 开发流程

**步骤1: 写测试（Red）**

```python
# tests/unit/core_v2/plan/test_plan_agent.py

def test_initial_plan():
    """测试初始规划"""
    plan_agent = PlanAgent(llm_client=mock_llm_client)
    
    user_request = "生成一个 ReLU 算子"
    current_state = {}
    
    plan = await plan_agent.plan(user_request, current_state)
    
    assert "plan" in plan
    assert "todo_list" in plan
    assert "next_steps" in plan
    assert len(plan["next_steps"]) > 0

def test_replan():
    """测试重新规划"""
    plan_agent = PlanAgent(llm_client=mock_llm_client)
    
    user_request = "生成一个 ReLU 算子"
    action_history = [
        {"tool_name": "call_designer", "result": {"design": "..."}},
        {"tool_name": "call_coder", "result": {"code": "..."}}
    ]
    current_state = {"op_name": "relu", "step": 10}
    
    plan = await plan_agent.replan(user_request, action_history, current_state)
    
    assert "plan" in plan
    assert "progress_analysis" in plan
    assert "next_steps" in plan

def test_plan_prompt_construction():
    """测试规划 prompt 构建"""
    plan_agent = PlanAgent(llm_client=mock_llm_client)
    
    prompt = plan_agent._build_plan_prompt(
        user_request="生成 ReLU",
        current_state={}
    )
    
    assert "任务" in prompt
    assert "生成 ReLU" in prompt
    assert "计划" in prompt

def test_is_plan_outdated_by_keywords():
    """测试通过关键词检测 plan 是否过期"""
    plan_agent = PlanAgent(llm_client=mock_llm_client)
    
    plan = {"todo_list": ["生成 ReLU"], "next_steps": ["call_coder"]}
    
    # 包含关键词 "改成" → plan 过期
    action_history = [
        {"action": "ask_user", "result": "用户回复: 不要 ReLU 了，改成 GELU"}
    ]
    assert plan_agent.is_plan_outdated(plan, action_history) == True
    
    # 不包含关键词 → plan 未过期
    action_history = [
        {"action": "ask_user", "result": "用户回复: 确认"}
    ]
    assert plan_agent.is_plan_outdated(plan, action_history) == False

def test_is_plan_outdated_by_deviation():
    """测试通过执行偏离检测 plan 是否过期"""
    plan_agent = PlanAgent(llm_client=mock_llm_client)
    
    plan = {
        "todo_list": ["生成 ReLU"],
        "next_steps": ["call_designer", "call_coder", "call_verifier"]
    }
    
    # 实际执行严重偏离计划 → plan 过期
    action_history = [
        {"action": "read_file"},
        {"action": "ask_user"},
        {"action": "write_file"}  # 与计划完全不符
    ]
    assert plan_agent.is_plan_outdated(plan, action_history) == True
```

**步骤2: 实现代码（Green）**

```python
# core_v2/plan/plan_agent.py

class PlanAgent:
    """规划 Agent
    
    职责:
    1. 初始规划 (plan)
    2. 每10步重新规划 (replan)
    3. 固化计划到 FileSystem
    """
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
    
    async def plan(self, user_request: str, current_state: Dict) -> Dict:
        """初始规划"""
        prompt = self._build_plan_prompt(user_request, current_state)
        
        response = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="你是一个任务规划专家",
            response_format=PlanSchema  # JSON Schema
        )
        
        return response
    
    async def replan(self, 
                    user_request: str,
                    action_history: List[Dict],
                    current_state: Dict) -> Dict:
        """重新规划"""
        prompt = self._build_replan_prompt(
            user_request, 
            action_history, 
            current_state
        )
        
        response = await self.llm_client.chat(
            messages=[{"role": "user", "content": prompt}],
            system_prompt="你是一个任务规划专家",
            response_format=PlanSchema
        )
        
        return response
    
    def is_plan_outdated(self, plan: Dict, action_history: List[Dict]) -> bool:
        """判断 plan 是否过期
        
        检测规则:
        1. 用户反馈包含关键词（"不要"、"改成"、"换成"、"重新"、"取消"、"修改"）
        2. 实际执行与计划严重偏离（超过 50%）
        """
        # 规则1: 检查最近的 ask_user 结果
        recent_ask_user = [
            a for a in action_history[-3:]  # 最近 3 步
            if a.get('action') == 'ask_user'
        ]
        
        if recent_ask_user:
            user_reply = recent_ask_user[-1].get('result', '')
            
            # 关键词检测
            change_keywords = ['不要', '改成', '换成', '重新', '取消', '修改']
            if any(kw in user_reply for kw in change_keywords):
                return True  # plan 过期
        
        # 规则2: 检查执行偏离
        planned_actions = plan.get('next_steps', [])[:len(action_history)]
        actual_actions = [a.get('action') for a in action_history]
        
        if len(planned_actions) > 0:
            mismatch_count = sum(
                1 for p, a in zip(planned_actions, actual_actions) 
                if p != a
            )
            if mismatch_count > len(action_history) * 0.5:  # 超过 50%
                return True
        
        return False
    
    def _build_plan_prompt(self, user_request: str, current_state: Dict) -> str:
        """构建规划 prompt"""
        return f"""
# 任务规划

## 用户需求
{user_request}

## 当前状态
{json.dumps(current_state, indent=2, ensure_ascii=False)}

## 请制定执行计划

输出 JSON 格式:
{{
    "plan": "整体计划概述",
    "todo_list": ["任务1", "任务2", "任务3"],
    "next_steps": ["下一步具体动作1", "下一步具体动作2"]
}}
"""
    
    def _build_replan_prompt(self, 
                            user_request: str,
                            action_history: List[Dict],
                            current_state: Dict) -> str:
        """构建重新规划 prompt"""
        return f"""
# 任务重新规划

## 用户需求
{user_request}

## 已完成的动作
{self._format_action_history(action_history)}

## 当前状态
{json.dumps(current_state, indent=2, ensure_ascii=False)}

## 请重新规划

分析进度，调整计划，输出 JSON:
{{
    "progress_analysis": "进度分析",
    "plan": "调整后的计划",
    "todo_list": ["剩余任务"],
    "next_steps": ["下一步动作"]
}}
"""
```

**交付物**:
- ✅ `core_v2/plan/plan_agent.py` 实现完成
- ✅ 所有单元测试通过

---

### 阶段3: Tool 封装层（第3周）

**目标**: 将 Workflow 和 SubAgent 封装成 tools

#### 3.1 TDD 开发流程

**步骤1: 写测试（Red）**

```python
# tests/unit/core_v2/tools/test_tool_factory.py

def test_create_workflow_tool():
    """测试创建 workflow tool"""
    mock_workflow = MockWorkflow()
    
    tool = ToolFactory.create_workflow_tool(
        name="standard",
        workflow=mock_workflow
    )
    
    assert tool.name == "use_standard_workflow"
    assert callable(tool)
    
    # 调用 tool
    result = await tool(task_desc="ReLU", op_name="relu")
    assert result["success"] == True
    assert result["workflow"] == "standard"

def test_create_subagent_tool():
    """测试创建 subagent tool"""
    mock_subagent = MockSubAgent("designer")
    
    tool = ToolFactory.create_subagent_tool(
        name="designer",
        subagent=mock_subagent
    )
    
    assert tool.name == "call_designer"
    assert callable(tool)
    
    # 调用 tool
    result = await tool(task_desc="ReLU")
    assert result["success"] == True
    assert result["subagent"] == "designer"

def test_create_all_tools():
    """测试创建所有 tools"""
    workflows = {"standard": MockWorkflow()}
    subagents = {"designer": MockSubAgent("designer")}
    basic_tools = [read_file_tool, write_file_tool]
    
    all_tools = ToolFactory.create_all_tools(
        workflows=workflows,
        subagents=subagents,
        basic_tools=basic_tools
    )
    
    # 验证数量
    assert len(all_tools) == 4  # 1 workflow + 1 subagent + 2 basic
    
    # 验证名称
    tool_names = [tool.name for tool in all_tools]
    assert "use_standard_workflow" in tool_names
    assert "call_designer" in tool_names
```

**步骤2: 实现代码（Green）**

```python
# core_v2/tools/tool_factory.py

from langchain.tools import tool
from typing import Dict, Any, List

class ToolFactory:
    """Tool 工厂：将 Workflow 和 SubAgent 封装成 tools"""
    
    @staticmethod
    def create_workflow_tool(name: str, workflow):
        """将 Workflow 封装成 tool"""
        
        @tool(f"use_{name}_workflow")
        async def workflow_tool(task_desc: str, op_name: str, **kwargs) -> Dict[str, Any]:
            """
            使用完整工作流（适合标准场景）
            
            Args:
                task_desc: 任务描述
                op_name: 算子名称
            
            说明: 此 workflow 会执行完整流程，无法中途干预
            适用: 需求清晰、流程标准的场景
            """
            # 执行 workflow (LangGraph)
            result = await workflow.ainvoke({
                "task_desc": task_desc,
                "op_name": op_name,
                **kwargs
            })
            
            return {
                "success": True,
                "workflow": name,
                "result": result
            }
        
        return workflow_tool
    
    @staticmethod
    def create_subagent_tool(name: str, subagent):
        """将 SubAgent 封装成 tool"""
        
        @tool(f"call_{name}")
        async def subagent_tool(**kwargs) -> Dict[str, Any]:
            """
            调用单个 SubAgent（适合复杂场景 + HITL）
            
            说明: 可以单步执行，配合 ask_user 实现 HITL
            适用: 需要讨论、多轮迭代的场景
            """
            success, result = await subagent.execute(**kwargs)
            return {
                "success": success,
                "subagent": name,
                "result": result
            }
        
        return subagent_tool
    
    @staticmethod
    def create_all_tools(
        workflows: Dict,
        subagents: Dict,
        basic_tools: List
    ) -> List:
        """创建所有 tools"""
        tools = []
        
        # 1. Workflow tools
        for name, workflow in workflows.items():
            tool = ToolFactory.create_workflow_tool(name, workflow)
            tools.append(tool)
        
        # 2. SubAgent tools
        for name, subagent in subagents.items():
            tool = ToolFactory.create_subagent_tool(name, subagent)
            tools.append(tool)
        
        # 3. 基础 tools
        tools.extend(basic_tools)
        
        return tools
```

**交付物**:
- ✅ `core_v2/tools/tool_factory.py` 实现完成
- ✅ 所有单元测试通过

---

### 阶段4: KernelAgent 主循环（第4-5周）

**目标**: 实现 KernelAgent 主循环（核心！）

#### 4.1 TDD 开发流程

**步骤1: 写测试（Red）**

```python
# tests/unit/core_v2/agents/test_kernel_agent.py

@pytest.mark.asyncio
async def test_kernel_agent_simple_workflow():
    """测试简单场景：调用 workflow"""
    # Mock 组件
    mock_llm = MockLLMClient()
    mock_workflow = MockWorkflow()
    mock_fs = MockFileSystemState()
    
    # 配置 LLM 返回 workflow tool call
    mock_llm.set_response({
        "tool_calls": [{
            "function": {
                "name": "use_standard_workflow",
                "arguments": json.dumps({
                    "task_desc": "ReLU",
                    "op_name": "relu"
                })
            }
        }]
    })
    
    # 创建 KernelAgent
    agent = KernelAgent(
        workflows={"standard": mock_workflow},
        subagents={},
        tools=[],
        plan_agent=mock_plan_agent,
        trace_system=mock_trace,
        llm_client=mock_llm,
        file_system_state=mock_fs
    )
    
    # 执行
    result = await agent.execute("生成一个 ReLU 算子")
    
    # 验证
    assert result["success"] == True
    assert mock_workflow.invoked == True

@pytest.mark.asyncio
async def test_kernel_agent_complex_hitl():
    """测试复杂场景：SubAgent + HITL"""
    # ... 类似上面的测试
    pass

@pytest.mark.asyncio
async def test_kernel_agent_10_steps_replan():
    """测试每10步触发 PlanAgent"""
    # ... 测试重新规划逻辑
    pass
```

**步骤2: 实现代码（Green）**

```python
# core_v2/agents/kernel_agent.py

class KernelAgent:
    """算子生成主 Agent"""
    
    def __init__(self, 
                 workflows: dict,
                 subagents: dict,
                 tools: list,
                 plan_agent,
                 trace_system,
                 llm_client,
                 file_system_state):
        
        self.workflows = {
            name: workflow.compile() 
            for name, workflow in workflows.items()
        }
        self.subagents = subagents
        
        # 创建所有 tools
        from core_v2.tools.tool_factory import ToolFactory
        self.all_tools = ToolFactory.create_all_tools(
            workflows=self.workflows,
            subagents=self.subagents,
            basic_tools=tools
        )
        
        self.plan_agent = plan_agent
        self.trace_system = trace_system
        self.llm_client = llm_client
        self.file_system_state = file_system_state
        
        self.action_history = []
        self.current_plan = None
    
    async def execute(self, user_request: str, max_turns: int = 100):
        """主执行循环"""
        # 1. 初始规划
        self.current_plan = await self.plan_agent.plan(
            user_request=user_request,
            current_state=self.file_system_state.get_state_summary()
        )
        
        # 保存 thinking
        self.file_system_state.save_thinking(self.current_plan)
        
        # 2. 主循环
        for turn in range(max_turns):
            # 2.1 构建上下文
            context = self._build_context(
                user_request=user_request,
                plan=self.current_plan,
                action_history=self.action_history,
                file_state=self.file_system_state.get_state_summary()
            )
            
            # 2.2 LLM 决策
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": "请输出下一个动作"}],
                system_prompt=context,
                tools=self.all_tools,
                tool_choice="required"
            )
            
            # 2.3 执行 tools
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # 执行 tool
                tool_func = self._get_tool(tool_name)
                result = await tool_func(**tool_args)
                
                # 记录
                action = {
                    "turn": turn,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result
                }
                
                self.action_history.append(action)
                self.file_system_state.save_action(action)
                self.trace_system.record_action(action)
                
                # 检查完成
                if tool_name == "finish":
                    return result
            
            # 2.4 每10步重新规划
            if (turn + 1) % 10 == 0:
                self.current_plan = await self.plan_agent.replan(
                    user_request=user_request,
                    action_history=self.action_history,
                    current_state=self.file_system_state.get_state_summary()
                )
                
                self.file_system_state.save_thinking(self.current_plan)
                self.action_history = []  # 清空
        
        return {"error": "Max turns reached"}
    
    def _build_context(self, user_request, plan, action_history, file_state):
        """构建上下文"""
        # ... (见前面的代码)
        pass
```

**步骤3: 集成测试**

```python
# tests/integration/test_kernel_agent_integration.py

@pytest.mark.integration
async def test_kernel_agent_end_to_end_simple():
    """端到端测试：简单场景"""
    # 使用真实的组件
    agent = create_real_kernel_agent()
    
    result = await agent.execute("生成一个 ReLU 算子")
    
    assert result["success"] == True
    # 验证生成的代码
    # 验证 FileSystem 状态
    # 验证 Trace 记录

@pytest.mark.integration
async def test_kernel_agent_end_to_end_complex():
    """端到端测试：复杂场景 + HITL"""
    # ... 测试复杂场景
    pass
```

**交付物**:
- ✅ `core_v2/agents/kernel_agent.py` 实现完成
- ✅ 所有单元测试通过
- ✅ 集成测试通过

---

### 阶段5: CLI 适配（第5-6周）

**目标**: 适配 CLI，支持新的 KernelAgent

#### 5.1 适配步骤

1. **创建新的 Executor**
   ```python
   # cli/runtime/kernel_agent_executor.py
   
   class KernelAgentExecutor:
       """KernelAgent 执行器"""
       
       def __init__(self, config):
           # 初始化 KernelAgent
           self.agent = self._create_kernel_agent(config)
       
       async def execute(self, user_request: str):
           """执行"""
           return await self.agent.execute(user_request)
   ```

2. **更新 CLI 入口**
   ```python
   # cli/main.py
   
   @cli.command("op")
   @click.option("--use-new-agent", is_flag=True, help="使用新的 KernelAgent")
   async def op_command(use_new_agent):
       if use_new_agent:
           executor = KernelAgentExecutor(config)
       else:
           executor = LocalExecutor(config)  # 旧的
       
       result = await executor.execute(user_request)
   ```

3. **测试**
   ```bash
   # 测试新 agent
   akg_cli op --use-new-agent
   
   # 测试旧 agent（回归测试）
   akg_cli op
   ```

**交付物**:
- ✅ CLI 适配完成
- ✅ 新旧 agent 都可工作
- ✅ 回归测试通过

---

## 三、测试策略

### 3.1 测试金字塔

```
        /\
       /  \
      /E2E \      ← 端到端测试（10%）
     /------\
    /        \
   / 集成测试  \   ← 集成测试（30%）
  /----------\
 /            \
/ 单元测试     \  ← 单元测试（60%）
/--------------\
```

### 3.2 测试类型

#### 单元测试（60%）

**目标**: 每个模块独立测试

```python
tests/unit/core_v2/
├── agents/
│   └── test_kernel_agent.py
├── plan/
│   └── test_plan_agent.py
├── state/
│   └── test_file_system_state.py
└── tools/
    └── test_tool_factory.py
```

**覆盖率要求**: > 80%

#### 集成测试（30%）

**目标**: 测试模块之间的交互

```python
tests/integration/
├── test_kernel_agent_integration.py    # KernelAgent + 所有组件
├── test_file_system_persistence.py     # FileSystem 持久化
└── test_plan_agent_integration.py      # PlanAgent + KernelAgent
```

#### 端到端测试（10%）

**目标**: 测试完整流程

```python
tests/e2e/
├── test_simple_operator.py    # 简单算子生成
├── test_complex_hitl.py        # 复杂场景 + HITL
└── test_multi_turn.py          # 多轮对话
```

### 3.3 回归测试

**每个阶段完成后**:

```bash
# 1. 跑所有新测试
pytest tests/unit/core_v2/ -v
pytest tests/integration/ -v

# 2. 跑所有旧测试（回归）
pytest tests/ -v --ignore=tests/unit/core_v2/

# 3. 对比基线
pytest tests/ --compare-baseline
```

---

## 四、里程碑与交付

### 里程碑1: 基础设施（第2周结束）

**交付物**:
- ✅ FileSystemState 实现 + 测试
- ✅ 测试覆盖率 > 80%

**验收标准**:
- 所有单元测试通过
- 可以创建、保存、加载状态

---

### 里程碑2: 规划能力（第3周结束）

**交付物**:
- ✅ PlanAgent 实现 + 测试
- ✅ Tool 封装层 实现 + 测试

**验收标准**:
- 可以生成初始规划
- 可以重新规划
- Workflow/SubAgent 可以作为 tools 调用

---

### 里程碑3: 核心 Agent（第5周结束）

**交付物**:
- ✅ KernelAgent 实现 + 测试
- ✅ 集成测试通过

**验收标准**:
- 可以执行简单场景（调用 workflow）
- 可以执行复杂场景（SubAgent + HITL）
- 每10步触发 PlanAgent

---

### 里程碑4: 完整功能（第6周结束）

**交付物**:
- ✅ CLI 适配完成
- ✅ 端到端测试通过
- ✅ 文档完善

**验收标准**:
- 新旧 agent 都可工作
- 回归测试全部通过
- 用户可以通过 CLI 使用新 agent

---

## 五、风险与应对

### 风险1: LLM 决策不稳定

**描述**: LLM 可能不总是返回正确的 tool calls

**应对**:
- Mock LLM 进行单元测试
- 真实 LLM 进行集成测试
- 添加重试机制
- 添加错误处理

### 风险2: FileSystem 性能问题

**描述**: 频繁读写文件可能影响性能

**应对**:
- 添加缓存机制
- 批量写入
- 性能测试

### 风险3: 现有代码兼容性

**描述**: 现有代码可能依赖旧的接口

**应对**:
- 新旧 agent 并存
- 渐进式迁移
- 充分的回归测试

---

## 六、总结

### 开发时间线

| 阶段 | 时间 | 关键任务 | 交付物 |
|-----|------|---------|-------|
| **阶段0** | 第1周 | 测试框架搭建 | 测试基线 |
| **阶段1** | 第1-2周 | FileSystemState | 状态管理 |
| **阶段2** | 第2-3周 | PlanAgent | 规划能力 |
| **阶段3** | 第3周 | Tool 封装层 | Tools |
| **阶段4** | 第4-5周 | KernelAgent | 核心 Agent |
| **阶段5** | 第5-6周 | CLI 适配 | 完整功能 |

**总计**: 6周（1.5个月）

### 核心原则

1. **测试驱动**: Red → Green → Refactor
2. **渐进式**: 每个阶段都有可交付的成果
3. **回归测试**: 确保不破坏现有功能
4. **代码复用**: 最大化复用现有代码

---

**更新时间**: 2026-01-20  
**状态**: 实施方案确定  
**下一步**: 开始阶段0 - 搭建测试框架
