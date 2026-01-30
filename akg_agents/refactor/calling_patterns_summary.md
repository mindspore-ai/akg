# KernelAgent 调用模式最终方案

> 基于 ReAct 的灵活架构
> 更新时间: 2026-01-21

## 一、核心架构

### 1.1 整体设计

```
KernelAgent (非 LangGraph，手动主循环)
    ↓ 可以调用（都是 tools）
    ├─ Workflow (LangGraph，固化流程) → 简单标准场景
    ├─ SubAgent (LangGraph，单步) → 复杂场景 + HITL
    ├─ Tool (基础工具)
    └─ ask_user (HITL 工具)

状态管理: FileSystem (~/.akg/conversations/{task_id}/)
```

### 1.2 关键特性

1. **KernelAgent 是智能 agent**
   - 手动主循环（不用 LangGraph create_agent）
   - 使用 OpenAI function calling (ReAct 风格的 tool calling)
   - PlanAgent + 每10步固化（定期规划机制）
   - FileSystem 作为状态存储

2. **Workflow 和 SubAgent 都是 tools**
   - Workflow: 快捷方式（简单标准场景）
   - SubAgent: 细粒度控制（复杂场景 + HITL）
   - LLM 自主决策用哪个

3. **完全支持 HITL**
   - `ask_user` tool 暂停执行，等待用户输入
   - 可以多轮讨论设计、迭代优化

---

## 二、代码实现

### 2.1 KernelAgent 主循环

```python
# op/agents/kernel_agent.py

class KernelAgent:
    """算子生成主 Agent
    
    特点:
    - 手动主循环
    - OpenAI function calling（ReAct 风格）
    - PlanAgent + 每10步固化
    - FileSystem 状态管理
    """
    
    def __init__(self, 
                 workflows: dict,      # LangGraph workflows
                 subagents: dict,      # SubAgent 实例
                 tools: list,          # 基础 tools
                 plan_agent,           # PlanAgent
                 trace_system,         # Trace System
                 llm_client,
                 file_system_state):   # FileSystem 管理
        
        # Workflows (LangGraph)
        self.workflows = {
            name: workflow.compile() 
            for name, workflow in workflows.items()
        }
        
        # SubAgents
        self.subagents = subagents
        
        # 所有 tools (包含 workflow/subagent 的封装)
        self.all_tools = self._create_all_tools(tools)
        
        # PlanAgent
        self.plan_agent = plan_agent
        
        # Trace System
        self.trace_system = trace_system
        
        # FileSystem
        self.file_system_state = file_system_state
        
        # LLM
        self.llm_client = llm_client
        
        # 状态
        self.action_history = []
        self.current_plan = None
    
    def _create_all_tools(self, basic_tools):
        """创建所有 tools (Workflow + SubAgent + 基础 tools)"""
        tools = []
        
        # 1. Workflow tools
        for name, workflow in self.workflows.items():
            tool = self._create_workflow_tool(name, workflow)
            tools.append(tool)
        
        # 2. SubAgent tools
        for name, subagent in self.subagents.items():
            tool = self._create_subagent_tool(name, subagent)
            tools.append(tool)
        
        # 3. 基础 tools (read_file, write_file, ask_user, ...)
        tools.extend(basic_tools)
        
        return tools
    
    def _create_workflow_tool(self, name, workflow):
        """将 Workflow 封装成 tool"""
        @tool(f"use_{name}_workflow")
        async def workflow_tool(task_desc: str, op_name: str, **kwargs) -> dict:
            """
            使用完整工作流（适合标准场景）
            
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
    
    def _create_subagent_tool(self, name, subagent):
        """将 SubAgent 封装成 tool"""
        @tool(f"call_{name}")
        async def subagent_tool(**kwargs) -> dict:
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
    
    async def execute(self, user_request: str, max_turns: int = 100):
        """
        主执行循环
        
        流程:
        1. PlanAgent 初始规划
        2. 主循环: 构建上下文 → LLM 决策 → 执行 action
        3. 每10步触发 PlanAgent 重新规划
        """
        # 1. PlanAgent 初始规划
        self.current_plan = await self.plan_agent.plan(
            user_request=user_request,
            current_state=self.file_system_state.get_state()
        )
        
        # 2. 主循环
        for turn in range(max_turns):
            # 2.1 构建上下文
            context = self._build_context(
                user_request=user_request,
                plan=self.current_plan,
                action_history=self.action_history,
                file_state=self.file_system_state.get_state()
            )
            
            # 2.2 LLM 决策 (OpenAI function calling)
            response = await self.llm_client.chat(
                messages=[{"role": "user", "content": "请输出下一个动作"}],
                system_prompt=context,
                tools=self.all_tools,  # 传入所有 tools
                tool_choice="required"
            )
            
            # 2.3 执行 tool calls
            for tool_call in response.tool_calls:
                tool_name = tool_call.function.name
                tool_args = json.loads(tool_call.function.arguments)
                
                # 执行 tool
                tool_func = self._get_tool(tool_name)
                result = await tool_func(**tool_args)
                
                # 记录到 action_history
                self.action_history.append({
                    "turn": turn,
                    "tool_name": tool_name,
                    "tool_args": tool_args,
                    "result": result
                })
                
                # 记录到 Trace System
                self.trace_system.record_action(
                    action={
                        "type": self._get_action_type(tool_name),
                        "name": tool_name,
                        "args": tool_args
                    },
                    result=result
                )
                
                # 更新 FileSystem
                self.file_system_state.update(result)
                
                # 检查是否完成
                if tool_name == "finish":
                    return result
            
            # 2.4 每10步触发 PlanAgent（定期规划）
            if (turn + 1) % 10 == 0:
                self.current_plan = await self.plan_agent.replan(
                    user_request=user_request,
                    action_history=self.action_history,
                    current_state=self.file_system_state.get_state()
                )
                
                # 清空 action_history（减少上下文）
                self.action_history = []
                
                # 固化到 FileSystem
                self.file_system_state.save_thinking(self.current_plan)
        
        return {"error": "Max turns reached"}
    
    def _build_context(self, user_request, plan, action_history, file_state):
        """构建上下文 (System Prompt)"""
        return f"""
# 算子生成 Agent

## 任务
{user_request}

## 当前计划
{plan}

## 可用 Actions

### 1. Workflow (完整流程 - 适合标准场景)
- use_standard_workflow(task_desc, op_name): Designer → Coder → Verifier
- use_evolve_workflow(task_desc, op_name): 进化搜索优化

**适用场景**:
- 需求清晰、流程标准
- 不需要人工干预
- 快速生成标准算子

**示例**: "生成一个标准的 ReLU 算子"

### 2. SubAgent (单步 - 适合复杂场景 + HITL)
- call_designer(task_desc, user_requirements): 单独设计算法
- call_coder(design, user_requirements): 单独生成代码
- call_kernel_verifier(code, task_code): 单独验证

**适用场景**:
- 需要讨论设计方案
- 需要多轮迭代
- 需要人工介入决策

**配合**: ask_user() 实现 HITL

**示例**: "生成融合算子，需要讨论设计方案"

### 3. Tools (基础工具)
- ask_user(message): 询问用户意见 (HITL)
- read_file(path): 读取文件
- write_file(path, content): 写入文件
- finish(result): 完成任务

## 决策指南
1. 用户需求清晰 + 标准流程 → use_*_workflow
2. 用户需求模糊 + 需要讨论 → call_designer + ask_user
3. 需要迭代优化 → 逐步调用 SubAgent + ask_user

## 动作历史 (最近 {len(action_history)} 步)
{self._format_action_history(action_history)}

## 当前文件系统状态
{file_state}
"""
    
    def _get_action_type(self, tool_name):
        """判断 action 类型"""
        if tool_name.startswith("use_") and tool_name.endswith("_workflow"):
            return "workflow"
        elif tool_name.startswith("call_"):
            return "subagent"
        else:
            return "tool"
```

---

## 三、使用场景

### 场景1: 简单标准算子

```
用户: "生成一个标准的 ReLU 算子"

KernelAgent 主循环:
    Turn 1:
        Context: "任务: 生成 ReLU，计划: ..., 可用 Actions: ..."
        LLM 决策:
            Thought: "标准算子，流程清晰，使用完整 workflow"
            Action: use_standard_workflow(task_desc="ReLU", op_name="relu")
        
        执行: StandardWorkflow (Designer → Coder → Verifier)
            - Designer 设计算法
            - Coder 生成代码
            - Verifier 验证通过
        
        结果: 返回完整算子
        → 完成
```

### 场景2: 复杂场景 + HITL

```
用户: "生成一个融合算子，需要讨论设计方案"

KernelAgent 主循环:
    Turn 1:
        LLM 决策:
            Thought: "需求不明确，先设计方案"
            Action: call_designer(task_desc="融合算子")
        执行: Designer 生成设计方案 A
    
    Turn 2:
        LLM 决策:
            Thought: "有设计方案了，询问用户意见"
            Action: ask_user("设计了方案A（包含...），您觉得如何？")
        执行: 暂停，等待用户输入
        用户输入: "不行，我要用方案B，要二次切分"
    
    Turn 3:
        LLM 决策:
            Thought: "用户要求方案B + 二次切分，重新设计"
            Action: call_designer(
                task_desc="融合算子",
                user_requirements="二次切分，方案B"
            )
        执行: Designer 生成设计方案 B
    
    Turn 4:
        LLM 决策:
            Thought: "新方案设计好了，再次确认"
            Action: ask_user("修改后的方案B（包含...），可以吗？")
        执行: 暂停，等待用户输入
        用户输入: "可以，生成代码"
    
    Turn 5:
        LLM 决策:
            Thought: "用户确认方案B，开始生成代码"
            Action: call_coder(
                design="方案B的设计",
                user_requirements="二次切分"
            )
        执行: Coder 生成代码 v1
    
    Turn 6:
        LLM 决策:
            Thought: "代码生成完毕，验证"
            Action: call_kernel_verifier(code="v1")
        执行: Verifier 验证
        结果: 精度通过，性能不足
    
    Turn 7:
        LLM 决策:
            Thought: "性能不足，询问用户"
            Action: ask_user("代码功能正确但性能不足，要优化吗？")
        执行: 暂停，等待用户输入
        用户输入: "要，用向量化"
    
    Turn 8:
        LLM 决策:
            Thought: "用户要求向量化优化"
            Action: call_coder(
                current_code="v1",
                user_requirements="向量化优化"
            )
        执行: Coder 生成代码 v2
    
    Turn 9:
        LLM 决策:
            Thought: "优化后的代码，再次验证"
            Action: call_kernel_verifier(code="v2")
        执行: Verifier 验证
        结果: 精度和性能都通过
    
    Turn 10:
        LLM 决策:
            Thought: "验证通过，任务完成"
            Action: finish(result="成功生成优化后的融合算子")
        → 完成
        
    Turn 10 触发: PlanAgent 重新规划，清空 action_history
```

---

## 四、FileSystem 状态管理

### 4.1 目录结构

```
~/.akg/conversations/{task_id}/
├─ trace.json              # Trace 树（记录所有分支）
├─ state.json              # 当前状态快照
├─ thinking.json           # PlanAgent 固化信息（每10步更新）
├─ actions/                # 每个 action 的详细记录
│   ├─ turn_001_call_designer.json
│   ├─ turn_002_ask_user.json
│   ├─ turn_003_call_designer.json
│   ├─ turn_004_ask_user.json
│   ├─ turn_005_call_coder.json
│   ├─ turn_006_call_kernel_verifier.json
│   ├─ turn_007_ask_user.json
│   ├─ turn_008_call_coder.json
│   ├─ turn_009_call_kernel_verifier.json
│   └─ turn_010_finish.json
├─ code/                   # 生成的代码和文档
│   ├─ design_v1.md        # 方案A
│   ├─ design_v2.md        # 方案B (最终)
│   ├─ kernel_v1.py        # 初版代码
│   ├─ kernel_v2.py        # 优化后代码 (最终)
│   └─ verification_results.json
└─ logs/                   # 日志
    └─ execution.log
```

### 4.2 核心机制

**"十步策略"**:

1. **每10步触发 PlanAgent**
   - 分析当前进度
   - 重新规划 (todo_list, next_steps)
   - 固化到 `thinking.json`

2. **清空 action_history**
   - 释放上下文窗口
   - 历史操作的**效果**已保存在文件中
   - 通过读取文件了解历史状态

3. **文件系统 = 真相源**
   - Agent 通过读文件了解历史
   - 支持断点续跑
   - 支持回溯到任意节点

---

## 五、关键优势

### 5.1 灵活性

| 场景 | 调用方式 | HITL | 示例 |
|-----|---------|------|------|
| **简单标准算子** | Workflow | ❌ | ReLU, Add, Mul |
| **需要讨论设计** | SubAgent + ask_user | ✅ | 融合算子设计讨论 |
| **迭代优化** | SubAgent + ask_user | ✅ | 性能优化多轮 |
| **快速原型** | Workflow | ❌ | 快速验证想法 |

### 5.2 智能决策

- ✅ LLM 自主判断用 Workflow 还是 SubAgent
- ✅ 可以根据用户反馈动态调整策略
- ✅ PlanAgent 每10步重新规划

### 5.3 完整 HITL 支持

- ✅ `ask_user` tool 暂停执行
- ✅ 可以多轮讨论设计
- ✅ 可以人工介入任意环节

### 5.4 无限运行能力

- ✅ 每10步清空 action_history
- ✅ PlanAgent 固化关键信息
- ✅ FileSystem 记录所有效果

---

## 六、对比总结

| 维度 | AIKG 当前 | 新架构 (最终方案) |
|-----|----------|------------------|
| **KernelAgent** | ReAct (LangGraph) | 手动主循环 |
| **Tool calling** | ✅ OpenAI function calling | ✅ OpenAI function calling |
| **Workflow** | ❌ 无法调用 | ✅ 作为 tool 调用 |
| **SubAgent** | ✅ 封装成 tool | ✅ 封装成 tool |
| **HITL** | ⚠️ 有限支持 | ✅ 完整支持 (ask_user) |
| **PlanAgent** | ❌ 无 | ✅ 每10步规划 |
| **无限运行** | ❌ 无 | ✅ action_history 清空 |
| **状态管理** | Trace (日志) | FileSystem (真相源) |
| **决策方式** | LLM 自主 | LLM 自主 |

---

## 七、实施步骤

### 阶段1: 核心框架 (1-2周)

1. **实现 KernelAgent 主循环**
   - 手动循环 + OpenAI function calling
   - action_history 管理

2. **实现 FileSystemState**
   - 目录结构
   - 状态存储和加载

3. **实现 PlanAgent**
   - 初始规划
   - 每10步重新规划

4. **封装 Workflow/SubAgent 为 tools**
   - `_create_workflow_tool()`
   - `_create_subagent_tool()`

### 阶段2: HITL 集成 (1周)

1. **实现 ask_user tool**
   - 暂停执行机制
   - 用户输入收集

2. **测试多轮 HITL 场景**
   - 设计讨论
   - 迭代优化

### 阶段3: Trace System (1周)

1. **实现 Trace System**
   - 记录所有 action
   - 支持回溯
   - 多方案探索

2. **集成到 KernelAgent**
   - 每个 action 记录到 Trace
   - FileSystem 同步

### 阶段4: 测试与优化 (1-2周)

1. **端到端测试**
   - 简单场景 (Workflow)
   - 复杂场景 (SubAgent + HITL)

2. **性能优化**
   - 上下文管理
   - FileSystem 读写优化

---

**更新时间**: 2026-01-20  
**状态**: 最终方案确定  
**下一步**: 开始实现 KernelAgent 主循环
