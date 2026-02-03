# test_kernel_agent.py 完整分析

## 📋 测试文件结构

```
test_kernel_agent.py
├── test_kernel_agent()      # 交互式测试（主测试）
├── test_simple_case()        # 简单测试案例
└── main                      # 测试入口
```

## 🔍 详细流程分析

### 1. 初始化阶段（lines 31-48）

```python
api_key = os.getenv("OPENAI_API_KEY") or os.getenv("AIKG_API_KEY")
agent = KernelAgent(task_id="test_kernel_agent", model_level="standard")
```

**检查点**：
- ✅ API Key 必须设置（OPENAI_API_KEY 或 AIKG_API_KEY）
- ✅ KernelAgent 初始化成功
- ✅ 加载 available_tools 和 agent_registry

**可能的错误**：
- ❌ API Key 未设置 → 返回 False，退出测试
- ❌ 导入失败 → 抛出 ImportError
- ❌ 初始化失败 → 抛出异常，被 catch 处理

---

### 2. 交互循环（lines 51-125）

#### 2.1 用户输入（lines 58-73）

```python
user_input = input("👤 您的需求: ").strip()
```

**处理逻辑**：
- 'quit', 'exit', 'q', '退出' → 退出循环
- 空输入 → 提示重新输入，round_num -= 1
- 正常输入 → 继续执行

#### 2.2 执行 agent.run()（line 80）

```python
result = await agent.run(user_input)
```

**KernelAgent.run() 返回格式**：
```python
{
    "status": "waiting_for_user" | "success" | "error",
    "output": str,                    # 输出信息
    "error_information": str,         # 错误信息（可选）
    "message": str,                   # ask_user 时的询问消息（可选）
    "plan_list": List[Dict],          # 执行计划
    "history": List[Dict],            # 执行历史
    "total_actions": int              # 总动作数（可选）
}
```

#### 2.3 显示结果（lines 82-113）

**基本信息**：
```python
print(f"状态: {result.get('status')}")
print(f"输出: {result.get('output')}")
if result.get('error_information'):
    print(f"错误: {result.get('error_information')}")
```

**plan_list 格式**（修复后兼容两种格式）：
```python
# 格式1：plan agent 返回（高层描述）
{
    "step_id": 1,
    "desc": "生成 ReLU 算子的 task_desc 定义",
    "status": "pending"
}

# 格式2：具体执行计划（可能由 LLM 扩展）
{
    "step_id": 1,
    "tool": "call_kernel_gen",
    "description": "生成 ReLU 内核代码",
    "status": "pending",
    "retry_count": 0,
    "max_retries": 3,
    "arguments": {...},
    "depends_on": [...]
}
```

**显示逻辑**（已修复）：
```python
step_desc = step.get('desc') or step.get('description') or step.get('tool', 'Unknown')
```

**history 格式**：
```python
{
    "action_id": "action_1",
    "tool_name": "ask_user",
    "arguments": {...},
    "result": {"status": "waiting", ...},
    "duration_ms": 0
}
```

#### 2.4 状态判断（lines 115-125）

**waiting_for_user**（line 116-118）：
```python
if result.get('status') == 'waiting_for_user':
    print(f"\n💬 Agent 询问: {result.get('message')}")
    continue  # 继续下一轮，等待用户回答
```

**success/error/timeout**（line 121-125）：
```python
if result.get('status') in ['success', 'error', 'timeout']:
    choice = input(">>> ").strip().lower()
    if choice != 'y':
        break  # 退出循环
```

---

### 3. 简单测试案例（lines 141-177）

#### Round 1: 提出需求
```python
result1 = await agent.run("生成一个 ReLU 算子")
```

**预期流程**：
1. LLM 发现缺少信息（DSL、Framework、Backend）
2. 返回 `{"status": "waiting_for_user", "message": "缺少信息..."}`

#### Round 2: 回答问题
```python
result2 = await agent.run("使用 triton_cuda，默认配置")
```

**预期流程**：
1. LLM 调用 plan 工具
2. plan agent 生成 plan_list
3. 返回包含 plan_list 的结果

---

## ⚠️ 关键问题和修复

### 问题 1: plan_list 字段不一致 ✅ 已修复

**原问题**：
- 测试期望：`step.get('tool')`, `step.get('description')`
- 实际返回：`step.get('desc')`（plan agent 返回格式）

**修复方案**：
```python
# 兼容两种格式
step_desc = step.get('desc') or step.get('description') or step.get('tool', 'Unknown')
```

### 问题 2: 响应格式缺少字段 ✅ 已修复

**原问题**：
- 测试期望 `error_information` 字段
- success_response 中缺少该字段

**修复方案**：
```python
def _build_success_response(self) -> Dict[str, Any]:
    return {
        "status": "success",
        "output": "所有任务已完成",
        "error_information": "",  # ✅ 添加此字段
        "plan_list": self.plan_list,
        "history": [...],
        "total_actions": len(self.history)
    }
```

---

## 🚀 测试执行流程示例

### 完整交互示例

```
[Round 1] 用户: "生成 ReLU 算子"
    ↓
KernelAgent.run() 执行
    ↓
LLM: ask_user（缺少信息）
    ↓
返回: {"status": "waiting_for_user", "message": "缺少以下信息..."}
    ↓
[Round 2] 用户: "使用 triton，torch，cuda"
    ↓
KernelAgent.run() 执行
    ↓
LLM: 调用 plan 工具
    ↓
plan agent 返回: 
{
  "arguments": {
    "steps": [
      {"step_id": 1, "desc": "生成 task_desc", "status": "pending"},
      {"step_id": 2, "desc": "实现 kernel", "status": "pending"},
      {"step_id": 3, "desc": "验证精度", "status": "pending"}
    ]
  },
  "result": {"status": "success", "desc": "规划成功"}
}
    ↓
_update_plan_from_result() 提取 plan_list
    ↓
LLM: ask_user（确认计划）
    ↓
[Round 3] 用户: "确认"
    ↓
LLM: 根据 plan_list 执行 step 1
    ↓
... 继续执行 ...
```

---

## ✅ 测试检查清单

运行前检查：
- [ ] 设置 API Key：`export OPENAI_API_KEY='your-key'`
- [ ] 确认 PlanAgent TOOL_NAME = "plan"
- [ ] 确认 plan 已注册到 agent_registry
- [ ] 确认响应格式包含所有必需字段

运行中检查：
- [ ] 第一轮能正常接收用户输入
- [ ] agent.run() 不抛出异常
- [ ] 返回的 result 包含 status 字段
- [ ] plan_list 能正确显示（兼容 desc/tool 两种格式）
- [ ] waiting_for_user 状态能正常循环
- [ ] success/error 状态能正确退出

---

## 🐛 常见错误和解决方案

### 错误 1: ImportError
```
ModuleNotFoundError: No module named 'akg_agents'
```
**解决**：设置 PYTHONPATH 或在 akg_agents 目录下运行

### 错误 2: KeyError: 'tool'
```
KeyError: 'tool'
```
**解决**：已修复，使用 `step.get('desc') or step.get('tool')`

### 错误 3: API Key 未设置
```
[ERROR] 未找到 API Key
```
**解决**：`export OPENAI_API_KEY='your-key'`

### 错误 4: plan 工具未注册
```
❌ plan 工具未注册
```
**解决**：确认 PlanAgent.TOOL_NAME = "plan"

---

## 📊 预期输出示例

```
================================================================================
KernelAgent 交互式测试
================================================================================

[OK] 检测到 API Key: sk-xxx...

================================================================================
第 1 轮交互
================================================================================

请输入您的需求（输入 'quit' 或 'exit' 退出）:
👤 您的需求: 生成 ReLU 算子

--------------------------------------------------------------------------------
开始处理需求（迭代式执行）...
--------------------------------------------------------------------------------

--------------------------------------------------------------------------------
本轮执行结果:
--------------------------------------------------------------------------------
状态: waiting_for_user
输出: 等待用户响应: 缺少以下信息，请一并提供：
1. DSL（如 triton-cuda、triton-ascend 等）
2. Framework（如 torch、mindspore 等）
3. Backend（如 cuda、ascend 等）

执行历史: 共 1 个动作
  [1] ask_user - waiting
--------------------------------------------------------------------------------

💬 Agent 询问: 缺少以下信息，请一并提供：
1. DSL（如 triton-cuda、triton-ascend 等）
2. Framework（如 torch、mindspore 等）
3. Backend（如 cuda、ascend 等）

================================================================================
第 2 轮交互
================================================================================

请输入您的需求（输入 'quit' 或 'exit' 退出）:
👤 您的需求: 使用 triton，torch，cuda
...
```

---

## 🎯 总结

测试文件已经过完整分析和修复，关键修复点：

1. ✅ 兼容 plan_list 的两种格式（desc/tool）
2. ✅ 添加 error_information 字段到 success_response
3. ✅ 修复 PlanAgent TOOL_NAME 为 "plan"
4. ✅ 更新 kernel_agent_system.j2 prompt

测试现在应该可以正常运行！
