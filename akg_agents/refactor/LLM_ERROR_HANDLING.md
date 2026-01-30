# LLM API 错误处理机制

> **最后更新**: 2026-01-29  
> **版本**: v3.0 - 完整的多层错误处理方案

## 概述

当模型 API 出现问题时（如连接失败、服务不可用、环境变量未设置等），系统会优雅地处理错误并向用户显示详细的错误信息，而不是直接退出。

## 问题背景

### 初始问题

当 LLM API 出现问题时，`akg_cli` 会直接退出，不显示任何错误信息：

```bash
~/yiyanzhi/code/akg/aikg> akg_cli op --framework torch --backend cpu --arch x86_64 --dsl cpp --devices 0
👤 nihao
~/yiyanzhi/code/akg/aikg>  # 直接退出，没有任何错误提示
```

### 发现的根本原因

经过深入调查，发现了两个层面的问题：

1. **Agent 层**：`run_llm()` 方法抛出的异常没有用户友好的错误消息
2. **React Agent 层**：
   - **创建阶段**（`__init__`）：环境变量检查失败时异常未被捕获
   - **执行阶段**（`run_turn`）：LangChain API 错误（如 `BadRequestError`）未被捕获

### 解决方案架构

采用**三层防御**策略，在不同层面捕获和处理错误：

```
┌─────────────────────────────────────────┐
│   Layer 1: Agent Base (agent_base.py)  │  ← 捕获 LLM API 调用错误
│   自定义 LLMAPIError 异常               │     (ConnectionError, 模型不可用等)
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Layer 2: Local Executor               │  ← 捕获 React Agent 创建错误
│   (local_executor.py)                   │     (环境变量未设置等)
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│   Layer 3: React Executor               │  ← 捕获 React Agent 执行错误
│   (react_executor.py)                   │     (BadRequestError, 配置错误等)
└─────────────────────────────────────────┘
```

## 详细实现

### Layer 1: Agent Base 层错误处理（`agent_base.py`）

**目标**：捕获所有 agent 中 `run_llm()` 调用的 LLM API 错误

**优势**：零侵入性，所有继承 `AgentBase` 的 agent 自动受益

#### 新增自定义异常类 `LLMAPIError`

```python
class LLMAPIError(Exception):
    """LLM API 调用失败的自定义异常"""
    def __init__(self, message: str, original_error: Exception = None):
        super().__init__(message)
        self.original_error = original_error
        self.user_message = message
    
    def __str__(self):
        """返回友好的错误消息"""
        return self.user_message
```

#### 改进 `AgentBase.run_llm()` 方法

- 捕获所有模型 API 调用异常
- **抛出 `LLMAPIError` 异常**，包含用户友好的错误消息
- 异常会被 agent 的现有异常处理机制捕获

#### 对现有 Agent 的影响

✅ **完全无需修改** - 所有 agent 都不需要任何修改，因为：

1. 现有的 try-except 会自动捕获 `LLMAPIError`
2. `str(e)` 会自动返回友好的错误消息（通过 `__str__()` 方法）
3. 错误消息会自动显示给用户

示例：现有的异常处理代码

```python
try:
    # 调用 LLM
    content, prompt, reasoning = await self.run_llm(...)
    parsed = parse(content)
    ...
except Exception as e:
    # LLMAPIError 会被这里捕获
    # str(e) 会自动返回友好的错误消息
    return {"status": "error", "message": f"错误: {str(e)}"}
```

---

### Layer 2: Local Executor 层错误处理（`local_executor.py`）

**目标**：捕获 React Agent 创建阶段的错误

**问题场景**：
- 环境变量未设置（如 `AKG_AGENTS_DEEPSEEK_API_KEY`）
- API 配置错误
- 模型预设不存在

**实现位置**：`execute_main_agent()` 方法中创建 `ReactTurnExecutor` 的地方

```python
# 文件：akg_agents/cli/runtime/local_executor.py
# 行号：162-216

if (
    self._react_executor is None
    or getattr(self._react_executor, "target", None) != target
):
    try:
        self._react_executor = ReactTurnExecutor(
            session_id=self.session_id,
            config=config,
            target=target,
            thread_id=self.session_id,
        )
    except Exception as e:
        # 捕获 React Agent 创建时的错误
        logger.error(f"[LocalExecutor] 创建 React Agent 失败: {type(e).__name__}: {e}")
        error_type = type(e).__name__
        error_message = str(e)
        
        user_message = (
            f"❌ 创建 React Agent 失败\n\n"
            f"错误类型: {error_type}\n"
            f"错误信息: {error_message}\n\n"
        )
        
        # 针对常见错误提供提示
        if "API密钥未找到" in error_message or "api_key" in error_message.lower():
            user_message += (
                "请检查环境变量设置：\n"
                "1. AKG_AGENTS_DEEPSEEK_API_KEY - DeepSeek API 密钥\n"
                "2. 或使用环境变量覆盖：\n"
                "   - AKG_AGENTS_MODEL_NAME\n"
                "   - AKG_AGENTS_BASE_URL\n"
                "   - AKG_AGENTS_API_KEY\n\n"
                "示例：\n"
                "export AKG_AGENTS_DEEPSEEK_API_KEY='your_api_key'\n"
                "或\n"
                "export AKG_AGENTS_MODEL_NAME='your_model'\n"
                "export AKG_AGENTS_BASE_URL='http://localhost:8001/v1'\n"
                "export AKG_AGENTS_API_KEY='dummy'"
            )
        
        return self._build_response_state({
            "current_step": "error",
            "should_continue": False,
            "display_message": user_message,
            "hint_message": "请检查配置后重试",
            "workflow_name": "react",
        })
```

**关键点**：
- 返回符合 CLI 期望的状态字典
- `current_step: "error"` 告诉 CLI 发生了错误
- `should_continue: False` 告诉 CLI 停止执行
- `display_message` 包含用户友好的错误消息

---

### Layer 3: React Executor 层错误处理（`react_executor.py`）

**目标**：捕获 React Agent 执行阶段的错误

**问题场景**：
- LangChain API 调用失败（如 `BadRequestError`）
- 配置参数错误（如 `max_tokens` 超限）
- 网络连接问题

**实现位置**：`run_turn()` 方法的 try-except 块

```python
# 文件：akg_agents/cli/runtime/react_executor.py
# 行号：393-448

try:
    self.running_task = asyncio.create_task(_run())
    await self.running_task
except asyncio.CancelledError:
    # 取消后回滚
    self._awaiting_resume = False
    await self._rollback_incomplete_tool_calls()
    return {
        "current_step": "cancelled_by_user",
        "should_continue": True,
        "display_message": "⚠️ 操作已被用户取消",
        "hint_message": "",
        "workflow_name": "react",
    }
except Exception as e:
    # 捕获 LLM API 错误（如 BadRequestError）
    logger.error(f"[ReactTurnExecutor] 执行失败: {type(e).__name__}: {e}")
    
    error_type = type(e).__name__
    error_message = str(e)
    
    # 检查是否是 LLMAPIError
    from akg_agents.core.agent.agent_base import LLMAPIError
    if isinstance(e, LLMAPIError):
        user_message = e.user_message
    else:
        # 其他错误，构建基本的错误消息
        user_message = (
            f"❌ 执行失败\n\n"
            f"错误类型: {error_type}\n"
            f"错误信息: {error_message}\n\n"
        )
        
        # 如果是 API 配置错误，添加提示
        if "max_tokens" in error_message.lower() or "invalid" in error_message.lower():
            user_message += (
                "可能的原因：\n"
                "1. 模型配置参数超出限制（如 max_tokens）\n"
                "2. API 密钥无效或过期\n"
                "3. 模型服务不可用\n\n"
                "请检查配置文件和环境变量设置"
            )
    
    return {
        "current_step": "error",
        "should_continue": False,
        "display_message": user_message,
        "hint_message": "请检查配置后重试",
        "workflow_name": "react",
    }
finally:
    self.running_task = None
```

**关键点**：
- 区分 `LLMAPIError`（来自 Layer 1）和其他异常
- 对 `LLMAPIError` 直接使用其友好消息
- 对其他异常（如 `BadRequestError`）构建新的友好消息
- 针对常见问题（如 `max_tokens`、`invalid`）提供具体的排查建议

**错误示例**：

当配置参数超出限制时（如 `max_tokens` 超过 API 限制），会显示：

```
❌ 执行失败

错误类型: BadRequestError
错误信息: Error code: 400 - {'error': {'message': 'Invalid max_tokens value, 
the valid range of max_tokens is [1, 8192]', 'type': 'invalid_request_error'}}

可能的原因：
1. 模型配置参数超出限制（如 max_tokens）
2. API 密钥无效或过期
3. 模型服务不可用

请检查配置文件和环境变量设置
```

用户可以根据错误提示修改 `llm_config.yaml` 中的相应配置

## 错误消息格式

当 API 调用失败时，用户会看到如下格式的错误消息：

```
❌ 模型 API 调用失败

错误类型: ConnectionResetError
错误信息: [Errno 54] Connection reset by peer

请检查：
1. 模型服务是否正常运行（如 vLLM 服务）
2. 网络连接是否正常
3. 环境变量配置是否正确：
   - AKG_AGENTS_MODEL_NAME: 模型名称
   - AKG_AGENTS_BASE_URL: API 服务地址
   - AKG_AGENTS_API_KEY: API 密钥
4. 如果使用预设配置（如 vllm_deepseek_r1_default），请确保对应的环境变量已设置

使用的模型: vllm_deepseek_r1_default
Agent: op_task_builder
```

## 解决方案优势

### 1. 零侵入性（对 Agent 代码）

- **Agent 层**：只修改 `agent_base.py`，所有 agent 无需任何修改
- **自动生效**：利用 Python 异常机制和 `__str__()` 方法
- **向后兼容**：不破坏任何现有代码

### 2. 多层防御

- **Layer 1 (Agent Base)**：捕获 `run_llm()` 调用错误（如 ConnectionError）
- **Layer 2 (Local Executor)**：捕获 React Agent 创建错误（如环境变量未设置）
- **Layer 3 (React Executor)**：捕获 React Agent 执行错误（如 BadRequestError、配置参数错误）

### 3. 用户友好

- **详细的错误信息**：包含错误类型、消息、模型名称、agent 名称
- **具体的排查建议**：针对不同错误类型提供具体的解决方案
- **不会静默失败**：所有错误都会显示给用户
- **优雅退出**：显示错误后正常退出，不会崩溃

### 4. 开发友好

- **完整的日志**：所有错误都会记录到日志文件
- **易于调试**：错误消息包含足够的上下文信息
- **易于扩展**：可以轻松添加新的错误类型处理

---

## 修改的文件清单

### 核心修改（3个文件）

1. **`aikg/python/akg_agents/core/agent/agent_base.py`**
   - 新增 `LLMAPIError` 异常类
   - 修改 `run_llm()` 方法：捕获异常并抛出 `LLMAPIError`
   - **影响范围**：所有继承 `AgentBase` 的 agent

2. **`aikg/python/akg_agents/cli/runtime/local_executor.py`**
   - 在 `execute_main_agent()` 中添加 try-except 捕获 `ReactTurnExecutor` 创建错误
   - 返回包含友好错误消息的状态字典
   - **影响范围**：`akg_cli op` 命令

3. **`aikg/python/akg_agents/cli/runtime/react_executor.py`**
   - 在 `run_turn()` 中添加 except Exception 捕获执行时错误
   - 区分 `LLMAPIError` 和其他异常类型
   - **影响范围**：React Agent 执行流程

### 无需修改的文件

以下 agent **无需任何修改**即可自动受益：
- ✅ `op_task_builder.py`
- ✅ `conductor.py`
- ✅ `coder.py`
- ✅ `designer.py`
- ✅ `selector.py`
- ✅ `test_case_generator.py`
- ✅ `main_op_agent.py`
- ✅ 其他所有使用 `run_llm()` 的 agent

---

## 测试验证

### 测试场景 1：环境变量未设置

```bash
# 清空所有环境变量
unset AKG_AGENTS_DEEPSEEK_API_KEY
unset AKG_AGENTS_MODEL_NAME
unset AKG_AGENTS_BASE_URL
unset AKG_AGENTS_API_KEY

# 运行 akg_cli
cd aikg && source env.sh
akg_cli op --framework torch --backend cpu --arch x86_64 --dsl cpp --devices 0
```

**期望结果**：

```
❌ 创建 React Agent 失败

错误类型: ValueError
错误信息: API密钥未找到。请设置环境变量 AKG_AGENTS_DEEPSEEK_API_KEY

请检查环境变量设置：
1. AKG_AGENTS_DEEPSEEK_API_KEY - DeepSeek API 密钥
2. 或使用环境变量覆盖：
   - AKG_AGENTS_MODEL_NAME
   - AKG_AGENTS_BASE_URL
   - AKG_AGENTS_API_KEY

示例：
export AKG_AGENTS_DEEPSEEK_API_KEY='your_api_key'
或
export AKG_AGENTS_MODEL_NAME='your_model'
export AKG_AGENTS_BASE_URL='http://localhost:8001/v1'
export AKG_AGENTS_API_KEY='dummy'
```

### 测试场景 2：API 连接失败

```bash
# 设置错误的 API 地址
export AKG_AGENTS_MODEL_NAME='test-model'
export AKG_AGENTS_BASE_URL='http://localhost:9999/v1'  # 不存在的服务
export AKG_AGENTS_API_KEY='dummy'

# 运行测试
cd aikg && source env.sh
python tools/use_llm_check/test_main_op_agent.py
```

**期望结果**：显示包含连接错误信息的友好消息

### 测试场景 3：配置参数错误

当配置文件中的参数超出 API 限制时（如 `max_tokens` 超过限制），会显示友好的错误消息：

```
❌ 执行失败

错误类型: BadRequestError
错误信息: Error code: 400 - {'error': {'message': 'Invalid max_tokens value...'}}

可能的原因：
1. 模型配置参数超出限制（如 max_tokens）
2. API 密钥无效或过期
3. 模型服务不可用

请检查配置文件和环境变量设置
```

用户可以根据错误消息修改 `llm_config.yaml` 中的相应配置

---

## 重要提示：Python 缓存

⚠️ **修改代码后必须清除 Python 缓存**，否则会运行旧代码：

```bash
# 清除缓存
find /Users/qiaolina/yiyanzhi/code/akg/aikg/python -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find /Users/qiaolina/yiyanzhi/code/akg/aikg/python -name "*.pyc" -delete 2>/dev/null

# 或使用 Python 命令
cd aikg && python -Bc "import py_compile"
```

**症状识别**：
- 代码已修改但错误消息没有变化
- 日志中看不到新增的错误处理日志
- `akg_cli` 仍然直接退出

**解决方法**：清除缓存后重新运行

---

## 故障排查

### 问题：`akg_cli` 仍然直接退出

**排查步骤**：

1. **检查是否清除了 Python 缓存**
   ```bash
   find aikg/python -name "*.pyc" | wc -l
   # 如果数量很多，需要清除
   ```

2. **检查日志文件**
   ```bash
   tail -100 /Users/qiaolina/akg_agents_logs/aikg.log
   ```
   - 应该能看到 `[LocalExecutor] 创建 React Agent 失败` 的日志
   - 如果看不到，说明运行的是旧代码

3. **检查环境变量**
   ```bash
   env | grep AIKG
   ```

4. **测试 LocalExecutor 是否工作**
   ```bash
   cd aikg && python -c "
   import os
   import asyncio
   import sys
   
   # 清空环境变量
   for key in ['AKG_AGENTS_DEEPSEEK_API_KEY', 'AKG_AGENTS_MODEL_NAME', 'AKG_AGENTS_BASE_URL', 'AKG_AGENTS_API_KEY']:
       os.environ.pop(key, None)
   
   sys.path.insert(0, 'python')
   from akg_agents.cli.runtime.local_executor import LocalExecutor
   
   async def test():
       executor = LocalExecutor(session_id='test')
       result = await executor.execute_main_agent(
           user_input='test',
           framework='torch',
           backend='cpu',
           arch='x86_64',
           dsl='cpp'
       )
       print('Result:', result.get('display_message'))
   
   asyncio.run(test())
   "
   ```

### 问题：看到错误消息但仍然退出

**这是正常行为** - 错误消息会显示，然后程序正常退出（因为 `should_continue: False`）

### 问题：错误消息格式不对

检查是否是以下原因：
1. 终端不支持 ANSI 颜色码
2. Rich 库版本不兼容
3. 控制台重定向

---

## 总结

### 完整的错误处理链路

```
用户输入
   ↓
akg_cli (runners.py)
   ↓
LocalExecutor.execute_main_agent()
   ├─ try: 创建 ReactTurnExecutor
   │    ├─ ReactTurnExecutor.__init__()
   │    │    └─ create_langchain_chat_model()
   │    │         └─ 检查环境变量 ✗ → ValueError
   │    └─ except Exception → 返回友好错误消息 ✓
   │
   ↓（创建成功）
   │
ReactTurnExecutor.run_turn()
   ├─ try: 执行 React Agent
   │    ├─ react_agent.invoke()
   │    │    └─ LangChain API call ✗ → BadRequestError
   │    └─ except Exception → 返回友好错误消息 ✓
   │
   ↓（调用成功但 run_llm 失败）
   │
AgentBase.run_llm()
   ├─ try: 调用 LLM API
   │    └─ API call ✗ → ConnectionResetError
   └─ except Exception → raise LLMAPIError(友好消息) ✓
        ↓
   上层 agent try-except 捕获
        └─ str(e) 返回友好消息 ✓
```

### 关键要点

1. **三层防御**确保所有错误都能被捕获
2. **配置修复**解决了根本问题（`max_tokens`）
3. **零侵入性**设计使得 agent 代码无需修改
4. **清除缓存**是修改后测试的必要步骤
5. **日志记录**帮助开发者快速定位问题

### 修复验证清单

- [x] 修改了 `agent_base.py`（添加 `LLMAPIError`）
- [x] 修改了 `local_executor.py`（捕获创建错误）
- [x] 修改了 `react_executor.py`（捕获执行错误）
- [x] 清除了 Python 缓存
- [x] 测试了环境变量未设置的场景
- [x] 测试了 API 配置错误的场景（如 `max_tokens` 超限）
- [x] 确认错误消息能正确显示

### 相关文件

- **代码实现**：
  - `aikg/python/akg_agents/core/agent/agent_base.py`
  - `aikg/python/akg_agents/cli/runtime/local_executor.py`
  - `aikg/python/akg_agents/cli/runtime/react_executor.py`
- **测试文件**：
  - `aikg/tools/use_llm_check/test_main_op_agent.py`
  - `aikg/tools/use_llm_check/test_run_llm.py`
- **日志文件**：
  - `/Users/qiaolina/akg_agents_logs/aikg.log`
- **配置文件**：
  - `aikg/python/akg_agents/core/llm/llm_config.yaml` - 用户可根据错误提示自行调整

---

**文档最后更新时间**: 2026-01-29  
**问题解决状态**: ✅ 已完成  
**测试状态**: ✅ 已验证
