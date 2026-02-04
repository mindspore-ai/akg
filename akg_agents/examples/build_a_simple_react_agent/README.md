# SimpleReActAgent 示例

一个最小化的 ReAct Agent 实现，演示如何继承 `ReActAgent` 基类创建自己的 Agent。

## 特点

- **无 Plan 功能**：直接响应用户请求，不需要复杂的规划
- **简单工具集**：包含计算、时间、天气、知识搜索四个工具
- **对话式交互**：支持多轮对话

## 文件结构

```
build_a_simple_react_agent/
├── README.md           # 本文件
├── run.py              # 运行入口
├── simple_react_agent.py  # Agent 实现
└── tools.py            # 工具定义
```

## 运行方法

```bash
# 1. 进入项目目录并设置环境
cd akg/aikg && source env.sh

# 2. 运行示例
python examples/build_a_simple_react_agent/run.py
```

## 示例对话

```
👤 你: 现在几点了？
🤖 助手: 现在是 2026-02-05 14:30:25，星期三。

👤 你: 计算 (100 + 200) * 3
🤖 助手: 计算结果是 900。

👤 你: 北京天气怎么样？
🤖 助手: 北京今天多云，温度 15°C，湿度 65%。

👤 你: 什么是深度学习？
🤖 助手: 深度学习是机器学习的子领域，使用多层神经网络来学习数据的层次表示。

👤 你: 你好
🤖 助手: 你好！我是一个智能助手，可以帮你计算数学问题、查询时间、天气和知识。有什么可以帮助你的吗？
```

## 可用工具

| 工具名 | 功能 | 示例 |
|--------|------|------|
| `calculate` | 数学计算 | "计算 sqrt(16) + 2^3" |
| `get_current_time` | 获取当前时间 | "现在几点了？" |
| `weather` | 查询天气 | "上海天气怎么样？" |
| `search_knowledge` | 搜索知识库 | "什么是 transformer？" |

## 代码说明

### 继承 ReActAgent

```python
from akg_agents.core_v2.agents.react_agent import ReActAgent

class SimpleReActAgent(ReActAgent):
    def _get_agent_name(self) -> str:
        return "SimpleReActAgent"
    
    def _load_prompt_template(self) -> Jinja2TemplateWrapper:
        # 返回 Jinja2 模板
        ...
    
    def _build_prompt_context(self) -> Dict[str, Any]:
        # 返回模板变量
        return {}
    
    def _get_agent_context(self) -> Dict[str, Any]:
        # 返回上下文
        return {"task_id": self.task_id}
    
    def _load_available_tools(self) -> List[Dict]:
        # 返回工具定义
        return TOOL_DEFINITIONS
```

### 添加自定义工具

在 `tools.py` 中添加：

```python
def my_tool(param: str) -> Dict[str, Any]:
    """工具实现"""
    return {
        "status": "success",
        "output": f"结果: {param}"
    }

# 添加工具定义
TOOL_DEFINITIONS.append({
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "工具描述",
        "parameters": {
            "type": "object",
            "properties": {
                "param": {"type": "string", "description": "参数描述"}
            },
            "required": ["param"]
        }
    }
})

# 注册执行函数
TOOL_FUNCTIONS["my_tool"] = my_tool
```

## 扩展方向

1. **添加更多工具**：如网络搜索、文件操作等
2. **优化 Prompt**：根据实际需求调整系统提示
3. **添加记忆**：实现长期记忆功能
4. **接入外部 API**：如真实的天气 API、搜索 API 等
