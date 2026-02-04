# SimpleReActAgent 示例

一个最小化的 ReAct Agent 实现，演示如何继承 `ReActAgent` 基类创建自己的 Agent。

## 特点

- **无 Plan 功能**：直接响应用户请求，不需要复杂的规划
- **使用 basic_tools**：使用 core_v2 中的基础工具（文件读写、代码检查等）
- **对话式交互**：支持多轮对话

## 文件结构

```
build_a_simple_react_agent/
├── README.md              # 本文件
├── run.py                 # 运行入口
└── simple_react_agent.py  # Agent 实现
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
👤 你: 帮我读取 README.md 文件
🤖 助手: [读取文件内容...]

👤 你: 检查这个 Python 代码是否有语法错误
🤖 助手: [调用 check_python_code 工具...]

👤 你: 你好
🤖 助手: 你好！我是一个智能助手，可以帮你读写文件、检查代码等。有什么可以帮助你的吗？
```

## 可用工具

| 工具名 | 功能 | 说明 |
|--------|------|------|
| `read_file` | 读取文件 | 支持相对路径和绝对路径 |
| `write_file` | 写入文件 | 创建新文件或覆盖现有文件 |
| `check_python_code` | 检查 Python 代码 | 语法检查和自动格式化 |
| `check_markdown` | 检查 Markdown | 使用 markdownlint 检查格式 |
| `execute_script` | 执行脚本 | 运行 Shell 或 Python 脚本 |

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

1. 在 `core_v2/tools/basic_tools.py` 中添加工具函数：

```python
def my_custom_tool(param: str) -> Dict[str, Any]:
    """自定义工具实现"""
    return {
        "status": "success",
        "output": f"结果: {param}",
        "error_information": ""
    }
```

2. 在 `core_v2/config/tools.yaml` 中添加工具定义：

```yaml
  my_custom_tool:
    type: "basic_tool"
    function:
      name: "my_custom_tool"
      description: "自定义工具的描述"
      parameters:
        type: "object"
        properties:
          param:
            type: "string"
            description: "参数描述"
        required: ["param"]
```

## 扩展方向

1. **添加更多工具**：在 basic_tools.py 中扩展
2. **优化 Prompt**：根据实际需求调整系统提示
3. **添加记忆**：实现长期记忆功能
4. **集成更多功能**：如网络搜索、数据库操作等
