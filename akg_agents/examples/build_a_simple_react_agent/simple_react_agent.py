"""
SimpleReActAgent - 简单的 ReAct Agent 示例

一个最小化的 ReActAgent 实现，演示如何：
- 继承 ReActAgent 基类
- 定义和调用工具
- 进行对话式交互

不包含 plan 功能，适合简单的对话和工具调用场景。
"""

import logging
import json
from typing import Dict, Any, List, Optional
from pathlib import Path

from akg_agents.core_v2.agents.react_agent import ReActAgent
from akg_agents.core_v2.agents.base import Jinja2TemplateWrapper
from akg_agents.core_v2.tools.tool_executor import ToolExecutor

from tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS

logger = logging.getLogger(__name__)


class SimpleToolExecutor:
    """简单的工具执行器"""
    
    def __init__(self, tool_functions: Dict[str, callable]):
        self.tool_functions = tool_functions
        self.history = []
        self.agent_context = {}
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """执行工具"""
        if tool_name not in self.tool_functions:
            return {
                "status": "error",
                "output": "",
                "error_information": f"未知工具: {tool_name}"
            }
        
        try:
            func = self.tool_functions[tool_name]
            result = func(**arguments)
            return result
        except Exception as e:
            return {
                "status": "error",
                "output": "",
                "error_information": f"工具执行失败: {str(e)}"
            }


class SimpleReActAgent(ReActAgent):
    """
    简单的 ReAct Agent
    
    特点：
    - 不包含 plan 功能
    - 使用简单的工具集
    - 适合对话式交互
    """
    
    def __init__(
        self,
        task_id: str = "simple_task",
        model_level: str = "standard",
        base_dir: Optional[str] = None
    ):
        """
        初始化 SimpleReActAgent
        
        Args:
            task_id: 任务 ID
            model_level: 模型级别
            base_dir: 基础目录
        """
        super().__init__(
            task_id=task_id,
            model_level=model_level,
            config={},
            base_dir=base_dir
        )
        
        # 使用简单的工具执行器替换默认的
        self.tool_executor = SimpleToolExecutor(TOOL_FUNCTIONS)
    
    def _get_agent_name(self) -> str:
        """获取 Agent 名称"""
        return "SimpleReActAgent"
    
    def _load_prompt_template(self) -> Jinja2TemplateWrapper:
        """加载 prompt 模板"""
        # 使用内嵌的简单 prompt 模板
        template = """# Simple ReAct Agent

你是一个智能助手，可以使用工具来帮助用户解决问题。

## 可用工具

{{ available_tools }}

## 对话历史

{% if action_history %}
{{ action_history }}
{% else %}
暂无历史
{% endif %}

## 用户请求

{{ user_input }}

---

## 执行规则

1. **分析用户请求**：理解用户想要什么
2. **选择合适的工具**：如果需要，调用工具获取信息
3. **直接回答**：如果可以直接回答，使用 `finish` 工具

## 输出格式

**必须返回 JSON 格式，不要有任何其他文本！**

```json
{
  "tool_name": "工具名称",
  "arguments": {"参数名": "参数值"},
  "reason": "为什么选择这个工具"
}
```

### 可用的 tool_name:
- `calculate`: 计算数学表达式
- `get_current_time`: 获取当前时间
- `search_knowledge`: 搜索知识库
- `weather`: 获取天气信息
- `ask_user`: 向用户提问（当信息不足时）
- `finish`: 任务完成，直接回复用户

### finish 工具的使用

当你可以直接回答用户问题时，使用 finish:

```json
{
  "tool_name": "finish",
  "arguments": {
    "message": "你要对用户说的话"
  },
  "reason": "可以直接回答用户"
}
```

### ask_user 工具的使用

当需要更多信息时：

```json
{
  "tool_name": "ask_user",
  "arguments": {
    "message": "你想问用户的问题"
  },
  "reason": "需要更多信息"
}
```

---

## 注意事项

1. **优先使用工具**：如果问题涉及计算、时间、天气或知识查询，使用对应工具
2. **直接回答简单问题**：对于闲聊或简单问题，直接使用 finish 回复
3. **保持友好**：用自然的语言与用户交流
4. **只返回 JSON**：不要添加任何额外的解释文字

现在，请分析用户请求并决定下一步行动：
"""
        return Jinja2TemplateWrapper(template)
    
    def _build_prompt_context(self) -> Dict[str, Any]:
        """构建 prompt 上下文变量"""
        return {}  # 简单实现，不需要额外变量
    
    def _get_agent_context(self) -> Dict[str, Any]:
        """获取 agent 上下文"""
        return {
            "task_id": self.task_id,
            "model_level": self.model_level or "standard"
        }
    
    def _load_available_tools(self) -> List[Dict]:
        """加载可用工具列表"""
        return TOOL_DEFINITIONS.copy()
    
    def _load_agent_registry(self) -> Dict[str, Any]:
        """不需要加载 agent registry"""
        return {}
    
    def _load_workflow_registry(self) -> Dict[str, Any]:
        """不需要加载 workflow registry"""
        return {}
    
    async def run(self, user_input: str) -> Dict[str, Any]:
        """
        执行 ReAct 循环（简化版，支持 finish with message）
        """
        if not self._initialized:
            self._initialize_task(user_input)
            self._original_user_input = user_input
            self.tool_executor.agent_context["user_input"] = user_input
            self._initialized = True
        else:
            # 判断是对 ask_user 的响应，还是新任务
            current_node = self.trace.get_node(self.current_node_id)
            is_ask_user_response = (current_node.action and 
                                   current_node.action.get("type") == "ask_user")
            
            if is_ask_user_response:
                self._handle_user_response(user_input)
            else:
                logger.info(f"[新对话] 开始新对话")
                self._original_user_input = user_input
                self.tool_executor.agent_context["user_input"] = user_input
        
        max_iterations = 10
        iteration = 0
        
        while iteration < max_iterations:
            iteration += 1
            llm_response = await self._get_next_action()
            
            if not llm_response:
                return self._build_error_response("LLM 调用失败")
            
            tool_name = llm_response.get("tool_name")
            arguments = llm_response.get("arguments", {})
            reason = llm_response.get("reason", "")
            
            logger.info(f"[ReAct] {tool_name} - {reason}")
            
            # 处理 finish（带消息）
            if tool_name == "finish":
                message = arguments.get("message", "任务完成")
                return self._build_success_response_with_message(message)
            
            # 处理 ask_user
            if tool_name == "ask_user":
                return self._handle_ask_user(arguments)
            
            # 执行工具
            result = await self._execute_tool(tool_name, arguments)
            logger.info(f"[Tool Result] {tool_name}: {result.get('status')}")
        
        return self._build_error_response("达到最大迭代次数")
    
    def _build_success_response_with_message(self, message: str) -> Dict[str, Any]:
        """构建带消息的成功响应"""
        full_history = self.trace.get_full_action_history(self.current_node_id)
        return {
            "status": "success",
            "output": message,
            "message": message,
            "error_information": "",
            "history": [self._format_action_record(r) for r in full_history],
            "total_actions": len(full_history),
            "current_node": self.current_node_id
        }
