# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
ReActAgent - 基于 ReAct 架构的算子生成 Agent
"""

import logging
from typing import List, Optional, Any
from pathlib import Path

from langchain.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langchain.agents import AgentState
from langchain.agents.middleware import before_model
from langgraph.runtime import Runtime

from ai_kernel_generator.core.agent.agent_base_v2 import AgentBaseV2
from ai_kernel_generator.core.tools.sub_agent_tool import create_sub_agent_tools
from ai_kernel_generator.core.tools.basic_tools import create_basic_tools
from ai_kernel_generator.core.sub_agent_registry import get_registry
from ai_kernel_generator.core.skills import SkillLoader

logger = logging.getLogger(__name__)

# 默认最大消息数
DEFAULT_MAX_MESSAGES = 50

PROMPT_DIR = Path(__file__).parent.parent.parent / "resources" / "prompts" / "react_agent"
SYSTEM_PROMPT_FILE = PROMPT_DIR / "system_prompt.md"
def create_checkpointer(backend: str = "memory", db_path: Optional[str] = None) -> Any:
    """
    创建 checkpointer 用于保存对话历史（短期记忆）
    
    """
    #内存
    if backend == "memory":
        from langgraph.checkpoint.memory import InMemorySaver
        logger.info("Created InMemorySaver for short-term memory")
        return InMemorySaver()
    # 可在一个.db的文件中持久化
    elif backend == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            path = db_path or ":memory:"
            logger.info(f"Created SqliteSaver at: {path}")
            return SqliteSaver.from_conn_string(path)
        except ImportError:
            logger.warning("langgraph-checkpoint-sqlite not installed, falling back to InMemorySaver")
            from langgraph.checkpoint.memory import InMemorySaver
            return InMemorySaver()
    else:
        logger.warning(f"Unknown checkpointer backend '{backend}', using InMemorySaver")
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()

def _find_safe_trim_index(messages: list, target_keep: int) -> int:
    """
    为了保证截取之后的message是能够配对的，因为aimessage(有tool_Call)后面必须紧跟对应的toolmessage
    """
    from langchain_core.messages import AIMessage, ToolMessage
    
    if len(messages) <= target_keep:
        return 0
    
    target_start = len(messages) - target_keep
    safe_start = target_start
    
    if safe_start > 0 and safe_start < len(messages):
        start_msg = messages[safe_start]
        
        # 如果起始位置是 ToolMessage，要向前包含对应的 AIMessage
        if isinstance(start_msg, ToolMessage):
            for i in range(safe_start - 1, -1, -1):
                msg = messages[i]
                if isinstance(msg, AIMessage):
                    tool_calls = getattr(msg, 'tool_calls', None)
                    if tool_calls:
                        safe_start = i
                        break
                # 如果遇到非 ToolMessage 且非相关 AIMessage不截取
                elif not isinstance(msg, ToolMessage):
                    break
    
    #检查起始位置前一条是否是带有 tool_calls 的 AIMessage
    if safe_start > 0:
        prev_msg = messages[safe_start - 1]
        if isinstance(prev_msg, AIMessage):
            tool_calls = getattr(prev_msg, 'tool_calls', None)
            if tool_calls:
                if safe_start < len(messages) and isinstance(messages[safe_start], ToolMessage):
                    safe_start -= 1
    
    logger.debug(f"Safe trim: target_start={target_start}, safe_start={safe_start}")
    return safe_start


@before_model
def trim_messages(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """
    保存对话历史
    """
    messages = state["messages"]
    max_messages = DEFAULT_MAX_MESSAGES
    
    if len(messages) <= max_messages:
        return None
    
    first_msg = messages[0]
    remaining_messages = messages[1:]
    
    target_keep = max_messages - 1
    
    safe_start = _find_safe_trim_index(remaining_messages, target_keep)
    recent_messages = remaining_messages[safe_start:]
    
    logger.info(f"Trimming messages: {len(messages)} -> {1 + len(recent_messages)} (safe_start={safe_start})")
    
    return {
        "messages": [
            RemoveMessage(id=REMOVE_ALL_MESSAGES),
            first_msg,
            *recent_messages
        ]
    }

class MainOpAgent(AgentBaseV2):
    """
    使用 ReAct 循环（Thought → Action → Observation）来生成算子代码。
    
    """
    
    def __init__(self,
                 config: dict,
                 model,
                 framework: str = "torch",
                 backend: str = "cuda",
                 arch: str = "a100",
                 dsl: str = "triton",
                 checkpointer=None,
                 enable_memory: bool = True,
                 memory_backend: str = "memory",
                 enable_trim: bool = True):
        """
        初始化 ReActAgent
        
        Args:
            config: 配置字典
            model: LangChain BaseChatModel 实例
            framework: 框架类型
            backend: 后端类型（cuda/ascend/cpu）
            arch: 硬件架构
            dsl: DSL 类型
            checkpointer: 自定义 Memory checkpointer
            enable_memory: 是否启用短期记忆（默认 True）
            memory_backend: 记忆后端类型，"memory"（内存）
            enable_trim: 是否启用消息裁剪（默认 True）
        """
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.dsl = dsl
        
        self.skill_loader = SkillLoader()
        
        # 创建 checkpointer 用于短期记忆
        if checkpointer is None and enable_memory:
            checkpointer = create_checkpointer(memory_backend)
            logger.info(f"Memory enabled with backend: {memory_backend}")
        
        middleware = [trim_messages] if enable_trim else None
        
        super().__init__(
            config=config,
            model=model,
            checkpointer=checkpointer,
            middleware=middleware
        )
        
        logger.info("ReActAgent initialized successfully")
        logger.info(f"  - Skills: {self.skill_loader.get_skill_names()}")
    
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        """
        base_prompt = SYSTEM_PROMPT_FILE.read_text(encoding="utf-8")
        logger.info(f"Loaded system prompt from: {SYSTEM_PROMPT_FILE}")
        
        # 注入 Skills 元数据到 system prompt
        skills_metadata = self._build_skills_metadata()
        if skills_metadata:
            base_prompt = base_prompt + "\n\n" + skills_metadata
            logger.info(f"Injected {len(self.skill_loader.skills)} skills metadata into system prompt")
        
        return base_prompt
    
    def _build_skills_metadata(self) -> str:
        """构建 Skills 元数据"""
        if not self.skill_loader.skills:
            return ""
        
        lines = ["## Skills"]
        for skill in self.skill_loader.skills:
            name = skill["name"]
            desc = skill["description"]
            path = skill.get("path", "")
            lines.append(f"- **{name}**: {desc} (`{path}`)")
        
        return "\n".join(lines)
    
    def create_tools(self) -> List:
        """
        创建 tools
        
        包括：
        - SubAgent tools：call_op_task_builder, call_codeonly, call_evolve 等
        - Basic tools：ask_user, finish, read_file
        """
        tools = []
        
        registry = get_registry()
        sub_agent_tools = create_sub_agent_tools(
            registry=registry,
            config=self.config,
            framework=self.framework,
            backend=self.backend,
            arch=self.arch,
            dsl=self.dsl
        )
        tools.extend(sub_agent_tools)
        
        # 基础 tools
        basic_tools = create_basic_tools()
        tools.extend(basic_tools)
        
        logger.info(f"  - SubAgent tools: {len(sub_agent_tools)}")
        logger.info(f"  - Basic tools: {len(basic_tools)}")
        
        return tools
