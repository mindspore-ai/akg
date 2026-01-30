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
ReActAgentBase - ReAct 架构基类

"""

import logging
from typing import List, Sequence, Optional
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool
from langchain.agents import create_agent

logger = logging.getLogger(__name__)


class AgentBaseV2(ABC):
    """
    AgentBaseV2 基类：扩展ReAct的能力
    
    """
    
    def __init__(self, 
                 config: dict,
                 model,
                 checkpointer=None,
                 middleware: Optional[Sequence] = None):
        self.config = config
        
        if model is None:
            raise ValueError("ReActAgentBase requires a model")
        self.llm = model
        logger.info(f"使用 LLM: {type(model).__name__}")
        
        self.tools = self.create_tools()
        logger.info(f"Created {len(self.tools)} tools: {[t.name for t in self.tools]}")
        
        try:
            agent_kwargs = {
                "model": self.llm,
                "tools": self.tools,
                "system_prompt": self.get_system_prompt(),
            }
            if checkpointer is not None:
                agent_kwargs["checkpointer"] = checkpointer
                logger.info("Agent memory (checkpointer) enabled")
            if middleware:
                agent_kwargs["middleware"] = middleware
                logger.info(f"Agent middleware enabled: {len(middleware)} middleware(s)")
            
            self.agent = create_agent(**agent_kwargs)
            logger.info("Created agent with LangChain create_agent API")
        except Exception as e:
            logger.error(f"Failed to create agent: {e}")
            raise RuntimeError(f"Agent creation failed: {e}")
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        获取系统提示
        
        Returns:
            系统提示字符串
        """
        pass
    
    @abstractmethod
    def create_tools(self) -> List[BaseTool]:
        """
        创建工具列表
        
        Returns:
            tools 列表
        """
        pass
