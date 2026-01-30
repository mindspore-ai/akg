# Copyright 2026 Huawei Technologies Co., Ltd
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
AgentRegistry - Agent 注册中心

提供 Agent 的注册和发现机制。

使用示例：
    # 方式 1：使用装饰器注册
    @register_agent
    class Coder(AgentBase):
        ...
    
    # 方式 2：手动注册
    AgentRegistry.register(Coder)
    
    # 创建 Agent 实例
    coder = AgentRegistry.create_agent("Coder", context={...}, config={...})
    
    # 列出所有已注册的 Agent
    agent_names = AgentRegistry.list_agents()
"""

import logging
from typing import Dict, List, Type, TypeVar, Optional

logger = logging.getLogger(__name__)

# 泛型类型，用于装饰器类型提示
T = TypeVar("T")


class AgentRegistry:
    """
    Agent 注册中心（单例模式）
    
    管理所有已注册的 Agent 类，提供：
    - 注册：通过装饰器或手动调用
    - 发现：列出所有已注册的 Agent
    - 创建：根据名称创建 Agent 实例
    """
    
    # 存储已注册的 Agent 类
    _agents: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, agent_class: Type[T], name: Optional[str] = None) -> Type[T]:
        """
        注册 Agent 类
        
        Args:
            agent_class: Agent 类
            name: 可选的注册名称，默认使用类名
        
        Returns:
            原 Agent 类（支持装饰器使用）
        """
        agent_name = name or agent_class.__name__
        
        if agent_name in cls._agents:
            logger.warning(f"Agent '{agent_name}' is already registered, overwriting")
        
        cls._agents[agent_name] = agent_class
        logger.debug(f"Registered agent: {agent_name}")
        
        return agent_class
    
    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        取消注册 Agent
        
        Args:
            name: Agent 名称
        
        Returns:
            bool: 是否成功取消注册
        """
        if name in cls._agents:
            del cls._agents[name]
            logger.debug(f"Unregistered agent: {name}")
            return True
        return False
    
    @classmethod
    def get_agent_class(cls, name: str) -> Optional[Type]:
        """
        获取 Agent 类
        
        Args:
            name: Agent 名称
        
        Returns:
            Agent 类，如果未找到返回 None
        """
        return cls._agents.get(name)
    
    @classmethod
    def create_agent(cls, agent_type: str, **kwargs):
        """
        根据名称创建 Agent 实例
        
        Args:
            agent_type: Agent 类型名称
            **kwargs: 传递给 Agent 构造函数的参数
        
        Returns:
            Agent 实例
        
        Raises:
            ValueError: Agent 类型未注册
        """
        if agent_type not in cls._agents:
            available = ", ".join(cls._agents.keys()) or "None"
            raise ValueError(f"Agent '{agent_type}' not registered. Available: {available}")
        
        agent_class = cls._agents[agent_type]
        return agent_class(**kwargs)
    
    @classmethod
    def list_agents(cls) -> List[str]:
        """
        列出所有已注册的 Agent 名称
        
        Returns:
            List[str]: Agent 名称列表
        """
        return list(cls._agents.keys())
    
    @classmethod
    def is_registered(cls, name: str) -> bool:
        """
        检查 Agent 是否已注册
        
        Args:
            name: Agent 名称
        
        Returns:
            bool: 是否已注册
        """
        return name in cls._agents
    
    @classmethod
    def clear(cls) -> None:
        """
        清空所有注册的 Agent（主要用于测试）
        """
        cls._agents.clear()
        logger.debug("Cleared all registered agents")


def register_agent(cls_or_name=None):
    """
    Agent 注册装饰器
    
    使用方式：
        # 方式 1：直接使用
        @register_agent
        class Coder(AgentBase):
            ...
        
        # 方式 2：指定名称
        @register_agent("CustomCoder")
        class Coder(AgentBase):
            ...
    
    Args:
        cls_or_name: Agent 类或自定义名称
    
    Returns:
        装饰后的类或装饰器函数
    """
    # 判断是直接装饰（无参数）还是带参数
    if cls_or_name is None or isinstance(cls_or_name, str):
        # 带参数的装饰器 @register_agent("name") 或 @register_agent()
        name = cls_or_name
        
        def decorator(cls: Type[T]) -> Type[T]:
            return AgentRegistry.register(cls, name=name)
        
        return decorator
    else:
        # 无参数的装饰器 @register_agent
        return AgentRegistry.register(cls_or_name)
