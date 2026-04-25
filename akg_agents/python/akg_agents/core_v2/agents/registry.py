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
    # 方式 1：使用装饰器注册（无 scope，全局可见）
    @register_agent
    class GlobalAgent(AgentBase):
        ...
    
    # 方式 2：使用装饰器注册（指定 scope）
    @register_agent(scopes=["op"])
    class Coder(AgentBase):
        ...
    
    # 方式 3：使用装饰器注册（多个 scope）
    @register_agent(scopes=["op", "common"])
    class SharedAgent(AgentBase):
        ...
    
    # 方式 4：手动注册
    AgentRegistry.register(Coder, scopes=["op"])
    
    # 创建 Agent 实例
    coder = AgentRegistry.create_agent("Coder", context={...}, config={...})
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
    - Scope：支持应用范围隔离
    """
    
    # 存储已注册的 Agent 类
    _agents: Dict[str, Type] = {}
    
    # 存储每个 Agent 的应用范围（scope）
    # key: agent_name, value: set of scopes (如果为空set，表示全局可见)
    _agent_scopes: Dict[str, set] = {}
    
    @classmethod
    def register(cls, agent_class: Type[T], name: Optional[str] = None, scopes: Optional[List[str]] = None) -> Type[T]:
        """
        注册 Agent 类
        
        Args:
            agent_class: Agent 类
            name: 可选的注册名称，默认使用类名
            scopes: 可选的应用范围列表，如 ["op", "common"]。
                   如果为 None 或空列表，表示全局可见（所有应用都可用）
        
        Returns:
            原 Agent 类（支持装饰器使用）
        """
        agent_name = name or agent_class.__name__
        
        if agent_name in cls._agents:
            logger.warning(f"Agent '{agent_name}' is already registered, overwriting")
        
        cls._agents[agent_name] = agent_class
        
        # 存储 scope 信息（空 set 表示全局可见）
        if scopes:
            cls._agent_scopes[agent_name] = set(scopes)
        else:
            cls._agent_scopes[agent_name] = set()
        
        scope_info = f" (scopes: {list(cls._agent_scopes[agent_name])})" if cls._agent_scopes[agent_name] else " (global)"
        logger.debug(f"Registered agent: {agent_name}{scope_info}")
        
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
            # 同时清理 scope 信息
            if name in cls._agent_scopes:
                del cls._agent_scopes[name]
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
    def list_agents(cls, scope: Optional[str] = None) -> List[str]:
        """
        列出已注册的 Agent 名称
        
        Args:
            scope: 可选的应用范围过滤，如 "op"。
                  如果为 None，返回所有已注册的 Agent
        
        Returns:
            List[str]: Agent 名称列表
        """
        if scope is None:
            # 返回所有 Agent
            return list(cls._agents.keys())
        
        # 返回指定 scope 的 Agent（包括全局可见的 Agent）
        result = []
        for agent_name in cls._agents.keys():
            agent_scopes = cls._agent_scopes.get(agent_name, set())
            # 如果 agent_scopes 为空（全局可见）或包含指定的 scope，则添加
            if not agent_scopes or scope in agent_scopes:
                result.append(agent_name)
        return result
    
    @classmethod
    def is_registered(cls, name: str, scope: Optional[str] = None) -> bool:
        """
        检查 Agent 是否已注册（可选：在指定范围内）
        
        Args:
            name: Agent 名称
            scope: 可选的应用范围，如 "op"。
                  如果为 None，只检查 Agent 是否存在
        
        Returns:
            bool: 是否已注册（在指定范围内）
        """
        if name not in cls._agents:
            return False
        
        if scope is None:
            # 只检查是否存在
            return True
        
        # 检查是否在指定的 scope 中
        agent_scopes = cls._agent_scopes.get(name, set())
        # 如果 agent_scopes 为空（全局可见）或包含指定的 scope，返回 True
        return not agent_scopes or scope in agent_scopes
    
    @classmethod
    def get_agent_scopes(cls, name: str) -> Optional[List[str]]:
        """
        获取 Agent 的应用范围
        
        Args:
            name: Agent 名称
        
        Returns:
            List[str]: Agent 的应用范围列表，如果不存在返回 None
                      空列表表示全局可见（所有应用都可用）
        """
        if name not in cls._agents:
            return None
        
        agent_scopes = cls._agent_scopes.get(name, set())
        return list(agent_scopes) if agent_scopes else []
    
    @classmethod
    def clear(cls) -> None:
        """
        清空所有注册的 Agent（主要用于测试）
        """
        cls._agents.clear()
        cls._agent_scopes.clear()
        logger.debug("Cleared all registered agents")


def register_agent(cls_or_name=None, scopes: Optional[List[str]] = None):
    """
    Agent 注册装饰器
    
    使用方式：
        # 方式 1：直接使用（全局可见）
        @register_agent
        class Coder(AgentBase):
            ...
        
        # 方式 2：指定名称
        @register_agent("CustomCoder")
        class Coder(AgentBase):
            ...
        
        # 方式 3：指定应用范围
        @register_agent(scopes=["op"])
        class OpCoder(AgentBase):
            ...
        
        # 方式 4：指定名称和应用范围
        @register_agent("CustomCoder", scopes=["op", "common"])
        class Coder(AgentBase):
            ...
    
    Args:
        cls_or_name: Agent 类或自定义名称
        scopes: 可选的应用范围列表，如 ["op", "common"]
    
    Returns:
        装饰后的类或装饰器函数
    """
    # 判断是直接装饰（无参数）还是带参数
    if cls_or_name is None or isinstance(cls_or_name, str):
        # 带参数的装饰器 @register_agent("name") 或 @register_agent() 或 @register_agent(scopes=[...])
        name = cls_or_name if isinstance(cls_or_name, str) else None
        
        def decorator(cls: Type[T]) -> Type[T]:
            return AgentRegistry.register(cls, name=name, scopes=scopes)
        
        return decorator
    else:
        # 无参数的装饰器 @register_agent
        return AgentRegistry.register(cls_or_name, scopes=scopes)
