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
WorkflowRegistry - Workflow 注册中心

提供独立于 Agent 的 Workflow 注册和发现机制。

使用示例：
    # 注册 workflow
    @register_workflow(scopes=["op"])
    class CoderOnlyWorkflow(OpBaseWorkflow):
        TOOL_NAME = "use_coder_only_workflow"
        DESCRIPTION = "使用 CoderOnly workflow 生成代码"
        PARAMETERS_SCHEMA = {...}
        ...
"""

import logging
from typing import Dict, List, Type, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class WorkflowRegistry:
    """
    Workflow 注册中心（单例模式）
    
    管理所有已注册的 Workflow 类，提供：
    - 注册：通过装饰器或手动调用
    - 发现：列出所有已注册的 Workflow
    - 获取：根据名称获取 Workflow 类
    - Scope：支持应用范围隔离
    """
    
    # 存储已注册的 Workflow 类
    _workflows: Dict[str, Type] = {}
    
    # 存储每个 Workflow 的应用范围（scope）
    _workflow_scopes: Dict[str, set] = {}
    
    # 存储 Workflow 的工具配置元数据
    _workflow_metadata: Dict[str, Dict] = {}
    
    @classmethod
    def register(
        cls, 
        workflow_class: Type[T], 
        name: Optional[str] = None, 
        scopes: Optional[List[str]] = None
    ) -> Type[T]:
        """
        注册 Workflow 类
        
        Args:
            workflow_class: Workflow 类
            name: 可选的注册名称，默认使用类名
            scopes: 可选的应用范围列表，如 ["op", "common"]
        
        Returns:
            原 Workflow 类（支持装饰器使用）
        """
        workflow_name = name or workflow_class.__name__
        
        if workflow_name in cls._workflows:
            logger.warning(f"Workflow '{workflow_name}' is already registered, overwriting")
        
        cls._workflows[workflow_name] = workflow_class
        
        # 存储 scope 信息
        if scopes:
            cls._workflow_scopes[workflow_name] = set(scopes)
        else:
            cls._workflow_scopes[workflow_name] = set()
        
        # 提取并存储工具配置元数据
        cls._extract_metadata(workflow_name, workflow_class)
        
        scope_info = f" (scopes: {list(cls._workflow_scopes[workflow_name])})" if cls._workflow_scopes[workflow_name] else " (global)"
        logger.debug(f"Registered workflow: {workflow_name}{scope_info}")
        
        return workflow_class
    
    @classmethod
    def _extract_metadata(cls, workflow_name: str, workflow_class: Type):
        """
        提取 Workflow 的工具配置元数据
        
        Workflow 类需要定义：
        - TOOL_NAME: 工具名称
        - DESCRIPTION: 功能描述
        - PARAMETERS_SCHEMA: 参数 schema
        """
        metadata = {}
        
        if hasattr(workflow_class, 'TOOL_NAME'):
            metadata['tool_name'] = workflow_class.TOOL_NAME
        if hasattr(workflow_class, 'DESCRIPTION'):
            metadata['description'] = workflow_class.DESCRIPTION
        if hasattr(workflow_class, 'PARAMETERS_SCHEMA'):
            metadata['parameters_schema'] = workflow_class.PARAMETERS_SCHEMA
        
        cls._workflow_metadata[workflow_name] = metadata
    
    @classmethod
    def get_workflow_class(cls, name: str) -> Optional[Type]:
        """
        获取 Workflow 类
        
        Args:
            name: Workflow 名称
        
        Returns:
            Workflow 类，如果未找到返回 None
        """
        return cls._workflows.get(name)
    
    @classmethod
    def list_workflows(cls, scope: Optional[str] = None) -> List[str]:
        """
        列出已注册的 Workflow 名称
        
        Args:
            scope: 可选的应用范围过滤，如 "op"
        
        Returns:
            Workflow 名称列表
        """
        if scope is None:
            return list(cls._workflows.keys())
        
        result = []
        for workflow_name in cls._workflows.keys():
            workflow_scopes = cls._workflow_scopes.get(workflow_name, set())
            if not workflow_scopes or scope in workflow_scopes:
                result.append(workflow_name)
        return result
    
    @classmethod
    def get_tool_config(cls, workflow_name: str) -> Optional[Dict]:
        """
        获取 Workflow 的工具配置
        
        Args:
            workflow_name: Workflow 名称
        
        Returns:
            工具配置字典，如果没有配置返回 None
        """
        if workflow_name not in cls._workflow_metadata:
            return None
        
        metadata = cls._workflow_metadata[workflow_name]
        
        # 检查必需字段
        if not metadata.get('tool_name'):
            return None
        
        tool_name = metadata['tool_name']
        
        return {
            tool_name: {
                "type": "call_workflow",
                "workflow_name": workflow_name,
                "function": {
                    "name": tool_name,
                    "description": metadata.get('description', ''),
                    "parameters": metadata.get('parameters_schema', {})
                }
            }
        }
    
    @classmethod
    def is_registered(cls, name: str, scope: Optional[str] = None) -> bool:
        """
        检查 Workflow 是否已注册（可选：在指定范围内）
        
        Args:
            name: Workflow 名称
            scope: 可选的应用范围
        
        Returns:
            bool: 是否已注册（在指定范围内）
        """
        if name not in cls._workflows:
            return False
        
        if scope is None:
            return True
        
        workflow_scopes = cls._workflow_scopes.get(name, set())
        return not workflow_scopes or scope in workflow_scopes
    
    @classmethod
    def clear(cls) -> None:
        """清空所有注册的 Workflow（主要用于测试）"""
        cls._workflows.clear()
        cls._workflow_scopes.clear()
        cls._workflow_metadata.clear()
        logger.debug("Cleared all registered workflows")


def register_workflow(
    name: Optional[str] = None,
    scopes: Optional[List[str]] = None
):
    """
    Workflow 注册装饰器
    
    使用方式：
        # 方式 1：直接使用（全局可见）
        @register_workflow()
        class MyWorkflow(OpBaseWorkflow):
            TOOL_NAME = "use_my_workflow"
            DESCRIPTION = "..."
            PARAMETERS_SCHEMA = {...}
        
        # 方式 2：指定名称
        @register_workflow(name="custom_name")
        class MyWorkflow(OpBaseWorkflow):
            ...
        
        # 方式 3：指定应用范围
        @register_workflow(scopes=["op"])
        class MyWorkflow(OpBaseWorkflow):
            ...
    
    Args:
        name: 可选的注册名称
        scopes: 可选的应用范围列表
    
    Returns:
        装饰器函数
    """
    def decorator(cls: Type[T]) -> Type[T]:
        return WorkflowRegistry.register(cls, name=name, scopes=scopes)
    
    return decorator
