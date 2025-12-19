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
子 Agent 注册中心

提供即插即用的子 Agent 管理机制：
1. 注册新的子 Agent
2. 动态发现可用的子 Agent
3. 统一的调用接口
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from ai_kernel_generator.core.langgraph_task import LangGraphTask

logger = logging.getLogger(__name__)


class SubAgentBase(ABC):
    """
    子 Agent 基类
    
    所有子 Agent 必须继承这个类并实现 execute() 方法
    """
    
    def __init__(self, 
                 config: dict,
                 framework: str = "torch",
                 backend: str = "cuda",
                 arch: str = "a100",
                 dsl: str = "triton"):
        """
        初始化子 Agent
        
        Args:
            config: 配置字典
            framework: 框架类型
            backend: 后端类型
            arch: 硬件架构
            dsl: DSL 类型
        """
        self.config = config
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.dsl = dsl

    @abstractmethod
    async def execute(self, 
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        执行子 Agent
        
        Args:
            task_code: OpTaskBuildAgent 生成的 task 代码
            op_name: 算子名称
            task_id: 任务 ID
            **kwargs: 其他参数
            
        Returns:
            Tuple[bool, Dict[str, Any]]: (是否成功, 结果字典)
                结果字典应包含：
                - generated_code: 生成的代码
                - verification_result: 验证结果          
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """返回子 Agent 的名称"""
        pass
    
    @abstractmethod
    def get_description(self) -> str:
        """返回子 Agent 的描述（用于 LLM 决策）"""
        pass
    
    @abstractmethod
    def get_detailed_info(self) -> Dict[str, Any]:
        """
        返回子 Agent 的详细信息（用于 LLM 决策选择）
        
        Returns:
            Dict 包含：
            - name: 名称
            - description: 简短描述
            - workflow_steps: 工作流程步骤列表
            - use_cases: 适用场景列表
            - advantages: 优势列表
            - limitations: 限制/劣势列表
            - performance: 性能特点
        """
        pass


class CodeOnlySubAgent(SubAgentBase):
    """
    CodeOnly 子 Agent
    
    直接生成代码，跳过 Designer 阶段
    流程: Coder → Verifier
    """
    
    def get_name(self) -> str:
        return "codeonly"
    
    def get_description(self) -> str:
        return "Direct code generation without design phase (Coder → Verifier)"
    
    def get_detailed_info(self) -> Dict[str, Any]:
        """返回详细信息用于 LLM 决策"""
        return {
            "name": "codeonly",
            "description": "直接生成代码的标准流程",
            "workflow_steps": [
                "Coder: 使用 LLM 直接生成高性能算子代码",
                "Verifier: 编译、运行、验证生成的代码"
            ],
            "use_cases": [
                "各种算子实现（简单到中等复杂度）",
                "标准的数学运算算子（如 ReLU、Add、Mul、MatMul 等）",
                "大部分常见算子（如激活函数、归一化层等）",
                "需要快速生成和验证的场景",
                "算子逻辑清晰、实现直接的情况"
            ],
            "advantages": [
                "速度适中：直接生成代码，流程简洁",
                "流程简单：仅两个步骤（生成 + 验证）",
                "适用范围广：适合大多数标准算子",
                "资源消耗合理：LLM 调用次数适中",
                "成熟稳定：基于成熟的 Coder Agent"
            ],
            "limitations": [
                "不支持迭代优化：无法通过多轮优化提升性能",
                "性能可能不是最优：相比 evolve，缺少性能调优过程"
            ],
            "performance": "适中（约 30-90 秒），适合大部分算子的标准生成场景"
        }
    
    async def execute(self,
                     task_code: str,
                     op_name: str,
                     task_id: str,
                     **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """
        执行 codeonly workflow
        """
        logger.info(f"Executing CodeOnly sub-agent for {op_name}")

        try:
            # 延迟导入避免循环依赖
            from ai_kernel_generator.core.langgraph_task import LangGraphTask

            # 使用 LangGraphTask 调用 codeonly workflow
            task = LangGraphTask(
                op_name=op_name,
                task_desc=task_code,
                task_id=task_id,
                backend=self.backend,
                arch=self.arch,
                dsl=self.dsl,
                config=self.config,
                framework=self.framework,
                workflow="coder_only_workflow",  # codeonly 对应的 workflow
            )
            
            # 执行
            final_op_name, success, final_state = await task.run()
            
            # 提取结果
            result = {
                "generated_code": final_state.get("coder_code", ""),
                "verification_result": final_state.get("verifier_result", False),
                "verification_error": final_state.get("verifier_error", ""),
                "profile_result": final_state.get("profile_res"),
                "final_state": final_state  # 保存完整状态
            }
            
            return success, result
            
        except Exception as e:
            logger.error(f"CodeOnly sub-agent failed: {e}")
            return False, {
                "generated_code": "",
                "verification_result": False,
                "verification_error": str(e),
                "profile_result": None
            }

    # todo evolve sub-agent

class SubAgentRegistry:
    """
    子 Agent 注册中心
    
    管理所有可用的子 Agent
    """
    
    def __init__(self):
        """初始化注册中心"""
        self._agents: Dict[str, type] = {}
        
        # 自动注册内置的子 Agent
        self._register_builtin_agents()
    
    def _register_builtin_agents(self):
        """注册内置的子 Agent"""
        self.register(CodeOnlySubAgent)
        
        logger.info(f"Registered {len(self._agents)} built-in sub-agents")
    
    def register(self, agent_class: type):
        """
        注册一个子 Agent
        
        Args:
            agent_class: 子 Agent 类（必须继承 SubAgentBase）
        """
        if not issubclass(agent_class, SubAgentBase):
            raise ValueError(f"{agent_class} must inherit from SubAgentBase")
        
        # 创建临时实例获取名称
        temp_instance = agent_class(config={})
        agent_name = temp_instance.get_name()
        
        if agent_name in self._agents:
            logger.warning(f"Sub-agent '{agent_name}' already registered, overwriting")
        
        self._agents[agent_name] = agent_class
        logger.info(f"Registered sub-agent: {agent_name}")
    
    def get_agent(self, 
                  agent_name: str,
                  config: dict,
                  framework: str = "torch",
                  backend: str = "cuda",
                  arch: str = "a100",
                  dsl: str = "triton") -> Optional[SubAgentBase]:
        """
        获取子 Agent 实例
        
        Args:
            agent_name: 子 Agent 名称
            config: 配置字典
            framework: 框架类型
            backend: 后端类型
            arch: 硬件架构
            dsl: DSL 类型
            
        Returns:
            子 Agent 实例，如果不存在返回 None
        """
        agent_class = self._agents.get(agent_name)
        if agent_class is None:
            logger.error(f"Sub-agent '{agent_name}' not found")
            return None
        
        return agent_class(
            config=config,
            framework=framework,
            backend=backend,
            arch=arch,
            dsl=dsl
        )
    
    def list_agents(self) -> Dict[str, str]:
        """
        列出所有可用的子 Agent
        
        Returns:
            Dict[agent_name, description]
        """
        result = {}
        for agent_name, agent_class in self._agents.items():
            temp_instance = agent_class(config={})
            result[agent_name] = temp_instance.get_description()
        return result
    
    def get_agents_detailed_info(self) -> Dict[str, Dict[str, Any]]:
        """
        获取所有子 Agent 的详细信息（用于 LLM 决策）
        
        Returns:
            Dict[agent_name, detailed_info]
        """
        result = {}
        for agent_name, agent_class in self._agents.items():
            temp_instance = agent_class(config={})
            result[agent_name] = temp_instance.get_detailed_info()
        return result
    
    def is_registered(self, agent_name: str) -> bool:
        """检查子 Agent 是否已注册"""
        return agent_name in self._agents


# 全局注册中心实例
_global_registry = SubAgentRegistry()


def get_registry() -> SubAgentRegistry:
    """获取全局注册中心"""
    return _global_registry


def register_sub_agent(agent_class: type):
    _global_registry.register(agent_class)
    return agent_class

