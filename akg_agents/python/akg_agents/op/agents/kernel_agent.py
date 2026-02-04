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
KernelAgent - 算子生成 ReAct Agent

基于 ReActAgent 基类，专门用于算子内核代码生成任务。

功能：
- 支持多种 DSL（Triton, CUDA, AscendC 等）
- 支持多种 Backend（CUDA, Ascend 等）
- 支持多种 Framework（PyTorch, MindSpore 等）
- 动态加载 op 相关的 agents 和 workflows
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

from akg_agents.core_v2.agents.react_agent import ReActAgent
from akg_agents.core_v2.agents.base import Jinja2TemplateWrapper
from akg_agents.core_v2.agents.registry import register_agent

logger = logging.getLogger(__name__)


@register_agent(scopes=["op"])
class KernelAgent(ReActAgent):
    """
    算子生成 ReAct Agent
    
    继承 ReActAgent，实现算子特定的配置和逻辑。
    """
    
    def __init__(
        self,
        task_id: str,
        model_level: str = None,
        config: Dict = None,
        framework: str = "torch",
        backend: str = "cuda",
        arch: str = "a100",
        dsl: str = "triton",
        base_dir: Optional[str] = None
    ):
        """
        初始化 KernelAgent
        
        Args:
            task_id: 任务 ID
            model_level: 模型级别
            config: 配置信息
            framework: 框架（如 "torch", "mindspore"）
            backend: 后端（如 "cuda", "ascend"）
            arch: 架构（如 "a100", "ascend910b4"）
            dsl: DSL（如 "triton", "cuda_c", "ascendc"）
            base_dir: 基础目录
        """
        # 保存算子特定配置（在调用父类初始化之前）
        self.framework = framework
        self.backend = backend
        self.arch = arch
        self.dsl = dsl
        
        # Workflow 资源（延迟初始化）
        self._workflow_resources = None
        
        # 调用父类初始化
        super().__init__(
            task_id=task_id,
            model_level=model_level,
            config=config,
            base_dir=base_dir
        )
    
    # ==================== 实现抽象方法 ====================
    
    def _get_agent_name(self) -> str:
        """获取 Agent 名称"""
        return "KernelAgent"
    
    def _load_prompt_template(self) -> Jinja2TemplateWrapper:
        """加载 prompt 模板"""
        # 从 op/resources/prompts/kernel_agent/ 加载模板
        from akg_agents import get_project_root
        prompt_file = Path(get_project_root()) / "op" / "resources" / "prompts" / "kernel_agent" / "system.j2"
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            return Jinja2TemplateWrapper(f.read())
    
    def _build_prompt_context(self) -> Dict[str, Any]:
        """构建 prompt 上下文变量"""
        return {
            "framework": self.framework,
            "backend": self.backend,
            "arch": self.arch,
            "dsl": self.dsl,
        }
    
    def _get_agent_context(self) -> Dict[str, Any]:
        """获取 agent 上下文（传递给 ToolExecutor）"""
        return {
            "task_id": self.task_id,
            "dsl": self.dsl,
            "framework": self.framework,
            "backend": self.backend,
            "arch": self.arch,
            "model_level": self.model_level or "standard",
            # 提供获取 workflow 资源的回调
            "get_workflow_resources": lambda: self._get_workflow_resources()
        }
    
    def _load_available_tools(self) -> List[Dict]:
        """加载可用工具列表"""
        from akg_agents import get_project_root
        tools_file = Path(get_project_root()) / "core_v2" / "config" / "tools.yaml"
        
        with open(tools_file, "r", encoding="utf-8") as f:
            tools_config = yaml.safe_load(f)
        
        available_tools = []
        for tool_name, tool_def in tools_config.get("tools", {}).items():
            func = tool_def.get("function", {})
            if func:
                available_tools.append({
                    "type": "function",
                    "function": func
                })
        return available_tools
    
    # ==================== 覆盖可选方法 ====================
    
    def _load_agent_registry(self) -> Dict[str, Any]:
        """加载 Agent 注册表（包含 op agents）"""
        from akg_agents.core_v2.agents.registry import AgentRegistry
        
        # 导入 plan agent
        try:
            from akg_agents.core_v2.agents import plan  # noqa: F401
        except Exception as e:
            logger.warning(f"[KernelAgent] 导入 plan 失败: {e}")
        
        # 导入 op agents
        try:
            from akg_agents.op.agents import kernel_gen, kernel_designer, op_task_builder  # noqa: F401
        except Exception as e:
            logger.warning(f"[KernelAgent] 导入 op.agents 失败: {e}")
        
        agent_registry = {}
        all_agent_names = AgentRegistry.list_agents()
        logger.info(f"[KernelAgent] 发现 {len(all_agent_names)} 个已注册 agents")
        
        for agent_name in all_agent_names:
            try:
                agent_class = AgentRegistry.get_agent_class(agent_name)
                if not hasattr(agent_class, 'TOOL_NAME') or not agent_class.TOOL_NAME:
                    logger.debug(f"[KernelAgent] Agent '{agent_name}' 没有 TOOL_NAME，跳过")
                    continue
                
                # 加载 agent 的工具配置
                for tool_name, tool_def in agent_class.load_tool_config().items():
                    agent_registry[tool_name] = {
                        "agent_class": agent_class,
                        "config": tool_def
                    }
                    self.available_tools.append({
                        "type": "function",
                        "function": tool_def.get("function", {})
                    })
                    logger.info(f"[KernelAgent] 注册工具: {tool_name} (来自 {agent_name})")
            except Exception as e:
                logger.warning(f"[KernelAgent] 加载 Agent '{agent_name}' 失败: {e}", exc_info=True)
        
        return agent_registry
    
    def _load_workflow_registry(self) -> Dict[str, Any]:
        """动态加载所有注册的 Workflow"""
        from akg_agents.core_v2.workflows.registry import WorkflowRegistry
        
        # 导入所有 workflow 模块（触发 @register_workflow 装饰器）
        try:
            from akg_agents.op.workflows import (
                coder_only_workflow,      # noqa: F401
                default_workflow,         # noqa: F401
                verifier_only_workflow,   # noqa: F401
                connect_all_workflow,     # noqa: F401
            )
        except Exception as e:
            logger.warning(f"[KernelAgent] 导入 workflows 失败: {e}")
        
        workflow_registry = {}
        all_workflow_names = WorkflowRegistry.list_workflows(scope="op")
        logger.info(f"[KernelAgent] 发现 {len(all_workflow_names)} 个已注册 workflows")
        
        for workflow_name in all_workflow_names:
            try:
                workflow_class = WorkflowRegistry.get_workflow_class(workflow_name)
                tool_config = WorkflowRegistry.get_tool_config(workflow_name)
                
                if not tool_config:
                    logger.debug(f"[KernelAgent] Workflow '{workflow_name}' 没有工具配置，跳过")
                    continue
                
                # 注册到 registry
                for tool_name, tool_def in tool_config.items():
                    workflow_registry[tool_name] = {
                        "workflow_class": workflow_class,
                        "workflow_name": workflow_name,
                        "config": tool_def
                    }
                    
                    # 添加到 available_tools
                    self.available_tools.append({
                        "type": "function",
                        "function": tool_def.get("function", {})
                    })
                    
                    logger.info(f"[KernelAgent] 注册工具: {tool_name} (来自 workflow {workflow_name})")
            
            except Exception as e:
                logger.warning(f"[KernelAgent] 加载 Workflow '{workflow_name}' 失败: {e}", exc_info=True)
        
        return workflow_registry
    
    def _get_task_info_extra(self) -> Dict[str, Any]:
        """获取任务信息的额外字段"""
        return {
            "op_name": "",
            "dsl": self.dsl,
            "backend": self.backend,
            "arch": self.arch
        }
    
    # ==================== 算子特定方法 ====================
    
    def _get_workflow_resources(self) -> Dict[str, Any]:
        """
        获取 workflow 所需资源（延迟初始化）
        
        Returns:
            包含 agents, trace, config 等的资源字典
        """
        if self._workflow_resources is None:
            logger.info("[KernelAgent] 初始化 workflow 资源...")
            
            # 初始化 agents（延迟加载，避免循环依赖）
            from akg_agents.core.agent.designer import Designer
            from akg_agents.core.agent.coder import Coder
            from akg_agents.op.verifier.kernel_verifier import KernelVerifier
            
            # 创建 agents 配置
            agent_config = {
                "dsl": self.dsl,
                "backend": self.backend,
                "arch": self.arch,
                "framework": self.framework
            }
            
            try:
                agents = {
                    "designer": Designer(**agent_config),
                    "coder": Coder(**agent_config),
                    "verifier": KernelVerifier(backend=self.backend, arch=self.arch)
                }
                logger.info(f"[KernelAgent] 成功初始化 {len(agents)} 个 agents")
            except Exception as e:
                logger.error(f"[KernelAgent] 初始化 agents 失败: {e}", exc_info=True)
                # 提供空的 agents 作为降级
                agents = {}
            
            self._workflow_resources = {
                "agents": agents,
                "device_pool": None,  # 向后兼容
                "trace": self.trace,
                "config": {},  # 可以从配置文件加载
                "private_worker": None,
                "worker_manager": None,
                "backend": self.backend,
                "arch": self.arch
            }
        
        return self._workflow_resources
