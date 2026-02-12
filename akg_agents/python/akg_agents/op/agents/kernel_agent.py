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
- 基于 Skill 系统进行工具选择指导
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import yaml

from akg_agents.core_v2.agents.react_agent import ReActAgent
from akg_agents.core_v2.agents.base import Jinja2TemplateWrapper
from akg_agents.core_v2.agents.registry import register_agent
from akg_agents.core_v2.skill import SkillRegistry, SkillLevel

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
        
        # Skill 相关（延迟初始化）
        self._skill_registry: Optional[SkillRegistry] = None
        self._skills_content: Optional[str] = None  # 缓存的 Skills 内容
        
        # 调用父类初始化
        super().__init__(
            task_id=task_id,
            model_level=model_level,
            config=config,
            base_dir=base_dir
        )
        
        # 初始化后加载 Skills
        self._load_skills()
    
    # ==================== 实现抽象方法 ====================
    
    def _get_agent_name(self) -> str:
        """获取 Agent 名称"""
        return "KernelAgent"
    
    def _load_prompt_template(self) -> Jinja2TemplateWrapper:
        """加载 prompt 模板（使用包含 Skills 的 ReAct 版本）"""
        # 从 op/resources/prompts/kernel_agent/ 加载模板
        from akg_agents import get_project_root
        # 使用 system_react.j2，支持 Skills 注入
        prompt_file = Path(get_project_root()) / "op" / "resources" / "prompts" / "kernel_agent" / "system_react.j2"
        
        with open(prompt_file, "r", encoding="utf-8") as f:
            return Jinja2TemplateWrapper(f.read())
    
    def _build_prompt_context(self) -> Dict[str, Any]:
        """构建 prompt 上下文变量（包含 Skills 内容）"""
        return {
            "framework": self.framework,
            "backend": self.backend,
            "arch": self.arch,
            "dsl": self.dsl,
            # 注入 Skills 内容作为工具选择指南
            "skills_guide": self._get_skills_guide(),
        }
    
    def _get_agent_context(self) -> Dict[str, Any]:
        """获取 agent 上下文（传递给 ToolExecutor）"""
        ctx = {
            "task_id": self.task_id,
            "dsl": self.dsl,
            "framework": self.framework,
            "backend": self.backend,
            "arch": self.arch,
            "model_level": self.model_level or "standard",
            # 提供获取 workflow 资源的回调
            "get_workflow_resources": lambda: self._get_workflow_resources()
        }
        # 透传 session_id，使 workflow 内的 agent 能通过流式输出推送到 CLI
        session_id = self.context.get("session_id", "")
        if session_id:
            ctx["session_id"] = session_id
        return ctx
    
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
        
        # 导入 task constructor agent
        try:
            from akg_agents.op.agents import task_constructor  # noqa: F401
        except Exception as e:
            logger.warning(f"[KernelAgent] 导入 task_constructor 失败: {e}")
        
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
        
        # 导入 workflows（触发 @register_workflow 装饰器）
        try:
            from akg_agents.op.workflows import (
                kernelgen_only_workflow,  # noqa: F401  基于 Skill 系统的代码生成
                evolve_workflow,          # noqa: F401  进化式算子生成
                adaptive_search_workflow, # noqa: F401  自适应搜索算子生成
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
    
    def _get_domain(self) -> str:
        """返回 op 领域标识"""
        return "op"
    
    # ==================== 配置同步 ====================
    
    def _on_hardware_config_updated(self, updated: Dict[str, str]):
        """
        硬件配置更新回调
        
        当 LLM 在工具调用中指定了新的 backend/dsl/arch/framework 时，
        同步到 KernelAgent 自身属性，确保:
        1. prompt context 中的「当前配置」显示最新值
        2. workflow 资源使用正确的硬件配置
        
        Args:
            updated: 已更新的参数字典
        """
        for key, value in updated.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 使 workflow 资源失效，下次调用时重新创建
        # （因为 verifier 等组件可能依赖硬件配置）
        if self._workflow_resources is not None:
            logger.info(f"[KernelAgent] 硬件配置变更，workflow 资源将在下次调用时重建")
            self._workflow_resources = None
        
        # 清除 skills 缓存（不同 backend 可能需要不同 skills）
        self._skills_content = None
    
    # ==================== Skill 系统方法 ====================
    
    def _load_skills(self) -> None:
        """
        加载 KernelAgent 相关的 Skills
        
        Skills 目录结构：
        - kernel-agent-overview (L1): 工具体系概述和选择策略
        """
        from akg_agents import get_project_root
        
        try:
            # 创建 Registry 并加载 Skills
            self._skill_registry = SkillRegistry()
            skills_dir = Path(get_project_root()) / "op" / "resources" / "skills" / "kernel-agent"
            
            if skills_dir.exists():
                count = self._skill_registry.load_from_directory(skills_dir)
                logger.info(f"[KernelAgent] 成功加载 {count} 个 Skills from {skills_dir}")
                
                # 打印加载的 Skills 统计
                stats = self._skill_registry.get_statistics()
                logger.debug(f"[KernelAgent] Skills 统计: {stats}")
            else:
                logger.warning(f"[KernelAgent] Skills 目录不存在: {skills_dir}")
                
        except Exception as e:
            logger.error(f"[KernelAgent] 加载 Skills 失败: {e}", exc_info=True)
            self._skill_registry = None
    
    def _get_skills_guide(self) -> str:
        """
        获取 Skills 指南内容（用于注入 prompt）
        
        Returns:
            Skills 指南内容（Markdown 格式）
        """
        # 使用缓存
        if self._skills_content is not None:
            return self._skills_content
        
        if not self._skill_registry:
            self._skills_content = ""
            return self._skills_content
        
        try:
            # 获取 L1 overview skill
            overview_skill = self._skill_registry.get("kernel-agent-overview")
            
            if overview_skill:
                self._skills_content = overview_skill.content
                logger.info(f"[KernelAgent] 已加载工具选择指南: {overview_skill.name}")
            else:
                self._skills_content = ""
                logger.warning("[KernelAgent] 未找到任何可用的 Skills 指南")
            
        except Exception as e:
            logger.error(f"[KernelAgent] 获取 Skills 指南失败: {e}", exc_info=True)
            self._skills_content = ""
        
        return self._skills_content
    
    def get_skill_by_name(self, name: str) -> Optional[Any]:
        """
        根据名称获取 Skill（供外部调用）
        
        Args:
            name: Skill 名称
        
        Returns:
            SkillMetadata 对象，不存在返回 None
        """
        if self._skill_registry:
            return self._skill_registry.get(name)
        return None
    
    def get_skills_by_level(self, level: SkillLevel) -> List[Any]:
        """
        根据层级获取 Skills（供外部调用）
        
        Args:
            level: Skill 层级（L1, L2, L3 等）
        
        Returns:
            该层级的 Skill 列表
        """
        if self._skill_registry:
            return self._skill_registry.get_by_level(level)
        return []
    
    # ==================== 算子特定方法 ====================
    
    def _get_workflow_resources(self) -> Dict[str, Any]:
        """
        获取 workflow 所需资源（延迟初始化）
        
        Returns:
            包含 agents, trace, config 等的资源字典
        """
        if self._workflow_resources is None:
            logger.info("[KernelAgent] 初始化 workflow 资源...")
            
            from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
            
            workflow_config = OpBaseWorkflow.build_langgraph_task_config(
                dsl=self.dsl,
                backend=self.backend,
            )
            
            # 初始化 agents（延迟加载，避免循环依赖）
            from akg_agents.op.agents.kernel_gen import KernelGen
            from akg_agents.op.verifier.kernel_verifier import KernelVerifier
            
            try:
                # 创建 KernelGen（不需要 op 特定参数）
                kernel_gen = KernelGen()
                
                # 创建 KernelVerifier（使用占位符，实际值会在 verifier node 运行时从 state 更新）
                verifier = KernelVerifier(
                    op_name="placeholder",  # 会被 verifier node 从 state 更新
                    framework_code="",      # 会被 verifier node 从 state 更新
                    task_id="0",
                    framework=self.framework,
                    dsl=self.dsl,
                    backend=self.backend,
                    arch=self.arch,
                    config=workflow_config   # 使用完整 config
                )
                
                agents = {
                    "kernel_gen": kernel_gen,
                    "verifier": verifier
                }
                logger.info(f"[KernelAgent] 成功初始化 {len(agents)} 个 agents: {list(agents.keys())}")
            except Exception as e:
                logger.error(f"[KernelAgent] 初始化 agents 失败: {e}", exc_info=True)
                agents = {}
            
            # 使用全局 WorkerManager 管理 worker
            from akg_agents.core.worker.manager import get_worker_manager
            
            self._workflow_resources = {
                "agents": agents,
                "device_pool": None,
                "trace": self.trace,
                "config": workflow_config,   # 使用完整 config
                "private_worker": None,
                "worker_manager": get_worker_manager(),
                "backend": self.backend,
                "arch": self.arch
            }
        
        return self._workflow_resources
