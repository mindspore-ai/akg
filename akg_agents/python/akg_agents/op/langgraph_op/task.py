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

"""算子生成任务类

基于 LangGraph 的算子生成任务执行器，继承通用 BaseLangGraphTask。
"""

import logging
import os
from typing import Optional, Dict, Any, Tuple

from akg_agents.core_v2.langgraph_base.base_task import BaseLangGraphTask
from akg_agents.core.trace import Trace
from akg_agents.core.agent.designer import Designer
from akg_agents.core.agent.coder import Coder
from akg_agents.op.verifier.kernel_verifier import KernelVerifier
from akg_agents.core.async_pool.device_pool import DevicePool
from akg_agents.core.utils import check_task_config, check_task_type
from akg_agents.core.worker.manager import get_worker_manager

logger = logging.getLogger(__name__)

# 导入算子工作流
from akg_agents.op.workflows.default_workflow import DefaultWorkflow
from akg_agents.op.workflows.coder_only_workflow import CoderOnlyWorkflow
from akg_agents.op.workflows.verifier_only_workflow import VerifierOnlyWorkflow
from akg_agents.op.workflows.connect_all_workflow import ConnectAllWorkflow

# 算子工作流注册表（同时支持短名称和完整名称）
WORKFLOW_REGISTRY = {
    # 短名称
    "default": DefaultWorkflow,
    "coder_only": CoderOnlyWorkflow,
    "verifier_only": VerifierOnlyWorkflow,
    "connect_all": ConnectAllWorkflow,
    # 完整名称（与 Task 的 workflow 参数兼容）
    "default_workflow": DefaultWorkflow,
    "coder_only_workflow": CoderOnlyWorkflow,
    "verifier_only_workflow": VerifierOnlyWorkflow,
    "conductor_connect_all_workflow": ConnectAllWorkflow,
}


class LangGraphTask(BaseLangGraphTask):
    """基于 LangGraph 的算子生成任务执行器
    
    继承通用 BaseLangGraphTask，添加算子生成场景的专用逻辑：
    - 算子 Agent 初始化（Designer, Coder, Verifier）
    - 设备池和 Worker 管理
    - 算子专用初始状态准备
    """
    
    def __init__(self, 
                 op_name: str, 
                 task_desc: str, 
                 task_id: str, 
                 backend: str,
                 arch: str, 
                 dsl: str, 
                 config: dict, 
                 device_pool: Optional[DevicePool] = None, 
                 framework: str = "torch",
                 task_type: str = "precision_only", 
                 workflow: str = "default",
                 inspirations: Optional[list] = None, 
                 meta_prompts: Optional[str] = None,
                 handwrite_suggestions: Optional[list] = None,
                 source_backend: Optional[str] = None,
                 source_arch: Optional[str] = None,
                 user_requirements: Optional[str] = None):
        """初始化 LangGraphTask
        
        Args:
            op_name: 算子名称
            task_desc: 任务描述（框架代码）
            task_id: 任务ID
            backend: 后端名称
            arch: 架构名称
            dsl: DSL 类型
            config: 配置字典
            device_pool: 设备池（已废弃，使用 WorkerManager）
            framework: 框架名称
            task_type: 任务类型
            workflow: workflow 名称
            inspirations: 启发示例列表
            meta_prompts: 元提示
            handwrite_suggestions: 手写优化建议列表
            source_backend: 源后端，用于跨后端转换
            source_arch: 源架构，用于跨后端转换
            user_requirements: 用户额外需求（来自 ReAct 多轮对话）
        """
        # 验证任务配置
        normalized_dsl = check_task_config(framework, backend, arch, dsl)
        check_task_type(task_type)
        
        # 算子专用属性
        self.op_name = op_name
        self.task_desc = task_desc
        self.backend = backend.lower()
        self.arch = arch.lower()
        self.dsl = normalized_dsl.lower()
        self.framework = framework.lower()
        self.task_type = task_type
        self.device_pool = device_pool
        self.inspirations = inspirations
        self.meta_prompts = meta_prompts
        self.handwrite_suggestions = handwrite_suggestions or []
        self.source_backend = source_backend.lower() if source_backend else None
        self.source_arch = source_arch.lower() if source_arch else None
        self.user_requirements = user_requirements or config.get("user_requirements", "")
        
        # 调用父类初始化
        super().__init__(task_id, config, workflow)
        
        # 兼容旧代码：如果提供了 device_pool，创建私有 Worker
        self._private_worker = None
        if device_pool:
            import warnings
            warnings.warn(
                "⚠️  [DEPRECATED] 直接传递 device_pool 给 LangGraphTask() 是旧写法，将在未来版本移除。\n"
                "推荐的新写法：\n"
                "  1. 注册 LocalWorker 到 WorkerManager（一行代码）：\n"
                "     from akg_agents.core.worker.manager import register_local_worker\n"
                "     \n"
                "     await register_local_worker([0], backend='cuda', arch='a100')\n"
                "  2. 创建 LangGraphTask 时不传 device_pool：\n"
                "     task = LangGraphTask(\n"
                "         ...,\n"
                "         # device_pool=device_pool,  # 不再传递\n"
                "         ...\n"
                "     )\n"
                "参考示例：tests/st/test_langgraph_task_triton_cuda.py",
                DeprecationWarning,
                stacklevel=2
            )
            logger.warning("⚠️  检测到使用旧的 device_pool 参数，请参考日志中的警告信息迁移到新写法")
            
            from akg_agents.core.worker.local_worker import LocalWorker
            self._private_worker = LocalWorker(device_pool, backend=self.backend)
        
        self._worker_manager = get_worker_manager()
        
        # 工作流优先级：Task 参数 > 配置文件 > 默认值
        if workflow and workflow != "default":
            self.workflow_name = workflow
            logger.debug(f"[{op_name}] Using workflow from parameter: {self.workflow_name}")
        else:
            self.workflow_name = config.get("default_workflow", "default_workflow")
            logger.info(f"[{op_name}] Using workflow from config: {self.workflow_name}")
        
        # 初始化 Trace
        log_dir = config.get("log_dir")
        self.trace = Trace(op_name, task_id, log_dir)
        
        # 初始化 Agents
        self.agents = self._init_agents()
        
        # 初始化工作流
        self._init_workflow()
        
        logger.info(f"LangGraphTask initialized with workflow: {self.workflow_name}")
    
    def _init_agents(self) -> dict:
        """初始化算子 Agent（Designer, Coder, Verifier）"""
        agents = {}
        
        # 获取 parser 配置路径
        parser_config_path = self.config.get("parser_config_path")
        
        # Designer
        try:
            agents['designer'] = Designer(
                op_name=self.op_name,
                task_desc=self.task_desc,
                dsl=self.dsl,
                backend=self.backend,
                arch=self.arch,
                parser_config_path=parser_config_path,
                config=self.config
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Designer: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Coder
        try:
            logger.info(f"[LangGraphTask] _init_agents: self.config.get('rag')={self.config.get('rag')}")
            
            agents['coder'] = Coder(
                op_name=self.op_name,
                task_desc=self.task_desc,
                dsl=self.dsl,
                framework=self.framework,
                backend=self.backend,
                arch=self.arch,
                parser_config_path=parser_config_path,
                config=self.config,
                source_backend=self.source_backend,
                source_arch=self.source_arch
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Coder: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        # Verifier
        try:
            agents['verifier'] = KernelVerifier(
                op_name=self.op_name,
                framework_code=self.task_desc,
                task_id=self.task_id,
                framework=self.framework,
                dsl=self.dsl,
                backend=self.backend,
                arch=self.arch,
                config=self.config
            )
        except Exception as e:
            logger.warning(f"Failed to initialize Verifier: {e}")
            import traceback
            logger.debug(traceback.format_exc())
        
        return agents
    
    def _init_workflow(self):
        """初始化算子工作流"""
        workflow_class = WORKFLOW_REGISTRY.get(self.workflow_name, DefaultWorkflow)
        if workflow_class is None:
            logger.warning(f"Unknown workflow '{self.workflow_name}', using default")
            workflow_class = DefaultWorkflow
            
        self.workflow = workflow_class(
            agents=self.agents,
            device_pool=self.device_pool,
            private_worker=self._private_worker,
            worker_manager=self._worker_manager,
            backend=self.backend,
            arch=self.arch,
            trace=self.trace,
            config=self.config
        )
        
        # 编译图
        self.app = self.workflow.compile()
    
    def _prepare_initial_state(self, init_task_info: Optional[dict]) -> Dict[str, Any]:
        """准备算子专用初始状态"""
        # max_iterations 优先级：配置文件 > 默认值 20
        max_iterations = self.config.get("max_step", 20)

        # 获取 session_id（非 TUI 场景可为空；仅在流式输出开启时必填）
        session_id = str(self.config.get("session_id") or "").strip()
        stream_enabled = os.getenv("AKG_AGENTS_STREAM_OUTPUT", "off").lower() == "on"
        if stream_enabled and not session_id:
            raise ValueError("[LangGraphTask] config 中必须包含 session_id（AKG_AGENTS_STREAM_OUTPUT=on）！")
        task_label = str(self.config.get("task_label") or "none").strip()
        if not task_label:
            raise ValueError("[LangGraphTask] config 中必须包含 task_label")

        state = {
            # 通用字段（来自 BaseState）
            "task_id": self.task_id,
            "task_label": task_label,
            "iteration": 0,
            "step_count": 0,
            "max_iterations": max_iterations,
            "agent_history": [],
            "success": False,
            "error_message": None,
            
            # 算子专用字段
            "op_name": self.op_name,
            "task_desc": self.task_desc,
            "dsl": self.dsl,
            "framework": self.framework,
            "backend": self.backend,
            "arch": self.arch,
            "task_type": self.task_type,
            "verifier_result": False,
            "verifier_error": "",
            "history_attempts": [],
            "inspirations": self.inspirations,
            "meta_prompts": self.meta_prompts,
            "handwrite_suggestions": self.handwrite_suggestions,
            "user_requirements": self.user_requirements,
        }

        if session_id:
            state["session_id"] = session_id
        
        # 合并初始代码（如果有）
        if init_task_info:
            state.update(init_task_info)
        
        return state
    
    async def run(self, init_task_info: Optional[Dict[str, Any]] = None) -> Tuple[str, bool, dict]:
        """执行任务（API 兼容现有 Task）
        
        Args:
            init_task_info: 初始任务信息字典，包含初始代码等
            
        Returns:
            Tuple[str, bool, dict]: (算子名称, 是否成功, 最终状态)
        """
        try:
            # 准备初始状态
            initial_state = self._prepare_initial_state(init_task_info)
            max_iter = initial_state.get("max_iterations", 20)
            recursion_limit = max(25, max_iter * 3 + 5)
            
            logger.info(f"Task {self.task_id}, op_name: {self.op_name}")
            
            # 执行图
            final_state = await self.app.ainvoke(
                initial_state,
                config={"recursion_limit": recursion_limit}
            )
            # 处理结果
            success = final_state.get("verifier_result", False)
            
            logger.info(f"Task {self.task_id}, op_name: {self.op_name}, completed, success: {success}")
            
            return self.op_name, success, final_state
            
        except Exception as e:
            logger.error(f"LangGraphTask {self.task_id} failed: {e}")
            return self.op_name, False, {"error": str(e)}
    
    def visualize(self, output_path: str = None) -> str:
        """生成流程图
        
        Args:
            output_path: 输出路径（可选），如果提供则保存为 PNG
            
        Returns:
            Mermaid 格式的流程图字符串
        """
        from akg_agents.core_v2.langgraph_base.visualizer import WorkflowVisualizer
        
        if output_path:
            WorkflowVisualizer.save_png(self.app, output_path)
            return f"Workflow visualization saved to {output_path}"
        else:
            return WorkflowVisualizer.generate_mermaid(self.app)

