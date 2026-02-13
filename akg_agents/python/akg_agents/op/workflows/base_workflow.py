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

"""算子工作流基类

继承通用 BaseWorkflow，添加算子生成场景的专用逻辑。
"""

from pathlib import Path
from typing import Dict, Any, Optional

from akg_agents.core_v2.langgraph_base.base_workflow import BaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.utils.common_utils import get_prompt_path
from jinja2 import Template
import os
import logging
import tempfile

logger = logging.getLogger(__name__)

# DSL → docs_dir 映射表
# 来源：各 default_{dsl}_config.yaml 中的 docs_dir 配置
_DSL_DOCS_DIR_MAP: Dict[str, Dict[str, str]] = {
    "triton_ascend": {
        "designer": "op/resources/docs/sketch_docs",
        "coder": "op/resources/docs/triton_ascend_docs",
        "sketch": "op/resources/docs/sketch_docs",
    },
    "triton_cuda": {
        "designer": "op/resources/docs/sketch_docs",
        "coder": "op/resources/docs/triton_cuda_docs",
        "sketch": "op/resources/docs/sketch_docs",
    },
    "ascendc": {
        "designer": "op/resources/docs/ascendc_docs",
        "coder": "op/resources/docs/ascendc_docs",
    },
    "swft": {
        "designer": "op/resources/docs/aul_docs",
        "coder": "op/resources/docs/swft_docs",
    },
    "torch": {
        "designer": "op/resources/docs/torch_docs",
        "coder": "op/resources/docs/torch_docs",
    },
    "cuda_c": {
        "designer": "op/resources/docs/cuda_docs",
        "coder": "op/resources/docs/cuda_docs",
    },
    "cpp": {
        "designer": "op/resources/docs/cpu_docs",
        "coder": "op/resources/docs/cpu_docs",
    },
    "tilelang_cuda": {
        "designer": "op/resources/docs/tilelang_cuda_docs",
        "coder": "op/resources/docs/tilelang_cuda_docs",
    },
    "tilelang_npuir": {
        "designer": "op/resources/docs/sketch_docs",
        "coder": "op/resources/docs/tilelang_npuir_docs",
    },
}


class OpBaseWorkflow(BaseWorkflow[KernelGenState]):
    """算子工作流基类
    
    继承通用 BaseWorkflow，添加算子生成场景的专用属性：
    - agents: 算子 Agent 字典（designer, coder, verifier）
    - device_pool: 设备池（向后兼容）
    - private_worker / worker_manager: Worker 管理
    - conductor_template: Conductor 分析模板
    """
    
    def __init__(self, agents: dict, device_pool, trace, config: dict,
                 private_worker=None, worker_manager=None, backend=None, arch=None):
        """初始化算子工作流
        
        Args:
            agents: Agent 实例字典 (designer, coder, verifier)
            device_pool: 设备池（向后兼容）
            trace: Trace 实例（可以是旧的 Trace 或新的 TraceSystem）
            config: 配置字典
            private_worker: 私有 Worker 实例
            worker_manager: WorkerManager 实例
            backend: 后端类型（用于 Worker 选择）
            arch: 架构类型（用于 Worker 选择）
        """
        super().__init__(config, trace)
        
        # 算子专用属性
        self.agents = agents
        self.device_pool = device_pool  # 向后兼容
        self.private_worker = private_worker
        self.worker_manager = worker_manager
        self.backend = backend
        self.arch = arch
        
        # 加载 Conductor 模板
        self._load_conductor_template()
    
    def _load_conductor_template(self):
        """加载算子专用的 Conductor 模板"""
        try:
            prompt_dir = get_prompt_path()
            template_path = os.path.join(prompt_dir, "conductor/analyze.j2")
            with open(template_path, 'r', encoding='utf-8') as f:
                self.conductor_template = Template(f.read())
        except Exception as e:
            logger.warning(f"Failed to load conductor template: {e}")
            self.conductor_template = None
    
    @staticmethod
    def build_langgraph_task_config(
        dsl: str,
        backend: str,
        op_name: str = "",
        base_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """构建 LangGraphTask 所需的完整配置
        
        根据 DSL 和 backend 自动填充 docs_dir、agent_model_config、task_label 等配置，
        使 adaptive_search / evolve 内部创建的 LangGraphTask 能正确初始化 Agent。
        
        这些配置在原来的 run_single_adaptive_search.py / run_single_evolve.py 中
        由 load_config() + 手动补齐完成，现在硬编码到 workflow 中。
        
        Args:
            dsl: DSL 类型（如 'triton_ascend', 'triton_cuda'）
            backend: 后端（如 'cuda', 'ascend'）
            op_name: 算子名称（用于 task_label）
            base_config: 基础配置字典（已有的 config，新配置会在此基础上补充）
        
        Returns:
            完整的配置字典
        """
        config = dict(base_config or {})
        
        # 1. log_dir（若尚未设置）
        if "log_dir" not in config:
            log_root = os.path.expanduser("~/akg_agents_logs")
            config["log_dir"] = str(
                Path(log_root) / f"Task_{next(tempfile._get_candidate_names())}"
            )
        
        # 2. docs_dir —— 根据 DSL 硬编码映射
        if "docs_dir" not in config:
            dsl_lower = dsl.lower()
            docs_dir = _DSL_DOCS_DIR_MAP.get(dsl_lower)
            if docs_dir:
                config["docs_dir"] = dict(docs_dir)  # 拷贝
            else:
                logger.warning(
                    f"[OpBaseWorkflow] 未找到 DSL '{dsl}' 对应的 docs_dir 映射，"
                    f"Designer/Coder 可能无法加载文档"
                )
        
        # 3. agent_model_config —— 各 Agent 默认使用 "standard" 模型级别
        if "agent_model_config" not in config or not isinstance(
            config.get("agent_model_config"), dict
        ):
            config["agent_model_config"] = {}
        mc = config["agent_model_config"]
        default_level = mc.get("default") or "standard"
        mc.setdefault("default", default_level)
        for agent_name in [
            "designer", "coder", "conductor", "verifier", "selector", "op_task_builder"
        ]:
            mc.setdefault(agent_name, mc["default"])
        
        # 4. task_label
        if "task_label" not in config:
            try:
                from akg_agents.utils.task_label import resolve_task_label
                config["task_label"] = resolve_task_label(
                    op_name=op_name or "unknown",
                    parallel_index=1,
                )
            except Exception as e:
                logger.warning(f"[OpBaseWorkflow] 无法生成 task_label: {e}")
        
        # 5. profile_settings（硬编码，不允许用户或 LLM 修改）
        config["profile_settings"] = {
            "run_times": 50,
            "warmup_times": 5,
        }
        
        # 6. verify_timeout（硬编码，5 分钟）
        config["verify_timeout"] = 300
        
        # 7. default_workflow（硬编码）
        config["default_workflow"] = "default_workflow"
        
        # 8. max_step（硬编码）
        config["max_step"] = 20
        
        return config
    
    @classmethod
    def prepare_config(cls, workflow_resources: Dict[str, Any], arguments: Dict[str, Any]):
        """根据 arguments 调整 workflow 配置（Op 基础实现）
        
        功能：
        当指定了 cur_path 时（即被 KernelAgent 通过 ToolExecutor 调用）：
        1. 将 log_dir 重定向到 cur_path/logs
        2. 创建 WorkflowLogger 实例并注入到 workflow_resources["trace"]，
           替换掉原始的 TraceSystem 实例，使 workflow 内部的 nodes 统一使用
           WorkflowLogger 记录日志
        
        子类可以覆盖此方法添加额外逻辑，但应先调用 super().prepare_config()。
        
        Args:
            workflow_resources: workflow 资源字典（会被就地修改）
            arguments: 工具调用参数
        """
        # 确保 config 是一个独立副本
        workflow_resources["config"] = dict(workflow_resources.get("config") or {})
        config = workflow_resources["config"]
        
        cur_path = arguments.get("cur_path")
        if cur_path:
            log_dir = str(Path(cur_path) / "logs")
            config["log_dir"] = log_dir
            
            # 创建 WorkflowLogger
            # 使 workflow 内部的 nodes 统一使用 WorkflowLogger 记录日志
            from akg_agents.core_v2.workflow_logger import WorkflowLogger
            category = arguments.get("op_name", "unknown")
            task_id = arguments.get("task_id", "0")
            workflow_resources["trace"] = WorkflowLogger(
                log_dir=log_dir,
                category=category,
                task_id=task_id,
            )
            
            logger.info(f"[{cls.__name__}] cur_path 已设置，log_dir 重定向到: {log_dir}")
    
    @classmethod
    def build_initial_state(cls, arguments: Dict[str, Any], agent_context: Dict[str, Any]) -> Dict[str, Any]:
        """构建算子 workflow 的初始状态
        
        从 LLM 工具调用参数和 agent 上下文中提取所需字段，
        构建 KernelGenState 的初始值。
        
        Args:
            arguments: LLM 工具调用参数（已注入 cur_path）
            agent_context: Agent 上下文（包含硬件参数等）
        
        Returns:
            KernelGenState 初始状态字典
        """
        return {
            # 算子基础信息
            "op_name": arguments.get("op_name", ""),
            "task_desc": arguments.get("task_desc", ""),
            "dsl": arguments.get("dsl", agent_context.get("dsl", "")),
            "framework": arguments.get("framework", agent_context.get("framework", "")),
            "backend": arguments.get("backend", agent_context.get("backend", "")),
            "arch": arguments.get("arch", agent_context.get("arch", "")),
            "task_id": arguments.get("task_id", "0"),
            "user_requirements": arguments.get("user_requirements", ""),
            "previous_code": arguments.get("previous_code", ""),
            "cur_path": arguments.get("cur_path", ""),
            "task_type": arguments.get("task_type", "precision_only"),
            # 流程控制
            "iteration": 0,
            "step_count": 0,
            "max_iterations": arguments.get("max_iterations", 10),
            # Agent 输出（初始为空）
            "verifier_result": None,
            "verifier_error": None,
            "profile_res": {},
            "multi_case_error": None,
            "code_check_passed": None,
            "code_check_errors": None,
            "code_check_details": None,
            "conductor_suggestion": None,
            "conductor_decision": None,
            # 历史记录
            "agent_history": [],
            "history_attempts": [],
            # 会话信息
            "session_id": agent_context.get("session_id", ""),
        }

