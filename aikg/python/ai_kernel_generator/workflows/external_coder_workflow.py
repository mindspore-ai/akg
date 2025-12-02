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
External Coder Workflow

使用外部代码生成器（如 RL 训练的模型）的工作流。
流程：ExternalCoder → Verifier → (失败) → Conductor → ExternalCoder → ...

这是一个 Coder-Only 的变体，用外部代码生成器替换内置 Coder。
"""

from langgraph.graph import StateGraph, END
from ai_kernel_generator.workflows.base_workflow import BaseWorkflow
from ai_kernel_generator.utils.langgraph.state import KernelGenState
from ai_kernel_generator.utils.langgraph.nodes import NodeFactory
from ai_kernel_generator.utils.langgraph.routers import RouterFactory

import logging

logger = logging.getLogger(__name__)


class ExternalCoderWorkflow(BaseWorkflow):
    """
    外部代码生成器工作流
    
    用于接入外部训练的代码生成模型（如 RL 模型）。
    跳过 Designer 阶段，直接使用外部 Coder 生成代码。
    
    Flow:
        external_coder -> verifier
              ^              |
              |______________|
            (if verification fails, via conductor)
    
    配置:
        在 config 中需要包含 external_coder 配置：
        ```yaml
        external_coder:
          class: "ai_kernel_generator.core.agent.external_coders.triton_ascend_rl_coder.TritonAscendRLCoder"
          prompt_template_path: "path/to/template.j2"
          model_name: "vllm_xxx"
        ```
    """
    
    def __init__(self, agents: dict, device_pool, trace, config: dict,
                 private_worker=None, worker_manager=None, backend=None, arch=None):
        """
        初始化 ExternalCoderWorkflow
        
        Args:
            agents: Agent 实例字典 (此 workflow 会忽略 designer 和 coder)
            device_pool: 设备池
            trace: Trace 实例
            config: 配置字典（必须包含 external_coder 配置）
            private_worker: 私有 Worker 实例
            worker_manager: WorkerManager 实例
            backend: 后端类型
            arch: 架构类型
        """
        super().__init__(
            agents=agents,
            device_pool=device_pool,
            trace=trace,
            config=config,
            private_worker=private_worker,
            worker_manager=worker_manager,
            backend=backend,
            arch=arch
        )
        
        # 加载外部代码生成器
        self.external_coder = self._load_external_coder()
        
        if self.external_coder is None:
            raise ValueError(
                "Failed to load external coder. "
                "Please check your config.external_coder configuration."
            )
        
        logger.info(f"ExternalCoderWorkflow initialized with {self.external_coder.__class__.__name__}")
    
    def _load_external_coder(self):
        """
        动态加载外部代码生成器
        
        从 config.external_coder.class 中读取类路径，动态加载并实例化。
        
        Returns:
            外部代码生成器实例
        """
        import importlib
        
        external_config = self.config.get("external_coder", {})
        class_path = external_config.get("class")
        
        if not class_path:
            logger.error("No external_coder.class configured")
            return None
        
        try:
            # 解析模块路径和类名
            module_path, class_name = class_path.rsplit(".", 1)
            
            # 动态导入模块
            module = importlib.import_module(module_path)
            
            # 获取类
            coder_class = getattr(module, class_name)
            
            # 从 agents 中获取必要的信息（如果有的话）
            # 否则从 config 中读取
            coder = self.agents.get('coder')
            if coder:
                op_name = coder.op_name
                task_desc = coder.task_desc
                dsl = coder.dsl
                framework = coder.framework
                backend = coder.backend
                arch = coder.arch
            else:
                # 尝试从 config 或其他来源获取
                op_name = external_config.get("op_name", "unknown")
                task_desc = external_config.get("task_desc", "")
                dsl = external_config.get("dsl", "triton_ascend")
                framework = external_config.get("framework", "torch")
                backend = self.backend or external_config.get("backend", "ascend")
                arch = self.arch or external_config.get("arch", "")
            
            # 实例化外部代码生成器
            external_coder = coder_class(
                op_name=op_name,
                task_desc=task_desc,
                dsl=dsl,
                framework=framework,
                backend=backend,
                arch=arch,
                config=self.config
            )
            
            logger.info(f"Loaded external coder: {class_name}")
            return external_coder
            
        except Exception as e:
            logger.error(f"Failed to load external coder from {class_path}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def build_graph(self) -> StateGraph:
        """
        构建外部代码生成器工作流图
        
        复用 create_coder_node，因为 ExternalCoderBase 和 Coder 有相同的接口：
        run(task_info) -> (code, prompt, reasoning)
        
        Returns:
            StateGraph 实例
        """
        workflow = StateGraph(KernelGenState)
        
        # 创建节点（复用 create_coder_node，传入外部 Coder 实例）
        coder_node = NodeFactory.create_coder_node(
            self.external_coder,  # ExternalCoderBase 实例，接口与 Coder 兼容
            self.trace
        )
        
        verifier_node = NodeFactory.create_verifier_node(
            self.agents['verifier'],
            self.device_pool,
            self.trace,
            self.config,
            self.private_worker,
            self.worker_manager,
            self.backend,
            self.arch
        )
        
        conductor_node = NodeFactory.create_conductor_node(
            self.trace,
            self.config,
            self.conductor_template
        )
        
        # 添加节点
        workflow.add_node("coder", coder_node)  # 节点名仍为 "coder"，保持兼容
        workflow.add_node("verifier", verifier_node)
        workflow.add_node("conductor", conductor_node)
        
        # 添加边
        workflow.add_edge("coder", "verifier")
        
        # 条件边：verifier 后的路由
        verifier_router = RouterFactory.create_verifier_router_with_conductor(
            self.config
        )
        
        workflow.add_conditional_edges(
            "verifier",
            verifier_router,
            {
                "conductor": "conductor",  # 验证失败 → Conductor 分析
                "finish": END              # 验证通过 → 直接结束
            }
        )
        
        # Conductor 后的路由
        conductor_router = RouterFactory.create_conductor_router(self.config)
        
        workflow.add_conditional_edges(
            "conductor",
            conductor_router,
            {
                "coder": "coder",  # 继续修复
                "finish": END
            }
        )
        
        # 设置入口
        workflow.set_entry_point("coder")
        
        return workflow

