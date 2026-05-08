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

"""通用 LangGraph 任务基类

提供任务执行的框架，不包含任何领域专用逻辑。
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseLangGraphTask(ABC):
    """通用 LangGraph 任务基类
    
    提供任务执行的框架，不包含任何领域专用逻辑：
    - 无算子相关参数（op_name, task_desc, dsl, framework 等）
    - 无 Agent 初始化逻辑
    - 无 device_pool、worker 等硬件相关参数
    - 无 WORKFLOW_REGISTRY
    
    子类应继承此类并实现：
    - _init_workflow(): 初始化工作流
    - _prepare_initial_state(): 准备初始状态
    
    Example:
        class MyTask(BaseLangGraphTask):
            def __init__(self, task_id: str, config: dict, my_param: str):
                super().__init__(task_id, config)
                self.my_param = my_param
                self._init_workflow()
            
            def _init_workflow(self):
                self.workflow = MyWorkflow(config=self.config)
                self.app = self.workflow.compile()
            
            def _prepare_initial_state(self, init_info):
                return {"task_id": self.task_id, "my_param": self.my_param}
    """
    
    def __init__(self, task_id: str, config: dict, workflow_name: str = "default"):
        """初始化任务
        
        Args:
            task_id: 任务唯一标识
            config: 配置字典
            workflow_name: 工作流名称（子类可用于选择不同工作流）
        """
        self.task_id = task_id
        self.config = config
        self.workflow_name = workflow_name
        
        # 子类需要在 _init_workflow() 中设置
        self.workflow = None
        self.app = None
    
    @abstractmethod
    def _init_workflow(self):
        """初始化工作流
        
        子类必须实现此方法，设置 self.workflow 和 self.app。
        """
        pass
    
    @abstractmethod
    def _prepare_initial_state(self, init_info: Optional[dict]) -> Dict[str, Any]:
        """准备初始状态
        
        子类必须实现此方法，返回工作流的初始状态。
        
        Args:
            init_info: 可选的初始化信息
            
        Returns:
            初始状态字典
        """
        pass
    
    async def run(self, init_info: Optional[Dict[str, Any]] = None) -> Tuple[bool, dict]:
        """执行任务
        
        Args:
            init_info: 可选的初始化信息
            
        Returns:
            Tuple[bool, dict]: (是否成功, 最终状态)
        """
        if self.app is None:
            raise RuntimeError("Workflow not initialized. Call _init_workflow() first.")
        
        try:
            # 准备初始状态
            initial_state = self._prepare_initial_state(init_info)
            max_iter = initial_state.get("max_iterations", 20)
            recursion_limit = max(25, max_iter * 3 + 5)
            
            logger.info(f"Task {self.task_id} starting...")
            
            # 执行工作流
            final_state = await self.app.ainvoke(
                initial_state,
                config={"recursion_limit": recursion_limit}
            )
            
            # 判断成功
            success = final_state.get("success", False)
            
            logger.info(f"Task {self.task_id} completed, success: {success}")
            
            return success, final_state
            
        except Exception as e:
            logger.error(f"Task {self.task_id} failed: {e}")
            return False, {"error": str(e)}
    
    def visualize(self, output_path: str = None) -> str:
        """生成流程图
        
        Args:
            output_path: 可选的输出路径，如果提供则保存为 PNG
            
        Returns:
            Mermaid 格式的流程图字符串
        """
        from akg_agents.core_v2.langgraph_base.visualizer import WorkflowVisualizer
        
        if self.app is None:
            return "# Workflow not initialized"
        
        if output_path:
            WorkflowVisualizer.save_png(self.app, output_path)
            return f"Workflow visualization saved to {output_path}"
        else:
            return WorkflowVisualizer.generate_mermaid(self.app)

