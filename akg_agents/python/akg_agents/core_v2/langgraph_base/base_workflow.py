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

"""通用 LangGraph 工作流基类

提供领域无关的工作流抽象，不包含任何特定领域逻辑。
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Optional
from langgraph.graph import StateGraph
import logging

logger = logging.getLogger(__name__)

# 状态类型变量，允许子类指定具体状态类型
StateType = TypeVar('StateType')


class BaseWorkflow(ABC, Generic[StateType]):
    """通用 LangGraph 工作流基类
    
    提供工作流的基础框架，不包含任何领域专用逻辑：
    - 无 agents 参数（由子类按需添加）
    - 无 device_pool、worker 等硬件相关参数
    - 无 conductor_template 等领域模板
    
    子类应继承此类并实现 build_graph() 方法。
    
    Example:
        class MyWorkflow(BaseWorkflow[MyState]):
            def __init__(self, my_agents: dict, config: dict):
                super().__init__(config)
                self.my_agents = my_agents
            
            def build_graph(self) -> StateGraph:
                workflow = StateGraph(MyState)
                # 添加节点和边
                return workflow
    """
    
    def __init__(self, config: dict, trace=None):
        """初始化工作流
        
        Args:
            config: 配置字典，包含 max_step 等通用配置
            trace: Trace 实例（可选，用于记录执行过程）
        """
        self.config = config
        self.trace = trace
        self.max_iterations = config.get("max_step", 20)
    
    @abstractmethod
    def build_graph(self) -> StateGraph:
        """构建工作流图
        
        子类必须实现此方法，定义具体的图结构。
        
        Returns:
            StateGraph: 未编译的 LangGraph StateGraph 实例
        """
        pass
    
    def compile(self):
        """编译工作流图
        
        Returns:
            编译后的 LangGraph 应用，可直接调用 ainvoke()
        """
        graph = self.build_graph()
        return graph.compile()
    
    def visualize(self) -> str:
        """生成 Mermaid 格式的流程图
        
        Returns:
            str: Mermaid 图表字符串
        """
        try:
            app = self.compile()
            return app.get_graph().draw_mermaid()
        except Exception as e:
            logger.warning(f"Failed to generate visualization: {e}")
            return "# Error generating diagram"

