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

"""Verifier-only workflow: Verifier → Finish"""

from langgraph.graph import StateGraph, END
from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.langgraph_op.state import KernelGenState
from akg_agents.op.langgraph_op.nodes import NodeFactory
from akg_agents.core_v2.workflows.registry import register_workflow


@register_workflow(scopes=["op"])
class VerifierOnlyWorkflow(OpBaseWorkflow):
    """Verifier Only Workflow：仅验证已有代码
    
    Flow:
        verifier -> finish
    """
    
    # ========== 工具配置元数据（用于 KernelAgent 调用）==========
    TOOL_NAME = "use_verifier_only_workflow"
    
    DESCRIPTION = """
使用 VerifierOnly workflow 验证已有的 kernel 代码。

验证流程：
1. Verifier: 对比框架实现和生成代码的输出，验证数值精度和性能

适用场景：
- 已有生成的代码，需要验证其正确性
- 想测试某个手写或生成的 kernel 的性能
- 需要对比框架实现和自定义实现的差异
- 验证代码修改后是否仍然正确

与其他 workflow 的区别：
- Default/CoderOnly: 包含完整的生成+验证流程
- VerifierOnly: 仅验证，不生成代码

注意事项：
- 需要提供完整的框架代码（task_desc）和生成代码（result.code）
- 验证需要可用的设备环境（GPU/CPU）
- 验证时间取决于算子复杂度（通常 30秒-2分钟）
- 会输出正确性检查结果和性能对比

何时使用：
- 用户明确要求"验证"、"测试"已有代码时
- 用户提供了代码，想知道是否正确时
- 需要性能对比时
"""
    
    PARAMETERS_SCHEMA = {
        "type": "object",
        "properties": {
            "op_name": {
                "type": "string",
                "description": "算子名称，如 'relu', 'matmul', 'softmax'"
            },
            "task_desc": {
                "type": "string",
                "description": "任务描述（框架代码），必须包含框架实现的完整代码"
            },
            "generated_code": {
                "type": "string",
                "description": "待验证的生成代码，必须包含完整的实现"
            },
            "dsl": {
                "type": "string",
                "description": "生成代码的 DSL，如 'triton', 'cpp', 'cuda_c'"
            },
            "framework": {
                "type": "string",
                "description": "框架，如 'torch', 'mindspore', 'numpy'"
            },
            "backend": {
                "type": "string",
                "description": "后端，如 'cuda', 'cpu', 'npu'"
            },
            "arch": {
                "type": "string",
                "description": "架构，如 'a100', 'x86_64', 'ascend910b4'"
            },
            "task_id": {
                "type": "string",
                "description": "任务 ID（可选）",
                "default": ""
            }
        },
        "required": ["op_name", "task_desc", "generated_code", "dsl", "framework", "backend", "arch"]
    }
    
    # ========== Workflow 实现 ==========
    
    def build_graph(self) -> StateGraph:
        """构建 Verifier-only 工作流图"""
        workflow = StateGraph(KernelGenState)
        
        # 创建节点
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
        
        # 添加节点
        workflow.add_node("verifier", verifier_node)
        
        # 直接结束
        workflow.add_edge("verifier", END)
        
        # 设置入口
        workflow.set_entry_point("verifier")
        
        return workflow

