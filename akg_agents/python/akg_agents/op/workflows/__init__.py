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
op/workflows - 算子专用工作流

提供算子生成场景的工作流定义：
- OpBaseWorkflow: 算子工作流基类
- DefaultWorkflow: 默认工作流 (Designer → Coder ↔ Verifier)
- CoderOnlyWorkflow: 仅 Coder 工作流
- VerifierOnlyWorkflow: 仅 Verifier 工作流
- ConnectAllWorkflow: 全连接工作流
- OpTaskBuilderWorkflow: OpTaskBuilder 工作流
"""

from akg_agents.op.workflows.base_workflow import OpBaseWorkflow
from akg_agents.op.workflows.default_workflow import DefaultWorkflow
from akg_agents.op.workflows.coder_only_workflow import CoderOnlyWorkflow
from akg_agents.op.workflows.verifier_only_workflow import VerifierOnlyWorkflow
from akg_agents.op.workflows.connect_all_workflow import ConnectAllWorkflow
from akg_agents.op.workflows.op_task_builder_workflow import (
    OpTaskBuilderWorkflow,
    run_op_task_builder,
)

__all__ = [
    "OpBaseWorkflow",
    "DefaultWorkflow",
    "CoderOnlyWorkflow",
    "VerifierOnlyWorkflow",
    "ConnectAllWorkflow",
    "OpTaskBuilderWorkflow",
    "run_op_task_builder",
]

