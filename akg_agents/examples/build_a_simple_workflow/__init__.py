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

"""文档修复工作流示例

演示如何基于 langgraph_base 构建一个简单的多 Agent 工作流。

本示例可以直接通过 run_example.py 运行，不需要作为包导入。

使用方法:
    python examples/build_a_simple_workflow/run_example.py
"""

# 注意：由于本示例设计为独立运行脚本，不使用相对导入
# 如需作为包使用，请参考各模块中的导入方式

__all__ = [
    "DocFixerState",
    "DocFixerWorkflow", 
    "DocFixerTask",
]

