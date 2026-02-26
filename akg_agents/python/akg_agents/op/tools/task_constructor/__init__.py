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
任务构造器工具集

包含代码操作工具和文件操作工具，供 TaskConstructor agent 内部 ReAct 循环使用。
所有工具通过统一的 ToolRegistry 注册和管理。
"""

from akg_agents.core_v2.tools.tool_registry import ToolRegistry

# 导入以触发工具注册（模块加载时自动注册到 ToolRegistry）
import akg_agents.op.tools.task_constructor.code_tools  # noqa: F401
import akg_agents.op.tools.task_constructor.file_tools  # noqa: F401

__all__ = ["ToolRegistry"]
