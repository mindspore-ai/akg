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
工具函数遵循 akg_agents v2 规范（关键字参数、标准返回格式）。
"""

from akg_agents.op.tools.task_constructor.tool_registry import TaskToolRegistry

# 导入以触发工具注册（模块加载时自动注册）
import akg_agents.op.tools.task_constructor.code_tools  # noqa: F401
import akg_agents.op.tools.task_constructor.file_tools  # noqa: F401

__all__ = ["TaskToolRegistry"]
