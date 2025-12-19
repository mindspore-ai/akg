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

"""Widgets 组件包

这个包包含了 TUI 应用中使用的所有 widget 组件：

基础组件：
- LogPane: 日志区域组件
- InteractiveInput: 交互式输入框
- TraceListItem: Trace 列表项
- TraceListView: Trace 列表视图组件
- TaskTabs: 任务标签页组件

响应式面板：
- ReactivePanelBase: 响应式面板基类
- ReactiveInfoPanel: 响应式信息面板
- ReactiveWorkflowPanel: 响应式工作流面板
"""

from __future__ import annotations

from .log_pane import LogPane
from .interactive_input import InteractiveInput
from .trace_list_item import TraceListItem
from .trace_list_view import TraceListView
from .task_tabs import TaskTabs
from .reactive_panel_base import ReactivePanelBase
from .reactive_info_panel import ReactiveInfoPanel
from .reactive_workflow_panel import ReactiveWorkflowPanel

__all__ = [
    "LogPane",
    "InteractiveInput",
    "TraceListItem",
    "TraceListView",
    "TaskTabs",
    "ReactivePanelBase",
    "ReactiveInfoPanel",
    "ReactiveWorkflowPanel",
]
