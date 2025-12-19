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

"""UI 抽象层（与具体 TUI 实现解耦）。"""

from .commands import UICommand
from .factory import create_default_layout_manager
from .intents import UIIntent
from .protocols import LayoutManager
from .state import InfoPanelState, ReactiveState, WorkflowPanelState
from .types import MainContent

__all__ = [
    "LayoutManager",
    "UICommand",
    "UIIntent",
    "MainContent",
    "ReactiveState",
    "InfoPanelState",
    "WorkflowPanelState",
    "create_default_layout_manager",
]
