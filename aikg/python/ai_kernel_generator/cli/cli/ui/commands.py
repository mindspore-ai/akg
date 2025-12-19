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

from __future__ import annotations

from dataclasses import dataclass

from .state import InfoPanelState, WorkflowPanelState


class UICommand:
    """Presenter/业务线程 -> UI 的命令（类型化）。"""


@dataclass(frozen=True)
class Quit(UICommand):
    pass


@dataclass(frozen=True)
class SetInput(UICommand):
    enabled: bool


@dataclass(frozen=True)
class Focus(UICommand):
    target: str


@dataclass(frozen=True)
class UpdateInfoState(UICommand):
    state: InfoPanelState


@dataclass(frozen=True)
class PatchInfoState(UICommand):
    patch: dict[str, object]


@dataclass(frozen=True)
class UpdateWorkflowState(UICommand):
    state: WorkflowPanelState


@dataclass(frozen=True)
class PatchWorkflowState(UICommand):
    patch: dict[str, object]


@dataclass(frozen=True)
class AppendTrace(UICommand):
    text: str
    task_id: str
    event_idx: int = 0


@dataclass(frozen=True)
class ClearTrace(UICommand):
    pass


@dataclass(frozen=True)
class SetTraceTitle(UICommand):
    title: str


@dataclass(frozen=True)
class SetTrace(UICommand):
    items: list[tuple[str, str, int]]


@dataclass(frozen=True)
class SetTaskTabs(UICommand):
    items: list[tuple[str, str]]
    active_task_id: str = ""


@dataclass(frozen=True)
class SetActiveTaskTab(UICommand):
    task_id: str
