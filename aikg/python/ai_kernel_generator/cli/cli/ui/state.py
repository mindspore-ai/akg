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

import copy
from dataclasses import dataclass, field, fields
from typing import Any, Optional


class ReactiveState:
    """响应式 State 基类（提供 clone/diff/has_changes 的通用实现）。

    约定：子类必须是 dataclass。
    """

    def clone(self):  # type: ignore[no-untyped-def]
        """拷贝当前状态（尽量避免可变字段共享引用）。"""
        return type(self)(
            **{f.name: copy.deepcopy(getattr(self, f.name)) for f in fields(self)}
        )

    def diff(self, other):  # type: ignore[no-untyped-def]
        """计算与另一个状态的差异。

        Returns:
            {field_name: (old_value, new_value)}
        """
        if other is None:
            return {
                field.name: (None, getattr(self, field.name)) for field in fields(self)
            }

        changes: dict[str, tuple[Any, Any]] = {}
        for f in fields(self):
            old_val = getattr(other, f.name)
            new_val = getattr(self, f.name)
            if old_val != new_val:
                changes[f.name] = (old_val, new_val)
        return changes

    def has_changes(self, other) -> bool:  # type: ignore[no-untyped-def]
        """检查是否有变化。"""
        return len(self.diff(other)) > 0


@dataclass
class InfoPanelState(ReactiveState):
    """Info Panel 的状态模型（类型安全）"""

    framework: Optional[str] = None
    backend: Optional[str] = None
    arch: Optional[str] = None
    dsl: Optional[str] = None
    watch_task_id: Optional[str] = None


@dataclass
class WorkflowPanelState(ReactiveState):
    """Workflow Panel 的状态模型（结构化，渲染逻辑收敛在 widget 内）。"""

    workflow_name: str = ""
    current_node: str = ""
    current_node_status: str = ""  # running/done/idle
    seen_nodes: list[str] = field(default_factory=list)
    node_run_counts: dict[str, int] = field(default_factory=dict)
    current_node_run_no: int = 0
    progress: dict[str, Any] = field(default_factory=dict)
    watch_task_id: str = ""
