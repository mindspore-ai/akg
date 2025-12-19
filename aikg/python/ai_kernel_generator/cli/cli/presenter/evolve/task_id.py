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

from typing import Any

from ai_kernel_generator.cli.cli.constants import NodeName


class TaskIdPolicy:
    """task_id 解析/归一化与“是否渲染”策略（依赖 presenter 配置与 watch 选择）。"""

    def __init__(self, presenter: Any, *, get_watch_task_id) -> None:
        self._p = presenter
        self._get_watch_task_id = get_watch_task_id

    def should_render_task(self, task_id: str) -> bool:
        """只渲染当前 watch 的任务（未选择 watch 时不做过滤）。"""
        tid = (task_id or "").strip()

        # 兼容旧 server：没有 task_id 时不做过滤
        if not tid:
            return True

        # 如果用户已切换观察目标：直接按 task_id 精确过滤（切换后持续追底）
        watch_task_id = str(self._get_watch_task_id() or "").strip()
        if watch_task_id:
            return tid == watch_task_id
        return True

    def normalize_task_id(self, raw_task_id: str, *, node_hint: str = "") -> str:
        """
        统一 task_id 归一化（解决“task_init + default 多一栏”的历史问题）：
        - TaskInit 阶段永远归到 "task_init"
        - 如果 task_id 为空且已有 watch_task_id：归到 watch_task_id（避免切换 tab 后产生额外 "main" 窗格）
        - 其他情况下，task_id 为空则归到 "main"（即 main）
        """
        tid = str(raw_task_id or "").strip()
        node = str(node_hint or "").strip()
        node_l = node.lower()
        tid_l = tid.lower()

        # TaskInit 固定归并
        if node_l in ["task_init", "taskinit"] or node == NodeName.TASK_INIT:
            return "task_init"
        if tid_l in ["task_init", "taskinit"]:
            return "task_init"

        if tid:
            return tid

        watch_task_id = str(self._get_watch_task_id() or "").strip()
        if watch_task_id:
            return watch_task_id

        return "main"

    def task_id_from_message(self, message: Any, *, node_hint: str = "") -> str:
        raw = str(getattr(message, "task_id", "") or "").strip()
        return self.normalize_task_id(raw, node_hint=node_hint)
