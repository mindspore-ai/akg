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

"""响应式 Info Panel

用法示例：
    panel = ReactiveInfoPanel(widget)

    # 更新状态（只有真正变化时才会触发渲染）
    panel.update(InfoPanelState(
        framework="pytorch",
        backend="cuda"
    ))

    # 增量更新（只更新指定字段）
    panel.patch(framework="tensorflow", arch="x86")
"""

from __future__ import annotations

from typing import Callable, Optional, Any
from rich.text import Text
from textual.widgets import Static
from textual import log

from ai_kernel_generator.cli.cli.ui.state import InfoPanelState
from .reactive_panel_base import ReactivePanelBase


class ReactiveInfoPanel(ReactivePanelBase[InfoPanelState]):
    """响应式 Info Panel"""

    def __init__(
        self,
        widget: Optional[Static] = None,
        *,
        t_func: Optional[Callable[[str], str]] = None,
        empty_placeholder: str = "Waiting for task info...",
    ):
        super().__init__(
            widget,
            state_factory=InfoPanelState,
            empty_placeholder=empty_placeholder,
        )
        self._t_func = t_func or (lambda x: x)

        # 字段渲染器：可以自定义每个字段的渲染逻辑
        self._field_renderers: dict[str, Callable[[Any], str]] = {
            "framework": lambda v: self._render_field(
                "presenter.label.framework", v or "-"
            ),
            "backend": lambda v: self._render_field(
                "presenter.label.backend", v or "-"
            ),
            "arch": lambda v: self._render_field("presenter.label.arch", v or "-"),
            "dsl": lambda v: self._render_field("presenter.label.dsl", v or "-"),
        }

        # 字段顺序（控制显示顺序）
        self._field_order = [
            "framework",
            "backend",
            "arch",
            "dsl",
        ]

    def set_widget(self, widget: Static) -> None:  # type: ignore[override]
        super().set_widget(widget)
        if self._state is not None:
            self._sync_border_title(self._state)

    def update(self, new_state: InfoPanelState) -> None:  # type: ignore[override]
        super().update(new_state)
        self._sync_border_title(new_state)

    def _sync_border_title(self, state: InfoPanelState) -> None:
        if self._widget is None:
            return
        try:
            title = f"{self._t_func('tui.title.task_info')}"
        except Exception as e:
            log.debug("[ReactiveInfoPanel] compute title failed; ignore", exc_info=e)
            return

        try:
            self._widget.border_title = title
        except Exception as e:
            log.debug(
                "[ReactiveInfoPanel] set border_title failed; ignore",
                title=title,
                exc_info=e,
            )
            return

        try:
            self._widget.refresh()
        except Exception as e:
            log.debug("[ReactiveInfoPanel] widget.refresh failed; ignore", exc_info=e)

    def set_translator(self, t_func: Callable[[str], str]) -> None:
        """设置翻译函数"""
        self._t_func = t_func
        try:
            self.refresh()
        except Exception as e:
            log.debug(
                "[ReactiveInfoPanel] refresh failed after translator set", exc_info=e
            )
        if self._state is not None:
            self._sync_border_title(self._state)

    def _render_state(self, state: InfoPanelState) -> Text:
        """根据状态渲染 UI。"""
        lines = []

        # 按照预定义的顺序渲染字段
        for field_name in self._field_order:
            value = getattr(state, field_name, None)

            # 跳过 None 值（除了必须显示的字段）
            if value is None and field_name not in [
                "framework",
                "backend",
                "arch",
                "dsl",
            ]:
                continue

            # 使用对应的渲染器
            renderer = self._field_renderers.get(field_name)
            if renderer:
                try:
                    line = renderer(value)
                    if line:  # 跳过空行
                        lines.append(line)
                except Exception as e:
                    log.debug(
                        "[ReactiveInfoPanel] render field failed; skip",
                        field=field_name,
                        exc_info=e,
                    )
                    continue

        # 如果没有任何内容，显示占位符
        text = "\n".join(lines).strip() or self._empty_placeholder

        return Text(text)

    def _render_field(self, label_key: str, value: str) -> str:
        """渲染单个字段

        Args:
            label_key: 翻译键（如 "presenter.label.framework"）
            value: 字段值

        Returns:
            格式化的字符串
        """
        label = self._t_func(label_key)
        return f"{label}: {value}"

    # ========= 自定义渲染器 =========

    def set_field_renderer(
        self, field_name: str, renderer: Callable[[Any], str]
    ) -> None:
        """自定义字段渲染器

        用法：
            panel.set_field_renderer(
                "framework",
                lambda v: f"🔥 Framework: {v}"
            )
        """
        self._field_renderers[field_name] = renderer

    def set_field_order(self, order: list[str]) -> None:
        """自定义字段显示顺序"""
        self._field_order = order
