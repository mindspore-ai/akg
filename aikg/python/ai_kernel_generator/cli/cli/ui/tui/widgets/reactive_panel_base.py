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

"""响应式面板基类：统一 widget 绑定、差分更新与 patch 逻辑。

相比原始的 panel.update(text) 方式，响应式系统提供：
1. 类型安全的状态模型
2. 自动差分更新（只在真正变化时才渲染）
3. 组件化的渲染逻辑
4. 更好的可测试性和可维护性
"""

from __future__ import annotations

import traceback
from typing import Optional, Any, Callable, TYPE_CHECKING, Generic, TypeVar
from rich.text import Text
from textual import log

from ai_kernel_generator.cli.cli.ui.state import ReactiveState

if TYPE_CHECKING:
    from textual.widgets import Static


StateT = TypeVar("StateT", bound=ReactiveState)


class ReactivePanelBase(Generic[StateT]):
    """响应式面板基类：统一 widget 绑定、差分更新与 patch 逻辑。"""

    def __init__(
        self,
        widget: Optional["Static"] = None,
        *,
        state_factory: Callable[[], StateT],
        empty_placeholder: str,
    ):
        self._widget = widget
        self._state: Optional[StateT] = None
        self._state_factory = state_factory
        self._empty_placeholder = empty_placeholder

    def set_widget(self, widget: "Static") -> None:
        """延迟设置 widget（用于初始化顺序问题）。"""
        self._widget = widget
        log.debug(
            "[ReactivePanel] set_widget",
            panel=type(self).__name__,
            widget_id=str(getattr(widget, "id", "") or ""),
            has_cached_state=self._state is not None,
        )
        # 若之前已缓存 state，立即补一次渲染，避免"挂载后不更新"。
        if self._state is not None:
            try:
                self._safe_widget_update(self._render_state(self._state))
            except Exception as e:
                log.warning("[ReactivePanel] render cached state failed", exc_info=e)

    def update(self, new_state: StateT) -> None:
        """响应式更新状态：仅在状态真正变化时触发渲染。"""
        if not self._widget:
            self._state = new_state.clone()
            log.debug(
                "[ReactivePanel] cache_state(no_widget)",
                panel=type(self).__name__,
            )
            return

        if not new_state.has_changes(self._state):
            return

        try:
            changes = new_state.diff(self._state)
            log.debug(
                "[ReactivePanel] update",
                panel=type(self).__name__,
                changed=list(changes.keys()),
            )
        except Exception as e:
            log.warning(
                "[ReactivePanel] diff failed; continue with empty changes", exc_info=e
            )
            changes = {}

        try:
            renderable = self._render_state(new_state)
        except Exception as e:
            log.error(
                "[ReactivePanel] render_failed",
                panel=type(self).__name__,
                error=f"{type(e).__name__}: {e}",
                changed=list(getattr(changes, "keys", lambda: [])()),
                exc_info=e,
            )
            log.error(traceback.format_exc())
            return

        ok = self._safe_widget_update(renderable)
        if ok:
            self._state = new_state.clone()

    def refresh(self) -> None:
        """强制重绘当前状态（用于 i18n/主题等渲染上下文变化）。"""
        if not self._widget or self._state is None:
            return
        try:
            renderable = self._render_state(self._state)
        except Exception as e:
            log.error(
                "[ReactivePanel] refresh_render_failed",
                panel=type(self).__name__,
                error=f"{type(e).__name__}: {e}",
                exc_info=e,
            )
            log.error(traceback.format_exc())
            return
        self._safe_widget_update(renderable)

    def patch(self, **kwargs: Any) -> None:
        """增量更新（只更新指定字段）。"""
        current = self._state.clone() if self._state else self._state_factory()

        for key, value in kwargs.items():
            if hasattr(current, key):
                setattr(current, key, value)
            else:
                log.error(
                    "[ReactivePanel] patch_unknown_field",
                    panel=type(self).__name__,
                    field=key,
                )
                raise ValueError(f"Unknown field: {key}")

        self.update(current)

    def clear(self) -> None:
        """清空面板（同时清空内部 state）。"""
        self._state = None
        if self._widget:
            log.debug("[ReactivePanel] clear", panel=type(self).__name__)
            self._safe_widget_update("")

    def get_state(self) -> Optional[StateT]:
        """获取当前状态（只读）。"""
        return self._state.clone() if self._state else None

    def _safe_widget_update(self, renderable: Any) -> bool:
        """安全更新 widget：吞掉异常避免打断 UI 主循环。"""
        if not self._widget:
            return False
        try:
            self._widget.update(renderable)
            return True
        except Exception as e:
            log.error(
                "[ReactivePanel] widget_update_failed",
                panel=type(self).__name__,
                widget_id=str(getattr(self._widget, "id", "") or ""),
                error=f"{type(e).__name__}: {e}",
                exc_info=e,
            )
            log.error(traceback.format_exc())
        try:
            if isinstance(renderable, str):
                self._widget.update(Text(renderable))
            else:
                self._widget.update(str(renderable))
            return True
        except Exception as e:
            log.error(
                "[ReactivePanel] widget_update_failed_fallback",
                panel=type(self).__name__,
                widget_id=str(getattr(self._widget, "id", "") or ""),
                error=f"{type(e).__name__}: {e}",
                exc_info=e,
            )
            log.error(traceback.format_exc())
            return False

    def _render_state(self, state: StateT) -> Any:
        raise NotImplementedError
