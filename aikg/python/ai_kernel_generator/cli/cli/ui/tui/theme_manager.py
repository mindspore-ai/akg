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

from typing import TYPE_CHECKING

from textual.theme import Theme
from textual import log

if TYPE_CHECKING:
    from .app import SplitViewApp


class ThemeManager:
    """主题与配色辅助（Textual + Rich）。"""

    def __init__(self, app: "SplitViewApp") -> None:
        self.app = app

    def register_builtin_themes(self) -> None:
        try:
            self.app.register_theme(
                Theme(
                    name="aikg-light",
                    # 亮色主题收敛为：黑白 + 单一强调色（primary）
                    primary="#2563EB",
                    secondary="#2563EB",
                    warning="#2563EB",
                    error="#2563EB",
                    success="#2563EB",
                    accent="#2563EB",
                    foreground="#111827",
                    background="#FFFFFF",
                    surface="#FFFFFF",
                    panel="#FFFFFF",
                    # 在部分终端（如 MobaXterm）里，Textual 的 alpha/派生色会导致“发灰”
                    text_alpha=1.0,
                    luminosity_spread=0.0,
                    dark=False,
                    variables={
                        "footer-key-foreground": "#2563EB",
                        # 避免派生/默认的 muted 走到灰阶，统一使用强调色
                        "primary-muted": "#2563EB",
                        "border": "#111827",
                        "border-blurred": "#111827",
                    },
                )
            )
            self.app.register_theme(
                Theme(
                    name="aikg-dark",
                    # 暗色主题收敛为：黑白 + 单一强调色（primary）
                    primary="#60A5FA",
                    secondary="#60A5FA",
                    warning="#60A5FA",
                    error="#60A5FA",
                    success="#60A5FA",
                    accent="#60A5FA",
                    foreground="#E5E7EB",
                    background="#0B1021",
                    surface="#0B1021",
                    panel="#0B1021",
                    text_alpha=1.0,
                    luminosity_spread=0.0,
                    dark=True,
                    variables={
                        "footer-key-foreground": "#60A5FA",
                        "primary-muted": "#60A5FA",
                        "border": "#E5E7EB",
                        "border-blurred": "#E5E7EB",
                    },
                )
            )
        except Exception as e:
            log.warning("[Theme] register_builtin_themes failed", exc_info=e)
            return

    def theme_color(self, name: str, fallback: str) -> str:
        """从当前主题变量中取色（用于 Rich/Text 样式）。"""
        try:
            value = self.app.get_css_variables().get(name)
            if value:
                return str(value)
        except Exception as e:
            log.debug("[Theme] get_css_variables failed; fallback", exc_info=e)
        return fallback

    def is_dark(self) -> bool:
        try:
            return bool(getattr(getattr(self.app, "current_theme", None), "dark", True))
        except Exception as e:
            log.debug("[Theme] read current_theme.dark failed; assume dark", exc_info=e)
            return True

    @staticmethod
    def pick_pygments_theme(candidates: list[str], default: str) -> str:
        try:
            from pygments.styles import get_all_styles

            available = set(get_all_styles())
            for name in candidates:
                if name in available:
                    return name
        except ImportError as e:
            log.debug("[Theme] pygments not installed; use default", exc_info=e)
        return default

    def syntax_theme_name(self) -> str:
        if self.is_dark():
            return self.pick_pygments_theme(
                ["gruvbox-dark", "github-dark", "monokai"], "monokai"
            )
        return self.pick_pygments_theme(
            ["xcode", "default", "gruvbox-light"], "default"
        )
