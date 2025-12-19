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

import urllib.parse
import urllib.request

from rich.console import Console
from textual import log

from ai_kernel_generator.cli.cli.constants import DisplayStyle


class BarkNotifier:
    """Bark 推送通知（可禁用）。"""

    def __init__(
        self, console: Console, *, bark_key: str, enabled: bool = True
    ) -> None:
        self.console = console
        self.bark_key = bark_key
        self.enabled = enabled

    def send(self, title: str, content: str) -> None:
        if not self.enabled:
            return
        if not self.bark_key:
            return

        try:
            encoded_title = urllib.parse.quote(title)
            encoded_content = urllib.parse.quote(content)
            bark_url = (
                f"https://api.day.app/{self.bark_key}/{encoded_title}/{encoded_content}"
            )

            with urllib.request.urlopen(bark_url, timeout=5) as response:
                if response.status == 200:
                    self.console.print(
                        f"[{DisplayStyle.DIM}]📱 已发送 Bark 通知[/{DisplayStyle.DIM}]"
                    )
                else:
                    self.console.print(
                        f"[{DisplayStyle.DIM}]⚠️  Bark 通知发送失败: {response.status}[/{DisplayStyle.DIM}]"
                    )
        except Exception as e:
            log.warning("[Notify] Bark send failed", exc_info=e)
            self.console.print(
                f"[{DisplayStyle.DIM}]⚠️  Bark 通知发送异常: {e}[/{DisplayStyle.DIM}]"
            )
