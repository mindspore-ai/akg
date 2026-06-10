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

from typing import Union

from rich.console import Console
from rich.text import Text

from akg_agents.cli.constants import (
    DisplayStyle,
    make_gradient_logo,
)
from akg_agents.cli.console import AKGConsole


def print_logo_once(console: Union[Console, AKGConsole]) -> None:
    """进入命令时打印 AKG CLI Logo。``AKG_CLI_QUIET=1`` 时跳过 —— 用于
    SSH 递归调用（remote_dispatch 起远端 daemon）这种场景，避免本机 +
    远端各打一次，且远端 SSH 窄 TTY 会把 logo 截断成两段。"""
    import os
    if os.environ.get("AKG_CLI_QUIET") == "1":
        return
    if isinstance(console, AKGConsole):
        console.print("\n")
        console.print(make_gradient_logo())
        console.print("\n")
    else:
        # 兼容旧代码，但应该使用 AKGConsole
        console.print("\n")
        console.print(make_gradient_logo())
        console.print("\n")
