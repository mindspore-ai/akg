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

from rich.console import Console
from rich.text import Text

from ai_kernel_generator.cli.cli.constants import (
    AKG_CLI_LOGO,
    DisplayStyle,
    make_gradient_logo,
)


def print_logo_once(console: Console) -> None:
    """进入命令时打印 AKG CLI Logo。"""
    console.print("\n")
    # 用 Text 输出，避免 Rich markup 与 ASCII 字符冲突
    console.print(make_gradient_logo())
    console.print("\n")
