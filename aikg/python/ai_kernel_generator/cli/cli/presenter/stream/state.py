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

from dataclasses import dataclass, field
from typing import List


@dataclass
class RenderState:
    """维护渲染过程中的瞬时状态。"""

    line_number: int = 1

    # 代码块状态
    in_code_block: bool = False
    code_lang: str = ""
    code_buffer: List[str] = field(default_factory=list)

    # JSON 状态
    in_json_block: bool = False
    is_implicit_json: bool = False
    brace_balance: int = 0
    in_json_code_field: bool = False

    def reset(self) -> None:
        self.line_number = 1
        self.in_code_block = False
        self.code_lang = ""
        self.code_buffer.clear()
        self.in_json_block = False
        self.is_implicit_json = False
        self.brace_balance = 0
        self.in_json_code_field = False
