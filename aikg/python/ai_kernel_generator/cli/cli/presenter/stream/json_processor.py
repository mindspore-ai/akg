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

import re

from .state import RenderState
from .painter import ConsolePainter
from .logging import init_stream_renderer_logger


logger = init_stream_renderer_logger(__name__)


class JsonStreamProcessor:
    """处理流式 JSON 内容的解析与呈现。"""

    CODE_KEY_PATTERN = re.compile(r'^(\s*\{?\s*)"(task_code|code|sketch)":\s*"')

    def __init__(self, painter: ConsolePainter):
        self.painter = painter

    def process_line(self, line: str, state: RenderState, *, dim: bool = False) -> bool:
        state.brace_balance += line.count("{") - line.count("}")

        current_text = line

        if not state.in_json_code_field:
            match = self.CODE_KEY_PATTERN.search(current_text)
            if match:
                state.in_json_code_field = True
                end_pos = match.end()
                key_part = current_text[:end_pos]
                self.painter.print_json_structure_line(
                    key_part, state.get_line_number(dim=dim), dim=dim
                )
                state.advance_line_number(dim=dim)
                current_text = current_text[end_pos:]

        if state.in_json_code_field:
            return self._process_json_code_value(current_text, state, dim=dim)

        display_lines = current_text.replace("\\n", "\n").split("\n")
        for dl in display_lines:
            if dl or len(display_lines) == 1:
                self.painter.print_json_line(
                    dl, state.get_line_number(dim=dim), dim=dim
                )
                state.advance_line_number(dim=dim)

        self._check_implicit_json_end(state)
        return True

    def _process_json_code_value(
        self, text: str, state: RenderState, *, dim: bool = False
    ) -> bool:
        end_idx = -1
        idx = 0
        while idx < len(text):
            if text[idx] == '"':
                escaped = False
                back = idx - 1
                while back >= 0 and text[back] == "\\":
                    escaped = not escaped
                    back -= 1
                if not escaped:
                    end_idx = idx
                    break
            idx += 1

        if end_idx != -1:
            code_part = text[:end_idx]
            suffix = text[end_idx:]

            if code_part:
                display_code = (
                    code_part.replace("\\n", "\n")
                    .replace('\\"', '"')
                    .replace("\\'", "'")
                )
                if display_code.endswith("\n"):
                    display_code = display_code[:-1]
                consumed = self.painter.print_syntax_block(
                    display_code,
                    "python",
                    add_end_separator=False,
                    line_number_start=state.get_line_number(dim=dim),
                    dim=dim,
                )
                state.advance_line_number(dim=dim, count=int(consumed or 0))

            self.painter.print_json_structure_line(
                suffix, state.get_line_number(dim=dim), dim=dim
            )
            state.advance_line_number(dim=dim)
            state.in_json_code_field = False
        else:
            if text:
                display = (
                    text.replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")
                )
                if "\n" in display:
                    if display.endswith("\n"):
                        display = display.rstrip("\n")
                    consumed = self.painter.print_syntax_block(
                        display,
                        "python",
                        add_end_separator=False,
                        line_number_start=state.get_line_number(dim=dim),
                        dim=dim,
                    )
                    state.advance_line_number(dim=dim, count=int(consumed or 0))
                else:
                    self.painter.print_json_line(
                        display,
                        state.get_line_number(dim=dim),
                        is_code=True,
                        dim=dim,
                    )
                    state.advance_line_number(dim=dim)

        return True

    def _check_implicit_json_end(self, state: RenderState) -> None:
        if (
            state.is_implicit_json
            and state.brace_balance <= 0
            and not state.in_json_code_field
        ):
            logger.debug("隐式 JSON 块结束")
            dim_used = bool(state.json_dim)
            state.in_json_block = False
            state.is_implicit_json = False
            self.painter.print_json_structure_line(
                "", state.get_line_number(dim=dim_used), dim=dim_used
            )
            state.json_dim = False
            state.advance_line_number(dim=dim_used)
