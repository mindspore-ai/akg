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

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SafetyState:
    """Markdown 解析状态"""

    in_code_block: bool
    in_json_block: bool

    @property
    def is_safe(self) -> bool:
        """
        当前状态是否安全可渲染。
        - 普通代码块中间：不安全（需要等待闭合）。
        - JSON 块中间：安全（允许流式输出 JSON）。
        - 无代码块：安全。
        """
        return not self.in_code_block


class StreamSafetyChecker:
    # 匹配 Markdown 代码块标记：行首(允许缩进) + ``` + 可选语言标识
    FENCE_PATTERN = re.compile(r"^\s*```(.*)", re.MULTILINE)

    def __init__(self, initial_state: SafetyState):
        self.initial_state = initial_state

    def find_safe_length(self, content: str) -> int:
        """
        分析内容，返回最长可安全渲染的长度。
        - 如果内容末尾是安全的，返回 len(content)。
        - 如果内容末尾处于未闭合的普通代码块中，返回该代码块开始之前的位置。
        """
        state = SafetyState(
            self.initial_state.in_code_block, self.initial_state.in_json_block
        )

        # 记录导致“当前处于不安全状态”的那个代码块的起始位置
        # 如果初始就是不安全的，起始位置为 0
        unsafe_start_idx: Optional[int] = 0 if not state.is_safe else None

        for match in self.FENCE_PATTERN.finditer(content):
            # 再次确认匹配的是 ``` 开头
            if not match.group(0).strip().startswith("```"):
                continue

            fence_start = match.start()

            # 状态转换逻辑
            if state.in_code_block or state.in_json_block:
                # 闭合当前块 -> 状态变更为安全
                state.in_code_block = False
                state.in_json_block = False
                unsafe_start_idx = None  # 清除不安全标记
            else:
                # 开启新块
                lang = match.group(1).strip().lower()
                if lang in ("json", "output_json"):
                    state.in_json_block = True
                    # JSON 块视为安全，不需要记录 unsafe_start_idx
                else:
                    state.in_code_block = True
                    # 进入不安全区域，记录起始点
                    unsafe_start_idx = fence_start

        # 如果最终状态不安全，截断到不安全区域开始的位置
        if unsafe_start_idx is not None:
            logger.debug(f"检测到未闭合代码块，截断至位置: {unsafe_start_idx}")
            return unsafe_start_idx

        return len(content)


def _should_use_json_splitting(content: str, in_json_block: bool) -> bool:
    """
    判断是否应用 JSON 优化策略（允许按转义换行符切分）。
    """
    # 1. 明确在 JSON block 中
    if in_json_block:
        return True

    # 2. 启发式：内容看起来像是在流式输出 JSON 字符串
    # (例如包含 {"code": ... 且有转义换行符，但没有 markdown fence)
    has_json_marker = '{"' in content
    has_escaped_newlines = "\\n" in content
    has_code_fence = "```" in content

    if has_json_marker:
        return True

    if has_escaped_newlines and not has_code_fence:
        return True

    return False


def _find_candidate_limit(content: str, use_json_opt: bool) -> int:
    """
    寻找物理上的最佳切分点（换行符）。
    """
    last_newline = content.rfind("\n")
    # 基础位置：最后一个物理换行符之后
    limit = last_newline + 1

    if use_json_opt:
        # 尝试扩展到最后一个转义换行符 \n
        tail = content[limit:]
        last_escaped = tail.rfind("\\n")
        if last_escaped != -1:
            # +2 是为了包含 \n 两个字符
            limit += last_escaped + 2
            logger.debug(f"JSON 优化：扩展渲染位置到 {limit}")

    return limit


def find_safe_render_position(
    content: str, in_code_block: bool, in_output_json_block: bool
) -> int:
    """
    主入口：找到可以安全渲染的位置。
    """
    if not content:
        return 0

    # 1. 确定切分策略
    use_json_opt = _should_use_json_splitting(content, in_output_json_block)

    # 2. 找到候选截止位置（物理层面）
    candidate_pos = _find_candidate_limit(content, use_json_opt)

    if candidate_pos == 0:
        return 0

    # 3. 安全性检查（逻辑层面）
    # 只检查候选位置之前的内容
    content_to_check = content[:candidate_pos]

    checker = StreamSafetyChecker(SafetyState(in_code_block, in_output_json_block))
    safe_length = checker.find_safe_length(content_to_check)

    return safe_length
