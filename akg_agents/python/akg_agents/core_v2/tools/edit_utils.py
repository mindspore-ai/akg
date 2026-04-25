# Copyright 2026 Huawei Technologies Co., Ltd
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

"""
编辑容错工具 (Edit Replacer Chain)

借鉴 opencode 的 9 级 Replacer 链设计，提供多级容错的字符串替换能力。
当 LLM 生成的 old_string 不完全精确时（空白差异、缩进差异等），
依次尝试更宽容的匹配策略。

使用方式:
    from akg_agents.core_v2.tools.edit_utils import find_and_replace

    new_content, match_info = find_and_replace(file_content, old_string, new_string)
    if new_content is not None:
        # 替换成功
    else:
        # 所有策略均失败
"""

import re
import logging
from difflib import SequenceMatcher
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


# ==================== Replacer 函数 ====================


def _exact_match(content: str, old: str, new: str) -> Optional[str]:
    """Level 1: 精确匹配

    直接使用 str.replace()，要求完全一致。
    """
    if old in content:
        # 确保只有一处匹配（避免歧义替换）
        count = content.count(old)
        if count == 1:
            return content.replace(old, new)
        else:
            logger.debug(f"[edit] 精确匹配发现 {count} 处，尝试替换第一处")
            idx = content.index(old)
            return content[:idx] + new + content[idx + len(old):]
    return None


def _line_trimmed(content: str, old: str, new: str) -> Optional[str]:
    """Level 2: 行首尾空白容差

    将 old_string 和文件内容的每一行去除首尾空白后比较。
    匹配成功后用原始缩进 + new_string 的内容替换。
    """
    old_lines = [line.strip() for line in old.splitlines()]
    content_lines = content.splitlines()

    if not old_lines:
        return None

    # 滑动窗口匹配
    for i in range(len(content_lines) - len(old_lines) + 1):
        window = [content_lines[i + j].strip() for j in range(len(old_lines))]
        if window == old_lines:
            # 找到匹配位置，替换
            before = content_lines[:i]
            after = content_lines[i + len(old_lines):]
            new_lines = new.splitlines()
            result_lines = before + new_lines + after
            return "\n".join(result_lines)

    return None


def _block_anchor(content: str, old: str, new: str) -> Optional[str]:
    """Level 3: 首尾行锚点 + 中间相似度

    使用 old_string 的第一行和最后一行作为锚点在文件中定位，
    然后检查中间部分的相似度是否达到阈值。
    """
    old_lines = old.splitlines()
    if len(old_lines) < 2:
        return None

    first_line = old_lines[0].strip()
    last_line = old_lines[-1].strip()
    content_lines = content.splitlines()

    if not first_line or not last_line:
        return None

    # 找到首行匹配
    for start in range(len(content_lines)):
        if content_lines[start].strip() == first_line:
            # 找到尾行匹配（在首行之后）
            for end in range(start + len(old_lines) - 1,
                             min(start + len(old_lines) * 2, len(content_lines))):
                if content_lines[end].strip() == last_line:
                    # 检查中间部分相似度
                    block = "\n".join(content_lines[start:end + 1])
                    similarity = SequenceMatcher(None, old, block).ratio()
                    if similarity >= 0.6:
                        before = content_lines[:start]
                        after = content_lines[end + 1:]
                        new_lines = new.splitlines()
                        result_lines = before + new_lines + after
                        return "\n".join(result_lines)

    return None


def _whitespace_normalized(content: str, old: str, new: str) -> Optional[str]:
    """Level 4: 空白归一化

    将所有连续空白（空格、制表符）归一化为单个空格后比较。
    """
    def normalize(text: str) -> str:
        return re.sub(r'[ \t]+', ' ', text)

    normalized_old = normalize(old)
    content_lines = content.splitlines()
    old_lines = old.splitlines()

    if not old_lines:
        return None

    for i in range(len(content_lines) - len(old_lines) + 1):
        window = "\n".join(content_lines[i:i + len(old_lines)])
        if normalize(window) == normalized_old:
            before = content_lines[:i]
            after = content_lines[i + len(old_lines):]
            new_lines = new.splitlines()
            result_lines = before + new_lines + after
            return "\n".join(result_lines)

    return None


def _indentation_flexible(content: str, old: str, new: str) -> Optional[str]:
    """Level 5: 缩进灵活匹配

    忽略缩进差异进行匹配。匹配成功后，
    保持文件原有的缩进基准，将 new_string 调整到相同缩进级别。
    """
    old_lines = old.splitlines()
    content_lines = content.splitlines()

    if not old_lines:
        return None

    # 去除所有行的缩进后比较
    stripped_old = [line.lstrip() for line in old_lines]

    for i in range(len(content_lines) - len(old_lines) + 1):
        window = [content_lines[i + j].lstrip() for j in range(len(old_lines))]
        if window == stripped_old:
            # 匹配成功，计算原始缩进
            original_indent = ""
            first_content_line = content_lines[i]
            stripped_first = first_content_line.lstrip()
            if stripped_first:
                original_indent = first_content_line[:len(first_content_line) - len(stripped_first)]

            # 计算 old_string 的缩进（用于计算 new_string 的相对缩进）
            old_indent = ""
            stripped_old_first = old_lines[0].lstrip()
            if stripped_old_first:
                old_indent = old_lines[0][:len(old_lines[0]) - len(stripped_old_first)]

            # 调整 new_string 的缩进
            new_lines = new.splitlines()
            adjusted_new = []
            for line in new_lines:
                if not line.strip():
                    adjusted_new.append("")
                else:
                    # 去除 old 的缩进基准，加上文件的缩进基准
                    if line.startswith(old_indent):
                        relative = line[len(old_indent):]
                    else:
                        relative = line.lstrip()
                    adjusted_new.append(original_indent + relative)

            before = content_lines[:i]
            after = content_lines[i + len(old_lines):]
            result_lines = before + adjusted_new + after
            return "\n".join(result_lines)

    return None


# ==================== Replacer Chain ====================


# 按优先级排序的 Replacer 列表
REPLACERS: List[Tuple[str, callable]] = [
    ("exact_match", _exact_match),
    ("line_trimmed", _line_trimmed),
    ("block_anchor", _block_anchor),
    ("whitespace_normalized", _whitespace_normalized),
    ("indentation_flexible", _indentation_flexible),
]


def find_and_replace(
    content: str,
    old_string: str,
    new_string: str,
) -> Tuple[Optional[str], str]:
    """使用 Replacer 链进行容错替换

    依次尝试从精确到宽松的 5 级匹配策略。
    第一个成功的策略返回替换结果。

    Args:
        content: 文件原始内容
        old_string: 要替换的目标字符串
        new_string: 替换后的字符串

    Returns:
        (new_content, match_info) 元组:
        - new_content: 替换后的内容，所有策略失败时为 None
        - match_info: 匹配信息（使用的策略名称 或 失败原因）
    """
    if not old_string:
        return None, "old_string 不能为空"

    for level, (name, replacer) in enumerate(REPLACERS, 1):
        try:
            result = replacer(content, old_string, new_string)
            if result is not None:
                info = f"Level {level} ({name}) 匹配成功"
                if level > 1:
                    logger.info(f"[edit] {info}")
                return result, info
        except Exception as e:
            logger.debug(f"[edit] {name} 异常: {e}")
            continue

    return None, f"所有 {len(REPLACERS)} 级匹配策略均失败"
