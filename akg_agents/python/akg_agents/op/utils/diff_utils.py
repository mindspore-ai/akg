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

"""FixCodeGen 增量修复工具

提供 Search/Replace 模式的代码增量修复能力：
- CodeMatcher: 多级代码匹配器（当前实现 L1 精确匹配 + L2 行级 trim 匹配）
- DiffApplier: 差异应用器，执行替换并生成 unified diff
- Modification / DiffResult: 数据类
"""

import difflib
import json
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Modification:
    """单个 Search/Replace 修改操作"""
    old_string: str
    new_string: str
    reason: str = ""


@dataclass
class DiffResult:
    """修改操作的完整结果"""
    success: bool
    modified_code: str
    original_code: str
    diff_text: str = ""
    applied_count: int = 0
    errors: List[str] = field(default_factory=list)


class CodeMatcher:
    """多级代码匹配器

    当前实现：
      L1 — 精确匹配（str.find）
      L2 — 行级 trim 匹配（逐行 strip 后滑动窗口比较）

    所有匹配级别找到匹配后，返回原始代码中的对应片段（而非搜索串本身），
    确保后续 str.replace() 一定能替换成功。
    """

    @classmethod
    def find_match(cls, content: str, search: str) -> Tuple[Optional[str], str]:
        """依次尝试各级匹配策略，返回 (匹配到的原始片段, 匹配级别)。

        Returns:
            (matched_text, level): matched_text 为 None 表示全部失败，
            level 取值 "exact" / "trimmed" / "none"。
        """
        result = cls.exact_match(content, search)
        if result is not None:
            return result, "exact"

        result = cls.trimmed_line_match(content, search)
        if result is not None:
            return result, "trimmed"

        return None, "none"

    @classmethod
    def exact_match(cls, content: str, search: str) -> Optional[str]:
        """L1: 精确字符串匹配"""
        if not search:
            return None
        pos = content.find(search)
        if pos == -1:
            return None
        return content[pos:pos + len(search)]

    @classmethod
    def trimmed_line_match(cls, content: str, search: str) -> Optional[str]:
        """L2: 行级 trim 匹配

        将 content 和 search 都按行 strip，用滑动窗口在 content 行上寻找
        与 search 各行逐行匹配的位置。匹配成功后返回 content 中对应的原始行
        （保留原始缩进）。
        """
        if not search:
            return None

        content_lines = content.splitlines()
        search_lines = search.splitlines()

        if not search_lines:
            return None

        stripped_search = [line.strip() for line in search_lines]
        # 过滤全空的搜索行序列
        if all(s == "" for s in stripped_search):
            return None

        window_size = len(search_lines)
        if window_size > len(content_lines):
            return None

        for i in range(len(content_lines) - window_size + 1):
            window = content_lines[i:i + window_size]
            stripped_window = [line.strip() for line in window]
            if stripped_window == stripped_search:
                # 返回原始内容中的对应片段（保留原始换行符）
                matched = "\n".join(window)
                return matched

        return None


class DiffApplier:
    """差异应用器：将 Modification 列表逐个应用到代码上"""

    @classmethod
    def apply_modifications(
        cls,
        code: str,
        modifications: List[Modification],
    ) -> DiffResult:
        """逐个顺序应用 modifications，返回 DiffResult。

        每次替换后在修改后的代码上执行下一个匹配。
        如果某个 modification 匹配失败则跳过并记录错误。
        """
        original_code = code
        current_code = code
        applied_count = 0
        errors: List[str] = []

        for idx, mod in enumerate(modifications):
            if mod.old_string == mod.new_string:
                errors.append(
                    f"Modification {idx + 1}: old_string 与 new_string 相同，跳过"
                )
                continue

            matched_text, level = CodeMatcher.find_match(current_code, mod.old_string)

            if matched_text is None:
                errors.append(
                    f"Modification {idx + 1}: 在代码中未找到匹配 "
                    f"(old_string 前 60 字符: '{mod.old_string[:60]}...')"
                )
                continue

            # 执行替换（只替换第一次出现）
            current_code = current_code.replace(matched_text, mod.new_string, 1)
            applied_count += 1
            logger.debug(
                f"Modification {idx + 1} 应用成功 (level={level}): {mod.reason}"
            )

        # 生成 unified diff
        diff_text = cls._generate_diff(original_code, current_code)

        success = applied_count > 0
        return DiffResult(
            success=success,
            modified_code=current_code,
            original_code=original_code,
            diff_text=diff_text,
            applied_count=applied_count,
            errors=errors,
        )

    @staticmethod
    def _generate_diff(original: str, modified: str) -> str:
        """生成 unified diff 文本"""
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile="original",
            tofile="modified",
        )
        return "".join(diff)


def parse_modifications(llm_output: str) -> List[Modification]:
    """从 LLM 的 JSON 输出中提取 Modification 列表。

    支持两种格式：
    1. 完整 JSON: {"analysis": "...", "modifications": [...], "summary": "..."}
    2. 仅 modifications 数组: [{"old_string": "...", "new_string": "..."}]

    对 JSON 外层包裹 markdown 代码块（```json ... ```）也做容错处理。
    """
    text = llm_output.strip()

    # 去除 markdown 代码块包裹
    if text.startswith("```"):
        lines = text.split("\n")
        # 去掉第一行（```json）和最后一行（```）
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        else:
            lines = lines[1:]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON 解析失败: {e}")
        return []

    # 提取 modifications 列表
    if isinstance(data, dict):
        mods_raw = data.get("modifications", [])
    elif isinstance(data, list):
        mods_raw = data
    else:
        logger.warning(f"意外的 JSON 顶层类型: {type(data)}")
        return []

    modifications = []
    for item in mods_raw:
        if not isinstance(item, dict):
            continue
        old = item.get("old_string")
        new = item.get("new_string")
        if old is None or new is None:
            logger.warning(f"跳过缺少 old_string/new_string 的修改项: {item}")
            continue
        modifications.append(Modification(
            old_string=old,
            new_string=new,
            reason=item.get("reason", ""),
        ))

    return modifications
