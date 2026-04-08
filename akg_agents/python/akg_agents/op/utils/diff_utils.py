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
- CodeMatcher: 多级代码匹配器（L1 精确 / L2 行级 trim / L3 空白规范化 / L4 模糊匹配）
- DiffApplier: 差异应用器，执行替换并生成 unified diff
- Modification / DiffResult: 数据类
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class Modification:
    """单个 Search/Replace 修改操作"""
    old_string: str
    new_string: str
    reason: str = ""
    replace_all: bool = False
    anchor: str = ""


@dataclass
class DiffResult:
    """修改操作的完整结果"""
    success: bool
    modified_code: str
    original_code: str
    diff_text: str = ""
    applied_count: int = 0
    errors: List[str] = field(default_factory=list)
    raw_llm_output: str = ""
    match_levels: Dict[str, int] = field(default_factory=dict)


class CodeMatcher:
    """多级代码匹配器

    匹配降级链：
      L1 — 精确匹配（str.find）
      L2 — 行级 trim 匹配（逐行 strip 后滑动窗口比较）
      L3 — 空白规范化匹配（连续空白合并为单个空格后比较）
      L4 — 模糊匹配（基于编辑距离，含置信度检查和窗口 +/-1 行容差）

    所有匹配级别找到匹配后，返回原始代码中的对应片段（而非搜索串本身），
    确保后续 str.replace() 一定能替换成功。
    """

    FUZZY_THRESHOLD = 0.8
    FUZZY_CONFIDENCE_GAP = 0.1

    @classmethod
    def find_match(cls, content: str, search: str) -> Tuple[Optional[str], str]:
        """依次尝试各级匹配策略，返回 (匹配到的原始片段, 匹配级别)。

        Returns:
            (matched_text, level): matched_text 为 None 表示全部失败，
            level 取值 "exact" / "trimmed" / "whitespace_normalized" / "fuzzy" / "none"。
        """
        result = cls.exact_match(content, search)
        if result is not None:
            return result, "exact"

        result = cls.trimmed_line_match(content, search)
        if result is not None:
            return result, "trimmed"

        result = cls.whitespace_normalized_match(content, search)
        if result is not None:
            return result, "whitespace_normalized"

        result = cls.fuzzy_match(content, search, threshold=cls.FUZZY_THRESHOLD)
        if result is not None:
            return result, "fuzzy"

        return None, "none"

    @classmethod
    def find_match_with_anchor(
        cls, content: str, search: str, anchor: str,
    ) -> Tuple[Optional[str], str]:
        """anchor 消歧匹配：先定位 anchor，再在 anchor 附近搜索 old_string。

        anchor 可能位于 old_string 的中间，此时 old_string 从 anchor 之前开始。
        通过检查 anchor 在 search 中的位置精确计算回退偏移量。

        Returns:
            (matched_text, level): 同 find_match。
        """
        if not anchor:
            return cls.find_match(content, search)

        anchor_pos = content.find(anchor)
        if anchor_pos == -1:
            logger.warning(f"Anchor not found: '{anchor[:60]}'")
            return None, "none"

        anchor_in_search = search.find(anchor)
        if anchor_in_search > 0:
            search_start = max(0, anchor_pos - anchor_in_search)
        else:
            search_start = anchor_pos

        sub_content = content[search_start:]
        matched, level = cls.find_match(sub_content, search)
        if matched is None:
            return None, "none"
        return matched, level

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
        if all(s == "" for s in stripped_search):
            return None

        window_size = len(search_lines)
        if window_size > len(content_lines):
            return None

        for i in range(len(content_lines) - window_size + 1):
            window = content_lines[i:i + window_size]
            stripped_window = [line.strip() for line in window]
            if stripped_window == stripped_search:
                matched = "\n".join(window)
                return matched

        return None

    @classmethod
    def whitespace_normalized_match(cls, content: str, search: str) -> Optional[str]:
        """L3: 空白规范化匹配

        将连续空白统一为单个空格后比较。使用行级滑动窗口，匹配成功返回
        content 中的原始片段。
        """
        if not search:
            return None

        def normalize(s: str) -> str:
            return re.sub(r'\s+', ' ', s).strip()

        norm_search = normalize(search)
        if not norm_search:
            return None

        content_lines = content.splitlines()
        search_lines = search.splitlines()
        window_size = len(search_lines)

        if window_size == 0 or window_size > len(content_lines):
            return None

        for i in range(len(content_lines) - window_size + 1):
            window = "\n".join(content_lines[i:i + window_size])
            if normalize(window) == norm_search:
                return window

        return None

    @classmethod
    def fuzzy_match(
        cls, content: str, search: str, threshold: float = 0.8,
    ) -> Optional[str]:
        """L4: 模糊匹配（基于 SequenceMatcher）

        滑动窗口尝试 search_line_count +/-1 行的窗口大小（容差），
        找到相似度最高的片段。含置信度检查：最佳与次佳相似度差距不足
        FUZZY_CONFIDENCE_GAP 时拒绝匹配，防止选错位置。
        """
        if not search or not content:
            return None

        content_lines = content.splitlines()
        search_lines = search.splitlines()
        search_line_count = len(search_lines)

        if search_line_count == 0:
            return None

        best_match: Optional[str] = None
        best_ratio = 0.0
        second_best_ratio = 0.0

        for delta in [0, -1, 1]:
            window_size = search_line_count + delta
            if window_size <= 0 or window_size > len(content_lines):
                continue
            for i in range(len(content_lines) - window_size + 1):
                window = "\n".join(content_lines[i:i + window_size])
                ratio = difflib.SequenceMatcher(None, search, window).ratio()
                if ratio > best_ratio:
                    second_best_ratio = best_ratio
                    best_ratio = ratio
                    best_match = window
                elif ratio > second_best_ratio:
                    second_best_ratio = ratio

        if best_ratio < threshold:
            return None

        if best_ratio - second_best_ratio < cls.FUZZY_CONFIDENCE_GAP:
            logger.warning(
                f"Fuzzy match rejected: confidence gap too small "
                f"(best={best_ratio:.3f}, second={second_best_ratio:.3f})"
            )
            return None

        return best_match


class DiffApplier:
    """差异应用器：将 Modification 列表逐个应用到代码上

    支持 replace_all 全局替换、anchor 锚点消歧、冲突预检测、匹配级别追踪。
    """

    @classmethod
    def apply_modifications(
        cls,
        code: str,
        modifications: List[Modification],
        raw_llm_output: str = "",
    ) -> DiffResult:
        """逐个顺序应用 modifications，返回 DiffResult。

        每次替换后在修改后的代码上执行下一个匹配。
        如果某个 modification 匹配失败则跳过并记录错误。
        """
        original_code = code
        current_code = code
        applied_count = 0
        errors: List[str] = []
        match_levels: Dict[str, int] = {}

        conflict_warnings = cls.detect_conflicts(modifications)
        for w in conflict_warnings:
            logger.warning(w)

        for idx, mod in enumerate(modifications):
            if mod.old_string == mod.new_string:
                errors.append(
                    f"Modification {idx + 1}: old_string 与 new_string 相同，跳过"
                )
                continue

            matched_text, level = CodeMatcher.find_match_with_anchor(
                current_code, mod.old_string, mod.anchor,
            )

            match_levels[level] = match_levels.get(level, 0) + 1

            if matched_text is None:
                if mod.anchor and current_code.find(mod.anchor) == -1:
                    errors.append(
                        f"Modification {idx + 1}: anchor 未找到 "
                        f"(anchor: '{mod.anchor[:60]}')"
                    )
                else:
                    errors.append(
                        f"Modification {idx + 1}: 在代码中未找到匹配 "
                        f"(old_string 前 60 字符: "
                        f"'{mod.old_string[:60]}...')"
                    )
                continue

            if mod.replace_all:
                count = current_code.count(matched_text)
                current_code = current_code.replace(matched_text, mod.new_string)
                applied_count += count
                logger.debug(
                    f"Modification {idx + 1} replace_all={count} (level={level}): "
                    f"{mod.reason}"
                )
            elif mod.anchor:
                anchor_pos = current_code.find(mod.anchor)
                anchor_in_old = mod.old_string.find(mod.anchor)
                if anchor_in_old > 0:
                    search_start = max(0, anchor_pos - anchor_in_old)
                else:
                    search_start = anchor_pos
                sub_content = current_code[search_start:]
                replaced_sub = sub_content.replace(matched_text, mod.new_string, 1)
                current_code = current_code[:search_start] + replaced_sub
                applied_count += 1
                logger.debug(
                    f"Modification {idx + 1} 应用成功 (anchor, level={level}): "
                    f"{mod.reason}"
                )
            else:
                current_code = current_code.replace(matched_text, mod.new_string, 1)
                applied_count += 1
                logger.debug(
                    f"Modification {idx + 1} 应用成功 (level={level}): {mod.reason}"
                )

        diff_text = cls._generate_diff(original_code, current_code)

        success = applied_count > 0
        return DiffResult(
            success=success,
            modified_code=current_code,
            original_code=original_code,
            diff_text=diff_text,
            applied_count=applied_count,
            errors=errors,
            raw_llm_output=raw_llm_output,
            match_levels=match_levels,
        )

    @staticmethod
    def detect_conflicts(modifications: List[Modification]) -> List[str]:
        """冲突预检测：扫描 old_string 之间是否存在包含关系"""
        warnings: List[str] = []
        for i, mod_a in enumerate(modifications):
            for j, mod_b in enumerate(modifications):
                if i >= j:
                    continue
                if (mod_a.old_string in mod_b.old_string
                        or mod_b.old_string in mod_a.old_string):
                    warnings.append(
                        f"Modification {i + 1} and {j + 1} may conflict "
                        f"(overlapping old_string regions)"
                    )
        return warnings

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
            replace_all=bool(item.get("replace_all", False)),
            anchor=str(item.get("anchor", "")),
        ))

    return modifications


def truncate_error_log(error_log: str, max_len: int = 5000) -> str:
    """截断过长的错误日志，保留头部 1/3 + 尾部 2/3。

    Traceback 的关键信息（实际错误类型和最近的调用帧）在尾部，
    因此尾部分配更多空间。
    """
    if len(error_log) <= max_len:
        return error_log
    head_len = max_len // 3
    tail_len = max_len - head_len - 50
    return (
        error_log[:head_len]
        + f"\n\n... ({len(error_log) - head_len - tail_len} chars truncated) ...\n\n"
        + error_log[-tail_len:]
    )
