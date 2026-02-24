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
输出截断基础设施

借鉴 opencode 的 Truncate 系统，为工具输出提供统一的截断机制。
当输出超过阈值时，自动截断并提供摘要信息。

使用方式:
    from akg_agents.core_v2.tools.truncation import Truncate

    # 截断文本输出
    output = Truncate.text(long_text, max_chars=50000)

    # 截断并保存完整内容到文件
    output = Truncate.text_with_save(long_text, save_path="/tmp/full_output.txt")

    # 截断工具结果字典
    result = Truncate.result(tool_result, max_chars=50000)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# 默认截断配置
DEFAULT_MAX_CHARS = 50000  # 单次工具输出最大字符数
DEFAULT_MAX_LINES = 500    # 单次工具输出最大行数


class Truncate:
    """统一的输出截断工具"""

    @staticmethod
    def text(
        content: str,
        max_chars: int = DEFAULT_MAX_CHARS,
        max_lines: Optional[int] = DEFAULT_MAX_LINES,
    ) -> str:
        """截断文本内容

        优先按字符数截断，其次按行数截断。
        截断时保留首尾各一半内容，中间插入截断提示。

        Args:
            content: 原始文本
            max_chars: 最大字符数
            max_lines: 最大行数（None 不限）

        Returns:
            截断后的文本（如果超限）或原文
        """
        if not content:
            return content

        # 按行数截断
        if max_lines is not None:
            lines = content.splitlines()
            if len(lines) > max_lines:
                half = max_lines // 2
                head = "\n".join(lines[:half])
                tail = "\n".join(lines[-half:])
                content = (
                    f"{head}\n\n"
                    f"... [已截断: 总计 {len(lines)} 行, 仅显示首尾各 {half} 行] ...\n\n"
                    f"{tail}"
                )

        # 按字符数截断
        if len(content) > max_chars:
            half = max_chars // 2
            content = (
                content[:half]
                + f"\n\n... [已截断: 总计 {len(content)} 字符, 仅显示首尾各 {half} 字符] ...\n\n"
                + content[-half:]
            )

        return content

    @staticmethod
    def text_with_save(
        content: str,
        save_path: str,
        max_chars: int = DEFAULT_MAX_CHARS,
    ) -> str:
        """截断文本并将完整内容保存到文件

        当内容超过阈值时：
        1. 保存完整内容到 save_path
        2. 返回截断版本 + 文件路径提示

        Args:
            content: 原始文本
            save_path: 保存完整内容的文件路径
            max_chars: 最大字符数

        Returns:
            截断后的文本（含文件路径提示）
        """
        if not content or len(content) <= max_chars:
            return content

        # 保存完整内容
        try:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            save_info = f"\n[完整内容已保存到: {save_path}]"
        except Exception as e:
            logger.warning(f"[Truncate] 保存完整输出失败: {e}")
            save_info = ""

        # 截断
        truncated = Truncate.text(content, max_chars=max_chars)
        return truncated + save_info

    @staticmethod
    def result(
        tool_result: Dict[str, Any],
        max_chars: int = DEFAULT_MAX_CHARS,
        save_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """截断工具结果字典中的 output 和 error_information 字段

        Args:
            tool_result: 工具返回的标准结果字典
            max_chars: 最大字符数
            save_dir: 保存完整输出的目录（可选）

        Returns:
            截断后的结果字典
        """
        if not isinstance(tool_result, dict):
            return tool_result

        result = dict(tool_result)

        for field in ("output", "error_information"):
            value = result.get(field)
            if isinstance(value, str) and len(value) > max_chars:
                if save_dir:
                    save_path = str(Path(save_dir) / f"full_{field}.txt")
                    result[field] = Truncate.text_with_save(value, save_path, max_chars)
                else:
                    result[field] = Truncate.text(value, max_chars)

        return result
