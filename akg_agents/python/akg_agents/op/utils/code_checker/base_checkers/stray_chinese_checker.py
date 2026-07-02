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

"""Stray Chinese text check for CodeChecker."""

import io
import logging
import re
import tokenize
from typing import Dict, List

from akg_agents.op.utils.code_checker.base import BlockingCodeChecker

logger = logging.getLogger(__name__)


class StrayChineseChecker(BlockingCodeChecker):
    """Detect un-commented Chinese prose outside strings and comments."""

    name = "stray_chinese"

    _CHINESE_RUN_RE = re.compile(r"[\u4e00-\u9fff]{3,}")

    def check(self, code: str) -> List[Dict]:
        """
        检测代码中混入的中文文本（LLM 常见问题）。

        规则：连续 >=3 个汉字出现在注释和字符串之外，视为误混入的中文描述。
        通过 tokenize 精确剥离注释和字符串，只扫描真正的代码 token。
        """
        errors = []
        try:
            tokens = list(tokenize.generate_tokens(io.StringIO(code).readline))
        except (tokenize.TokenError, IndentationError):
            return errors

        for tok in tokens:
            if tok.type in (tokenize.COMMENT, tokenize.STRING):
                continue
            if tok.type in (
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
                tokenize.ENCODING,
            ):
                continue

            match = self._CHINESE_RUN_RE.search(tok.string)
            if match:
                line_num = tok.start[0]
                chinese_text = match.group()
                errors.append(
                    {
                        "line": line_num,
                        "error_type": "stray_chinese_text",
                        "detail": f"代码中混入了中文文本 '{chinese_text}'，疑似未注释的中文描述",
                        "suggestion": (
                            f"第 {line_num} 行包含非代码的中文文本，请删除或改为注释（在行首加 #）。"
                            f"如果是有意使用的中文变量名，请忽略此警告。"
                        ),
                        "code_snippet": "",
                    }
                )
                logger.warning(
                    f"CodeChecker: stray Chinese text at line {line_num}: '{chinese_text}'"
                )

        return errors
