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

"""py_compile check for CodeChecker."""

import logging
import os
import py_compile
import re
import tempfile
from typing import Dict, List

from akg_agents.op.utils.code_checker.base import BlockingCodeChecker

logger = logging.getLogger(__name__)


class PyCompileChecker(BlockingCodeChecker):
    """Run py_compile to catch compile-time Python issues."""

    name = "py_compile"

    def check(self, code: str) -> List[Dict]:
        """
        使用 py_compile 进行编译级别检查。
        比 ast.parse 更严格，能捕获部分 ast.parse 遗漏的编译问题
        （如 SyntaxWarning 升级、重复关键字参数等）。
        """
        errors = []
        tmp_src = None
        tmp_pyc = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".py", delete=False, encoding="utf-8"
            ) as f:
                f.write(code)
                tmp_src = f.name

            # 临时文件写入系统临时目录（Linux: /tmp, Windows: %TEMP%），不在当前工作目录。
            # 用独立的临时文件接收 .pyc 输出，避免往 __pycache__ 写入导致权限问题。
            fd, tmp_pyc = tempfile.mkstemp(suffix=".pyc")
            os.close(fd)

            py_compile.compile(tmp_src, cfile=tmp_pyc, doraise=True)
        except py_compile.PyCompileError as e:
            line_num = 0
            error_str = str(e)
            match = re.search(r"line (\d+)", error_str)
            if match:
                line_num = int(match.group(1))

            code_lines = code.split("\n")
            code_snippet = ""
            if 0 < line_num <= len(code_lines):
                code_snippet = code_lines[line_num - 1].rstrip()

            errors.append(
                {
                    "line": line_num,
                    "error_type": "compile_error",
                    "detail": f"Python 编译错误: {error_str}",
                    "suggestion": f"""请检查第 {line_num} 行附近的代码：
  - 检查是否有不合法的表达式或语法结构
  - 检查变量名、函数名是否合法
  - 检查是否有 Python 版本不兼容的写法""",
                    "code_snippet": code_snippet,
                }
            )
            logger.warning(f"CodeChecker: py_compile error at line {line_num}: {error_str}")
        except Exception as e:
            logger.warning(f"CodeChecker: py_compile check failed unexpectedly: {e}")
        finally:
            for path in (tmp_src, tmp_pyc):
                if path:
                    try:
                        os.unlink(path)
                    except OSError:
                        pass

        return errors
