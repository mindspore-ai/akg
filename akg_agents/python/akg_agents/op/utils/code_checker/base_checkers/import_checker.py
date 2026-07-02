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

"""Import availability check for CodeChecker."""

import ast
import importlib.util
import logging
from typing import Dict, List

from akg_agents.op.utils.code_checker.base import BlockingCodeChecker

logger = logging.getLogger(__name__)


class ImportAvailabilityChecker(BlockingCodeChecker):
    """Check imported top-level modules with importlib.util.find_spec."""

    name = "import_availability"

    def check(self, code: str) -> List[Dict]:
        """
        检查代码中 import 语句引用的模块是否可用。

        通过 AST 提取所有 import / from ... import 语句，
        使用 importlib.util.find_spec 验证顶层模块是否存在。
        """
        errors = []
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return errors

        checked = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    top_module = alias.name.split(".")[0]
                    if top_module in checked:
                        continue
                    checked.add(top_module)
                    if not self.is_module_available(top_module):
                        errors.append(
                            {
                                "line": node.lineno,
                                "error_type": "import_error",
                                "detail": f"模块 '{alias.name}' 无法导入（环境中不存在此模块）",
                                "suggestion": f"请检查模块名 '{alias.name}' 是否拼写正确，或确认该模块是否需要安装",
                                "code_snippet": "",
                            }
                        )
                        logger.warning(
                            f"CodeChecker: import error at line {node.lineno}: "
                            f"module '{alias.name}' not found"
                        )

            elif isinstance(node, ast.ImportFrom):
                if node.level and node.level > 0:
                    continue
                if node.module:
                    top_module = node.module.split(".")[0]
                    if top_module in checked:
                        continue
                    checked.add(top_module)
                    if not self.is_module_available(top_module):
                        errors.append(
                            {
                                "line": node.lineno,
                                "error_type": "import_error",
                                "detail": f"模块 '{node.module}' 无法导入（环境中不存在此模块）",
                                "suggestion": f"请检查模块名 '{node.module}' 是否拼写正确，或确认该模块是否需要安装",
                                "code_snippet": "",
                            }
                        )
                        logger.warning(
                            f"CodeChecker: import error at line {node.lineno}: "
                            f"module '{node.module}' not found"
                        )

        return errors

    @staticmethod
    def is_module_available(module_name: str) -> bool:
        """检查模块在当前环境中是否可用"""
        try:
            return importlib.util.find_spec(module_name) is not None
        except (ModuleNotFoundError, ValueError):
            return False
