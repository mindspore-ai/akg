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

"""
CodeChecker: 代码检查器

纯静态检查流程（不调用 LLM）：
1. 空代码检查
2. ast.parse 语法检查
3. py_compile 编译检查
4. import 可用性检查
5. 中文文本混入检测
6. DSL 合规性检测（反作弊：确保代码真正使用了指定的 DSL）
7. Triton 非阻塞诊断（API/signature/高置信语义风险，不影响 verifier 路由）
"""

import logging
from typing import Dict, List, Optional, Tuple

from akg_agents.op.utils.code_checker.base import (
    CheckError,
    format_errors,
)
from akg_agents.op.utils.code_checker.base_checkers import BaseCheckerRunner
from akg_agents.op.utils.code_checker.registry import (
    build_checker_registry,
    checker_group_config,
)
from akg_agents.op.utils.code_checker.triton_checkers import TritonCheckerRunner

logger = logging.getLogger(__name__)


class CodeChecker:
    """
    代码检查器：在 Coder 生成代码后、Verifier 验证前，进行快速的纯静态检查

    基础 blocking 检查失败时仍按原逻辑回到 Coder/KernelGen。
    Triton 诊断检查只写入 last_diagnostic_*，作为 verifier 失败后的修复参考。
    """

    def __init__(self, backend: str, dsl: str, config: Optional[dict] = None):
        self.backend = backend.lower() if backend else ""
        self.dsl = dsl.lower() if dsl else ""
        self.config = config or {}
        self.last_diagnostic_passed = True
        self.last_diagnostic_errors: List[Dict] = []
        self.last_diagnostic_error_message = ""

        self._registry = build_checker_registry(
            backend=self.backend,
            dsl=self.dsl,
            config=self.config,
        )

        base_specs = self._registry.select(
            "base",
            checker_group_config(self.config, "base_checkers"),
        )
        self.base_checkers = [spec.factory() for spec in base_specs]
        self._base_runner = BaseCheckerRunner(self.base_checkers)

        triton_selection = checker_group_config(self.config, "triton_checkers")
        if triton_selection is None:
            checker_config = self.config.get("code_diagnostic_checker", {}) or {}
            if not isinstance(checker_config, dict):
                checker_config = {}
            triton_selection = checker_config.get("checkers")

        triton_specs = self._registry.select("triton", triton_selection)
        self.triton_checkers = [spec.factory() for spec in triton_specs]
        self._triton_runner = TritonCheckerRunner(
            self.triton_checkers,
            backend=self.backend,
            dsl=self.dsl,
            config=self.config,
        )
        logger.info(f"CodeChecker initialized: backend={self.backend}, dsl={self.dsl}")

    async def check(self, code: str, task_info: Optional[dict] = None) -> Tuple[bool, str, List[Dict]]:
        """
        检查代码（纯静态检查，不调用 LLM）

        Args:
            code: 要检查的代码
            task_info: 任务信息（保留参数以兼容接口）

        Returns:
            Tuple[bool, str, List[Dict]]:
                - passed: 是否通过检查
                - error_message: 格式化的错误信息（用于传递给 Coder）
                - errors: 详细错误列表
        """
        self._reset_diagnostics()

        errors = self._base_runner.run(code)

        # Step 6: 非阻塞 Triton 诊断检查（仅在 base checks 完全通过后执行）。
        #
        # 这些诊断只写入 last_diagnostic_*，不参与 passed，也不会改变
        # 原有 CodeChecker 路由行为：基础 blocking 检查失败仍返回 coder，
        # 基础检查通过则继续进入 verifier。
        if not errors:
            (
                self.last_diagnostic_passed,
                self.last_diagnostic_errors,
                self.last_diagnostic_error_message,
            ) = self._triton_runner.run(code, task_info)

        passed = len(errors) == 0
        code_lines = code.split("\n")
        error_message = format_errors(errors, code_lines) if errors else ""

        if errors:
            logger.warning(f"CodeChecker: Found {len(errors)} issue(s)")
            for err in errors:
                logger.warning(f"  Line {err['line']}: {err['detail']}")
        else:
            logger.info("CodeChecker: All checks passed")

        return passed, error_message, errors

    def _reset_diagnostics(self) -> None:
        self.last_diagnostic_passed = True
        self.last_diagnostic_errors = []
        self.last_diagnostic_error_message = ""


__all__ = ["CheckError", "CodeChecker"]
