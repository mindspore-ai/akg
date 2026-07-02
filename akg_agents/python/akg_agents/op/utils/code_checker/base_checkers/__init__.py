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

"""Blocking base checkers used by CodeChecker."""

from akg_agents.op.utils.code_checker.base_checkers.empty_code_checker import EmptyCodeChecker
from akg_agents.op.utils.code_checker.base_checkers.import_checker import ImportAvailabilityChecker
from akg_agents.op.utils.code_checker.base_checkers.py_compile_checker import PyCompileChecker
from akg_agents.op.utils.code_checker.base_checkers.python_syntax_checker import PythonSyntaxChecker
from akg_agents.op.utils.code_checker.base_checkers.stray_chinese_checker import StrayChineseChecker
from akg_agents.op.utils.code_checker.base_checkers.triton_dsl_compliance_checker import (
    TritonDslComplianceChecker,
)
from akg_agents.op.utils.code_checker.base_checkers.runner import BaseCheckerRunner
from akg_agents.op.utils.code_checker.registry import CheckerSpec

__all__ = [
    "BaseCheckerRunner",
    "EmptyCodeChecker",
    "ImportAvailabilityChecker",
    "PyCompileChecker",
    "PythonSyntaxChecker",
    "StrayChineseChecker",
    "TritonDslComplianceChecker",
    "register_base_checkers",
]


def register_base_checkers(registry, *, backend: str, dsl: str, config: dict | None) -> None:
    """Register blocking base checkers in their legacy execution order."""
    del backend, config
    registry.register(
        CheckerSpec(
            name=EmptyCodeChecker.name,
            group="base",
            factory=EmptyCodeChecker,
        )
    )
    registry.register(
        CheckerSpec(
            name=PythonSyntaxChecker.name,
            group="base",
            factory=PythonSyntaxChecker,
        )
    )
    registry.register(
        CheckerSpec(
            name=PyCompileChecker.name,
            group="base",
            factory=PyCompileChecker,
        )
    )
    registry.register(
        CheckerSpec(
            name=ImportAvailabilityChecker.name,
            group="base",
            factory=ImportAvailabilityChecker,
        )
    )
    registry.register(
        CheckerSpec(
            name=StrayChineseChecker.name,
            group="base",
            factory=StrayChineseChecker,
        )
    )
    registry.register(
        CheckerSpec(
            name=TritonDslComplianceChecker.name,
            group="base",
            factory=lambda: TritonDslComplianceChecker(dsl),
        )
    )
