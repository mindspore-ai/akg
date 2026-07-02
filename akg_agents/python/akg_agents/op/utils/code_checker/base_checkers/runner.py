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

"""Blocking base checker execution for CodeChecker."""

from __future__ import annotations

from typing import Dict, Iterable, List

from akg_agents.op.utils.code_checker.base import BlockingCodeChecker


class BaseCheckerRunner:
    """Run selected blocking checkers while preserving legacy short-circuits."""

    def __init__(self, checkers: Iterable[BlockingCodeChecker]):
        self._checkers = list(checkers)

    def run(self, code: str) -> List[Dict]:
        errors: List[Dict] = []
        for checker in self._checkers:
            name = checker.name
            if name == "empty_code":
                empty_errors = checker.check(code)
                if empty_errors:
                    return empty_errors
                continue

            if name in {"py_compile", "import_availability"} and errors:
                continue

            if name == "triton_dsl_compliance" and self._has_syntax_or_compile_errors(errors):
                continue

            errors.extend(checker.check(code))
        return errors

    @staticmethod
    def _has_syntax_or_compile_errors(errors: List[Dict]) -> bool:
        return any(e.get("error_type") in ("syntax_error", "compile_error") for e in errors)
