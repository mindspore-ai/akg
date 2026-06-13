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

"""Shared types and utilities for CodeChecker."""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set


@dataclass
class CheckError:
    """检查错误信息"""

    line: int
    error_type: str
    detail: str
    suggestion: str
    code_snippet: str


@dataclass(frozen=True)
class Location:
    """Source location for non-blocking diagnostic issues."""

    lineno: int = -1
    col: int = 0
    end_lineno: int = -1
    end_col: int = 0


@dataclass(frozen=True)
class Issue:
    """Non-blocking diagnostic issue."""

    severity: str
    rule_id: str
    title: str
    message: str
    location: Location = Location()
    hint: Optional[str] = None
    tags: Set[str] = field(default_factory=set)


@dataclass
class CheckContext:
    """Context used by non-blocking diagnostic checkers."""

    backend: str = ""
    dsl: str = ""
    file_path: Optional[str] = None
    only_errors: bool = True
    enabled_rules: Optional[Set[str]] = None
    enabled_tags: Optional[Set[str]] = None
    dedup: bool = True


class CodeCheckerUnit(ABC):
    """Common interface for checker units selected by YAML."""

    name: str = ""


class BlockingCodeChecker(CodeCheckerUnit):
    """Base interface for blocking CodeChecker sub-checkers."""

    @abstractmethod
    def check(self, code: str) -> List[Dict]:
        """Return a list of CodeChecker-compatible error dictionaries."""


class TritonDiagnosticChecker(CodeCheckerUnit):
    """Base interface for non-blocking Triton diagnostic checkers."""

    checker_id: str = ""
    rule_id: str = ""

    @abstractmethod
    def run(self, code, ctx):
        """Return diagnostic Issue objects."""

    def syntax_error_issue(self, exc: SyntaxError):
        return Issue(
            severity="ERROR",
            rule_id="PYTHON_SYNTAX_ERROR",
            title="Python syntax error",
            message=str(exc),
            location=Location(
                lineno=exc.lineno or -1,
                col=exc.offset or 0,
                end_lineno=exc.end_lineno or -1,
                end_col=exc.end_offset or 0,
            ),
            hint="Fix Python syntax before Triton diagnostics can run.",
            tags={"python", "syntax", "host"},
        )

    def internal_error_issue(self, exc: Exception):
        return Issue(
            severity="ERROR",
            rule_id="TRITON_DIAGNOSTIC_CHECKER_INTERNAL_ERROR",
            title="Triton diagnostic checker internal error",
            message=f"{self.checker_id or self.name} failed: {type(exc).__name__}: {exc}",
            hint="This diagnostic checker is non-blocking; use verifier output as the source of truth.",
            tags={"checker"},
        )


def coerce_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def coerce_str_list(value) -> Optional[List[str]]:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [value]
    try:
        return [str(item) for item in value if item]
    except TypeError:
        return [str(value)]


def format_errors(errors: List[Dict], code_lines: Optional[List[str]] = None) -> str:
    """格式化错误信息，便于传递给 Coder"""
    if not errors:
        return ""

    lines = [
        "## CodeChecker 静态检查报告",
        "",
        f"**发现 {len(errors)} 个问题，请修复后重新生成代码：**",
        "",
    ]

    for i, err in enumerate(errors, 1):
        error_line = err["line"]
        lines.append(f"### 问题 {i}: 第 {error_line} 行 [{err.get('error_type', 'unknown')}]")
        lines.append(f"  {err['detail']}")

        if code_lines is not None and error_line > 0:
            start_line = max(1, error_line - 3)
            end_line = min(len(code_lines), error_line + 3)

            lines.append(f"  上下文（第 {start_line}-{end_line} 行）：")
            for ctx_line_num in range(start_line, end_line + 1):
                ctx_line = code_lines[ctx_line_num - 1]
                if ctx_line_num == error_line:
                    lines.append(f"  >>> {ctx_line_num:4d} | {ctx_line}")
                else:
                    lines.append(f"      {ctx_line_num:4d} | {ctx_line}")
        elif err.get("code_snippet"):
            lines.append(f"  出错代码: {err['code_snippet']}")

        if err.get("suggestion"):
            lines.append("  建议：")
            for sug_line in err["suggestion"].strip().split("\n"):
                lines.append(f"    {sug_line}")

        lines.append("")

    lines.append("**注意：语法检查每次只能定位到首个错误，修复后可能还有后续问题，请仔细检查整段代码。**")

    return "\n".join(lines)


def get_check_summary(errors: List[Dict]) -> str:
    """获取检查摘要（简短版本，用于日志）"""
    if not errors:
        return "代码检查通过"

    error_types = set(err.get("error_type", "unknown") for err in errors)
    return f"发现 {len(errors)} 个问题: {', '.join(error_types)}"


def parse_ast(code: str) -> ast.AST:
    return ast.parse(code)
