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

"""Non-blocking Triton checker execution for CodeChecker."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Dict, Iterable, List, Optional, Tuple

from akg_agents.op.utils.code_checker.base import (
    CheckContext,
    Issue,
    TritonDiagnosticChecker,
    coerce_bool,
    coerce_str_list,
)

logger = logging.getLogger(__name__)


class TritonCheckerRunner:
    """Run selected Triton diagnostics without affecting blocking pass."""

    def __init__(
        self,
        checkers: Iterable[TritonDiagnosticChecker],
        *,
        backend: str,
        dsl: str,
        config: Optional[dict],
    ):
        self._checkers = list(checkers)
        self.backend = backend.lower() if backend else ""
        self.dsl = dsl.lower() if dsl else ""
        self.config = config or {}

    def diagnostic_checker_config(self) -> Dict:
        checker_config = self.config.get("code_diagnostic_checker", {}) or {}
        if not isinstance(checker_config, dict):
            checker_config = {}
        return checker_config

    def diagnostic_checker_enabled(self) -> bool:
        checker_config = self.diagnostic_checker_config()
        if "enabled" in checker_config:
            return coerce_bool(checker_config.get("enabled"), True)
        return coerce_bool(self.config.get("enable_code_diagnostic_checker"), True)

    def run(self, code: str, task_info: Optional[dict] = None) -> Tuple[bool, List[Dict], str]:
        if not self.dsl.startswith("triton"):
            return True, [], ""
        if not self.diagnostic_checker_enabled():
            return True, [], ""
        if not self._checkers:
            return True, [], ""

        try:
            checker_config = self.diagnostic_checker_config()
            enabled_rules = coerce_str_list(checker_config.get("enabled_rules"))
            enabled_tags = coerce_str_list(checker_config.get("enabled_tags"))
            ctx = CheckContext(
                backend=self.backend,
                dsl=self.dsl,
                only_errors=coerce_bool(checker_config.get("only_errors"), True),
                enabled_rules=set(enabled_rules) if enabled_rules else None,
                enabled_tags=set(enabled_tags) if enabled_tags else None,
                dedup=coerce_bool(checker_config.get("dedup"), True),
            )

            issues = []
            for checker in self._checkers:
                issues.extend(_run_single_checker(checker, code, ctx))

            issues = self._filter_and_sort_issues(issues, ctx)
            if issues:
                task_id = (task_info or {}).get("task_id", "0")
                logger.warning(
                    "[Task %s] CodeChecker non-blocking diagnostics found %d issue(s)",
                    task_id,
                    len(issues),
                )
            return len(issues) == 0, issues_to_dicts(issues), format_issues_text(issues)
        except Exception as exc:
            logger.warning("CodeChecker diagnostic checks failed: %s", exc)
            return (
                False,
                [
                    {
                        "line": 0,
                        "column": 0,
                        "severity": "ERROR",
                        "rule_id": "TRITON_DIAGNOSTIC_CHECKER_INTERNAL_ERROR",
                        "title": "Triton diagnostic checker internal error",
                        "detail": f"{type(exc).__name__}: {exc}",
                        "suggestion": "This non-blocking diagnostic failed; rely on verifier output.",
                        "tags": ["checker"],
                    }
                ],
                self._format_internal_error(exc),
            )

    @staticmethod
    def _filter_and_sort_issues(issues, ctx):
        if ctx.enabled_tags is not None:
            issues = [issue for issue in issues if issue.tags & ctx.enabled_tags]
        if ctx.only_errors:
            issues = [issue for issue in issues if issue.severity == "ERROR"]
        if ctx.dedup:
            seen = set()
            unique = []
            for issue in issues:
                key = (
                    issue.severity,
                    issue.rule_id,
                    issue.location.lineno,
                    issue.location.col,
                    issue.message,
                )
                if key in seen:
                    continue
                seen.add(key)
                unique.append(issue)
            issues = unique
        issues.sort(key=lambda item: (item.location.lineno, item.location.col, item.rule_id))
        return issues

    @staticmethod
    def _format_internal_error(exc: Exception) -> str:
        return "\n".join(
            [
                "## CodeChecker 非阻塞诊断报告",
                "",
                "**Triton 诊断检查器自身执行失败；该错误不会阻止 verifier 运行。**",
                "",
                f"- {type(exc).__name__}: {exc}",
            ]
        )


def run_diagnostic_checkers(
    code: str,
    *,
    ctx: Optional[CheckContext] = None,
    checkers: Optional[Iterable[TritonDiagnosticChecker]] = None,
) -> List[Issue]:
    """Run the default non-blocking Triton diagnostics directly."""
    if checkers is None:
        from akg_agents.op.utils.code_checker.triton_checkers import default_triton_checkers

        checkers = default_triton_checkers()

    ctx = ctx or CheckContext()
    issues: List[Issue] = []
    for checker in checkers:
        issues.extend(_run_single_checker(checker, code, ctx))
    return TritonCheckerRunner._filter_and_sort_issues(issues, ctx)


def issues_to_dicts(issues: List[Issue]) -> List[dict]:
    out = []
    for issue in issues:
        out.append(
            {
                "line": issue.location.lineno,
                "column": issue.location.col,
                "end_line": issue.location.end_lineno,
                "end_column": issue.location.end_col,
                "severity": issue.severity,
                "rule_id": issue.rule_id,
                "title": issue.title,
                "detail": issue.message,
                "suggestion": issue.hint or "",
                "tags": sorted(issue.tags),
            }
        )
    return out


def format_issues_text(issues: List[Issue], *, grouped: bool = True) -> str:
    if not issues:
        return ""
    return _format_grouped(issues) if grouped else _format_flat(issues)


def _run_single_checker(
    checker: TritonDiagnosticChecker,
    code: str,
    ctx: CheckContext,
) -> List[Issue]:
    if ctx.enabled_rules is not None and checker.rule_id not in ctx.enabled_rules:
        return []
    try:
        return checker.run(code, ctx) or []
    except SyntaxError as exc:
        return [checker.syntax_error_issue(exc)]
    except Exception as exc:
        return [checker.internal_error_issue(exc)]


def _format_flat(issues: List[Issue]) -> str:
    lines = ["## CodeChecker 非阻塞诊断报告", "", f"**发现 {len(issues)} 个 Triton 诊断问题：**", ""]
    for issue in issues:
        loc = issue.location
        lines.append(f"- [{loc.lineno}:{loc.col}] {issue.severity} {issue.rule_id}: {issue.title}")
        lines.append(f"  {issue.message}")
        if issue.hint:
            lines.append(f"  建议: {issue.hint}")
        lines.append("")
    lines.append("**注意：这些诊断不会阻止 verifier 运行；请结合 verifier 运行错误一起修复。**")
    return "\n".join(lines).rstrip()


def _format_grouped(issues: List[Issue]) -> str:
    groups: OrderedDict[tuple, List[Issue]] = OrderedDict()
    for issue in issues:
        groups.setdefault((issue.severity, issue.rule_id, issue.title), []).append(issue)

    lines = [
        "## CodeChecker 非阻塞诊断报告",
        "",
        f"**发现 {len(issues)} 个 Triton 诊断问题（{len(groups)} 类）：**",
        "",
    ]

    for (severity, rule_id, title), group in groups.items():
        locs = ", ".join(f"[{item.location.lineno}:{item.location.col}]" for item in group)
        count = f" (x{len(group)})" if len(group) > 1 else ""
        lines.append(f"### {severity} {rule_id}: {title}{count}")
        lines.append(f"位置: {locs}")

        messages = list(dict.fromkeys(item.message for item in group))
        for message in messages:
            lines.append(f"- {message}")

        hint = next((item.hint for item in group if item.hint), None)
        if hint:
            lines.append(f"建议: {hint}")
        lines.append("")

    lines.append("**注意：这些诊断不会阻止 verifier 运行；请结合 verifier 运行错误一起修复。**")
    return "\n".join(lines).rstrip()
