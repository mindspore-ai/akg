#!/usr/bin/env python3
"""
AKG Issue 规范校验脚本。

用法:
    python validate_issue.py <json_file> [--json-output]

从 .json 元数据文件读取信息，定位对应的 .md body 文件，执行规范校验。
校验结果写入 .json 的 validation 字段，并输出到 stdout。
"""

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

TITLE_PREFIX = "[AKG_AGENTS]"


@dataclass
class CheckResult:
    rule_id: str
    description: str
    level: str  # "error" | "warning"
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    passed: bool = True
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    results: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Common rules
# ---------------------------------------------------------------------------

def check_title_prefix(body: str, meta: dict) -> CheckResult:
    title = meta.get("title", "")
    passed = title.startswith(TITLE_PREFIX)
    detail = f"title: {title}" if passed else f"title 缺少 {TITLE_PREFIX} 前缀"
    return CheckResult("ISS-000", f"title 必须以 {TITLE_PREFIX} 开头", "error", passed, detail)


def check_issue_title(body: str, meta: dict) -> CheckResult:
    title = meta.get("title", "")
    passed = len(title) >= 10
    detail = f"标题长度 {len(title)} 字符" if passed else f"标题仅 {len(title)} 字符，需至少 10"
    return CheckResult("ISS-001", "title 不得为空，至少 10 字符", "error", passed, detail)


def check_issue_labels(body: str, meta: dict) -> CheckResult:
    labels = meta.get("labels", [])
    has_kind = any(l.startswith("kind/") for l in labels)
    detail = f"labels: {labels}" if has_kind else "缺少 kind/* 标签"
    return CheckResult("ISS-002", "labels 至少包含一个 kind/* 标签", "error", has_kind, detail)


COMMON_CHECKS = [check_title_prefix, check_issue_title, check_issue_labels]


# ---------------------------------------------------------------------------
# Bug-specific rules
# ---------------------------------------------------------------------------

def check_bug_device(body: str, _meta: dict) -> CheckResult:
    passed = bool(re.search(r"^/device\s+(ascend|gpu|cpu)\s*$", body, re.MULTILINE))
    detail = "/device 已设置" if passed else "缺少 /device 标记"
    return CheckResult("BUG-001", "/device 必须存在", "error", passed, detail)


def check_bug_software_env(body: str, _meta: dict) -> CheckResult:
    has_akg = bool(re.search(r"AKG version.*?:", body)) and bool(
        re.search(r"AKG version.*?:\s*\S+", body)
    )
    has_python = bool(re.search(r"Python version.*?:\s*\S+", body))
    passed = has_akg and has_python
    missing = []
    if not has_akg:
        missing.append("AKG 版本")
    if not has_python:
        missing.append("Python 版本")
    detail = "软件环境完整" if passed else f"缺少: {', '.join(missing)}"
    return CheckResult("BUG-002", "至少包含 AKG 版本和 Python 版本", "error", passed, detail)


def check_bug_current_behavior(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Describe the current behavior\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    passed = len(text) > 0
    return CheckResult("BUG-003", "当前行为不得为空", "error", passed, "")


def check_bug_expected_behavior(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Describe the expected behavior\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    passed = len(text) > 0
    return CheckResult("BUG-004", "期望行为不得为空", "error", passed, "")


def check_bug_repro_steps(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Steps to reproduce the issue\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    has_steps = bool(re.search(r"\d+\.\s*\S", text))
    detail = "复现步骤已填写" if has_steps else "复现步骤为空，请补充"
    return CheckResult("BUG-005", "复现步骤至少包含 1 个步骤", "error", has_steps, detail)


def check_bug_log(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Related log / screenshot\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    passed = len(text) > 0
    detail = "已附带日志" if passed else "建议附带相关日志以便排查"
    return CheckResult("BUG-006", "建议附带相关日志", "warning", passed, detail)


BUG_CHECKS = [
    check_bug_device, check_bug_software_env, check_bug_current_behavior,
    check_bug_expected_behavior, check_bug_repro_steps, check_bug_log,
]


# ---------------------------------------------------------------------------
# RFC-specific rules
# ---------------------------------------------------------------------------

def check_rfc_background(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Background\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    passed = len(text) >= 20
    detail = f"背景长度 {len(text)} 字符" if passed else f"背景仅 {len(text)} 字符，需至少 20"
    return CheckResult("RFC-001", "背景不得为空，至少 20 字符", "error", passed, detail)


def check_rfc_design(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Introduction\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    passed = len(text) >= 30
    detail = f"方案长度 {len(text)} 字符" if passed else f"方案仅 {len(text)} 字符，需至少 30"
    return CheckResult("RFC-002", "方案设计不得为空，至少 30 字符", "error", passed, detail)


def check_rfc_trail(body: str, _meta: dict) -> CheckResult:
    passed = bool(re.search(r"\|\s*\d+\s*\|.*\|.*\|", body))
    detail = "已包含任务拆分" if passed else "建议包含任务拆分表格"
    return CheckResult("RFC-003", "建议包含任务拆分表格", "warning", passed, detail)


RFC_CHECKS = [check_rfc_background, check_rfc_design, check_rfc_trail]


# ---------------------------------------------------------------------------
# Task-specific rules
# ---------------------------------------------------------------------------

def check_task_description(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Task Description\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    passed = len(text) > 0
    return CheckResult("TSK-001", "任务描述不得为空", "error", passed, "")


def check_task_goal(body: str, _meta: dict) -> CheckResult:
    match = re.search(r"## Task Goal\s*(.*?)(?=##|\Z)", body, re.DOTALL)
    text = match.group(1).strip() if match else ""
    passed = len(text) > 0
    return CheckResult("TSK-002", "任务目标不得为空", "error", passed, "")


def check_task_subtasks(body: str, _meta: dict) -> CheckResult:
    passed = bool(re.search(r"\|\s*\d+\s*\|.*\|.*\|", body))
    detail = "已包含子任务拆分" if passed else "建议包含子任务拆分"
    return CheckResult("TSK-003", "建议包含子任务拆分", "warning", passed, detail)


TASK_CHECKS = [check_task_description, check_task_goal, check_task_subtasks]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_checks(checks: list, body: str, meta: dict) -> ValidationReport:
    report = ValidationReport()
    for check_fn in checks:
        result = check_fn(body, meta)
        report.results.append(result)
        if not result.passed:
            entry = {"rule_id": result.rule_id, "description": result.description, "detail": result.detail}
            if result.level == "error":
                report.errors.append(entry)
                report.passed = False
            else:
                report.warnings.append(entry)
    return report


def format_report_text(report: ValidationReport) -> str:
    lines = []
    for r in report.results:
        icon = "✅" if r.passed else ("❌" if r.level == "error" else "⚠️ ")
        detail = f": {r.detail}" if r.detail else ""
        lines.append(f"{icon} {r.rule_id}: {r.description}{detail}")
    lines.append("")
    if report.passed:
        if report.warnings:
            lines.append(f"校验通过（{len(report.warnings)} 个 warning）")
        else:
            lines.append("校验通过 ✅")
    else:
        lines.append(f"校验未通过：{len(report.errors)} 个 error，{len(report.warnings)} 个 warning")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="AKG Issue 规范校验")
    parser.add_argument("json_file", help="Issue 的 .json 元数据文件路径")
    parser.add_argument("--json-output", action="store_true", help="以 JSON 格式输出")
    args = parser.parse_args()

    json_path = Path(args.json_file).resolve()
    if not json_path.exists():
        print(f"错误: 文件不存在 {json_path}", file=sys.stderr)
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    body_filename = meta.get("body_file", "")
    body_path = json_path.parent / body_filename
    if not body_path.exists():
        print(f"错误: body 文件不存在 {body_path}", file=sys.stderr)
        sys.exit(1)

    with open(body_path, "r", encoding="utf-8") as f:
        body = f.read()

    issue_type = meta.get("issue_type", "")
    checks = list(COMMON_CHECKS)
    if issue_type == "bug":
        checks.extend(BUG_CHECKS)
    elif issue_type == "rfc":
        checks.extend(RFC_CHECKS)
    elif issue_type == "task":
        checks.extend(TASK_CHECKS)

    report = run_checks(checks, body, meta)

    meta["validation"] = {
        "passed": report.passed,
        "errors": report.errors,
        "warnings": report.warnings,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    if args.json_output:
        print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
    else:
        print(format_report_text(report))

    sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
