#!/usr/bin/env python3
"""
AKG PR 规范校验脚本。

用法:
    python validate_pr.py <json_file> [--json-output]

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


def check_title_prefix(body: str, meta: dict) -> CheckResult:
    title = meta.get("title", "")
    passed = title.startswith(TITLE_PREFIX)
    detail = f"title: {title}" if passed else f"title 缺少 {TITLE_PREFIX} 前缀"
    return CheckResult("PR-000", f"title 必须以 {TITLE_PREFIX} 开头", "error", passed, detail)


def check_pr_kind(body: str, meta: dict) -> CheckResult:
    kind = meta.get("kind", "")
    valid_kinds = {"bug", "task", "feature"}
    found_in_body = re.search(r"^/kind\s+(bug|task|feature)\s*$", body, re.MULTILINE)
    passed = kind in valid_kinds or found_in_body is not None
    detail = f"kind={kind}" if passed else "缺少 /kind 或值无效"
    return CheckResult("PR-001", "/kind 必须存在且为 bug/task/feature", "error", passed, detail)


def check_pr_description(body: str, _meta: dict) -> CheckResult:
    desc_match = re.search(
        r"\*\*What does this PR do.*?\*\*:?\s*(.*?)(?=\*\*Which issue|---|\Z)",
        body, re.DOTALL
    )
    desc_text = desc_match.group(1).strip() if desc_match else ""
    length = len(desc_text)
    passed = length >= 20
    detail = f"描述长度 {length} 字符" if passed else f"描述仅 {length} 字符，需至少 20"
    return CheckResult("PR-002", "PR 描述不得为空，至少 20 字", "error", passed, detail)


def check_pr_fixes_link(body: str, meta: dict) -> CheckResult:
    kind = meta.get("kind", "")
    if kind != "bug":
        return CheckResult("PR-003", "bug fix PR 必须关联 issue", "error", True, "非 bug 类型，跳过")
    has_fixes = bool(re.search(
        r"(Fixes|Closes|Resolves)\s+(#\d+|https?://\S+/issues/\d+)", body, re.IGNORECASE
    ))
    detail = "已关联 issue" if has_fixes else "bug fix PR 未关联 issue，请补充 Fixes <issue_url>"
    return CheckResult("PR-003", "bug fix PR 必须关联 issue", "error", has_fixes, detail)


def check_pr_merge_commits(body: str, _meta: dict) -> CheckResult:
    merge_lines = re.findall(r"^[a-f0-9]+ Merge ", body, re.MULTILINE)
    passed = len(merge_lines) == 0
    detail = "无 merge commit" if passed else f"发现 {len(merge_lines)} 个 merge commit，建议 rebase 清理"
    return CheckResult("PR-004", "commit 历史中不应有 merge commit", "warning", passed, detail)


def check_pr_title_format(body: str, meta: dict) -> CheckResult:
    title = meta.get("title", "")
    raw = title.replace(TITLE_PREFIX, "").strip()
    pattern = r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert|task|bug|feature)[\(:]"
    passed = bool(re.match(pattern, raw, re.IGNORECASE))
    detail = f"title: {title}" if passed else f"title 不符合 conventional 格式: {title}"
    return CheckResult("PR-006", "title 建议符合 [AKG_AGENTS] <kind>: <描述> 格式", "warning", passed, detail)


PR_CHECKS = [
    check_title_prefix,
    check_pr_kind,
    check_pr_description,
    check_pr_fixes_link,
    check_pr_merge_commits,
    check_pr_title_format,
]


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
    parser = argparse.ArgumentParser(description="AKG PR 规范校验")
    parser.add_argument("json_file", help="PR 的 .json 元数据文件路径")
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

    report = run_checks(PR_CHECKS, body, meta)

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
