#!/usr/bin/env python3
# Copyright 2025-2026 Huawei Technologies Co., Ltd
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
审查报告生成脚本

汇总所有检查结果，生成统一的 Markdown 报告和 JSON 元数据。
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _status_icon(result: Dict, key: str = "status") -> str:
    """返回状态图标"""
    val = result.get(key)
    if val == "pass":
        return "✅"
    if val in ("conflict", "fail"):
        return "❌"
    return "❌" if result.get("total_errors", 0) > 0 else "✅"


def _table_row(name, result, extras="0") -> str:
    """生成概览表的一行"""
    icon = _status_icon(result)
    errs = result.get("total_errors", 0)
    warns = result.get("total_warnings", 0)
    return f"| {name} | {icon} | {errs} | {warns} | {extras} |\n"


def _format_issues(issues, header, show_suggestion=True):
    """格式化一组 issues 为 Markdown"""
    if not issues:
        return ""
    lines = [f"### {header}\n\n"]
    for issue in issues:
        rule = issue["rule"]
        f = issue["file"]
        ln = issue["line"]
        lines.append(f"**[{rule}] {f}:{ln}**\n")
        lines.append(f"- 问题: {issue['message']}\n")
        if show_suggestion:
            sug = issue.get("suggestion", "—")
            lines.append(f"- 建议: {sug}\n")
        lines.append("\n")
    return "".join(lines)


def _report_header(current_branch, target_branch, status):
    """报告头部"""
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return (
        "# AKG Code Review Report\n\n"
        f"**分支**: {current_branch}\n"
        f"**目标**: {target_branch}\n"
        f"**时间**: {now}\n"
        f"**状态**: {status}\n\n---\n\n"
    )


def _report_rebase(rebase_result, target_branch):
    """Rebase 检查部分"""
    msg = rebase_result.get("message", "未知")
    parts = [
        "## 1️⃣ Rebase 冲突检查\n\n",
        f"**状态**: {msg}\n\n",
    ]
    if rebase_result.get("conflicts"):
        parts.append("### 冲突文件\n\n")
        for c in rebase_result["conflicts"]:
            parts.append(f"- **{c['file']}**\n")
            parts.append(f"  - 类型: {c['type']}\n")
            parts.append(f"  - 详情: {c['details']}\n\n")
        remote = (
            target_branch.split("/")[0]
            if "/" in target_branch
            else "origin"
        )
        parts.append("**修复建议**:\n```bash\n")
        parts.append(f"git fetch {remote}\n")
        parts.append(f"git rebase {target_branch}\n")
        parts.append("# 解决冲突后\ngit rebase --continue\n")
        parts.append("```\n\n")
    parts.append("---\n\n")
    return "".join(parts)


def _report_code_style(code_style_result):
    """代码规范检查部分"""
    parts = ["## 2️⃣ 代码规范检查 (Ruff + 自定义)\n\n"]
    issues = code_style_result.get("issues", [])
    errs = [i for i in issues if i["level"] == "error"]
    warns = [i for i in issues if i["level"] == "warning"]
    infos = [i for i in issues if i["level"] == "info"]

    parts.append(_format_issues(errs, "❌ 错误（必须修复）"))
    parts.append(_format_issues(warns, "⚠️ 警告（建议修复）"))
    if infos:
        parts.append(
            _format_issues(infos[:10], "ℹ️ 信息（可选优化）",
                           show_suggestion=False)
        )
        if len(infos) > 10:
            n = len(infos) - 10
            parts.append(f"... 还有 {n} 个信息级别问题\n\n")
    if not errs and not warns and not infos:
        parts.append("✅ 代码规范检查通过\n\n")
    parts.append("---\n\n")
    return "".join(parts), errs


def _report_bandit(bandit_result):
    """Bandit 安全检查部分"""
    parts = ["## 3️⃣ 安全检查 (Bandit)\n\n"]
    issues = bandit_result.get("issues", [])
    err_levels = ("error", "HIGH", "MEDIUM")
    warn_levels = ("warning", "LOW")
    errs = [i for i in issues if i["level"] in err_levels]
    warns = [i for i in issues if i["level"] in warn_levels]

    parts.append(_format_issues(errs, "❌ 安全问题（必须修复）"))
    parts.append(
        _format_issues(warns, "⚠️ 低风险（建议修复）",
                       show_suggestion=False)
    )
    if not errs and not warns:
        parts.append("✅ 安全检查通过\n\n")
    parts.append("---\n\n")
    return "".join(parts), errs


def _report_spec(spec_result):
    """SPEC.md 合规性检查部分"""
    parts = ["## 4️⃣ SPEC.md 合规性检查\n\n"]
    issues = spec_result.get("issues", [])
    errs = [i for i in issues if i["level"] == "error"]
    warns = [i for i in issues if i["level"] == "warning"]

    parts.append(_format_issues(errs, "❌ 错误（必须修复）"))
    parts.append(_format_issues(warns, "⚠️ 警告（建议修复）"))
    if not errs and not warns:
        parts.append("✅ SPEC.md 合规性检查通过\n\n")
    parts.append("---\n\n")
    return "".join(parts), errs


def generate_markdown_report(
    current_branch: str,
    target_branch: str,
    rebase_result: Dict,
    code_style_result: Dict,
    bandit_result: Dict,
    spec_result: Dict,
    diff_stat: str
) -> str:
    """生成 Markdown 格式的审查报告"""
    has_errors = (
        rebase_result.get("status") == "conflict"
        or code_style_result.get("total_errors", 0) > 0
        or bandit_result.get("total_errors", 0) > 0
        or spec_result.get("total_errors", 0) > 0
    )
    has_warnings = (
        code_style_result.get("total_warnings", 0) > 0
        or bandit_result.get("total_warnings", 0) > 0
        or spec_result.get("total_warnings", 0) > 0
    )
    if has_errors:
        status = "❌ FAIL"
    elif has_warnings:
        status = "⚠️ WARNING"
    else:
        status = "✅ PASS"

    parts = [_report_header(current_branch, target_branch, status)]

    # 概览表
    infos = code_style_result.get("total_infos", 0)
    parts.append("## 📋 检查概览\n\n")
    parts.append("| 检查项 | 状态 | 错误 | 警告 | 信息 |\n")
    parts.append("|--------|------|------|------|------|\n")
    rebase_row = {
        "total_errors": len(rebase_result.get("conflicts", [])),
        "total_warnings": 0,
        "status": rebase_result.get("status"),
    }
    parts.append(_table_row("Rebase 冲突", rebase_row))
    parts.append(
        _table_row("代码规范", code_style_result, str(infos))
    )
    parts.append(_table_row("安全检查", bandit_result))
    parts.append(_table_row("SPEC.md 合规", spec_result))
    parts.append("\n---\n\n")

    parts.append(_report_rebase(rebase_result, target_branch))

    cs_text, cs_errs = _report_code_style(code_style_result)
    parts.append(cs_text)

    bd_text, bd_errs = _report_bandit(bandit_result)
    parts.append(bd_text)

    sp_text, sp_errs = _report_spec(spec_result)
    parts.append(sp_text)

    all_errs = cs_errs + bd_errs + sp_errs
    if all_errs:
        parts.append("## 📝 修复建议\n\n")
        parts.append("**必须修复的问题**:\n\n")
        for i, issue in enumerate(all_errs[:5], 1):
            parts.append(f"{i}. **{issue['file']}**")
            if issue.get("line", 0) > 0:
                parts.append(f" (第 {issue['line']} 行)")
            parts.append(f"\n   - {issue['message']}\n")
            sug = issue.get("suggestion", "—")
            parts.append(f"   - {sug}\n\n")
        if len(all_errs) > 5:
            n = len(all_errs) - 5
            parts.append(f"... 还有 {n} 个错误需要修复\n\n")

    parts.append("---\n\n## 📊 变更统计\n\n```\n")
    parts.append(diff_stat)
    parts.append("```\n")

    return "".join(parts)


def _check_mark(metadata, check_name, cmp="pass"):
    """返回 checklist 的 [x] 或 [ ]"""
    st = metadata["checks"][check_name]["status"]
    if cmp == "pass":
        return "x" if st == "pass" else " "
    return "x" if st != "fail" else " "


def generate_pr_checklist(
    current_branch: str,
    target_branch: str,
    json_metadata: Dict,
) -> str:
    """生成用于贴到 PR 的极简 checklist"""
    status = json_metadata["status"]
    summary = json_metadata["summary"]
    ts = datetime.now().strftime('%Y-%m-%d %H:%M')
    icon = {
        "pass": "✅", "warning": "⚠️",
    }.get(status, "❌")

    rb = _check_mark(json_metadata, "rebase")
    cs = _check_mark(json_metadata, "code_style", "fail")
    bd = _check_mark(json_metadata, "bandit", "fail")
    sp = _check_mark(json_metadata, "spec_compliance", "fail")

    lines = [
        "## Pre-submit Review Checklist\n",
        f"\n**Branch**: `{current_branch}` → `{target_branch}`\n",
        f"**Review Time**: {ts}\n",
        f"**Status**: {icon} {status.upper()}\n",
        "\n### Checks Performed\n",
        f"\n- [{rb}] Rebase conflict check\n",
        f"- [{cs}] Code style (Ruff + custom)\n",
        f"- [{bd}] Security check (Bandit)\n",
        f"- [{sp}] SPEC.md compliance\n",
        "\n### Summary\n",
        f"\n- **Errors**: {summary['total_errors']}\n",
        f"- **Warnings**: {summary['total_warnings']}\n",
        f"- **Files Changed**: {summary['files_changed']}\n\n",
    ]

    if summary["total_errors"] > 0:
        lines.append("### Issues Found\n\n")
        all_errors = []
        checks = json_metadata["checks"]
        for name in ("rebase", "code_style", "spec_compliance"):
            data = checks[name]
            if name == "rebase" and data.get("conflicts"):
                for c in data["conflicts"][:3]:
                    all_errors.append(
                        f"Rebase conflict: {c['file']}"
                    )
            elif "errors" in data:
                for e in data["errors"][:3]:
                    all_errors.append(
                        f"{e['rule']}: {e['file']}"
                    )
        for i, err in enumerate(all_errors[:5], 1):
            lines.append(f"{i}. {err}\n")
        if len(all_errors) > 5:
            n = len(all_errors) - 5
            lines.append(f"\n... and {n} more issues\n")
        lines.append(
            "\n**Action Required**: "
            "Fix all errors before merge.\n"
        )
    else:
        lines.append(
            "**Result**: ✅ All checks passed. "
            "Safe to merge.\n"
        )
    lines.append("\n---\n*Generated by `/akg-review`*\n")
    return "".join(lines)


def _filter_issues(result, level_match):
    """从结果中按 level 过滤 issues"""
    issues = result.get("issues", [])
    if isinstance(level_match, str):
        return [i for i in issues if i["level"] == level_match]
    return [i for i in issues if i["level"] in level_match]


def generate_json_metadata(
    current_branch: str,
    target_branch: str,
    rebase_result: Dict,
    code_style_result: Dict,
    bandit_result: Dict,
    spec_result: Dict,
    report_filename: str,
) -> Dict:
    """生成 JSON 元数据"""
    total_errors = (
        len(rebase_result.get("conflicts", []))
        + code_style_result.get("total_errors", 0)
        + bandit_result.get("total_errors", 0)
        + spec_result.get("total_errors", 0)
    )
    total_warnings = (
        code_style_result.get("total_warnings", 0)
        + bandit_result.get("total_warnings", 0)
        + spec_result.get("total_warnings", 0)
    )
    total_infos = code_style_result.get("total_infos", 0)

    if total_errors > 0:
        status = "fail"
    elif total_warnings > 0:
        status = "warning"
    else:
        status = "pass"

    files_changed = (
        code_style_result
        .get("summary", {})
        .get("files_checked", 0)
    )
    err_levels = ("error", "HIGH", "MEDIUM")
    warn_levels = ("warning", "LOW")

    return {
        "version": "1.0",
        "type": "review",
        "current_branch": current_branch,
        "target_branch": target_branch,
        "status": status,
        "generated_at": datetime.now().isoformat(),
        "checks": {
            "rebase": {
                "status": rebase_result.get("status", "error"),
                "conflicts": rebase_result.get("conflicts", []),
                "message": rebase_result.get("message", ""),
            },
            "code_style": {
                "status": code_style_result.get("status", "pass"),
                "errors": _filter_issues(code_style_result, "error"),
                "warnings": _filter_issues(code_style_result, "warning"),
                "infos": _filter_issues(code_style_result, "info"),
            },
            "bandit": {
                "status": bandit_result.get("status", "pass"),
                "errors": _filter_issues(bandit_result, err_levels),
                "warnings": _filter_issues(bandit_result, warn_levels),
            },
            "spec_compliance": {
                "status": spec_result.get("status", "pass"),
                "errors": _filter_issues(spec_result, "error"),
                "warnings": _filter_issues(spec_result, "warning"),
            },
        },
        "summary": {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_infos": total_infos,
            "files_changed": files_changed,
        },
        "report_file": report_filename,
    }


def _normalize_ruff_issues(ruff_data) -> List[Dict]:
    """将 ruff JSON 输出转换为统一的 issue 格式"""
    if not isinstance(ruff_data, list):
        return []
    issues = []
    for item in ruff_data:
        loc = item.get("location", {})
        issues.append({
            "rule": item.get("code", "RUFF"),
            "level": "error",
            "file": item.get("filename", ""),
            "line": loc.get("row", 0),
            "message": item.get("message", ""),
            "suggestion": (
                item.get("fix", {}).get("message", "—")
                if item.get("fix") else "—"
            )
        })
    return issues


def _normalize_bandit_issues(bandit_data) -> List[Dict]:
    """将 bandit JSON 输出转换为统一的 issue 格式"""
    results = bandit_data.get("results", []) if isinstance(bandit_data, dict) else []
    issues = []
    severity_map = {"HIGH": "error", "MEDIUM": "error", "LOW": "warning"}
    for item in results:
        issues.append({
            "rule": item.get("test_id", "BANDIT"),
            "level": severity_map.get(item.get("issue_severity", ""), "warning"),
            "file": item.get("filename", ""),
            "line": item.get("line_number", 0),
            "message": item.get("issue_text", ""),
            "suggestion": f"Confidence: {item.get('issue_confidence', 'N/A')}"
        })
    return issues


def _load_json(path: str, default: Dict = None) -> Dict:
    """安全加载 JSON 文件"""
    if default is None:
        default = {}
    if not path or not Path(path).exists():
        return default
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def main():
    parser = argparse.ArgumentParser(description="生成审查报告")
    parser.add_argument("--current-branch", required=True, help="当前分支")
    parser.add_argument("--target-branch", required=True, help="目标分支")
    parser.add_argument("--rebase-result", required=True, help="Rebase 检查结果 JSON")
    parser.add_argument("--ruff-result", help="Ruff 检查结果 JSON（可选）")
    parser.add_argument("--bandit-result", help="Bandit 检查结果 JSON（可选）")
    parser.add_argument(
        "--custom-style-result", required=True,
        help="自定义规范检查结果 JSON",
    )
    parser.add_argument("--spec-result", required=True, help="SPEC 合规性检查结果 JSON")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--repo-path", default=".", help="仓库路径")

    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rebase_result = _load_json(
        args.rebase_result,
        {"status": "error", "message": "结果文件缺失", "conflicts": []}
    )

    # Ruff: 转换原始 JSON 数组为统一格式，合并到 code_style
    ruff_raw = _load_json(args.ruff_result) if args.ruff_result else {}
    ruff_issues = _normalize_ruff_issues(ruff_raw if isinstance(ruff_raw, list) else [])

    code_style_result = _load_json(
        args.custom_style_result,
        {"status": "pass", "issues": [], "total_errors": 0, "total_infos": 0}
    )
    code_style_result["issues"] = (
        ruff_issues + code_style_result.get("issues", [])
    )
    ruff_errs = sum(
        1 for i in ruff_issues if i["level"] == "error"
    )
    ruff_warns = sum(
        1 for i in ruff_issues if i["level"] == "warning"
    )
    prev_errs = code_style_result.get("total_errors", 0)
    prev_warns = code_style_result.get("total_warnings", 0)
    code_style_result["total_errors"] = prev_errs + ruff_errs
    code_style_result["total_warnings"] = prev_warns + ruff_warns

    # Bandit: 转换原始 JSON 为统一格式
    bandit_raw = _load_json(args.bandit_result) if args.bandit_result else {}
    bandit_issues = _normalize_bandit_issues(bandit_raw)
    has_bandit_err = any(
        i["level"] == "error" for i in bandit_issues
    )
    bandit_result = {
        "status": "fail" if has_bandit_err else "pass",
        "issues": bandit_issues,
        "total_errors": sum(1 for i in bandit_issues if i["level"] == "error"),
        "total_warnings": sum(1 for i in bandit_issues if i["level"] == "warning"),
    }

    spec_result = _load_json(
        args.spec_result,
        {"status": "pass", "issues": [], "total_errors": 0, "total_warnings": 0}
    )

    try:
        result = subprocess.run(
            ["git", "diff", f"{args.target_branch}...HEAD", "--stat"],
            cwd=repo_path, capture_output=True, text=True
        )
        diff_stat = result.stdout if result.returncode == 0 else "无法获取变更统计"
    except Exception:
        diff_stat = "无法获取变更统计"

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    branch_slug = args.current_branch.replace('/', '_')
    report_filename = f"review_{branch_slug}_{timestamp}.md"
    json_filename = f"review_{branch_slug}_{timestamp}.json"

    markdown_content = generate_markdown_report(
        args.current_branch, args.target_branch,
        rebase_result, code_style_result, bandit_result, spec_result,
        diff_stat
    )

    json_content = generate_json_metadata(
        args.current_branch, args.target_branch,
        rebase_result, code_style_result, bandit_result, spec_result,
        report_filename
    )

    checklist_filename = f"checklist_{branch_slug}_{timestamp}.md"
    checklist_content = generate_pr_checklist(
        args.current_branch, args.target_branch, json_content
    )

    report_path = output_dir / report_filename
    json_path = output_dir / json_filename
    checklist_path = output_dir / checklist_filename

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=2, ensure_ascii=False)
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write(checklist_content)

    result = {
        "report_file": str(report_path),
        "json_file": str(json_path),
        "checklist_file": str(checklist_path),
        "status": json_content["status"],
        "summary": json_content["summary"]
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))

    sys.exit(1 if json_content["status"] == "fail" else 0)


if __name__ == "__main__":
    main()
