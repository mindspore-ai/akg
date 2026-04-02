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


def generate_markdown_report(
    current_branch: str,
    target_branch: str,
    rebase_result: Dict,
    code_style_result: Dict,
    spec_result: Dict,
    diff_stat: str
) -> str:
    """生成 Markdown 格式的审查报告"""
    
    # 计算总体状态
    has_errors = (
        rebase_result.get("status") == "conflict" or
        code_style_result.get("total_errors", 0) > 0 or
        spec_result.get("total_errors", 0) > 0
    )
    
    has_warnings = (
        code_style_result.get("total_warnings", 0) > 0 or
        spec_result.get("total_warnings", 0) > 0
    )
    
    if has_errors:
        overall_status = "❌ FAIL"
    elif has_warnings:
        overall_status = "⚠️ WARNING"
    else:
        overall_status = "✅ PASS"
    
    # 生成报告
    report = f"""# AKG Code Review Report

**分支**: {current_branch}
**目标**: {target_branch}
**时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**状态**: {overall_status}

---

## 📋 检查概览

| 检查项 | 状态 | 错误 | 警告 | 信息 |
|--------|------|------|------|------|
| Rebase 冲突 | {"✅" if rebase_result.get("status") == "pass" else "❌"} | {len(rebase_result.get("conflicts", []))} | 0 | 0 |
| 代码规范 | {"✅" if code_style_result.get("total_errors", 0) == 0 else "❌"} | {code_style_result.get("total_errors", 0)} | {code_style_result.get("total_warnings", 0)} | {code_style_result.get("total_infos", 0)} |
| SPEC.md 合规 | {"✅" if spec_result.get("total_errors", 0) == 0 else "❌"} | {spec_result.get("total_errors", 0)} | {spec_result.get("total_warnings", 0)} | 0 |

---

## 1️⃣ Rebase 冲突检查

**状态**: {rebase_result.get("message", "未知")}

"""
    
    # Rebase 冲突详情
    if rebase_result.get("conflicts"):
        report += "### 冲突文件\n\n"
        for conflict in rebase_result["conflicts"]:
            report += f"- **{conflict['file']}**\n"
            report += f"  - 类型: {conflict['type']}\n"
            report += f"  - 详情: {conflict['details']}\n\n"
        
        report += "**修复建议**:\n"
        report += "```bash\n"
        report += f"git fetch origin_gitcode\n"
        report += f"git rebase {target_branch}\n"
        report += "# 解决冲突后\n"
        report += "git rebase --continue\n"
        report += "```\n\n"
    
    report += "---\n\n"
    
    # 代码规范检查
    report += "## 2️⃣ 代码规范检查\n\n"
    
    code_issues = code_style_result.get("issues", [])
    errors = [i for i in code_issues if i["level"] == "error"]
    warnings = [i for i in code_issues if i["level"] == "warning"]
    infos = [i for i in code_issues if i["level"] == "info"]
    
    if errors:
        report += "### ❌ 错误（必须修复）\n\n"
        for issue in errors:
            report += f"**[{issue['rule']}] {issue['file']}:{issue['line']}**\n"
            report += f"- 问题: {issue['message']}\n"
            report += f"- 建议: {issue['suggestion']}\n\n"
    
    if warnings:
        report += "### ⚠️ 警告（建议修复）\n\n"
        for issue in warnings:
            report += f"**[{issue['rule']}] {issue['file']}:{issue['line']}**\n"
            report += f"- 问题: {issue['message']}\n"
            report += f"- 建议: {issue['suggestion']}\n\n"
    
    if infos:
        report += "### ℹ️ 信息（可选优化）\n\n"
        # 只显示前 10 个 info 级别问题
        for issue in infos[:10]:
            report += f"**[{issue['rule']}] {issue['file']}:{issue['line']}**\n"
            report += f"- {issue['message']}\n\n"
        
        if len(infos) > 10:
            report += f"... 还有 {len(infos) - 10} 个信息级别问题\n\n"
    
    if not errors and not warnings and not infos:
        report += "✅ 代码规范检查通过\n\n"
    
    report += "---\n\n"
    
    # SPEC.md 合规性检查
    report += "## 3️⃣ SPEC.md 合规性检查\n\n"
    
    spec_issues = spec_result.get("issues", [])
    spec_errors = [i for i in spec_issues if i["level"] == "error"]
    spec_warnings = [i for i in spec_issues if i["level"] == "warning"]
    
    if spec_errors:
        report += "### ❌ 错误（必须修复）\n\n"
        for issue in spec_errors:
            report += f"**[{issue['rule']}] {issue['file']}"
            if issue.get('line', 0) > 0:
                report += f":{issue['line']}"
            report += "**\n"
            report += f"- 问题: {issue['message']}\n"
            report += f"- 建议: {issue['suggestion']}\n\n"
    
    if spec_warnings:
        report += "### ⚠️ 警告（建议修复）\n\n"
        for issue in spec_warnings:
            report += f"**[{issue['rule']}] {issue['file']}"
            if issue.get('line', 0) > 0:
                report += f":{issue['line']}"
            report += "**\n"
            report += f"- 问题: {issue['message']}\n"
            report += f"- 建议: {issue['suggestion']}\n\n"
    
    if not spec_errors and not spec_warnings:
        report += "✅ SPEC.md 合规性检查通过\n\n"
    
    report += "---\n\n"
    
    # 修复建议汇总
    all_errors = errors + spec_errors
    if all_errors:
        report += "## 📝 修复建议\n\n"
        report += "**必须修复的问题**:\n\n"
        for i, issue in enumerate(all_errors[:5], 1):
            report += f"{i}. **{issue['file']}**"
            if issue.get('line', 0) > 0:
                report += f" (第 {issue['line']} 行)"
            report += f"\n   - {issue['message']}\n"
            report += f"   - {issue['suggestion']}\n\n"
        
        if len(all_errors) > 5:
            report += f"... 还有 {len(all_errors) - 5} 个错误需要修复\n\n"
    
    # 变更统计
    report += "---\n\n"
    report += "## 📊 变更统计\n\n"
    report += "```\n"
    report += diff_stat
    report += "```\n"
    
    return report


def generate_pr_checklist(
    current_branch: str,
    target_branch: str,
    json_metadata: Dict
) -> str:
    """生成用于贴到 PR 的极简 checklist"""
    
    status = json_metadata["status"]
    summary = json_metadata["summary"]
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # 状态图标
    status_icon = "✅" if status == "pass" else ("⚠️" if status == "warning" else "❌")
    
    checklist = f"""## Pre-submit Review Checklist

**Branch**: `{current_branch}` → `{target_branch}`  
**Review Time**: {timestamp}  
**Status**: {status_icon} {status.upper()}

### Checks Performed

- [{"x" if json_metadata["checks"]["rebase"]["status"] == "pass" else " "}] Rebase conflict check
- [{"x" if summary["total_errors"] == 0 else " "}] Code style check (ruff + bandit + custom)
- [{"x" if json_metadata["checks"]["spec_compliance"]["status"] != "fail" else " "}] SPEC.md compliance check

### Summary

- **Errors**: {summary["total_errors"]}
- **Warnings**: {summary["total_warnings"]}
- **Files Changed**: {summary["files_changed"]}

"""
    
    # 如果有错误，列出关键问题
    if summary["total_errors"] > 0:
        checklist += "### Issues Found\n\n"
        
        all_errors = []
        for check_name in ["rebase", "code_style", "spec_compliance"]:
            check_data = json_metadata["checks"][check_name]
            if check_name == "rebase" and check_data.get("conflicts"):
                all_errors.extend([f"Rebase conflict: {c['file']}" for c in check_data["conflicts"][:3]])
            elif "errors" in check_data:
                all_errors.extend([f"{e['rule']}: {e['file']}" for e in check_data["errors"][:3]])
        
        for i, error in enumerate(all_errors[:5], 1):
            checklist += f"{i}. {error}\n"
        
        if len(all_errors) > 5:
            checklist += f"\n... and {len(all_errors) - 5} more issues\n"
        
        checklist += "\n**Action Required**: Fix all errors before merge.\n"
    else:
        checklist += "**Result**: ✅ All checks passed. Safe to merge.\n"
    
    checklist += f"\n---\n*Generated by `/akg-review`*\n"
    
    return checklist


def generate_json_metadata(
    current_branch: str,
    target_branch: str,
    rebase_result: Dict,
    code_style_result: Dict,
    spec_result: Dict,
    report_filename: str
) -> Dict:
    """生成 JSON 元数据"""
    
    # 计算总体状态
    total_errors = (
        len(rebase_result.get("conflicts", [])) +
        code_style_result.get("total_errors", 0) +
        spec_result.get("total_errors", 0)
    )
    
    total_warnings = (
        code_style_result.get("total_warnings", 0) +
        spec_result.get("total_warnings", 0)
    )
    
    total_infos = code_style_result.get("total_infos", 0)
    
    if total_errors > 0:
        status = "fail"
    elif total_warnings > 0:
        status = "warning"
    else:
        status = "pass"
    
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
                "message": rebase_result.get("message", "")
            },
            "code_style": {
                "status": code_style_result.get("status", "pass"),
                "errors": [i for i in code_style_result.get("issues", []) if i["level"] == "error"],
                "warnings": [i for i in code_style_result.get("issues", []) if i["level"] == "warning"],
                "infos": [i for i in code_style_result.get("issues", []) if i["level"] == "info"]
            },
            "spec_compliance": {
                "status": spec_result.get("status", "pass"),
                "errors": [i for i in spec_result.get("issues", []) if i["level"] == "error"],
                "warnings": [i for i in spec_result.get("issues", []) if i["level"] == "warning"]
            }
        },
        "summary": {
            "total_errors": total_errors,
            "total_warnings": total_warnings,
            "total_infos": total_infos,
            "files_changed": code_style_result.get("summary", {}).get("files_checked", 0)
        },
        "report_file": report_filename
    }


def main():
    parser = argparse.ArgumentParser(description="生成审查报告")
    parser.add_argument("--current-branch", required=True, help="当前分支")
    parser.add_argument("--target-branch", required=True, help="目标分支")
    parser.add_argument("--rebase-result", required=True, help="Rebase 检查结果 JSON")
    parser.add_argument("--ruff-result", help="Ruff 检查结果 JSON（可选）")
    parser.add_argument("--custom-style-result", required=True, help="自定义规范检查结果 JSON")
    parser.add_argument("--spec-result", required=True, help="SPEC 合规性检查结果 JSON")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--repo-path", default=".", help="仓库路径")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path).resolve()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取检查结果
    try:
        with open(args.rebase_result, 'r', encoding='utf-8') as f:
            rebase_result = json.load(f)
    except Exception as e:
        rebase_result = {"status": "error", "message": str(e), "conflicts": []}
    
    # 读取 ruff 结果（可选）
    ruff_result = {"issues": [], "total_errors": 0}
    if args.ruff_result and Path(args.ruff_result).exists():
        try:
            with open(args.ruff_result, 'r', encoding='utf-8') as f:
                ruff_data = json.load(f)
                # ruff 输出格式转换
                ruff_result["issues"] = ruff_data if isinstance(ruff_data, list) else []
                ruff_result["total_errors"] = len(ruff_result["issues"])
        except Exception:
            pass
    
    # 读取自定义规范结果
    try:
        with open(args.custom_style_result, 'r', encoding='utf-8') as f:
            code_style_result = json.load(f)
    except Exception as e:
        code_style_result = {"status": "error", "issues": [], "total_errors": 0, "total_infos": 0}
    
    # 合并 ruff 和自定义结果
    code_style_result["issues"] = ruff_result.get("issues", []) + code_style_result.get("issues", [])
    code_style_result["total_errors"] = ruff_result.get("total_errors", 0) + code_style_result.get("total_errors", 0)
    
    try:
        with open(args.spec_result, 'r', encoding='utf-8') as f:
            spec_result = json.load(f)
    except Exception as e:
        spec_result = {"status": "error", "issues": [], "total_errors": 0, "total_warnings": 0}
    
    # 获取 diff stat
    try:
        result = subprocess.run(
            ["git", "diff", f"{args.target_branch}...HEAD", "--stat"],
            cwd=repo_path,
            capture_output=True,
            text=True
        )
        diff_stat = result.stdout if result.returncode == 0 else "无法获取变更统计"
    except Exception:
        diff_stat = "无法获取变更统计"
    
    # 生成文件名
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    branch_slug = args.current_branch.replace('/', '_')
    report_filename = f"review_{branch_slug}_{timestamp}.md"
    json_filename = f"review_{branch_slug}_{timestamp}.json"
    
    # 生成 Markdown 报告
    markdown_content = generate_markdown_report(
        args.current_branch,
        args.target_branch,
        rebase_result,
        code_style_result,
        spec_result,
        diff_stat
    )
    
    # 生成 JSON 元数据
    json_content = generate_json_metadata(
        args.current_branch,
        args.target_branch,
        rebase_result,
        code_style_result,
        spec_result,
        report_filename
    )
    
    # 生成极简 checklist（用于贴到 PR）
    checklist_filename = f"checklist_{branch_slug}_{timestamp}.md"
    checklist_content = generate_pr_checklist(
        args.current_branch,
        args.target_branch,
        json_content
    )
    
    # 写入文件
    report_path = output_dir / report_filename
    json_path = output_dir / json_filename
    checklist_path = output_dir / checklist_filename
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(markdown_content)
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_content, f, indent=2, ensure_ascii=False)
    
    with open(checklist_path, 'w', encoding='utf-8') as f:
        f.write(checklist_content)
    
    # 输出结果
    result = {
        "report_file": str(report_path),
        "json_file": str(json_path),
        "checklist_file": str(checklist_path),
        "status": json_content["status"],
        "summary": json_content["summary"]
    }
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    # 返回退出码
    if json_content["status"] == "fail":
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
