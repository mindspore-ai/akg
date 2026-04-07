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
SPEC.md 合规性检查脚本

检查代码变更是否符合各层级 SPEC.md 的约束。
"""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict


SPEC_RULES = {
    "SPEC-001": {"level": "error", "name": "在 core_v2/ 中写业务逻辑"},
    "SPEC-002": {"level": "error", "name": "在 op/ 中写通用框架代码"},
    "SPEC-003": {"level": "error", "name": "在 core_v2/tests/ 中写新测试"},
    "SPEC-004": {"level": "error", "name": "新增目录缺少 __init__.py"},
    "SPEC-005": {"level": "warning", "name": "新增/删除模块但未更新 SPEC.md"},
    "SPEC-006": {"level": "warning", "name": "修改对外接口但未更新 SPEC.md"},
    "SPEC-007": {"level": "warning", "name": "测试文件缺少必需的 marker"},
}


def find_spec_files(changed_files: List[str], repo_path: Path) -> Dict[str, Path]:
    """根据变更文件找到对应的 SPEC.md"""
    spec_map = {}

    for file_str in changed_files:
        file_path = Path(file_str)

        # 从文件路径向上查找 SPEC.md
        current = file_path.parent
        while current != Path('.'):
            spec_path = repo_path / current / "SPEC.md"
            if spec_path.exists():
                spec_map[file_str] = spec_path
                break
            current = current.parent

        # 如果没找到，检查是否有顶层 AGENTS.md
        if file_str not in spec_map:
            agents_md = repo_path / "AGENTS.md"
            if agents_md.exists():
                spec_map[file_str] = agents_md

    return spec_map


def parse_spec_constraints(spec_path: Path) -> Dict:
    """解析 SPEC.md 中的约束"""
    try:
        with open(spec_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception:
        return {"forbidden": [], "required": [], "directory_rules": []}

    constraints = {
        "forbidden": [],
        "required": [],
        "directory_rules": []
    }

    # 提取"不做什么"章节
    pat = r'##\s*不做什么\s*\n(.*?)(?=\n##|\Z)'
    forbidden_section = re.search(pat, content, re.DOTALL)
    if forbidden_section:
        forbidden_text = forbidden_section.group(1)
        # 提取所有 "不要" 开头的条目
        pat = r'-\s*\*\*不要\*\*(.*?)(?=\n-|\n\n|\Z)'
        forbidden_items = re.findall(
            pat, forbidden_text, re.DOTALL
        )
        constraints["forbidden"] = [item.strip() for item in forbidden_items]

    # 提取"开发约定"章节
    pat = r'##\s*开发约定\s*\n(.*?)(?=\n##|\Z)'
    required_section = re.search(pat, content, re.DOTALL)
    if required_section:
        required_text = required_section.group(1)
        # 提取关键规则
        constraints["required"] = [required_text]

    # 提取目录结构规则
    dir_section = re.search(r'##\s*目录结构\s*\n(.*?)(?=\n##|\Z)', content, re.DOTALL)
    if dir_section:
        constraints["directory_rules"] = [dir_section.group(1)]

    return constraints


def check_core_v2_business_logic(file_path: Path, content: str) -> List[Dict]:
    """检查 core_v2/ 中是否有业务逻辑"""
    issues = []

    if not str(file_path).startswith('python/akg_agents/core_v2/'):
        return issues

    # 检查是否包含算子相关的业务逻辑关键词
    business_keywords = [
        r'\bop_name\b',
        r'\bkernel\s+code\b',
        r'\btriton\s+kernel\b',
        r'\bverify\s+kernel\b',
        r'\bcompile\s+kernel\b',
    ]

    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        for pattern in business_keywords:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append({
                    "rule": "SPEC-001",
                    "level": "error",
                    "file": str(file_path),
                    "line": i,
                    "message": f"在 core_v2/ 中包含业务逻辑: {line.strip()[:60]}",
                    "suggestion": "业务逻辑应放在场景层（如 op/）"
                })
                break

    return issues


def check_op_framework_code(file_path: Path, content: str) -> List[Dict]:
    """检查 op/ 中是否有通用框架代码"""
    issues = []

    if not str(file_path).startswith('python/akg_agents/op/'):
        return issues

    # 检查是否定义了通用基类（AgentBase、BaseWorkflow 等）
    framework_patterns = [
        r'class\s+AgentBase',
        r'class\s+BaseWorkflow',
        r'class\s+BaseTool',
        r'class\s+SkillLoader',
    ]

    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        for pattern in framework_patterns:
            if re.search(pattern, line):
                issues.append({
                    "rule": "SPEC-002",
                    "level": "error",
                    "file": str(file_path),
                    "line": i,
                    "message": f"在 op/ 中定义通用框架类: {line.strip()}",
                    "suggestion": "通用框架代码应放在 core_v2/"
                })

    return issues


def check_test_location(file_path: Path) -> List[Dict]:
    """检查测试文件位置"""
    issues = []

    # 检查是否在 core_v2/tests/ 中
    if 'core_v2/tests/' in str(file_path) or 'core_v2\\tests\\' in str(file_path):
        issues.append({
            "rule": "SPEC-003",
            "level": "error",
            "file": str(file_path),
            "line": 0,
            "message": "在 core_v2/tests/ 中写新测试",
            "suggestion": "新测试应放在仓库根目录的 tests/ 下"
        })

    return issues


def check_init_py(
    changed_files: List[str], repo_path: Path,
    base_branch: str,
) -> List[Dict]:
    """检查新增目录是否有 __init__.py"""
    issues = []

    # 获取新增的目录
    new_dirs = set()
    for file_str in changed_files:
        file_path = Path(file_str)

        # 只检查 python/ 下的 .py 文件
        if not str(file_path).startswith('python/') or file_path.suffix != '.py':
            continue

        diff_cmd = [
            "git", "diff", f"{base_branch}...HEAD",
            "--diff-filter=A", "--name-only", "--relative",
        ]
        result = subprocess.run(
            diff_cmd, cwd=repo_path,
            capture_output=True, text=True,
        )

        if file_str in result.stdout:
            # 这是新增文件，检查其目录
            parent_dir = file_path.parent
            new_dirs.add(parent_dir)

    # 检查每个新目录是否有 __init__.py
    for dir_path in new_dirs:
        init_file = dir_path / "__init__.py"
        full_init_path = repo_path / init_file

        if not full_init_path.exists():
            issues.append({
                "rule": "SPEC-004",
                "level": "error",
                "file": str(dir_path),
                "line": 0,
                "message": f"新增目录缺少 __init__.py: {dir_path}",
                "suggestion": f"创建 {init_file}"
            })

    return issues


def check_spec_updates(
    changed_files: List[str], repo_path: Path,
    base_branch: str,
) -> List[Dict]:
    """检查是否需要更新 SPEC.md"""
    issues = []

    # 检查是否有新增/删除的模块文件
    added_files = []
    deleted_files = []

    diff_cmd = [
        "git", "diff", f"{base_branch}...HEAD",
        "--diff-filter=A", "--name-only", "--relative",
    ]
    result = subprocess.run(
        diff_cmd, cwd=repo_path,
        capture_output=True, text=True,
    )
    skip_prefixes = (
        ".opencode/", ".claude/", ".cursor/",
        "scripts/", "tests/",
    )
    added_files = [
        f for f in result.stdout.strip().split('\n')
        if f.endswith('.py')
        and not any(f.startswith(p) for p in skip_prefixes)
    ]

    diff_cmd = [
        "git", "diff", f"{base_branch}...HEAD",
        "--diff-filter=D", "--name-only", "--relative",
    ]
    result = subprocess.run(
        diff_cmd, cwd=repo_path,
        capture_output=True, text=True,
    )
    deleted_files = [
        f for f in result.stdout.strip().split('\n')
        if f.endswith('.py')
        and not any(f.startswith(p) for p in skip_prefixes)
    ]

    # 检查是否有对应的 SPEC.md 更新
    spec_files = [f for f in changed_files if 'SPEC.md' in f or 'AGENTS.md' in f]

    if (added_files or deleted_files) and not spec_files:
        issues.append({
            "rule": "SPEC-005",
            "level": "warning",
            "file": "SPEC.md",
            "line": 0,
            "message": (
                f"新增/删除了 "
                f"{len(added_files) + len(deleted_files)}"
                f" 个模块文件，但未更新 SPEC.md"
            ),
            "suggestion": "检查是否需要更新对应目录的 SPEC.md"
        })

    return issues


def check_test_markers(file_path: Path, content: str) -> List[Dict]:
    """检查测试文件的 marker"""
    issues = []

    # 只检查 op-st 和 op-bench 测试
    if not ('tests/op/st/' in str(file_path) or 'tests/op/bench/' in str(file_path)):
        return issues

    # 检查是否有必需的 marker
    required_markers = ['framework', 'dsl', 'backend', 'arch']
    found_markers = set()

    lines = content.split('\n')
    for line in lines:
        if '@pytest.mark.' in line:
            for marker in required_markers:
                if f'@pytest.mark.{marker}' in line:
                    found_markers.add(marker)

    missing_markers = set(required_markers) - found_markers
    if missing_markers:
        issues.append({
            "rule": "SPEC-007",
            "level": "warning",
            "file": str(file_path),
            "line": 0,
            "message": (
                "测试文件缺少必需的 marker: "
                f"{', '.join(missing_markers)}"
            ),
            "suggestion": (
                "op-st 和 op-bench 必须同时指定 "
                "framework, dsl, backend, arch 四类 marker"
            )
        })

    return issues


def main():
    parser = argparse.ArgumentParser(description="SPEC.md 合规性检查")
    parser.add_argument("--files", required=True, help="要检查的文件列表（空格分隔）")
    parser.add_argument("--base-branch", required=True, help="基准分支")
    parser.add_argument("--repo-path", default=".", help="仓库路径")
    parser.add_argument("--output", help="输出 JSON 文件路径")

    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    files = [f for f in args.files.split() if f]

    all_issues = []

    # 1. 检查每个文件的 SPEC 约束
    for file_str in files:
        file_path = Path(file_str)
        full_path = repo_path / file_path

        if not full_path.exists() or file_path.suffix != '.py':
            continue

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception:
            continue

        # 执行各项检查
        all_issues.extend(check_core_v2_business_logic(file_path, content))
        all_issues.extend(check_op_framework_code(file_path, content))
        all_issues.extend(check_test_location(file_path))
        all_issues.extend(check_test_markers(file_path, content))

    # 2. 检查目录级别的约束
    all_issues.extend(check_init_py(files, repo_path, args.base_branch))
    all_issues.extend(check_spec_updates(files, repo_path, args.base_branch))

    # 统计
    errors = [i for i in all_issues if i["level"] == "error"]
    warnings = [i for i in all_issues if i["level"] == "warning"]

    result = {
        "status": "fail" if errors else ("warning" if warnings else "pass"),
        "total_errors": len(errors),
        "total_warnings": len(warnings),
        "issues": all_issues,
        "summary": {
            "files_checked": len(files),
            "errors": len(errors),
            "warnings": len(warnings)
        }
    }

    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # 返回退出码
    if errors:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
