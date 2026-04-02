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
自定义代码规范检查（仅检查 ruff 无法覆盖的项目特定规则）
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict


RULES = {
    "CODE-001": {"level": "error", "name": "License 头缺失或格式错误"},
    "CODE-002": {"level": "error", "name": "从 core/ 导入（应使用 core_v2/）"},
    "CODE-004": {"level": "error", "name": "参数值不在有效范围内"},
    "CODE-013": {"level": "error", "name": "错误的包名引用"},
    "CODE-010": {"level": "info", "name": "TODO/FIXME/HACK 注释"},
}


def check_license_header(file_path: Path, content: str) -> List[Dict]:
    """检查 License 头"""
    issues = []
    
    if file_path.name == "__init__.py" or "test_" in file_path.name:
        return issues
    
    lines = content.split('\n')[:3]
    header_text = '\n'.join(lines)
    
    if "Copyright" not in header_text or "Huawei Technologies" not in header_text:
        issues.append({
            "rule": "CODE-001",
            "level": "error",
            "file": str(file_path),
            "line": 1,
            "message": "License 头缺失或格式不正确",
            "suggestion": "添加 Apache 2.0 License 头（年份 2025-2026）"
        })
    elif "2025-2026" not in header_text and "2026" not in header_text:
        issues.append({
            "rule": "CODE-001",
            "level": "error",
            "file": str(file_path),
            "line": 1,
            "message": "License 头年份不正确",
            "suggestion": "年份应为 '2025-2026' 或 '2026'"
        })
    
    return issues


def check_imports(file_path: Path, content: str) -> List[Dict]:
    """检查导入规范"""
    issues = []
    
    if "kernel_verifier.py" in str(file_path):
        return issues
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('#'):
            continue
        
        if re.search(r'from\s+akg_agents\.core\.', line):
            issues.append({
                "rule": "CODE-002",
                "level": "error",
                "file": str(file_path),
                "line": i,
                "message": f"从 core/ 导入: {line.strip()}",
                "suggestion": "使用 core_v2/ 替代（core/ 正在迁移）"
            })
    
    return issues


def check_parameter_values(file_path: Path, content: str) -> List[Dict]:
    """检查参数值"""
    issues = []
    
    VALID_VALUES = {
        "backend": ["cuda", "ascend", "cpu"],
        "framework": ["torch", "mindspore"],
        "dsl": ["triton_cuda", "triton_ascend", "cpp", "cuda_c", "tilelang_cuda", "ascendc", "pypto"],
    }
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('#'):
            continue
        
        for param, valid in VALID_VALUES.items():
            match = re.search(rf'{param}\s*=\s*["\']([^"\']+)["\']', line)
            if match and match.group(1) not in valid:
                issues.append({
                    "rule": "CODE-004",
                    "level": "error",
                    "file": str(file_path),
                    "line": i,
                    "message": f"{param}='{match.group(1)}' 不在有效值范围",
                    "suggestion": f"有效值: {', '.join(valid)}"
                })
    
    return issues


def check_package_name(file_path: Path, content: str) -> List[Dict]:
    """检查包名"""
    issues = []
    
    # 检查错误的包名（完整匹配，避免误报 akg_agents）
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if line.strip().startswith('#') or 'import' not in line:
            continue
        
        # 检查 ai_kernel_generator（完整包名）
        if 'ai_kernel_generator' in line:
            issues.append({
                "rule": "CODE-013",
                "level": "error",
                "file": str(file_path),
                "line": i,
                "message": f"错误的包名: {line.strip()}",
                "suggestion": "应为 'akg_agents'，不是 'ai_kernel_generator'"
            })
    
    return issues


def check_todo_comments(file_path: Path, content: str) -> List[Dict]:
    """检查 TODO 注释"""
    issues = []
    
    lines = content.split('\n')
    for i, line in enumerate(lines, 1):
        if re.search(r'#\s*(TODO|FIXME|XXX|HACK)', line, re.IGNORECASE):
            issues.append({
                "rule": "CODE-010",
                "level": "info",
                "file": str(file_path),
                "line": i,
                "message": f"待办注释: {line.strip()[:80]}",
                "suggestion": "考虑创建 Issue 跟踪或在本次提交中完成"
            })
    
    return issues


def check_file(file_path: Path, repo_path: Path) -> List[Dict]:
    """检查单个文件"""
    issues = []
    
    if file_path.suffix != '.py':
        return issues
    
    try:
        with open(repo_path / file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        issues.append({
            "rule": "ERROR",
            "level": "error",
            "file": str(file_path),
            "line": 0,
            "message": f"无法读取文件: {str(e)}",
            "suggestion": "检查文件是否存在且可读"
        })
        return issues
    
    # 执行检查（仅项目特定规则，bandit 已覆盖危险函数）
    issues.extend(check_license_header(file_path, content))
    issues.extend(check_imports(file_path, content))
    issues.extend(check_parameter_values(file_path, content))
    issues.extend(check_package_name(file_path, content))
    issues.extend(check_todo_comments(file_path, content))
    
    return issues


def main():
    parser = argparse.ArgumentParser(description="自定义代码规范检查")
    parser.add_argument("--files", required=True, help="文件列表（空格分隔）")
    parser.add_argument("--repo-path", default=".", help="仓库路径")
    parser.add_argument("--output", help="输出 JSON 文件")
    
    args = parser.parse_args()
    
    repo_path = Path(args.repo_path).resolve()
    files = [f for f in args.files.split() if f]
    
    all_issues = []
    for file_str in files:
        all_issues.extend(check_file(Path(file_str), repo_path))
    
    errors = [i for i in all_issues if i["level"] == "error"]
    infos = [i for i in all_issues if i["level"] == "info"]
    
    result = {
        "status": "fail" if errors else "pass",
        "total_errors": len(errors),
        "total_infos": len(infos),
        "issues": all_issues,
        "summary": {
            "files_checked": len(files),
            "errors": len(errors),
            "infos": len(infos)
        }
    }
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    
    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
