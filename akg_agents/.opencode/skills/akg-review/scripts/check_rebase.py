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
Rebase 冲突检查脚本

在临时 worktree 中执行 rebase dry-run，检测冲突但不影响当前工作目录。
"""

import argparse
import json
import subprocess
import tempfile
import shutil
import sys
from pathlib import Path
from typing import Dict, List


def run_command(
    cmd: List[str], cwd: Path = None, check: bool = True
) -> subprocess.CompletedProcess:
    """执行命令并返回结果"""
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result
    except subprocess.CalledProcessError as e:
        return e


def parse_conflicts(worktree_path: Path) -> List[Dict]:
    """解析冲突文件"""
    conflicts = []

    # 获取冲突文件列表
    result = run_command(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=worktree_path,
        check=False
    )

    if result.returncode != 0:
        return conflicts

    conflict_files = [f for f in result.stdout.strip().split('\n') if f]

    for file_path in conflict_files:
        full_path = worktree_path / file_path
        if not full_path.exists():
            conflicts.append({
                "file": file_path,
                "type": "deleted",
                "details": "文件在 rebase 过程中被删除或重命名"
            })
            continue

        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查是否有冲突标记
            has_conflict = (
                '<<<<<<< HEAD' in content
                or '=======' in content
                or '>>>>>>>' in content
            )
            if has_conflict:
                conflict_type = "content"
                # 统计冲突块数量
                conflict_count = content.count('<<<<<<< HEAD')
                details = f"发现 {conflict_count} 个冲突块"
            else:
                conflict_type = "unknown"
                details = "文件标记为冲突但未找到冲突标记"

            conflicts.append({
                "file": file_path,
                "type": conflict_type,
                "details": details
            })
        except Exception as e:
            conflicts.append({
                "file": file_path,
                "type": "error",
                "details": f"无法读取文件: {str(e)}"
            })

    return conflicts


def check_rebase(target_branch: str, current_branch: str, repo_path: Path) -> Dict:
    """
    在临时 worktree 中检查 rebase 冲突

    Args:
        target_branch: 目标分支（如 origin/br_agents）
        current_branch: 当前分支
        repo_path: 仓库根目录

    Returns:
        {
            "status": "pass" | "conflict" | "error",
            "conflicts": [{"file": str, "type": str, "details": str}],
            "message": str,
            "target_branch": str,
            "current_branch": str
        }
    """
    result = {
        "status": "error",
        "conflicts": [],
        "message": "",
        "target_branch": target_branch,
        "current_branch": current_branch
    }

    # 创建临时目录
    temp_dir = tempfile.mkdtemp(prefix="akg_review_rebase_")
    worktree_path = Path(temp_dir) / "worktree"

    try:
        head_commit = run_command(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            check=False
        )
        if head_commit.returncode == 0:
            commit_hash = head_commit.stdout.strip()
        else:
            commit_hash = current_branch

        print(f"创建临时 worktree: {worktree_path}", file=sys.stderr)
        cmd_result = run_command(
            ["git", "worktree", "add", "--detach", str(worktree_path), commit_hash],
            cwd=repo_path,
            check=False
        )

        if cmd_result.returncode != 0:
            result["message"] = f"创建 worktree 失败: {cmd_result.stderr}"
            return result

        # 2. 在 worktree 中执行 rebase
        print(f"执行 rebase dry-run: {target_branch}", file=sys.stderr)
        cmd_result = run_command(
            ["git", "rebase", target_branch],
            cwd=worktree_path,
            check=False
        )

        if cmd_result.returncode == 0:
            # Rebase 成功，无冲突
            result["status"] = "pass"
            result["message"] = "✅ 无 rebase 冲突"
        else:
            # Rebase 失败，检查冲突
            conflicts = parse_conflicts(worktree_path)

            if conflicts:
                result["status"] = "conflict"
                result["conflicts"] = conflicts
                n = len(conflicts)
                result["message"] = (
                    f"❌ 发现 {n} 个冲突文件，需要先 rebase"
                )
            else:
                result["status"] = "error"
                result["message"] = f"Rebase 失败但未检测到冲突: {cmd_result.stderr}"

    except Exception as e:
        result["status"] = "error"
        result["message"] = f"检查过程出错: {str(e)}"

    finally:
        # 3. 清理临时 worktree
        try:
            # 中止 rebase（如果有）
            run_command(["git", "rebase", "--abort"], cwd=worktree_path, check=False)

            # 移除 worktree
            run_command(
                ["git", "worktree", "remove", str(worktree_path), "--force"],
                cwd=repo_path,
                check=False
            )

            # 删除临时目录
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            print(f"清理临时文件失败: {e}", file=sys.stderr)

    return result


def main():
    parser = argparse.ArgumentParser(description="Rebase 冲突检查")
    parser.add_argument("--target-branch", required=True, help="目标分支")
    parser.add_argument("--current-branch", required=True, help="当前分支")
    parser.add_argument("--repo-path", default=".", help="仓库路径")
    parser.add_argument("--output", help="输出 JSON 文件路径")

    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    result = check_rebase(args.target_branch, args.current_branch, repo_path)

    # 输出结果
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
    else:
        print(json.dumps(result, indent=2, ensure_ascii=False))

    # 返回退出码
    if result["status"] == "pass":
        sys.exit(0)
    elif result["status"] == "conflict":
        sys.exit(1)
    else:
        sys.exit(2)


if __name__ == "__main__":
    main()
