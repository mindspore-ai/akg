#!/usr/bin/env python3
"""验证参赛选手的提交包格式是否符合规范。

用法:
    python validate_submission.py <submission.tar.gz> [--extract-to <dir>]

检查项:
    1. tar 包结构: team_name/{t1,t2,...}/*.py
    2. meta.json 存在且格式正确
    3. .py 文件名在题库中存在
    4. 每个 .py 文件包含 ModelNew 类
    5. 基本安全检查（禁止危险 import）
"""

import argparse
import ast
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path

DEFAULT_BENCH_ROOT = Path(__file__).resolve().parent.parent

BLOCKED_MODULES = {
    "subprocess", "shutil", "socket", "http", "urllib",
    "ftplib", "smtplib", "ctypes", "multiprocessing",
}


def _discover_tiers(bench_root: Path) -> list[str]:
    return sorted(
        d.name for d in bench_root.iterdir()
        if d.is_dir() and d.name.startswith("t") and d.name[1:].isdigit()
    )


def _get_case_registry(bench_root: Path) -> dict[str, set[str]]:
    """扫描题库，返回 {tier: {case_name, ...}}。"""
    registry: dict[str, set[str]] = {}
    for tier in _discover_tiers(bench_root):
        tier_dir = bench_root / tier
        cases = {
            f.stem for f in tier_dir.glob("*.py") if f.name != "__init__.py"
        }
        registry[tier] = cases
    return registry


def _check_meta(meta_path: Path) -> list[str]:
    errors = []
    if not meta_path.exists():
        errors.append("缺少 meta.json")
        return errors
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"meta.json 格式错误: {e}")
        return errors

    for field in ("team_name",):
        if field not in meta:
            errors.append(f"meta.json 缺少必填字段: {field}")
    return errors


def _check_py_file(py_path: Path) -> list[str]:
    errors = []
    try:
        source = py_path.read_text(encoding="utf-8")
    except Exception as e:
        errors.append(f"无法读取 {py_path.name}: {e}")
        return errors

    try:
        tree = ast.parse(source, filename=str(py_path))
    except SyntaxError as e:
        errors.append(f"{py_path.name} 语法错误: {e}")
        return errors

    has_model_new = any(
        isinstance(node, ast.ClassDef) and node.name == "ModelNew"
        for node in ast.walk(tree)
    )
    if not has_model_new:
        errors.append(f"{py_path.name} 缺少 ModelNew 类")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in BLOCKED_MODULES:
                    errors.append(f"{py_path.name} 使用了禁止的模块: {alias.name}")
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top in BLOCKED_MODULES:
                errors.append(f"{py_path.name} 使用了禁止的模块: {node.module}")

    return errors


def validate_tar(tar_path: str, extract_to: str | None = None, bench_root: Path | None = None) -> bool:
    tar_path = Path(tar_path)
    bench_root = bench_root or DEFAULT_BENCH_ROOT
    if not tar_path.exists():
        print(f"[ERROR] 文件不存在: {tar_path}")
        return False

    case_registry = _get_case_registry(bench_root)
    all_errors: list[str] = []

    with tarfile.open(tar_path, "r:*") as tf:
        members = tf.getmembers()
        top_dirs = {m.name.split("/")[0] for m in members if "/" in m.name}

        if len(top_dirs) != 1:
            all_errors.append(
                f"tar 包应包含唯一的顶层目录(队伍名), 发现: {top_dirs}"
            )
            print_errors(all_errors)
            return False

        team_name = top_dirs.pop()

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            tf.extractall(tmpdir, filter="data")
            team_dir = Path(tmpdir) / team_name

            all_errors.extend(_check_meta(team_dir / "meta.json"))

            submitted_tiers = sorted(
                d.name for d in team_dir.iterdir()
                if d.is_dir() and d.name.startswith("t")
            )
            if not submitted_tiers:
                all_errors.append("未找到任何 tier 目录 (t1, t2, ...)")

            for tier in submitted_tiers:
                if tier not in case_registry:
                    all_errors.append(f"未知的 tier: {tier}")
                    continue
                tier_dir = team_dir / tier
                for py_file in sorted(tier_dir.glob("*.py")):
                    if py_file.name == "__init__.py" or py_file.name.startswith("._"):
                        continue
                    if py_file.stem not in case_registry[tier]:
                        all_errors.append(
                            f"{tier}/{py_file.name} 不在题库中"
                        )
                    all_errors.extend(_check_py_file(py_file))

            if all_errors:
                print_errors(all_errors)
                return False

            if extract_to:
                dst = Path(extract_to) / team_name
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(team_dir, dst)
                print(f"[OK] 已解压到 {dst}")

    print(f"[OK] 验证通过: {team_name}")
    return True


def print_errors(errors: list[str]):
    print("[FAILED] 验证失败:")
    for e in errors:
        print(f"  - {e}")


def main():
    parser = argparse.ArgumentParser(description="验证参赛提交包")
    parser.add_argument("tarball", help="提交的 tar.gz 文件路径")
    parser.add_argument(
        "--extract-to",
        default=None,
        help="验证通过后解压到指定目录 (默认: 不解压)",
    )
    parser.add_argument(
        "--bench-dir",
        default=None,
        help="题库根目录 (默认: 脚本所在目录的上级目录)",
    )
    args = parser.parse_args()
    bench_root = Path(args.bench_dir).resolve() if args.bench_dir else None
    ok = validate_tar(args.tarball, args.extract_to, bench_root=bench_root)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
