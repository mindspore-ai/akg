# Copyright 2026 Huawei Technologies Co., Ltd
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

"""代码清理工具：import 去重、私有模块移除、from-import 合并"""

import ast
import re
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from akg_agents.op.tools.task_constructor.ast_utils import (
    extract_file_header,
    extract_functions,
    clean_unused_imports,
    has_private_segments,
    resolve_attribute_chain,
)

logger = logging.getLogger(__name__)


def polish_task_code(code: str, private_prefixes: List[str] = None) -> str:
    """
    文本级别的最终清理（AST 清理的补充/兜底）:
    - 去重完全相同的 import 行
    - 移除内部私有模块 import
    - 移除 __all__ 定义、mypy 注释
    - 移除私有模块赋值
    - 移除 section 注释标记 (# ===== ... =====)
    - 合并连续空行

    Args:
        code: 待清理的代码
        private_prefixes: 私有模块前缀列表，默认 ["torch._"]
    """
    if private_prefixes is None:
        private_prefixes = ["torch._"]

    lines = code.splitlines()
    seen_imports = set()
    result = []

    # 构建正则模式
    prefix_pattern = "|".join(re.escape(p) for p in private_prefixes)
    private_import_re = re.compile(
        rf'^\s*import\s+(?:{prefix_pattern})\w+'
        rf'|^\s*from\s+(?:{prefix_pattern})\w+\s+import'
    )
    private_assign_re = re.compile(
        rf'^\s*\w+\s*=\s*(?:{prefix_pattern})'
    )
    all_re = re.compile(r'^\s*__all__\s*[:=]')
    mypy_re = re.compile(r'^\s*#\s*mypy:')
    section_re = re.compile(r'^\s*#\s*=====.*=====\s*$')
    import_line_re = re.compile(r'^\s*(import\s+\S+|from\s+\S+\s+import\s+.+)\s*$')

    for line in lines:
        stripped = line.strip()

        if not stripped:
            result.append(line)
            continue

        if mypy_re.match(stripped):
            continue

        if all_re.match(stripped):
            continue

        if private_import_re.match(stripped):
            if ' as ' not in stripped:
                continue

        if private_assign_re.match(stripped):
            continue

        if section_re.match(stripped):
            continue

        m = import_line_re.match(stripped)
        if m:
            canon = re.sub(r'\s+', ' ', stripped.strip())
            if canon in seen_imports:
                continue
            seen_imports.add(canon)

        result.append(line)

    # 合并连续空行（最多2个）
    cleaned = []
    blank_count = 0
    for line in result:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    while cleaned and cleaned[0].strip() == '':
        cleaned.pop(0)
    while cleaned and cleaned[-1].strip() == '':
        cleaned.pop()

    return '\n'.join(cleaned)


def cleanup_task_code(code: str) -> str:
    """
    清理组装后的任务代码（AST + 文本双重清理）:
    - 移除内部私有模块的 import 和赋值
    - 移除 __all__、mypy 注释等非功能性代码
    - 合并同模块的 from-import 并移除未使用的名字
    - 去重 import 语句
    - 清理多余空行
    """
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        logger.warning(f"cleanup_task_code: AST 解析失败 (SyntaxError: {e})，跳过 AST 清理")
        return polish_task_code(code)

    try:
        return _cleanup_task_code_ast(code, tree)
    except Exception as e:
        logger.warning(f"cleanup_task_code: AST 清理异常 ({type(e).__name__}: {e})，使用文本清理兜底")
        return polish_task_code(code)


def _cleanup_task_code_ast(code: str, tree: ast.Module) -> str:
    """AST-based cleanup"""
    lines = code.splitlines()

    local_defs = {
        node.name for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }

    used_names = set()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            for child in ast.walk(node):
                if isinstance(child, ast.Name):
                    used_names.add(child.id)
                elif isinstance(child, ast.Attribute):
                    root = child
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    if isinstance(root, ast.Name):
                        used_names.add(root.id)

    remove_lines = set()
    seen_imports = set()
    from_import_groups = {}

    for node in ast.iter_child_nodes(tree):
        lr = set(range(node.lineno - 1, node.end_lineno))

        if isinstance(node, ast.Import):
            should_remove = False
            for alias in node.names:
                mod = alias.name
                name = alias.asname or mod.split('.')[0]
                key = f"import {mod}" + (f" as {alias.asname}" if alias.asname else "")

                if has_private_segments(mod):
                    if alias.asname:
                        if name in local_defs or name not in used_names:
                            should_remove = True
                    else:
                        should_remove = True
                elif key in seen_imports:
                    should_remove = True
                else:
                    seen_imports.add(key)

            if should_remove:
                remove_lines |= lr

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if has_private_segments(mod):
                needed = any(
                    (a.asname or a.name) in used_names
                    and (a.asname or a.name) not in local_defs
                    for a in node.names
                )
                if not needed:
                    remove_lines |= lr
                    continue
            from_import_groups.setdefault(mod, []).append((lr, node.names))

        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                name = target.id
                if name == '__all__':
                    remove_lines |= lr
                    continue
                source = resolve_attribute_chain(node.value)
                if source and has_private_segments(source):
                    if name in local_defs or name not in used_names:
                        remove_lines |= lr
                        continue
                if name not in used_names and name not in local_defs:
                    remove_lines |= lr

    replacements = {}
    for mod, groups in from_import_groups.items():
        all_names = {}
        active_groups = []
        for lr, aliases in groups:
            if lr & remove_lines:
                continue
            active_groups.append((lr, aliases))
            for alias in aliases:
                all_names[alias.name] = alias.asname

        if not active_groups:
            continue

        if len(active_groups) == 1:
            all_used = all(
                (a.asname or a.name) in used_names for a in active_groups[0][1]
            )
            if all_used:
                continue

        needed = {}
        for orig, asname in all_names.items():
            effective = asname or orig
            if effective in used_names:
                needed[orig] = asname

        all_lr = set()
        first_line = None
        for lr, _ in active_groups:
            all_lr |= lr
            line_min = min(lr)
            if first_line is None or line_min < first_line:
                first_line = line_min

        remove_lines |= all_lr

        if needed and first_line is not None:
            sorted_names = sorted(needed.keys())
            parts = []
            for n in sorted_names:
                asname = needed[n]
                if asname and asname != n:
                    parts.append(f"{n} as {asname}")
                else:
                    parts.append(n)
            replacements[first_line] = f"from {mod} import {', '.join(parts)}"

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith('# mypy:'):
            remove_lines.add(i)

    result = []
    for i, line in enumerate(lines):
        if i in replacements:
            result.append(replacements[i])
        elif i not in remove_lines:
            result.append(line)

    cleaned = []
    blank_count = 0
    for line in result:
        if line.strip() == '':
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)

    while cleaned and cleaned[-1].strip() == '':
        cleaned.pop()

    ast_result = '\n'.join(cleaned)
    return polish_task_code(ast_result)


def merge_from_imports_text(code: str) -> str:
    """文本级别合并同模块 from-import 语句"""
    lines = code.splitlines()
    from_imports: Dict[str, list] = {}
    from_import_re = re.compile(r'^from\s+(\S+)\s+import\s+(.+)$')

    for i, line in enumerate(lines):
        m = from_import_re.match(line.strip())
        if m:
            mod, names_str = m.group(1), m.group(2)
            from_imports.setdefault(mod, []).append((i, names_str))

    remove_lines = set()
    replacements = {}
    for mod, entries in from_imports.items():
        if len(entries) <= 1:
            continue
        all_names = set()
        for _, names_str in entries:
            for name in names_str.split(','):
                name = name.strip()
                if name:
                    all_names.add(name)
        first_idx = entries[0][0]
        sorted_names = sorted(all_names, key=lambda x: x.lower())
        replacements[first_idx] = f"from {mod} import {', '.join(sorted_names)}"
        for idx, _ in entries[1:]:
            remove_lines.add(idx)

    result = []
    for i, line in enumerate(lines):
        if i in replacements:
            result.append(replacements[i])
        elif i not in remove_lines:
            result.append(line)

    return '\n'.join(result)


def extract_functions_from_file(file_path: Path, function_names: List[str]) -> Tuple[str, str, List[str]]:
    """从文件中提取文件头 + 指定函数"""
    content = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return "", "", function_names

    header = extract_file_header(content, tree, clean_local_imports=True)
    funcs_code, not_found = extract_functions(content, tree, function_names)

    if funcs_code:
        header = clean_unused_imports(header, funcs_code)

    return header, funcs_code, not_found
